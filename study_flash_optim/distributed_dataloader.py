"""
Distributed Stateful Dataloader for Parquet from Azure Blob Storage.

Reads parquet files (via SAS token URLs), tokenizes text, packs tokens into
fixed-length sequences, batches them, prefetches in a background thread,
and supports save/load of state for resumption.

Each parquet file's rows are sharded across ranks (row-level sharding).
"""

import argparse
import json
import logging
import queue
import random
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import pyarrow.parquet as pq
import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataloaderState:
    """Serializable per-rank dataloader state."""
    dataset_idx: int = 0
    file_idx: int = 0
    row_offset: int = 0
    epoch: int = 0
    total_samples_seen: int = 0
    buffer_remainder: List[int] = field(default_factory=list)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DataloaderState":
        with open(path) as f:
            return cls(**json.load(f))


class DistributedParquetDataloader:
    """
    Distributed dataloader that reads parquet files from HTTP/SAS URLs,
    tokenizes text, packs into sequences, and prefetches batches.
    """

    def __init__(self, config_path: str, rank: int, world_size: int,
                 device: torch.device = None, tokenizer=None):
        self.rank = rank
        self.world_size = world_size
        self.device = device or torch.device("cpu")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.seq_length = self.config["seq_length"]
        self.batch_size = self.config["batch_size"]
        self.seed = self.config.get("seed", 42)
        self.datasets = self.config["datasets"]

        # Normalize weights
        total_w = sum(d["weight"] for d in self.datasets)
        self._weights = [d["weight"] / total_w for d in self.datasets]

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["tokenizer"])

        # State
        self.state = DataloaderState()
        self._rng = random.Random(self.seed + rank)

        # Track per-dataset file/row cursors
        self._dataset_cursors = [
            {"file_idx": 0, "row_offset": 0} for _ in self.datasets
        ]

        # Token buffer for packing
        self._token_buffer: List[int] = []

        # Prefetch
        self._prefetch_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._prefetch_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start the prefetch thread."""
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return
        self._stop_event.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, daemon=True
        )
        self._prefetch_thread.start()

    def stop(self):
        """Stop the prefetch thread and drain the queue."""
        self._stop_event.set()
        # Drain to unblock the producer
        while not self._prefetch_queue.empty():
            try:
                self._prefetch_queue.get_nowait()
            except queue.Empty:
                break
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5)
            self._prefetch_thread = None

    def get_batch(self, timeout: float = 60.0) -> torch.Tensor:
        """
        Get the next batch from the prefetch queue.
        Returns tensor of shape (batch_size, seq_length) on self.device.
        """
        try:
            batch = self._prefetch_queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError("Dataloader prefetch queue timed out")
        if isinstance(batch, Exception):
            raise batch
        self.state.total_samples_seen += batch.shape[0]
        return batch.to(self.device)

    def save_state(self, path: str):
        """Save current dataloader state to JSON."""
        self.state.buffer_remainder = list(self._token_buffer)
        self.state.save(path)
        logger.info(f"Rank {self.rank}: state saved to {path}")

    def load_state(self, path: str):
        """Load dataloader state and fast-forward to the saved position."""
        self.state = DataloaderState.load(path)
        self._token_buffer = list(self.state.buffer_remainder)
        # Restore dataset cursors from state
        if self.state.dataset_idx < len(self._dataset_cursors):
            self._dataset_cursors[self.state.dataset_idx]["file_idx"] = self.state.file_idx
            self._dataset_cursors[self.state.dataset_idx]["row_offset"] = self.state.row_offset
        # Re-seed RNG deterministically
        self._rng = random.Random(self.seed + self.rank + self.state.total_samples_seen)
        logger.info(f"Rank {self.rank}: state loaded from {path}, "
                    f"epoch={self.state.epoch}, samples_seen={self.state.total_samples_seen}")

    def __iter__(self):
        self.start()
        return self

    def __next__(self) -> torch.Tensor:
        return self.get_batch()

    def __del__(self):
        self.stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pick_dataset(self) -> int:
        """Weighted random choice of dataset index."""
        return self._rng.choices(range(len(self.datasets)), weights=self._weights, k=1)[0]

    def _read_parquet_rows(self, url: str, text_column: str) -> List[str]:
        """Read a parquet file from URL and return this rank's sharded rows."""
        table = pq.read_table(url)
        col = table.column(text_column)
        all_rows = col.to_pylist()
        # Row-level sharding: rank r takes rows[r::world_size]
        return all_rows[self.rank :: self.world_size]

    def _tokenize_rows(self, rows: List[str]) -> List[int]:
        """Tokenize a list of text rows and concatenate into a token stream."""
        tokens = []
        for text in rows:
            if text is None:
                continue
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            tokens.extend(encoded)
        return tokens

    def _fill_buffer_from_dataset(self, ds_idx: int) -> bool:
        """
        Read the next file from dataset ds_idx, tokenize, and append to buffer.
        Returns True if data was read, False if dataset is exhausted (triggers epoch bump).
        """
        ds = self.datasets[ds_idx]
        cursor = self._dataset_cursors[ds_idx]
        files = ds["files"]

        if cursor["file_idx"] >= len(files):
            # Dataset exhausted — reset for next epoch
            cursor["file_idx"] = 0
            cursor["row_offset"] = 0
            self.state.epoch += 1
            logger.info(f"Rank {self.rank}: dataset '{ds['name']}' epoch {self.state.epoch}")
            return False

        url = files[cursor["file_idx"]]
        text_col = ds.get("text_column", "text")

        try:
            rows = self._read_parquet_rows(url, text_col)
        except Exception as e:
            logger.error(f"Rank {self.rank}: failed to read {url}: {e}")
            cursor["file_idx"] += 1
            cursor["row_offset"] = 0
            return False

        # Fast-forward past already-consumed rows
        offset = cursor["row_offset"]
        if offset > 0:
            rows = rows[offset:]

        tokens = self._tokenize_rows(rows)
        self._token_buffer.extend(tokens)

        # Advance cursor
        cursor["file_idx"] += 1
        cursor["row_offset"] = 0
        return True

    def _extract_batch(self) -> Optional[torch.Tensor]:
        """
        Try to extract one batch from the token buffer.
        Returns (batch_size, seq_length) tensor or None if not enough tokens.
        """
        needed = self.batch_size * self.seq_length
        if len(self._token_buffer) < needed:
            return None

        sequences = []
        for _ in range(self.batch_size):
            seq = self._token_buffer[: self.seq_length]
            self._token_buffer = self._token_buffer[self.seq_length :]
            sequences.append(seq)

        batch = torch.tensor(sequences, dtype=torch.long)
        return batch

    def _snapshot_state(self, ds_idx: int):
        """Snapshot cursor position into the state for save/load."""
        cursor = self._dataset_cursors[ds_idx]
        self.state.dataset_idx = ds_idx
        self.state.file_idx = cursor["file_idx"]
        self.state.row_offset = cursor["row_offset"]

    def _prefetch_loop(self):
        """Background thread: continuously produce batches into the queue."""
        try:
            while not self._stop_event.is_set():
                # Try to form a batch from existing buffer
                batch = self._extract_batch()
                if batch is not None:
                    try:
                        while not self._stop_event.is_set():
                            try:
                                self._prefetch_queue.put(batch, timeout=0.5)
                                break
                            except queue.Full:
                                continue
                    except Exception:
                        break
                    continue

                # Buffer insufficient — refill from a dataset
                ds_idx = self._pick_dataset()
                filled = self._fill_buffer_from_dataset(ds_idx)
                self._snapshot_state(ds_idx)

                if not filled:
                    # Dataset wrapped around; try again (next epoch data)
                    filled2 = self._fill_buffer_from_dataset(ds_idx)
                    self._snapshot_state(ds_idx)
                    if not filled2:
                        logger.warning(f"Rank {self.rank}: dataset {ds_idx} returned no data")
        except Exception as e:
            # Push the exception so get_batch() can raise it
            try:
                self._prefetch_queue.put(e, timeout=5)
            except queue.Full:
                pass


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _self_test(config_path: str):
    """Run a self-test simulating 2 ranks to verify correctness."""
    import tempfile
    import os
    import pandas as pd

    print("=== Self-Test Mode ===\n")

    # Create a temporary parquet file with sample data
    tmpdir = tempfile.mkdtemp()
    sample_texts = [f"This is sample text number {i}. " * 20 for i in range(100)]
    df = pd.DataFrame({"text": sample_texts})
    parquet_path = os.path.join(tmpdir, "test_data.parquet")
    df.to_parquet(parquet_path)

    # Create a temporary config
    test_config = {
        "datasets": [
            {
                "name": "test_dataset",
                "files": [parquet_path],
                "text_column": "text",
                "weight": 1.0,
            }
        ],
        "tokenizer": "gpt2",
        "seq_length": 128,
        "batch_size": 4,
        "seed": 42,
    }
    config_path_tmp = os.path.join(tmpdir, "test_config.yaml")
    with open(config_path_tmp, "w") as f:
        yaml.dump(test_config, f)

    world_size = 2
    num_batches = 3

    for rank in range(world_size):
        print(f"--- Rank {rank}/{world_size} ---")
        loader = DistributedParquetDataloader(
            config_path=config_path_tmp, rank=rank, world_size=world_size
        )
        loader.start()

        for i in range(num_batches):
            batch = loader.get_batch(timeout=30)
            print(f"  Batch {i}: shape={tuple(batch.shape)}, "
                  f"dtype={batch.dtype}, "
                  f"first_tokens={batch[0, :5].tolist()}")
            assert batch.shape == (4, 128), f"Expected (4, 128), got {batch.shape}"
            assert batch.dtype == torch.long

        # Test state save/load
        state_path = os.path.join(tmpdir, f"state_rank{rank}.json")
        loader.save_state(state_path)
        print(f"  State saved: samples_seen={loader.state.total_samples_seen}")

        # Stop and reload
        loader.stop()

        loader2 = DistributedParquetDataloader(
            config_path=config_path_tmp, rank=rank, world_size=world_size
        )
        loader2.load_state(state_path)
        loader2.start()
        batch_resumed = loader2.get_batch(timeout=30)
        print(f"  Resumed batch: shape={tuple(batch_resumed.shape)}")
        assert batch_resumed.shape == (4, 128)
        loader2.stop()

        print(f"  Rank {rank} PASSED\n")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)
    print("=== All tests PASSED ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Parquet Dataloader")
    parser.add_argument("--config", type=str, default="dataloader_config.yaml")
    parser.add_argument("--test", action="store_true", help="Run self-test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.test:
        _self_test(args.config)
    else:
        print("Use --test to run self-test, or import as a module.")
        print("Example usage:")
        print("  loader = DistributedParquetDataloader('config.yaml', rank=0, world_size=4)")
        print("  loader.start()")
        print("  batch = loader.get_batch()")

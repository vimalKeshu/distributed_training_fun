import io
import json
import logging
import os
import queue
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq
import torch
import yaml

logger = logging.getLogger(__name__)


# -- State ------------------------------------------------------------------

@dataclass
class DataloaderState:
    epoch: int = 0
    total_tokens_produced: int = 0
    total_batches_produced: int = 0
    buffer_remainder: List[int] = field(default_factory=list)
    dataset_cursors: List[Dict[str, int]] = field(default_factory=list)
    file_orders: Dict[str, List[int]] = field(default_factory=dict)

    def save(self, path: str):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "DataloaderState":
        with open(path) as f:
            return cls(**json.load(f))


# -- URL Resolver (pluggable auth) -----------------------------------------

class _UrlResolver:
    """
    Resolves raw file URLs using the `auth` block from config.
    Supports: sas_env, sas_inline, identity, connection_string, none.
    """

    def __init__(self, config: Dict[str, Any]):
        auth = config.get("auth", {})
        self._auth_type = auth.get("type", "none")
        self._sas_token: Optional[str] = None
        self._blob_client: Optional[Any] = None
        self._container: Optional[str] = auth.get("container")

        if self._auth_type == "sas_env":
            env_var = auth.get("env_var", "SAS_TOKEN")
            self._sas_token = os.environ.get(env_var)
            if not self._sas_token:
                raise EnvironmentError(
                    f"auth.type='sas_env' but ${env_var} is not set."
                )
            self._sas_token = self._sas_token.lstrip("?")

        elif self._auth_type == "sas_inline":
            self._sas_token = auth["sas_token"].lstrip("?")

        elif self._auth_type == "identity":
            from azure.identity import DefaultAzureCredential
            from azure.storage.blob import BlobServiceClient
            self._blob_client = BlobServiceClient(
                auth["account_url"], credential=DefaultAzureCredential()
            )

        elif self._auth_type == "connection_string":
            from azure.storage.blob import BlobServiceClient
            self._blob_client = BlobServiceClient.from_connection_string(
                auth["connection_string"]
            )

        elif self._auth_type != "none":
            raise ValueError(f"Unknown auth.type: {self._auth_type}")

    def resolve(self, source: str) -> str:
        if self._sas_token and source.startswith(("http://", "https://")):
            sep = "&" if "?" in source else "?"
            return f"{source}{sep}{self._sas_token}"
        return source

    @property
    def blob_client(self):
        return self._blob_client

    @property
    def container(self):
        return self._container


# -- Parquet reader with retry ----------------------------------------------

def _read_parquet_bytes(
    source: str, resolver: _UrlResolver, max_retries: int = 3, backoff: float = 1.0
) -> io.BytesIO:
    for attempt in range(1, max_retries + 1):
        try:
            if (resolver.blob_client and resolver.container
                    and not source.startswith(("http://", "https://", "/"))):
                blob = resolver.blob_client.get_blob_client(
                    container=resolver.container, blob=source
                )
                stream = io.BytesIO()
                blob.download_blob().readinto(stream)
                stream.seek(0)
                return stream

            if source.startswith(("http://", "https://")):
                import requests
                resp = requests.get(resolver.resolve(source), timeout=120)
                resp.raise_for_status()
                return io.BytesIO(resp.content)

            with open(source, "rb") as fh:
                return io.BytesIO(fh.read())
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))
    raise RuntimeError("Unreachable")


# -- Dataloader -------------------------------------------------------------

class DistributedParquetDataloader:

    def __init__(self, config_path: str, rank: int, world_size: int,
                 device: torch.device = None, tokenizer=None):
        self.rank, self.world_size = rank, world_size
        self.device = device or torch.device("cpu")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.seq_length = self.config["seq_length"]
        self.batch_size = self.config["batch_size"]
        self.seed = self.config.get("seed", 42)
        self.prefetch_depth = self.config.get("prefetch_depth", 2)
        self.max_retries = self.config.get("max_retries", 3)
        self.add_eos = self.config.get("add_eos", True)
        self.shuffle_files = self.config.get("shuffle_files", True)
        self.datasets = self.config["datasets"]

        total_w = sum(d["weight"] for d in self.datasets)
        self._weights = [d["weight"] / total_w for d in self.datasets]

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["tokenizer"],
                model_max_length=self.seq_length,  # align with our actual seq_length
            )
        self.eos_id = self.tokenizer.eos_token_id or 0

        self._resolver = _UrlResolver(self.config)
        self.state = DataloaderState(
            dataset_cursors=[{"file_idx": 0, "row_offset": 0} for _ in self.datasets]
        )
        self._rng = random.Random(self.seed + rank)
        self._init_file_orders()
        self._token_buffer: List[int] = []

        self._queue: queue.Queue = queue.Queue(maxsize=self.prefetch_depth)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        logger.info(
            f"Rank {self.rank}: init done — seq_length={self.seq_length}, "
            f"batch_size={self.batch_size}, "
            f"tokens_per_batch={self.seq_length * self.batch_size}, "
            f"total_files={sum(len(d['files']) for d in self.datasets)}, "
            f"world_size={self.world_size}"
        )

    # -- File order ----------------------------------------------------------

    def _init_file_orders(self):
        for i, ds in enumerate(self.datasets):
            if ds["name"] not in self.state.file_orders:
                order = list(range(len(ds["files"])))
                if self.shuffle_files:
                    random.Random(self.seed + self.state.epoch + i).shuffle(order)
                self.state.file_orders[ds["name"]] = order

    def _reshuffle(self, ds_idx: int):
        ds = self.datasets[ds_idx]
        order = list(range(len(ds["files"])))
        if self.shuffle_files:
            random.Random(self.seed + self.state.epoch + ds_idx).shuffle(order)
        self.state.file_orders[ds["name"]] = order

    # -- Public API ----------------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._prefetch_loop, daemon=True, name=f"prefetch-r{self.rank}"
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        while not self._queue.empty():
            try: self._queue.get_nowait()
            except queue.Empty: break
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

    def get_batch(self, timeout: float = 300.0) -> torch.Tensor:
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(
                f"Rank {self.rank}: prefetch timed out after {timeout}s — "
                f"buffer_len={len(self._token_buffer)}, "
                f"needed={self.batch_size * self.seq_length}, "
                f"thread_alive={self._thread.is_alive() if self._thread else False}"
            )
        if isinstance(item, Exception):
            raise item
        self.state.total_batches_produced += 1
        self.state.total_tokens_produced += item.numel()
        return item.to(self.device, non_blocking=True)

    def save_state(self, path: str):
        self.state.buffer_remainder = list(self._token_buffer)
        self.state.save(path)
        logger.info(f"Rank {self.rank}: state saved → {path}")

    def load_state(self, path: str):
        self.state = DataloaderState.load(path)
        self._token_buffer = list(self.state.buffer_remainder)
        self._rng = random.Random(self.seed + self.rank + self.state.total_tokens_produced)
        self._init_file_orders()
        logger.info(f"Rank {self.rank}: state loaded ← {path}")

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "epoch": self.state.epoch,
            "total_batches": self.state.total_batches_produced,
            "total_tokens": self.state.total_tokens_produced,
            "buffer_len": len(self._token_buffer),
            "queue_size": self._queue.qsize(),
            "thread_alive": self._thread.is_alive() if self._thread else False,
        }        

    def __iter__(self):
        self.start(); return self

    def __next__(self):
        return self.get_batch()

    def __del__(self):
        self.stop()

    # -- Data reading --------------------------------------------------------

    def _read_sharded(self, source: str, text_col: str) -> List[str]:
        # Local files: read directly (no network copy into BytesIO)
        is_local = os.path.isfile(source)
        if is_local:
            pf = pq.ParquetFile(source)
        else:
            buf = _read_parquet_bytes(source, self._resolver, self.max_retries)
            pf = pq.ParquetFile(buf)

        rows, gidx = [], 0
        for rg in range(pf.metadata.num_row_groups):
            for val in pf.read_row_group(rg, columns=[text_col]).column(text_col).to_pylist():
                if gidx % self.world_size == self.rank:
                    rows.append(val)
                gidx += 1
        return rows

    def _tokenize(self, rows: List[str]) -> List[int]:
        tokens: List[int] = []
        for text in rows:
            if text is None:
                continue
            if self.add_eos and tokens:
                tokens.append(self.eos_id)
            tokens.extend(self.tokenizer.encode(text, add_special_tokens=False))
        return tokens

    def _fill_buffer(self, ds_idx: int) -> bool:
        ds = self.datasets[ds_idx]
        cursor = self.state.dataset_cursors[ds_idx]
        file_order = self.state.file_orders[ds["name"]]

        if cursor["file_idx"] >= len(file_order):
            cursor["file_idx"] = cursor["row_offset"] = 0
            self.state.epoch += 1
            self._reshuffle(ds_idx)
            logger.info(f"Rank {self.rank}: dataset '{ds['name']}' → epoch {self.state.epoch}")
            return False

        actual_idx = file_order[cursor["file_idx"]]
        url = ds["files"][actual_idx]
        text_col = ds.get("text_column", "text")

        try:
            logger.info(
                f"Rank {self.rank}: reading file {cursor['file_idx']}/{len(file_order)} "
                f"(actual={actual_idx}) from '{ds['name']}'"
            )
            rows = self._read_sharded(url, text_col)
            logger.info(f"Rank {self.rank}: got {len(rows)} rows from file {actual_idx}")
        except Exception as e:
            logger.error(f"Rank {self.rank}: FAILED to read file {actual_idx} ({url}): {e}")
            cursor["file_idx"] += 1; cursor["row_offset"] = 0
            return False

        offset = cursor["row_offset"]
        if offset > 0:
            rows = rows[offset:] if offset < len(rows) else []

        tokens = self._tokenize(rows)
        self._token_buffer.extend(tokens)
        cursor["file_idx"] += 1; cursor["row_offset"] = 0

        logger.info(
            f"Rank {self.rank}: tokenized {len(tokens)} tokens, "
            f"buffer now {len(self._token_buffer)} tokens"
        )
        return True

    # -- Batching & prefetch -------------------------------------------------

    def _extract_batch(self) -> Optional[torch.Tensor]:
        needed = self.batch_size * self.seq_length
        if len(self._token_buffer) < needed:
            return None
        t = torch.tensor(self._token_buffer[:needed], dtype=torch.long)
        self._token_buffer = self._token_buffer[needed:]
        return t.view(self.batch_size, self.seq_length)

    def _prefetch_loop(self):
        """Background thread: keep filling queue with batches."""
        try:
            logger.info(f"Rank {self.rank}: prefetch thread started")
            consecutive_failures = 0
            max_failures = len(self.datasets) * 2

            while not self._stop.is_set():
                # 1. Try to extract a batch from existing buffer
                batch = self._extract_batch()
                if batch is not None:
                    consecutive_failures = 0
                    self._enqueue(batch)
                    continue

                # 2. Buffer too small — keep loading files until we have enough
                needed = self.batch_size * self.seq_length
                while len(self._token_buffer) < needed and not self._stop.is_set():
                    ds_idx = self._rng.choices(
                        range(len(self.datasets)), self._weights, k=1
                    )[0]
                    filled = self._fill_buffer(ds_idx)
                    if not filled:
                        # Dataset exhausted — epoch wrapped, try again with fresh epoch
                        filled = self._fill_buffer(ds_idx)
                    if not filled:
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            raise RuntimeError(
                                f"Rank {self.rank}: {consecutive_failures} consecutive "
                                f"fill failures — datasets may be empty or unreadable. "
                                f"buffer={len(self._token_buffer)}, needed={needed}"
                            )
                    else:
                        consecutive_failures = 0

        except Exception as exc:
            logger.error(f"Rank {self.rank}: prefetch thread CRASHED: {exc}")
            try:
                self._queue.put(exc, timeout=10)
            except queue.Full:
                logger.error(f"Rank {self.rank}: could not push exception to queue")

    def _enqueue(self, batch: torch.Tensor):
        while not self._stop.is_set():
            try: self._queue.put(batch, timeout=0.5); return
            except queue.Full: continue
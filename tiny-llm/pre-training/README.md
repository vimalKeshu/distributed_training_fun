# Pre-Training

TinyStories pretraining run.

- prepare text into a token cache
- train a small decoder-only Transformer
- verify train/validation loss moves down
- generate a few samples
- record speed, memory, and failure cases


## Setup

Install data and training dependencies from the `pre-training` directory:

```bash
pip install -r data/requirements.txt
pip install -r training/requirements.txt
```

Download a TinyStories subset:

```bash
python data/download_tinystories.py subset --target-train-tokens 50000000
```

## Tokenizer

The current configs use a ~4k-vocab ByteLevel BPE tokenizer instead of raw
bytes. A whole story fits in a few hundred tokens, so the model spends its
capacity on story structure rather than spelling, and a full story fits in the
context window. Train it once:

```bash
python data/train_tokenizer.py --vocab-size 4096
```

This writes `data/tinystories/tokenizer.json` (with a `<|endoftext|>` token).
Every config requires `data.tokenizer_path`, and `model.vocab_size` must match
the tokenizer's vocab size (the trainer prints it).

## Debug Run

Start with the small debug model:

```bash
python pre-training/train.py pre-training/configs/assignment_01_debug.json
```

This should run on CPU, MPS, or CUDA. It is intended to catch data, shape, and
training-loop bugs before using the larger model.

## First 30M-Scale Run

After the debug run is healthy, the 30M RoPE config (BPE tokenizer, AdamW,
QK-norm) is a solid baseline:

```bash
python pre-training/train.py pre-training/configs/assignment_01_30m_rope.json
```

- `block_size`: 512 (a full story fits, thanks to the subword tokenizer)
- `n_layer`/`n_head`/`n_embd`: 8 / 8 / 512
- `dtype`: float32, position encoding: RoPE

Attention uses `scaled_dot_product_attention` (fused / memory-efficient), so the
`B x H x T x T` score matrix is never materialized; with the subword tokenizer a
512-token context already spans a whole TinyStory. If you ever want more headroom
just raise `model.block_size` (and lower `batch_size` to fit 12GB).

## Metrics & TensorBoard

Each run writes scalars to `<out_dir>/tensorboard/` and a raw
`<out_dir>/metrics.jsonl`. Watch training live with:

```bash
tensorboard --logdir pre-training/outputs
```

Logged: `train/loss`, `train/lr`, `perf/tokens_per_sec`, `eval/train_loss`,
`eval/valid_loss`, and `perf/peak_cuda_gb`.

## Efficient 30M Run (BPE + Muon + QK-norm)

The recommended from-scratch recipe bundles the efficiency improvements:

```bash
python data/train_tokenizer.py --vocab-size 4096
python pre-training/train.py pre-training/configs/assignment_01_30m_bpe_muon.json
```

What this config turns on (all also available individually via config/`GPTConfig`):

- **Subword tokenizer** (`data.tokenizer_path`, `model.vocab_size: 4096`) — the
  biggest efficiency win; shorter sequences, whole stories in context.
- **Muon optimizer** (`training.optimizer: "muon"`, `muon_lr`) — orthogonalized
  updates on the 2D block weights, AdamW on embeddings/head/norms. Tends to reach
  a target loss in fewer steps; valuable here since bf16/TF32 are unavailable on
  Maxwell/Pascal.
- **QK-norm** (`model.qk_norm: true`) and **scaled residual init** (always on) —
  training stability, tolerates the higher LR.
- **Gradient checkpointing** (`model.grad_checkpoint: true`) — optional; trades
  compute for memory so you can push `batch_size`/`block_size` on 12GB.
- **`torch.compile`** (`training.compile: true`) — kernel fusion throughput.

Muon lives in `muon.py` (a self-contained equivalent of `torch.optim.Muon`,
which only exists in torch >= 2.8).

## Sampling

Generate a sample from a checkpoint:

```bash
python pre-training/sample.py \
  --checkpoint pre-training/outputs/assignment_01_debug/best.pt \
  --prompt "Once upon a time"
```




# Instruction Tuning

Supervised instruction fine-tuning (SFT) of the GPT on
[`roneneldan/TinyStoriesInstruct`](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct).

It takes the pre-trained base model from `pre-training/` and continues training
on instruction-formatted examples, so the model learns to write a story given a
few instruction fields.

- reuse the exact same model, tokenizer, and Muon optimizer as pre-training
  (`model.py`, `tokenizer_util.py`, `muon.py` re-export them)
- load a pre-trained checkpoint and continue training
- mask the loss on the prompt so only the story tokens are learned
- generate stories from instruction fields

This must use the **same subword tokenizer the base model was trained with**;
the run asserts the tokenizer's vocab size matches the checkpoint.

## How the data is used

Each `<|endoftext|>`-delimited block looks like:

```
Features: Dialogue
Summary: Tom and Anna fly on a big plane to a sunny place with their parents.
Words: plane, holiday, excited
Story: 
Tom and Anna are brother and sister...
```

Everything up to and including `Story:` is the **prompt**; the story body is the
**response**. Prompt and response are tokenized separately (so the mask falls on
a clean token boundary) and an `<|endoftext|>` token is appended so the model
learns to stop. Training uses standard next-token prediction, but the loss is
masked (`ignore_index = -100`) over the prompt tokens so the model is only scored
on generating the story. Blocks without a `Story:` marker (e.g. the reverse
story→summary task) are skipped by default; set `require_story_marker` to `false`
in the config to train on them as plain text.

## Setup

Install dependencies:

```bash
pip install -r data/requirements.txt
pip install -r instruction-tuning/requirements.txt
```

Train the tokenizer (if not already done for pre-training) and download a
token-capped TinyStories-Instruct subset:

```bash
python data/train_tokenizer.py --vocab-size 4096
python data/download_tinystories_instruct.py subset --target-train-tokens 20000000
```

This writes `data/tinystories_instruct/train_20M_tokens.txt` and
`valid_1M_tokens.txt`, preserving the `<|endoftext|>` block delimiters. The SFT
config points at the tokenizer via `data.tokenizer_path` and at the base
checkpoint via `training.init_from`.

## Debug Run

Sanity-check the data, masking, and training loop on the small validation file:

```bash
python instruction-tuning/train.py instruction-tuning/configs/instruct_30m_debug.json
```

## Fine-Tuning Run

After the debug run is healthy, fine-tune the 30M base model:

```bash
python instruction-tuning/train.py instruction-tuning/configs/instruct_30m_rope.json
```

Key config fields:

- `training.init_from`: path to the pre-trained checkpoint to start from
  (default `pre-training/outputs/assignment_01_30m_rope/best.pt`)
- `model.block_size`: context length for fine-tuning (default `512`, matching the
  base model). Position buffers are rebuilt if you change it; learned weights
  transfer either way.
- `learning_rate`: lower than pre-training (`5e-5`), since we start from a trained model

The architecture (layers, heads, embedding size) is taken from the pre-trained
checkpoint, so the config only needs to override `block_size`. To benefit from a
long-context base, pretrain with `assignment_01_30m_rope_ctx1024.json` first,
then set `init_from` to that checkpoint and `model.block_size` to `1024` here.

Each run writes TensorBoard scalars to `<out_dir>/tensorboard/` and a raw
`<out_dir>/metrics.jsonl`:

```bash
tensorboard --logdir instruction-tuning/outputs
```

## Evaluation

Masked validation loss for the best checkpoint:

```bash
python instruction-tuning/eval.py instruction-tuning/configs/instruct_30m_rope.json
```

## Instruction-Following Metrics

Loss tells you about next-byte prediction, not whether the model *obeys* the
instruction. This generates a story per validation prompt and measures it:

```bash
python instruction-tuning/eval_following.py \
  --checkpoint instruction-tuning/outputs/instruct_30m_rope/best.pt \
  --valid-path data/tinystories_instruct/valid_1M_tokens.txt \
  --num-examples 100 --device cuda
```

- `words_used`: fraction of the requested `Words:` that appear in the story
- `sentence_exact`: fraction whose `Random sentence:` appears verbatim
- `sentence_fuzzy`: average longest-common-substring coverage of that sentence

Metrics are computed only on the generated continuation (the prompt is
excluded). The `reference` column is the ceiling — the score the ground-truth
stories achieve — so compare `generated` against it rather than against 1.0.
Generation is slow on CPU; use `--device cuda` and a modest `--num-examples`.

## Sampling

Generate a story from instruction fields:

```bash
python instruction-tuning/sample.py \
  --checkpoint instruction-tuning/outputs/instruct_30m_rope/best.pt \
  --summary "A little fish makes friends with a crab at the beach." \
  --words "fish, crab, friends" \
  --features "Dialogue"
```

Or pass a raw prompt that ends at `Story:`:

```bash
python instruction-tuning/sample.py \
  --checkpoint instruction-tuning/outputs/instruct_30m_rope/best.pt \
  --prompt "Summary: A cat learns to share its toys.
Story:"
```

# On-Policy Distillation

Train a new instruction model by **on-policy distillation**: the *student*
(starting from the pretrained base) generates its own responses to instruction
prompts, and a frozen *teacher* (the SFT instruct model) grades those responses
token-by-token. The student is trained to match the teacher's distribution on
the tokens **it itself produced**.

```
prompt ──► student samples a response ──► teacher scores each token ──► KL loss ──► update student
              (on-policy rollout)            (dense per-token target)
```

## Why this, and how it differs from SFT

| method | training sequences | target | signal |
|---|---|---|---|
| **SFT** (`instruction-tuning/`) | fixed ground-truth stories | the one correct next token (one-hot) | dense, **off-policy** |
| **On-policy distillation** (here) | the **student's own** generations | teacher's full next-token distribution | **dense + on-policy** |
| RL / RLHF | the student's own generations | one scalar reward at the end | sparse, on-policy |

On-policy distillation combines the strengths of the other two: like RL, the
student learns on the states **it actually visits at inference** (no exposure
bias from only ever seeing ground-truth prefixes); like SFT, it gets a **dense**
signal — a full target distribution at *every* token — instead of one sparse
reward. The per-token `-KL(student ‖ teacher)` is effectively a dense reward.
(See GKD, Agarwal et al. 2023, for the on/off-policy + KL-direction framework.)

## What to expect here (be realistic)

Teacher and student are the **same 35M architecture**, and the teacher only
knows what SFT taught it. So distillation **cannot exceed the teacher** — the
best case is the student *matches* it. The point of this folder is to **learn
the technique** and to observe a real property: because the targets are soft
distributions on the student's own states, on-policy distillation often
**generalizes a bit better / overfits less** than the equivalent SFT. The
real-world payoff (compressing a *larger* teacher into a smaller student) is the
same algorithm at a different scale.

A nice consequence of this project's tiny **vocab (4096)**: we compute the
**exact full-distribution KL** at every token. Production LLMs (vocab 50k–200k)
must approximate this with top-k or sampled tokens; here you get the clean
version.

## The loss, precisely

For each response position `t` (prompt and padding masked out):

- **reverse KL** (default, mode-seeking): `Σ_v p_S(v) · [log p_S(v) − log p_T(v)]`
- **forward KL** (mode-covering, classic distillation): `Σ_v p_T(v) · [log p_T(v) − log p_S(v)]`

`p_S` = student softmax (with gradient), `p_T` = teacher softmax (detached). Set
`distill.kl_type` to `"reverse"` or `"forward"` to compare the two directions.
Only positions whose *target* token is part of the student's generated response
contribute (same masking idea as SFT).

## Requirements

```bash
pip install -r on-policy-distillation/requirements.txt
```

Teacher, student, and tokenizer must all share the **same BPE vocab (4096)** —
the KL aligns the two models token-by-token. The run asserts this.

## Debug run

Sanity-check rollouts, masking, and the loss on a tiny slice (20 steps):

```bash
python on-policy-distillation/distill.py on-policy-distillation/configs/distill_30m_debug.json
```

## Full run

```bash
python on-policy-distillation/distill.py on-policy-distillation/configs/distill_30m.json
```

Key config fields:

- `distill.teacher_from`: the frozen SFT checkpoint (default the instruct model).
- `distill.student_init_from`: the student's starting weights (default the base).
- `distill.kl_type`: `reverse` (default) or `forward`.
- `distill.rollout_temperature` / `rollout_top_k`: how the student samples its
  on-policy responses (default `1.0` / none = pure on-policy).
- `distill.kl_temperature`: softmax temperature for the KL targets (default `1.0`).
- `distill.max_new_tokens`: rollout length cap (prompt + response must fit
  `block_size`). **Generation is the bottleneck** — bigger = slower per step.
- `training.batch_size` × `gradient_accumulation_steps`: rollouts per optimizer
  step. Each rollout is generated sequentially, so keep these modest on a single
  older GPU.

Metrics never overwrite: each run writes `metrics_<timestamp>.jsonl` and a
`tensorboard/<timestamp>/` subdir.

```bash
tensorboard --logdir on-policy-distillation/outputs
```

`train/kl` and `eval/valid_kl` should fall toward 0 as the student's
distribution approaches the teacher's. `best.pt` is the lowest-`valid_kl`
checkpoint.

## Evaluating the student

The student is saved in the **same checkpoint format** as the other stages, so
the instruction-tuning eval/sample scripts work on it directly:

```bash
# instruction-following metrics (compare against the teacher's numbers)
python instruction-tuning/eval_following.py \
  --checkpoint on-policy-distillation/outputs/distill_30m/best.pt \
  --valid-path data/tinystories_instruct/full/TinyStories-Instruct-valid.txt \
  --num-examples 100 --device cuda

# generate a story
python instruction-tuning/sample.py \
  --checkpoint on-policy-distillation/outputs/distill_30m/best.pt \
  --summary "A little fish makes friends with a crab at the beach." \
  --words "fish, crab, friends" --features "Dialogue"
```

The interesting comparison is the student's `words_used` / `sentence_exact` /
`sentence_fuzzy` against the teacher's (the SFT model's) scores — does on-policy
distillation recover the teacher's instruction-following?

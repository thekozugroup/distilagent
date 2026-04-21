# MLX LoRA training data

Two variants of the pilot dataset (`../distilagent_pilot.jsonl`), each split 80/10/10 into `train.jsonl` / `valid.jsonl` / `test.jsonl`. Format is `mlx-lm.lora --data` compatible (messages-style JSONL).

## Variants

| dir | assistant target | use when |
|-----|------------------|----------|
| `gen_only/` | the tournament-refined final answer only | classic SFT — faster to train, safer first attempt |
| `cot/` | `<thinking>` + teacher reasoning + `</thinking>` + final answer | process-supervision — student learns the reasoning too |

The `cot` variant includes *only* the teacher role's reasoning — critic/author/judge reasoning is tournament machinery and stays out of student targets.

## Quick start — LoRA fine-tune on Apple Silicon

```bash
# Install mlx-lm if needed
pip install -U mlx-lm

# Train — pick a base model from mlx-community (see note below)
mlx_lm.lora \
  --model mlx-community/Qwen3-4B-Instruct-4bit \
  --train \
  --data datasets/mlx_lora/gen_only \
  --iters 400 \
  --batch-size 1 \
  --num-layers 16 \
  --learning-rate 1e-4 \
  --adapter-path adapters/qwen3-4b-gen-only

# Evaluate on the held-out test set
mlx_lm.lora \
  --model mlx-community/Qwen3-4B-Instruct-4bit \
  --adapter-path adapters/qwen3-4b-gen-only \
  --data datasets/mlx_lora/gen_only \
  --test

# Chat with the tuned adapter
mlx_lm.generate \
  --model mlx-community/Qwen3-4B-Instruct-4bit \
  --adapter-path adapters/qwen3-4b-gen-only \
  --prompt "You are planning a task: migrate PostgreSQL 13 → 16 with zero downtime. Produce a 6-step plan and flag the riskiest step."
```

## Base model choice

`mlx-community/Qwen3-4B-Instruct-4bit` is a reasonable first target at this dataset scale. Alternatives worth trying:

- `mlx-community/Qwen3-1.7B-Instruct-4bit` — faster iteration, weaker ceiling
- `mlx-community/Qwen3-8B-Instruct-4bit` — better quality, needs more RAM
- `mlx-community/Qwen3-14B-Instruct-4bit` — MacBook Pro / M3 Max territory
- `mlx-community/Llama-3.2-3B-Instruct-4bit` — alternative family

## Caveats at n=25

This pilot is **too small to generalize**. Expect to see loss drop nicely on training data and **probably overfit** — the model will learn the 21 training answers but not the *skill*. That's expected: this run is to validate the training loop end-to-end, not to produce a production adapter. Scale to 500-5000 samples before evaluating student quality meaningfully.

## Files

- `gen_only/train.jsonl` — 21 rows, user→assistant pairs
- `gen_only/valid.jsonl` — 2 rows held-out for training-time eval
- `gen_only/test.jsonl` — 2 rows untouched for final eval
- `cot/...` — same splits, with `<thinking>` block in assistant targets

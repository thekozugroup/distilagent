DistilAgent fuses [distilabel](https://github.com/argilla-io/distilabel) with [NousResearch's AutoReason](https://github.com/NousResearch/autoreason) tournament refinement so one teacher model produces distillation labels that are measurably better than its own single-shot output. Each prompt is answered, critiqued, adversarially revised, synthesized, and ranked by a blind Borda panel until "do nothing" wins twice. Full reasoning traces are captured per role for process-supervision fine-tuning.

## Screenshots

![DistilAgent pipeline — raw prompts funnel through an A/B/AB tournament into a condensed refined dataset](./docs/screenshot.png)

## How it works

A new `AutoReasonedGeneration` Task drops into any distilabel pipeline. Each prompt runs through five fresh-context agent roles, each solving a different problem:

- **Teacher** — writes the incumbent draft **A**. Defines the quality ceiling; should be the strongest model you can afford.
- **Critic** — finds *concrete, quotable* flaws in A or replies exactly `NO FLAWS`. Needs discrimination, not creativity. Anti-hallucination directives keep it honest.
- **Author B** — adversarial revision. Rewrites A to address the critique without padding or scope creep.
- **Synthesizer** — conservative synthesis **AB**. Minimum repair: keep what worked, fix only what was flagged.
- **Judges (N)** — blind Borda panel. Each judge sees A, B, AB under randomised labels and returns one ranking line. Panel size beats panel quality — 7 noisy judges outperform 3 careful ones.

If A defends twice in a row the tournament converges. Otherwise the winner is promoted and the loop repeats. Every call is rate-limited and retry-wrapped: 429s, 503s, Metal working-set rejections, and httpx timeouts all backoff-and-retry instead of dropping votes.

A 69-sample pilot on Gemma 4 26B-A4B (local vMLX, thinking on) produced rich traces at ~8 min/sample with zero unrecovered failures. Blind external-judge evals on a prior pilot picked AutoReason over single-shot baseline on all 3 randomised runs.

## Use cases

- **SFT fine-tuning** — train students on the refined `generation` field (higher-quality targets than raw teacher output).
- **DPO / preference training** — every tournament iteration produces winner/loser pairs with Borda scores for free.
- **Chain-of-thought distillation** — per-role reasoning traces teach students *how* to think, not just what to say.
- **Critic-model training** — `(draft, critique, revision)` triples for self-correcting students.
- **Agentic specialisation** — paired hand-authored agentic prompts cover tool use, planning, reflection, error recovery, and multi-agent coordination.

## How this rethinks distillation

Vanilla distillation is *one teacher pass per prompt* — whatever the model says first is what the student learns. Best-of-N sampling picks the best of several drafts but still treats each draft as atomic. AutoReason refuses both frames: the teacher's first answer is a starting point, its own critic is invited to stress-test it, and "no change needed" is a first-class winning move rather than a default assumption. The same model is three different workers, and self-evaluation is almost always stronger than self-generation — that's the lever we pull.

## Stack

- Python 3.11, asyncio, pydantic v2
- distilabel (forked from develop) as the pipeline runtime
- AutoReason tournament (A / B / AB + blind Borda) adapted from the NousResearch paper
- OpenAI async SDK — OpenRouter, local vMLX, any OpenAI-compatible endpoint
- MLX-LM for on-device Apple Silicon inference (Gemma 4, Qwen3)
- pytest + pytest-asyncio (61 unit + integration tests)
- Hugging Face datasets (`ianncity/General-Distillation-Prompts-1M`)

## Status

Active

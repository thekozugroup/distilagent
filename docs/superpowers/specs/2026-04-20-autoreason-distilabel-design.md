# AutoReason × Distilabel — Design Spec

**Date:** 2026-04-20
**Status:** Approved
**Goal:** Fork `distilabel` (develop branch), add `AutoReasonedGeneration` Task that applies the AutoReason tournament refinement loop (A/B/AB + blind Borda judges) to teacher-model generations during distillation. Maximizes label quality before student training.

## Approved Choices

| # | Decision | Choice |
|---|---|---|
| 1 | Integration style | Fork distilabel, add new Task module (no upstream modifications) |
| 2 | Model-per-role | Single teacher LLM default, hooks for per-role override later |
| 3 | Output format | Full trace preserved in dataset (winner + all iterations + votes) |
| 4 | Compute defaults | Paper-faithful: 7 judges, max 15 iterations, convergence k=2 |
| 5 | API shape | Wrapper `Task` (`AutoReasonedGeneration`) — works with any teacher prompt |
| 6 | Concurrency | Iteration-level async (parallel judges + B/AB authoring) + per-model token-bucket rate limiter. Ship first; sample-level concurrency + model-fallback chains as follow-up |
| + | Constraint | Free-tier OpenRouter protection — respect RPM/RPD, exp backoff on 429 |

## Architecture

### Module Layout
```
src/distilabel/steps/tasks/autoreason/
├── __init__.py          # Exports AutoReasonedGeneration
├── task.py              # AutoReasonedGeneration(Task)
├── tournament.py        # Tournament loop + convergence
├── roles.py             # Role prompt templates (critic, author_b, synthesizer, judge)
├── borda.py             # Blind Borda count
├── rate_limit.py        # Per-model AsyncTokenBucket (RPM + RPD)
└── types.py             # TraceEntry, IterationResult, JudgeVote dataclasses

tests/unit/steps/tasks/autoreason/
├── __init__.py
├── test_borda.py        # Unit — pure function, easy
├── test_rate_limit.py   # Unit — token bucket semantics, 429 backoff
├── test_roles.py        # Unit — prompt templates render correctly
├── test_tournament.py   # Unit — loop with mocked LLM
└── test_task.py         # Integration — full AutoReasonedGeneration with DummyLLM
```

### Task API

```python
from distilabel.steps.tasks import AutoReasonedGeneration
from distilabel.models import OpenAILLM

task = AutoReasonedGeneration(
    llm=OpenAILLM(model="openrouter/auto", base_url="https://openrouter.ai/api/v1"),
    max_iterations=15,
    num_judges=7,
    convergence_k=2,
    rpm=20,            # free-tier OpenRouter default
    rpd=1000,
    max_concurrency=8, # semaphore across role calls
)
```

### Input Columns
- `instruction` (str) — the task prompt (same as `TextGeneration`)

### Output Columns
- `generation` (str) — final refined winner
- `autoreason_trace` (dict) — full trace: `{iterations: [...], converged: bool, winner_source: "A"|"B"|"AB", total_calls: int}`
- `autoreason_iterations` (int) — how many iterations ran
- `model_name` (str) — teacher model

Each trace iteration entry:
```python
{
  "iteration": 0,
  "A": "...incumbent text...",
  "critique": "...",
  "B": "...adversarial revision...",
  "AB": "...synthesis...",
  "votes": [{"judge_id": 0, "ranking": ["AB", "B", "A"]}, ...],
  "borda": {"A": 7, "B": 12, "AB": 14},
  "winner": "AB",
}
```

### Tournament Loop

```
Given prompt P:
  A = teacher_generate(P)                       # seed incumbent
  consecutive_A_wins = 0
  trace = []
  for i in 0..max_iterations:
      critique = critic(P, A)                   # 1 call
      # PARALLEL
      B  = author_b(P, A, critique)             # 1 call
      AB = synthesizer(P, A, critique)          # 1 call
      # PARALLEL (N judges, fresh context each)
      votes = [judge(P, A, B, AB) for _ in num_judges]  # N calls
      borda = borda_count(votes)
      winner = argmax(borda)
      trace.append(...)
      if winner == "A":
          consecutive_A_wins += 1
          if consecutive_A_wins >= convergence_k:
              break                             # A defended k times → done
      else:
          consecutive_A_wins = 0
          A = {B, AB}[winner]                   # promote new incumbent
  return A, trace
```

### Role Prompts (roles.py)

Each role is a Jinja2 template with explicit role framing. Fresh chat context per call — no role sees another's reasoning. Templates:

- **Teacher seed** — plain task prompt (no priming).
- **Critic** — "Identify concrete, specific flaws in the draft below. If there are none, say 'NO FLAWS'."
- **Author B** — "Given the draft and this critique, write an improved version. Address the critique; do not expand scope."
- **Synthesizer** — "Given the draft and this critique, produce a synthesis that keeps the draft's strengths and repairs only the issues the critique identifies."
- **Judge** — "Rank these three anonymized responses (labeled X1/X2/X3) from best to worst. Reply with exactly 'RANKING: X? > X? > X?'. Labels are randomized per judge (blind)."

Blinding: each judge receives {A, B, AB} under randomly permuted labels (X1/X2/X3) to prevent position bias. Label→role map is stored in the trace for audit.

### Borda Count (borda.py)

Pure function. For each judge's ranking, candidates receive points:
- 1st place → 2 points
- 2nd place → 1 point
- 3rd place → 0 points

Sum across judges. Highest total = winner. On tie, prefer "A" (do-nothing bias — matches paper's anti-scope-creep principle).

### Rate Limiter (rate_limit.py)

`AsyncTokenBucket` keyed by model name:
- **RPM bucket**: refills continuously at rate RPM/60 per second, capacity = RPM.
- **RPD bucket**: refills once per 24h, capacity = RPD.
- `async acquire(n=1)`: blocks until tokens available in both buckets, decrements both.
- Wraps LLM calls: on HTTP 429, apply exponential backoff (1s, 2s, 4s, ..., max 60s), retry up to 5 times.
- On RPD exhaustion: configurable policy — `"wait"` (sleep until midnight UTC) or `"fail"` (raise).

Limiter is a singleton-per-model registry: `get_limiter(model_name, rpm, rpd)`.

### Concurrency Inside an Iteration

```python
# After critic (sequential)
B_task = asyncio.create_task(author_b(...))
AB_task = asyncio.create_task(synthesizer(...))
B, AB = await asyncio.gather(B_task, AB_task)

# Judges in parallel, bounded by semaphore
sem = asyncio.Semaphore(max_concurrency)
async def bounded_judge(i):
    async with sem:
        return await judge_role(P, A, B, AB, seed=i)
votes = await asyncio.gather(*[bounded_judge(i) for i in range(num_judges)])
```

All wrapped by the rate limiter's `acquire()`.

### Error Handling

- **Malformed judge ranking** (can't parse `RANKING: X? > X? > X?`): drop that judge's vote. If >50% of judges fail to produce a parseable ranking, abort iteration and keep current A (fail-safe toward "do nothing").
- **Critic returns "NO FLAWS"**: skip iteration, increment `consecutive_A_wins`. If k reached, converge.
- **API timeout/error after retries**: propagate as `AutoReasonError`; distilabel's `impute_step_outputs` handles row-level failures.
- **Empty/null generation from teacher seed**: abort with clear error; retry handled at distilabel's pipeline level.

## Testing Strategy

- **Unit tests** (fast, deterministic): Borda math, rate-limit token-bucket behavior, Jinja template rendering, tournament loop with `DummyLLM` returning canned sequences.
- **Integration test** (mock LLM): Full `AutoReasonedGeneration.process()` on a 2-row fake dataset using a scripted `DummyLLM` that exercises: (a) converges at k=2, (b) hits max_iterations, (c) NO FLAWS early exit, (d) malformed judge handling, (e) rate-limited backoff.
- **Smoke test** (optional, behind env flag `AUTOREASON_LIVE_TEST=1` + `OPENROUTER_API_KEY`): 1 instruction end-to-end against free OpenRouter model. Not run in CI.

Target: 100% unit coverage on `borda.py`, `rate_limit.py`, `roles.py`, `tournament.py`. Integration test asserts trace shape + winner progression.

## Non-Goals (for this iteration)

- Per-role model configuration (scaffolded via `role_llms: dict | None = None` field, default None → use `self.llm` for all).
- Multi-model fallback chains.
- Process-supervision student-training format converters (keep trace generic; conversion is downstream concern).
- Sample-level concurrency (distilabel pipeline handles that natively).
- Judge panel ≠ 7 optimization (configurable but default is paper-faithful).

## Dependencies

New runtime deps to add to `pyproject.toml`:
- `aiolimiter` (token-bucket primitive) — or implement inline to avoid new dep. Decision: **implement inline** (no new dep; token bucket is ~50 LOC).

No other new deps — Jinja2, asyncio, pydantic already in distilabel.

## Rollout Plan

1. Parallel agents build: `borda.py`, `rate_limit.py`, `roles.py`, `types.py`, `tournament.py` + tests.
2. Sequential: `task.py` integration, module `__init__.py`, wire into `distilabel/steps/tasks/__init__.py` export.
3. Run full test suite; iterate until green.
4. Report ready for API key (end-to-end live smoke test with user's OpenRouter key).

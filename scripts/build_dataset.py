"""Build a distillation dataset using AutoReasonedGeneration.

Sources:
  * ianncity/General-Distillation-Prompts-1M  (HF, 1.28M lines)
  * scripts/agentic_prompts.py                (hand-authored agentic workflows)

Features:
  * Per-prompt checkpointing — resumable on 429, crash, or Ctrl-C.
  * JSONL append-only output (one row per completed sample).
  * Per-category budget controls via env vars.
  * Conservative AutoReason config tuned for free-tier throughput.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from agentic_prompts import get_prompts as get_agentic_prompts  # noqa: E402
from reasoning_llm import LocalInlineThinkLLM, ReasoningOpenRouterLLM  # noqa: E402
from pools import (  # noqa: E402
    CODE_SPECIALIST,
    GENERALIST_POOL,
    JUDGE_POOL,
    REASONING_SPECIALIST,
    classify_route,
    route_teacher,
)

from distilabel.steps.tasks import AutoReasonedGeneration  # noqa: E402  # re-exported for CLI smoke  # noqa: F401
from distilabel.steps.tasks.autoreason.rate_limit import get_limiter  # noqa: E402
from distilabel.steps.tasks.autoreason.tournament import TournamentRunner  # noqa: E402


@dataclass
class Config:
    out_jsonl: Path
    checkpoint_json: Path
    provider: str            # "openrouter" | "local"
    base_url: str
    model: str
    n_hf: int
    hf_seed: int
    include_agentic: bool
    num_judges: int
    max_iterations: int
    convergence_k: int
    max_concurrency: int
    rpm: int
    rpd: int
    temperature: Optional[float]
    top_p: Optional[float]
    repetition_penalty: Optional[float]
    enable_thinking: bool
    max_tokens: Optional[int]
    use_pool: bool
    teacher_pool: List[str]
    judge_pool: List[str]
    rng_seed: int


def prompt_id(text: str, prefix: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{h}"


def load_hf_samples(n: int, seed: int) -> List[Dict]:
    """Random-sample n lines from the 1M-line prompts.txt."""
    path = hf_hub_download(
        "ianncity/General-Distillation-Prompts-1M",
        filename="prompts.txt",
        repo_type="dataset",
    )
    rng = random.Random(seed)
    # Reservoir sample for fairness across 1.28M lines.
    reservoir: List[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i < n:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = line
    return [
        {
            "id": prompt_id(t, "hf"),
            "source": "ianncity/General-Distillation-Prompts-1M",
            "category": "general",
            "instruction": t,
        }
        for t in reservoir
    ]


def load_agentic_samples() -> List[Dict]:
    return [
        {
            "id": p["id"],
            "source": "distilagent/agentic-authored",
            "category": p["category"],
            "instruction": p["instruction"],
        }
        for p in get_agentic_prompts()
    ]


def load_checkpoint(path: Path) -> Dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_ids": [], "failed": {}}


def save_checkpoint(path: Path, state: Dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(path)


def _all_llms(runner: TournamentRunner) -> List:
    """Return every unique LLM instance the runner might call."""
    seen = {id(runner.llm): runner.llm}
    for v in runner.role_llms.values():
        seen.setdefault(id(v), v)
    for v in runner.judge_pool:
        seen.setdefault(id(v), v)
    return list(seen.values())


async def run_one(
    runner: TournamentRunner, sample: Dict
) -> Optional[Dict]:
    # Reset reasoning buffers on every LLM the runner may use.
    llms = _all_llms(runner)
    for llm in llms:
        if hasattr(llm, "drain_reasoning"):
            llm.drain_reasoning()

    t0 = time.monotonic()
    try:
        trace = await runner.run(sample["instruction"])
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    elapsed = time.monotonic() - t0

    # Merge reasoning buffers across all participating LLMs, tagging each
    # entry with the source model so pool-mode datasets stay auditable.
    reasoning_entries = []
    for llm in llms:
        if not hasattr(llm, "drain_reasoning"):
            continue
        mn = getattr(llm, "model_name", "unknown")
        for role_hint, reasoning, text in llm.drain_reasoning():
            reasoning_entries.append(
                {
                    "call_index": len(reasoning_entries),
                    "role_hint": role_hint,
                    "source_model": mn,
                    "reasoning": reasoning,
                    "text": text,
                }
            )

    row = {
        **sample,
        "generation": trace.final_answer,
        "autoreason_trace": trace.to_dict(),
        "autoreason_iterations": len(trace.iterations),
        "autoreason_converged": trace.converged,
        "total_calls": trace.total_calls,
        "elapsed_seconds": round(elapsed, 2),
        "model_name": runner.llm.model_name,
        "reasoning_trace": reasoning_entries,
        "reasoning_total_chars": sum(len(e["reasoning"]) for e in reasoning_entries),
    }
    return row


async def amain(cfg: Config) -> int:
    samples: List[Dict] = []
    if cfg.n_hf > 0:
        print(f"[load] sampling {cfg.n_hf} prompts from HF...", flush=True)
        samples.extend(load_hf_samples(cfg.n_hf, cfg.hf_seed))
    if cfg.include_agentic:
        agentic = load_agentic_samples()
        print(f"[load] {len(agentic)} agentic-authored prompts", flush=True)
        samples.extend(agentic)

    checkpoint = load_checkpoint(cfg.checkpoint_json)
    done_ids = set(checkpoint["completed_ids"])
    remaining = [s for s in samples if s["id"] not in done_ids]
    print(
        f"[plan] total={len(samples)}  done={len(done_ids)}  remaining={len(remaining)}",
        flush=True,
    )

    # Build generation_kwargs from sampling + thinking config.
    gen_kwargs: Dict[str, Any] = {}
    if cfg.temperature is not None:
        gen_kwargs["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        gen_kwargs["top_p"] = cfg.top_p
    if cfg.max_tokens is not None:
        gen_kwargs["max_tokens"] = cfg.max_tokens
    extra_body: Dict[str, Any] = {}
    if cfg.repetition_penalty is not None:
        extra_body["repetition_penalty"] = cfg.repetition_penalty
    if cfg.enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    if extra_body:
        gen_kwargs["extra_body"] = extra_body

    # Thinking-capable models can take minutes per call; raise the HTTP
    # client timeout well above the 120s distilabel default.
    call_timeout_s = int(os.environ.get("LLM_CALL_TIMEOUT", "900"))

    def _make_llm(model_id: str):
        if cfg.provider == "openrouter":
            return ReasoningOpenRouterLLM(
                model=model_id,
                base_url=cfg.base_url,
                api_key=os.environ["OPENROUTER_API_KEY"],
                reasoning_effort=os.environ.get("REASONING_EFFORT", "high"),
                generation_kwargs=gen_kwargs,
                timeout=call_timeout_s,
            )
        if cfg.provider == "local":
            return LocalInlineThinkLLM(
                model=model_id,
                base_url=cfg.base_url,
                api_key=os.environ.get("OPENROUTER_API_KEY", "local-dummy"),
                generation_kwargs=gen_kwargs,
                timeout=call_timeout_s,
            )
        raise ValueError(f"unknown provider: {cfg.provider!r}")

    llm_cache: Dict[str, Any] = {}

    def _get_llm(model_id: str):
        if model_id not in llm_cache:
            obj = _make_llm(model_id)
            obj.load()
            llm_cache[model_id] = obj
        return llm_cache[model_id]

    def _get_limiter(model_id: str):
        return get_limiter(name=model_id, rpm=cfg.rpm, rpd=cfg.rpd)

    # Pre-warm cache for pool mode so first sample isn't a cold start.
    if cfg.use_pool:
        all_ids = set(cfg.teacher_pool) | set(cfg.judge_pool) | {
            CODE_SPECIALIST, REASONING_SPECIALIST,
        }
        print(f"[pool] warming {len(all_ids)} LLM clients: "
              f"teachers={len(cfg.teacher_pool)} + judges={len(cfg.judge_pool)} "
              f"+ 2 specialists", flush=True)
        for mid in all_ids:
            _get_llm(mid)

    # Single-model mode (backward compat with the --model flag).
    single_llm = None if cfg.use_pool else _get_llm(cfg.model)
    single_limiter = None if cfg.use_pool else _get_limiter(cfg.model)

    def build_runner_for(sample: Dict) -> TournamentRunner:
        if not cfg.use_pool:
            return TournamentRunner(
                llm=single_llm,
                num_judges=cfg.num_judges,
                max_iterations=cfg.max_iterations,
                convergence_k=cfg.convergence_k,
                max_concurrency=cfg.max_concurrency,
                rate_limiter=single_limiter,
                rng_seed_base=cfg.rng_seed,
            )

        # Pool mode: route teacher by domain, reuse for author/synth,
        # use mixed judge pool for discriminators.
        teacher_id = route_teacher(
            sample["instruction"], cfg.teacher_pool, seed=cfg.rng_seed
        )
        teacher_llm = _get_llm(teacher_id)
        role_llms = {
            "teacher": teacher_llm,
            "author_b": teacher_llm,      # match teacher tier
            "synthesizer": teacher_llm,   # match teacher tier
            "critic": teacher_llm,        # keep critic strong; cheap to downgrade later
        }
        judge_llms = [_get_llm(mid) for mid in cfg.judge_pool]
        limiter_map = {
            mid: _get_limiter(mid)
            for mid in ({teacher_id} | set(cfg.judge_pool))
        }
        sample["_teacher_id"] = teacher_id
        sample["_route"] = classify_route(sample["instruction"])
        sample["_judge_pool"] = list(cfg.judge_pool)
        return TournamentRunner(
            llm=teacher_llm,
            num_judges=cfg.num_judges,
            max_iterations=cfg.max_iterations,
            convergence_k=cfg.convergence_k,
            max_concurrency=cfg.max_concurrency,
            rate_limiter=limiter_map.get(teacher_id),
            rng_seed_base=cfg.rng_seed,
            role_llms=role_llms,
            judge_pool=judge_llms,
            limiter_map=limiter_map,
        )

    cfg.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_fh = open(cfg.out_jsonl, "a", encoding="utf-8")

    try:
        for i, sample in enumerate(remaining):
            label = f"{i + 1}/{len(remaining)}"
            runner = build_runner_for(sample)
            tag = ""
            if cfg.use_pool:
                tag = f" [route={sample.get('_route')} teacher={sample.get('_teacher_id')}]"
            print(
                f"\n[{label}] {sample['id']} [{sample['category']}]{tag} "
                f"\"{sample['instruction'][:80]}...\"",
                flush=True,
            )
            row = await run_one(runner, sample)
            if row is None or "error" in row:
                err = (row or {}).get("error", "unknown")
                print(f"  ✗ FAIL: {err}", flush=True)
                checkpoint["failed"][sample["id"]] = err
                save_checkpoint(cfg.checkpoint_json, checkpoint)
                continue
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_fh.flush()
            checkpoint["completed_ids"].append(sample["id"])
            save_checkpoint(cfg.checkpoint_json, checkpoint)
            trace_summary = (
                f"iters={row['autoreason_iterations']} "
                f"converged={row['autoreason_converged']} "
                f"calls={row['total_calls']} "
                f"{row['elapsed_seconds']}s"
            )
            print(f"  ✓ {trace_summary}", flush=True)
    finally:
        out_fh.close()

    print(
        f"\n[done] completed {len(checkpoint['completed_ids'])} "
        f"failed {len(checkpoint['failed'])}"
    )
    print(f"output: {cfg.out_jsonl}")
    return 0


def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/distilagent_pilot.jsonl"))
    ap.add_argument("--checkpoint", type=Path, default=Path("datasets/.checkpoint.json"))
    ap.add_argument(
        "--provider",
        choices=["openrouter", "local"],
        default=os.environ.get("PROVIDER", "openrouter"),
    )
    ap.add_argument("--base-url", default=None,
                    help="override default base URL (defaults: openrouter=https://openrouter.ai/api/v1, "
                         "local=http://localhost:8081/v1)")
    ap.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", "minimax/minimax-m2.5:free"))
    ap.add_argument("--n-hf", type=int, default=10)
    ap.add_argument("--hf-seed", type=int, default=2026)
    ap.add_argument("--no-agentic", action="store_true")
    ap.add_argument("--num-judges", type=int, default=3)
    ap.add_argument("--max-iterations", type=int, default=2)
    ap.add_argument("--convergence-k", type=int, default=2)
    ap.add_argument("--max-concurrency", type=int, default=3)
    ap.add_argument("--rpm", type=int, default=15)
    ap.add_argument("--rpd", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="pass chat_template_kwargs.enable_thinking=True to the server")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap completion tokens per call (forwarded as OpenAI max_tokens)")
    ap.add_argument("--use-pool", action="store_true",
                    help="route teacher per prompt + use mixed judge pool (pool mode)")
    ap.add_argument("--teacher-pool", type=str, default=None,
                    help="comma-separated teacher model IDs; defaults to pools.GENERALIST_POOL")
    ap.add_argument("--judge-pool", type=str, default=None,
                    help="comma-separated judge model IDs; defaults to pools.JUDGE_POOL")
    ap.add_argument("--rng-seed", type=int, default=42)
    args = ap.parse_args()
    base_url = args.base_url or (
        "http://localhost:8081/v1" if args.provider == "local" else "https://openrouter.ai/api/v1"
    )
    teacher_pool = (
        [s.strip() for s in args.teacher_pool.split(",") if s.strip()]
        if args.teacher_pool else list(GENERALIST_POOL)
    )
    judge_pool = (
        [s.strip() for s in args.judge_pool.split(",") if s.strip()]
        if args.judge_pool else list(JUDGE_POOL)
    )
    return Config(
        out_jsonl=args.out,
        checkpoint_json=args.checkpoint,
        provider=args.provider,
        base_url=base_url,
        model=args.model,
        n_hf=args.n_hf,
        hf_seed=args.hf_seed,
        include_agentic=not args.no_agentic,
        num_judges=args.num_judges,
        max_iterations=args.max_iterations,
        convergence_k=args.convergence_k,
        max_concurrency=args.max_concurrency,
        rpm=args.rpm,
        rpd=args.rpd,
        use_pool=args.use_pool,
        teacher_pool=teacher_pool,
        judge_pool=judge_pool,
        rng_seed=args.rng_seed,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(parse_args())))

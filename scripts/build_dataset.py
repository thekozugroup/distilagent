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
from typing import Dict, Iterable, List, Optional

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from agentic_prompts import get_prompts as get_agentic_prompts  # noqa: E402

from distilabel.models import OpenAILLM  # noqa: E402
from distilabel.steps.tasks import AutoReasonedGeneration  # noqa: E402
from distilabel.steps.tasks.autoreason.rate_limit import get_limiter  # noqa: E402
from distilabel.steps.tasks.autoreason.tournament import TournamentRunner  # noqa: E402


@dataclass
class Config:
    out_jsonl: Path
    checkpoint_json: Path
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


async def run_one(
    runner: TournamentRunner, sample: Dict
) -> Optional[Dict]:
    t0 = time.monotonic()
    try:
        trace = await runner.run(sample["instruction"])
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    elapsed = time.monotonic() - t0
    return {
        **sample,
        "generation": trace.final_answer,
        "autoreason_trace": trace.to_dict(),
        "autoreason_iterations": len(trace.iterations),
        "autoreason_converged": trace.converged,
        "total_calls": trace.total_calls,
        "elapsed_seconds": round(elapsed, 2),
        "model_name": runner.llm.model_name,
    }


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

    llm = OpenAILLM(
        model=cfg.model,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    llm.load()

    limiter = get_limiter(name=cfg.model, rpm=cfg.rpm, rpd=cfg.rpd)
    runner = TournamentRunner(
        llm=llm,
        num_judges=cfg.num_judges,
        max_iterations=cfg.max_iterations,
        convergence_k=cfg.convergence_k,
        max_concurrency=cfg.max_concurrency,
        rate_limiter=limiter,
        rng_seed_base=42,
    )

    cfg.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_fh = open(cfg.out_jsonl, "a", encoding="utf-8")

    try:
        for i, sample in enumerate(remaining):
            label = f"{i + 1}/{len(remaining)}"
            print(
                f"\n[{label}] {sample['id']} [{sample['category']}] "
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
    args = ap.parse_args()
    return Config(
        out_jsonl=args.out,
        checkpoint_json=args.checkpoint,
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
    )


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(parse_args())))

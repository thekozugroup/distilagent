"""Live smoke test for AutoReasonedGeneration against OpenRouter free-tier.

Reads OPENROUTER_API_KEY and OPENROUTER_MODEL from the environment.
Uses a modest config (3 judges, 1 iteration) to limit call volume.
"""
from __future__ import annotations

import json
import os
import sys
import time

from distilabel.models import OpenAILLM
from distilabel.steps.tasks import AutoReasonedGeneration

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ.get("OPENROUTER_MODEL", "minimax/minimax-m2.5:free")
INSTRUCTION = os.environ.get(
    "OPENROUTER_INSTRUCTION",
    "In three sentences, explain what makes the Transformer architecture different from RNNs.",
)


def main() -> int:
    print(f"Model: {MODEL}")
    print(f"Instruction: {INSTRUCTION}\n")

    llm = OpenAILLM(
        model=MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
    )

    task = AutoReasonedGeneration(
        llm=llm,
        num_judges=int(os.environ.get("AR_JUDGES", "3")),
        max_iterations=int(os.environ.get("AR_MAX_ITERS", "1")),
        convergence_k=int(os.environ.get("AR_K", "1")),
        max_concurrency=int(os.environ.get("AR_CONC", "3")),
        rpm=int(os.environ.get("AR_RPM", "15")),
        rpd=int(os.environ.get("AR_RPD", "200")),
        rng_seed_base=42,
    )
    task.load()

    t0 = time.monotonic()
    try:
        batch = next(task.process([{"instruction": INSTRUCTION}]))
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.monotonic() - t0

    row = batch[0]
    trace = row.get("autoreason_trace") or {}

    print(f"=== Final answer ({elapsed:.1f}s) ===")
    print(row.get("generation"))
    print("\n=== Trace summary ===")
    print(f"iterations:  {row.get('autoreason_iterations')}")
    print(f"converged:   {row.get('autoreason_converged')}")
    print(f"total_calls: {trace.get('total_calls')}")
    for i, it in enumerate(trace.get("iterations", [])):
        print(f"\n--- iter {i} ---")
        print(f"  winner:   {it.get('winner')} (no_flaws={it.get('no_flaws')})")
        print(f"  borda:    {it.get('borda')}")
        print(f"  critique: {(it.get('critique') or '')[:200]!r}")
        parseable = sum(1 for v in it.get("votes", []) if v.get("parsed_ok"))
        print(f"  judges parseable: {parseable}/{len(it.get('votes', []))}")

    out_path = os.path.join(os.path.dirname(__file__), "smoke_trace.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    print(f"\nFull trace saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

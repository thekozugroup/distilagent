"""Summary stats for a distilagent JSONL dataset produced by build_dataset.py."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main(path: Path) -> int:
    if not path.exists():
        print(f"no such file: {path}")
        return 1
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("empty dataset")
        return 0

    n = len(rows)
    cats = Counter(r.get("category", "?") for r in rows)
    srcs = Counter(r.get("source", "?") for r in rows)
    converged = sum(1 for r in rows if r.get("autoreason_converged"))
    iters = [r.get("autoreason_iterations", 0) for r in rows]
    calls = [r.get("total_calls", 0) for r in rows]
    elapsed = [r.get("elapsed_seconds", 0) for r in rows]
    gen_len = [len((r.get("generation") or "").split()) for r in rows]

    def stats(xs):
        if not xs:
            return "n/a"
        xs = sorted(xs)
        med = xs[len(xs) // 2]
        return f"min={xs[0]} med={med} max={xs[-1]} avg={sum(xs) / len(xs):.1f}"

    print(f"samples:            {n}")
    print(f"converged:          {converged}/{n} ({100 * converged / n:.0f}%)")
    print(f"iterations:         {stats(iters)}")
    print(f"total_calls:        {stats(calls)}")
    print(f"elapsed_seconds:    {stats(elapsed)}")
    print(f"generation_words:   {stats(gen_len)}")
    print()
    print("by category:")
    for c, k in cats.most_common():
        print(f"  {c:24s} {k}")
    print()
    print("by source:")
    for s, k in srcs.most_common():
        print(f"  {s:48s} {k}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, nargs="?", default=Path("datasets/distilagent_pilot.jsonl"))
    args = ap.parse_args()
    raise SystemExit(main(args.path))

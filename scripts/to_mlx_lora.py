"""Convert a distilagent JSONL dataset into MLX-LoRA training format.

Emits two variants side-by-side so you can A/B the distillation recipe:

  gen-only   {user: instruction, assistant: generation}
             Pure SFT on the tournament-refined answer.

  cot        {user: instruction,
              assistant: "<thinking>" + reasoning + "</thinking>\n\n" + generation}
             Process-supervision: student learns the reasoning too.

Splits each variant into 80/10/10 train/valid/test, writing:
  <out_dir>/<variant>/train.jsonl
  <out_dir>/<variant>/valid.jsonl
  <out_dir>/<variant>/test.jsonl

Format is mlx-lm --data compatible:
  {"messages": [{"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}]}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def load_rows(path: Path) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def teacher_reasoning(row: Dict) -> str:
    """Return the reasoning text from the teacher-role call, if any.

    We include only the *teacher-seed* reasoning (the first call). The
    critic/author/synthesizer/judge reasoning is tournament machinery
    and should not leak into the student target.
    """
    for entry in row.get("reasoning_trace", []):
        if entry.get("role_hint") == "teacher":
            return entry.get("reasoning", "") or ""
    return ""


def row_to_gen_only(row: Dict) -> Dict:
    return {
        "messages": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["generation"] or ""},
        ]
    }


def row_to_cot(row: Dict) -> Dict:
    reasoning = teacher_reasoning(row)
    if reasoning:
        assistant = f"<thinking>\n{reasoning.strip()}\n</thinking>\n\n{row['generation'] or ''}"
    else:
        assistant = row["generation"] or ""
    return {
        "messages": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": assistant},
        ]
    }


def split(rows: List[Dict], seed: int) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_valid = max(1, n // 10)
    n_test = max(1, n // 10)
    n_train = n - n_valid - n_test
    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train : n_train + n_valid],
        "test": shuffled[n_train + n_valid :],
    }


def write_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("datasets/distilagent_pilot.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("datasets/mlx_lora"))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    src = load_rows(args.inp)
    print(f"loaded {len(src)} source rows from {args.inp}")

    variants = {
        "gen_only": [row_to_gen_only(r) for r in src],
        "cot": [row_to_cot(r) for r in src],
    }

    for name, rows in variants.items():
        split_rows = split(rows, args.seed)
        for part, part_rows in split_rows.items():
            path = args.out / name / f"{part}.jsonl"
            write_jsonl(part_rows, path)
            total_chars = sum(
                sum(len(m["content"]) for m in r["messages"]) for r in part_rows
            )
            print(f"  {name}/{part}.jsonl  n={len(part_rows):>3}  chars={total_chars:>9,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

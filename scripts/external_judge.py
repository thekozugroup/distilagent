"""Independent blind judge: compare baseline (teacher seed) vs AutoReason output.

Uses a DIFFERENT model as judge (not the one that produced either answer).
Labels are randomly permuted. Runs 5 judge seeds to reduce noise.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
from collections import Counter

from openai import AsyncOpenAI

API_KEY = os.environ["OPENROUTER_API_KEY"]
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "google/gemini-2.0-flash-exp:free")

INSTRUCTION = (
    "Explain quantum entanglement in exactly 2 sentences using a compelling "
    "metaphor that a 12-year-old would find both accurate and memorable."
)

with open(os.path.join(os.path.dirname(__file__), "smoke_trace.json")) as f:
    trace = json.load(f)

BASELINE = trace["iterations"][0]["A"]      # teacher seed (single-shot)
AUTOREASON = trace["final_answer"]          # tournament winner

JUDGE_SYSTEM = (
    "You are an impartial judge. You will be given an instruction and two "
    "anonymized candidate responses labeled X1 and X2. Evaluate each for: "
    "(1) factual accuracy of the physics, (2) faithfulness to the instruction, "
    "(3) clarity for a 12-year-old. Do not produce chain of thought. Reply "
    "with a single line in exactly the format: WINNER: X1 or WINNER: X2 or "
    "WINNER: TIE."
)


def user_msg(instruction: str, t1: str, t2: str) -> str:
    return (
        f"INSTRUCTION:\n{instruction}\n\n"
        f"--- X1 ---\n{t1}\n\n"
        f"--- X2 ---\n{t2}\n\n"
        "Which is better? Reply only with 'WINNER: X1' or 'WINNER: X2' or 'WINNER: TIE'."
    )


async def judge_once(client: AsyncOpenAI, seed: int) -> tuple[str, str, dict]:
    rng = random.Random(seed)
    labels = ["baseline", "autoreason"]
    rng.shuffle(labels)
    mapping = {"X1": labels[0], "X2": labels[1]}
    texts = {"baseline": BASELINE, "autoreason": AUTOREASON}

    delay = 5.0
    last_err = None
    for attempt in range(6):
        try:
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg(INSTRUCTION, texts[labels[0]], texts[labels[1]])},
                ],
                temperature=0.3,
            )
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            if "429" not in msg and "rate" not in msg.lower():
                raise
            print(f"  429 backoff {delay:.0f}s (attempt {attempt + 1})")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 90.0)
    else:
        raise last_err  # type: ignore[misc]
    raw = resp.choices[0].message.content or ""
    m = re.search(r"WINNER:\s*(X1|X2|TIE)", raw, re.IGNORECASE)
    verdict = m.group(1).upper() if m else "UNPARSED"
    if verdict in ("X1", "X2"):
        real_winner = mapping[verdict]
    elif verdict == "TIE":
        real_winner = "TIE"
    else:
        real_winner = "UNPARSED"
    return real_winner, raw, mapping


async def main():
    client = AsyncOpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")
    n = int(os.environ.get("JUDGE_RUNS", "5"))
    tally: Counter = Counter()
    details = []
    for i in range(n):
        try:
            winner, raw, mapping = await judge_once(client, seed=1000 + i)
        except Exception as exc:  # noqa: BLE001
            print(f"run {i}: FAILED {type(exc).__name__}: {exc}")
            continue
        tally[winner] += 1
        details.append({"seed": 1000 + i, "mapping": mapping, "winner": winner, "raw": raw.strip()})
        print(f"run {i}: {winner}  (mapping={mapping})")
        await asyncio.sleep(float(os.environ.get("JUDGE_SLEEP", "20")))  # gentle on free tier

    print("\n=== TALLY ===")
    for k, v in tally.most_common():
        print(f"  {k}: {v}")
    print(f"\njudge model: {JUDGE_MODEL}")
    with open(os.path.join(os.path.dirname(__file__), "judge_results.json"), "w") as f:
        json.dump({"tally": dict(tally), "details": details, "judge_model": JUDGE_MODEL}, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())

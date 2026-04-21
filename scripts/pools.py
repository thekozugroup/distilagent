"""Free-model teacher / judge pool config + domain-aware teacher router.

Catalog reflects OpenRouter's :free tier as of 2026-04-21. Tiers based on
capability class, not raw parameter count — small MoE models with large
active-parameter budgets sit with the large dense models.

Rules of thumb this encodes:
  * Teacher pool = top-tier generation models. Tournament ceiling = max of
    the pool, so don't add weak models here.
  * Judge pool = mid-tier models from DIFFERENT families. Panel diversity
    outperforms panel size of identical judges.
  * Specialist routing beats random rotation: code prompts go to
    qwen3-coder, math/reasoning to minimax-m2.5, everything else to the
    strongest generalist in rotation.
"""
from __future__ import annotations

import random
import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Pool definitions
# ---------------------------------------------------------------------------

# Strong generalists. Tier-equivalent so rotation doesn't dilute the ceiling.
GENERALIST_POOL: List[str] = [
    "google/gemma-4-26b-a4b-it:free",        # 26B MoE-ish, 262k ctx
    "google/gemma-4-31b-it:free",            # 31B dense, 262k ctx
    "qwen/qwen3-next-80b-a3b-instruct:free", # 80B MoE, 262k ctx
    "meta-llama/llama-3.3-70b-instruct:free",# 70B dense
    "nvidia/nemotron-3-super-120b-a12b:free",# 120B MoE
    "openai/gpt-oss-120b:free",              # 120B
    "z-ai/glm-4.5-air:free",                 # GLM family
]

# Deep-reasoning specialist (handles math, multi-step proofs, hard logic).
REASONING_SPECIALIST: str = "minimax/minimax-m2.5:free"

# Code specialist (trained on code corpora).
CODE_SPECIALIST: str = "qwen/qwen3-coder:free"

# Judge pool — diverse mid-tier families. Panel diversity > panel size.
JUDGE_POOL: List[str] = [
    "openai/gpt-oss-20b:free",               # OpenAI family
    "google/gemma-3-27b-it:free",            # Google Gemma
    "google/gemma-3-12b-it:free",            # Google Gemma (smaller)
    "nvidia/nemotron-nano-9b-v2:free",       # NVIDIA Nemotron
    "nvidia/nemotron-3-nano-30b-a3b:free",   # NVIDIA Nemotron (larger)
    "meta-llama/llama-3.3-70b-instruct:free",# Meta (also in generalist — fine for judging)
    "z-ai/glm-4.5-air:free",                 # Zhipu GLM
]


# ---------------------------------------------------------------------------
# Domain routing
# ---------------------------------------------------------------------------

_CODE_HINTS = re.compile(
    r"\b("
    r"python|javascript|typescript|golang|go |rust|swift|c\+\+|c#|ruby|php|kotlin|"
    r"function|class\s+\w+|def\s+\w+|import\s+|async\s+def|struct|interface|"
    r"api|rest|graphql|sql|query|database|schema|migration|docker|kubernetes|k8s|"
    r"algorithm|recursion|optimization|complexity|regex|unit test|pytest|"
    r"fastapi|django|flask|react|vue|angular|next\.js|litestar|fiber|quarkus|"
    r"leetcode|codeforces|implement|refactor|debug|stacktrace|segfault|exception"
    r")\b",
    re.IGNORECASE,
)

_MATH_HINTS = re.compile(
    r"\b("
    r"theorem|lemma|proof|integral|derivative|matrix|eigenvalue|probability|"
    r"statistic|bayesian|optimization problem|linear algebra|calculus|"
    r"differential equation|topology|manifold|vector space|equation solve|"
    r"\\boxed|\\frac|\\sum|\\int"
    r")\b",
    re.IGNORECASE,
)


def route_teacher(
    instruction: str,
    generalist_pool: List[str],
    seed: int,
) -> str:
    """Pick a teacher model for this prompt.

    Routing:
      - Code-looking prompt → CODE_SPECIALIST (deterministic).
      - Math/reasoning-looking prompt → REASONING_SPECIALIST (deterministic).
      - Otherwise round-robin over ``generalist_pool`` seeded by hash(instruction).
    """
    if _CODE_HINTS.search(instruction):
        return CODE_SPECIALIST
    if _MATH_HINTS.search(instruction):
        return REASONING_SPECIALIST
    if not generalist_pool:
        raise ValueError("empty generalist pool")
    rng = random.Random(seed ^ (hash(instruction) & 0x7FFFFFFF))
    return rng.choice(generalist_pool)


def classify_route(instruction: str) -> str:
    """Returns 'code' | 'reasoning' | 'general' — for logging/telemetry."""
    if _CODE_HINTS.search(instruction):
        return "code"
    if _MATH_HINTS.search(instruction):
        return "reasoning"
    return "general"

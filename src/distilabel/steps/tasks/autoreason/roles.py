# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prompt templates for the AutoReason tournament roles.

Each role is rendered as a chat-style list of ``{"role": ..., "content": ...}``
messages, matching the format consumed by ``distilabel``'s ``TextGeneration``
task. Templates are compiled once at import time as :class:`jinja2.Template`
objects.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Tuple

from jinja2 import Template

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# Caveman-style directive — applies ONLY to the model's internal reasoning
# stream (<think> blocks / reasoning_content / chain of thought), NOT to
# the final output. Source: github.com/JuliusBrussee/caveman. Rationale:
# reasoning-capable models otherwise ramble ("wait, let me reconsider…
# actually…") inside the thinking stream, burning tokens without gaining
# correctness. Caveman style compresses internal deliberation while the
# visible output stays natural prose for downstream student training.
_CAVEMAN_REASONING = (
    " INTERNAL REASONING (thinking/<think>/reasoning trace): terse like "
    "caveman. Drop articles, filler (just/really/basically/perhaps/maybe), "
    "pleasantries, hedging. Fragments OK. Short synonyms. Pattern: [thing] "
    "[action] [reason]. No 'wait/let me/actually/hmm/on second thought'. "
    "First-pass commit. No re-deliberation. "
    "FINAL OUTPUT (the response the user sees): unchanged — natural, "
    "complete, technically precise prose. Code unchanged in both."
)

TEACHER_SEED_SYSTEM = (
    "Expert assistant. Answer direct, concrete, technically correct. No "
    "preamble, no meta, no restating the request. Code: minimal, runnable, "
    "no placeholder comments. Uncertain → state briefly."
    + _CAVEMAN_REASONING
)

CRITIC_SYSTEM = (
    "Rigorous critic. Find concrete, specific flaws in draft. Quote the "
    "exact phrase or line and explain briefly why it is wrong, misleading, "
    "or insufficient. If the draft is already strong and no real flaw "
    "exists, reply exactly: NO FLAWS. Do not invent weaknesses to seem "
    "thorough."
    + _CAVEMAN_REASONING
)

AUTHOR_B_SYSTEM = (
    "Adversarial reviser. Rewrite the draft so it directly addresses the "
    "critique's specific flaws. Do not expand scope. Do not pad length. "
    "Keep the same overall length, structure, and scope as the original "
    "draft. Output the revised response only."
    + _CAVEMAN_REASONING
)

SYNTHESIZER_SYSTEM = (
    "Conservative synthesizer. Produce a minimal repair: keep what the "
    "draft does well, change only what the critique specifically identifies "
    "as wrong. Smallest possible change. If the critique is weak, the "
    "synthesis is nearly identical to the draft. Output the synthesis only."
    + _CAVEMAN_REASONING
)

JUDGE_SYSTEM = (
    "Impartial judge. Rank three anonymized candidates (X1, X2, X3) against "
    "the instruction. No position bias. No chain of thought. No "
    "deliberation. No 'wait/let me/actually'. Reply with the ranking line "
    "only, exact format: RANKING: X? > X? > X?"
)


# ---------------------------------------------------------------------------
# Jinja2 templates (user messages)
# ---------------------------------------------------------------------------

_TEACHER_SEED_USER_TMPL = Template(
    "{{ instruction }}",
    keep_trailing_newline=False,
)

_CRITIC_USER_TMPL = Template(
    "INSTRUCTION:\n{{ instruction }}\n\n"
    "DRAFT RESPONSE:\n{{ draft }}\n\n"
    "Identify concrete, specific flaws in the DRAFT RESPONSE above. "
    "For each flaw, quote the exact phrase or line that is wrong and briefly "
    "explain why. Be specific: do not generalize, do not hedge, do not invent "
    "issues that are not actually present.\n\n"
    "If the draft is already strong and you cannot point to a real flaw, reply "
    "with exactly: NO FLAWS",
    keep_trailing_newline=False,
)

_AUTHOR_B_USER_TMPL = Template(
    "INSTRUCTION:\n{{ instruction }}\n\n"
    "ORIGINAL DRAFT:\n{{ draft }}\n\n"
    "CRITIQUE:\n{{ critique }}\n\n"
    "Rewrite the entire response to address the critique, but keep the same "
    "overall length, structure, and scope as the original draft. Do not expand "
    "scope. Do not pad length. Address only the specific flaws named in the "
    "critique. Output only the revised response.",
    keep_trailing_newline=False,
)

_SYNTHESIZER_USER_TMPL = Template(
    "INSTRUCTION:\n{{ instruction }}\n\n"
    "ORIGINAL DRAFT:\n{{ draft }}\n\n"
    "CRITIQUE:\n{{ critique }}\n\n"
    "Produce a minimal synthesis: keep the draft's strengths and repair only "
    "the issues the critique identifies. Make the smallest possible change "
    "that addresses the critique. Preserve the draft's wording wherever the "
    "critique does not object. If the critique is weak, the synthesis should "
    "be nearly identical to the draft. Output only the synthesized response.",
    keep_trailing_newline=False,
)

_JUDGE_USER_TMPL = Template(
    "INSTRUCTION:\n{{ instruction }}\n\n"
    "Three anonymized candidate responses follow. Rank them from best to "
    "worst. Do not reveal reasoning. Reply with exactly one line in the "
    "format: RANKING: X? > X? > X?\n\n"
    "=== {{ label1 }} ===\n{{ text1 }}\n\n"
    "=== {{ label2 }} ===\n{{ text2 }}\n\n"
    "=== {{ label3 }} ===\n{{ text3 }}\n\n"
    "RANKING:",
    keep_trailing_newline=False,
)


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------

ChatMessage = Dict[str, str]
ChatType = List[ChatMessage]


def render_teacher_seed(instruction: str) -> ChatType:
    """Render the plain teacher/seed prompt."""
    return [
        {"role": "system", "content": TEACHER_SEED_SYSTEM},
        {"role": "user", "content": _TEACHER_SEED_USER_TMPL.render(instruction=instruction)},
    ]


def render_critic(instruction: str, draft: str) -> ChatType:
    """Render the critic prompt."""
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {
            "role": "user",
            "content": _CRITIC_USER_TMPL.render(instruction=instruction, draft=draft),
        },
    ]


def render_author_b(instruction: str, draft: str, critique: str) -> ChatType:
    """Render the adversarial author B prompt."""
    return [
        {"role": "system", "content": AUTHOR_B_SYSTEM},
        {
            "role": "user",
            "content": _AUTHOR_B_USER_TMPL.render(
                instruction=instruction, draft=draft, critique=critique
            ),
        },
    ]


def render_synthesizer(instruction: str, draft: str, critique: str) -> ChatType:
    """Render the conservative synthesizer prompt."""
    return [
        {"role": "system", "content": SYNTHESIZER_SYSTEM},
        {
            "role": "user",
            "content": _SYNTHESIZER_USER_TMPL.render(
                instruction=instruction, draft=draft, critique=critique
            ),
        },
    ]


def render_judge(
    instruction: str,
    candidate_a: str,
    candidate_b: str,
    candidate_ab: str,
    rng_seed: int,
) -> Tuple[ChatType, Dict[str, str]]:
    """Render the judge prompt with a deterministic random label permutation.

    Returns
    -------
    (messages, label_permutation)
        ``label_permutation`` maps displayed labels ``"X1"``, ``"X2"``, ``"X3"``
        to the real candidate labels ``"A"``, ``"B"``, ``"AB"``. The three
        candidate texts appear in the user message under their permuted
        display labels.
    """
    rng = random.Random(rng_seed)
    real_labels = ["A", "B", "AB"]
    rng.shuffle(real_labels)

    display_labels = ["X1", "X2", "X3"]
    label_permutation: Dict[str, str] = {
        display: real for display, real in zip(display_labels, real_labels)
    }

    text_by_real = {"A": candidate_a, "B": candidate_b, "AB": candidate_ab}
    text1 = text_by_real[label_permutation["X1"]]
    text2 = text_by_real[label_permutation["X2"]]
    text3 = text_by_real[label_permutation["X3"]]

    user_content = _JUDGE_USER_TMPL.render(
        instruction=instruction,
        label1="X1",
        label2="X2",
        label3="X3",
        text1=text1,
        text2=text2,
        text3=text3,
    )
    messages: ChatType = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    return messages, label_permutation


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_critique(raw: str) -> Tuple[str, bool]:
    """Parse a critic response.

    Returns ``(critique_text, no_flaws_flag)``. ``no_flaws_flag`` is ``True``
    when the substring ``"NO FLAWS"`` (case-insensitive) appears anywhere in
    ``raw``.
    """
    text = (raw or "").strip()
    no_flaws = "no flaws" in text.lower()
    return text, no_flaws


_VALID_DISPLAY_LABELS = {"X1", "X2", "X3"}


def parse_judge_ranking(
    raw: str,
    label_permutation: Dict[str, str],
) -> Tuple[List[str], bool]:
    """Parse a judge ranking response.

    Accepts formats like ``"RANKING: X1 > X3 > X2"``, tolerating case,
    missing ``RANKING:`` prefix, and ``>`` / ``,`` / ``and`` / newline
    separators. Returns ``(ranking_in_real_labels, parsed_ok)``. On any
    structural problem (wrong count, duplicate label, unknown label) returns
    ``([], False)``.
    """
    if not raw:
        return [], False

    text = raw.strip()
    # Drop a leading RANKING: prefix if present (case-insensitive).
    text = re.sub(r"(?i)^\s*ranking\s*:\s*", "", text, count=1)
    # If there are multiple lines, take the first non-empty line that mentions
    # an X-label; otherwise fall back to the first non-empty line.
    candidate_line = None
    for line in text.splitlines():
        if re.search(r"(?i)\bx[123]\b", line):
            candidate_line = line
            break
    if candidate_line is None:
        for line in text.splitlines():
            if line.strip():
                candidate_line = line
                break
    if candidate_line is None:
        return [], False

    # Extract X1/X2/X3 tokens in order.
    tokens = re.findall(r"(?i)\bx\s*([123])\b", candidate_line)
    if len(tokens) != 3:
        return [], False

    display_seq = [f"X{t}" for t in tokens]

    # Duplicate detection.
    if len(set(display_seq)) != 3:
        return [], False

    # Unknown label detection (shouldn't happen because regex restricts to 1-3).
    if any(lbl not in _VALID_DISPLAY_LABELS for lbl in display_seq):
        return [], False

    # Permutation must cover exactly X1/X2/X3 with known real labels.
    try:
        real_seq = [label_permutation[lbl] for lbl in display_seq]
    except KeyError:
        return [], False

    if any(r not in ("A", "B", "AB") for r in real_seq):
        return [], False

    if len(set(real_seq)) != 3:
        return [], False

    return real_seq, True

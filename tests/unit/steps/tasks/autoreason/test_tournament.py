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
"""Unit tests for the AutoReason TournamentRunner loop."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Union

import pytest

from distilabel.steps.tasks.autoreason.rate_limit import AsyncTokenBucket
from distilabel.steps.tasks.autoreason.roles import (
    AUTHOR_B_SYSTEM,
    CRITIC_SYSTEM,
    JUDGE_SYSTEM,
    SYNTHESIZER_SYSTEM,
    TEACHER_SEED_SYSTEM,
)
from distilabel.steps.tasks.autoreason.tournament import TournamentRunner

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# ScriptedLLM test double
# ---------------------------------------------------------------------------


ScriptEntry = Union[List[Any], Callable[[int, list], str]]


class ScriptedLLM:
    """Lightweight async LLM stub for tournament tests.

    The `script` maps role names ("teacher", "critic", "author_b",
    "synthesizer", "judge") to either a list of canned responses
    (consumed pop-from-front) or a callable receiving
    ``(call_idx, input)`` and returning a string.

    Routing is by the *system prompt* of ``input[0]``, matched against the
    role system constants from :mod:`distilabel.steps.tasks.autoreason.roles`.
    """

    _SYSTEM_TO_ROLE = {
        TEACHER_SEED_SYSTEM: "teacher",
        CRITIC_SYSTEM: "critic",
        AUTHOR_B_SYSTEM: "author_b",
        SYNTHESIZER_SYSTEM: "synthesizer",
        JUDGE_SYSTEM: "judge",
    }

    def __init__(self, script: Dict[str, ScriptEntry]) -> None:
        self.script = script
        self.counters: Dict[str, int] = {
            "teacher": 0,
            "critic": 0,
            "author_b": 0,
            "synthesizer": 0,
            "judge": 0,
        }

    def _resolve_role(self, input: list) -> str:
        if not input or input[0].get("role") != "system":
            raise AssertionError(
                f"ScriptedLLM: expected first message to be system, got {input!r}"
            )
        sys_content = input[0]["content"]
        role = self._SYSTEM_TO_ROLE.get(sys_content)
        if role is None:
            raise AssertionError(
                f"ScriptedLLM: unrecognized system prompt: {sys_content!r}"
            )
        return role

    def _next_response(self, role: str, input: list) -> str:
        entry = self.script.get(role)
        if entry is None:
            raise AssertionError(
                f"ScriptedLLM: no script entry for role {role!r}"
            )
        idx = self.counters[role]
        self.counters[role] = idx + 1
        if callable(entry):
            return entry(idx, input)
        # List: pop-from-front semantics.
        if not isinstance(entry, list):
            raise AssertionError(
                f"ScriptedLLM: script entry for {role!r} must be list or callable; got {type(entry).__name__}"
            )
        if idx >= len(entry):
            raise AssertionError(
                f"ScriptedLLM: script for role {role!r} exhausted at call {idx}"
            )
        return entry[idx]

    async def agenerate(
        self, input: list, num_generations: int = 1, **kwargs: Any
    ) -> dict:
        role = self._resolve_role(input)
        text = self._next_response(role, input)
        return {
            "generations": [text],
            "statistics": {"input_tokens": [0], "output_tokens": [0]},
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def judge_ranking_for(target_text: str, user_content: str) -> str:
    """Return ``"RANKING: <target> > <other1> > <other2>"`` where ``<target>``
    is the display label (X1/X2/X3) whose section wraps ``target_text``.

    Parses blocks of the form::

        === X1 ===
        <text1>

        === X2 ===
        <text2>

        === X3 ===
        <text3>

    and finds which display label contains ``target_text``. Remaining
    labels are appended in natural order to form a valid 3-way ranking.
    """
    # Extract (label, body) pairs. Body runs until the next "=== Xn ===" or
    # the final "RANKING:" marker.
    pattern = re.compile(
        r"=== (X[123]) ===\n(.*?)(?=\n=== X[123] ===|\nRANKING:|\Z)",
        re.DOTALL,
    )
    sections = {m.group(1): m.group(2) for m in pattern.finditer(user_content)}
    if len(sections) != 3:
        raise AssertionError(
            f"judge_ranking_for: expected 3 sections, got {list(sections)!r}"
        )

    winner_label = None
    for label, body in sections.items():
        if target_text in body:
            winner_label = label
            break
    if winner_label is None:
        raise AssertionError(
            f"judge_ranking_for: target {target_text!r} not found in any section"
        )

    others = [lbl for lbl in ("X1", "X2", "X3") if lbl != winner_label]
    return f"RANKING: {winner_label} > {others[0]} > {others[1]}"


def _user_content(input: list) -> str:
    # Second message is the user message.
    return input[1]["content"]


# ---------------------------------------------------------------------------
# Constants (distinctive texts so judges can locate candidates)
# ---------------------------------------------------------------------------

SEED_TEXT = "<<<TEACHER_SEED>>>"
B_TEXT = "<<<BEE>>>"
AB_TEXT = "<<<SYNTH>>>"
CRITIQUE_TEXT = "There is a specific flaw in paragraph 2."


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_converges_when_A_defends_twice() -> None:
    """Critic always returns a critique; judges rank A first in both rounds."""
    script: Dict[str, ScriptEntry] = {
        "teacher": [SEED_TEXT],
        "critic": [CRITIQUE_TEXT, CRITIQUE_TEXT],
        "author_b": [B_TEXT, B_TEXT + "2"],
        "synthesizer": [AB_TEXT, AB_TEXT + "2"],
        "judge": lambda idx, inp: judge_ranking_for(SEED_TEXT, _user_content(inp)),
    }
    llm = ScriptedLLM(script)
    runner = TournamentRunner(
        llm=llm,
        num_judges=3,
        max_iterations=5,
        convergence_k=2,
    )
    trace = await runner.run("What is 2+2?")

    assert trace.converged is True
    assert len(trace.iterations) == 2
    assert trace.final_answer == SEED_TEXT
    assert trace.iterations[0].winner == "A"
    assert trace.iterations[1].winner == "A"


async def test_no_flaws_early_exit() -> None:
    """Critic returns 'NO FLAWS' twice with convergence_k=2: stop at 2 iters."""
    script: Dict[str, ScriptEntry] = {
        "teacher": [SEED_TEXT],
        "critic": ["NO FLAWS", "NO FLAWS"],
        # B/AB/judge must not be called under no-flaws branch.
        "author_b": lambda idx, inp: (_ for _ in ()).throw(
            AssertionError("author_b should not be called on NO FLAWS")
        ),
        "synthesizer": lambda idx, inp: (_ for _ in ()).throw(
            AssertionError("synthesizer should not be called on NO FLAWS")
        ),
        "judge": lambda idx, inp: (_ for _ in ()).throw(
            AssertionError("judge should not be called on NO FLAWS")
        ),
    }
    llm = ScriptedLLM(script)
    runner = TournamentRunner(
        llm=llm, num_judges=3, max_iterations=5, convergence_k=2
    )
    trace = await runner.run("instr")

    assert trace.converged is True
    assert len(trace.iterations) == 2
    for it in trace.iterations:
        assert it.no_flaws is True
        assert it.B == ""
        assert it.AB == ""
        assert it.winner == "A"
    assert trace.final_answer == SEED_TEXT


async def test_b_wins_promotes_new_incumbent() -> None:
    """Iter 0: B wins. Iter 1 & 2: the promoted B (now A) wins twice to converge."""
    # Round 0 produces b_text = B_TEXT. After promotion, the new "A" is B_TEXT.
    # In rounds 1 and 2, we want judges to rank A (B_TEXT) first.
    def judge_fn(idx: int, inp: list) -> str:
        uc = _user_content(inp)
        # In iter 0, SEED_TEXT is A. We want B to win -> target B_TEXT.
        # In iter >=1, A is B_TEXT; we want A to win -> target B_TEXT.
        # So target is always B_TEXT.
        return judge_ranking_for(B_TEXT, uc)

    script: Dict[str, ScriptEntry] = {
        "teacher": [SEED_TEXT],
        "critic": [CRITIQUE_TEXT, CRITIQUE_TEXT, CRITIQUE_TEXT],
        "author_b": [B_TEXT, "B_round2", "B_round3"],
        "synthesizer": [AB_TEXT, "AB_round2", "AB_round3"],
        "judge": judge_fn,
    }
    llm = ScriptedLLM(script)
    runner = TournamentRunner(
        llm=llm, num_judges=3, max_iterations=10, convergence_k=2
    )
    trace = await runner.run("instr")

    assert trace.converged is True
    assert trace.iterations[0].winner == "B"
    assert trace.iterations[1].winner == "A"
    assert trace.iterations[2].winner == "A"
    assert trace.final_answer == B_TEXT


async def test_ab_wins_promotes_synthesis() -> None:
    """Iter 0: AB wins. Iter 1 & 2: promoted AB defends twice."""
    def judge_fn(idx: int, inp: list) -> str:
        uc = _user_content(inp)
        # Always rank AB_TEXT first: it's AB in iter 0, then A in iter 1+.
        return judge_ranking_for(AB_TEXT, uc)

    script: Dict[str, ScriptEntry] = {
        "teacher": [SEED_TEXT],
        "critic": [CRITIQUE_TEXT, CRITIQUE_TEXT, CRITIQUE_TEXT],
        "author_b": [B_TEXT, "B_round2", "B_round3"],
        "synthesizer": [AB_TEXT, "AB_round2", "AB_round3"],
        "judge": judge_fn,
    }
    llm = ScriptedLLM(script)
    runner = TournamentRunner(
        llm=llm, num_judges=3, max_iterations=10, convergence_k=2
    )
    trace = await runner.run("instr")

    assert trace.converged is True
    assert trace.iterations[0].winner == "AB"
    assert trace.iterations[1].winner == "A"
    assert trace.iterations[2].winner == "A"
    assert trace.final_answer == AB_TEXT


async def test_max_iterations_hit_without_convergence() -> None:
    """Judges alternate winners so A never defends twice in a row."""
    # Iter 0: B wins -> A becomes B_texts[0]
    # Iter 1: B wins -> A becomes B_texts[1]
    # Iter 2: B wins -> A becomes B_texts[2]
    # max_iterations=3, convergence_k=2 but no two consecutive A-wins.
    b_texts = ["B_iter0", "B_iter1", "B_iter2"]

    def judge_fn(idx: int, inp: list) -> str:
        # Each iteration has num_judges judge calls, so iteration = idx // num_judges.
        # We want B to win every iteration. The current-iteration B text is the
        # one currently in this judge prompt that is NOT the incumbent A and NOT
        # AB. Easiest: our B text pool is distinct — locate whichever of b_texts
        # appears as a fresh B (it's always the last b_texts up through this
        # iteration... actually B is generated fresh each round). Simplest
        # trick: the B text for iter i is b_texts[i]. Compute iteration from idx.
        iteration = idx // 3  # num_judges=3
        target = b_texts[iteration]
        return judge_ranking_for(target, _user_content(inp))

    script: Dict[str, ScriptEntry] = {
        "teacher": [SEED_TEXT],
        "critic": [CRITIQUE_TEXT] * 5,
        "author_b": list(b_texts),
        "synthesizer": ["AB_iter0", "AB_iter1", "AB_iter2"],
        "judge": judge_fn,
    }
    llm = ScriptedLLM(script)
    runner = TournamentRunner(
        llm=llm, num_judges=3, max_iterations=3, convergence_k=2
    )
    trace = await runner.run("instr")

    assert trace.converged is False
    assert len(trace.iterations) == 3
    # B wins all three; final incumbent is the last promoted B.
    assert trace.final_answer == b_texts[-1]


async def test_majority_malformed_judges_keeps_A() -> None:
    """With >=4/7 judges unparseable, fail-safe keeps A; run 2 iters to converge."""
    def judge_fn(idx: int, inp: list) -> str:
        # Make judges 0..3 (4 of 7) unparseable each iteration; rest give
        # a ranking. judge index within iteration = idx % 7.
        j_in_iter = idx % 7
        if j_in_iter < 4:
            return "I cannot rank"
        # Remaining judges rank A (SEED_TEXT) first — but fail-safe kicks in
        # regardless because only 3/7 parse. Still fine.
        return judge_ranking_for(SEED_TEXT, _user_content(inp))

    script: Dict[str, ScriptEntry] = {
        "teacher": [SEED_TEXT],
        "critic": [CRITIQUE_TEXT, CRITIQUE_TEXT],
        "author_b": [B_TEXT, B_TEXT + "2"],
        "synthesizer": [AB_TEXT, AB_TEXT + "2"],
        "judge": judge_fn,
    }
    llm = ScriptedLLM(script)
    runner = TournamentRunner(
        llm=llm, num_judges=7, max_iterations=5, convergence_k=2
    )
    trace = await runner.run("instr")

    assert trace.converged is True
    assert len(trace.iterations) == 2
    for it in trace.iterations:
        assert it.winner == "A"
        # Borda stays all-zero under fail-safe.
        assert it.borda == {"A": 0, "B": 0, "AB": 0}
    assert trace.final_answer == SEED_TEXT


async def test_rate_limiter_is_invoked() -> None:
    """Confirm AsyncTokenBucket.acquire runs at least once."""
    bucket = AsyncTokenBucket(rpm=10000, name="test")
    start_remaining = bucket.state["rpm_remaining"]

    script: Dict[str, ScriptEntry] = {
        "teacher": [SEED_TEXT],
        "critic": ["NO FLAWS", "NO FLAWS"],
        "author_b": [],
        "synthesizer": [],
        "judge": [],
    }
    llm = ScriptedLLM(script)
    runner = TournamentRunner(
        llm=llm,
        num_judges=3,
        max_iterations=5,
        convergence_k=2,
        rate_limiter=bucket,
    )
    await runner.run("instr")

    end_remaining = bucket.state["rpm_remaining"]
    assert end_remaining < start_remaining, (
        f"Rate limiter not invoked: remaining {end_remaining} >= start {start_remaining}"
    )


async def test_total_calls_counter() -> None:
    """With num_judges=3, max_iterations=1 and a critique: expect 7 calls."""
    script: Dict[str, ScriptEntry] = {
        "teacher": [SEED_TEXT],
        "critic": [CRITIQUE_TEXT],
        "author_b": [B_TEXT],
        "synthesizer": [AB_TEXT],
        "judge": lambda idx, inp: judge_ranking_for(SEED_TEXT, _user_content(inp)),
    }
    llm = ScriptedLLM(script)
    runner = TournamentRunner(
        llm=llm, num_judges=3, max_iterations=1, convergence_k=2
    )
    trace = await runner.run("instr")

    # teacher(1) + critic(1) + author_b(1) + synthesizer(1) + judges(3) = 7
    assert trace.total_calls == 7

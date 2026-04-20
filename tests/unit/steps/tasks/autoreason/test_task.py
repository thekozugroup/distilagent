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

"""Integration tests for :class:`AutoReasonedGeneration`.

These tests wire up a scripted async LLM, drive the task end-to-end, and
assert on the shape and contents of the emitted rows. The scripted LLM
subclasses :class:`distilabel.models.llms.base.AsyncLLM` so that pydantic
validation on ``Task.llm`` passes without gymnastics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import pytest
from pydantic import PrivateAttr

from distilabel.models.llms.base import AsyncLLM
from distilabel.steps.tasks.autoreason.roles import (
    AUTHOR_B_SYSTEM,
    CRITIC_SYSTEM,
    JUDGE_SYSTEM,
    SYNTHESIZER_SYSTEM,
    TEACHER_SEED_SYSTEM,
)
from distilabel.steps.tasks.autoreason.task import AutoReasonedGeneration

if TYPE_CHECKING:
    from distilabel.typing import FormattedInput, GenerateOutput


ScriptEntry = Union[List[Any], Callable[[int, list], str]]


class ScriptedAsyncLLM(AsyncLLM):
    """Scripted :class:`AsyncLLM` that routes responses by role system prompt."""

    structured_output: Any = None

    _script: Dict[str, ScriptEntry] = PrivateAttr(default_factory=dict)
    _counters: Dict[str, int] = PrivateAttr(default_factory=dict)
    _SYSTEM_TO_ROLE = {
        TEACHER_SEED_SYSTEM: "teacher",
        CRITIC_SYSTEM: "critic",
        AUTHOR_B_SYSTEM: "author_b",
        SYNTHESIZER_SYSTEM: "synthesizer",
        JUDGE_SYSTEM: "judge",
    }

    def set_script(self, script: Dict[str, ScriptEntry]) -> None:
        self._script = script
        self._counters = {k: 0 for k in self._SYSTEM_TO_ROLE.values()}

    def load(self) -> None:
        super().load()
        if not self._counters:
            self._counters = {k: 0 for k in self._SYSTEM_TO_ROLE.values()}

    @property
    def model_name(self) -> str:
        return "scripted"

    def _resolve_role(self, input: list) -> str:
        assert input and input[0].get("role") == "system", (
            f"ScriptedAsyncLLM: expected first msg to be system, got {input!r}"
        )
        sys_content = input[0]["content"]
        role = self._SYSTEM_TO_ROLE.get(sys_content)
        assert role is not None, (
            f"ScriptedAsyncLLM: unrecognized system prompt: {sys_content!r}"
        )
        return role

    def _next_response(self, role: str, input: list) -> str:
        entry = self._script.get(role)
        assert entry is not None, f"no script entry for role {role!r}"
        idx = self._counters.get(role, 0)
        self._counters[role] = idx + 1
        if callable(entry):
            return entry(idx, input)
        assert isinstance(entry, list), f"role {role!r} needs list or callable"
        assert idx < len(entry), f"script exhausted for role {role!r} at {idx}"
        return entry[idx]

    async def agenerate(  # type: ignore[override]
        self,
        input: "FormattedInput",
        num_generations: int = 1,
        **kwargs: Any,
    ) -> "GenerateOutput":
        role = self._resolve_role(input)  # type: ignore[arg-type]
        text = self._next_response(role, input)  # type: ignore[arg-type]
        return {
            "generations": [text],
            "statistics": {"input_tokens": [0], "output_tokens": [0]},
        }


class FailingAsyncLLM(AsyncLLM):
    """AsyncLLM that raises on every call."""

    structured_output: Any = None
    _exc_message: str = PrivateAttr(default="boom")

    def load(self) -> None:
        super().load()

    @property
    def model_name(self) -> str:
        return "failing"

    async def agenerate(  # type: ignore[override]
        self,
        input: "FormattedInput",
        num_generations: int = 1,
        **kwargs: Any,
    ) -> "GenerateOutput":
        raise RuntimeError(self._exc_message)


# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------


def _no_flaws_script(seed_text: str = "Seed answer for X") -> Dict[str, ScriptEntry]:
    """Script where the critic returns NO FLAWS every time (fast convergence)."""
    return {
        "teacher": [seed_text] * 10,
        "critic": ["NO FLAWS"] * 10,
        "author_b": [],
        "synthesizer": [],
        "judge": [],
    }


def _echoing_teacher_no_flaws() -> Dict[str, ScriptEntry]:
    """Teacher echoes the user instruction; critic says NO FLAWS."""

    def teacher_fn(_idx: int, inp: list) -> str:
        # user content is the rendered instruction
        user = next(m for m in inp if m.get("role") == "user")
        return f"ANSWER[{user['content']}]"

    return {
        "teacher": teacher_fn,
        "critic": ["NO FLAWS"] * 10,
        "author_b": [],
        "synthesizer": [],
        "judge": [],
    }


def _one_round_then_converge(seed: str = "S") -> Dict[str, ScriptEntry]:
    """Run one real iteration (critique has flaws, judges vote), then NO FLAWS."""
    return {
        "teacher": [seed],
        "critic": ["This draft lacks detail on edge case Z.", "NO FLAWS", "NO FLAWS"],
        "author_b": ["B variant"],
        "synthesizer": ["AB variant"],
        # Judges all pick A first -> incumbent keeps, then NO FLAWS converges.
        # Ranking must be in display labels X1/X2/X3; we don't know the permutation
        # ahead of time, so return a format that parses: we'll craft via callable.
        "judge": (lambda _idx, inp: _vote_for_A(inp)),  # type: ignore[dict-item]
    }


def _vote_for_A(messages: list) -> str:
    """Produce a judge RANKING line that ranks the real 'A' candidate first.

    The judge user content includes blocks like ``=== X1 ===\\n<text>``; we can
    locate whichever X? corresponds to A by pattern-matching the seed text.
    But simpler: the test seed is unique, so find which X? block contains it.
    """
    user = next(m for m in messages if m.get("role") == "user")["content"]
    # Find X1 / X2 / X3 block mapping to their texts.
    blocks: Dict[str, str] = {}
    for label in ("X1", "X2", "X3"):
        start_marker = f"=== {label} ===\n"
        i = user.find(start_marker)
        if i < 0:
            continue
        i += len(start_marker)
        # Block ends at the next "===" or at "RANKING:" near the end
        j_next = user.find("=== ", i)
        j_rank = user.find("RANKING:", i)
        end = min(j for j in (j_next, j_rank) if j >= 0) if (j_next >= 0 or j_rank >= 0) else len(user)
        blocks[label] = user[i:end].strip()
    # Identify A by the unique seed marker "S" alone as the block content.
    a_label = next((lbl for lbl, t in blocks.items() if t == "S"), "X1")
    others = [lbl for lbl in ("X1", "X2", "X3") if lbl != a_label]
    return f"RANKING: {a_label} > {others[0]} > {others[1]}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_task(
    llm, *, num_judges: int = 7, max_iterations: int = 15, **kwargs: Any
) -> AutoReasonedGeneration:
    task = AutoReasonedGeneration(
        llm=llm,
        num_judges=num_judges,
        max_iterations=max_iterations,
        **kwargs,
    )
    task.load()
    return task


def test_single_row_converges():
    llm = ScriptedAsyncLLM()
    llm.set_script(_no_flaws_script("Seed for X"))
    task = _make_task(llm, num_judges=7, max_iterations=5, convergence_k=2)

    results = next(task.process([{"instruction": "Explain X"}]))

    assert len(results) == 1
    row = results[0]
    assert row["instruction"] == "Explain X"
    assert row["generation"] == "Seed for X"
    assert isinstance(row["autoreason_trace"], dict)
    assert row["autoreason_converged"] is True
    assert row["autoreason_iterations"] >= 1
    assert row["model_name"] == "scripted"


def test_multiple_rows():
    llm = ScriptedAsyncLLM()
    llm.set_script(_echoing_teacher_no_flaws())
    task = _make_task(llm, num_judges=3, max_iterations=3, convergence_k=2)

    results = next(
        task.process(
            [
                {"instruction": "Question one"},
                {"instruction": "Question two"},
            ]
        )
    )

    assert len(results) == 2
    gens = [r["generation"] for r in results]
    assert "Question one" in gens[0]
    assert "Question two" in gens[1]
    assert gens[0] != gens[1]
    assert results[0]["autoreason_trace"] != results[1]["autoreason_trace"]


def test_runtime_params_respected():
    llm = ScriptedAsyncLLM()
    # First critique has flaws so we get a full judged round; then NO FLAWS.
    llm.set_script(_one_round_then_converge("S"))
    task = _make_task(
        llm,
        num_judges=3,
        max_iterations=1,
        convergence_k=2,
        rng_seed_base=0,
    )

    results = next(task.process([{"instruction": "Q"}]))
    row = results[0]
    trace = row["autoreason_trace"]

    # max_iterations=1 bounds total iterations
    assert row["autoreason_iterations"] <= 1
    # With one iteration that had flaws, we must have 3 votes (one per judge).
    iters = trace["iterations"]
    assert len(iters) == 1
    iteration = iters[0]
    if not iteration["no_flaws"]:
        assert len(iteration["votes"]) == 3


def test_template_with_custom_columns():
    llm = ScriptedAsyncLLM()
    llm.set_script(_echoing_teacher_no_flaws())

    task = AutoReasonedGeneration(
        llm=llm,
        template="{{ topic }}: {{ detail }}",
        columns=["topic", "detail"],
        num_judges=3,
        max_iterations=2,
        convergence_k=2,
    )
    task.load()

    results = next(
        task.process([{"topic": "Python", "detail": "async"}])
    )

    row = results[0]
    # The scripted teacher echoes the rendered user prompt back.
    assert "Python: async" in row["generation"]


def test_failure_produces_empty_row_with_error_metadata():
    llm = FailingAsyncLLM()
    task = _make_task(llm, num_judges=3, max_iterations=2, convergence_k=2)

    results = next(task.process([{"instruction": "Will fail"}]))

    assert len(results) == 1
    row = results[0]
    assert row["generation"] is None
    assert row["autoreason_trace"] is None
    assert row["autoreason_iterations"] == 0
    assert row["autoreason_converged"] is False
    # Error is stashed in distilabel_metadata
    from distilabel.constants import DISTILABEL_METADATA_KEY

    meta = row[DISTILABEL_METADATA_KEY]
    assert any("autoreason_error" in k for k in meta.keys())
    # The full batch didn't crash — we got a row back.

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

"""Distilabel Task integration for the AutoReason tournament.

This module exposes :class:`AutoReasonedGeneration`, a
:class:`distilabel.steps.tasks.base.Task` that runs an AutoReason tournament
per input row. Unlike ``TextGeneration``, which calls the LLM once per
input, AutoReason orchestrates many LLM calls per row (seed, critique,
adversarial revision, synthesis, and a judge panel) via
:class:`~distilabel.steps.tasks.autoreason.tournament.TournamentRunner`. To
accommodate that, this task overrides ``process`` directly instead of
relying on the default ``_format_inputs`` / ``llm.generate_outputs`` /
``_format_outputs`` pipeline.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from jinja2 import Template
from pydantic import Field, PrivateAttr

from distilabel.constants import DISTILABEL_METADATA_KEY
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.autoreason.rate_limit import get_limiter
from distilabel.steps.tasks.autoreason.tournament import TournamentRunner
from distilabel.steps.tasks.base import Task
from distilabel.utils.template import check_column_in_template

if TYPE_CHECKING:
    from distilabel.steps.base import StepInput
    from distilabel.typing import ChatType, StepColumns, StepOutput


# Distilabel depends on ``nest_asyncio`` and applies it when running inside a
# notebook. Sub-processes launched by the pipeline orchestrator can also be
# running inside an asyncio loop (e.g. via the LLM client), so we apply
# ``nest_asyncio`` at import time to make ``asyncio.run`` resilient to already-
# running loops. This mirrors ``distilabel.models.llms.base``.
try:  # pragma: no cover - import side effect, best-effort
    import nest_asyncio

    nest_asyncio.apply()
except Exception:  # pragma: no cover - nest_asyncio is optional at runtime
    nest_asyncio = None  # type: ignore[assignment]


class AutoReasonedGeneration(Task):
    """Generate a response through an AutoReason tournament.

    AutoReason refines a teacher's answer by pitting three candidates against
    each other under a panel of blinded judges. Each input row produces:

    - a final (possibly refined) answer,
    - the full tournament trace (critique, revisions, votes),
    - the number of refinement iterations the tournament ran,
    - whether the tournament converged before ``max_iterations``.

    Unlike ``TextGeneration``, this task issues many LLM calls per input.
    ``llm.generate`` / ``llm.agenerate`` is invoked by the underlying
    :class:`~distilabel.steps.tasks.autoreason.tournament.TournamentRunner`
    rather than the task's own ``format_input`` → ``generate`` →
    ``format_output`` pipeline. The ``process`` method is overridden to run
    one tournament per row.

    Attributes:
        num_judges: Size of the judge panel per iteration. Defaults to ``7``.
        max_iterations: Upper bound on refinement passes. Defaults to ``15``.
        convergence_k: Stop after ``k`` consecutive iterations in which the
            incumbent A wins. Defaults to ``2``.
        max_concurrency: Maximum concurrent in-flight role calls per
            tournament. Defaults to ``8``.
        rpm: Optional requests-per-minute cap applied via an async token
            bucket. ``None`` disables rate limiting.
        rpd: Optional requests-per-day cap. Only meaningful alongside ``rpm``.
        rng_seed_base: Base seed for deterministic judge-label permutations.
        system_prompt: Reserved passthrough. Currently unused by the
            tournament (roles carry their own system prompts) but kept for
            symmetry with ``TextGeneration``.
        template: Jinja2 template rendered per input row to produce the
            instruction string.
        columns: Template variables populated from each input row.

    Input columns:
        - dynamic (``columns``): By default ``instruction``. The listed
          columns are fed into ``template`` to render the instruction used
          to seed the tournament.

    Output columns:
        - generation (``str``): The final answer the tournament produced.
        - autoreason_trace (``Dict``): Serialized tournament trace
          (iterations, critiques, votes, Borda totals).
        - autoreason_iterations (``int``): Number of iterations executed.
        - autoreason_converged (``bool``): Whether the tournament converged.
        - model_name (``str``): Name of the underlying LLM.

    Categories:
        - text-generation

    Examples:
        Refine a teacher answer through an AutoReason tournament:

        ```python
        from distilabel.steps.tasks import AutoReasonedGeneration
        from distilabel.models import OpenAILLM

        task = AutoReasonedGeneration(
            llm=OpenAILLM(model="gpt-4o-mini"),
            num_judges=7,
            max_iterations=5,
            convergence_k=2,
        )
        task.load()

        result = next(
            task.process([{"instruction": "Explain quicksort."}])
        )
        # result[0]["generation"]          -> refined answer
        # result[0]["autoreason_trace"]    -> full tournament record
        # result[0]["autoreason_converged"] -> True / False
        ```
    """

    num_judges: RuntimeParameter[int] = Field(
        default=7,
        description="Number of judges in the panel. The paper uses 7.",
    )
    max_iterations: RuntimeParameter[int] = Field(
        default=15,
        description="Upper bound on refinement passes per tournament.",
    )
    convergence_k: RuntimeParameter[int] = Field(
        default=2,
        description=(
            "Stop when incumbent A wins this many consecutive iterations."
        ),
    )
    max_concurrency: RuntimeParameter[int] = Field(
        default=8,
        description="Semaphore bound for concurrent role calls per tournament.",
    )
    rpm: RuntimeParameter[Optional[int]] = Field(
        default=None,
        description="Optional requests-per-minute cap. None disables rate limiting.",
    )
    rpd: RuntimeParameter[Optional[int]] = Field(
        default=None,
        description="Optional requests-per-day cap. Only used when rpm is set.",
    )
    rng_seed_base: RuntimeParameter[int] = Field(
        default=0,
        description="Base seed for deterministic judge-label permutations.",
    )

    system_prompt: Union[str, None] = None
    template: str = Field(
        default="{{ instruction }}",
        description=(
            "Jinja2 template rendered per input row to produce the "
            "instruction fed into the tournament."
        ),
    )
    columns: Union[str, List[str]] = Field(
        default="instruction",
        description=(
            "Custom column or list of columns referenced by the template."
        ),
    )

    _template: Optional["Template"] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self.columns = [self.columns] if isinstance(self.columns, str) else self.columns
        super().model_post_init(__context)

    def load(self) -> None:
        super().load()
        for column in self.columns:
            check_column_in_template(column, self.template)
        self._template = Template(self.template)

    def unload(self) -> None:
        super().unload()
        self._template = None

    @property
    def inputs(self) -> "StepColumns":
        """Input columns are driven by ``columns``. ``system_prompt`` is optional."""
        cols = {column: True for column in self.columns}
        cols["system_prompt"] = False
        return cols

    @property
    def outputs(self) -> List[str]:
        """Output columns emitted per input row."""
        return [
            "generation",
            "autoreason_trace",
            "autoreason_iterations",
            "autoreason_converged",
            "model_name",
        ]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """Return a chat-style rendering of the instruction.

        The AutoReason tournament does not consume this directly — each role
        crafts its own chat messages — but the base class uses this to build
        ``raw_input_*`` metadata and the ``print`` preview.
        """
        fields = {column: input[column] for column in self.columns}
        user_content = (
            self._template.render(**fields)
            if self._template is not None
            else Template(self.template).render(**fields)
        )
        messages: List[Dict[str, str]] = []
        row_system_prompt = input.get("system_prompt")
        if row_system_prompt:
            messages.append({"role": "system", "content": row_system_prompt})
        elif self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages  # type: ignore[return-value]

    def format_output(
        self,
        output: Union[str, None],
        input: Union[Dict[str, Any], None] = None,
    ) -> Dict[str, Any]:
        """Minimal fallback formatter.

        The real output assembly happens in ``process`` since a tournament
        yields much more than a single string. This method is still required
        by the abstract base and is used by the framework's failure paths.
        """
        return {"generation": output}

    # ------------------------------------------------------------------
    # Core: per-row tournament execution
    # ------------------------------------------------------------------
    def process(self, inputs: "StepInput") -> "StepOutput":  # type: ignore[override]
        """Run one AutoReason tournament per input row and yield enriched rows."""
        if self._template is None:
            # ``process`` may be called directly in tests without ``load``.
            self._template = Template(self.template)

        limiter = None
        if self.rpm is not None:
            limiter = get_limiter(
                name=self.llm.model_name,
                rpm=self.rpm,
                rpd=self.rpd,
            )

        runner = TournamentRunner(
            llm=self.llm,
            num_judges=self.num_judges,
            max_iterations=self.max_iterations,
            convergence_k=self.convergence_k,
            max_concurrency=self.max_concurrency,
            rate_limiter=limiter,
            rng_seed_base=self.rng_seed_base,
        )

        task_outputs: List[Dict[str, Any]] = []
        for row in inputs:
            fields = {column: row[column] for column in self.columns if column in row}
            instruction = self._template.render(**fields)
            try:
                trace = self._run_async(runner.run(instruction))
                task_outputs.append(self._row_from_trace(row, trace))
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    f"AutoReason failed for row: {exc}"  # type: ignore[attr-defined]
                )
                task_outputs.append(self._empty_row(row, str(exc)))

        yield task_outputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _run_async(coro):
        """Execute ``coro`` even if we are already inside a running loop.

        ``nest_asyncio.apply()`` at module load lets ``asyncio.run`` nest,
        but some execution contexts still forbid it. Fall back to a
        dedicated thread with a fresh event loop if needed.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    def _row_from_trace(self, row: Dict[str, Any], trace) -> Dict[str, Any]:
        """Assemble a success row from a tournament trace."""
        out = dict(row)
        out["generation"] = trace.final_answer
        out["autoreason_trace"] = trace.to_dict()
        out["autoreason_iterations"] = len(trace.iterations)
        out["autoreason_converged"] = bool(trace.converged)
        out["model_name"] = self.llm.model_name
        # Per-framework convention, stash metadata under DISTILABEL_METADATA_KEY.
        meta = out.get(DISTILABEL_METADATA_KEY, {}) or {}
        meta[f"statistics_{self.name}"] = {
            "total_calls": trace.total_calls,
            "winner_source": trace.winner_source,
        }
        out[DISTILABEL_METADATA_KEY] = meta
        return out

    def _empty_row(self, row: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Assemble a failure row. All tournament outputs are blanked out."""
        out = dict(row)
        out["generation"] = None
        out["autoreason_trace"] = None
        out["autoreason_iterations"] = 0
        out["autoreason_converged"] = False
        out["model_name"] = self.llm.model_name
        meta = out.get(DISTILABEL_METADATA_KEY, {}) or {}
        meta[f"autoreason_error_{self.name}"] = error_message
        out[DISTILABEL_METADATA_KEY] = meta
        return out

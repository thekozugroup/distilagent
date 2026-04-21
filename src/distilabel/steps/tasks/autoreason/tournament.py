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

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from distilabel.steps.tasks.autoreason.borda import borda_count, pick_winner
from distilabel.steps.tasks.autoreason.rate_limit import (
    AsyncTokenBucket,
    rate_limited_call,
)
from distilabel.steps.tasks.autoreason.roles import (
    parse_critique,
    parse_judge_ranking,
    render_author_b,
    render_critic,
    render_judge,
    render_synthesizer,
    render_teacher_seed,
)
from distilabel.steps.tasks.autoreason.types import (
    AutoReasonError,
    IterationResult,
    JudgeVote,
    TournamentTrace,
)

if TYPE_CHECKING:
    from distilabel.models.llms.base import LLM

_LOG = logging.getLogger(__name__)


RoleLLMs = Dict[str, "LLM"]          # keys: teacher, critic, author_b, synthesizer
JudgePool = List["LLM"]               # round-robin per judge call
LimiterMap = Dict[str, AsyncTokenBucket]  # keyed by LLM.model_name


async def _invoke(llm: "LLM", messages: list, limiter: Optional[AsyncTokenBucket]) -> str:
    """Single-shot LLM call returning plain text. Applies rate limiting when given.

    Supports both sync `LLM.generate` and `AsyncLLM.agenerate`. Extracts the
    first generation string from `GenerateOutput`.
    """
    async def _call():
        agen = getattr(llm, "agenerate", None)
        if agen is not None and asyncio.iscoroutinefunction(agen):
            result = await agen(input=messages, num_generations=1)
            generations = result.get("generations") or []
        else:
            result_list = await asyncio.to_thread(
                llm.generate, [messages], 1
            )
            generations = (result_list[0] or {}).get("generations") or []

        if not generations:
            raise AutoReasonError("LLM returned no generations")
        text = generations[0]
        if text is None:
            raise AutoReasonError("LLM returned None generation")
        return text

    if limiter is None:
        return await _call()
    return await rate_limited_call(limiter, _call)


class TournamentRunner:
    """Executes a single AutoReason tournament for one instruction.

    Parameters
    ----------
    llm
        The teacher LLM. Used for all roles by default.
    num_judges
        Number of judges in the panel. Paper uses 7.
    max_iterations
        Upper bound on refinement passes.
    convergence_k
        Stop when the incumbent A wins ``k`` consecutive iterations.
    max_concurrency
        Semaphore bound for concurrent role calls within an iteration.
    rate_limiter
        Optional ``AsyncTokenBucket`` that every LLM call acquires from.
    rng_seed_base
        Base seed for judge label permutations. Per-judge seed is
        ``rng_seed_base + iteration * 1000 + judge_index``.
    """

    def __init__(
        self,
        llm: "LLM",
        num_judges: int = 7,
        max_iterations: int = 15,
        convergence_k: int = 2,
        max_concurrency: int = 8,
        rate_limiter: Optional[AsyncTokenBucket] = None,
        rng_seed_base: int = 0,
        role_llms: Optional[RoleLLMs] = None,
        judge_pool: Optional[JudgePool] = None,
        limiter_map: Optional[LimiterMap] = None,
    ) -> None:
        """
        Parameters
        ----------
        llm
            Fallback / default LLM. Used for any role not overridden by
            ``role_llms``, and for judges when ``judge_pool`` is not given.
        role_llms
            Optional per-role LLM assignment. Keys may include ``teacher``,
            ``critic``, ``author_b``, ``synthesizer``. Missing keys fall
            back to ``llm``.
        judge_pool
            Optional list of LLMs for the judge panel. Judges are assigned
            round-robin: judge ``j`` at iteration ``i`` uses
            ``pool[(i * num_judges + j) % len(pool)]``. If empty/None, all
            judges use ``llm``.
        limiter_map
            Per-model-name rate-limit buckets. Keyed by ``LLM.model_name``.
            If given, the runner looks up the right bucket per call; else
            ``rate_limiter`` is used for every call.
        """
        if num_judges < 1:
            raise ValueError("num_judges must be >= 1")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if convergence_k < 1:
            raise ValueError("convergence_k must be >= 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        self.llm = llm
        self.num_judges = num_judges
        self.max_iterations = max_iterations
        self.convergence_k = convergence_k
        self.max_concurrency = max_concurrency
        self.rate_limiter = rate_limiter
        self.rng_seed_base = rng_seed_base
        self.role_llms: RoleLLMs = dict(role_llms or {})
        self.judge_pool: JudgePool = list(judge_pool or [])
        self.limiter_map: LimiterMap = dict(limiter_map or {})

    def _llm_for_role(self, role: str) -> "LLM":
        return self.role_llms.get(role, self.llm)

    def _llm_for_judge(self, iteration: int, j: int) -> "LLM":
        if self.judge_pool:
            idx = (iteration * self.num_judges + j) % len(self.judge_pool)
            return self.judge_pool[idx]
        return self._llm_for_role("critic") if False else self.llm

    def _limiter_for(self, llm: "LLM") -> Optional[AsyncTokenBucket]:
        name = getattr(llm, "model_name", None)
        if name and self.limiter_map:
            return self.limiter_map.get(name, self.rate_limiter)
        return self.rate_limiter

    async def run(self, instruction: str) -> TournamentTrace:
        trace = TournamentTrace()
        sem = asyncio.Semaphore(self.max_concurrency)

        async def call_role(role: str, messages):
            llm = self._llm_for_role(role)
            limiter = self._limiter_for(llm)
            async with sem:
                trace.total_calls += 1
                return await _invoke(llm, messages, limiter)

        async def call_judge(iteration: int, j: int, messages):
            llm = self._llm_for_judge(iteration, j)
            limiter = self._limiter_for(llm)
            async with sem:
                trace.total_calls += 1
                return await _invoke(llm, messages, limiter)

        # 1. Seed incumbent A
        a_text = await call_role("teacher", render_teacher_seed(instruction))

        consecutive_a_wins = 0

        for i in range(self.max_iterations):
            # 2. Critique (sequential — B and AB need it)
            critique_raw = await call_role("critic", render_critic(instruction, a_text))
            critique_text, no_flaws = parse_critique(critique_raw)

            if no_flaws:
                # Critic says draft is strong. Count as an A-win.
                consecutive_a_wins += 1
                trace.iterations.append(
                    IterationResult(
                        iteration=i,
                        A=a_text,
                        critique=critique_text,
                        B="",
                        AB="",
                        votes=[],
                        borda={"A": 0, "B": 0, "AB": 0},
                        winner="A",
                        no_flaws=True,
                    )
                )
                if consecutive_a_wins >= self.convergence_k:
                    trace.converged = True
                    break
                continue

            # 3. B + AB in parallel
            b_task = asyncio.create_task(call_role("author_b", render_author_b(instruction, a_text, critique_text)))
            ab_task = asyncio.create_task(call_role("synthesizer", render_synthesizer(instruction, a_text, critique_text)))
            b_text, ab_text = await asyncio.gather(b_task, ab_task)

            # 4. Judge panel in parallel (blind labels per judge)
            votes: List[JudgeVote] = []
            judge_tasks = []
            permutations: List[dict] = []
            for j in range(self.num_judges):
                seed = self.rng_seed_base + i * 1000 + j
                messages, permutation = render_judge(
                    instruction, a_text, b_text, ab_text, rng_seed=seed
                )
                permutations.append(permutation)
                judge_tasks.append(asyncio.create_task(call_judge(i, j, messages)))

            judge_raws = await asyncio.gather(*judge_tasks, return_exceptions=True)

            for j, (raw, permutation) in enumerate(zip(judge_raws, permutations)):
                if isinstance(raw, BaseException):
                    _LOG.warning("Judge %d raised %s; dropping vote", j, raw)
                    votes.append(
                        JudgeVote(
                            judge_id=j,
                            ranking=[],
                            raw_response=None,
                            parsed_ok=False,
                            label_permutation=permutation,
                        )
                    )
                    continue
                ranking_real, parsed_ok = parse_judge_ranking(raw, permutation)
                votes.append(
                    JudgeVote(
                        judge_id=j,
                        ranking=ranking_real,  # type: ignore[arg-type]
                        raw_response=raw,
                        parsed_ok=parsed_ok,
                        label_permutation=permutation,
                    )
                )

            # Fail-safe: if majority of judges failed to parse, keep A (do nothing).
            parseable = sum(1 for v in votes if v.parsed_ok)
            if parseable * 2 <= self.num_judges:  # i.e. <= 50% valid
                _LOG.warning(
                    "Iteration %d: only %d/%d judges parseable; keeping A",
                    i, parseable, self.num_judges,
                )
                borda = {"A": 0, "B": 0, "AB": 0}
                winner = "A"
            else:
                borda = borda_count(votes)
                winner = pick_winner(borda)

            trace.iterations.append(
                IterationResult(
                    iteration=i,
                    A=a_text,
                    critique=critique_text,
                    B=b_text,
                    AB=ab_text,
                    votes=votes,
                    borda=borda,
                    winner=winner,
                    no_flaws=False,
                )
            )

            # 5. Convergence update
            if winner == "A":
                consecutive_a_wins += 1
                if consecutive_a_wins >= self.convergence_k:
                    trace.converged = True
                    break
            else:
                consecutive_a_wins = 0
                a_text = b_text if winner == "B" else ab_text

        trace.final_answer = a_text
        trace.winner_source = (
            trace.iterations[-1].winner if trace.iterations else "A"
        )
        return trace

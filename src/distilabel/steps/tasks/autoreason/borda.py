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

"""Borda count aggregation for AutoReason tournament judging.

Three candidates are always present: "A" (incumbent), "B" (adversarial
revision) and "AB" (synthesis). Each valid judge ranking contributes
2 points to its 1st place, 1 point to its 2nd place, and 0 points to
its 3rd place. Invalid votes (parsed_ok=False or malformed ranking)
are ignored entirely.
"""

from __future__ import annotations

from typing import Dict, List

from distilabel.steps.tasks.autoreason.types import CandidateLabel, JudgeVote

CANDIDATES: List[CandidateLabel] = ["A", "B", "AB"]
# Tie-breaker order: prefer A (do-nothing bias), then AB (synthesis), then B.
_TIE_BREAK_ORDER: List[CandidateLabel] = ["A", "AB", "B"]


def borda_count(votes: List[JudgeVote]) -> Dict[CandidateLabel, int]:
    """Aggregate judge rankings into Borda totals.

    For each valid vote (``parsed_ok=True`` and ``ranking`` is a permutation
    of ``{"A", "B", "AB"}``): 1st place gets 2 points, 2nd gets 1 point,
    3rd gets 0. Invalid votes are skipped.

    Args:
        votes: Judge votes produced by the panel.

    Returns:
        A dict mapping every candidate in ``{"A", "B", "AB"}`` to its total
        points. All three keys are always present; zero is used when a
        candidate received no points (or when no votes were valid).
    """
    totals: Dict[CandidateLabel, int] = {c: 0 for c in CANDIDATES}
    expected = set(CANDIDATES)

    for vote in votes:
        if not vote.parsed_ok:
            continue
        ranking = vote.ranking
        if ranking is None or len(ranking) != len(CANDIDATES):
            continue
        if set(ranking) != expected:
            # Missing or duplicated candidates make the ranking invalid.
            continue
        # 1st -> 2 pts, 2nd -> 1 pt, 3rd -> 0 pts.
        for position, candidate in enumerate(ranking):
            totals[candidate] += (len(CANDIDATES) - 1) - position

    return totals


def pick_winner(borda: Dict[CandidateLabel, int]) -> CandidateLabel:
    """Return the winning candidate from a Borda totals dict.

    Highest total wins. On a tie, prefer ``"A"`` (do-nothing bias); if
    ``"A"`` is not involved in the tie, prefer ``"AB"`` (synthesis) over
    ``"B"`` (adversarial).

    Args:
        borda: Output of :func:`borda_count`.

    Returns:
        The winning candidate label.
    """
    best_score = max(borda.get(c, 0) for c in CANDIDATES)
    for candidate in _TIE_BREAK_ORDER:
        if borda.get(candidate, 0) == best_score:
            return candidate
    # Unreachable: _TIE_BREAK_ORDER covers every candidate.
    return "A"

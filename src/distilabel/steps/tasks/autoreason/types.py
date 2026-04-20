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

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional

CandidateLabel = Literal["A", "B", "AB"]


@dataclass
class JudgeVote:
    judge_id: int
    ranking: List[CandidateLabel]
    raw_response: Optional[str] = None
    parsed_ok: bool = True
    label_permutation: Optional[Dict[str, CandidateLabel]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class IterationResult:
    iteration: int
    A: str
    critique: str
    B: str
    AB: str
    votes: List[JudgeVote]
    borda: Dict[CandidateLabel, int]
    winner: CandidateLabel
    no_flaws: bool = False

    def to_dict(self) -> Dict:
        return {
            "iteration": self.iteration,
            "A": self.A,
            "critique": self.critique,
            "B": self.B,
            "AB": self.AB,
            "votes": [v.to_dict() for v in self.votes],
            "borda": dict(self.borda),
            "winner": self.winner,
            "no_flaws": self.no_flaws,
        }


@dataclass
class TournamentTrace:
    iterations: List[IterationResult] = field(default_factory=list)
    converged: bool = False
    winner_source: Optional[CandidateLabel] = None
    total_calls: int = 0
    final_answer: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "iterations": [it.to_dict() for it in self.iterations],
            "converged": self.converged,
            "winner_source": self.winner_source,
            "total_calls": self.total_calls,
            "final_answer": self.final_answer,
        }


class AutoReasonError(Exception):
    """Raised when the AutoReason tournament cannot complete."""

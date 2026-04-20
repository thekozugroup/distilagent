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

from distilabel.steps.tasks.autoreason.borda import borda_count, pick_winner
from distilabel.steps.tasks.autoreason.types import JudgeVote


class TestBordaCount:
    def test_single_judge_simple_ranking(self) -> None:
        votes = [JudgeVote(judge_id=0, ranking=["AB", "A", "B"])]
        totals = borda_count(votes)

        assert totals == {"A": 1, "B": 0, "AB": 2}
        assert pick_winner(totals) == "AB"

    def test_seven_judges_mixed_rankings(self) -> None:
        # 4 judges prefer AB outright, 2 prefer A, 1 prefers B.
        votes = [
            JudgeVote(judge_id=0, ranking=["AB", "A", "B"]),
            JudgeVote(judge_id=1, ranking=["AB", "A", "B"]),
            JudgeVote(judge_id=2, ranking=["AB", "B", "A"]),
            JudgeVote(judge_id=3, ranking=["AB", "B", "A"]),
            JudgeVote(judge_id=4, ranking=["A", "AB", "B"]),
            JudgeVote(judge_id=5, ranking=["A", "AB", "B"]),
            JudgeVote(judge_id=6, ranking=["B", "AB", "A"]),
        ]
        totals = borda_count(votes)

        # A: 2*2 (judges 4,5) + 1*2 (judges 0,1) + 0*2 (judges 2,3) + 0 (j6) = 6
        # B: 0*2 (j0,j1) + 1*2 (j2,j3) + 0*2 (j4,j5) + 2 (j6) = 4
        # AB: 2*4 (j0-3) + 1*2 (j4,j5) + 1 (j6) = 11
        assert totals == {"A": 6, "B": 4, "AB": 11}
        assert pick_winner(totals) == "AB"

    def test_tie_between_a_and_b_prefers_a(self) -> None:
        # Two judges split A vs B at 1st place with AB always last.
        votes = [
            JudgeVote(judge_id=0, ranking=["A", "B", "AB"]),
            JudgeVote(judge_id=1, ranking=["B", "A", "AB"]),
        ]
        totals = borda_count(votes)

        assert totals == {"A": 3, "B": 3, "AB": 0}
        assert pick_winner(totals) == "A"

    def test_tie_between_b_and_ab_prefers_ab(self) -> None:
        # A is beaten; B and AB tie; synthesis (AB) should win.
        votes = [
            JudgeVote(judge_id=0, ranking=["B", "AB", "A"]),
            JudgeVote(judge_id=1, ranking=["AB", "B", "A"]),
        ]
        totals = borda_count(votes)

        assert totals == {"A": 0, "B": 3, "AB": 3}
        assert pick_winner(totals) == "AB"

    def test_all_invalid_votes_returns_zero_and_a(self) -> None:
        votes = [
            JudgeVote(judge_id=0, ranking=[], parsed_ok=False),
            JudgeVote(
                judge_id=1,
                ranking=["A", "B", "AB"],
                parsed_ok=False,
            ),
        ]
        totals = borda_count(votes)

        assert totals == {"A": 0, "B": 0, "AB": 0}
        assert pick_winner(totals) == "A"

    def test_empty_votes_list(self) -> None:
        totals = borda_count([])

        assert totals == {"A": 0, "B": 0, "AB": 0}
        assert pick_winner(totals) == "A"

    def test_mixed_valid_and_invalid_votes(self) -> None:
        votes = [
            JudgeVote(judge_id=0, ranking=["AB", "A", "B"]),
            JudgeVote(
                judge_id=1,
                ranking=["B", "AB", "A"],
                parsed_ok=False,  # ignored despite well-formed ranking
            ),
            # Malformed ranking: missing "AB" -> must be skipped.
            JudgeVote(judge_id=2, ranking=["A", "B"], parsed_ok=True),
            # Malformed ranking: duplicate candidate -> skipped.
            JudgeVote(
                judge_id=3,
                ranking=["A", "A", "B"],
                parsed_ok=True,
            ),
            JudgeVote(judge_id=4, ranking=["A", "AB", "B"]),
        ]
        totals = borda_count(votes)

        # Only judges 0 and 4 count.
        # Judge 0: AB=2, A=1, B=0
        # Judge 4: A=2, AB=1, B=0
        assert totals == {"A": 3, "B": 0, "AB": 3}
        # Tie between A and AB at 3 -> "A" wins via do-nothing bias.
        assert pick_winner(totals) == "A"

    def test_unanimous_ranking_max_points(self) -> None:
        n_judges = 5
        votes = [
            JudgeVote(judge_id=i, ranking=["AB", "A", "B"])
            for i in range(n_judges)
        ]
        totals = borda_count(votes)

        assert totals == {"A": n_judges, "B": 0, "AB": 2 * n_judges}
        assert pick_winner(totals) == "AB"


class TestPickWinnerTieBreakers:
    def test_three_way_tie_prefers_a(self) -> None:
        assert pick_winner({"A": 0, "B": 0, "AB": 0}) == "A"

    def test_a_wins_outright(self) -> None:
        assert pick_winner({"A": 10, "B": 5, "AB": 7}) == "A"

    def test_b_wins_outright(self) -> None:
        assert pick_winner({"A": 1, "B": 9, "AB": 4}) == "B"

    def test_ab_wins_outright(self) -> None:
        assert pick_winner({"A": 2, "B": 3, "AB": 11}) == "AB"

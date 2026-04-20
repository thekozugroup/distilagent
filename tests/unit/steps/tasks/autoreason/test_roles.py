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

import pytest

from distilabel.steps.tasks.autoreason.roles import (
    parse_critique,
    parse_judge_ranking,
    render_author_b,
    render_critic,
    render_judge,
    render_synthesizer,
    render_teacher_seed,
)


INSTRUCTION = "Explain why the sky appears blue during the day."
DRAFT = "The sky is blue because of Rayleigh scattering of sunlight by air molecules."
CRITIQUE = "The intro is weak because it does not mention wavelength dependence."


def _roles(messages):
    return [m["role"] for m in messages]


def _joined(messages):
    return "\n".join(m["content"] for m in messages)


class TestRenderTeacherSeed:
    def test_two_messages_system_then_user(self):
        msgs = render_teacher_seed(INSTRUCTION)
        assert len(msgs) == 2
        assert _roles(msgs) == ["system", "user"]

    def test_user_content_contains_instruction_verbatim(self):
        msgs = render_teacher_seed(INSTRUCTION)
        assert INSTRUCTION in msgs[1]["content"]


class TestRenderCritic:
    def test_contains_draft_and_no_flaws_literal(self):
        msgs = render_critic(INSTRUCTION, DRAFT)
        full = _joined(msgs)
        assert DRAFT in full
        assert "NO FLAWS" in full

    def test_structure(self):
        msgs = render_critic(INSTRUCTION, DRAFT)
        assert _roles(msgs) == ["system", "user"]
        assert INSTRUCTION in msgs[1]["content"]


class TestRenderAuthorB:
    def test_contains_draft_and_critique(self):
        msgs = render_author_b(INSTRUCTION, DRAFT, CRITIQUE)
        full = _joined(msgs)
        assert DRAFT in full
        assert CRITIQUE in full
        assert INSTRUCTION in full

    def test_scope_bound_language_present(self):
        msgs = render_author_b(INSTRUCTION, DRAFT, CRITIQUE)
        full = _joined(msgs).lower()
        # adversarial but scope-bounded
        assert "do not expand scope" in full
        assert "do not pad length" in full


class TestRenderSynthesizer:
    def test_contains_draft_and_critique(self):
        msgs = render_synthesizer(INSTRUCTION, DRAFT, CRITIQUE)
        full = _joined(msgs)
        assert DRAFT in full
        assert CRITIQUE in full

    def test_stronger_minimal_change_wording_than_author_b(self):
        synth = _joined(render_synthesizer(INSTRUCTION, DRAFT, CRITIQUE)).lower()
        author = _joined(render_author_b(INSTRUCTION, DRAFT, CRITIQUE)).lower()
        # Synthesizer should explicitly demand the "smallest possible change"
        # / "minimal" framing which is not present in author_b.
        assert "smallest possible change" in synth
        assert "smallest possible change" not in author
        assert "minimal" in synth


class TestRenderJudge:
    def test_deterministic_permutation(self):
        _, perm1 = render_judge(INSTRUCTION, "a_text", "b_text", "ab_text", rng_seed=42)
        _, perm2 = render_judge(INSTRUCTION, "a_text", "b_text", "ab_text", rng_seed=42)
        assert perm1 == perm2

    def test_permutation_is_bijection(self):
        _, perm = render_judge(INSTRUCTION, "a_text", "b_text", "ab_text", rng_seed=42)
        assert set(perm.keys()) == {"X1", "X2", "X3"}
        assert set(perm.values()) == {"A", "B", "AB"}

    def test_candidates_appear_under_permuted_labels(self):
        texts = {"A": "ALPHA_TEXT_42", "B": "BRAVO_TEXT_42", "AB": "AB_TEXT_42"}
        msgs, perm = render_judge(
            INSTRUCTION, texts["A"], texts["B"], texts["AB"], rng_seed=42
        )
        user_content = msgs[-1]["content"]
        for display_label, real_label in perm.items():
            text = texts[real_label]
            assert display_label in user_content
            assert text in user_content
            # The text must appear after the corresponding display label.
            label_idx = user_content.index(display_label)
            text_idx = user_content.index(text)
            assert text_idx > label_idx

    def test_instruction_appears_in_prompt(self):
        msgs, _ = render_judge(
            INSTRUCTION, "a_text", "b_text", "ab_text", rng_seed=42
        )
        assert INSTRUCTION in _joined(msgs)

    def test_ranking_keyword_present(self):
        msgs, _ = render_judge(
            INSTRUCTION, "a_text", "b_text", "ab_text", rng_seed=42
        )
        assert "RANKING" in _joined(msgs)

    def test_different_seeds_yield_different_permutations(self):
        perms = []
        for seed in (0, 1, 2, 3, 4):
            _, perm = render_judge(
                INSTRUCTION, "a_text", "b_text", "ab_text", rng_seed=seed
            )
            perms.append(tuple(sorted(perm.items())))
        assert len(set(perms)) >= 2


class TestParseCritique:
    def test_no_flaws_exact(self):
        text, flag = parse_critique("NO FLAWS")
        assert flag is True
        assert text == "NO FLAWS"

    def test_no_flaws_case_insensitive_embedded(self):
        text, flag = parse_critique("no flaws found")
        assert flag is True
        assert text  # stripped raw text retained

    def test_substantive_critique(self):
        text, flag = parse_critique("The intro is weak because X")
        assert flag is False
        assert "intro is weak" in text

    def test_empty(self):
        text, flag = parse_critique("")
        assert flag is False
        assert text == ""


class TestParseJudgeRanking:
    perm = {"X1": "A", "X2": "B", "X3": "AB"}

    def test_canonical(self):
        ranking, ok = parse_judge_ranking("RANKING: X1 > X3 > X2", self.perm)
        assert ok is True
        assert ranking == ["A", "AB", "B"]

    def test_lowercase(self):
        ranking, ok = parse_judge_ranking("ranking: x2 > x1 > x3", self.perm)
        assert ok is True
        assert ranking == ["B", "A", "AB"]

    def test_missing_prefix(self):
        ranking, ok = parse_judge_ranking("X1 > X2 > X3", self.perm)
        assert ok is True
        assert ranking == ["A", "B", "AB"]

    def test_comma_separators(self):
        ranking, ok = parse_judge_ranking("RANKING: X3, X1, X2", self.perm)
        assert ok is True
        assert ranking == ["AB", "A", "B"]

    def test_and_separator(self):
        ranking, ok = parse_judge_ranking("X1 and X2 and X3", self.perm)
        assert ok is True
        assert ranking == ["A", "B", "AB"]

    def test_wrong_count(self):
        ranking, ok = parse_judge_ranking("RANKING: X1 > X2", self.perm)
        assert ok is False
        assert ranking == []

    def test_duplicate_label(self):
        ranking, ok = parse_judge_ranking("RANKING: X1 > X1 > X2", self.perm)
        assert ok is False
        assert ranking == []

    def test_unknown_label(self):
        ranking, ok = parse_judge_ranking("RANKING: X1 > X2 > X4", self.perm)
        assert ok is False
        assert ranking == []

    def test_empty_response(self):
        ranking, ok = parse_judge_ranking("", self.perm)
        assert ok is False
        assert ranking == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

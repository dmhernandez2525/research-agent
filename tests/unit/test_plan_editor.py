"""Unit tests for research_agent.plan_editor - plan editing utilities."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from research_agent.plan_editor import (
    EditableSubQuestion,
    EditedPlan,
    _get_editor,
    _yaml_quote,
    edit_plan_in_editor,
    edit_plan_inline,
    plan_to_yaml,
    yaml_to_plan,
)

# ---------------------------------------------------------------------------
# EditableSubQuestion model tests
# ---------------------------------------------------------------------------


class TestEditableSubQuestion:
    """EditableSubQuestion Pydantic model validation."""

    def test_valid_construction(self) -> None:
        sq = EditableSubQuestion(id=1, question="What is X?")
        assert sq.id == 1
        assert sq.question == "What is X?"
        assert sq.rationale == ""

    def test_with_rationale(self) -> None:
        sq = EditableSubQuestion(id=2, question="Q?", rationale="Because")
        assert sq.rationale == "Because"

    def test_id_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            EditableSubQuestion(id=0, question="Q?")

    def test_id_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            EditableSubQuestion(id=-1, question="Q?")

    def test_question_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            EditableSubQuestion(id=1, question="")


# ---------------------------------------------------------------------------
# EditedPlan model tests
# ---------------------------------------------------------------------------


class TestEditedPlan:
    """EditedPlan Pydantic model validation."""

    def test_single_sub_question(self) -> None:
        plan = EditedPlan(subtopics=[EditableSubQuestion(id=1, question="Q1")])
        assert len(plan.subtopics) == 1

    def test_renumbers_ids(self) -> None:
        plan = EditedPlan(
            subtopics=[
                EditableSubQuestion(id=10, question="First"),
                EditableSubQuestion(id=20, question="Second"),
                EditableSubQuestion(id=30, question="Third"),
            ]
        )
        assert plan.subtopics[0].id == 1
        assert plan.subtopics[1].id == 2
        assert plan.subtopics[2].id == 3

    def test_rejects_empty_list(self) -> None:
        with pytest.raises(ValueError):
            EditedPlan(subtopics=[])

    def test_max_20_sub_questions(self) -> None:
        sqs = [EditableSubQuestion(id=i, question=f"Q{i}") for i in range(1, 21)]
        plan = EditedPlan(subtopics=sqs)
        assert len(plan.subtopics) == 20

    def test_rejects_over_20_sub_questions(self) -> None:
        sqs = [EditableSubQuestion(id=i, question=f"Q{i}") for i in range(1, 22)]
        with pytest.raises(ValueError):
            EditedPlan(subtopics=sqs)


# ---------------------------------------------------------------------------
# _yaml_quote tests
# ---------------------------------------------------------------------------


class TestYamlQuote:
    """YAML string quoting utility."""

    def test_plain_string_unquoted(self) -> None:
        assert _yaml_quote("simple text") == "simple text"

    def test_colon_triggers_quoting(self) -> None:
        result = _yaml_quote("key: value")
        assert result.startswith('"')
        assert result.endswith('"')

    def test_question_mark_triggers_quoting(self) -> None:
        result = _yaml_quote("What is this?")
        assert result.startswith('"')

    def test_hash_triggers_quoting(self) -> None:
        result = _yaml_quote("topic #1")
        assert result.startswith('"')

    def test_newline_triggers_quoting(self) -> None:
        result = _yaml_quote("line1\nline2")
        assert result.startswith('"')

    def test_escapes_backslash(self) -> None:
        result = _yaml_quote("path\\to\\file")
        assert '\\\\"' not in result  # Should escape backslash, not double-escape
        assert "\\\\" in result

    def test_escapes_double_quote(self) -> None:
        result = _yaml_quote('He said "hello"')
        assert '\\"hello\\"' in result

    def test_empty_string(self) -> None:
        assert _yaml_quote("") == ""

    def test_brackets_trigger_quoting(self) -> None:
        result = _yaml_quote("[array] syntax")
        assert result.startswith('"')

    def test_braces_trigger_quoting(self) -> None:
        result = _yaml_quote("{mapping} syntax")
        assert result.startswith('"')


# ---------------------------------------------------------------------------
# plan_to_yaml tests
# ---------------------------------------------------------------------------


class TestPlanToYaml:
    """YAML serialization of sub-questions."""

    def test_contains_header(self) -> None:
        yaml_str = plan_to_yaml([])
        assert "Research Plan Editor" in yaml_str
        assert "subtopics:" in yaml_str

    def test_includes_question_text(self) -> None:
        sqs = [{"id": 1, "question": "What is RAG?", "rationale": "Core"}]
        yaml_str = plan_to_yaml(sqs)
        assert "What is RAG?" in yaml_str

    def test_includes_rationale(self) -> None:
        sqs = [{"id": 1, "question": "Q?", "rationale": "Important topic"}]
        yaml_str = plan_to_yaml(sqs)
        assert "Important topic" in yaml_str

    def test_multiple_sub_questions(self) -> None:
        sqs = [
            {"id": 1, "question": "Q1?", "rationale": "R1"},
            {"id": 2, "question": "Q2?", "rationale": "R2"},
        ]
        yaml_str = plan_to_yaml(sqs)
        assert "Q1?" in yaml_str
        assert "Q2?" in yaml_str

    def test_missing_fields_default(self) -> None:
        sqs = [{"other_field": "value"}]
        yaml_str = plan_to_yaml(sqs)
        assert "id: 0" in yaml_str

    def test_special_characters_properly_quoted(self) -> None:
        sqs = [
            {"id": 1, "question": "What's the impact: big or small?", "rationale": ""}
        ]
        yaml_str = plan_to_yaml(sqs)
        # Should be parseable
        plan = yaml_to_plan(yaml_str)
        assert plan is not None


# ---------------------------------------------------------------------------
# yaml_to_plan tests
# ---------------------------------------------------------------------------


class TestYamlToPlan:
    """YAML deserialization back to EditedPlan."""

    def test_valid_yaml(self) -> None:
        yaml_str = (
            "subtopics:\n"
            "  - id: 1\n"
            "    question: Test question\n"
            "    rationale: Test rationale\n"
        )
        plan = yaml_to_plan(yaml_str)
        assert plan is not None
        assert plan.subtopics[0].question == "Test question"

    def test_empty_string_returns_none(self) -> None:
        assert yaml_to_plan("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert yaml_to_plan("   \n\n  ") is None

    def test_invalid_yaml_returns_none(self) -> None:
        assert yaml_to_plan("{{invalid: yaml:::") is None

    def test_non_dict_returns_none(self) -> None:
        assert yaml_to_plan("- item1\n- item2") is None

    def test_missing_sub_questions_key_returns_none(self) -> None:
        assert yaml_to_plan("other_key: value") is None

    def test_empty_sub_questions_returns_none(self) -> None:
        assert yaml_to_plan("subtopics: []") is None

    def test_sub_questions_not_list_returns_none(self) -> None:
        assert yaml_to_plan("subtopics: not_a_list") is None

    def test_invalid_sub_question_returns_none(self) -> None:
        yaml_str = 'subtopics:\n  - id: 0\n    question: ""\n'
        assert yaml_to_plan(yaml_str) is None

    def test_comments_ignored(self) -> None:
        yaml_str = "# A comment\nsubtopics:\n  - id: 1\n    question: Q1\n"
        plan = yaml_to_plan(yaml_str)
        assert plan is not None


# ---------------------------------------------------------------------------
# _get_editor tests
# ---------------------------------------------------------------------------


class TestGetEditor:
    """Editor detection from environment."""

    def test_visual_preferred(self) -> None:
        with patch.dict(os.environ, {"VISUAL": "code", "EDITOR": "nano"}):
            assert _get_editor() == "code"

    def test_editor_fallback(self) -> None:
        with patch.dict(os.environ, {"EDITOR": "nano"}, clear=False):
            env = os.environ.copy()
            env.pop("VISUAL", None)
            with patch.dict(os.environ, env, clear=True):
                assert _get_editor() == "nano"

    def test_default_vi(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _get_editor() == "vi"


# ---------------------------------------------------------------------------
# edit_plan_in_editor tests
# ---------------------------------------------------------------------------


class TestEditPlanInEditor:
    """$EDITOR-based plan editing."""

    def test_successful_edit(self) -> None:
        sqs = [{"id": 1, "question": "Original", "rationale": "R"}]
        edited_yaml = (
            "subtopics:\n"
            "  - id: 1\n"
            "    question: Modified\n"
            "    rationale: Updated\n"
        )

        def mock_run(args: list[str], **kwargs: Any) -> MagicMock:
            with open(args[1], "w") as f:
                f.write(edited_yaml)
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("research_agent.plan_editor.subprocess.run", side_effect=mock_run):
            plan = edit_plan_in_editor(sqs)

        assert plan is not None
        assert plan.subtopics[0].question == "Modified"

    def test_editor_exit_nonzero(self) -> None:
        sqs = [{"id": 1, "question": "Q", "rationale": "R"}]
        mock_result = MagicMock(returncode=1)

        with patch(
            "research_agent.plan_editor.subprocess.run", return_value=mock_result
        ):
            plan = edit_plan_in_editor(sqs)

        assert plan is None

    def test_user_empties_file_cancels(self) -> None:
        sqs = [{"id": 1, "question": "Q", "rationale": "R"}]

        def mock_run(args: list[str], **kwargs: Any) -> MagicMock:
            with open(args[1], "w") as f:
                f.write("")
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("research_agent.plan_editor.subprocess.run", side_effect=mock_run):
            plan = edit_plan_in_editor(sqs)

        assert plan is None

    def test_temp_file_cleaned_up(self) -> None:
        sqs = [{"id": 1, "question": "Q", "rationale": "R"}]
        captured_path = {}

        def mock_run(args: list[str], **kwargs: Any) -> MagicMock:
            captured_path["path"] = args[1]
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("research_agent.plan_editor.subprocess.run", side_effect=mock_run):
            edit_plan_in_editor(sqs)

        assert not os.path.exists(captured_path["path"])


# ---------------------------------------------------------------------------
# edit_plan_inline tests
# ---------------------------------------------------------------------------


class TestEditPlanInline:
    """Inline plan editing via numbered removal."""

    def test_remove_single_item(self) -> None:
        sqs = [
            {"id": 1, "question": "Q1", "rationale": "R1"},
            {"id": 2, "question": "Q2", "rationale": "R2"},
        ]
        with patch("builtins.input", return_value="1"):
            plan = edit_plan_inline(sqs)
        assert plan is not None
        assert len(plan.subtopics) == 1
        assert plan.subtopics[0].question == "Q2"

    def test_remove_multiple_items(self) -> None:
        sqs = [
            {"id": 1, "question": "Q1", "rationale": "R1"},
            {"id": 2, "question": "Q2", "rationale": "R2"},
            {"id": 3, "question": "Q3", "rationale": "R3"},
        ]
        with patch("builtins.input", return_value="1, 3"):
            plan = edit_plan_inline(sqs)
        assert plan is not None
        assert len(plan.subtopics) == 1
        assert plan.subtopics[0].question == "Q2"

    def test_remove_all_returns_none(self) -> None:
        sqs = [{"id": 1, "question": "Q1", "rationale": "R1"}]
        with patch("builtins.input", return_value="1"):
            plan = edit_plan_inline(sqs)
        assert plan is None

    def test_cancel_with_c(self) -> None:
        sqs = [{"id": 1, "question": "Q1", "rationale": "R1"}]
        with patch("builtins.input", return_value="c"):
            plan = edit_plan_inline(sqs)
        assert plan is None

    def test_cancel_with_uppercase_c(self) -> None:
        sqs = [{"id": 1, "question": "Q1", "rationale": "R1"}]
        with patch("builtins.input", return_value="C"):
            plan = edit_plan_inline(sqs)
        assert plan is None

    def test_empty_input_returns_none(self) -> None:
        sqs = [{"id": 1, "question": "Q1", "rationale": "R1"}]
        with patch("builtins.input", return_value=""):
            plan = edit_plan_inline(sqs)
        assert plan is None

    def test_invalid_number_returns_none(self) -> None:
        sqs = [{"id": 1, "question": "Q1", "rationale": "R1"}]
        with patch("builtins.input", return_value="abc"):
            plan = edit_plan_inline(sqs)
        assert plan is None

    def test_eof_returns_none(self) -> None:
        sqs = [{"id": 1, "question": "Q1", "rationale": "R1"}]
        with patch("builtins.input", side_effect=EOFError):
            plan = edit_plan_inline(sqs)
        assert plan is None

    def test_keyboard_interrupt_returns_none(self) -> None:
        sqs = [{"id": 1, "question": "Q1", "rationale": "R1"}]
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            plan = edit_plan_inline(sqs)
        assert plan is None

    def test_nonexistent_id_keeps_all(self) -> None:
        sqs = [
            {"id": 1, "question": "Q1", "rationale": "R1"},
            {"id": 2, "question": "Q2", "rationale": "R2"},
        ]
        with patch("builtins.input", return_value="99"):
            plan = edit_plan_inline(sqs)
        assert plan is not None
        assert len(plan.subtopics) == 2

    def test_renumbers_after_removal(self) -> None:
        sqs = [
            {"id": 1, "question": "Q1", "rationale": "R1"},
            {"id": 2, "question": "Q2", "rationale": "R2"},
            {"id": 3, "question": "Q3", "rationale": "R3"},
        ]
        with patch("builtins.input", return_value="2"):
            plan = edit_plan_inline(sqs)
        assert plan is not None
        assert plan.subtopics[0].id == 1
        assert plan.subtopics[1].id == 2

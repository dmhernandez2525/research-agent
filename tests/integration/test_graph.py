"""Integration tests for graph construction and conditional edge routing."""

from __future__ import annotations

import unittest.mock
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from research_agent.graph import (
    _MAX_SEARCH_RETRIES,
    _all_subtopics_done,
    _should_continue_scrape,
    _should_continue_search,
    build_graph,
    compile_graph,
)

if TYPE_CHECKING:
    from pathlib import Path

    from research_agent.state import ResearchState

pytestmark = pytest.mark.integration


# ---- Graph construction ------------------------------------------------------


class TestBuildGraph:
    """Verify build_graph returns a well-formed StateGraph."""

    @pytest.fixture()
    def settings(self) -> MagicMock:
        s = MagicMock()
        s.checkpoints.enabled = False
        return s

    def test_returns_state_graph(self, settings: MagicMock) -> None:
        graph = build_graph(settings)
        assert graph is not None

    def test_graph_has_five_nodes(self, settings: MagicMock) -> None:
        graph = build_graph(settings)
        node_names = set(graph.nodes.keys())
        expected = {"plan", "search", "scrape", "summarize", "synthesize"}
        assert expected.issubset(node_names)

    def test_graph_compiles(self, settings: MagicMock) -> None:
        graph = build_graph(settings)
        compiled = graph.compile()
        assert compiled is not None


# ---- _should_continue_search -------------------------------------------------


class TestShouldContinueSearch:
    """Conditional routing after search node."""

    def test_routes_to_scrape_with_enough_results(self) -> None:
        state: ResearchState = {
            "search_results": [MagicMock()] * 5,
            "search_retry_count": 0,
        }
        assert _should_continue_search(state) == "scrape"

    def test_routes_to_search_with_insufficient_results(self) -> None:
        state: ResearchState = {
            "search_results": [MagicMock()],
            "search_retry_count": 0,
        }
        assert _should_continue_search(state) == "search"

    def test_routes_to_scrape_at_max_retries(self) -> None:
        state: ResearchState = {
            "search_results": [],
            "search_retry_count": _MAX_SEARCH_RETRIES,
        }
        assert _should_continue_search(state) == "scrape"

    def test_routes_to_scrape_above_max_retries(self) -> None:
        state: ResearchState = {
            "search_results": [],
            "search_retry_count": _MAX_SEARCH_RETRIES + 1,
        }
        assert _should_continue_search(state) == "scrape"

    def test_empty_state_retries(self) -> None:
        state: ResearchState = {}
        assert _should_continue_search(state) == "search"

    def test_exactly_min_results_routes_to_scrape(self) -> None:
        state: ResearchState = {
            "search_results": [MagicMock()] * 3,
            "search_retry_count": 0,
        }
        assert _should_continue_search(state) == "scrape"


# ---- _should_continue_scrape ------------------------------------------------


class TestShouldContinueScrape:
    """Conditional routing after scrape node."""

    def test_routes_to_summarize_with_content(self) -> None:
        state: ResearchState = {"scraped_pages": [MagicMock()]}
        assert _should_continue_scrape(state) == "summarize"

    def test_routes_to_end_without_content(self) -> None:
        state: ResearchState = {"scraped_pages": []}
        result = _should_continue_scrape(state)
        assert result == "__end__"

    def test_empty_state_routes_to_end(self) -> None:
        state: ResearchState = {}
        result = _should_continue_scrape(state)
        assert result == "__end__"


# ---- _all_subtopics_done ----------------------------------------------------


class TestAllSubtopicsDone:
    """Conditional routing after summarize node for subtopic iteration."""

    def test_routes_to_search_when_subtopics_remain(self) -> None:
        state: ResearchState = {
            "subtopics": [MagicMock(), MagicMock(), MagicMock()],
            "current_subtopic_index": 1,
        }
        assert _all_subtopics_done(state) == "search"

    def test_routes_to_synthesize_when_all_done(self) -> None:
        state: ResearchState = {
            "subtopics": [MagicMock(), MagicMock()],
            "current_subtopic_index": 2,
        }
        assert _all_subtopics_done(state) == "synthesize"

    def test_routes_to_synthesize_when_index_exceeds(self) -> None:
        state: ResearchState = {
            "subtopics": [MagicMock()],
            "current_subtopic_index": 5,
        }
        assert _all_subtopics_done(state) == "synthesize"

    def test_empty_subtopics_routes_to_synthesize(self) -> None:
        state: ResearchState = {
            "subtopics": [],
            "current_subtopic_index": 0,
        }
        assert _all_subtopics_done(state) == "synthesize"

    def test_index_zero_with_subtopics_routes_to_search(self) -> None:
        state: ResearchState = {
            "subtopics": [MagicMock()],
            "current_subtopic_index": 0,
        }
        assert _all_subtopics_done(state) == "search"

    def test_empty_state_routes_to_synthesize(self) -> None:
        state: ResearchState = {}
        assert _all_subtopics_done(state) == "synthesize"


# ---- compile_graph -----------------------------------------------------------


class TestCompileGraph:
    """Verify compile_graph compiles the graph with/without checkpointing."""

    @staticmethod
    def _sqlite_module_patches():
        """Context manager that patches the langgraph.checkpoint.sqlite.aio module.

        Returns a tuple of (context_manager, mock_saver) where mock_saver
        is the mock for AsyncSqliteSaver.
        """
        import sys

        mock_aio_module = MagicMock()
        mock_saver = MagicMock()
        mock_aio_module.AsyncSqliteSaver = mock_saver

        modules_patch = {
            "langgraph.checkpoint.sqlite": MagicMock(),
            "langgraph.checkpoint.sqlite.aio": mock_aio_module,
        }
        return unittest.mock.patch.dict(sys.modules, modules_patch), mock_saver

    def test_compile_without_checkpoints(self) -> None:
        """When checkpoints are disabled, compile_graph returns a compiled graph
        without creating a checkpointer."""
        settings = MagicMock()
        settings.checkpoints.enabled = False

        ctx, mock_saver = self._sqlite_module_patches()
        with ctx, unittest.mock.patch(
            "research_agent.graph.build_graph"
        ) as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            compiled = compile_graph(settings)

        assert compiled is not None
        mock_saver.from_conn_string.assert_not_called()
        mock_graph.compile.assert_called_once_with(checkpointer=None)

    def test_compile_with_checkpoints_creates_db(self, tmp_path: Path) -> None:
        """When checkpoints are enabled and a checkpoint_db path is provided,
        the parent directory is created and AsyncSqliteSaver is initialized."""
        settings = MagicMock()
        settings.checkpoints.enabled = True

        db_path = tmp_path / "sub" / "langgraph.db"

        ctx, mock_saver = self._sqlite_module_patches()
        mock_checkpointer = MagicMock()
        mock_saver.from_conn_string.return_value = mock_checkpointer

        with ctx, unittest.mock.patch(
            "research_agent.graph.build_graph"
        ) as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            compiled = compile_graph(settings, checkpoint_db=db_path)

        assert compiled is not None
        assert db_path.parent.exists()
        mock_saver.from_conn_string.assert_called_once_with(str(db_path))
        mock_graph.compile.assert_called_once_with(checkpointer=mock_checkpointer)

    def test_compile_with_checkpoints_uses_default_db_path(
        self, tmp_path: Path
    ) -> None:
        """When checkpoints are enabled and checkpoint_db is None, the default
        path from settings.checkpoints.directory / 'langgraph.db' is used."""
        settings = MagicMock()
        settings.checkpoints.enabled = True
        settings.checkpoints.directory = tmp_path

        expected_db = tmp_path / "langgraph.db"

        ctx, mock_saver = self._sqlite_module_patches()
        mock_checkpointer = MagicMock()
        mock_saver.from_conn_string.return_value = mock_checkpointer

        with ctx, unittest.mock.patch(
            "research_agent.graph.build_graph"
        ) as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            compiled = compile_graph(settings, checkpoint_db=None)

        assert compiled is not None
        mock_saver.from_conn_string.assert_called_once_with(str(expected_db))
        mock_graph.compile.assert_called_once_with(checkpointer=mock_checkpointer)

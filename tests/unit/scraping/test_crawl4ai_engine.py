"""Unit tests for research_agent.scraping.crawl4ai_engine."""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_agent.scraping.crawl4ai_engine import crawl4ai_extract

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_crawl4ai_module(mock_crawler_cls: MagicMock) -> types.ModuleType:
    """Create a fake crawl4ai module with AsyncWebCrawler."""
    mod = types.ModuleType("crawl4ai")
    mod.AsyncWebCrawler = mock_crawler_cls  # type: ignore[attr-defined]
    return mod


def _make_crawler_mock(
    success: bool = True,
    markdown: str = "Content.",
    title: str = "",
    error_message: str = "",
    side_effect: Exception | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Build mock crawler class and instance.

    Returns:
        (mock_cls, mock_instance) where mock_cls() returns mock_instance.
    """
    mock_result = MagicMock()
    mock_result.success = success
    mock_result.markdown = markdown
    mock_result.title = title
    if error_message:
        mock_result.error_message = error_message

    mock_instance = AsyncMock()
    if side_effect:
        mock_instance.arun = AsyncMock(side_effect=side_effect)
    else:
        mock_instance.arun = AsyncMock(return_value=mock_result)
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=None)

    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls, mock_instance


# ---------------------------------------------------------------------------
# TestCrawl4aiExtract
# ---------------------------------------------------------------------------


class TestCrawl4aiExtract:
    """crawl4ai_extract handles JS-heavy page extraction."""

    @pytest.mark.asyncio()
    async def test_returns_none_when_crawl4ai_not_installed(self) -> None:
        with patch.dict(sys.modules, {"crawl4ai": None}):
            result = await crawl4ai_extract("https://example.com")
        assert result is None

    @pytest.mark.asyncio()
    async def test_successful_extraction(self) -> None:
        mock_cls, _ = _make_crawler_mock(
            success=True,
            markdown="# Page Title\n\nSome extracted content here.",
            title="Page Title",
        )
        fake_mod = _make_crawl4ai_module(mock_cls)

        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await crawl4ai_extract("https://example.com")

        assert result is not None
        assert result["success"] is True
        assert "extracted content" in result["content"]
        assert result["title"] == "Page Title"

    @pytest.mark.asyncio()
    async def test_returns_none_on_failed_result(self) -> None:
        mock_cls, _ = _make_crawler_mock(
            success=False,
            error_message="Page load timeout",
        )
        fake_mod = _make_crawl4ai_module(mock_cls)

        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await crawl4ai_extract("https://example.com")

        assert result is None

    @pytest.mark.asyncio()
    async def test_returns_none_on_empty_content(self) -> None:
        mock_cls, _ = _make_crawler_mock(success=True, markdown="   ")
        fake_mod = _make_crawl4ai_module(mock_cls)

        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await crawl4ai_extract("https://example.com")

        assert result is None

    @pytest.mark.asyncio()
    async def test_returns_none_on_exception(self) -> None:
        mock_cls, _ = _make_crawler_mock(side_effect=RuntimeError("Browser crash"))
        fake_mod = _make_crawl4ai_module(mock_cls)

        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await crawl4ai_extract("https://example.com")

        assert result is None

    @pytest.mark.asyncio()
    async def test_handles_missing_title_attribute(self) -> None:
        _mock_cls, _ = _make_crawler_mock(success=True, markdown="Some content.")
        # Remove the title attribute to test getattr fallback
        mock_result = MagicMock(spec=["success", "markdown"])
        mock_result.success = True
        mock_result.markdown = "Some content."

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_cls_custom = MagicMock(return_value=mock_instance)
        fake_mod = _make_crawl4ai_module(mock_cls_custom)

        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await crawl4ai_extract("https://example.com")

        assert result is not None
        assert result["title"] == ""

    @pytest.mark.asyncio()
    async def test_passes_timeout_to_crawler(self) -> None:
        mock_cls, mock_instance = _make_crawler_mock(
            success=True, markdown="Content."
        )
        fake_mod = _make_crawl4ai_module(mock_cls)

        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            await crawl4ai_extract("https://example.com", timeout=60_000)

        mock_instance.arun.assert_called_once_with(
            url="https://example.com",
            timeout=60_000,
        )

    @pytest.mark.asyncio()
    async def test_returns_none_on_none_markdown(self) -> None:
        _mock_cls, _ = _make_crawler_mock(success=True, markdown="")
        # Override markdown to be None
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = None
        mock_result.title = ""

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_cls_none = MagicMock(return_value=mock_instance)
        fake_mod = _make_crawl4ai_module(mock_cls_none)

        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await crawl4ai_extract("https://example.com")

        assert result is None

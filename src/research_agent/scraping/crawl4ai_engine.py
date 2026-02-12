"""Crawl4AI fallback engine for JS-heavy sites.

Provides async content extraction using Crawl4AI's headless browser when
Trafilatura extraction yields low quality scores (e.g., JS-rendered SPAs).

Crawl4AI is an optional dependency. Install with:
``pip install research-agent[js]``
"""

from __future__ import annotations

from typing import Any

import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_DEFAULT_TIMEOUT = 30_000  # milliseconds


async def crawl4ai_extract(
    url: str,
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict[str, Any] | None:
    """Extract content from a URL using Crawl4AI's headless browser.

    Launches a headless Chromium browser, navigates to the URL, waits
    for JS rendering, then extracts the page content.

    Args:
        url: The URL to crawl.
        timeout: Page load timeout in milliseconds.

    Returns:
        A dict with ``content``, ``title``, and ``success`` keys,
        or ``None`` if extraction failed.

    Raises:
        ImportError: If crawl4ai is not installed (caught and logged).
    """
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        logger.warning(
            "crawl4ai_not_installed",
            hint="Install with: pip install research-agent[js]",
        )
        return None

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=url,
                timeout=timeout,
            )

        if not result.success:
            logger.warning(
                "crawl4ai_extraction_failed",
                url=url,
                error=getattr(result, "error_message", "unknown"),
            )
            return None

        content = result.markdown or ""
        if not content.strip():
            logger.debug("crawl4ai_empty_content", url=url)
            return None

        title = getattr(result, "title", "") or ""

        logger.info(
            "crawl4ai_extract_ok",
            url=url,
            content_length=len(content),
            title=title[:80] if title else "",
        )

        return {
            "content": content,
            "title": title,
            "success": True,
        }

    except Exception as exc:
        logger.warning("crawl4ai_error", url=url, error=str(exc))
        return None

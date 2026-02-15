"""RSS/Atom feed monitoring with incremental polling support."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

import httpx

from research_agent.intelligence.models import FeedEntry

if TYPE_CHECKING:
    from pathlib import Path


class RSSMonitor:
    """Monitor RSS feeds with ETag/If-Modified-Since state."""

    def __init__(self, state_path: Path, client: httpx.Client | None = None) -> None:
        self._state_path = state_path
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = client or httpx.Client(timeout=10.0)

    def close(self) -> None:
        self._client.close()

    def import_opml(self, opml_path: Path) -> list[str]:
        """Load feed URLs from an OPML document."""
        root = ET.fromstring(opml_path.read_text(encoding="utf-8"))
        urls: list[str] = []
        for node in root.findall(".//outline"):
            url = node.attrib.get("xmlUrl")
            if url:
                urls.append(url)
        return sorted(set(urls))

    def poll(
        self,
        feeds: list[str],
        existing_urls: set[str] | None = None,
    ) -> list[FeedEntry]:
        """Poll feeds incrementally and return deduplicated entries."""
        state = self._load_state()
        existing_urls = existing_urls or set()
        entries: list[FeedEntry] = []

        for feed_url in feeds:
            headers: dict[str, str] = {}
            cached = state.get(feed_url, {})
            if isinstance(cached, dict):
                etag = cached.get("etag")
                modified = cached.get("last_modified")
                if isinstance(etag, str):
                    headers["If-None-Match"] = etag
                if isinstance(modified, str):
                    headers["If-Modified-Since"] = modified

            response = self._client.get(feed_url, headers=headers)
            if response.status_code == 304:
                continue
            response.raise_for_status()

            state[feed_url] = {
                "etag": response.headers.get("etag", ""),
                "last_modified": response.headers.get("last-modified", ""),
            }

            parsed = self._parse_feed(response.text)
            for entry in parsed:
                if entry.link in existing_urls:
                    continue
                existing_urls.add(entry.link)
                entries.append(entry)

        self._save_state(state)
        return entries

    def _parse_feed(self, xml_text: str) -> list[FeedEntry]:
        root = ET.fromstring(xml_text)
        entries: list[FeedEntry] = []

        channel = root.find("channel")
        if channel is not None:
            feed_title = channel.findtext("title", default="")
            for item in channel.findall("item"):
                entry = self._entry_from_item(feed_title, item)
                entries.append(entry)
            return entries

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        feed_title = root.findtext("atom:title", default="", namespaces=ns)
        for item in root.findall("atom:entry", ns):
            entries.append(self._entry_from_atom(feed_title, item, ns))
        return entries

    def _entry_from_item(self, feed_title: str, item: ET.Element) -> FeedEntry:
        link = item.findtext("link", default="")
        return FeedEntry(
            feed_title=feed_title,
            entry_id=item.findtext("guid", default=link),
            title=item.findtext("title", default=""),
            link=link,
            published=item.findtext("pubDate", default=""),
            summary=item.findtext("description", default=""),
            full_text=item.findtext("content:encoded", default="")
            or item.findtext("description", default=""),
        )

    def _entry_from_atom(
        self, feed_title: str, item: ET.Element, ns: dict[str, str]
    ) -> FeedEntry:
        link_node = item.find("atom:link", ns)
        link = link_node.attrib.get("href", "") if link_node is not None else ""
        summary = item.findtext("atom:summary", default="", namespaces=ns)
        content = item.findtext("atom:content", default=summary, namespaces=ns)

        return FeedEntry(
            feed_title=feed_title,
            entry_id=item.findtext("atom:id", default=link, namespaces=ns),
            title=item.findtext("atom:title", default="", namespaces=ns),
            link=link,
            published=item.findtext("atom:updated", default="", namespaces=ns),
            summary=summary,
            full_text=content,
        )

    def _load_state(self) -> dict[str, dict[str, Any]]:
        if not self._state_path.exists():
            return {}
        payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
        return {}

    def _save_state(self, payload: dict[str, dict[str, Any]]) -> None:
        self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

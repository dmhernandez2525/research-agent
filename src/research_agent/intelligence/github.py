"""GitHub repository analysis for implementation intelligence."""

from __future__ import annotations

import base64
import re

import httpx

from research_agent.intelligence.models import GitHubRepositoryInsight

_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)


class GitHubRepositoryAnalyzer:
    """Search and summarize GitHub repositories."""

    def __init__(
        self, token: str | None = None, client: httpx.Client | None = None
    ) -> None:
        self._client = client or httpx.Client(timeout=10.0)
        self._token = token

    def close(self) -> None:
        self._client.close()

    def search_repositories(
        self,
        query: str,
        project_dependencies: list[str],
        limit: int = 5,
    ) -> list[GitHubRepositoryInsight]:
        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        search_response = self._client.get(
            "https://api.github.com/search/repositories",
            headers=headers,
            params={"q": query, "sort": "stars", "order": "desc", "per_page": limit},
        )
        search_response.raise_for_status()
        self._ensure_rate_limit(search_response)

        payload = search_response.json()
        items = payload.get("items", [])
        if not isinstance(items, list):
            return []

        insights: list[GitHubRepositoryInsight] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            full_name = str(item.get("full_name", ""))
            readme = self._fetch_readme(full_name, headers=headers)
            snippets = self.extract_code_snippets(readme)

            dependency_match = self.match_dependencies(readme, project_dependencies)
            insights.append(
                GitHubRepositoryInsight(
                    full_name=full_name,
                    description=str(item.get("description", "")),
                    stars=int(item.get("stargazers_count", 0) or 0),
                    language=str(item.get("language", "")),
                    license_name=str((item.get("license") or {}).get("spdx_id", "")),
                    last_commit_date=str(item.get("pushed_at", "")),
                    readme_excerpt=readme[:600],
                    shared_dependencies=dependency_match,
                    code_snippets=snippets,
                )
            )

        return insights

    def _fetch_readme(self, full_name: str, headers: dict[str, str]) -> str:
        if not full_name:
            return ""

        response = self._client.get(
            f"https://api.github.com/repos/{full_name}/readme",
            headers=headers,
        )
        if response.status_code == 404:
            return ""
        response.raise_for_status()
        self._ensure_rate_limit(response)

        payload = response.json()
        content = payload.get("content", "")
        if not isinstance(content, str):
            return ""

        decoded = base64.b64decode(content).decode("utf-8", errors="ignore")
        return decoded

    def match_dependencies(
        self, readme: str, project_dependencies: list[str]
    ) -> list[str]:
        lower = readme.lower()
        matches = [dep for dep in project_dependencies if dep.lower() in lower]
        return sorted(set(matches))

    def extract_code_snippets(self, readme: str, limit: int = 4) -> list[str]:
        snippets = [match.strip() for match in _CODE_BLOCK_RE.findall(readme)]
        snippets = [snippet for snippet in snippets if snippet and len(snippet) <= 1200]
        return snippets[:limit]

    def _ensure_rate_limit(self, response: httpx.Response) -> None:
        remaining = response.headers.get("x-ratelimit-remaining")
        if remaining is None:
            return
        if int(remaining) <= 0:
            raise RuntimeError("GitHub rate limit exceeded")

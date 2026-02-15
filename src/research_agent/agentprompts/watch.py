"""File-watch helpers for AgentPrompts RESEARCH_PROMPT triggers."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from research_agent.agentprompts.bridge import run_for_project

if TYPE_CHECKING:
    from research_agent.agentprompts.bridge import ForProjectResult
    from research_agent.agentprompts.registry import ProjectRegistry


class PromptWatcher:
    """Poll project prompts and trigger research on change."""

    def __init__(
        self,
        projects_dir: Path,
        registry: ProjectRegistry,
        debounce_seconds: float = 2.0,
        poll_interval: float = 1.0,
        notify: bool = True,
    ) -> None:
        self._projects_dir = projects_dir
        self._registry = registry
        self._debounce_seconds = debounce_seconds
        self._poll_interval = poll_interval
        self._notify = notify
        self._seen_mtime: dict[Path, float] = {}
        self._last_triggered: dict[Path, float] = {}

    def run_once(self) -> list[ForProjectResult]:
        """Process changed prompts exactly once."""
        results: list[ForProjectResult] = []
        now = time.monotonic()

        for prompt_path in self._prompt_files():
            mtime = prompt_path.stat().st_mtime
            prior_mtime = self._seen_mtime.get(prompt_path)
            self._seen_mtime[prompt_path] = mtime

            changed = prior_mtime is None or mtime > prior_mtime
            if not changed:
                continue

            last = self._last_triggered.get(prompt_path, 0.0)
            if now - last < self._debounce_seconds:
                continue
            self._last_triggered[prompt_path] = now

            project_name = prompt_path.parent.name
            result = run_for_project(
                project_name=project_name,
                projects_dir=self._projects_dir,
                registry=self._registry,
            )
            results.append(result)
            self._notify_completion(result)

        return results

    def run_forever(self) -> None:
        """Continuously poll prompt files until interrupted."""
        while True:
            self.run_once()
            time.sleep(self._poll_interval)

    def _prompt_files(self) -> list[Path]:
        if not self._projects_dir.exists():
            return []
        paths = [
            path
            for path in self._projects_dir.glob("*/RESEARCH_PROMPT.md")
            if path.is_file()
        ]
        return sorted(paths)

    def _notify_completion(self, result: ForProjectResult) -> None:
        if not self._notify or not result.quality_gate_passed:
            return
        if not sys_platform_is_macos():
            return

        message = (
            f"Research completed for {result.project_name}. "
            f"Output: {result.output_path.name}"
        )
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{message}" with title "research-agent"',
            ],
            check=False,
            capture_output=True,
            text=True,
        )


def sys_platform_is_macos() -> bool:
    """Return true on macOS."""
    return Path("/System/Library/CoreServices/Finder.app").exists()

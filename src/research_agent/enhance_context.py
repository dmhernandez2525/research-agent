"""Project context loading for enhancement mode."""

from __future__ import annotations

import fnmatch
import json
import tomllib
from pathlib import Path

from research_agent.enhance_models import FileSummary, ProjectContext, ProjectDependency

_SKIPPED_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "__pycache__",
}

_RELEVANT_FILENAMES = {
    "README.md",
    "README",
    "README.rst",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
}

_RELEVANT_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java"}

_LANGUAGE_MAP = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}

_FRAMEWORK_HINTS = {
    "react": "react",
    "next": "nextjs",
    "nextjs": "nextjs",
    "vite": "vite",
    "fastapi": "fastapi",
    "django": "django",
    "flask": "flask",
    "express": "express",
    "nestjs": "nestjs",
    "langgraph": "langgraph",
}


def load_gitignore_patterns(project_path: Path) -> list[str]:
    """Load non-comment patterns from .gitignore if present."""
    gitignore = project_path / ".gitignore"
    if not gitignore.exists():
        return []
    patterns: list[str] = []
    for line in gitignore.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    return patterns


def _is_ignored(rel_path: str, patterns: list[str]) -> bool:
    normalized = rel_path.replace("\\", "/")
    for pattern in patterns:
        p = pattern.strip()
        if p.endswith("/"):
            if normalized.startswith(p):
                return True
            continue
        if fnmatch.fnmatch(normalized, p):
            return True
        if fnmatch.fnmatch(Path(normalized).name, p):
            return True
    return False


def _iter_relevant_files(project_path: Path, ignore_patterns: list[str]) -> list[Path]:
    relevant: list[Path] = []
    for path in sorted(project_path.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(project_path)
        rel_path_str = rel_path.as_posix()

        if any(part in _SKIPPED_DIRS for part in rel_path.parts):
            continue
        if _is_ignored(rel_path_str, ignore_patterns):
            continue

        if path.name in _RELEVANT_FILENAMES or path.suffix in _RELEVANT_EXTENSIONS:
            relevant.append(path)
    return relevant


def _summarize_file(path: Path, max_chars: int = 4000) -> FileSummary:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    truncated = len(raw) > max_chars
    if truncated:
        head = raw[: max_chars // 2]
        tail = raw[-(max_chars // 2) :]
        summary = f"[truncated from {len(raw)} chars]\n{head}\n...\n{tail}"
    else:
        summary = raw
    return FileSummary(
        path=str(path),
        size_bytes=path.stat().st_size,
        summary=summary,
        truncated=truncated,
    )


def _extract_description(key_files: list[FileSummary]) -> str:
    for item in key_files:
        name = Path(item.path).name.lower()
        if not name.startswith("readme"):
            continue
        for line in item.summary.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            return stripped[:300]
    return ""


def _parse_dependencies(project_path: Path) -> list[ProjectDependency]:
    deps: dict[str, ProjectDependency] = {}

    package_json = project_path / "package.json"
    if package_json.exists():
        data = json.loads(package_json.read_text(encoding="utf-8"))
        dep_maps = [data.get("dependencies", {}), data.get("devDependencies", {})]
        for dep_map in dep_maps:
            if not isinstance(dep_map, dict):
                continue
            for name, version in dep_map.items():
                deps[name] = ProjectDependency(
                    name=name,
                    version=str(version),
                    source="package.json",
                )

    pyproject = project_path / "pyproject.toml"
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project_data = data.get("project", {})
        if isinstance(project_data, dict):
            for raw in project_data.get("dependencies", []):
                if not isinstance(raw, str):
                    continue
                name = raw.split("[")[0].split("<")[0].split(">")[0].split("=")[0]
                deps[name.strip()] = ProjectDependency(
                    name=name.strip(),
                    version=raw,
                    source="pyproject.toml",
                )

    requirements = project_path / "requirements.txt"
    if requirements.exists():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            name = stripped.split("==")[0].split(">=")[0].split("<=")[0]
            deps[name] = ProjectDependency(
                name=name,
                version=stripped,
                source="requirements.txt",
            )

    return sorted(deps.values(), key=lambda item: item.name.lower())


def _detect_languages(files: list[Path]) -> list[str]:
    counts: dict[str, int] = {}
    for path in files:
        language = _LANGUAGE_MAP.get(path.suffix.lower())
        if language is None:
            continue
        counts[language] = counts.get(language, 0) + 1
    return [name for name, _count in sorted(counts.items(), key=lambda item: -item[1])]


def _detect_framework(dependencies: list[ProjectDependency]) -> str:
    names = {dep.name.lower() for dep in dependencies}
    for hint, framework in _FRAMEWORK_HINTS.items():
        if hint in names:
            return framework
    return "unknown"


def build_project_context(
    project_path: Path,
    max_files: int = 25,
    max_chars_per_file: int = 4000,
) -> ProjectContext:
    """Scan project files and build a structured enhancement context."""
    root = project_path.expanduser().resolve()
    ignore_patterns = load_gitignore_patterns(root)
    candidate_files = _iter_relevant_files(root, ignore_patterns)

    selected_files: list[Path] = []
    priority_names = ["README.md", "README", "pyproject.toml", "package.json"]
    for name in priority_names:
        for path in candidate_files:
            if path.name == name and path not in selected_files:
                selected_files.append(path)

    for path in candidate_files:
        if path not in selected_files:
            selected_files.append(path)
        if len(selected_files) >= max_files:
            break

    key_files = [
        _summarize_file(path, max_chars=max_chars_per_file) for path in selected_files
    ]
    dependencies = _parse_dependencies(root)
    languages = _detect_languages(selected_files)

    return ProjectContext(
        project_path=str(root),
        project_name=root.name,
        description=_extract_description(key_files),
        languages=languages,
        framework=_detect_framework(dependencies),
        dependencies=dependencies,
        key_files=key_files,
    )

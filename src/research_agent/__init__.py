"""research-agent: Crash-resilient deep research agent."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("research-agent")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]

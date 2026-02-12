"""Prompt versioning via content hashing for cache invalidation.

Computes SHA-256 hashes of prompt YAML files so that LLM cache entries
are automatically invalidated when prompt content changes. Each prompt
file gets a stable hash that can be included in the LLM cache key.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# In-memory cache of computed hashes
_hash_cache: dict[str, str] = {}


def prompt_hash(prompt_name: str) -> str:
    """Compute a SHA-256 hash of a prompt YAML file's content.

    Results are cached in memory so repeated calls for the same prompt
    return instantly without re-reading the file.

    Args:
        prompt_name: The prompt name (without extension), e.g. "summarizer".

    Returns:
        A hex SHA-256 digest of the file content, or an empty string
        if the file does not exist.
    """
    if prompt_name in _hash_cache:
        return _hash_cache[prompt_name]

    path = _PROMPTS_DIR / f"{prompt_name}.yaml"
    if not path.exists():
        logger.warning("prompt_file_not_found", prompt_name=prompt_name, path=str(path))
        _hash_cache[prompt_name] = ""
        return ""

    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    _hash_cache[prompt_name] = digest

    logger.debug(
        "prompt_hash_computed",
        prompt_name=prompt_name,
        hash_prefix=digest[:12],
    )
    return digest


def prompt_hash_combined(*prompt_names: str) -> str:
    """Compute a combined hash from multiple prompt files.

    Useful when a node uses templates from multiple prompt files and
    the cache entry should invalidate if any of them change.

    Args:
        *prompt_names: One or more prompt names (without extension).

    Returns:
        A hex SHA-256 digest combining all individual hashes.
    """
    individual_hashes = [prompt_hash(name) for name in sorted(prompt_names)]
    combined = "|".join(individual_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()


def clear_hash_cache() -> None:
    """Clear the in-memory hash cache.

    Useful for testing or when prompt files have been modified at runtime.
    """
    _hash_cache.clear()
    logger.debug("prompt_hash_cache_cleared")


def known_hashes() -> dict[str, str]:
    """Return a copy of all currently cached prompt hashes.

    Returns:
        Dict mapping prompt_name to its SHA-256 hex digest.
    """
    return dict(_hash_cache)

"""Cross-session knowledge synthesis and consolidation logic."""

from __future__ import annotations

from datetime import UTC, datetime

from research_agent.knowledge.models import KnowledgeFinding


def consolidate_findings(findings: list[KnowledgeFinding]) -> list[KnowledgeFinding]:
    """Merge redundant findings by topic + normalized statement."""
    grouped: dict[tuple[str, str], KnowledgeFinding] = {}

    for finding in findings:
        key = (finding.topic.lower(), normalize_statement(finding.statement))
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = finding
            continue

        combined_sources = sorted(set(existing.sources + finding.sources))
        grouped[key] = KnowledgeFinding(
            id=existing.id,
            topic=existing.topic,
            statement=existing.statement,
            sources=combined_sources,
            confidence=max(existing.confidence, finding.confidence),
            cluster=existing.cluster,
            updated_at=max(existing.updated_at, finding.updated_at),
        )

    return list(grouped.values())


def detect_conflicts(
    findings: list[KnowledgeFinding],
) -> list[tuple[KnowledgeFinding, KnowledgeFinding]]:
    """Flag contradictory findings for human review."""
    conflicts: list[tuple[KnowledgeFinding, KnowledgeFinding]] = []

    for i, left in enumerate(findings):
        for right in findings[i + 1 :]:
            if left.topic.lower() != right.topic.lower():
                continue
            if is_conflicting(left.statement, right.statement):
                conflicts.append((left, right))

    return conflicts


def score_confidence(finding: KnowledgeFinding) -> float:
    """Compute confidence from source count and recency."""
    source_score = min(len(finding.sources) / 5, 1.0)

    updated = datetime.fromisoformat(finding.updated_at)
    age_days = (datetime.now(tz=UTC) - updated).days
    recency_score = max(0.0, 1 - (age_days / 120))

    return round(0.65 * source_score + 0.35 * recency_score, 3)


def summarize_by_cluster(findings: list[KnowledgeFinding]) -> dict[str, str]:
    """Generate short summaries grouped by cluster/topic."""
    clusters: dict[str, list[KnowledgeFinding]] = {}
    for finding in findings:
        clusters.setdefault(finding.cluster, []).append(finding)

    summaries: dict[str, str] = {}
    for cluster, items in clusters.items():
        ranked = sorted(items, key=lambda item: item.confidence, reverse=True)
        top = ranked[:3]
        bullets = [f"- {item.topic}: {item.statement}" for item in top]
        summaries[cluster] = "\n".join(bullets)
    return summaries


def normalize_statement(statement: str) -> str:
    """Normalize statement text for dedupe comparison."""
    cleaned = " ".join(statement.lower().split())
    return cleaned.strip(". ")


def is_conflicting(left: str, right: str) -> bool:
    """Heuristic conflict detection between statements."""
    pairs = [
        ("recommended", "not recommended"),
        ("supports", "does not support"),
        ("safe", "unsafe"),
        ("stable", "unstable"),
    ]
    left_lower = left.lower()
    right_lower = right.lower()

    for positive, negative in pairs:
        if positive in left_lower and negative in right_lower:
            return True
        if positive in right_lower and negative in left_lower:
            return True

    return False

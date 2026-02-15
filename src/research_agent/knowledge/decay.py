"""Knowledge decay, refresh scheduling, and change tracking."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from research_agent.knowledge.models import KnowledgeFinding, KnowledgeRefreshRecord


def apply_confidence_decay(
    findings: list[KnowledgeFinding],
    decay_per_day: float = 0.003,
) -> list[KnowledgeFinding]:
    """Reduce confidence scores for aging findings."""
    now = datetime.now(tz=UTC)
    updated_findings: list[KnowledgeFinding] = []

    for finding in findings:
        updated_at = datetime.fromisoformat(finding.updated_at)
        age_days = max((now - updated_at).days, 0)
        decayed = max(0.0, finding.confidence - decay_per_day * age_days)
        updated_findings.append(
            finding.model_copy(update={"confidence": round(decayed, 3)})
        )

    return updated_findings


def refresh_due(
    finding: KnowledgeFinding,
    topic_refresh_days: dict[str, int],
    default_days: int = 30,
) -> bool:
    """Determine if a finding is due for refresh by schedule."""
    schedule_days = topic_refresh_days.get(finding.topic.lower(), default_days)
    updated_at = datetime.fromisoformat(finding.updated_at)
    return datetime.now(tz=UTC) - updated_at > timedelta(days=schedule_days)


def should_trigger_research(
    finding: KnowledgeFinding,
    threshold: float,
    topic_refresh_days: dict[str, int],
) -> bool:
    """Trigger re-research when low confidence or refresh schedule elapsed."""
    return finding.confidence < threshold or refresh_due(
        finding,
        topic_refresh_days=topic_refresh_days,
    )


def create_refresh_record(
    topic: str,
    old_confidence: float,
    new_confidence: float,
    old_statement: str,
    new_statement: str,
) -> KnowledgeRefreshRecord:
    """Build refresh-history record with concise change summary."""
    if old_statement.strip() == new_statement.strip():
        summary = "No statement changes; confidence updated."
    else:
        summary = "Statement updated after refresh review."

    return KnowledgeRefreshRecord(
        topic=topic,
        old_confidence=old_confidence,
        new_confidence=new_confidence,
        change_summary=summary,
    )

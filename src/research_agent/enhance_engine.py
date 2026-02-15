"""Enhancement opportunity analysis and incremental knowledge management."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from research_agent.enhance_models import (
    DeltaSummary,
    EnhancementOpportunity,
    KnowledgeEntry,
    OpportunityCategory,
    ProjectContext,
)

if TYPE_CHECKING:
    from research_agent.enhance_store import KnowledgeStore


def _has_tests(context: ProjectContext) -> bool:
    for file_summary in context.key_files:
        path = Path(file_summary.path)
        if "test" in path.name.lower() or "tests" in path.parts:
            return True
    return False


def _is_documented(context: ProjectContext) -> bool:
    return bool(context.description.strip())


def _dependency_health(context: ProjectContext) -> tuple[int, str]:
    if not context.dependencies:
        return 2, "No explicit dependencies were detected."

    risky = [
        dep
        for dep in context.dependencies
        if "*" in dep.version
        or dep.version.strip() == ""
        or dep.version.startswith("^")
    ]
    if risky:
        return 4, f"{len(risky)} dependencies use loose or missing version constraints."
    return 3, "Dependencies are versioned but should be reviewed for upgrades."


def _base_opportunities(context: ProjectContext) -> list[EnhancementOpportunity]:
    dep_impact, dep_rationale = _dependency_health(context)

    opportunities = [
        EnhancementOpportunity(
            title="Harden dependency upgrade strategy",
            category=OpportunityCategory.DEPENDENCIES,
            impact_score=dep_impact,
            effort_score=2,
            rationale=dep_rationale,
            suggested_query=(
                f"{context.framework} dependency upgrade checklist and breaking changes"
            ),
        ),
        EnhancementOpportunity(
            title="Establish architecture decision records",
            category=OpportunityCategory.ARCHITECTURE,
            impact_score=4,
            effort_score=3,
            rationale="Documenting architecture decisions reduces regressions during scale.",
            suggested_query=f"{context.project_name} architecture ADR template best practices",
        ),
        EnhancementOpportunity(
            title="Security controls and secrets handling review",
            category=OpportunityCategory.SECURITY,
            impact_score=5,
            effort_score=3,
            rationale="Security review lowers risk around auth, secrets, and input handling.",
            suggested_query=f"{context.framework} security hardening checklist 2026",
        ),
        EnhancementOpportunity(
            title="Performance profiling and hotspot optimization",
            category=OpportunityCategory.PERFORMANCE,
            impact_score=4,
            effort_score=3,
            rationale="Profiling reveals bottlenecks that affect user-facing latency.",
            suggested_query=f"{context.framework} performance profiling techniques",
        ),
    ]

    if not _has_tests(context):
        opportunities.append(
            EnhancementOpportunity(
                title="Create automated regression test suite",
                category=OpportunityCategory.TESTING,
                impact_score=5,
                effort_score=4,
                rationale="No test footprint detected in key files.",
                suggested_query=f"{context.framework} testing pyramid practical guide",
            )
        )

    if not _is_documented(context):
        opportunities.append(
            EnhancementOpportunity(
                title="Improve project documentation quality",
                category=OpportunityCategory.DOCUMENTATION,
                impact_score=3,
                effort_score=1,
                rationale="README description is missing or too sparse.",
                suggested_query=f"{context.framework} README template for maintainable projects",
            )
        )

    return opportunities


def identify_opportunities(
    context: ProjectContext,
    focus_areas: set[OpportunityCategory] | None = None,
) -> list[EnhancementOpportunity]:
    """Identify and rank enhancement opportunities from project context."""
    opportunities = _base_opportunities(context)

    if focus_areas:
        opportunities = [op for op in opportunities if op.category in focus_areas]

    opportunities.sort(key=lambda op: (-op.impact_score, op.effort_score, op.title))
    return opportunities


def plan_incremental_research(
    project_id: str,
    opportunities: list[EnhancementOpportunity],
    store: KnowledgeStore,
    stale_days: int,
    force_refresh: bool,
) -> tuple[list[EnhancementOpportunity], DeltaSummary, list[KnowledgeEntry]]:
    """Filter opportunities by staleness and collect cross-project shared findings."""
    existing = store.get_project_entries(project_id)
    now = datetime.now(tz=UTC)

    refresh_targets: list[EnhancementOpportunity] = []
    skipped_recent: list[str] = []
    shared_entries: list[KnowledgeEntry] = []

    for opp in opportunities:
        entry = existing.get(opp.title)
        if entry and not force_refresh:
            updated_at = datetime.fromisoformat(entry.updated_at)
            if now - updated_at < timedelta(days=stale_days):
                skipped_recent.append(opp.title)
                continue

        refresh_targets.append(opp)
        shared_entries.extend(store.cross_project_matches(project_id, opp.category))

    shared_topics = sorted({entry.topic for entry in shared_entries})
    delta = DeltaSummary(
        new_topics=[opp.title for opp in refresh_targets],
        skipped_recent_topics=skipped_recent,
        shared_topics=shared_topics,
    )
    return refresh_targets, delta, shared_entries


def _snippet_for_category(category: OpportunityCategory) -> tuple[str, str]:
    snippets: dict[OpportunityCategory, tuple[str, str]] = {
        OpportunityCategory.TESTING: (
            "def calc(x, y):\n    return x + y",
            "def calc(x: int, y: int) -> int:\n    return x + y\n\n\ndef test_calc() -> None:\n    assert calc(2, 3) == 5",
        ),
        OpportunityCategory.SECURITY: (
            'token = os.environ["TOKEN"]',
            "from research_agent.config import Settings\nsettings = Settings()\ntoken = settings.llm.model",
        ),
    }
    return snippets.get(
        category,
        (
            "# current implementation",
            "# improved implementation with explicit typing and validation",
        ),
    )


def generate_enhancement_report(
    context: ProjectContext,
    opportunities: list[EnhancementOpportunity],
    delta: DeltaSummary,
    shared_entries: list[KnowledgeEntry],
) -> str:
    """Build a COMPILED_RESEARCH-compatible enhancement report."""
    lines: list[str] = [
        "# COMPILED_RESEARCH",
        "",
        "## Current State",
        f"- Project: `{context.project_name}`",
        f"- Framework: `{context.framework}`",
        f"- Languages: {', '.join(context.languages) or 'unknown'}",
        f"- Dependencies detected: {len(context.dependencies)}",
        f"- Description: {context.description or 'No README description found.'}",
        "",
        "## Recommendations",
    ]

    for idx, opp in enumerate(opportunities, start=1):
        lines.append(
            f"{idx}. **{opp.title}** ({opp.category.value}) - impact {opp.impact_score}/5, effort {opp.effort_score}/5"
        )
        lines.append(f"   - Why: {opp.rationale}")
        lines.append(f"   - Research query: `{opp.suggested_query}`")

    lines.extend(["", "## Implementation Steps"])
    for idx, opp in enumerate(opportunities, start=1):
        lines.append(f"{idx}. Scope and plan changes for **{opp.title}**.")
        lines.append(
            f"{idx + 1}. Prototype and benchmark impact for `{opp.category.value}` improvements."
        )

    lines.extend(
        [
            "",
            "## Priority Matrix",
            "| Opportunity | Impact | Effort |",
            "|---|---:|---:|",
        ]
    )
    for opp in opportunities:
        lines.append(f"| {opp.title} | {opp.impact_score} | {opp.effort_score} |")

    lines.extend(
        [
            "",
            "## Delta Findings",
            f"- New topics: {', '.join(delta.new_topics) or 'none'}",
            f"- Skipped as fresh: {', '.join(delta.skipped_recent_topics) or 'none'}",
            f"- Shared cross-project topics: {', '.join(delta.shared_topics) or 'none'}",
            "",
            "## Code Snippet Suggestions",
        ]
    )

    for opp in opportunities[:3]:
        before, after = _snippet_for_category(opp.category)
        lines.append(f"### {opp.title}")
        lines.append("```python")
        lines.append(before)
        lines.append("```")
        lines.append("```python")
        lines.append(after)
        lines.append("```")

    lines.extend(["", "## Dependency Upgrade Analysis"])
    for dep in context.dependencies[:20]:
        warning = "Potential breaking changes - verify release notes."
        lines.append(f"- `{dep.name}` ({dep.version}) - {warning}")

    if shared_entries:
        lines.extend(["", "## Shared Knowledge from Other Projects"])
        for entry in shared_entries[:10]:
            lines.append(
                f"- **{entry.topic}** ({entry.category.value}): {entry.finding}"
            )

    return "\n".join(lines).strip() + "\n"


def persist_findings(
    project_id: str,
    opportunities: list[EnhancementOpportunity],
    store: KnowledgeStore,
) -> None:
    entries = [
        KnowledgeEntry(
            topic=op.title,
            category=op.category,
            finding=op.rationale,
            query=op.suggested_query,
        )
        for op in opportunities
    ]
    store.upsert_entries(project_id, entries)

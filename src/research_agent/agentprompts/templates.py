"""Research prompt templates for AgentPrompts workflows."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_BUILTIN_TEMPLATES: dict[str, str] = {
    "technology-evaluation": """# Topic
Evaluate {{PROJECT_NAME}} technology options for {{FOCUS_AREA}}.

## Constraints
- Prioritize maintainability and long-term support
- Prefer low operational complexity

## Output Requirements
- Decision matrix with tradeoffs
- Recommendation with rationale

## Existing Context
Primary language: {{LANGUAGE}}
""",
    "competitive-analysis": """# Topic
Competitive analysis for {{PROJECT_NAME}} in {{FOCUS_AREA}}.

## Constraints
- Compare at least 3 competitors
- Focus on differentiators and risks

## Output Requirements
- Competitor summary table
- Strategic positioning recommendations

## Existing Context
Primary language: {{LANGUAGE}}
""",
    "architecture-research": """# Topic
Architecture research for {{PROJECT_NAME}} focused on {{FOCUS_AREA}}.

## Constraints
- Align to production reliability and security
- Consider scaling to 10x expected traffic

## Output Requirements
- Reference architecture proposal
- Dependency and integration recommendations

## Existing Context
Primary language: {{LANGUAGE}}
""",
    "security-audit": """# Topic
Security audit research for {{PROJECT_NAME}}.

## Constraints
- Include OWASP-aligned controls
- Prioritize exploitability and blast radius

## Output Requirements
- Vulnerability categories with mitigations
- Security hardening checklist

## Existing Context
Primary language: {{LANGUAGE}}
Focus area: {{FOCUS_AREA}}
""",
    "dependency-review": """# Topic
Dependency review for {{PROJECT_NAME}}.

## Constraints
- Focus on outdated/high-risk dependencies
- Include migration effort estimates

## Output Requirements
- Upgrade plan and compatibility notes
- Recommended replacements for risky packages

## Existing Context
Primary language: {{LANGUAGE}}
Focus area: {{FOCUS_AREA}}
""",
}

_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def list_templates(custom_dir: Path | None = None) -> list[str]:
    """List built-in and custom template names."""
    names = set(_BUILTIN_TEMPLATES)
    if custom_dir and custom_dir.exists():
        for path in custom_dir.glob("*.md"):
            names.add(path.stem)
    return sorted(names)


def load_template(name: str, custom_dir: Path | None = None) -> str:
    """Load a template by name from built-ins or custom directory."""
    key = name.strip().lower()
    if custom_dir:
        custom_path = custom_dir / f"{key}.md"
        if custom_path.exists():
            return custom_path.read_text(encoding="utf-8")
    if key not in _BUILTIN_TEMPLATES:
        raise ValueError(f"Unknown template: {name}")
    return _BUILTIN_TEMPLATES[key]


def render_template(template_text: str, variables: dict[str, str]) -> str:
    """Render placeholders like {{PROJECT_NAME}} with provided variables."""

    def replace(match: re.Match[str]) -> str:
        placeholder = match.group(1)
        return variables.get(placeholder, "")

    return _PLACEHOLDER_RE.sub(replace, template_text)

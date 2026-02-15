"""AgentPrompts ecosystem bridge and automation helpers."""

from research_agent.agentprompts.bridge import ForProjectResult, run_for_project
from research_agent.agentprompts.registry import ProjectRegistry
from research_agent.agentprompts.watch import PromptWatcher

__all__ = ["ForProjectResult", "ProjectRegistry", "PromptWatcher", "run_for_project"]

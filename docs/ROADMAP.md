# Research Agent Roadmap

## Phase 1: MVP (Core Pipeline)

**Target:** 2 weeks
**Goal:** `research-agent "topic"` produces a report end-to-end

| Task | Status |
|------|--------|
| Project scaffolding with uv + pyproject.toml | Done |
| Configuration system (pydantic-settings + config.yaml) | Not Started |
| LangGraph StateGraph with 5 nodes (plan, search, scrape, summarize, synthesize) | Not Started |
| Tavily search integration | Not Started |
| Trafilatura scraping with quality scoring | Not Started |
| Basic model routing (Anthropic primary, OpenAI fallback) | Not Started |
| SQLite checkpointing via LangGraph | Not Started |
| Basic Rich CLI with progress bars | Not Started |
| Markdown report output | Not Started |
| Unit tests with VCR.py recording setup | Not Started |

**Milestone:** Running `research-agent "topic"` produces a complete Markdown report with cited sources from end-to-end pipeline execution.

---

## Phase 2: Resilience & Quality

**Target:** 2 weeks
**Goal:** Agent recovers from crash mid-research and stays within budget

| Task | Status |
|------|--------|
| Atomic file-based checkpoints (per-subtask JSON files) | Not Started |
| Observation masking for context management | Not Started |
| Graceful degradation state machine (4-tier) | Not Started |
| Budget tracking and cost caps | Not Started |
| Self-evaluation (LLM-as-judge) | Not Started |
| Prompt caching optimization (Anthropic + OpenAI) | Not Started |
| Content quality scoring (pre-LLM filtering) | Not Started |
| Structured logging with provenance chains (structlog) | Not Started |

**Milestone:** Agent recovers from a crash mid-research, resumes from the last checkpoint, and enforces cost caps throughout execution.

---

## Phase 3: Advanced Features

**Target:** 3 weeks
**Goal:** Handles complex multi-topic research with deduplication

| Task | Status |
|------|--------|
| ChromaDB vector store for deduplication | Not Started |
| nomic-embed-text-v1.5 local embeddings | Not Started |
| Serial section-by-section synthesis for long reports | Not Started |
| Crawl4AI for JavaScript-heavy sites | Not Started |
| MCP server integration for extensibility | Not Started |
| API key rotation for heavy workloads | Not Started |
| Human-in-the-loop plan review via LangGraph interrupt() | Not Started |
| Multi-format output (Markdown, PDF) | Not Started |

**Milestone:** Complex multi-topic research queries produce deduplicated, high-quality reports with vector-backed source management.

---

## Phase 4: Ecosystem Integration

**Target:** 2 weeks
**Goal:** Drop-in replacement for Claude.ai research step

| Task | Status |
|------|--------|
| Direct integration with RESEARCH_PROMPT.md format | Not Started |
| COMPILED_RESEARCH.md output matching BUILD_PROMPT.md expectations | Not Started |
| AgentPromptsManager "Run Research" button | Not Started |
| Enhancement research mode for existing projects | Not Started |
| Cross-session knowledge/memory (ChromaDB persistence) | Not Started |
| Web UI for real-time progress monitoring | Not Started |

**Milestone:** Research agent plugs directly into the Apps That Build Apps pipeline, replacing the manual Claude.ai research step with automated, reproducible output.

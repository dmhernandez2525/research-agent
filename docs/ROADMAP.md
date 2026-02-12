# Research Agent Roadmap

> **Research Foundation:** This roadmap was derived from a comprehensive survey of 16+ open-source research agents (GPT Researcher, DeerFlow, Open Deep Research, STORM, Perplexica, and others) across 5 deep-dive research sessions totaling ~40,000+ words. The session 4 gap analysis identified 20 specific gaps (G-01 through G-20) in existing implementations that this roadmap addresses.

## Phase 1: MVP (Core Pipeline)

**Goal:** `research-agent "topic"` produces a report end-to-end

| Task | Gap Ref | Status |
|------|---------|--------|
| Project scaffolding with uv + pyproject.toml | - | Done |
| Configuration system (pydantic-settings + config.yaml) | G-16 | Done |
| LangGraph StateGraph with 5 nodes (plan, search, scrape, summarize, synthesize) | G-01 | Done |
| Tavily search with ExpandSearch pattern (3 variations, 34.3% improvement) | G-03 | Done |
| Trafilatura scraping with quality scoring (F1=0.958) | G-04 | Done |
| Basic model routing (litellm with Anthropic primary, OpenAI fallback) | G-07 | Done |
| SQLite checkpointing via LangGraph SqliteSaver | G-05 | Done |
| Basic Rich CLI with progress bars and plan approval | G-13 | Done |
| Markdown report output with citation management | G-09 | Done |
| Unit tests with pytest-asyncio setup | G-15 | Done |

**Key research insights driving Phase 1:**
- Only 4 of 16+ surveyed agents have genuine crash resilience (G-05)
- ExpandSearch pattern: 34.3% improvement over single-query baselines (arXiv 2510.10009)
- Trafilatura: F1=0.958, precision 93.8%, recall 97.8% on ScrapingHub benchmark (SIGIR 2023)
- Tavily: 93.3% accuracy on SimpleQA benchmark, natural language queries

---

## Phase 2: Resilience & Quality

**Goal:** Agent recovers from crash mid-research and stays within budget

| Task | Gap Ref | Status |
|------|---------|--------|
| Atomic file-based checkpoints (temp+fsync+os.replace with SHA-256) | G-05, G-06 | Done |
| Token estimation for context management | G-08 | Done |
| Model routing with tenacity retry and fallback chains | G-07, G-10 | Done |
| Budget tracking with configurable cap and warning threshold | G-10 | Done |
| Self-evaluation: LLM-as-judge, 5 dimensions, scoring | G-11 | Done |
| Prompt caching (stability ordering) | G-17 | Done |
| Content quality scoring (reject below threshold, flag marginal) | G-04 | Done |
| Structured logging with provenance chains (structlog) | G-14 | Done |
| Scraping: paywall detection, freshness scoring, sanitization | G-04 | Done |
| Scraping: structured data extraction (JSON-LD, microdata, Open Graph) | G-04 | Done |
| Metrics collection and dashboard display | G-14 | Done |
| Human-in-the-loop plan review ($EDITOR and inline editing) | G-13 | Done |
| Event logging and progress tracking | G-14 | Done |

**Key research insights driving Phase 2:**
- 83.9% of context tokens come from tool observations (NeurIPS 2025, "The Complexity Trap")
- Prompt caching: 90% Anthropic discount, break-even after 2 calls; 50-90% OpenAI automatic
- Session cost reduction: $0.06-$0.15 optimized vs $2.43 naive (96-97% savings)
- Model routing: ~82% savings vs single-model Sonnet 4.5
- LLM-as-judge achieves ~80% agreement with human evaluators

---

## Phase 3: Advanced Features

**Goal:** Handles complex multi-topic research with deduplication

| Task | Gap Ref | Status |
|------|---------|--------|
| Centralized exception hierarchy | - | Done |
| LiteLLM migration (replace langchain for LLM calls) | G-07 | Done |
| Data model rename (Subtopic, ScrapedPage, SubtopicSummary) | - | Done |
| Coverage improvements (98% overall) | G-15 | Done |
| Plan node implementation (LLM-powered query decomposition) | G-01 | In Progress |
| ChromaDB vector store for deduplication (0.85 similarity, 0.95 exact) | G-12 | Not Started |
| nomic-embed-text-v1.5 local embeddings (768d, 8K context) | G-12 | Not Started |
| Serial section-by-section synthesis (bypasses ~2,000 word ceiling) | G-09 | Not Started |
| Crawl4AI for JavaScript-heavy sites | G-04 | Not Started |
| API key rotation for heavy workloads | G-18 | Not Started |
| Multi-format output (Markdown, PDF via pymupdf) | G-20 | Not Started |
| DiskCache for LLM deduplication | G-17 | Not Started |
| Prompt versioning for cache invalidation | G-17 | Not Started |
| Adaptive rate limiting | G-18 | Not Started |

---

## Phase 4: Ecosystem Integration

**Goal:** Drop-in replacement for Claude.ai research step

| Task | Gap Ref | Status |
|------|---------|--------|
| Direct integration with RESEARCH_PROMPT.md format | G-02 | Not Started |
| COMPILED_RESEARCH.md output matching BUILD_PROMPT.md expectations | G-02 | Not Started |
| Cross-session knowledge/memory (ChromaDB persistence) | G-12 | Not Started |

---

## Deferred to Phase 5

| Task | Gap Ref | Reason |
|------|---------|--------|
| MCP server integration for extensibility | G-19 | Full protocol server; better as standalone phase |
| AgentPromptsManager "Run Research" button | G-19 | Requires external project integration |
| Web UI for real-time progress monitoring | G-20 | Requires FastAPI + WebSocket + React frontend |
| Enhancement research mode for existing projects | - | Follows after RESEARCH_PROMPT integration |

---

## Gap Analysis Reference

The following gaps were identified during Session 4 of the research phase. Each gap is mapped to roadmap tasks above.

| Gap ID | Description | Phase | Severity |
|--------|-------------|-------|----------|
| G-01 | End-to-end graph wiring (5-node pipeline) | 1 | Critical |
| G-02 | Integration with agent-prompts ecosystem | 4 | High |
| G-03 | ExpandSearch multi-query pattern | 1 | High |
| G-04 | Content quality scoring and filtering | 1-3 | High |
| G-05 | Atomic checkpoint system with integrity verification | 1-2 | Critical |
| G-06 | Schema versioning for checkpoint evolution | 2 | Medium |
| G-07 | Three-tier model routing with fallback chains | 1-2 | High |
| G-08 | Observation masking for context management | 2 | High |
| G-09 | Citation management and synthesis architecture | 1-3 | High |
| G-10 | Budget tracking and graceful degradation | 2 | Critical |
| G-11 | Self-evaluation with LLM-as-judge | 2 | Medium |
| G-12 | Vector store deduplication | 3-4 | Medium |
| G-13 | CLI UX (plan approval, progress, Ctrl+C) | 1-3 | High |
| G-14 | Structured logging with provenance | 2 | Medium |
| G-15 | Test infrastructure (pytest, fixtures) | 1 | High |
| G-16 | 4-layer configuration resolution | 1 | High |
| G-17 | Prompt caching optimization | 2 | Medium |
| G-18 | API key rotation | 3 | Low |
| G-19 | MCP/extensibility integration | 3-4 | Low |
| G-20 | Multi-format output and web UI | 3-4 | Low |

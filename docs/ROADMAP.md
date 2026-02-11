# Research Agent Roadmap

> **Research Foundation:** This roadmap was derived from a comprehensive survey of 16+ open-source research agents (GPT Researcher, DeerFlow, Open Deep Research, STORM, Perplexica, and others) across 5 deep-dive research sessions totaling ~40,000+ words. The session 4 gap analysis identified 20 specific gaps (G-01 through G-20) in existing implementations that this roadmap addresses.

## Phase 1: MVP (Core Pipeline)

**Target:** 2 weeks
**Goal:** `research-agent "topic"` produces a report end-to-end

| Task | Gap Ref | Status |
|------|---------|--------|
| Project scaffolding with uv + pyproject.toml | - | Done |
| Configuration system (pydantic-settings + config.yaml) | G-16 | Not Started |
| LangGraph StateGraph with 5 nodes (plan, search, scrape, summarize, synthesize) | G-01 | Not Started |
| Tavily search with ExpandSearch pattern (3 variations, 34.3% improvement) | G-03 | Not Started |
| Trafilatura scraping with quality scoring (F1=0.958) | G-04 | Not Started |
| Basic model routing (Anthropic primary, OpenAI fallback) | G-07 | Not Started |
| SQLite checkpointing via LangGraph SqliteSaver | G-05 | Not Started |
| Basic Rich CLI with progress bars and plan approval | G-13 | Not Started |
| Markdown report output with citation management | G-09 | Not Started |
| Unit tests with VCR.py recording setup | G-15 | Not Started |

**Milestone:** Running `research-agent "topic"` produces a complete Markdown report with cited sources from end-to-end pipeline execution.

**Key research insights driving Phase 1:**
- Only 4 of 16+ surveyed agents have genuine crash resilience (G-05)
- ExpandSearch pattern: 34.3% improvement over single-query baselines (arXiv 2510.10009)
- Trafilatura: F1=0.958, precision 93.8%, recall 97.8% on ScrapingHub benchmark (SIGIR 2023)
- Tavily: 93.3% accuracy on SimpleQA benchmark, natural language queries

---

## Phase 2: Resilience & Quality

**Target:** 2 weeks
**Goal:** Agent recovers from crash mid-research and stays within budget

| Task | Gap Ref | Status |
|------|---------|--------|
| Atomic file-based checkpoints (temp+fsync+os.replace with SHA-256) | G-05, G-06 | Not Started |
| Observation masking (83.9% token savings, M=10 rolling window) | G-08 | Not Started |
| Four-tier graceful degradation (FULL/REDUCED/CACHED/PARTIAL) | G-07, G-10 | Not Started |
| Budget tracking with $2.00 cap and 80% warning threshold | G-10 | Not Started |
| Self-evaluation: LLM-as-judge, 5 dimensions, 3.5/5.0 threshold | G-11 | Not Started |
| Prompt caching (Anthropic 90% discount, stability ordering) | G-17 | Not Started |
| Content quality scoring (reject below 0.4, flag 0.4-0.7) | G-04 | Not Started |
| Structured logging with provenance chains (structlog) | G-14 | Not Started |

**Milestone:** Agent recovers from a crash mid-research, resumes from the last checkpoint, and enforces cost caps throughout execution.

**Key research insights driving Phase 2:**
- 83.9% of context tokens come from tool observations (NeurIPS 2025, "The Complexity Trap")
- Prompt caching: 90% Anthropic discount, break-even after 2 calls; 50-90% OpenAI automatic
- Session cost reduction: $0.06-$0.15 optimized vs $2.43 naive (96-97% savings)
- Model routing: ~82% savings vs single-model Sonnet 4.5
- LLM-as-judge achieves ~80% agreement with human evaluators

---

## Phase 3: Advanced Features

**Target:** 3 weeks
**Goal:** Handles complex multi-topic research with deduplication

| Task | Gap Ref | Status |
|------|---------|--------|
| ChromaDB vector store for deduplication (0.85 similarity, 0.95 exact) | G-12 | Not Started |
| nomic-embed-text-v1.5 local embeddings (768d, 8K context) | G-12 | Not Started |
| Serial section-by-section synthesis (bypasses ~2,000 word ceiling) | G-09 | Not Started |
| Crawl4AI for JavaScript-heavy sites | G-04 | Not Started |
| MCP server integration for extensibility | G-19 | Not Started |
| API key rotation for heavy workloads | G-18 | Not Started |
| Human-in-the-loop plan review via LangGraph interrupt() | G-13 | Not Started |
| Multi-format output (Markdown, PDF via pymupdf) | G-20 | Not Started |

**Milestone:** Complex multi-topic research queries produce deduplicated, high-quality reports with vector-backed source management.

---

## Phase 4: Ecosystem Integration

**Target:** 2 weeks
**Goal:** Drop-in replacement for Claude.ai research step

| Task | Gap Ref | Status |
|------|---------|--------|
| Direct integration with RESEARCH_PROMPT.md format | G-02 | Not Started |
| COMPILED_RESEARCH.md output matching BUILD_PROMPT.md expectations | G-02 | Not Started |
| AgentPromptsManager "Run Research" button | G-19 | Not Started |
| Enhancement research mode for existing projects | - | Not Started |
| Cross-session knowledge/memory (ChromaDB persistence) | G-12 | Not Started |
| Web UI for real-time progress monitoring | G-20 | Not Started |

**Milestone:** Research agent plugs directly into the Apps That Build Apps pipeline, replacing the manual Claude.ai research step with automated, reproducible output.

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
| G-15 | Test infrastructure (VCR.py, fixtures) | 1 | High |
| G-16 | 4-layer configuration resolution | 1 | High |
| G-17 | Prompt caching optimization | 2 | Medium |
| G-18 | API key rotation | 3 | Low |
| G-19 | MCP/extensibility integration | 3-4 | Low |
| G-20 | Multi-format output and web UI | 3-4 | Low |

# Feature Backlog

## Phase 1: MVP (Core Pipeline)

| Feature | Priority | Effort | Dependencies | Acceptance Criteria |
|---------|----------|--------|--------------|---------------------|
| Project scaffolding (uv + pyproject.toml) | P0 | S | None | `uv sync` installs all deps; `research-agent --help` prints usage |
| Configuration system | P0 | S | Scaffolding | Loads config.yaml with pydantic-settings validation; env vars override file values |
| LangGraph StateGraph (5 nodes) | P0 | L | Configuration | Graph executes plan -> search -> scrape -> summarize -> synthesize in sequence |
| Tavily search integration | P0 | M | StateGraph | Returns ranked results for a query; respects max_results config |
| Trafilatura scraping | P0 | M | Tavily search | Extracts clean text from URLs; assigns quality score 0.0-1.0; filters below threshold |
| Model routing (Anthropic + OpenAI) | P0 | M | Configuration | Routes to Anthropic by default; falls back to OpenAI on failure; respects config |
| SQLite checkpointing | P1 | M | StateGraph | LangGraph SqliteSaver persists state; re-running resumes from last checkpoint |
| Rich CLI with progress bars | P1 | M | StateGraph | Shows live progress for each pipeline stage; displays token/cost summary on completion |
| Markdown report output | P0 | M | Synthesis node | Produces structured .md file with Executive Summary, Findings, and Sources sections |
| Unit tests with VCR.py | P1 | M | All nodes | 80% coverage on node functions; recorded cassettes for search/scrape calls |

## Phase 2: Resilience & Quality

| Feature | Priority | Effort | Dependencies | Acceptance Criteria |
|---------|----------|--------|--------------|---------------------|
| Atomic file-based checkpoints | P0 | M | Phase 1 checkpoint | Per-subtask JSON files written atomically (temp -> fsync -> replace); SHA-256 verified |
| Observation masking | P1 | M | StateGraph | Raw scraped content replaced with summaries after summarize node; state size stays bounded |
| Graceful degradation (4-tier) | P0 | M | Model routing | Tier 1: retry -> Tier 2: fallback model -> Tier 3: skip subtopic -> Tier 4: partial report |
| Budget tracking and cost caps | P0 | S | Model routing | Tracks cumulative cost per run; halts execution at max_cost_per_run; warns at threshold |
| Self-evaluation (LLM-as-judge) | P1 | M | Synthesis | Scores report on coverage, accuracy, coherence (1-5 scale); logged with report |
| Prompt caching optimization | P2 | S | Model routing | Enables prompt caching headers for Anthropic; uses cached_tokens for OpenAI |
| Content quality scoring | P1 | S | Scraping | Scores scraped content before LLM processing; drops content below 0.3 threshold |
| Structured logging (structlog) | P1 | M | None | JSON-formatted logs with provenance chain; each step logs parent_id and step_id |

## Phase 3: Advanced Features

| Feature | Priority | Effort | Dependencies | Acceptance Criteria |
|---------|----------|--------|--------------|---------------------|
| ChromaDB vector store | P0 | M | Phase 2 | Stores document embeddings; deduplicates content with cosine similarity > 0.92 threshold |
| nomic-embed-text-v1.5 embeddings | P0 | M | ChromaDB | Generates 768-dim embeddings locally; no API calls required for embedding |
| Serial section-by-section synthesis | P1 | L | Synthesis node | Synthesizes report section-by-section to handle long context; maintains coherence across sections |
| Crawl4AI for JS-heavy sites | P1 | M | Scraping | Falls back to Crawl4AI when trafilatura returns low-quality content from JS-rendered pages |
| MCP server integration | P2 | L | StateGraph | Exposes research capabilities via MCP protocol; external tools can trigger research |
| API key rotation | P2 | S | Model routing | Round-robins across multiple API keys; tracks rate limits per key |
| Human-in-the-loop plan review | P1 | M | Plan node, CLI | LangGraph interrupt() pauses after plan; user approves/edits subtopics before search |
| Multi-format output (MD, PDF) | P2 | M | Report output | Generates PDF via pymupdf in addition to Markdown; same content, formatted layout |

## Phase 4: Ecosystem Integration

| Feature | Priority | Effort | Dependencies | Acceptance Criteria |
|---------|----------|--------|--------------|---------------------|
| RESEARCH_PROMPT.md integration | P0 | M | Phase 1 | Parses RESEARCH_PROMPT.md format; extracts topic, constraints, and output requirements |
| COMPILED_RESEARCH.md output | P0 | M | Report output | Output matches BUILD_PROMPT.md expected format; includes all required sections |
| AgentPromptsManager button | P1 | L | RESEARCH_PROMPT.md | "Run Research" button in web UI triggers agent; status displayed in real-time |
| Enhancement research mode | P1 | M | Full pipeline | Accepts existing project context; researches targeted improvements rather than full topics |
| Cross-session memory | P2 | M | ChromaDB | Persists research findings across sessions; avoids re-researching known topics |
| Web UI for progress monitoring | P2 | L | Full pipeline | Real-time dashboard showing current node, progress, cost, and intermediate results |

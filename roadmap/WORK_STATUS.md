# Work Status

## Current Phase

**Phase 3: Advanced Features** (in progress)

## Completed Work

### Phase 1: MVP (Core Pipeline) -- Complete
- Project scaffolding (uv, pyproject.toml, directory structure)
- Configuration system (pydantic-settings, YAML validation, 4-layer resolution)
- ResearchState TypedDict with accumulator annotations
- 5-node LangGraph StateGraph (plan, search, scrape, summarize, synthesize)
- Tavily search with ExpandSearch pattern (3 query variations via LLM)
- Trafilatura scraping with quality scoring
- Model routing (litellm, Anthropic primary with OpenAI fallback)
- SQLite checkpointing via LangGraph and atomic file-based checkpoints
- Rich CLI with progress bars, plan approval, Ctrl+C handling
- Markdown report output with citation management

### Phase 2: Resilience & Quality -- Complete
- Atomic checkpoints with SHA-256 integrity verification
- Token estimation for context window management
- Budget tracking with configurable caps and warning thresholds
- Self-evaluation (LLM-as-judge, 5 dimensions, scoring with rubrics)
- Prompt caching with stability ordering
- Content quality scoring (reject, flag, accept thresholds)
- Structured logging with structlog and provenance chains
- Scraping: paywall detection, freshness scoring, sanitization
- Scraping: structured data extraction (JSON-LD, microdata, Open Graph)
- Metrics collection and dashboard display
- Human-in-the-loop plan review ($EDITOR and inline editing)
- Event logging and progress tracking

### Phase 3 (completed branches)
- F11.1: Centralized exception hierarchy (PR #43)
- F11.2: LiteLLM migration, replacing langchain for all LLM calls (PR #44)
- F11.3: Data model rename (Subtopic, ScrapedPage, SubtopicSummary) (PR #45)
- F11.4: Coverage improvements (graph.py 100%, cli.py 99%, structured.py 99%, overall 98%)

## Current Branch

`feature/F11.4-coverage-gaps` (based on `feature/F11.3-data-model-rename`)

## Test Coverage

- **1,306 tests**, all passing
- **98% overall branch coverage** (above 80% threshold)
- Key files: graph.py 100%, cli.py 99%, structured.py 99%, state.py 100%

## Remaining Work (Phase 3)

| Branch | Description | Status |
|--------|-------------|--------|
| F12.1 | Plan node implementation (LLM-powered decomposition) | Next |
| F12.2 | Embeddings + ChromaDB vector store | Pending |
| F12.3 | Serial section-by-section synthesis | Pending |
| F12.4 | Crawl4AI fallback for JS-heavy sites | Pending |
| F12.5 | API key rotation | Pending |
| F12.6 | PDF output via pymupdf | Pending |
| FB.1 | DiskCache for LLM deduplication | Pending |
| FB.2 | Prompt versioning | Pending |
| FB.3 | Adaptive rate limiting | Pending |

## Remaining Work (Phase 4)

| Branch | Description | Status |
|--------|-------------|--------|
| F13.1 | Cross-session memory (ChromaDB persistence) | Pending |
| F13.2 | RESEARCH_PROMPT.md integration | Pending |
| F13.3 | COMPILED_RESEARCH.md output | Pending |

## Blockers

None.

## Architecture Notes

- **LLM calls**: All via `litellm.acompletion` with JSON format instructions + `_extract_json()` parsing
- **State management**: LangGraph with TypedDict + `Annotated[list, operator.add]` accumulators
- **Branch strategy**: Stacked feature branches (each PR targets the previous branch)
- **Model IDs**: `anthropic/claude-sonnet-4-5-20250929` (SMART), `anthropic/claude-haiku-3-5-20241022` (FAST)

# Architecture

## System Overview

```
                          +------------------+
                          |   CLI (Typer)    |
                          |   Rich Progress  |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          |   LangGraph      |
                          |   StateGraph     |
                          +--------+---------+
                                   |
              +--------------------+--------------------+
              |          |         |         |          |
              v          v         v         v          v
         +--------+ +--------+ +-------+ +--------+ +----------+
         |  Plan  | | Search | | Scrape| |Summarize| |Synthesize|
         |  Node  | |  Node  | | Node  | |  Node   | |  Node    |
         +--------+ +--------+ +-------+ +--------+ +----------+
              |          |         |         |          |
              v          v         v         v          v
         +--------+ +--------+ +-------+ +--------+ +----------+
         | Model  | | Tavily | |Trafil.| | Model  | | Model    |
         | Router | |  API   | |Crawl4A| | Router | | Router   |
         +--------+ +--------+ +-------+ +--------+ +----------+
                                   |
                          +--------+---------+
                          |   Persistence    |
                          |  JSONL / SQLite  |
                          |  / Checkpoint    |
                          +------------------+
```

## Core Pattern: Decompose -> Parallel Search -> Compress -> Synthesize

The research pipeline follows a four-stage information processing pattern:

1. **Decompose** -- The plan node breaks a broad research topic into focused subtopics. Each subtopic becomes an independent unit of work with its own search queries and scrape targets.

2. **Parallel Search** -- The search node fans out across subtopics, generating multiple query variations per subtopic and executing them against the Tavily API. Results are collected and deduplicated.

3. **Compress** -- The scrape and summarize nodes extract content from discovered URLs and compress it into dense summaries. Raw content is replaced with summaries (observation masking) to keep state size bounded.

4. **Synthesize** -- The synthesize node combines all subtopic summaries into a coherent final report with citations and structured sections.

## Fan-Out / Fan-In Pattern

```
                    Plan
                     |
            +--------+--------+
            |        |        |
            v        v        v
        Subtopic  Subtopic  Subtopic
           A        B        C
            |        |        |
            v        v        v
         Search   Search   Search
            |        |        |
            v        v        v
         Scrape   Scrape   Scrape
            |        |        |
            v        v        v
        Summarize Summarize Summarize
            |        |        |
            +--------+--------+
                     |
                     v
                 Synthesize
                     |
                     v
                  Report
```

The fan-out occurs after planning, where each subtopic is processed independently through search, scrape, and summarize. The fan-in happens at synthesis, where all subtopic summaries converge into a single report. This pattern enables:

- **Incremental progress** -- Each subtopic checkpoint is independent.
- **Fault isolation** -- A failed subtopic does not block others.
- **Cost control** -- Budget can be allocated per subtopic.

## LangGraph StateGraph Wiring

The graph is defined as a `StateGraph[ResearchState]` with five nodes and conditional edges:

```
START -> plan -> should_search? --(yes)--> search -> scrape -> summarize -> all_subtopics_done?
                      |                                                          |
                      |                                                   (no)---+---> search
                      |                                                          |
                      +---(budget exceeded)---> synthesize                 (yes)--+---> synthesize
                                                    |
                                                    v
                                                   END
```

**Conditional edges:**

- `should_search` -- Checks if the plan produced subtopics and budget remains. Routes to `search` or directly to `synthesize` (if budget exceeded or no subtopics).
- `all_subtopics_done` -- After summarizing, checks if more subtopics remain. Routes back to `search` for the next subtopic or forward to `synthesize`.

**State management:**

The `ResearchState` TypedDict uses `Annotated` reducers for list fields (e.g., `Annotated[list[SearchResult], operator.add]`) so that each node appends to shared lists rather than overwriting them.

## Persistence Layer (3 Layers)

The system uses three complementary persistence mechanisms:

### Layer 1: JSONL Event Log

Every significant event (node entry, node exit, error, result) is appended to a `.jsonl` file. This provides a complete audit trail for debugging and provenance tracking. Each entry includes a timestamp, step ID, parent ID, and event payload.

**Location:** `data/checkpoints/{run_id}/events.jsonl`

### Layer 2: Atomic Checkpoint Files

After each node completes, the full `ResearchState` is serialized to a JSON checkpoint file using an atomic write pattern:

1. Write to a temporary file in the same directory.
2. Call `fsync` to ensure data hits disk.
3. Use `os.replace` to atomically swap the temp file into place.
4. Compute SHA-256 hash and store in a sidecar `.sha256` file.

On resume, the checkpoint is loaded and its hash verified before continuing.

**Location:** `data/checkpoints/{run_id}/checkpoint_{step}.json`

### Layer 3: Progressive Markdown

As each subtopic is summarized, its content is appended to a progressive Markdown file. Even if the agent crashes before synthesis, this file contains all completed work in a readable format.

**Location:** `data/checkpoints/{run_id}/progress.md`

## Model Router (3 Tiers)

The model router provides resilient LLM access with automatic fallback:

| Tier | Provider | Model | Use Case |
|------|----------|-------|----------|
| 1 (Primary) | Anthropic | claude-sonnet-4-5-20250929 | All planning, summarization, synthesis |
| 2 (Fallback) | OpenAI | gpt-4o | Used when Anthropic is unavailable or rate-limited |
| 3 (Budget) | Google | gemini-2.0-flash | Used when cost cap is approaching; cheapest option |

The router:
- Tracks cumulative token usage and cost per provider.
- Retries with exponential backoff (via tenacity) before falling to the next tier.
- Logs every model call with provider, model, token counts, and latency.

## Configuration System (4 Layers)

Configuration is resolved in order of increasing priority:

| Layer | Source | Example |
|-------|--------|---------|
| 1. Defaults | Hardcoded in pydantic-settings models | `temperature: 0.1` |
| 2. Config file | `config.yaml` in project root | `llm.model: claude-sonnet-4-5-20250929` |
| 3. Environment variables | Shell env or `.env` file | `RESEARCH_AGENT_LLM__MODEL=gpt-4o` |
| 4. CLI arguments | Typer options | `--model gpt-4o` |

Each layer overrides the one below it. The pydantic-settings model validates the merged configuration at startup, providing clear error messages for invalid values.

## Data Flow

```
User Query
    |
    v
[Plan Node]
    - Input:  query (str)
    - Output: subtopics (list[Subtopic]), search_queries (list[str])
    - LLM:    Generates decomposition plan
    |
    v
[Search Node]
    - Input:  search_queries for current subtopic
    - Output: search_results (list[SearchResult])
    - API:    Tavily search with 3 query variations per subtopic
    |
    v
[Scrape Node]
    - Input:  URLs from search_results
    - Output: scraped_content (list[ScrapedPage])
    - Tool:   Trafilatura (primary), Crawl4AI (fallback)
    - Filter: Quality score >= 0.3 threshold
    |
    v
[Summarize Node]
    - Input:  scraped_content for current subtopic
    - Output: subtopic_summaries (list[SubtopicSummary])
    - LLM:    Condenses scraped content into dense summary with citations
    - Side effect: Observation masking (raw content dropped from state)
    |
    v
[Synthesize Node]
    - Input:  All subtopic_summaries
    - Output: final_report (str), report_metadata (ReportMetadata)
    - LLM:    Combines summaries into structured report
    |
    v
Markdown Report File
```

Each node reads from and writes to the shared `ResearchState`. The state is checkpointed after every node transition.

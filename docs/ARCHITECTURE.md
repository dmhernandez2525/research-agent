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

## Design Rationale / Ecosystem Context

A survey of 16+ open-source research agent projects informed every major design decision in this system. The landscape ranges from mature, high-star projects to experimental prototypes, and each revealed specific strengths and blind spots that shaped our architecture.

### Key Projects Analyzed

| Project | Stars | Key Takeaway |
|---------|-------|--------------|
| Perplexica | ~28.5K | Real-time search focus; lightweight but limited depth |
| DeerFlow | ~18.1K | Multi-agent coordination patterns; heavy infrastructure |
| GPT Researcher | ~17K | Pioneered parallel subtopic decomposition; weak on crash recovery |
| STORM | ~14-15K | Academic rigor with outline-driven synthesis; single-model dependency |
| Open Deep Research | ~10.3K | Minimal viable pipeline; useful baseline for comparison |

### The Checkpointing Gap

Only 4 of the 16+ surveyed projects implement genuine crash resilience -- the ability to resume a partially completed research session without re-running completed work or losing intermediate results. Most projects treat the pipeline as an all-or-nothing transaction. This gap directly motivated our three-layer persistence design (JSONL event log, atomic checkpoints, progressive Markdown).

### Research-Backed Design Principles

- **Token budget is the dominant cost lever.** Anthropic's internal benchmarking found that token usage explains approximately 80% of the variance in agentic system performance. Our budget tracking and observation masking are responses to this finding.
- **Multi-agent outperforms single-agent.** The same research showed multi-agent architectures outperforming single-agent by 90.2% on complex research tasks. Our decompose-and-fan-out pattern leverages this by treating each subtopic as an independent agent-like unit of work.
- **Multi-agent systems are fragile in production.** The MAST taxonomy documents failure rates of 41-86.7% across production multi-agent deployments. This informed our conservative approach: fan-out within a single graph (not separate processes), with graceful degradation rather than complex inter-agent protocols.

### Commercial System Patterns

Analysis of commercial research systems revealed three dominant architectural patterns:

- **Claude (Anthropic):** Orchestrator-worker pattern where a central orchestrator delegates subtasks to specialized workers. Our plan node plays the orchestrator role.
- **Gemini (Google):** Async task manager with parallel execution streams. Our fan-out/fan-in pattern mirrors this approach within LangGraph.
- **OpenAI Deep Research:** Three-step pipeline (plan, gather, synthesize). Our four-stage pipeline extends this with an explicit compress stage to control context growth.

## Core Pattern: Decompose -> Parallel Search -> Compress -> Synthesize

The research pipeline follows a four-stage information processing pattern:

1. **Decompose** -- The plan node breaks a broad research topic into focused subtopics. Each subtopic becomes an independent unit of work with its own search queries and scrape targets.

2. **Parallel Search** -- The search node fans out across subtopics, generating multiple query variations per subtopic and executing them against the Tavily API (93.3% accuracy on the SimpleQA benchmark). Results are collected and deduplicated. Generating multiple query reformulations per subtopic follows the ExpandSearch strategy, which demonstrates a 34.3% improvement over single-query retrieval (arXiv 2510.10009).

3. **Compress** -- The scrape and summarize nodes extract content from discovered URLs (Trafilatura: F1=0.958, precision 93.8%, recall 97.8% on the SIGIR 2023 ScrapingHub benchmark) and compress it into dense summaries. Raw content is replaced with summaries (observation masking) to keep state size bounded. This is critical because 83.9% of context tokens in agentic systems come from tool observations (NeurIPS 2025, JetBrains Research).

4. **Synthesize** -- The synthesize node combines all subtopic summaries into a coherent final report with citations and structured sections. Reports are assembled section-by-section rather than in a single pass, since LLMs hit a ~2,000 word glass ceiling for single-pass generation beyond which quality degrades sharply.

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

## Observation Masking

Research from NeurIPS 2025 (JetBrains Research) shows that 83.9% of context tokens in agentic LLM systems originate from tool observations -- the raw content returned by search APIs, scraped pages, and intermediate results. Left unmanaged, context windows fill rapidly, degrading both quality and cost.

### Three-Stage Capacity Ladder

Observation masking activates progressively as context utilization rises:

| Trigger | Action | Effect |
|---------|--------|--------|
| 75% context capacity | **Masking** -- Drop raw scraped content from state after summarization; retain only the dense summary and source URLs | Prevents unbounded growth |
| 80% context capacity | **Summarization** -- Compress existing subtopic summaries into shorter variants using a budget-tier model | Reclaims additional headroom |
| 85% context capacity | **File pointer replacement** -- Replace remaining large text blocks with file paths pointing to on-disk storage; the synthesize node reads files directly | Last-resort measure before forced termination |

### Rolling Window

The optimal rolling window is **M=10 turns** of conversation history retained in state. Earlier turns are evicted or summarized. The hybrid approach of masking combined with summarization yields an additional 7-11% token savings over masking alone.

## Four-Tier Graceful Degradation

When budget pressure or API failures accumulate, the system transitions through a four-tier state machine rather than failing abruptly:

```
FULL  -----(80% budget consumed)----->  REDUCED
  ^                                        |
  |                                  (95% budget OR
  |                                   5 consecutive failures)
  |                                        |
  |                                        v
FULL  <--(success + budget < 75%)---  CACHED
                                           |
                                     (all APIs down OR
                                      budget exhausted)
                                           |
                                           v
                                       PARTIAL
```

### Tier Definitions

| Tier | Behavior |
|------|----------|
| **FULL** | All features enabled. Primary model (Claude Sonnet), full search depth, complete scraping. |
| **REDUCED** | Fewer query variations per subtopic. Summarization uses the budget-tier model. Search depth reduced. |
| **CACHED** | No new API calls. The system works exclusively with already-retrieved content and cached results. |
| **PARTIAL** | Emergency synthesis from whatever content is available. Produces a partial report with explicit coverage gaps noted. |

### Model Fallback Chain

Within each degradation tier, individual LLM calls follow a provider fallback chain:

```
Claude Sonnet -> GPT-4o-mini -> local Ollama
```

A call falls to the next provider after retry exhaustion (exponential backoff via tenacity). The local Ollama tier ensures the system can produce output even with zero API connectivity, at reduced quality.

### Recovery

Recovery is automatic: if a call succeeds and the cumulative budget is below 75%, the system recovers upward by one tier. This prevents a single transient failure from permanently degrading the session.

## Prompt Caching Architecture

Prompt caching reduces LLM costs by reusing cached prefixes across calls. The savings are significant but require careful message structure.

### Provider Economics

| Provider | Cache Discount | Break-Even | Minimum Cacheable Size | Code Changes Required |
|----------|---------------|------------|------------------------|----------------------|
| Anthropic | 90% on cache reads | After 2 calls with same prefix | 1,024 tokens | Explicit `cache_control` markers |
| OpenAI | 50-90% automatic | Immediate | Automatic detection | None |

### Four Rules for Cache Stability

Cache hit rates depend on prefix stability. The message structure follows a strict ordering from most-static to most-dynamic:

```
1. Tools definition     (static -- never changes between calls)
2. System prompt        (static -- set once per session)
3. Prior conversation   (append-only -- grows but prefix is stable)
4. Latest user message  (dynamic -- changes every call)
```

Violating this ordering (e.g., injecting timestamps into the system prompt, reordering tools between calls) invalidates the cache prefix and eliminates savings.

**Implementation rules:**

1. No dynamic data (timestamps, random IDs) in the system prompt.
2. Conversation context is append-only -- never edit or reorder prior messages.
3. Tool definitions must not change between calls within a session.
4. JSON serialization of tool schemas must be deterministic (sorted keys, no whitespace variation).

### Cost Impact

With prompt caching and model routing combined, session cost drops from approximately $2.43 (naive single-model, no caching) to $0.06-0.15 (optimized), a 96-97% reduction. Model routing alone contributes approximately 82% savings versus using Claude Sonnet 4.5 for every call.

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
- Achieves approximately 82% cost savings versus routing all calls through a single Sonnet 4.5 model, by reserving the primary tier for planning and synthesis while delegating search-heavy and summarization work to cheaper tiers.

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
    - Tool:   Trafilatura (primary, F1=0.958), Crawl4AI (fallback)
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
    - Eval:   LLM-as-judge scoring (~80% agreement with human evaluators)
    |
    v
Markdown Report File
```

Each node reads from and writes to the shared `ResearchState`. The state is checkpointed after every node transition.

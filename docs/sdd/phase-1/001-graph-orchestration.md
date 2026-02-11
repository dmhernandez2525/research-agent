# SDD-001: Graph Orchestration

## Overview

This document defines the LangGraph StateGraph that orchestrates the research pipeline. The graph manages five nodes (plan, search, scrape, summarize, synthesize), routes between them with conditional edges, and persists state via SQLite checkpointing.

## ResearchState TypedDict

```python
from typing import Annotated, TypedDict
import operator

class Subtopic(TypedDict):
    id: str
    title: str
    description: str
    search_queries: list[str]
    status: str  # "pending" | "searching" | "scraping" | "summarizing" | "done" | "failed"

class SearchResult(TypedDict):
    url: str
    title: str
    snippet: str
    score: float
    subtopic_id: str

class ScrapedPage(TypedDict):
    url: str
    content: str
    quality_score: float
    word_count: int
    subtopic_id: str

class SubtopicSummary(TypedDict):
    subtopic_id: str
    title: str
    summary: str
    citations: list[str]
    token_count: int

class ResearchState(TypedDict):
    # Input
    query: str
    config: dict

    # Plan
    subtopics: list[Subtopic]
    current_subtopic_index: int

    # Search (reducer: append)
    search_results: Annotated[list[SearchResult], operator.add]

    # Scrape (reducer: append)
    scraped_pages: Annotated[list[ScrapedPage], operator.add]

    # Summarize (reducer: append)
    subtopic_summaries: Annotated[list[SubtopicSummary], operator.add]

    # Synthesize
    final_report: str
    report_metadata: dict

    # Tracking
    total_cost: float
    total_tokens: int
    errors: Annotated[list[dict], operator.add]
    seen_urls: set[str]
```

Fields using `Annotated[list[T], operator.add]` are append-only. Each node returns new items in these fields, and the reducer concatenates them with existing values. Non-annotated fields are overwritten on each update.

## Node Function Signatures

Each node is an async function that accepts `ResearchState` and returns a partial state dict:

```python
async def plan_node(state: ResearchState) -> dict:
    """Decomposes query into subtopics with search queries."""
    ...

async def search_node(state: ResearchState) -> dict:
    """Executes search queries for the current subtopic."""
    ...

async def scrape_node(state: ResearchState) -> dict:
    """Scrapes URLs from search results, scores quality."""
    ...

async def summarize_node(state: ResearchState) -> dict:
    """Summarizes scraped content for the current subtopic."""
    ...

async def synthesize_node(state: ResearchState) -> dict:
    """Combines all subtopic summaries into a final report."""
    ...
```

**Return contract:** Each node returns only the fields it modifies. For reducer fields, it returns new items to append. For scalar fields, it returns the updated value.

## Graph Definition

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(ResearchState)

# Add nodes
graph.add_node("plan", plan_node)
graph.add_node("search", search_node)
graph.add_node("scrape", scrape_node)
graph.add_node("summarize", summarize_node)
graph.add_node("synthesize", synthesize_node)

# Edges
graph.add_edge(START, "plan")
graph.add_conditional_edges("plan", should_search)
graph.add_edge("search", "scrape")
graph.add_edge("scrape", "summarize")
graph.add_conditional_edges("summarize", all_subtopics_done)
graph.add_edge("synthesize", END)
```

## Conditional Edge Routing

### `should_search`

```python
def should_search(state: ResearchState) -> str:
    if not state["subtopics"]:
        return "synthesize"
    if state["total_cost"] >= state["config"]["costs"]["max_cost_per_run"]:
        return "synthesize"
    return "search"
```

Routes to `synthesize` if the plan produced no subtopics or if the cost cap has been reached. Otherwise, proceeds to `search`.

### `all_subtopics_done`

```python
def all_subtopics_done(state: ResearchState) -> str:
    next_index = state["current_subtopic_index"] + 1
    if next_index >= len(state["subtopics"]):
        return "synthesize"
    if state["total_cost"] >= state["config"]["costs"]["max_cost_per_run"]:
        return "synthesize"
    return "search"
```

After summarizing a subtopic, checks if more remain. Routes back to `search` for the next subtopic or forward to `synthesize` when all are done (or budget is exhausted).

## Checkpoint Integration

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def build_graph(checkpoint_path: str):
    memory = AsyncSqliteSaver.from_conn_string(checkpoint_path)
    return graph.compile(checkpointer=memory)
```

The compiled graph automatically checkpoints state after every node transition. To resume a previous run:

```python
config = {"configurable": {"thread_id": run_id}}
result = await compiled_graph.ainvoke(None, config=config)
```

Passing `None` as input with an existing `thread_id` resumes from the last checkpoint.

## Error Handling

Each node wraps its logic in a try/except. On failure:

1. The error is appended to `state["errors"]` with the node name, timestamp, and traceback.
2. The node returns its partial state (including the error) so the checkpoint captures the failure.
3. The conditional edge logic checks error state and can route to `synthesize` for a partial report if too many errors accumulate.

## File Location

```
src/research_agent/
    graph.py          # StateGraph definition, conditional edges, build_graph()
    state.py          # ResearchState TypedDict and related types
    nodes/
        plan.py       # plan_node
        search.py     # search_node
        scrape.py     # scrape_node
        summarize.py  # summarize_node
        synthesize.py # synthesize_node
```

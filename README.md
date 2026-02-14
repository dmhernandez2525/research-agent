# Research Agent

Crash-resilient deep research, locally run.

A standalone Python CLI tool that replaces manual browser-based research workflows with an automated, checkpointed pipeline. It programmatically searches the web, scrapes content, and synthesizes comprehensive cited reports -- without character limits, session timeouts, or manual copy-paste orchestration.

## Features

- **Crash Resilient** -- Checkpoints after every step. Resume exactly where you left off after any failure (power outage, API timeout, Ctrl+C).
- **No Character Limits** -- Reports can be any length. Each sub-task saves results to disk independently; synthesis reads from files, not context.
- **Automated Pipeline** -- One command triggers the full pipeline: plan, search, scrape, analyze, synthesize.
- **Cost Controlled** -- Budget caps, three-tier model routing, and graceful degradation prevent runaway API costs.

## Quick Start

```bash
# Install
pip install research-agent
# or with uv:
uv tool install research-agent

# Configure API keys
export ANTHROPIC_API_KEY=your-key
export TAVILY_API_KEY=your-key

# Run health diagnostics
research-agent doctor

# Run research
research-agent "Best practices for building SaaS applications in 2026"
```

## Docker

```bash
# Build and run in container
docker compose up --build

# Run an ad-hoc query
docker compose run --rm research-agent run "AI agent evaluation framework"
```

## API Server

```bash
# Start API server
research-agent serve --port 8000

# Manage API keys
research-agent api-keys --create local-dev --admin
research-agent api-keys --list
```

## How It Works

```
Query --> Planner --> [Sub-task 1] --> save to disk
                      [Sub-task 2] --> save to disk  (parallel)
                      [Sub-task 3] --> save to disk
                               |
                               v
                      Synthesizer reads ALL files
                               |
                               v
                      Final Report (Markdown)
```

1. **Plan** -- Decomposes your query into 3-7 independent sub-questions
2. **Search** -- Tavily semantic search with 3 query variations per sub-question
3. **Scrape** -- Trafilatura content extraction with quality scoring
4. **Analyze** -- Per-subtask summarization with context management
5. **Synthesize** -- Final report generation from all findings on disk

## CLI Usage

```bash
# Basic research
research-agent "What are the best practices for building SaaS applications?"

# With options
research-agent "Competitor analysis for photography ERP" \
    --depth 7 \
    --model claude-sonnet-4 \
    --budget 3.0 \
    --output ./research/COMPILED_RESEARCH.md

# Resume after interruption (same query = auto-resume)
research-agent "Competitor analysis for photography ERP"

# Auto-approve research plan
research-agent "Market analysis for AI podcast generators" --yes
```

## Configuration

Configuration uses a 4-layer resolution: defaults -> `config.yaml` -> environment variables -> CLI arguments.

```yaml
# config.yaml
llm:
  provider: "anthropic"
  model: "claude-sonnet-4-5-20250929"
  temperature: 0.1

search:
  provider: "tavily"
  max_results: 10

costs:
  max_cost_per_run: 2.00
  warn_at_percentage: 80
```

Override via environment variables:

```bash
LLM__MODEL=gpt-4o-mini research-agent "your query"
```

## Development

```bash
# Clone and setup
git clone https://github.com/dmhernandez2525/research-agent.git
cd research-agent
uv sync

# Run tests
uv run pytest

# Lint and type check
uv run ruff check src/
uv run mypy src/
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph 1.0 |
| LLM Providers | langchain-anthropic, langchain-openai |
| Search | Tavily API |
| Scraping | Trafilatura |
| Configuration | pydantic-settings |
| CLI | Typer + Rich |
| Logging | structlog |
| Testing | pytest + VCR.py |

## License

MIT

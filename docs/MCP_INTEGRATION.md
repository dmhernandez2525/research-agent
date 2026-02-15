# MCP Integration Guide

## Server Capabilities
- Name: `research-agent`
- Transports: `stdio`, `sse`
- Tools: `research`, `recall`, `evaluate`, `status`
- Resources: `reports://`, `sessions://`, `memory://`

## Claude Desktop Configuration
Use this shape in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "research-agent": {
      "command": "uv",
      "args": ["run", "research-agent", "mcp", "serve"]
    }
  }
}
```

A sample file is also stored at `docs/claude_desktop_config.json`.

## Cursor / VS Code
Use the same command/args in your MCP extension settings. For SSE transport:
- Run `research-agent mcp serve --transport sse --port 8765`
- Point client base URL to `http://localhost:8765`

## Performance Benchmark
Benchmark helper:

```bash
uv run research-agent mcp benchmark --query "vector database tradeoffs"
```

This reports milliseconds from tool invocation to first result payload.

# openviking-mcp

Local semantic code search for Claude Code, powered by [Ollama](https://ollama.ai) embeddings and a SQLite vector store.

No cloud APIs. No API keys. Everything runs on your machine.

## What it does

An [MCP](https://modelcontextprotocol.io/) server that indexes your project files locally using BGE-M3 embeddings via Ollama, stores them in a SQLite vector database, and lets any MCP client (Claude Code, Claude Desktop, etc.) search for relevant code context.

## Tools

| Tool | Description |
|---|---|
| `search_context` | Semantic search over indexed files — finds code relevant to a natural language query |
| `index_project` | Index a directory (runs in background, check progress via status) |
| `openviking_status` | Check Ollama, model, index stats, and indexing progress |

## Quick Start

### Prerequisites

- [Go 1.25+](https://go.dev/dl/)
- [Ollama](https://ollama.ai) installed and running
- BGE-M3 model: `ollama pull bge-m3`

### Install

```bash
go install github.com/caio-silva/openviking-mcp@latest
```

Or build from source:

```bash
git clone https://github.com/caio-silva/openviking-mcp.git
cd openviking-mcp
make build
```

### Connect to Claude Code

```bash
claude mcp add openviking /path/to/openviking-mcp
```

### Use

In Claude Code, index your project first:

```
index this project for semantic search
```

Then just ask questions — Claude will use `search_context` automatically when it needs project context.

## Configuration

Optional. Create `~/.config/openviking-mcp/config.json`:

```json
{
  "ollamaEndpoint": "http://localhost:11434",
  "model": "bge-m3",
  "maxContextTokens": 4096,
  "excludePatterns": [".git", "node_modules", "vendor", ".viking_db"]
}
```

Or set `OPENVIKING_MCP_CONFIG=/path/to/config.json`.

Works with zero config — defaults to local Ollama with BGE-M3.

## How it works

1. `index_project` scans files, chunks them, generates embeddings via Ollama, stores in `.viking_db/vectors.db`
2. `search_context` embeds your query, finds similar chunks via cosine similarity
3. Index is incremental — re-indexing only processes changed files
4. `.viking_db/` lives in the project directory, portable between machines (same model required)

## License

Apache-2.0

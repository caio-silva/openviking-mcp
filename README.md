# openviking-mcp

Local semantic code search for Claude Code, powered by [Ollama](https://ollama.ai) embeddings (BGE-M3) and a SQLite vector store.

No cloud APIs. No API keys. Everything runs on your machine.

## Prerequisites

- [Go 1.25+](https://go.dev/dl/)
- [Ollama](https://ollama.ai) installed and running
- BGE-M3 model pulled: `ollama pull bge-m3`

## Install

```bash
go install github.com/caio-silva/openviking-mcp@latest
```

Or build from source:

```bash
git clone https://github.com/caio-silva/openviking-mcp.git
cd openviking-mcp
make build
```

## Connect to Claude Code

```bash
claude mcp add openviking /path/to/openviking-mcp
```

Then in Claude Code, index your project:

```
index this project for semantic search
```

After that, Claude uses `search_context` automatically when it needs project context.

## CLI Usage

```bash
# Index a project
openviking-mcp index /path/to/project

# Index with a custom DB location
openviking-mcp index /path/to/project --out /tmp/my-index

# Search indexed files
openviking-mcp search "payment processing logic"

# Check Ollama status and index stats
openviking-mcp status

# List all registered projects
openviking-mcp projects
```

## MCP Tools

| Tool | Description |
|---|---|
| `search_context` | Semantic search over indexed files. Accepts `query` (required), `limit`, and `project` (optional name/path). |
| `index_project` | Index a directory in the background. Accepts `path`. |
| `openviking_status` | Ollama reachability, model status, index stats, indexing progress, registered projects. |
| `list_projects` | List all projects in the registry with their names, paths, and DB locations. |

## Project Registry

When you index a project (via CLI or MCP), it gets registered automatically. The registry maps project names to their database locations.

This means `search_context` works from any working directory -- if you're inside a registered project, the correct index is found automatically. You can also pass a `project` parameter to search a specific project by name.

The registry lives at `~/.config/openviking-mcp/projects.json`.

Resolution order for finding the database:
1. `project` parameter (looked up in registry by name or path)
2. CWD-based lookup (if CWD is inside a registered project)
3. Fallback to `<cwd>/.viking_db/`

## Configuration

Optional. Create `~/.config/openviking-mcp/config.json`:

```json
{
  "ollamaEndpoint": "http://localhost:11434",
  "model": "bge-m3",
  "contextDepth": 1,
  "maxContextTokens": 4096,
  "excludePatterns": [".git", "node_modules", "vendor", ".viking_db", "__pycache__", ".idea"]
}
```

Or set `OPENVIKING_MCP_CONFIG=/path/to/config.json`.

Config precedence:
1. `$OPENVIKING_MCP_CONFIG` env var
2. `$XDG_CONFIG_HOME/openviking-mcp/config.json`
3. `~/.config/openviking-mcp/config.json`

Works with zero config -- defaults to local Ollama with BGE-M3.

## Portable Indexes

The `.viking_db/` directory contains a single `vectors.db` SQLite file. You can copy it between machines as long as the same embedding model (bge-m3) is used. File paths stored in the index are relative, so projects can live at different absolute paths.

## Ollama Tuning

The included `ollama-env.sh` sets environment variables optimized for embedding workloads:

```bash
source ollama-env.sh && ollama serve
```

## How It Works

1. `index_project` scans files, chunks them, generates embeddings via Ollama, stores in `.viking_db/vectors.db`
2. `search_context` embeds your query, finds similar chunks via cosine similarity
3. Indexing is incremental -- re-indexing only processes changed files (based on modtime)
4. Background indexing with progress tracking and stall detection

## License

Apache-2.0

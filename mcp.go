package main

import (
	"encoding/json"
	"fmt"
)

// --- JSON-RPC / MCP types ---

type jsonRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type jsonRPCResponse struct {
	JSONRPC string    `json:"jsonrpc"`
	ID      any       `json:"id,omitempty"`
	Result  any       `json:"result,omitempty"`
	Error   *rpcError `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type mcpToolInfo struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

type mcpToolResult struct {
	Content []mcpContent `json:"content"`
	IsError bool         `json:"isError,omitempty"`
}

type mcpContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func textResult(text string) mcpToolResult {
	return mcpToolResult{Content: []mcpContent{{Type: "text", Text: text}}}
}

func errResult(msg string) mcpToolResult {
	return mcpToolResult{Content: []mcpContent{{Type: "text", Text: msg}}, IsError: true}
}

// handle dispatches a JSON-RPC request to the appropriate handler.
func (s *server) handle(req jsonRPCRequest) *jsonRPCResponse {
	switch req.Method {
	case "initialize":
		return s.handleInitialize(req)
	case "notifications/initialized":
		return nil
	case "tools/list":
		return s.handleToolsList(req)
	case "tools/call":
		return s.handleToolsCall(req)
	case "ping":
		return &jsonRPCResponse{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{}}
	default:
		return &jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &rpcError{Code: -32601, Message: "method not found: " + req.Method},
		}
	}
}

func (s *server) handleInitialize(req jsonRPCRequest) *jsonRPCResponse {
	return &jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result: map[string]any{
			"protocolVersion": "2024-11-05",
			"capabilities": map[string]any{
				"tools": map[string]any{},
			},
			"serverInfo": map[string]any{
				"name":    "openviking-mcp",
				"version": "1.0.0",
			},
		},
	}
}

func (s *server) handleToolsList(req jsonRPCRequest) *jsonRPCResponse {
	tools := []mcpToolInfo{
		{
			Name:        "search_context",
			Description: "Search indexed project files for code relevant to a query. Uses local Ollama embeddings and a SQLite vector store.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "Natural language query or code snippet to search for",
					},
					"limit": map[string]any{
						"type":        "integer",
						"description": "Max results to return (default 5)",
						"default":     5,
					},
					"project": map[string]any{
						"type":        "string",
						"description": "Project name or path (optional — auto-detected from CWD or registry)",
					},
				},
				"required": []string{"query"},
			},
		},
		{
			Name:        "index_project",
			Description: "Index a directory for semantic search. Runs in the background — call openviking_status to check progress. Creates or updates the .viking_db/ vector store.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Absolute path to the directory to index",
					},
				},
				"required": []string{"path"},
			},
		},
		{
			Name:        "openviking_status",
			Description: "Check the status of the OpenViking context engine: Ollama reachability, model availability, index statistics, and indexing progress.",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
		{
			Name:        "list_projects",
			Description: "List all projects registered in the OpenViking index registry. Shows project names, paths, and database locations.",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
	}

	return &jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result:  map[string]any{"tools": tools},
	}
}

func (s *server) handleToolsCall(req jsonRPCRequest) *jsonRPCResponse {
	var params struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil {
		return &jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &rpcError{Code: -32602, Message: "invalid params: " + err.Error()},
		}
	}

	var result mcpToolResult
	switch params.Name {
	case "search_context":
		result = s.toolSearch(params.Arguments)
	case "index_project":
		result = s.toolIndex(params.Arguments)
	case "openviking_status":
		result = s.toolStatus()
	case "list_projects":
		result = s.toolListProjects()
	default:
		return &jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &rpcError{Code: -32602, Message: "unknown tool: " + params.Name},
		}
	}

	return &jsonRPCResponse{JSONRPC: "2.0", ID: req.ID, Result: result}
}

func (s *server) toolListProjects() mcpToolResult {
	projects := s.registry.All()
	if len(projects) == 0 {
		return textResult("No projects registered. Index a project first with index_project.")
	}

	type projectInfo struct {
		Name   string `json:"name"`
		Path   string `json:"path"`
		DBPath string `json:"dbPath"`
	}
	var list []projectInfo
	for _, p := range projects {
		list = append(list, projectInfo{
			Name:   p.Name,
			Path:   p.Path,
			DBPath: p.DBPath,
		})
	}

	out, _ := json.MarshalIndent(list, "", "  ")
	return textResult(fmt.Sprintf("Registered projects (%d):\n%s", len(list), string(out)))
}

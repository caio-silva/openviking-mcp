// Package mcp implements the JSON-RPC based MCP server for OpenViking.
package mcp

import (
	"encoding/json"
	"fmt"

	"github.com/caio-silva/openviking-mcp/internal/config"
	"github.com/caio-silva/openviking-mcp/internal/registry"
)

// Server holds runtime state for the MCP server.
type Server struct {
	Cfg      config.Config
	Registry *registry.Registry
	Index    IndexState
}

// TextResult creates a successful text result.
func TextResult(text string) MCPToolResult {
	return MCPToolResult{Content: []MCPContent{{Type: "text", Text: text}}}
}

// ErrResult creates an error text result.
func ErrResult(msg string) MCPToolResult {
	return MCPToolResult{Content: []MCPContent{{Type: "text", Text: msg}}, IsError: true}
}

// Handle dispatches a JSON-RPC request to the appropriate handler.
func (s *Server) Handle(req JSONRPCRequest) *JSONRPCResponse {
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
		return &JSONRPCResponse{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{}}
	default:
		return &JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &RPCError{Code: -32601, Message: "method not found: " + req.Method},
		}
	}
}

func (s *Server) handleInitialize(req JSONRPCRequest) *JSONRPCResponse {
	return &JSONRPCResponse{
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

func (s *Server) handleToolsList(req JSONRPCRequest) *JSONRPCResponse {
	tools := []MCPToolInfo{
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

	return &JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
		Result:  map[string]any{"tools": tools},
	}
}

func (s *Server) handleToolsCall(req JSONRPCRequest) *JSONRPCResponse {
	var params struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil {
		return &JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &RPCError{Code: -32602, Message: "invalid params: " + err.Error()},
		}
	}

	var result MCPToolResult
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
		return &JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &RPCError{Code: -32602, Message: "unknown tool: " + params.Name},
		}
	}

	return &JSONRPCResponse{JSONRPC: "2.0", ID: req.ID, Result: result}
}

func (s *Server) toolListProjects() MCPToolResult {
	projects := s.Registry.All()
	if len(projects) == 0 {
		return TextResult("No projects registered. Index a project first with index_project.")
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
	return TextResult(fmt.Sprintf("Registered projects (%d):\n%s", len(list), string(out)))
}

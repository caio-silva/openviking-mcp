// openviking-mcp is a stdio MCP server that provides local semantic code
// search powered by Ollama embeddings (BGE-M3) and a SQLite vector store.
//
// It enables any MCP client (Claude Code, Claude Desktop, etc.) to search,
// index, and inspect locally indexed project files — all running on your
// machine with no cloud APIs.
//
// Usage:
//
//	claude mcp add openviking /path/to/openviking-mcp
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/caio-silva/openviking-mcp/internal/openviking"
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

// --- Tool input types ---

type searchInput struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

type indexInput struct {
	Path string `json:"path"`
}

// --- Server ---

type indexState struct {
	mu         sync.Mutex
	running    bool
	path       string
	current    int
	total      int
	startedAt  time.Time
	lastUpdate time.Time
	result     *openviking.IndexResult
	err        error
	cancel     context.CancelFunc
}

type server struct {
	cfg   Config
	index indexState
}

func main() {
	log.SetOutput(os.Stderr)
	log.SetPrefix("openviking-mcp: ")

	cfg := LoadConfig()

	// CLI mode: if args are provided, run as a command-line tool
	if len(os.Args) > 1 {
		runCLI(cfg, os.Args[1:])
		return
	}

	// MCP mode: no args, run as stdio MCP server
	s := &server{cfg: cfg}

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var req jsonRPCRequest
		if err := json.Unmarshal(line, &req); err != nil {
			log.Printf("parse error: %v", err)
			continue
		}

		resp := s.handle(req)
		if resp == nil {
			continue
		}

		out, _ := json.Marshal(resp)
		fmt.Fprintf(os.Stdout, "%s\n", out)
	}
}

func runCLI(cfg Config, args []string) {
	cmd := args[0]
	// Parse --out flag from anywhere in args
	outPath := ""
	var cleanArgs []string
	for i := 1; i < len(args); i++ {
		if (args[i] == "--out" || args[i] == "-o") && i+1 < len(args) {
			outPath = args[i+1]
			i++
		} else {
			cleanArgs = append(cleanArgs, args[i])
		}
	}

	switch cmd {
	case "index":
		if len(cleanArgs) < 1 {
			fmt.Fprintf(os.Stderr, "Usage: openviking-mcp index <path> [--out /path/to/output/dir]\n")
			os.Exit(1)
		}
		cliIndex(cfg, cleanArgs[0], outPath)
	case "status":
		cliStatus(cfg, outPath)
	case "search":
		if len(cleanArgs) < 1 {
			fmt.Fprintf(os.Stderr, "Usage: openviking-mcp search <query> [--out /path/to/db/dir]\n")
			os.Exit(1)
		}
		cliSearch(cfg, cleanArgs[0], outPath)
	case "help", "--help", "-h":
		fmt.Println("openviking-mcp — local semantic code search")
		fmt.Println()
		fmt.Println("CLI usage:")
		fmt.Println("  openviking-mcp index <path> [--out <dir>]   Index a directory")
		fmt.Println("  openviking-mcp status [--out <dir>]         Show index stats")
		fmt.Println("  openviking-mcp search <query> [--out <dir>] Search indexed files")
		fmt.Println()
		fmt.Println("Options:")
		fmt.Println("  --out, -o <dir>  Where to store the database (default: <path>/.viking_db/)")
		fmt.Println("                   The DB is a single file: <dir>/vectors.db")
		fmt.Println()
		fmt.Println("MCP usage (no args):")
		fmt.Println("  claude mcp add openviking /path/to/openviking-mcp")
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\nRun: openviking-mcp help\n", cmd)
		os.Exit(1)
	}
}

func cliIndex(cfg Config, path string, dbPath string) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid path: %v\n", err)
		os.Exit(1)
	}
	info, err := os.Stat(absPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Path does not exist: %s\n", absPath)
		os.Exit(1)
	}
	if !info.IsDir() {
		fmt.Fprintf(os.Stderr, "Not a directory: %s\n", absPath)
		os.Exit(1)
	}

	ctx := context.Background()
	client := openviking.NewOllamaClient(cfg.OllamaEndpoint)
	fmt.Printf("Connecting to Ollama at %s...\n", cfg.OllamaEndpoint)
	if err := client.Ping(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "Ollama not reachable: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Ollama: connected")

	dbDir := dbPath
	if dbDir == "" {
		dbDir = filepath.Join(absPath, ".viking_db")
	}
	store, err := openviking.OpenStore(dbDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Store error: %v\n", err)
		os.Exit(1)
	}
	defer store.Close()

	embedder := openviking.NewOllamaEmbedder(client, cfg.Model)
	indexer := openviking.NewIndexer(absPath, embedder, store, openviking.ChunkerOpts{
		MaxChunkSize: 1500,
		ContextDepth: cfg.ContextDepth,
		ExcludeGlobs: cfg.ExcludePatterns,
	})

	startTime := time.Now()
	fmt.Printf("Indexing %s...\n", absPath)
	fmt.Printf("Started:  %s\n", startTime.Format("2006-01-02 15:04:05"))
	fmt.Printf("Database: %s\n\n", dbDir)

	// Use async with progress so we get live updates.
	// Incremental: skips files already indexed with same modtime.
	// If you stop and restart, it picks up where it left off.
	progressCh := make(chan openviking.IndexProgress, 64)
	indexer.IndexProjectAsync(ctx, progressCh)

	lastPrint := time.Now()
	for p := range progressCh {
		if p.Done {
			if p.Err != nil {
				fmt.Fprintf(os.Stderr, "\nIndexing failed: %v\n", p.Err)
				os.Exit(1)
			}
			r := p.Result
			elapsed := time.Since(startTime).Round(time.Second)
			fmt.Printf("\n\nDone.\n")
			fmt.Printf("  Started:  %s\n", startTime.Format("2006-01-02 15:04:05"))
			fmt.Printf("  Finished: %s\n", time.Now().Format("2006-01-02 15:04:05"))
			fmt.Printf("  Elapsed:  %s\n", elapsed)
			fmt.Printf("  Scanned:  %d files\n", r.FilesScanned)
			fmt.Printf("  Indexed:  %d files\n", r.FilesIndexed)
			fmt.Printf("  Skipped:  %d files (unchanged)\n", r.FilesSkipped)
			fmt.Printf("  Chunks:   %d\n", r.ChunksCreated)
			fmt.Printf("  Database: %s\n", dbDir)
			return
		}
		// Live progress with elapsed time and ETA
		now := time.Now()
		if now.Sub(lastPrint) > 500*time.Millisecond || p.Current == p.Total {
			elapsed := now.Sub(startTime).Round(time.Second)
			eta := ""
			if p.Current > 0 {
				perFile := now.Sub(startTime) / time.Duration(p.Current)
				remaining := perFile * time.Duration(p.Total-p.Current)
				eta = fmt.Sprintf(" | ETA: %s", remaining.Round(time.Second))
			}
			fmt.Printf("\r  [%s] %d/%d files%s — %s", elapsed, p.Current, p.Total, eta, p.FilePath)
			// Clear rest of line in case previous filename was longer
			fmt.Printf("\033[K")
			lastPrint = now
		}
	}
}

func cliStatus(cfg Config, dbPath string) {
	ctx := context.Background()
	client := openviking.NewOllamaClient(cfg.OllamaEndpoint)

	fmt.Printf("Ollama: %s\n", cfg.OllamaEndpoint)
	if err := client.Ping(ctx); err != nil {
		fmt.Println("  Status: offline")
	} else {
		fmt.Println("  Status: running")
		models, err := client.ListModels(ctx)
		if err == nil {
			hasModel := false
			for _, m := range models {
				if m.Name == cfg.Model || (len(m.Name) > len(cfg.Model) && m.Name[:len(cfg.Model)] == cfg.Model) {
					hasModel = true
					break
				}
			}
			if hasModel {
				fmt.Printf("  Model: %s (available)\n", cfg.Model)
			} else {
				fmt.Printf("  Model: %s (not pulled)\n", cfg.Model)
			}
		}
	}

	statusDbDir := dbPath
	if statusDbDir == "" {
		cwd, _ := os.Getwd()
		statusDbDir = filepath.Join(cwd, ".viking_db")
	}
	if _, err := os.Stat(statusDbDir); err == nil {
		store, err := openviking.OpenStore(statusDbDir)
		if err == nil {
			defer store.Close()
			stats := store.Stats()
			fmt.Printf("\nIndex: %s\n", statusDbDir)
			fmt.Printf("  Files: %d\n", stats.TotalFiles)
			fmt.Printf("  Chunks: %d\n", stats.TotalRecords)
			if stats.LastModified > 0 {
				fmt.Printf("  Last indexed: %s\n", time.Unix(stats.LastModified, 0).Format("2006-01-02 15:04:05"))
			}
		}
	} else {
		fmt.Printf("\nNo index found at %s\n", statusDbDir)
	}
}

func cliSearch(cfg Config, query string, dbPath string) {
	ctx := context.Background()
	client := openviking.NewOllamaClient(cfg.OllamaEndpoint)
	if err := client.Ping(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "Ollama not reachable: %v\n", err)
		os.Exit(1)
	}

	searchDbDir := dbPath
	if searchDbDir == "" {
		cwd, _ := os.Getwd()
		searchDbDir = filepath.Join(cwd, ".viking_db")
	}
	store, err := openviking.OpenStore(searchDbDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Store error: %v\n", err)
		os.Exit(1)
	}
	defer store.Close()

	embedder := openviking.NewOllamaEmbedder(client, cfg.Model)
	retriever := openviking.NewRetriever(embedder, store, 5, cfg.MaxContextTokens)

	blocks, err := retriever.Retrieve(ctx, query)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Search error: %v\n", err)
		os.Exit(1)
	}

	if len(blocks) == 0 {
		fmt.Println("No results found.")
		return
	}

	for i, b := range blocks {
		fmt.Printf("\n--- %s (L%d-%d, score: %.2f) ---\n", b.FilePath, b.StartLine, b.EndLine, b.Score)
		fmt.Println(b.Content)
		if i < len(blocks)-1 {
			fmt.Println()
		}
	}
}

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
	default:
		return &jsonRPCResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &rpcError{Code: -32602, Message: "unknown tool: " + params.Name},
		}
	}

	return &jsonRPCResponse{JSONRPC: "2.0", ID: req.ID, Result: result}
}

func (s *server) toolSearch(args json.RawMessage) mcpToolResult {
	var input searchInput
	if err := json.Unmarshal(args, &input); err != nil {
		return errResult("invalid arguments: " + err.Error())
	}
	if input.Query == "" {
		return errResult("query is required")
	}
	if input.Limit <= 0 {
		input.Limit = 5
	}

	ctx := context.Background()
	client := openviking.NewOllamaClient(s.cfg.OllamaEndpoint)
	if err := client.Ping(ctx); err != nil {
		return errResult("Ollama not reachable at " + s.cfg.OllamaEndpoint + ": " + err.Error())
	}

	cwd, _ := os.Getwd()
	dbDir := filepath.Join(cwd, ".viking_db")
	if _, err := os.Stat(dbDir); err != nil {
		return errResult("No index found. Run index_project first.")
	}

	store, err := openviking.OpenStore(dbDir)
	if err != nil {
		return errResult("store error: " + err.Error())
	}
	defer store.Close()

	embedder := openviking.NewOllamaEmbedder(client, s.cfg.Model)
	retriever := openviking.NewRetriever(embedder, store, input.Limit, s.cfg.MaxContextTokens)

	blocks, err := retriever.Retrieve(ctx, input.Query)
	if err != nil {
		return errResult("search error: " + err.Error())
	}

	if len(blocks) == 0 {
		return textResult("No relevant results found.")
	}

	type result struct {
		File      string  `json:"file"`
		StartLine int     `json:"startLine"`
		EndLine   int     `json:"endLine"`
		Content   string  `json:"content"`
		Score     float64 `json:"score"`
	}
	var results []result
	for _, b := range blocks {
		results = append(results, result{
			File:      b.FilePath,
			StartLine: b.StartLine,
			EndLine:   b.EndLine,
			Content:   b.Content,
			Score:     b.Score,
		})
	}

	out, _ := json.MarshalIndent(results, "", "  ")
	return textResult(string(out))
}

func (s *server) toolIndex(args json.RawMessage) mcpToolResult {
	var input indexInput
	if err := json.Unmarshal(args, &input); err != nil {
		return errResult("invalid arguments: " + err.Error())
	}
	if input.Path == "" {
		return errResult("path is required")
	}

	absPath, err := filepath.Abs(input.Path)
	if err != nil {
		return errResult("invalid path: " + err.Error())
	}
	info, err := os.Stat(absPath)
	if err != nil {
		return errResult("path does not exist: " + absPath)
	}
	if !info.IsDir() {
		return errResult("path is not a directory: " + absPath)
	}

	s.index.mu.Lock()
	if s.index.running {
		s.index.mu.Unlock()
		return textResult(fmt.Sprintf("Indexing already in progress: %s (%d/%d files)\nUse openviking_status to check progress.",
			s.index.path, s.index.current, s.index.total))
	}

	if s.index.result != nil && s.index.path == absPath {
		r := s.index.result
		e := s.index.err
		s.index.result = nil
		s.index.err = nil
		s.index.mu.Unlock()

		if e != nil {
			return errResult("previous indexing failed: " + e.Error())
		}
		out, _ := json.MarshalIndent(map[string]any{
			"filesScanned":  r.FilesScanned,
			"filesIndexed":  r.FilesIndexed,
			"chunksCreated": r.ChunksCreated,
			"duration":      r.Duration.Round(time.Millisecond).String(),
		}, "", "  ")
		return textResult(string(out))
	}

	ctx, cancel := context.WithTimeout(context.Background(), 24*time.Hour)
	s.index.running = true
	s.index.path = absPath
	s.index.current = 0
	s.index.total = 0
	s.index.startedAt = time.Now()
	s.index.lastUpdate = time.Now()
	s.index.result = nil
	s.index.err = nil
	s.index.cancel = cancel
	s.index.mu.Unlock()

	go s.runIndex(ctx, absPath)

	return textResult(fmt.Sprintf("Indexing started in background: %s\nUse openviking_status to check progress. Call index_project again when done to get results.", absPath))
}

func (s *server) runIndex(ctx context.Context, absPath string) {
	defer func() {
		s.index.mu.Lock()
		if s.index.cancel != nil {
			s.index.cancel()
			s.index.cancel = nil
		}
		s.index.mu.Unlock()
	}()

	client := openviking.NewOllamaClient(s.cfg.OllamaEndpoint)

	if err := client.Ping(ctx); err != nil {
		s.index.mu.Lock()
		s.index.running = false
		s.index.err = fmt.Errorf("Ollama not reachable: %v", err)
		s.index.mu.Unlock()
		return
	}

	cwd, _ := os.Getwd()
	dbDir := filepath.Join(cwd, ".viking_db")
	store, err := openviking.OpenStore(dbDir)
	if err != nil {
		s.index.mu.Lock()
		s.index.running = false
		s.index.err = err
		s.index.mu.Unlock()
		return
	}
	defer store.Close()

	embedder := openviking.NewOllamaEmbedder(client, s.cfg.Model)
	indexer := openviking.NewIndexer(absPath, embedder, store, openviking.ChunkerOpts{
		MaxChunkSize: 1500,
		ContextDepth: s.cfg.ContextDepth,
		ExcludeGlobs: s.cfg.ExcludePatterns,
	})

	progressCh := make(chan openviking.IndexProgress, 64)
	indexer.IndexProjectAsync(ctx, progressCh)

	const stallTimeout = 30 * time.Minute

	for {
		select {
		case p, ok := <-progressCh:
			if !ok {
				return
			}
			s.index.mu.Lock()
			s.index.current = p.Current
			s.index.total = p.Total
			s.index.lastUpdate = time.Now()
			if p.Done {
				s.index.running = false
				s.index.result = p.Result
				s.index.err = p.Err
			}
			s.index.mu.Unlock()
			if p.Done {
				return
			}

		case <-time.After(stallTimeout):
			s.index.mu.Lock()
			elapsed := time.Since(s.index.lastUpdate)
			s.index.mu.Unlock()

			if elapsed >= stallTimeout {
				log.Printf("indexing stalled for %v, cancelling", elapsed)
				s.index.mu.Lock()
				s.index.running = false
				s.index.err = fmt.Errorf("indexing stalled: no progress for %v at file %d/%d", elapsed, s.index.current, s.index.total)
				s.index.mu.Unlock()
				return
			}

		case <-ctx.Done():
			s.index.mu.Lock()
			s.index.running = false
			s.index.err = fmt.Errorf("indexing timed out after %v", time.Since(s.index.startedAt).Round(time.Second))
			s.index.mu.Unlock()
			return
		}
	}
}

func (s *server) toolStatus() mcpToolResult {
	ctx := context.Background()
	endpoint := s.cfg.OllamaEndpoint
	model := s.cfg.Model

	status := map[string]any{
		"endpoint":        endpoint,
		"model":           model,
		"excludePatterns": s.cfg.ExcludePatterns,
	}

	client := openviking.NewOllamaClient(endpoint)
	if err := client.Ping(ctx); err != nil {
		status["ollamaRunning"] = false
	} else {
		status["ollamaRunning"] = true
		models, err := client.ListModels(ctx)
		if err == nil {
			hasModel := false
			for _, m := range models {
				if m.Name == model || len(m.Name) > len(model) && m.Name[:len(model)] == model {
					hasModel = true
					break
				}
			}
			status["modelAvailable"] = hasModel
		}
	}

	cwd, _ := os.Getwd()
	dbDir := filepath.Join(cwd, ".viking_db")
	if _, err := os.Stat(dbDir); err == nil {
		store, err := openviking.OpenStore(dbDir)
		if err == nil {
			defer store.Close()
			stats := store.Stats()
			status["indexedFiles"] = stats.TotalFiles
			status["totalChunks"] = stats.TotalRecords
			if stats.LastModified > 0 {
				status["lastIndexed"] = time.Unix(stats.LastModified, 0).Format("2006-01-02 15:04:05")
			} else {
				status["lastIndexed"] = "never"
			}
		}
	} else {
		status["indexedFiles"] = 0
		status["totalChunks"] = 0
		status["lastIndexed"] = "no index"
	}

	// Indexing state
	s.index.mu.Lock()
	if s.index.running {
		status["indexing"] = true
		status["indexingPath"] = s.index.path
		status["indexingProgress"] = fmt.Sprintf("%d/%d files", s.index.current, s.index.total)
		status["indexingElapsed"] = time.Since(s.index.startedAt).Round(time.Second).String()
		sinceUpdate := time.Since(s.index.lastUpdate).Round(time.Second)
		status["timeSinceLastProgress"] = sinceUpdate.String()
		if sinceUpdate > 2*time.Minute {
			status["warning"] = "indexing may be stalled — no progress for " + sinceUpdate.String()
		}
	} else if s.index.result != nil {
		status["indexing"] = false
		status["lastIndexResult"] = map[string]any{
			"path":          s.index.path,
			"filesScanned":  s.index.result.FilesScanned,
			"filesIndexed":  s.index.result.FilesIndexed,
			"chunksCreated": s.index.result.ChunksCreated,
			"duration":      s.index.result.Duration.Round(time.Millisecond).String(),
		}
		if s.index.err != nil {
			status["lastIndexError"] = s.index.err.Error()
		}
	} else if s.index.err != nil {
		status["indexing"] = false
		status["lastIndexError"] = s.index.err.Error()
		status["lastIndexPath"] = s.index.path
	}
	s.index.mu.Unlock()

	out, _ := json.MarshalIndent(status, "", "  ")
	return textResult(string(out))
}

func textResult(text string) mcpToolResult {
	return mcpToolResult{Content: []mcpContent{{Type: "text", Text: text}}}
}

func errResult(msg string) mcpToolResult {
	return mcpToolResult{Content: []mcpContent{{Type: "text", Text: msg}}, IsError: true}
}

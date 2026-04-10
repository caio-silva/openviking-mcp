package main

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// Config holds the OpenViking MCP server configuration.
type Config struct {
	OllamaEndpoint   string   `json:"ollamaEndpoint"`
	Model            string   `json:"model"`
	ContextDepth     int      `json:"contextDepth"`
	MaxContextTokens int      `json:"maxContextTokens"`
	ExcludePatterns  []string `json:"excludePatterns"`
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		OllamaEndpoint:   "http://localhost:11434",
		Model:            "bge-m3",
		ContextDepth:     1,
		MaxContextTokens: 4096,
		ExcludePatterns:  []string{".git", "node_modules", "vendor", ".viking_db", "__pycache__", ".idea"},
	}
}

// LoadConfig loads config from the resolved path, merging with defaults.
func LoadConfig() Config {
	cfg := DefaultConfig()
	path := resolveConfigPath()
	if path == "" {
		return cfg
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return cfg
	}

	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg
	}

	// Re-apply defaults for zero values
	defaults := DefaultConfig()
	if cfg.OllamaEndpoint == "" {
		cfg.OllamaEndpoint = defaults.OllamaEndpoint
	}
	if cfg.Model == "" {
		cfg.Model = defaults.Model
	}
	if cfg.MaxContextTokens == 0 {
		cfg.MaxContextTokens = defaults.MaxContextTokens
	}
	if len(cfg.ExcludePatterns) == 0 {
		cfg.ExcludePatterns = defaults.ExcludePatterns
	}

	return cfg
}

// resolveConfigPath finds the config file using this precedence:
// 1. $OPENVIKING_MCP_CONFIG env var
// 2. $XDG_CONFIG_HOME/openviking-mcp/config.json
// 3. ~/.config/openviking-mcp/config.json
func resolveConfigPath() string {
	if envPath := os.Getenv("OPENVIKING_MCP_CONFIG"); envPath != "" {
		return envPath
	}
	xdgConfig := os.Getenv("XDG_CONFIG_HOME")
	if xdgConfig == "" {
		home, _ := os.UserHomeDir()
		xdgConfig = filepath.Join(home, ".config")
	}
	path := filepath.Join(xdgConfig, "openviking-mcp", "config.json")
	if _, err := os.Stat(path); err == nil {
		return path
	}
	return ""
}

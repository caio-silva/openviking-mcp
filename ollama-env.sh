# Ollama environment — optimized for embedding workloads
# Usage: source ollama-env.sh && ollama serve
# Or:    source ollama-env.sh && brew services restart ollama
export OLLAMA_NUM_THREADS=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_FLASH_ATTENTION=1

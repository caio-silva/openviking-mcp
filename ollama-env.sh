# Ollama environment — optimized for embedding workloads
# Usage: source ollama-env.sh && ollama serve
# Or:    source ollama-env.sh && brew services restart ollama
export OLLAMA_NUM_THREADS=$(nproc)
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_FLASH_ATTENTION=1

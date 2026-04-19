"""Application configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_BASE_URL = "http://localhost:11434"
# For lower latency use a smaller tag, e.g. "llama3.2:1b" or "phi3:mini" (after `ollama pull`).
OLLAMA_LLM_MODEL = "llama3.2"

# Fewer chunks = smaller prompt to Ollama = faster generation.
RETRIEVER_K = 3

# Trim each retrieved chunk before sending to the LLM (indexing still uses full chunks).
CONTEXT_CHARS_PER_CHUNK = 550

# Cap context window for Ollama (smaller often faster on CPU; raise if answers get cut off).
OLLAMA_NUM_CTX = 4096

# Keep model loaded between requests (reduces cold-start delay on the next message).
OLLAMA_KEEP_ALIVE = "15m"

# How many prior chat turns to include (lower = smaller prompt = faster).
CHAT_HISTORY_MAX_TURNS = 4

# Default False: query rewriting is a full extra Ollama call before retrieval (much slower).
QUERY_REWRITE_ENABLED = False
QUERY_REWRITE_TEMPERATURE = 0.2

# Lower = faster answers from Ollama (num_predict caps generated tokens).
OLLAMA_NUM_PREDICT_REWRITE = 48
OLLAMA_NUM_PREDICT_SIMPLE = 220
OLLAMA_NUM_PREDICT_DETAILED = 550

GENERATION_TEMPERATURE_SIMPLE = 0.3
GENERATION_TEMPERATURE_DETAILED = 0.5

FAISS_INDEX_NAME = "faiss_index"

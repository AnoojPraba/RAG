import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Directories
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# Embedding config
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval config
TOP_K = 5
MIN_SIMILARITY = 0.3

# LLM config
MODEL = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 1024
TEMPERATURE = 0.7

# Create directories if they don't exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

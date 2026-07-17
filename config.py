import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")

# Qdrant Vector DB Settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = "rag_documents"

# Directory Settings
UPLOAD_DIR = "uploads"
LOG_DIR = "logs"
DOCS_DIR = "docs"
TESTS_DIR = "tests"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Chunking Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Embedding Constants
EMBEDDING_MODEL = "gemini-embedding-001"

# Retrieval Constants
DEFAULT_TOP_K = 3

# LLM Models Configuration
MODEL_GEMINI = "Gemini Flash"
MODEL_QWEN = "Qwen 2.5"

LLM_MODELS = [MODEL_GEMINI, MODEL_QWEN]

GEMINI_LLM_MODEL = "gemini-flash-latest"
QWEN_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

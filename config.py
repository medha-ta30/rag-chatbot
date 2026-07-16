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

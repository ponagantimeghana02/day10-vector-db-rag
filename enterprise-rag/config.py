# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# API Keys
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# Models
# =========================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.1-8b-instant"

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# =========================
# Vector Database
# =========================

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "enterprise_rag"

# =========================
# Chunk Settings
# =========================

CHUNK_SIZE = 500

CHUNK_OVERLAP = 50

# =========================
# Retrieval
# =========================

TOP_K = 5

RERANK_TOP_K = 3

# =========================
# Cache
# =========================

ENABLE_CACHE = True

# =========================
# Chat History
# =========================

CHAT_HISTORY_FILE = "./chat_history/history.json"

# =========================
# User Roles
# =========================

USER_ROLES = {

    "admin": {
        "access": "all"
    },

    "employee": {
        "access": "employee"
    },

    "guest": {
        "access": "public"
    }
}
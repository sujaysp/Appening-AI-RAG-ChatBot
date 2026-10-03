import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "Ebook-Agentic-AI.pdf"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-local-index",
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

LLM_MODEL = "gemini-3.5-flash-lite"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 8

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in the .env file.")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")
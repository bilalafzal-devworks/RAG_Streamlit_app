"""Settings, secret loading, and the shared error type."""
import os

from dotenv import load_dotenv

load_dotenv()  # Reads a local .env file if present (does nothing on Streamlit Cloud)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#PREFERRED_GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
PREFERRED_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 4


class RAGError(Exception):
    """An error whose message is safe and clear enough to show to the user."""


def get_secret(name):
    """Find a secret in environment variables (.env) or Streamlit Secrets."""
    value = os.environ.get(name)
    if not value:
        try:
            import streamlit as st
            value = st.secrets.get(name)
        except Exception:
            value = None
    if not value:
        return None
    return str(value).strip().strip('"').strip("'") or None

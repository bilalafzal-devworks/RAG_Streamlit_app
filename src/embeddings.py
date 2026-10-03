"""Create embeddings with the Hugging Face Inference API."""
import numpy as np
from huggingface_hub import InferenceClient

from src.config import EMBED_MODEL, RAGError, get_secret


def _explain_error(error):
    status = getattr(getattr(error, "response", None), "status_code", None)
    text = str(error)
    if status == 401 or "401" in text:
        return "Hugging Face rejected your token (401). Check that HF_TOKEN is correct and not expired."
    if status == 403 or "403" in text:
        return "Hugging Face denied access (403). Your token needs permission to call Inference Providers."
    if status == 404 or "404" in text:
        return f"The embedding model '{EMBED_MODEL}' is not available right now (404)."
    if status == 429 or "429" in text:
        return "Hugging Face rate limit reached (429). Wait a minute and try again."
    if status in (500, 502, 503, 504):
        return "The Hugging Face embedding service is busy or loading. Try again in 30 seconds."
    return f"Embedding request failed: {text[:200]}"


def embed_texts(texts, batch_size=16):
    """Turn a list of strings into a list of vectors."""
    if not texts:
        return []

    token = get_secret("HF_TOKEN")
    if not token:
        raise RAGError("Hugging Face token (HF_TOKEN) is missing. Add it to .env or Streamlit Secrets.")

    client = InferenceClient(provider="hf-inference", api_key=token)
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            output = client.feature_extraction(batch, model=EMBED_MODEL)
        except Exception as e:
            raise RAGError(_explain_error(e)) from e

        arr = np.array(output, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 3:            # token-level vectors: average into one per text
            arr = arr.mean(axis=1)
        vectors.extend(arr.tolist())
    return vectors
"""Groq LLM client."""
from groq import APIConnectionError, APIStatusError, AuthenticationError, Groq, RateLimitError

from src.config import PREFERRED_GROQ_MODELS, RAGError, get_secret

_model_cache = {}


def _get_client():
    key = get_secret("GROQ_API_KEY")
    if not key:
        raise RAGError("Groq API key (GROQ_API_KEY) is missing. Add it to .env or Streamlit Secrets.")
    return Groq(api_key=key)


def _choose_model(client):
    override = get_secret("GROQ_MODEL")      # optional: lets you force a model without code changes
    if override:
        return override
    if "model" in _model_cache:
        return _model_cache["model"]

    try:
        available = {m.id for m in client.models.list().data}
    except AuthenticationError as e:
        raise RAGError("Groq rejected your API key (401). Check GROQ_API_KEY.") from e
    except Exception as e:
        raise RAGError(f"Could not check available Groq models: {str(e)[:200]}") from e

    for name in PREFERRED_GROQ_MODELS:
        if name in available:
            _model_cache["model"] = name
            return name
    raise RAGError(
        "None of the preferred Groq models are available anymore. Pick a chat model from "
        "console.groq.com/docs/models and set it as GROQ_MODEL in your secrets."
    )


def generate_answer(messages):
    client = _get_client()
    model = _choose_model(client)
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.1, max_tokens=500
        )
    except AuthenticationError as e:
        raise RAGError("Groq rejected your API key (401). Check GROQ_API_KEY.") from e
    except RateLimitError as e:
        raise RAGError("Groq rate limit reached (429). Wait a moment and try again.") from e
    except APIConnectionError as e:
        raise RAGError("Could not reach Groq. Check your internet connection and try again.") from e
    except APIStatusError as e:
        raise RAGError(
            f"Groq returned an error ({e.status_code}) for model '{model}'. "
            "If the model was retired, set GROQ_MODEL to another chat model."
        ) from e

    content = response.choices[0].message.content
    if not content:
        raise RAGError("Groq returned an empty answer. Please try again.")
    return content.strip()
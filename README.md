# 📄 Document Q&A with RAG

A Retrieval-Augmented Generation app: upload a PDF, TXT or DOCX file and ask questions about it.
Answers are generated from your document only, and the app shows which chunks it used.

## How it works

```
Upload → Extract text → Clean → Chunk → Embed (Hugging Face) → Store in ChromaDB
Question → Embed → Similarity search → Top-k chunks → Prompt → Groq LLM → Answer + sources
```

## Tech stack

Python · Streamlit · ChromaDB · Hugging Face Inference API (`all-MiniLM-L6-v2` embeddings) · Groq API (Llama models)

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/rag-streamlit-app.git
cd rag-streamlit-app
python -m venv .venv
# Windows: .venv\Scripts\activate    |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
# edit .env and add your real keys
streamlit run app.py
```

Get keys: [Groq](https://console.groq.com/keys) · [Hugging Face](https://huggingface.co/settings/tokens)

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (never commit `.env`).
2. Go to share.streamlit.io, create an app, choose this repo, branch `main`, main file `app.py`.
3. In **Advanced settings → Secrets**, paste:
```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   HF_TOKEN = "your_huggingface_token_here"
```
4. Deploy.

## Project structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `src/document_loader.py` | Read files, clean text, chunk |
| `src/embeddings.py` | Hugging Face embeddings |
| `src/vector_store.py` | ChromaDB storage and search |
| `src/llm.py` | Groq client |
| `src/rag.py` | Full RAG pipeline and prompt |
| `src/config.py` | Settings and secret loading |

## Notes and limitations

- The vector database is in-memory: documents are lost when the session ends.
- Scanned PDFs (images) are not supported, since there is no OCR.
- Answers are limited to the uploaded documents by design.

## Possible improvements

Reranking, hybrid search, conversation memory, persistent storage, OCR for scanned PDFs, retrieval evaluation.
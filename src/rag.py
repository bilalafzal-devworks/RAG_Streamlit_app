"""The RAG pipeline: index documents and answer questions."""
from src.document_loader import chunk_text, extract_text
from src.embeddings import embed_texts
from src.llm import generate_answer
from src.config import RAGError

NOT_FOUND = "I couldn't find that in the provided documents."

SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions using ONLY the provided context from the user's documents.

Rules:
1. Base your answer strictly on the context. Do not use outside knowledge.
2. If the context does not contain the answer, say exactly: "{NOT_FOUND}"
3. Never invent facts, numbers, names, or policies.
4. Keep answers concise and clear. If the answer combines several parts of the context, combine them.
5. When useful, mention which source number(s) you used, like [Source 2]."""


def index_document(store, filename, data):
    """Load -> clean -> chunk -> embed -> store. Returns the number of chunks."""
    text = extract_text(filename, data)
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    store.add_document(filename, chunks, vectors)
    return len(chunks)


def build_context(hits):
    return "\n\n".join(
        f"[Source {i}] ({h['metadata']['source']}, chunk {h['metadata']['chunk_index']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )


def build_messages(question, hits):
    user_prompt = (
        f"Context:\n{build_context(hits)}\n\n"
        f"Question: {question}\n\nAnswer using only the context above."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def answer_question(question, store, top_k=4, max_distance=None):
    """Embed question -> search Chroma -> build prompt -> ask Groq."""
    if not question or not question.strip():
        raise RAGError("Please type a question first.")
    question = question.strip()

    query_vector = embed_texts([question])[0]
    hits = store.query(query_vector, top_k=top_k)

    if max_distance is not None:
        hits = [h for h in hits if h["distance"] <= max_distance]
    if not hits:
        return {"answer": NOT_FOUND, "hits": []}

    answer = generate_answer(build_messages(question, hits))
    return {"answer": answer, "hits": hits}
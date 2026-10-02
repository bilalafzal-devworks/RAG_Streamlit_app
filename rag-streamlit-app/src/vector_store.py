"""A small wrapper around ChromaDB (in-memory, one private collection per session)."""
import uuid

import chromadb

from src.config import RAGError


class VectorStore:
    def __init__(self):
        try:
            self.client = chromadb.EphemeralClient()
            # Unique name so different visitors never share data
            self.name = f"docs_{uuid.uuid4().hex[:12]}"
            self.collection = self._create_collection()
        except Exception as e:
            raise RAGError(f"Could not start ChromaDB: {e}") from e

    def _create_collection(self):
        return self.client.create_collection(
            name=self.name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,   # we supply embeddings from Hugging Face
        )

    def count(self):
        return self.collection.count()

    def add_document(self, filename, chunks, embeddings):
        try:
            if self.count() > 0:
                self.collection.delete(where={"source": filename})  # replace if re-uploaded
            doc_id = uuid.uuid4().hex[:8]
            self.collection.add(
                ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
                documents=chunks,
                embeddings=embeddings,
                metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))],
            )
        except Exception as e:
            raise RAGError(f"Could not store the document in ChromaDB: {e}") from e

    def query(self, query_embedding, top_k=4):
        total = self.count()
        if total == 0:
            raise RAGError("The database is empty. Upload and process a document first.")
        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise RAGError(f"ChromaDB search failed: {e}") from e

        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def reset(self):
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        try:
            self.collection = self._create_collection()
        except Exception as e:
            raise RAGError(f"Could not reset ChromaDB: {e}") from e
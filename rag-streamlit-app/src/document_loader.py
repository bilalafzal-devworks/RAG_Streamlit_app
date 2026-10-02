"""Read uploaded files, clean the text, and split it into chunks."""
import io
import re
from pathlib import Path

from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, RAGError

SUPPORTED_TYPES = {".txt", ".pdf", ".docx"}


def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(filename, data):
    """Return cleaned text from the raw bytes of an uploaded file."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_TYPES:
        raise RAGError(f"Unsupported file type '{ext}'. Please upload a PDF, TXT or DOCX file.")

    try:
        if ext == ".txt":
            text = data.decode("utf-8", errors="ignore")
        elif ext == ".pdf":
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        else:
            doc = Document(io.BytesIO(data))
            text = "\n\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        raise RAGError(
            f"Could not read '{filename}'. The file may be corrupted or password-protected."
        ) from e

    text = clean_text(text)
    if not text:
        raise RAGError(
            f"'{filename}' contains no readable text. Scanned PDFs (images of text) are not supported."
        )
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [c for c in splitter.split_text(text) if c.strip()]
    if not chunks:
        raise RAGError("The document could not be split into chunks.")
    return chunks
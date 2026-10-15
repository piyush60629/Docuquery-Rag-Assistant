"""Core Retrieval-Augmented Generation logic.

Pipeline:
    DocumentPart -> LangChain RecursiveCharacterTextSplitter -> Gemini embeddings
    -> ChromaDB vector store -> similarity search -> LangChain (LCEL) answer chain
"""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import DocumentPart


DEFAULT_GENERATION_MODEL = "gemini-3.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768

COLLECTION_PREFIX = "docuquery-"
COLLECTION_MAX_AGE_SECONDS = 6 * 60 * 60  # drop indexes from sessions idle for 6 hours
EMBEDDING_BATCH_SIZE = 50  # keeps each Gemini embedding request small on the free tier

NOT_FOUND_MESSAGE = "I could not find that information in the uploaded documents."


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TextChunk:
    source: str
    text: str
    page: int | None
    chunk_id: int

    @property
    def label(self) -> str:
        page_label = f", page {self.page}" if self.page else ""
        return f"{self.source}{page_label}"

    def to_document(self) -> Document:
        # Chroma metadata values cannot be None, so the page key is only set when known.
        metadata: dict[str, Any] = {"source": self.source, "chunk_id": self.chunk_id}
        if self.page is not None:
            metadata["page"] = self.page
        return Document(page_content=self.text, metadata=metadata)

    @classmethod
    def from_document(cls, document: Document) -> "TextChunk":
        metadata = document.metadata or {}
        return cls(
            source=str(metadata.get("source", "unknown")),
            text=document.page_content,
            page=metadata.get("page"),
            chunk_id=int(metadata.get("chunk_id", 0)),
        )


@dataclass(frozen=True)
class SearchResult:
    chunk: TextChunk
    score: float  # cosine similarity: 1.0 = identical meaning, 0.0 = unrelated


# --------------------------------------------------------------------------- #
# Chunking (LangChain text splitter)
# --------------------------------------------------------------------------- #


def clean_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def make_text_splitter(chunk_size: int = 900, overlap: int = 150) -> RecursiveCharacterTextSplitter:
    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    # Try to split on paragraphs first, then lines, sentences, clauses and words.
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        keep_separator="end",
        strip_whitespace=True,
    )


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks using LangChain's recursive splitter."""

    splitter = make_text_splitter(chunk_size, overlap)
    normalized = clean_text(text)
    if not normalized:
        return []
    return [chunk for chunk in splitter.split_text(normalized) if chunk.strip()]


def build_chunks(
    parts: Iterable[DocumentPart], chunk_size: int = 900, overlap: int = 150
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    chunk_id = 1
    for part in parts:
        for text in chunk_text(part.text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(TextChunk(source=part.source, text=text, page=part.page, chunk_id=chunk_id))
            chunk_id += 1
    return chunks


# --------------------------------------------------------------------------- #
# ChromaDB vector store
# --------------------------------------------------------------------------- #


def _ensure_modern_sqlite() -> None:
    """ChromaDB needs SQLite >= 3.35.

    Some Linux hosts (including older Streamlit Community Cloud images) ship an older
    SQLite. When that happens, swap in the bundled `pysqlite3-binary` build before
    ChromaDB is imported.
    """

    if sqlite3.sqlite_version_info >= (3, 35, 0):
        return
    try:
        import pysqlite3  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            f"ChromaDB needs SQLite 3.35+, but this system has {sqlite3.sqlite_version}. "
            "Install pysqlite3-binary (already listed in requirements.txt for Linux)."
        ) from exc
    sys.modules["sqlite3"] = pysqlite3


_client_lock = threading.Lock()
_chroma_client: Any = None


def get_chroma_client() -> Any:
    """One in-memory Chroma client per server process, shared by all sessions.

    Each Streamlit session gets its own uniquely named collection, so users never
    see each other's documents. Nothing is written to disk, which suits Streamlit
    Community Cloud's temporary file system.
    """

    global _chroma_client
    with _client_lock:
        if _chroma_client is None:
            _ensure_modern_sqlite()
            import chromadb
            from chromadb.config import Settings

            _chroma_client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False, allow_reset=False)
            )
        return _chroma_client


def prune_stale_collections(client: Any, max_age_seconds: int = COLLECTION_MAX_AGE_SECONDS) -> int:
    """Delete DocuQuery collections left behind by sessions that have ended."""

    now = time.time()
    removed = 0
    for collection in client.list_collections():
        name = getattr(collection, "name", collection)
        if not str(name).startswith(COLLECTION_PREFIX):
            continue
        metadata = getattr(collection, "metadata", None) or {}
        created_at = float(metadata.get("created_at", now))
        if now - created_at > max_age_seconds:
            try:
                client.delete_collection(name=name)
                removed += 1
            except Exception:
                pass
    return removed


class ChromaIndex:
    """A per-session ChromaDB collection wrapped in LangChain's Chroma vector store."""

    def __init__(self, embeddings: Embeddings, client: Any = None) -> None:
        from langchain_chroma import Chroma

        self.client = client or get_chroma_client()
        prune_stale_collections(self.client)
        self.collection_name = f"{COLLECTION_PREFIX}{uuid.uuid4().hex}"
        self.store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=embeddings,
            # Cosine distance, so similarity = 1 - distance (same scale as before).
            collection_metadata={"hnsw:space": "cosine", "created_at": time.time()},
        )

    def add_chunks(self, chunks: Sequence[TextChunk], batch_size: int = EMBEDDING_BATCH_SIZE) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.store.add_documents(
                [chunk.to_document() for chunk in batch],
                ids=[f"chunk-{chunk.chunk_id}" for chunk in batch],
            )

    def count(self) -> int:
        return self.store._collection.count()

    def search(self, question: str, top_k: int = 4) -> list[SearchResult]:
        total = self.count()
        if total == 0:
            return []
        k = max(1, min(top_k, total))
        matches = self.store.similarity_search_with_score(question, k=k)
        results = [
            SearchResult(chunk=TextChunk.from_document(document), score=1.0 - float(distance))
            for document, distance in matches
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)

    def delete(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Retrieval + generation
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are a document question-answering assistant. Answer only from the "
    "provided context. Do not use outside knowledge. If the context does not "
    f"contain the answer, say: '{NOT_FOUND_MESSAGE}' Keep the language clear and "
    "practical. Cite supporting passages using [Source 1], [Source 2], and so on."
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
    ]
)


def format_context(results: Sequence[SearchResult]) -> str:
    return "\n\n".join(
        f"[Source {index}: {result.chunk.label}]\n{result.chunk.text}"
        for index, result in enumerate(results, start=1)
    )


class GeminiRAG:
    """RAG service: Gemini embeddings + ChromaDB retrieval + a LangChain answer chain.

    `embeddings` and `llm` can be injected (for tests or other providers); otherwise
    Gemini models are created from the API key.
    """

    def __init__(
        self,
        api_key: str = "",
        generation_model: str = DEFAULT_GENERATION_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embeddings: Embeddings | None = None,
        llm: BaseChatModel | None = None,
        max_retries: int = 6,
        request_timeout: float | None = None,
    ) -> None:
        if (embeddings is None or llm is None) and not api_key:
            raise ValueError("A Gemini API key is required")

        if embeddings is None or llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        self.embeddings = embeddings or GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=api_key,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        )
        self.llm = llm or ChatGoogleGenerativeAI(
            model=generation_model,
            google_api_key=api_key,
            max_output_tokens=700,
            max_retries=max_retries,
            timeout=request_timeout,
        )
        self.answer_chain = ANSWER_PROMPT | self.llm | StrOutputParser()

    def build_index(self, chunks: Sequence[TextChunk], client: Any = None) -> ChromaIndex:
        index = ChromaIndex(self.embeddings, client=client)
        try:
            index.add_chunks(chunks)
        except Exception:
            index.delete()
            raise
        return index

    def retrieve(self, index: ChromaIndex, question: str, top_k: int = 4) -> list[SearchResult]:
        return index.search(question, top_k=top_k)

    def answer(self, question: str, results: Sequence[SearchResult]) -> str:
        if not results:
            return NOT_FOUND_MESSAGE
        text = self.answer_chain.invoke({"context": format_context(results), "question": question})
        return (text or "").strip() or "I could not generate an answer from the uploaded documents."

"""Core Retrieval-Augmented Generation logic.

Pipeline:
    DocumentPart -> section-aware chunking (headings + LangChain RecursiveCharacterTextSplitter)
    -> Gemini embeddings -> ChromaDB vector store -> similarity search
    -> LangChain (LCEL) answer chain on an open-weight LLM (gpt-oss-120b) via Groq, or Gemini
"""

from __future__ import annotations

import re
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

# Answers can come from Groq (open-source models, generous free tier) or Gemini.
# Embeddings always use Gemini, so retrieval results stay comparable.
LLM_PROVIDERS = ("groq", "gemini")
# Groq retires models regularly; see https://console.groq.com/docs/deprecations
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"  # OpenAI's open-weight model
DEFAULT_GROQ_JUDGE_MODEL = "qwen/qwen3.8-27b"  # a different model family grades the answers
REASONING_TOKEN_HEADROOM = 800  # reasoning models spend some output tokens thinking first
EMBEDDING_DIMENSIONS = 768

COLLECTION_PREFIX = "docuquery-"
COLLECTION_MAX_AGE_SECONDS = 6 * 60 * 60  # drop indexes from sessions idle for 6 hours
EMBEDDING_BATCH_SIZE = 50  # keeps each Gemini embedding request small on the free tier

NOT_FOUND_MESSAGE = "I could not find that information in the uploaded documents."

CHUNKING_STRATEGIES = ("sections", "fixed")
DEFAULT_CHUNKING = "sections"


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TextChunk:
    source: str
    text: str
    page: int | None
    chunk_id: int
    section: str | None = None  # heading path, e.g. "Data Quality Standards > Thresholds"

    @property
    def label(self) -> str:
        page_label = f", page {self.page}" if self.page else ""
        return f"{self.source}{page_label}"

    def to_document(self) -> Document:
        # Chroma metadata values cannot be None, so the page key is only set when known.
        metadata: dict[str, Any] = {"source": self.source, "chunk_id": self.chunk_id}
        if self.page is not None:
            metadata["page"] = self.page
        if self.section:
            metadata["section"] = self.section
        return Document(page_content=self.text, metadata=metadata)

    @classmethod
    def from_document(cls, document: Document) -> "TextChunk":
        metadata = document.metadata or {}
        return cls(
            source=str(metadata.get("source", "unknown")),
            text=document.page_content,
            page=metadata.get("page"),
            chunk_id=int(metadata.get("chunk_id", 0)),
            section=metadata.get("section"),
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


# --------------------------------------------------------------------------- #
# Section-aware chunking
# --------------------------------------------------------------------------- #

MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
MIN_HEADER_BUDGET = 200  # never let a long heading path squeeze the body below this


def _looks_like_plain_heading(line: str) -> bool:
    """A short title-like line such as 'Bronze Layer' or 'Failure Handling'."""

    text = line.strip()
    if not text or len(text) > 60 or len(text.split()) > 8:
        return False
    if text[0] in "-*•>|" or text[-1] in ".,;:!?)\"'":
        return False
    return text[0].isupper() or text[0].isdigit()


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (heading path, body) sections.

    Recognises Markdown headings (# / ## / ###, nested into a path) and plain-text
    headings: a short title-like line after a blank line, followed by text. A short
    first line followed by a blank line is treated as the document title.
    Text without headings comes back as one section with an empty heading path.
    """

    lines = [" ".join(line.split()) for line in text.splitlines()]
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []

    def flush() -> None:
        content = re.sub(r"\n{3,}", "\n\n", "\n".join(body)).strip()
        if content:
            sections.append((" > ".join(title for _, title in stack), content))
        body.clear()

    for index, line in enumerate(lines):
        previous_blank = index == 0 or not lines[index - 1]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        level, title = 0, ""

        markdown = MARKDOWN_HEADING.match(line)
        if markdown:
            level, title = len(markdown.group(1)), markdown.group(2).strip()
        elif _looks_like_plain_heading(line) and previous_blank:
            if next_line:
                level, title = 2, line
            elif index == 0:
                level, title = 1, line  # document title

        if title:
            flush()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body.append(line)

    flush()
    return sections


def section_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> list[tuple[str, str]]:
    """Chunk each section separately so no chunk mixes two topics.

    Every chunk starts with its heading path, which gives the embedding (and the LLM)
    the topic even when the body text never repeats it. Returns (section, chunk text).
    """

    results: list[tuple[str, str]] = []
    for heading, body in split_sections(text):
        header = f"{heading}\n" if heading else ""
        budget = max(MIN_HEADER_BUDGET, chunk_size - len(header))
        splitter = make_text_splitter(budget, min(overlap, budget // 2))
        for piece in splitter.split_text(body):
            if piece.strip():
                results.append((heading, f"{header}{piece.strip()}"))
    return results


def build_chunks(
    parts: Iterable[DocumentPart],
    chunk_size: int = 900,
    overlap: int = 150,
    strategy: str = DEFAULT_CHUNKING,
) -> list[TextChunk]:
    """Turn extracted document parts into chunks.

    strategy="sections": split on headings first, then by size (default).
    strategy="fixed":    size-based splitting only (the original approach, kept for comparison).
    """

    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"strategy must be one of {CHUNKING_STRATEGIES}")
    make_text_splitter(chunk_size, overlap)  # validate settings once, for both strategies

    chunks: list[TextChunk] = []
    chunk_id = 1
    for part in parts:
        if strategy == "fixed":
            pieces = [("", text) for text in chunk_text(part.text, chunk_size=chunk_size, overlap=overlap)]
        else:
            pieces = section_chunks(part.text, chunk_size=chunk_size, overlap=overlap)
        for section, text in pieces:
            chunks.append(
                TextChunk(source=part.source, text=text, page=part.page, chunk_id=chunk_id, section=section or None)
            )
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
# Chat models
# --------------------------------------------------------------------------- #


def default_model(provider: str) -> str:
    return DEFAULT_GROQ_MODEL if provider == "groq" else DEFAULT_GENERATION_MODEL


def resolve_llm_provider(explicit: str | None = None, groq_api_key: str = "") -> str:
    """Use the provider that was asked for; otherwise Groq when its key is set, else Gemini."""

    if explicit:
        provider = explicit.strip().lower()
        if provider not in LLM_PROVIDERS:
            raise ValueError(f"LLM provider must be one of {LLM_PROVIDERS}")
        return provider
    return "groq" if groq_api_key else "gemini"


def make_chat_model(
    provider: str,
    api_key: str,
    model: str | None = None,
    *,
    max_tokens: int = 700,
    temperature: float | None = None,
    max_retries: int = 6,
    timeout: float | None = None,
) -> BaseChatModel:
    """Create a LangChain chat model for Groq or Gemini with the same settings."""

    if not api_key:
        raise ValueError(f"A {provider.title()} API key is required")
    model = model or default_model(provider)
    if provider == "groq":
        from langchain_groq import ChatGroq

        options: dict[str, Any] = {}
        if model.startswith("openai/gpt-oss"):
            options["reasoning_effort"] = "low"  # keep thinking short; reasoning goes to a separate field
            max_tokens += REASONING_TOKEN_HEADROOM
        elif model.startswith("qwen/"):
            options["reasoning_format"] = "hidden"  # return only the final answer, no <think> text
            max_tokens += REASONING_TOKEN_HEADROOM
        if temperature is not None:
            options["temperature"] = temperature
        return ChatGroq(
            model=model, api_key=api_key, max_tokens=max_tokens, max_retries=max_retries, timeout=timeout, **options
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        options = {"temperature": temperature} if temperature is not None else {}
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            max_output_tokens=max_tokens,
            max_retries=max_retries,
            timeout=timeout,
            **options,
        )
    raise ValueError(f"LLM provider must be one of {LLM_PROVIDERS}")


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

    `api_key` is the Gemini key (embeddings, and answers when llm_provider="gemini").
    With llm_provider="groq", answers come from an open-source model on Groq using
    `llm_api_key`. `embeddings` and `llm` can be injected for tests.
    """

    def __init__(
        self,
        api_key: str = "",
        generation_model: str | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embeddings: Embeddings | None = None,
        llm: BaseChatModel | None = None,
        max_retries: int = 6,
        request_timeout: float | None = None,
        llm_provider: str = "gemini",
        llm_api_key: str = "",
    ) -> None:
        self.llm_provider = resolve_llm_provider(llm_provider)
        self.generation_model = generation_model or default_model(self.llm_provider)

        if embeddings is None:
            if not api_key:
                raise ValueError("A Gemini API key is required")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embeddings = GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=api_key,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            )
        self.embeddings = embeddings
        self.llm = llm or make_chat_model(
            self.llm_provider,
            llm_api_key if self.llm_provider == "groq" else api_key,
            self.generation_model,
            max_tokens=700,
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

"""REST API for DocuQuery, built with FastAPI.

Run locally:   uvicorn api:app --reload        (interactive docs at http://localhost:8000/docs)
Run in Docker: docker compose up api

Flow:
    1. POST /indexes (upload files) or POST /indexes/demo  -> returns an index_id
    2. POST /indexes/{index_id}/query with a question       -> grounded answer + cited sources
    3. DELETE /indexes/{index_id} when finished (idle indexes also expire automatically)

Environment variables:
    GEMINI_API_KEY          required (embeddings; also answers when LLM_PROVIDER=gemini)
    GROQ_API_KEY            optional; when set, answers come from an open-source model on Groq
    LLM_PROVIDER            optional; "groq" or "gemini" to force the answer provider
    DOCUQUERY_API_TOKEN     optional; when set, every request except /health needs header X-API-Key
    DOCUQUERY_INDEX_TTL     seconds an unused index is kept (default 3600)
    DOCUQUERY_MAX_INDEXES   maximum indexes kept in memory (default 20, oldest are dropped first)
"""

from __future__ import annotations

import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from document_loader import (
    NamedBytesIO,
    UnsupportedFileTypeError,
    load_many,
    open_paths,
    supported_files_in,
)
from rag_engine import (
    CHUNKING_STRATEGIES,
    DEFAULT_CHUNKING,
    ChromaIndex,
    GeminiRAG,
    build_chunks,
    default_model,
    resolve_llm_provider,
)

API_VERSION = "1.0.0"
MAX_FILES = 5
MAX_TOTAL_BYTES = 15 * 1024 * 1024
DEMO_DIR = Path(__file__).resolve().parent / "sample_documents"

ChunkingName = Literal["sections", "fixed"]
assert set(CHUNKING_STRATEGIES) == {"sections", "fixed"}


# --------------------------------------------------------------------------- #
# Response and request models (these also drive the OpenAPI docs at /docs)
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = API_VERSION
    gemini_configured: bool
    llm_provider: str
    llm_model: str
    auth_required: bool
    indexes: int


class IndexInfo(BaseModel):
    index_id: str
    documents: list[str]
    chunks: int
    chunking: ChunkingName
    chunk_size: int
    overlap: int
    expires_in_seconds: int


class DemoIndexRequest(BaseModel):
    chunking: ChunkingName = DEFAULT_CHUNKING
    chunk_size: int = Field(900, ge=200, le=4000)
    overlap: int = Field(150, ge=0, le=1000)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, examples=["What happens when a pipeline fails?"])
    top_k: int = Field(4, ge=1, le=10)


class Source(BaseModel):
    rank: int
    source: str
    page: int | None = None
    section: str | None = None
    score: float = Field(..., description="Cosine similarity, 1.0 = most similar")
    text: str


class QueryResponse(BaseModel):
    index_id: str
    question: str
    answer: str
    sources: list[Source]
    retrieval_ms: int
    generation_ms: int


# --------------------------------------------------------------------------- #
# In-memory index store with expiry
# --------------------------------------------------------------------------- #


@dataclass
class IndexEntry:
    index: ChromaIndex
    documents: list[str]
    chunks: int
    chunking: str
    chunk_size: int
    overlap: int
    last_used: float = field(default_factory=time.monotonic)


class IndexStore:
    """Keeps each uploaded document set as its own ChromaDB collection."""

    def __init__(self, ttl_seconds: int, max_indexes: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_indexes = max_indexes
        self._entries: dict[str, IndexEntry] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _drop(self, index_id: str) -> None:
        entry = self._entries.pop(index_id, None)
        if entry is not None:
            entry.index.delete()

    def _evict(self) -> None:
        now = time.monotonic()
        for index_id in [i for i, e in self._entries.items() if now - e.last_used > self.ttl_seconds]:
            self._drop(index_id)
        while len(self._entries) >= self.max_indexes:
            oldest = min(self._entries, key=lambda i: self._entries[i].last_used)
            self._drop(oldest)

    def add(self, entry: IndexEntry) -> str:
        with self._lock:
            self._evict()
            index_id = uuid.uuid4().hex
            self._entries[index_id] = entry
            return index_id

    def get(self, index_id: str) -> IndexEntry:
        with self._lock:
            entry = self._entries.get(index_id)
            if entry is None or time.monotonic() - entry.last_used > self.ttl_seconds:
                if entry is not None:
                    self._drop(index_id)
                raise HTTPException(status.HTTP_404_NOT_FOUND, "Index not found or expired. Create it again.")
            entry.last_used = time.monotonic()
            return entry

    def delete(self, index_id: str) -> bool:
        with self._lock:
            if index_id not in self._entries:
                return False
            self._drop(index_id)
            return True

    def info(self, index_id: str, entry: IndexEntry) -> IndexInfo:
        return IndexInfo(
            index_id=index_id,
            documents=entry.documents,
            chunks=entry.chunks,
            chunking=entry.chunking,  # type: ignore[arg-type]
            chunk_size=entry.chunk_size,
            overlap=entry.overlap,
            expires_in_seconds=self.ttl_seconds,
        )


store = IndexStore(
    ttl_seconds=int(os.getenv("DOCUQUERY_INDEX_TTL", "3600")),
    max_indexes=int(os.getenv("DOCUQUERY_MAX_INDEXES", "20")),
)


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #

_rag_lock = threading.Lock()
_rag: GeminiRAG | None = None


def get_rag() -> GeminiRAG:
    """One shared RAG service built from GEMINI_API_KEY (tests override this)."""

    global _rag
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "GEMINI_API_KEY is not configured on the server.")
    provider = _llm_provider()
    groq_key = os.getenv("GROQ_API_KEY", "")
    if provider == "groq" and not groq_key:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "GROQ_API_KEY is not configured on the server.")
    with _rag_lock:
        if _rag is None:
            _rag = GeminiRAG(api_key=api_key, request_timeout=60, llm_provider=provider, llm_api_key=groq_key)
        return _rag


def _llm_provider() -> str:
    return resolve_llm_provider(os.getenv("LLM_PROVIDER") or None, os.getenv("GROQ_API_KEY", ""))


def require_token(x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = os.getenv("DOCUQUERY_API_TOKEN", "")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid X-API-Key header.")


Rag = Annotated[GeminiRAG, Depends(get_rag)]


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="DocuQuery RAG API",
    version=API_VERSION,
    description=(
        "Upload documents, index them in ChromaDB with Gemini embeddings, and ask questions "
        "answered only from those documents by an open-source LLM on Groq (or Gemini), "
        "with cited sources and similarity scores."
    ),
)


def _build(rag: GeminiRAG, files: list[NamedBytesIO], chunking: str, chunk_size: int, overlap: int) -> IndexInfo:
    if chunking not in CHUNKING_STRATEGIES:
        raise HTTPException(422, f"chunking must be one of {CHUNKING_STRATEGIES}")
    try:
        parts = load_many(files)
        chunks = build_chunks(parts, chunk_size=chunk_size, overlap=overlap, strategy=chunking)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not chunks:
        raise HTTPException(422, "No readable text was found in the files.")

    try:
        index = rag.build_index(chunks)
    except Exception as exc:  # noqa: BLE001 - provider errors
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Embedding request failed: {str(exc)[:300]}") from exc

    entry = IndexEntry(
        index=index,
        documents=sorted({chunk.source for chunk in chunks}),
        chunks=len(chunks),
        chunking=chunking,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return store.info(store.add(entry), entry)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        gemini_configured=bool(os.getenv("GEMINI_API_KEY")),
        llm_provider=_llm_provider(),
        llm_model=default_model(_llm_provider()),
        auth_required=bool(os.getenv("DOCUQUERY_API_TOKEN")),
        indexes=len(store),
    )


@app.post(
    "/indexes",
    response_model=IndexInfo,
    status_code=status.HTTP_201_CREATED,
    tags=["indexes"],
    dependencies=[Depends(require_token)],
    summary="Upload documents and build a searchable index",
)
def create_index(
    rag: Rag,
    files: Annotated[list[UploadFile], File(description="Up to 5 PDF, DOCX, TXT or MD files (15 MB total)")],
    chunking: Annotated[ChunkingName, Form()] = DEFAULT_CHUNKING,
    chunk_size: Annotated[int, Form(ge=200, le=4000)] = 900,
    overlap: Annotated[int, Form(ge=0, le=1000)] = 150,
) -> IndexInfo:
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"Upload no more than {MAX_FILES} files.")
    payloads, total = [], 0
    for upload in files:
        data = upload.file.read()
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise HTTPException(413, "The combined file size is above 15 MB.")
        payloads.append(NamedBytesIO(data, Path(upload.filename or "document.txt").name))
    return _build(rag, payloads, chunking, chunk_size, overlap)


@app.post(
    "/indexes/demo",
    response_model=IndexInfo,
    status_code=status.HTTP_201_CREATED,
    tags=["indexes"],
    dependencies=[Depends(require_token)],
    summary="Index the included demo documents",
)
def create_demo_index(rag: Rag, request: DemoIndexRequest | None = None) -> IndexInfo:
    request = request or DemoIndexRequest()
    files = open_paths(supported_files_in(DEMO_DIR))
    return _build(rag, files, request.chunking, request.chunk_size, request.overlap)


@app.get("/indexes/{index_id}", response_model=IndexInfo, tags=["indexes"], dependencies=[Depends(require_token)])
def get_index(index_id: str) -> IndexInfo:
    return store.info(index_id, store.get(index_id))


@app.delete(
    "/indexes/{index_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["indexes"],
    dependencies=[Depends(require_token)],
)
def delete_index(index_id: str) -> None:
    if not store.delete(index_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Index not found or expired.")


@app.post(
    "/indexes/{index_id}/query",
    response_model=QueryResponse,
    tags=["questions"],
    dependencies=[Depends(require_token)],
    summary="Ask a question answered only from the indexed documents",
)
def query_index(index_id: str, request: QueryRequest, rag: Rag) -> QueryResponse:
    entry = store.get(index_id)
    try:
        started = time.perf_counter()
        results = rag.retrieve(entry.index, request.question, top_k=request.top_k)
        retrieved_at = time.perf_counter()
        answer = rag.answer(request.question, results)
        finished = time.perf_counter()
    except Exception as exc:  # noqa: BLE001 - provider errors
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Gemini request failed: {str(exc)[:300]}") from exc

    return QueryResponse(
        index_id=index_id,
        question=request.question,
        answer=answer,
        sources=[
            Source(
                rank=rank,
                source=result.chunk.source,
                page=result.chunk.page,
                section=result.chunk.section,
                score=round(result.score, 4),
                text=result.chunk.text,
            )
            for rank, result in enumerate(results, start=1)
        ],
        retrieval_ms=round((retrieved_at - started) * 1000),
        generation_ms=round((finished - retrieved_at) * 1000),
    )

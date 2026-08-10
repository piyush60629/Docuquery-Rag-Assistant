"""Core Retrieval-Augmented Generation logic using Gemini and NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from document_loader import DocumentPart


DEFAULT_GENERATION_MODEL = "gemini-3.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


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


@dataclass(frozen=True)
class SearchResult:
    chunk: TextChunk
    score: float


def clean_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into readable overlapping chunks without external frameworks."""

    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    normalized = clean_text(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        hard_end = min(start + chunk_size, len(normalized))
        end = hard_end

        if hard_end < len(normalized):
            search_start = start + int(chunk_size * 0.55)
            candidates = [
                normalized.rfind("\n", search_start, hard_end),
                normalized.rfind(". ", search_start, hard_end),
                normalized.rfind("; ", search_start, hard_end),
                normalized.rfind(" ", search_start, hard_end),
            ]
            best_break = max(candidates)
            if best_break > start:
                end = best_break + (2 if normalized[best_break : best_break + 2] == ". " else 1)

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)

    return chunks


def build_chunks(
    parts: Iterable[DocumentPart], chunk_size: int = 900, overlap: int = 150
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    chunk_id = 1
    for part in parts:
        for text in chunk_text(part.text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                TextChunk(
                    source=part.source,
                    text=text,
                    page=part.page,
                    chunk_id=chunk_id,
                )
            )
            chunk_id += 1
    return chunks


def normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_search(
    query_embedding: Sequence[float],
    document_embeddings: np.ndarray,
    chunks: Sequence[TextChunk],
    top_k: int = 4,
) -> list[SearchResult]:
    if len(chunks) == 0:
        return []
    if len(document_embeddings) != len(chunks):
        raise ValueError("Embedding count must match chunk count")

    query = normalize_embeddings(np.asarray(query_embedding, dtype=np.float32))[0]
    documents = normalize_embeddings(document_embeddings)
    scores = documents @ query
    top_indices = np.argsort(scores)[::-1][: max(1, min(top_k, len(chunks)))]
    return [SearchResult(chunk=chunks[i], score=float(scores[i])) for i in top_indices]


class GeminiRAG:
    """Small RAG service that uses Gemini embeddings and text generation."""

    def __init__(
        self,
        api_key: str,
        generation_model: str = DEFAULT_GENERATION_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        if not api_key:
            raise ValueError("A Gemini API key is required")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "The google-genai package is required. Install requirements.txt first."
            ) from exc

        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.generation_model = generation_model
        self.embedding_model = embedding_model

    def embed_documents(self, chunks: Sequence[TextChunk], batch_size: int = 50) -> np.ndarray:
        if not chunks:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=[chunk.text for chunk in batch],
                config=self.types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768,
                ),
            )
            all_embeddings.extend(embedding.values for embedding in response.embeddings)

        return normalize_embeddings(np.asarray(all_embeddings, dtype=np.float32))

    def embed_query(self, question: str) -> np.ndarray:
        response = self.client.models.embed_content(
            model=self.embedding_model,
            contents=question,
            config=self.types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )
        return normalize_embeddings(
            np.asarray(response.embeddings[0].values, dtype=np.float32)
        )[0]

    def retrieve(
        self,
        question: str,
        chunks: Sequence[TextChunk],
        embeddings: np.ndarray,
        top_k: int = 4,
    ) -> list[SearchResult]:
        query_embedding = self.embed_query(question)
        return cosine_search(query_embedding, embeddings, chunks, top_k=top_k)

    def answer(self, question: str, results: Sequence[SearchResult]) -> str:
        if not results:
            return "I could not find relevant information in the uploaded documents."

        context_blocks = []
        for index, result in enumerate(results, start=1):
            context_blocks.append(
                f"[Source {index}: {result.chunk.label}]\n{result.chunk.text}"
            )
        context = "\n\n".join(context_blocks)

        system_instruction = (
            "You are a document question-answering assistant. Answer only from the "
            "provided context. Do not use outside knowledge. If the context does not "
            "contain the answer, say: 'I could not find that information in the uploaded "
            "documents.' Keep the language clear and practical. Cite supporting passages "
            "using [Source 1], [Source 2], and so on."
        )
        prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

        response = self.client.models.generate_content(
            model=self.generation_model,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=700,
            ),
        )
        return (response.text or "").strip() or (
            "I could not generate an answer from the uploaded documents."
        )

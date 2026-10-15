
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from document_loader import DocumentPart
from rag_engine import (
    NOT_FOUND_MESSAGE,
    GeminiRAG,
    TextChunk,
    build_chunks,
    chunk_text,
    get_chroma_client,
)


class KeywordEmbeddings(Embeddings):
    """Deterministic offline embeddings: one dimension per keyword."""

    VOCAB = ["pipeline", "failure", "retry", "quality", "check", "incremental", "load", "watermark"]

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        vector = [float(lowered.count(word)) for word in self.VOCAB]
        return vector if any(vector) else [0.01] * len(self.VOCAB)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture()
def rag():
    llm = FakeListChatModel(responses=["The pipeline retries three times [Source 1]."])
    return GeminiRAG(embeddings=KeywordEmbeddings(), llm=llm)


CHUNKS = [
    TextChunk("runbook.txt", "On pipeline failure the job will retry three times before alerting.", None, 1),
    TextChunk("runbook.txt", "Data quality check rules validate nulls and duplicates.", None, 2),
    TextChunk("guide.pdf", "Incremental load uses a watermark column to pick new rows.", 4, 3),
]


def test_chunk_text_preserves_content_with_overlap():
    text = "Sentence one. " * 120
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_chunk_text_rejects_bad_settings():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=300, overlap=300)


def test_build_chunks_keeps_source_metadata():
    parts = [DocumentPart(source="runbook.txt", page=None, text="A " * 600)]
    chunks = build_chunks(parts, chunk_size=300, overlap=50)
    assert chunks
    assert all(chunk.source == "runbook.txt" for chunk in chunks)
    assert [chunk.chunk_id for chunk in chunks] == list(range(1, len(chunks) + 1))


def test_document_round_trip_keeps_page_and_handles_missing_page():
    with_page = TextChunk("guide.pdf", "text", 4, 3)
    without_page = TextChunk("notes.md", "text", None, 7)
    assert TextChunk.from_document(with_page.to_document()) == with_page
    assert "page" not in without_page.to_document().metadata
    assert TextChunk.from_document(without_page.to_document()) == without_page


def test_chroma_search_returns_best_match_first(rag):
    index = rag.build_index(CHUNKS)
    try:
        assert index.count() == 3
        results = rag.retrieve(index, "What happens on pipeline failure and retry?", top_k=2)
        assert len(results) == 2
        assert results[0].chunk.chunk_id == 1
        assert results[0].score > results[1].score
        assert 0.99 < results[0].score <= 1.0001  # cosine similarity scale

        page_hit = rag.retrieve(index, "incremental load watermark", top_k=1)[0]
        assert page_hit.chunk.label == "guide.pdf, page 4"
    finally:
        index.delete()


def test_top_k_larger_than_index_is_clamped(rag):
    index = rag.build_index(CHUNKS[:1])
    try:
        assert len(rag.retrieve(index, "pipeline", top_k=6)) == 1
    finally:
        index.delete()


def test_sessions_get_separate_collections(rag):
    first = rag.build_index(CHUNKS[:1])
    second = rag.build_index(CHUNKS[1:])
    try:
        assert first.collection_name != second.collection_name
        assert first.count() == 1 and second.count() == 2
    finally:
        first.delete()
        second.delete()
    names = {getattr(c, "name", c) for c in get_chroma_client().list_collections()}
    assert first.collection_name not in names


def test_answer_chain_uses_llm_and_handles_no_results(rag):
    index = rag.build_index(CHUNKS)
    try:
        results = rag.retrieve(index, "pipeline failure", top_k=2)
        assert rag.answer("pipeline failure?", results).startswith("The pipeline retries")
    finally:
        index.delete()
    assert rag.answer("anything", []) == NOT_FOUND_MESSAGE


def test_api_key_required_without_injected_models():
    with pytest.raises(ValueError):
        GeminiRAG(api_key="")

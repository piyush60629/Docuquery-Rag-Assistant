
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


RUNBOOK = """Pipeline Runbook

Overview
The pipeline loads sales data every night.

Bronze Layer
Raw files land here with a batch identifier.

Failure Handling
Engineers rerun the failed batch after approval.
"""


def test_split_sections_detects_plain_text_and_markdown_headings():
    from rag_engine import split_sections

    plain = split_sections(RUNBOOK)
    assert [h for h, _ in plain] == [
        "Pipeline Runbook > Overview",
        "Pipeline Runbook > Bronze Layer",
        "Pipeline Runbook > Failure Handling",
    ]
    assert plain[1][1] == "Raw files land here with a batch identifier."

    markdown = split_sections("# Guide\nIntro text.\n\n## Setup\nInstall it.\n\n### Windows\nUse PowerShell.\n\n## Usage\nRun it.")
    assert [h for h, _ in markdown] == ["Guide", "Guide > Setup", "Guide > Setup > Windows", "Guide > Usage"]


def test_split_sections_does_not_treat_sentences_or_bullets_as_headings():
    from rag_engine import split_sections

    text = "Intro line that ends with a period.\n\n- Bullet item\nmore text\n\nShort sentence.\nNext line."
    assert split_sections(text) == [("", text)]


def test_section_strategy_keeps_topics_apart_and_prefixes_headings():
    parts = [DocumentPart(source="runbook.txt", text=RUNBOOK)]
    fixed = build_chunks(parts, chunk_size=300, overlap=50, strategy="fixed")
    sections = build_chunks(parts, chunk_size=300, overlap=50, strategy="sections")
    assert len(fixed) == 1  # the whole runbook fits one fixed-size chunk, mixing three topics
    assert len(sections) == 3
    assert sections[1].text.startswith("Pipeline Runbook > Bronze Layer\n")
    assert sections[1].section == "Pipeline Runbook > Bronze Layer"
    assert "rerun" not in sections[1].text
    assert TextChunk.from_document(sections[1].to_document()) == sections[1]


def test_long_sections_are_split_within_the_size_limit():
    body = "Each sentence describes one more validation rule. " * 40
    parts = [DocumentPart(source="rules.md", text=f"## Validation Rules\n{body}")]
    chunks = build_chunks(parts, chunk_size=400, overlap=50, strategy="sections")
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)
    assert all(c.text.startswith("Validation Rules\n") for c in chunks)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        build_chunks([DocumentPart(source="a.txt", text="x")], strategy="magic")


def test_docx_headings_become_markdown_sections():
    from io import BytesIO

    from docx import Document as WordDocument

    from document_loader import extract_docx
    from rag_engine import split_sections

    word = WordDocument()
    word.add_heading("Onboarding Guide", level=0)
    word.add_paragraph("Welcome to the team.")
    word.add_heading("Access", level=1)
    word.add_paragraph("Raise a request in the portal.")
    buffer = BytesIO()
    word.save(buffer)

    parts = extract_docx(buffer.getvalue(), "guide.docx")
    assert parts[0].text.startswith("# Onboarding Guide\n\nWelcome")
    assert [h for h, _ in split_sections(parts[0].text)] == ["Onboarding Guide", "Onboarding Guide > Access"]

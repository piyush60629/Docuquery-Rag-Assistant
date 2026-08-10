from document_loader import DocumentPart
from rag_engine import TextChunk, build_chunks, chunk_text, cosine_search


def test_chunk_text_preserves_content_with_overlap():
    text = "Sentence one. " * 120
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 305 for chunk in chunks)


def test_build_chunks_keeps_source_metadata():
    parts = [DocumentPart(source="runbook.txt", page=None, text="A " * 600)]
    chunks = build_chunks(parts, chunk_size=300, overlap=50)
    assert chunks
    assert all(chunk.source == "runbook.txt" for chunk in chunks)


def test_cosine_search_returns_best_match_first():
    chunks = [
        TextChunk("a.txt", "alpha", None, 1),
        TextChunk("b.txt", "beta", None, 2),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    results = cosine_search([0.9, 0.1], embeddings, chunks, top_k=2)
    assert results[0].chunk.source == "a.txt"
    assert results[0].score > results[1].score

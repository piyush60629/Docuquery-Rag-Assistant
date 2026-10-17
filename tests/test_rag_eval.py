import csv
import hashlib
import math
import re
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from document_loader import load_paths, supported_files_in
from evaluate import run_evaluation, with_retries, write_outputs
from rag_engine import NOT_FOUND_MESSAGE, GeminiRAG, SearchResult, TextChunk, build_chunks
from rag_eval import (
    CaseResult,
    aggregate,
    evidence_rank,
    is_refusal,
    load_dataset,
    normalize,
    parse_judge_response,
    render_markdown,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "eval" / "dataset.json"


class HashingEmbeddings(Embeddings):
    """Offline bag-of-words embeddings so the harness can be tested without an API key."""

    def __init__(self, dims: int = 512):
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for word in re.findall(r"[a-z0-9_]+", text.lower()):
            if len(word) > 2:
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector] if any(vector) else [1e-3] * self.dims

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET)


@pytest.fixture(scope="module")
def corpus_parts(dataset):
    corpus_dir, _ = dataset
    return load_paths(supported_files_in(corpus_dir))


def test_dataset_is_well_formed(dataset):
    corpus_dir, cases = dataset
    assert corpus_dir.name == "sample_documents"
    assert len(cases) >= 30
    assert sum(not c.answerable for c in cases) >= 5
    assert all(not c.evidence for c in cases if not c.answerable)


@pytest.mark.parametrize("strategy,chunk_size", [("fixed", 900), ("sections", 900), ("sections", 500)])
def test_every_evidence_phrase_fits_inside_one_chunk(dataset, corpus_parts, strategy, chunk_size):
    """If evidence never appears inside a single chunk, retrieval could never score a hit."""

    _, cases = dataset
    built = build_chunks(corpus_parts, chunk_size=chunk_size, overlap=150, strategy=strategy)
    chunks = [normalize(c.text) for c in built]
    for case in cases:
        if case.answerable:
            assert any(normalize(p) in chunk for p in case.evidence for chunk in chunks), case.id


def test_evidence_rank_and_refusal_detection():
    results = [
        SearchResult(TextChunk("a.md", "The **finance analyst** approves any manual adjustment.", None, 1), 0.9),
        SearchResult(TextChunk("b.md", "Gold tables must be ready by 6:00 AM IST.", None, 2), 0.8),
    ]
    assert evidence_rank(results, ["finance analyst approves any manual adjustment"]) == 1
    assert evidence_rank(results, ["ready by 6:00 am ist"]) == 2
    assert evidence_rank(results, ["not there"]) is None
    assert is_refusal(NOT_FOUND_MESSAGE)
    assert not is_refusal("The watermark is updated after the batch succeeds [Source 1].")


def test_parse_judge_response_handles_fences_and_snaps_scores():
    text = 'Here you go:\n```json\n{"faithfulness": 0.8, "correctness": "1", "reason": "ok"}\n```'
    assert parse_judge_response(text) == {"faithfulness": 1.0, "correctness": 1.0, "reason": "ok"}
    assert parse_judge_response('{"faithfulness": 0.4, "correctness": 0}')["faithfulness"] == 0.5
    with pytest.raises(ValueError):
        parse_judge_response("no json here")


def test_with_retries_retries_rate_limits_only():
    calls, sleeps = {"n": 0}, []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "ok"

    assert with_retries(flaky, sleep=sleeps.append, base_delay=1) == "ok"
    assert sleeps == [1, 2]
    with pytest.raises(ValueError):
        with_retries(lambda: (_ for _ in ()).throw(ValueError("bad input")), sleep=sleeps.append)


def test_retrieval_mode_end_to_end(dataset, corpus_parts):
    _, cases = dataset
    rag = GeminiRAG(embeddings=HashingEmbeddings(), llm=FakeListChatModel(responses=["unused"]))
    results, summary, chunks = run_evaluation(rag, cases, corpus_parts, mode="retrieval", top_k=4, log=lambda _: None)
    assert chunks > 10
    assert summary["errors"] == 0
    assert summary["answerable_cases"] == sum(c.answerable for c in cases)
    # Even a crude bag-of-words baseline should find most evidence in the top 4.
    assert summary["hit_at_4"] >= 0.6
    assert 0 < summary["mrr"] <= 1
    assert "faithfulness" not in summary


def test_full_mode_scores_answers_and_refusals(dataset, corpus_parts, tmp_path):
    _, cases = dataset
    answers = ["It is described in [Source 1]." if c.answerable else NOT_FOUND_MESSAGE for c in cases]
    rag = GeminiRAG(embeddings=HashingEmbeddings(), llm=FakeListChatModel(responses=answers))
    judge = FakeListChatModel(responses=['{"faithfulness": 1, "correctness": 0.5, "reason": "partial"}'])
    results, summary, chunks = run_evaluation(
        rag, cases, corpus_parts, mode="full", judge=judge, top_k=4, sleep=lambda _: None, log=lambda _: None
    )
    assert summary["errors"] == 0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["false_refusal_rate"] == 0.0
    assert summary["faithfulness"] == 1.0
    assert summary["answer_correctness"] == 0.5

    config = {"timestamp": "t", "label": "test", "mode": "full", "chunk_size": 900, "overlap": 150, "top_k": 4, "chunks": chunks}
    report = write_outputs(tmp_path, results, summary, config)
    write_outputs(tmp_path, results, summary, config)
    assert "Answer faithfulness" in report.read_text()
    rows = list(csv.DictReader((tmp_path / "history.csv").open()))
    assert len(rows) == 2 and rows[0]["hit_at_k"] == str(summary["hit_at_4"])


def test_aggregate_counts_errors_separately():
    results = [
        CaseResult("a", "q", True, rank=1),
        CaseResult("b", "q", True, rank=None),
        CaseResult("c", "q", True, error="boom"),
    ]
    summary = aggregate(results, top_k=3)
    assert summary["errors"] == 1 and summary["hit_at_1"] == 0.5 and summary["hit_at_3"] == 0.5
    assert "| a | 1 |" in render_markdown(summary, results, {"mode": "retrieval", "chunk_size": 900, "overlap": 150, "top_k": 3})


def test_update_readme_replaces_only_the_results_block(tmp_path):
    from evaluate import README_END, README_START, update_readme

    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nold\n{README_END}\noutro\n", encoding="utf-8")
    summary = {"cases": 2, "answerable_cases": 2, "unanswerable_cases": 0, "errors": 0, "hit_at_1": 0.5, "hit_at_4": 1.0, "mrr": 0.75}
    config = {"mode": "retrieval", "chunk_size": 900, "overlap": 150, "top_k": 4, "chunks": 13}
    assert update_readme(readme, summary, config)
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n") and text.endswith("outro\n")
    assert "old" not in text and "| Retrieval Hit@1 | 50.0% |" in text and "Per-question" not in text
    assert update_readme(readme, summary, config)  # idempotent
    assert text == readme.read_text(encoding="utf-8")


def test_daily_quota_errors_are_not_retried():
    from evaluate import explain_error, is_daily_quota_error

    sleeps = []
    daily = "429 RESOURCE_EXHAUSTED: Quota exceeded for metric generate_content_free_tier_requests, limit: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    with pytest.raises(RuntimeError):
        with_retries(lambda: (_ for _ in ()).throw(RuntimeError(daily)), sleep=sleeps.append)
    assert sleeps == []
    assert is_daily_quota_error(daily)
    assert not is_daily_quota_error("429 RESOURCE_EXHAUSTED: GenerateRequestsPerMinutePerProjectPerModel")
    assert "daily quota" in explain_error(daily)
    assert "rejected" in explain_error("400 API key not valid. Please pass a valid API key.")


class BrokenEmbeddings(HashingEmbeddings):
    """Indexing works, but every question fails, like a key that hits its quota mid-run."""

    def embed_query(self, text):
        raise RuntimeError("400 API key not valid")


def test_run_stops_early_after_repeated_errors(dataset, corpus_parts):
    _, cases = dataset
    rag = GeminiRAG(embeddings=BrokenEmbeddings(), llm=FakeListChatModel(responses=["x"]))
    results, summary, _ = run_evaluation(rag, cases, corpus_parts, mode="retrieval", sleep=lambda _: None, log=lambda _: None)
    assert len(results) == 3
    assert summary["planned_cases"] == len(cases)
    assert summary["stopped_early"] and "rejected" in summary["stopped_early"]


def test_single_error_does_not_stop_the_run(dataset, corpus_parts):
    _, cases = dataset

    class OneBadQuestion(HashingEmbeddings):
        def embed_query(self, text):
            if text == cases[0].question:
                raise RuntimeError("500 internal error")
            return super().embed_query(text)

    rag = GeminiRAG(embeddings=OneBadQuestion(), llm=FakeListChatModel(responses=["x"]))
    results, summary, _ = run_evaluation(rag, cases, corpus_parts, mode="retrieval", sleep=lambda _: None, log=lambda _: None)
    assert len(results) == len(cases)
    assert summary["errors"] == 1 and summary["stopped_early"] is None


def test_history_file_is_upgraded_when_columns_change(tmp_path):
    from evaluate import HISTORY_FIELDS

    old = tmp_path / "history.csv"
    old.write_text(
        "timestamp,label,mode,chunk_size,overlap,top_k,chunks,cases,errors,hit_at_1,hit_at_k,mrr\n"
        "2026-10-15 17:40 UTC,baseline,retrieval,900,150,4,13,35,0,0.8,0.9667,0.8667\n",
        encoding="utf-8",
    )
    summary = {"cases": 1, "answerable_cases": 1, "unanswerable_cases": 0, "errors": 0, "hit_at_1": 1.0, "hit_at_4": 1.0, "mrr": 1.0}
    config = {"timestamp": "t", "label": "new", "mode": "retrieval", "chunking": "sections", "chunk_size": 900, "overlap": 150, "top_k": 4, "chunks": 30}
    write_outputs(tmp_path, [], summary, config)
    rows = list(csv.DictReader(old.open()))
    assert list(rows[0].keys()) == HISTORY_FIELDS
    assert [(r["label"], r["chunking"], r["hit_at_1"]) for r in rows] == [("baseline", "fixed", "0.8"), ("new", "sections", "1.0")]

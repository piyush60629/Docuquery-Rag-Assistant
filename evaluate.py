"""Evaluate DocuQuery's retrieval and answers on eval/dataset.json.

Usage:
    python evaluate.py --mode retrieval          # retrieval metrics only (embeddings calls only)
    python evaluate.py --mode full               # + answers and LLM-as-judge scores
    python evaluate.py --mode full --chunk-size 600 --top-k 3 --label "smaller chunks"

Needs GEMINI_API_KEY in the environment. Results go to eval/results/:
    latest.json / latest.md  - full results of this run
    history.csv              - one summary row per run, for before/after comparisons
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from document_loader import DocumentPart, load_paths, supported_files_in
from rag_engine import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    GeminiRAG,
    build_chunks,
    format_context,
)
from rag_eval import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    CaseResult,
    EvalCase,
    aggregate,
    evidence_rank,
    is_refusal,
    load_dataset,
    parse_judge_response,
    render_markdown,
)

RETRYABLE_MARKERS = ("429", "resource_exhausted", "quota", "rate limit", "503", "unavailable", "deadline")
HISTORY_FIELDS = [
    "timestamp", "label", "mode", "chunk_size", "overlap", "top_k", "chunks", "cases", "errors",
    "hit_at_1", "hit_at_k", "mrr", "faithfulness", "answer_correctness", "refusal_accuracy", "false_refusal_rate",
]


def with_retries(
    fn: Callable[[], Any],
    attempts: int = 5,
    base_delay: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Retry rate-limit / temporary errors with exponential backoff (free tier friendly)."""

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many error types
            message = str(exc).lower()
            if attempt == attempts or not any(marker in message for marker in RETRYABLE_MARKERS):
                raise
            sleep(base_delay * (2 ** (attempt - 1)))
    raise RuntimeError("unreachable")


def judge_answer(judge: BaseChatModel, case: EvalCase, context: str, answer: str) -> dict[str, Any]:
    prompt = JUDGE_USER_TEMPLATE.format(
        question=case.question, context=context, answer=answer, reference=case.reference_answer
    )
    message = judge.invoke([SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=prompt)])
    return parse_judge_response(StrOutputParser().invoke(message))


def run_evaluation(
    rag: GeminiRAG,
    cases: Sequence[EvalCase],
    parts: Sequence[DocumentPart],
    *,
    mode: str = "retrieval",
    chunk_size: int = 900,
    overlap: int = 150,
    top_k: int = 4,
    judge: BaseChatModel | None = None,
    delay: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> tuple[list[CaseResult], dict[str, Any], int]:
    if mode not in {"retrieval", "full"}:
        raise ValueError("mode must be 'retrieval' or 'full'")
    if mode == "full" and judge is None:
        raise ValueError("full mode needs a judge model")

    chunks = build_chunks(parts, chunk_size=chunk_size, overlap=overlap)
    log(f"Indexing {len(chunks)} chunks from {len({c.source for c in chunks})} documents...")
    index = with_retries(lambda: rag.build_index(chunks), sleep=sleep)

    results: list[CaseResult] = []
    try:
        for number, case in enumerate(cases, start=1):
            result = CaseResult(id=case.id, question=case.question, answerable=case.answerable)
            try:
                retrieved = with_retries(lambda: rag.retrieve(index, case.question, top_k=top_k), sleep=sleep)
                result.retrieved = [f"{r.chunk.label} #{r.chunk.chunk_id}" for r in retrieved]
                result.top_score = round(retrieved[0].score, 4) if retrieved else None
                if case.answerable:
                    result.rank = evidence_rank(retrieved, case.evidence)

                if mode == "full":
                    if delay:
                        sleep(delay)
                    answer = with_retries(lambda: rag.answer(case.question, retrieved), sleep=sleep)
                    result.answer = answer
                    result.refused = is_refusal(answer)
                    if case.answerable:
                        if result.refused:
                            result.faithfulness, result.correctness = None, 0.0
                            result.judge_reason = "Declined to answer an answerable question."
                        else:
                            if delay:
                                sleep(delay)
                            verdict = with_retries(
                                lambda: judge_answer(judge, case, format_context(retrieved), answer), sleep=sleep
                            )
                            result.faithfulness = verdict["faithfulness"]
                            result.correctness = verdict["correctness"]
                            result.judge_reason = verdict["reason"]
            except Exception as exc:  # noqa: BLE001
                result.error = f"{type(exc).__name__}: {exc}"[:300]

            results.append(result)
            status = result.error or (
                f"rank={result.rank}" if case.answerable else f"refused={result.refused}"
            )
            log(f"[{number:02d}/{len(cases)}] {case.id}: {status}")
    finally:
        index.delete()

    return results, aggregate(results, top_k), len(chunks)


def write_outputs(
    output_dir: Path,
    results: Sequence[CaseResult],
    summary: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, "summary": summary, "results": [r.to_dict() for r in results]}
    (output_dir / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = render_markdown(summary, results, config)
    (output_dir / "latest.md").write_text(markdown, encoding="utf-8")

    history = output_dir / "history.csv"
    is_new = not history.exists()
    row = {**{k: config.get(k) for k in HISTORY_FIELDS}, **summary}
    row["hit_at_k"] = summary.get(f"hit_at_{config['top_k']}")
    with history.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return output_dir / "latest.md"


README_START = "<!-- EVAL_RESULTS_START -->"
README_END = "<!-- EVAL_RESULTS_END -->"


def update_readme(readme: Path, summary: dict[str, Any], config: dict[str, Any]) -> bool:
    """Replace the results block in README.md with this run's summary table."""

    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    table = render_markdown(summary, [], config).split("## Per-question results")[0]
    table = table.replace("# DocuQuery evaluation results", "**Latest results**").strip()
    block = f"{README_START}\n{table}\n\nFull per-question results: `eval/results/latest.md`.\n{README_END}"
    start, end = text.index(README_START), text.index(README_END) + len(README_END)
    readme.write_text(text[:start] + block + text[end:], encoding="utf-8")
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="eval/dataset.json")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="full")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--delay", type=float, default=4.0, help="Seconds to wait between LLM calls (free tier).")
    parser.add_argument("--label", default="", help="Short note stored in history.csv, e.g. 'baseline'.")
    parser.add_argument("--output-dir", default="eval/results")
    parser.add_argument("--update-readme", action="store_true", help="Write the summary table into README.md.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("Set GEMINI_API_KEY first.", file=sys.stderr)
        return 2

    corpus_dir, cases = load_dataset(args.dataset)
    files = supported_files_in(corpus_dir)
    parts = load_paths(files)

    rag = GeminiRAG(api_key=api_key)
    judge = None
    if args.mode == "full":
        from langchain_google_genai import ChatGoogleGenerativeAI

        judge = ChatGoogleGenerativeAI(
            model=DEFAULT_GENERATION_MODEL, google_api_key=api_key, temperature=0, max_output_tokens=300
        )

    results, summary, chunk_count = run_evaluation(
        rag,
        cases,
        parts,
        mode=args.mode,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
        judge=judge,
        delay=args.delay if args.mode == "full" else 0.0,
    )

    config = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "label": args.label,
        "mode": args.mode,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "top_k": args.top_k,
        "chunks": chunk_count,
        "documents": [path.name for path in files],
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "generation_model": DEFAULT_GENERATION_MODEL,
        "judge_model": DEFAULT_GENERATION_MODEL if args.mode == "full" else "",
    }
    report = write_outputs(Path(args.output_dir), results, summary, config)
    if args.update_readme and summary["errors"] < summary["cases"]:
        update_readme(Path(__file__).resolve().parent / "README.md", summary, config)
    print()
    print(report.read_text(encoding="utf-8"))
    return 1 if summary["errors"] == summary["cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

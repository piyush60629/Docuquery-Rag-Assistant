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
    CHUNKING_STRATEGIES,
    DEFAULT_CHUNKING,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_GROQ_JUDGE_MODEL,
    LLM_PROVIDERS,
    GeminiRAG,
    build_chunks,
    format_context,
    make_chat_model,
    resolve_llm_provider,
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

RETRYABLE_MARKERS = ("429", "resource_exhausted", "quota", "rate limit", "503", "unavailable", "deadline", "timeout", "timed out")
# A daily quota will not reset within this run, so waiting and retrying only wastes time.
DAILY_QUOTA_MARKERS = ("perday", "per_day", "per day", "daily")
MAX_CONSECUTIVE_ERRORS = 3
HISTORY_FIELDS = [
    "timestamp", "label", "mode", "llm_provider", "chunking", "chunk_size", "overlap", "top_k", "chunks", "cases", "errors",
    "hit_at_1", "hit_at_k", "mrr", "faithfulness", "answer_correctness", "refusal_accuracy", "false_refusal_rate",
]


def is_daily_quota_error(message: str) -> bool:
    lowered = message.lower().replace(" ", "")
    return any(marker.replace(" ", "") in lowered for marker in DAILY_QUOTA_MARKERS) and (
        "quota" in lowered or "resource_exhausted" in lowered or "429" in lowered
    )


def explain_error(message: str) -> str:
    """Turn a provider error into a plain-English hint."""

    lowered = message.lower()
    if is_daily_quota_error(message):
        return "The LLM provider's free-tier daily quota is used up. Try again tomorrow, or run --mode retrieval (far fewer calls)."
    if "429" in lowered or "resource_exhausted" in lowered or "rate limit" in lowered:
        return "The LLM provider is rate limiting requests. Run again with a larger --delay, or use --mode retrieval."
    if any(marker in lowered for marker in ("api key not valid", "api_key_invalid", "invalid api key", "permission", "401", "403")):
        return "An API key was rejected. Check the GEMINI_API_KEY and GROQ_API_KEY secrets."
    if "not found" in lowered or "404" in lowered or "decommissioned" in lowered:
        return "A model name was not found or has been retired. Check the model names in rag_engine.py."
    return "Unexpected error; see the message above."


def with_retries(
    fn: Callable[[], Any],
    attempts: int = 4,
    base_delay: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Retry short-lived rate-limit / temporary errors with exponential backoff (10s, 20s, 40s).

    Daily-quota errors are raised immediately because waiting will not help.
    """

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many error types
            message = str(exc)
            retryable = any(marker in message.lower() for marker in RETRYABLE_MARKERS)
            if attempt == attempts or not retryable or is_daily_quota_error(message):
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
    chunking: str = DEFAULT_CHUNKING,
    judge: BaseChatModel | None = None,
    delay: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS,
) -> tuple[list[CaseResult], dict[str, Any], int]:
    """Score every case. Stops early (summary["stopped_early"]) after repeated failures,
    so a broken key or an exhausted quota fails in seconds instead of grinding on."""
    if mode not in {"retrieval", "full"}:
        raise ValueError("mode must be 'retrieval' or 'full'")
    if mode == "full" and judge is None:
        raise ValueError("full mode needs a judge model")

    chunks = build_chunks(parts, chunk_size=chunk_size, overlap=overlap, strategy=chunking)
    log(f"Indexing {len(chunks)} chunks from {len({c.source for c in chunks})} documents...")
    index = with_retries(lambda: rag.build_index(chunks), sleep=sleep)

    results: list[CaseResult] = []
    stopped_early: str | None = None
    consecutive_errors = 0
    started = time.monotonic()
    try:
        for number, case in enumerate(cases, start=1):
            case_started = time.monotonic()
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
            if case.answerable:
                status = f"rank={result.rank or 'miss'}"
                if result.correctness is not None:
                    status += f" correct={result.correctness:g}"
            else:
                status = f"refused={result.refused}" if mode == "full" else "unanswerable (checked in full mode)"
            if result.error:
                status = f"ERROR {result.error}"
            log(f"[{number:02d}/{len(cases)}] {case.id}: {status} ({time.monotonic() - case_started:.1f}s)")

            if result.error:
                consecutive_errors += 1
                if is_daily_quota_error(result.error) or consecutive_errors >= max_consecutive_errors:
                    stopped_early = f"Stopped after {number} of {len(cases)} questions. {explain_error(result.error)}"
                    log(stopped_early)
                    break
            else:
                consecutive_errors = 0
    finally:
        index.delete()

    summary = aggregate(results, top_k)
    summary["planned_cases"] = len(cases)
    summary["stopped_early"] = stopped_early
    summary["duration_seconds"] = round(time.monotonic() - started, 1)
    return results, summary, len(chunks)


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
    if history.exists():
        with history.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            old_fields, old_rows = reader.fieldnames or [], list(reader)
        if old_fields != HISTORY_FIELDS:
            # Columns were added in a later version: rewrite the file with the new header.
            # Runs recorded before the "chunking" column existed all used fixed-size chunking.
            with history.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
                writer.writeheader()
                for old in old_rows:
                    writer.writerow({"chunking": "fixed", **{k: v for k, v in old.items() if v not in (None, "")}})
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
    parser.add_argument("--chunking", choices=list(CHUNKING_STRATEGIES), default=DEFAULT_CHUNKING)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--delay", type=float, default=6.0, help="Seconds to wait between LLM calls (free tier).")
    parser.add_argument(
        "--llm-provider",
        choices=list(LLM_PROVIDERS),
        default=None,
        help="Who writes and grades answers. Default: groq when GROQ_API_KEY is set, otherwise gemini.",
    )
    parser.add_argument("--answer-model", default=None, help="Override the answer model name.")
    parser.add_argument("--judge-model", default=None, help="Override the judge model name.")
    parser.add_argument("--label", default="", help="Short note stored in history.csv, e.g. 'baseline'.")
    parser.add_argument("--output-dir", default="eval/results")
    parser.add_argument("--update-readme", action="store_true", help="Write the summary table into README.md.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    # Print progress line by line, so it shows up live in GitHub Actions logs.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("Set GEMINI_API_KEY first.", file=sys.stderr)
        return 2

    corpus_dir, cases = load_dataset(args.dataset)
    files = supported_files_in(corpus_dir)
    parts = load_paths(files)

    # Retries are handled once, in with_retries(). The SDK's own retries are turned off so
    # they do not stack on top (6 SDK retries x 5 script retries made one bad call take minutes).
    groq_key = os.getenv("GROQ_API_KEY", "")
    provider = resolve_llm_provider(args.llm_provider, groq_key)
    llm_key = groq_key if provider == "groq" else api_key
    if args.mode == "full" and not llm_key:
        print("Set GROQ_API_KEY first (or use --llm-provider gemini).", file=sys.stderr)
        return 2

    # Retrieval mode never calls the answer model, so a missing Groq key is fine there.
    rag = GeminiRAG(
        api_key=api_key,
        max_retries=0,
        request_timeout=60,
        llm_provider=provider if llm_key else "gemini",
        llm_api_key=groq_key,
        generation_model=args.answer_model,
    )
    judge = None
    judge_model = ""
    if args.mode == "full":
        # A different model grades the answers, so the answer model is not marking its own work.
        judge_model = args.judge_model or (DEFAULT_GROQ_JUDGE_MODEL if provider == "groq" else DEFAULT_GENERATION_MODEL)
        judge = make_chat_model(
            provider,
            llm_key,
            judge_model,
            temperature=0,
            max_tokens=1024 if provider == "groq" else 300,
            max_retries=0,
            timeout=60,
        )

    print(f"Mode: {args.mode} | {len(cases)} questions | search: Gemini embeddings", end="")
    if args.mode == "full":
        print(f" | answers: {rag.generation_model} | judge: {judge_model} ({provider}) | about {len(cases) * 2} LLM calls")
    else:
        print(f" | about {len(cases)} embedding calls")
    try:
        results, summary, chunk_count = run_evaluation(
            rag,
            cases,
            parts,
            mode=args.mode,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            top_k=args.top_k,
            chunking=args.chunking,
            judge=judge,
            delay=args.delay if args.mode == "full" else 0.0,
        )
    except Exception as exc:  # noqa: BLE001 - e.g. the index could not be built at all
        print(f"\nEvaluation could not start: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr)
        print(explain_error(str(exc)), file=sys.stderr)
        return 1

    config = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "label": args.label,
        "mode": args.mode,
        "chunking": args.chunking,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "top_k": args.top_k,
        "chunks": chunk_count,
        "documents": [path.name for path in files],
        "llm_provider": provider if args.mode == "full" else "",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "generation_model": rag.generation_model if args.mode == "full" else "",
        "judge_model": judge_model,
    }
    if summary["stopped_early"]:
        # Keep the partial results for debugging, but do not overwrite the published scores.
        output_dir = Path(args.output_dir) / "failed-run"
        report = write_outputs(output_dir, results, summary, config)
        print(f"\n{summary['stopped_early']}", file=sys.stderr)
        print(f"Partial results: {report}", file=sys.stderr)
        return 1

    report = write_outputs(Path(args.output_dir), results, summary, config)
    if args.update_readme:
        update_readme(Path(__file__).resolve().parent / "README.md", summary, config)
    print(f"\nFinished in {summary['duration_seconds']:.0f} seconds.\n")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

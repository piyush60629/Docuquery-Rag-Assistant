"""Evaluation helpers for DocuQuery.

Retrieval metrics (no LLM needed):
    Hit@1, Hit@k  - share of answerable questions whose evidence appears in the top 1 / top k chunks
    MRR           - mean reciprocal rank of the first chunk that contains the evidence

Answer metrics (LLM-as-judge with Gemini):
    Faithfulness       - is every claim in the answer supported by the retrieved passages? (answered cases)
    Answer correctness - does the answer match the reference answer? (all answerable cases)
    Refusal accuracy   - share of unanswerable questions where the assistant declines instead of guessing
    False refusals     - share of answerable questions where the assistant wrongly declined
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from rag_engine import SearchResult


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    evidence: tuple[str, ...]
    reference_answer: str
    answerable: bool = True


def load_dataset(path: str | Path) -> tuple[Path, list[EvalCase]]:
    """Return (corpus directory, cases). The corpus path is relative to the repo root."""

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent
    corpus_dir = (root / data.get("corpus", "sample_documents")).resolve()
    cases = [
        EvalCase(
            id=item["id"],
            question=item["question"],
            evidence=tuple(item.get("evidence", [])),
            reference_answer=item.get("reference_answer", ""),
            answerable=bool(item.get("answerable", True)),
        )
        for item in data["cases"]
    ]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Case ids must be unique")
    for case in cases:
        if case.answerable and not case.evidence:
            raise ValueError(f"Answerable case {case.id} needs at least one evidence phrase")
    return corpus_dir, cases


# --------------------------------------------------------------------------- #
# Retrieval scoring
# --------------------------------------------------------------------------- #


def normalize(text: str) -> str:
    """Lowercase, drop Markdown emphasis characters and collapse whitespace."""

    text = re.sub(r"[*`_]{2,}|[*`]", "", text.lower())
    return " ".join(text.split())


def evidence_rank(results: Sequence[SearchResult], evidence: Iterable[str]) -> int | None:
    """1-based rank of the first retrieved chunk containing any evidence phrase."""

    phrases = [normalize(phrase) for phrase in evidence if phrase.strip()]
    for rank, result in enumerate(results, start=1):
        chunk = normalize(result.chunk.text)
        if any(phrase in chunk for phrase in phrases):
            return rank
    return None


# --------------------------------------------------------------------------- #
# Answer scoring
# --------------------------------------------------------------------------- #

REFUSAL_PATTERNS = (
    "could not find",
    "couldn't find",
    "not find that information",
    "does not contain",
    "doesn't contain",
    "not mentioned in",
    "no information",
    "not provided in",
    "not stated in",
)


def is_refusal(answer: str) -> bool:
    lowered = normalize(answer)
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of a retrieval-augmented question answering system.
You receive a QUESTION, the RETRIEVED CONTEXT given to the system, the system's ANSWER and a REFERENCE ANSWER.

Score two things:
1. faithfulness: are all factual claims in the ANSWER supported by the RETRIEVED CONTEXT?
   1 = fully supported, 0.5 = mostly supported with a minor unsupported detail, 0 = contains an unsupported or contradicted claim.
   Judge only against the RETRIEVED CONTEXT, not your own knowledge. Ignore citation markers like [Source 1].
2. correctness: does the ANSWER give the same essential information as the REFERENCE ANSWER?
   1 = correct and complete, 0.5 = partly correct or missing an important part, 0 = wrong or does not answer.
   Extra correct detail is fine.

Respond with JSON only, no other text:
{"faithfulness": <0|0.5|1>, "correctness": <0|0.5|1>, "reason": "<one short sentence>"}"""

JUDGE_USER_TEMPLATE = """QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}

REFERENCE ANSWER:
{reference}"""


def _snap_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min((0.0, 0.5, 1.0), key=lambda allowed: abs(allowed - number))


def parse_judge_response(text: str) -> dict[str, Any]:
    """Extract the judge's JSON, tolerating code fences or extra text around it."""

    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return JSON: {text!r}")
    data = json.loads(match.group(0))
    return {
        "faithfulness": _snap_score(data.get("faithfulness")),
        "correctness": _snap_score(data.get("correctness")),
        "reason": str(data.get("reason", "")).strip(),
    }


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class CaseResult:
    id: str
    question: str
    answerable: bool
    rank: int | None = None
    retrieved: list[str] = field(default_factory=list)
    top_score: float | None = None
    answer: str | None = None
    refused: bool | None = None
    faithfulness: float | None = None
    correctness: float | None = None
    judge_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate(results: Sequence[CaseResult], top_k: int) -> dict[str, Any]:
    answerable = [r for r in results if r.answerable and r.error is None]
    unanswerable = [r for r in results if not r.answerable and r.error is None]

    summary: dict[str, Any] = {
        "cases": len(results),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "errors": sum(1 for r in results if r.error),
        "hit_at_1": _mean([1.0 if r.rank == 1 else 0.0 for r in answerable]),
        f"hit_at_{top_k}": _mean([1.0 if r.rank is not None else 0.0 for r in answerable]),
        "mrr": _mean([1.0 / r.rank if r.rank else 0.0 for r in answerable]),
    }

    generated = any(r.answer is not None for r in results)
    if generated:
        answered = [r for r in answerable if r.answer is not None and not r.refused]
        summary.update(
            {
                "faithfulness": _mean([r.faithfulness for r in answered if r.faithfulness is not None]),
                "answer_correctness": _mean(
                    [r.correctness if r.correctness is not None else 0.0 for r in answerable if r.answer is not None]
                ),
                "refusal_accuracy": _mean([1.0 if r.refused else 0.0 for r in unanswerable if r.answer is not None]),
                "false_refusal_rate": _mean([1.0 if r.refused else 0.0 for r in answerable if r.answer is not None]),
            }
        )
    return summary


METRIC_LABELS = [
    ("hit_at_1", "Retrieval Hit@1"),
    ("hit_at_k", "Retrieval Hit@{k}"),
    ("mrr", "Retrieval MRR"),
    ("faithfulness", "Answer faithfulness"),
    ("answer_correctness", "Answer correctness"),
    ("refusal_accuracy", "Refusal accuracy (unanswerable)"),
    ("false_refusal_rate", "False refusal rate (lower is better)"),
]


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def render_markdown(summary: dict[str, Any], results: Sequence[CaseResult], config: dict[str, Any]) -> str:
    top_k = config["top_k"]
    lines = [
        "# DocuQuery evaluation results",
        "",
        f"- Run: {config.get('timestamp', '')}  ",
        f"- Mode: {config['mode']} · chunking {config.get('chunking', 'fixed')} · chunk size {config['chunk_size']} · overlap {config['overlap']} · top-k {top_k}  ",
        f"- Models: embeddings `{config.get('embedding_model', '')}`, answers `{config.get('generation_model', '')}`, judge `{config.get('judge_model', '')}`  ",
        f"- Questions: {summary['cases']} ({summary['answerable_cases']} answerable, "
        f"{summary['unanswerable_cases']} unanswerable) · chunks indexed: {config.get('chunks', '')} · errors: {summary['errors']}",
        "",
        "| Metric | Score |",
        "|---|---|",
    ]
    for key, label in METRIC_LABELS:
        actual_key = f"hit_at_{top_k}" if key == "hit_at_k" else key
        if actual_key not in summary:
            continue
        value = summary[actual_key]
        shown = f"{value:.3f}" if key == "mrr" and value is not None else _fmt_pct(value)
        lines.append(f"| {label.format(k=top_k)} | {shown} |")

    lines += ["", "## Per-question results", "", "| ID | Rank | Refused | Faithful | Correct | Note |", "|---|---|---|---|---|---|"]
    for r in results:
        rank = "-" if not r.answerable else (str(r.rank) if r.rank else "miss")
        refused = "" if r.refused is None else ("yes" if r.refused else "no")
        faith = "" if r.faithfulness is None else f"{r.faithfulness:g}"
        correct = "" if r.correctness is None else f"{r.correctness:g}"
        note = (r.error or r.judge_reason or "").replace("|", "/")[:120]
        lines.append(f"| {r.id} | {rank} | {refused} | {faith} | {correct} | {note} |")
    lines.append("")
    return "\n".join(lines)

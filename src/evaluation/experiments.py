from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.metrics import EndToEndMetrics, compute_end_to_end_metrics
from src.pipeline.rag_pipeline import RAGResult
from src.utils.config import RESULTS_DIR, ensure_project_directories


@dataclass(slots=True)
class EvaluationSample:
    query: str
    reference_answer: str
    relevant_source_names: set[str]


@dataclass(slots=True)
class EvaluationReport:
    config_name: str
    sample_count: int
    k: int
    metrics: EndToEndMetrics
    created_at_utc: str

    def to_dict(self) -> dict[str, object]:
        return {
            "config_name": self.config_name,
            "sample_count": self.sample_count,
            "k": self.k,
            "metrics": self.metrics.to_dict(),
            "created_at_utc": self.created_at_utc,
        }


def _extract_contexts(result: RAGResult) -> list[str]:
    contexts: list[str] = []
    for chunk in result.retrieved_chunks:
        text = str(chunk.get("text", ""))
        if text:
            contexts.append(text)
    return contexts


def _extract_sources(result: RAGResult) -> list[str]:
    sources: list[str] = []
    for chunk in result.retrieved_chunks:
        source = str(chunk.get("source_name", ""))
        if source:
            sources.append(source)
    return sources


def evaluate_rag_results(
    config_name: str,
    rag_results: list[RAGResult],
    samples: list[EvaluationSample],
    *,
    k: int = 5,
    output_dir: Path | None = None,
) -> EvaluationReport:
    """Evaluate RAG outputs against labeled samples and persist a JSON report."""

    if len(rag_results) != len(samples):
        raise ValueError("rag_results and samples must have the same length")

    retrieved_source_names = [_extract_sources(result) for result in rag_results]
    relevant_source_names = [sample.relevant_source_names for sample in samples]
    answers = [result.answer for result in rag_results]
    reference_answers = [sample.reference_answer for sample in samples]
    retrieved_contexts = [_extract_contexts(result) for result in rag_results]

    metrics = compute_end_to_end_metrics(
        retrieved_source_names=retrieved_source_names,
        relevant_source_names=relevant_source_names,
        answers=answers,
        reference_answers=reference_answers,
        retrieved_contexts=retrieved_contexts,
        k=k,
    )

    report = EvaluationReport(
        config_name=config_name,
        sample_count=len(samples),
        k=k,
        metrics=metrics,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    ensure_project_directories()
    output_dir = output_dir or (RESULTS_DIR / "comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = config_name.lower().replace(" ", "_").replace("/", "_")
    output_path = output_dir / f"evaluation_{safe_name}.json"
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return report


def compare_evaluation_reports(
    reports: list[EvaluationReport],
    *,
    output_dir: Path | None = None,
) -> Path:
    """Write a compact JSON comparison for multiple evaluation reports."""

    ensure_project_directories()
    output_dir = output_dir or (RESULTS_DIR / "comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reports": [report.to_dict() for report in reports],
    }

    output_path = output_dir / "evaluation_comparison.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path

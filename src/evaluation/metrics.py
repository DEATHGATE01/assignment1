from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Iterable


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


@dataclass(slots=True)
class RetrievalMetrics:
    precision_at_k: float
    hit_rate_at_k: float
    mean_reciprocal_rank: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class GenerationMetrics:
    average_relevance: float
    average_grounding_score: float
    hallucination_rate: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class EndToEndMetrics:
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    overall_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "retrieval": self.retrieval.to_dict(),
            "generation": self.generation.to_dict(),
            "overall_score": self.overall_score,
        }


def compute_retrieval_metrics(
    retrieved_source_names: list[list[str]],
    relevant_source_names: list[set[str]],
    k: int = 5,
) -> RetrievalMetrics:
    """Compute retrieval metrics using source-name relevance labels."""

    if len(retrieved_source_names) != len(relevant_source_names):
        raise ValueError("retrieved_source_names and relevant_source_names must have the same length")

    precisions: list[float] = []
    hits: list[float] = []
    reciprocal_ranks: list[float] = []

    for retrieved, relevant in zip(retrieved_source_names, relevant_source_names):
        top_k = retrieved[:k]
        if not top_k:
            precisions.append(0.0)
            hits.append(0.0)
            reciprocal_ranks.append(0.0)
            continue

        match_count = sum(1 for name in top_k if name in relevant)
        precisions.append(_safe_div(match_count, len(top_k)))
        hits.append(1.0 if match_count > 0 else 0.0)

        rr = 0.0
        for rank, name in enumerate(top_k, start=1):
            if name in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return RetrievalMetrics(
        precision_at_k=_safe_div(sum(precisions), len(precisions)),
        hit_rate_at_k=_safe_div(sum(hits), len(hits)),
        mean_reciprocal_rank=_safe_div(sum(reciprocal_ranks), len(reciprocal_ranks)),
    )


def answer_relevance(answer: str, reference_answer: str) -> float:
    """Token Jaccard overlap as a lightweight relevance proxy."""

    answer_tokens = _tokenize(answer)
    reference_tokens = _tokenize(reference_answer)
    if not answer_tokens and not reference_tokens:
        return 1.0
    union = answer_tokens | reference_tokens
    intersection = answer_tokens & reference_tokens
    return _safe_div(len(intersection), len(union))


def grounding_score(answer: str, contexts: Iterable[str]) -> float:
    """Fraction of answer tokens that are supported by retrieved context tokens."""

    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0

    context_tokens: set[str] = set()
    for context in contexts:
        context_tokens |= _tokenize(context)

    grounded_tokens = answer_tokens & context_tokens
    return _safe_div(len(grounded_tokens), len(answer_tokens))


def compute_generation_metrics(
    answers: list[str],
    reference_answers: list[str],
    retrieved_contexts: list[list[str]],
) -> GenerationMetrics:
    """Compute average relevance, grounding, and hallucination estimates."""

    if not (len(answers) == len(reference_answers) == len(retrieved_contexts)):
        raise ValueError("answers, reference_answers, and retrieved_contexts must have the same length")

    relevances: list[float] = []
    groundings: list[float] = []

    for answer, reference, contexts in zip(answers, reference_answers, retrieved_contexts):
        relevances.append(answer_relevance(answer, reference))
        groundings.append(grounding_score(answer, contexts))

    average_relevance = _safe_div(sum(relevances), len(relevances))
    average_grounding = _safe_div(sum(groundings), len(groundings))

    return GenerationMetrics(
        average_relevance=average_relevance,
        average_grounding_score=average_grounding,
        hallucination_rate=1.0 - average_grounding,
    )


def compute_end_to_end_metrics(
    retrieved_source_names: list[list[str]],
    relevant_source_names: list[set[str]],
    answers: list[str],
    reference_answers: list[str],
    retrieved_contexts: list[list[str]],
    k: int = 5,
) -> EndToEndMetrics:
    """Compute retrieval + generation metrics and an aggregate score."""

    retrieval = compute_retrieval_metrics(
        retrieved_source_names=retrieved_source_names,
        relevant_source_names=relevant_source_names,
        k=k,
    )
    generation = compute_generation_metrics(
        answers=answers,
        reference_answers=reference_answers,
        retrieved_contexts=retrieved_contexts,
    )

    overall = (
        (retrieval.precision_at_k + retrieval.hit_rate_at_k + retrieval.mean_reciprocal_rank) / 3.0
        + (generation.average_relevance + generation.average_grounding_score) / 2.0
    ) / 2.0

    return EndToEndMetrics(retrieval=retrieval, generation=generation, overall_score=overall)

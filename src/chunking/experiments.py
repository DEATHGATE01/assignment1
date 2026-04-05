from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.chunking.chunker import Chunk, chunk_by_sentence, chunk_fixed_size, chunk_semantic
from src.pipeline.rag_pipeline import ProcessedDocument
from src.utils.config import RESULTS_DIR, ensure_project_directories


@dataclass(slots=True)
class ChunkingExperimentResult:
	strategy: str
	chunk_size: int
	overlap: int
	document_count: int
	chunk_count: int
	average_chunk_length: float
	min_chunk_length: int
	max_chunk_length: int

	def to_dict(self) -> dict[str, object]:
		return asdict(self)


def _length_stats(chunks: list[Chunk]) -> tuple[float, int, int]:
	lengths = [len(chunk.text.split()) for chunk in chunks]
	if not lengths:
		return 0.0, 0, 0
	return sum(lengths) / len(lengths), min(lengths), max(lengths)


def chunk_documents(
	documents: list[ProcessedDocument],
	*,
	strategy: str = "fixed",
	chunk_size: int = 300,
	overlap: int = 50,
) -> list[Chunk]:
	"""Chunk processed documents using the selected strategy."""

	all_chunks: list[Chunk] = []
	for document in documents:
		if strategy == "fixed":
			chunks = chunk_fixed_size(
				document.text,
				chunk_size=chunk_size,
				overlap=overlap,
				source_name=document.source_name,
				source_path=document.source_path,
				extension=document.extension,
			)
		elif strategy == "sentence":
			chunks = chunk_by_sentence(
				document.text,
				max_tokens=chunk_size,
				source_name=document.source_name,
				source_path=document.source_path,
				extension=document.extension,
			)
		elif strategy == "semantic":
			chunks = chunk_semantic(
				document.text,
				max_tokens=chunk_size,
				overlap=overlap,
				source_name=document.source_name,
				source_path=document.source_path,
				extension=document.extension,
			)
		else:
			raise ValueError(f"Unsupported chunking strategy: {strategy}")
		all_chunks.extend(chunks)
	return all_chunks


def run_chunking_experiment(
	documents: list[ProcessedDocument],
	*,
	strategy: str,
	chunk_size: int = 300,
	overlap: int = 50,
	output_dir: Path | None = None,
) -> ChunkingExperimentResult:
	"""Run a chunking experiment and optionally persist the result."""

	ensure_project_directories()
	output_dir = output_dir or (RESULTS_DIR / "comparisons")
	output_dir.mkdir(parents=True, exist_ok=True)

	chunks = chunk_documents(documents, strategy=strategy, chunk_size=chunk_size, overlap=overlap)
	average_chunk_length, min_chunk_length, max_chunk_length = _length_stats(chunks)

	result = ChunkingExperimentResult(
		strategy=strategy,
		chunk_size=chunk_size,
		overlap=overlap,
		document_count=len(documents),
		chunk_count=len(chunks),
		average_chunk_length=average_chunk_length,
		min_chunk_length=min_chunk_length,
		max_chunk_length=max_chunk_length,
	)

	output_path = output_dir / f"chunking_{strategy}_{chunk_size}_{overlap}.json"
	output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
	return result


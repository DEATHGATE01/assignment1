from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import re
from typing import Iterable


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class Chunk:
	source_name: str
	source_path: str
	extension: str
	chunk_id: int
	chunk_type: str
	text: str
	start_index: int
	end_index: int

	def to_dict(self) -> dict[str, object]:
		return asdict(self)


def _tokenize(text: str) -> list[str]:
	return [token for token in text.split() if token]


def _detokenize(tokens: Iterable[str]) -> str:
	return " ".join(tokens).strip()


def chunk_fixed_size(
	text: str,
	*,
	chunk_size: int = 300,
	overlap: int = 50,
	source_name: str = "unknown",
	source_path: str = "",
	extension: str = "",
) -> list[Chunk]:
	"""Split text into overlapping fixed-size token chunks."""

	if chunk_size <= 0:
		raise ValueError("chunk_size must be greater than 0")
	if overlap < 0:
		raise ValueError("overlap cannot be negative")
	if overlap >= chunk_size:
		raise ValueError("overlap must be smaller than chunk_size")

	tokens = _tokenize(text)
	if not tokens:
		return []

	chunks: list[Chunk] = []
	step = chunk_size - overlap
	start = 0
	chunk_id = 0
	while start < len(tokens):
		end = min(start + chunk_size, len(tokens))
		chunk_tokens = tokens[start:end]
		chunks.append(
			Chunk(
				source_name=source_name,
				source_path=source_path,
				extension=extension,
				chunk_id=chunk_id,
				chunk_type="fixed",
				text=_detokenize(chunk_tokens),
				start_index=start,
				end_index=end,
			)
		)
		chunk_id += 1
		if end >= len(tokens):
			break
		start += step
	return chunks


def chunk_by_sentence(
	text: str,
	*,
	max_tokens: int = 300,
	source_name: str = "unknown",
	source_path: str = "",
	extension: str = "",
) -> list[Chunk]:
	"""Split text into chunks that respect sentence boundaries where possible."""

	if max_tokens <= 0:
		raise ValueError("max_tokens must be greater than 0")

	sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(text) if sentence.strip()]
	if not sentences:
		return []

	chunks: list[Chunk] = []
	current_sentences: list[str] = []
	current_token_count = 0
	chunk_id = 0
	start_sentence_index = 0
	sentence_cursor = 0

	for sentence in sentences:
		sentence_tokens = len(_tokenize(sentence))
		if current_sentences and current_token_count + sentence_tokens > max_tokens:
			chunk_text = " ".join(current_sentences).strip()
			chunks.append(
				Chunk(
					source_name=source_name,
					source_path=source_path,
					extension=extension,
					chunk_id=chunk_id,
					chunk_type="sentence",
					text=chunk_text,
					start_index=start_sentence_index,
					end_index=sentence_cursor,
				)
			)
			chunk_id += 1
			current_sentences = [sentence]
			current_token_count = sentence_tokens
			start_sentence_index = sentence_cursor
		else:
			current_sentences.append(sentence)
			current_token_count += sentence_tokens
		sentence_cursor += 1

	if current_sentences:
		chunks.append(
			Chunk(
				source_name=source_name,
				source_path=source_path,
				extension=extension,
				chunk_id=chunk_id,
				chunk_type="sentence",
				text=" ".join(current_sentences).strip(),
				start_index=start_sentence_index,
				end_index=sentence_cursor,
			)
		)

	return chunks


def chunk_semantic(
	text: str,
	*,
	max_tokens: int = 300,
	overlap: int = 0,
	source_name: str = "unknown",
	source_path: str = "",
	extension: str = "",
) -> list[Chunk]:
	"""Semantic chunking approximation that groups related sentences with light overlap."""

	base_chunks = chunk_by_sentence(
		text,
		max_tokens=max_tokens,
		source_name=source_name,
		source_path=source_path,
		extension=extension,
	)
	if not base_chunks or overlap <= 0:
		return base_chunks

	merged: list[Chunk] = []
	for index, chunk in enumerate(base_chunks):
		if index == 0:
			merged.append(chunk)
			continue
		previous = merged[-1]
		previous_tokens = previous.text.split()
		overlap_tokens = previous_tokens[-overlap:] if len(previous_tokens) > overlap else previous_tokens
		combined_text = f"{_detokenize(overlap_tokens)} {chunk.text}".strip()
		merged.append(
			Chunk(
				source_name=chunk.source_name,
				source_path=chunk.source_path,
				extension=chunk.extension,
				chunk_id=chunk.chunk_id,
				chunk_type="semantic",
				text=combined_text,
				start_index=chunk.start_index,
				end_index=chunk.end_index,
			)
		)
	return merged


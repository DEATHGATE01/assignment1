from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from src.utils.config import SUPPORTED_EXTENSIONS


@dataclass(slots=True)
class RawDocument:
	path: Path
	extension: str
	source_name: str


def iter_source_files(root_dir: Path, recursive: bool = True) -> Iterator[Path]:
	"""Yield supported source files under a directory."""

	if not root_dir.exists():
		return

	pattern = "**/*" if recursive else "*"
	for path in root_dir.glob(pattern):
		if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
			yield path


def load_documents(root_dir: Path, recursive: bool = True) -> list[RawDocument]:
	"""Return supported documents discovered in the raw data directory."""

	documents: list[RawDocument] = []
	for path in iter_source_files(root_dir, recursive=recursive):
		documents.append(
			RawDocument(
				path=path,
				extension=path.suffix.lower(),
				source_name=path.stem,
			)
		)
	return documents


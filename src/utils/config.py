from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx"}
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_DOCUMENT_EXTENSIONS


@dataclass(slots=True)
class IngestionConfig:
	raw_dir: Path = RAW_DATA_DIR
	processed_dir: Path = PROCESSED_DATA_DIR
	recursive: bool = True
	remove_reference_sections: bool = True
	normalize_whitespace: bool = True


def ensure_project_directories() -> None:
	"""Create standard project directories if they do not already exist."""

	for directory in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, LOGS_DIR):
		directory.mkdir(parents=True, exist_ok=True)


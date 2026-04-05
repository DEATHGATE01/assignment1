from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile


def _extract_text_from_pdf(path: Path) -> str:
	"""Extract text from a PDF using PyMuPDF or pypdf if available."""

	try:
		import fitz  # type: ignore

		with fitz.open(path) as doc:
			return "\n".join(page.get_text("text") for page in doc)
	except ImportError:
		pass

	try:
		from pypdf import PdfReader  # type: ignore

		reader = PdfReader(str(path))
		return "\n".join(page.extract_text() or "" for page in reader.pages)
	except ImportError as exc:
		raise ImportError(
			"PDF parsing requires either PyMuPDF (fitz) or pypdf to be installed."
		) from exc


def _extract_text_from_docx(path: Path) -> str:
	"""Extract text from a DOCX file without additional dependencies."""

	with ZipFile(path) as archive:
		document_xml = archive.read("word/document.xml")
	root = ET.fromstring(document_xml)
	namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
	paragraphs: list[str] = []
	for paragraph in root.findall(".//w:p", namespace):
		texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
		if texts:
			paragraphs.append("".join(texts))
	return "\n".join(paragraphs)


def parse_document(path: Path) -> str:
	"""Parse a supported document into plain text."""

	extension = path.suffix.lower()
	if extension in {".txt", ".md", ".markdown"}:
		return path.read_text(encoding="utf-8", errors="ignore")
	if extension == ".pdf":
		return _extract_text_from_pdf(path)
	if extension == ".docx":
		return _extract_text_from_docx(path)
	raise ValueError(f"Unsupported file type: {extension}")


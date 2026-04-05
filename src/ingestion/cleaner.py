from __future__ import annotations

import re


_MULTIPLE_WHITESPACE_RE = re.compile(r"\s+")
_REFERENCE_SECTION_RE = re.compile(
	r"\n(?:references|bibliography|works cited)\b[\s\S]*$", re.IGNORECASE
)
_CITATION_MARKERS_RE = re.compile(
	r"\[(?:\d+(?:\s*,\s*\d+)*)\]|\((?:[^()]*?\d{4}[^()]*)\)",
	re.VERBOSE,
)


def clean_text(text: str, remove_reference_sections: bool = True, normalize_whitespace: bool = True) -> str:
	"""Clean extracted text for downstream chunking and retrieval."""

	cleaned = text.replace("\x00", " ")
	cleaned = _CITATION_MARKERS_RE.sub(" ", cleaned)
	if remove_reference_sections:
		cleaned = _REFERENCE_SECTION_RE.sub("", cleaned)
	if normalize_whitespace:
		cleaned = _MULTIPLE_WHITESPACE_RE.sub(" ", cleaned)
		cleaned = cleaned.replace("\n ", "\n").replace(" \n", "\n")
	return cleaned.strip()


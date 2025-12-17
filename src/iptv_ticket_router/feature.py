from __future__ import annotations

from typing import Iterable, Mapping

from .text_preprocess import normalize_text, safe_join


def build_text_from_row(row: Mapping[str, object], text_fields: Iterable[str]) -> str:
    parts = []
    for f in text_fields:
        v = row.get(f, "")
        parts.append(normalize_text("" if v is None else str(v)))
    return safe_join(parts, sep=" ")

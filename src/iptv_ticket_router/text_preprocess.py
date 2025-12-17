from __future__ import annotations

import re

# Common punctuation + whitespace normalization for Chinese ticket text
_RE_SPACES = re.compile(r"\s+")
_RE_NOISE = re.compile(r"[\u200b\ufeff]")


def normalize_text(text: str) -> str:
    """Normalize input text.

    - Robust to None
    - Collapse whitespace
    - Keep Chinese as-is; do NOT force lowercase (carrier systems often have codes)
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    t = text.strip()
    t = _RE_NOISE.sub("", t)
    t = _RE_SPACES.sub(" ", t)
    return t


def safe_join(parts: list[str], sep: str = " ") -> str:
    return sep.join([p for p in parts if p])

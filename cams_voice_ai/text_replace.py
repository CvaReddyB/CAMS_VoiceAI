"""Apply phrase replacements to text (e.g. TTS transcript post-processing)."""

from __future__ import annotations

from typing import Dict


def apply_replacements(text: str, mapping: Dict[str, str]) -> str:
    if not mapping or not (text or "").strip():
        return text
    out = text
    for old, new in mapping.items():
        if not old:
            continue
        out = out.replace(old, new)
    return out

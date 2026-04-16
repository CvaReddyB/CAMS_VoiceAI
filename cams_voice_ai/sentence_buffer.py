"""Buffer streaming LLM text and flush on sentence boundaries before TTS."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:\s+|$)")


def match_end_of_sentence(text: str) -> bool:
    """True if text ends with a completed sentence (gate before TTS)."""
    t = (text or "").rstrip()
    if len(t) < 2:
        return False
    return bool(_SENTENCE_BOUNDARY.search(t))


@dataclass
class SentenceBuffer:
    """Accumulates deltas; ``pop_flushed_sentences`` returns completed sentences."""

    _buf: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self._buf = ""

    def append(self, piece: str) -> None:
        self._buf += piece or ""

    def pop_flushed_sentences(self) -> list[str]:
        out: list[str] = []
        s = self._buf
        i = 0
        while i < len(s):
            ch = s[i]
            if ch in ".!?":
                if i + 1 >= len(s) or s[i + 1].isspace():
                    seg = s[: i + 1].strip()
                    if seg:
                        out.append(seg)
                    s = s[i + 1 :].lstrip()
                    i = 0
                    continue
            i += 1
        self._buf = s
        return out

    def flush_remainder(self) -> str:
        r = self._buf.strip()
        self._buf = ""
        return r

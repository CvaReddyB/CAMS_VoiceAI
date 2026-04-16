"""Deepgram pre-recorded transcription (HTTP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class ASRResult:
    transcript: str
    confidence: Optional[float]
    raw: dict


class DeepgramASR:
    def __init__(self, api_key: str, model: str = "nova-2") -> None:
        self.api_key = api_key
        self.model = model
        self._url = "https://api.deepgram.com/v1/listen"

    def transcribe_wav(self, wav_bytes: bytes, timeout: float = 60.0) -> ASRResult:
        params = {
            "model": self.model,
            "smart_format": "true",
            "language": "en",
        }
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav",
        }
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(self._url, params=params, content=wav_bytes, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        transcript, confidence = _parse_deepgram_json(data)
        return ASRResult(transcript=transcript, confidence=confidence, raw=data)


def _parse_deepgram_json(data: dict) -> tuple[str, Optional[float]]:
    try:
        alts = (
            data.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])
        )
        best = alts[0] if alts else {}
        text = (best.get("transcript") or "").strip()
        conf = best.get("confidence")
        if conf is not None:
            conf = float(conf)
        return text, conf
    except (KeyError, IndexError, TypeError, ValueError):
        return "", None

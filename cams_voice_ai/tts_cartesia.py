"""Cartesia TTS: SDK or HTTP, WAV batch and raw PCM SSE streaming."""

from __future__ import annotations

import base64
import binascii
import json
import logging
from dataclasses import dataclass
from typing import Iterator, Optional

import httpx

logger = logging.getLogger(__name__)

try:
    from cartesia import Cartesia
    from cartesia.types.sse_events import ChunkEvent
except ImportError:
    Cartesia = None
    ChunkEvent = None


def _patch_cartesia_tts_generate_params_pep563() -> None:
    try:
        import cartesia.types.tts_generate_params as _tg
        from cartesia.types.raw_encoding import RawEncoding
    except ImportError:
        return
    setattr(_tg, "RawEncoding", RawEncoding)


if Cartesia is not None:
    _patch_cartesia_tts_generate_params_pep563()

CARTESIA_BYTES_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_SSE_URL = "https://api.cartesia.ai/tts/sse"


@dataclass
class TTSResult:
    audio_bytes: bytes
    content_type: Optional[str]


class CartesiaTTS:
    def __init__(
        self,
        api_key: str,
        model_id: str,
        voice_id: str,
        api_version: str = "2025-04-16",
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.voice_id = voice_id
        self.api_version = api_version
        self._voice = {"mode": "id", "id": voice_id}
        self._sdk: Optional[Cartesia] = None
        if Cartesia is not None:
            self._sdk = Cartesia(
                api_key=api_key,
                default_headers={"Cartesia-Version": api_version},
            )
        else:
            logger.warning("Cartesia SDK not installed; using HTTP.")

    def synthesize(self, text: str, timeout: float = 120.0) -> TTSResult:
        if self._sdk is not None:
            try:
                resp = self._sdk.tts.generate(
                    model_id=self.model_id,
                    transcript=text,
                    voice=self._voice,
                    output_format={
                        "container": "wav",
                        "encoding": "pcm_s16le",
                        "sample_rate": 24000,
                    },
                    timeout=timeout,
                )
                data = b"".join(resp.iter_bytes())
                return TTSResult(
                    audio_bytes=data,
                    content_type=resp.headers.get("content-type", "audio/wav"),
                )
            except Exception as exc:
                logger.warning("Cartesia SDK generate failed (%s); HTTP fallback.", exc)
        return _synthesize_httpx(
            self.api_key,
            self.model_id,
            self.voice_id,
            self.api_version,
            text,
            timeout=timeout,
        )

    def iter_synthesize_sse(
        self,
        text: str,
        *,
        timeout: float = 120.0,
        chunk_decode: str = "pcm_s16le",
    ) -> Iterator[bytes]:
        if self._sdk is not None:
            try:
                stream = self._sdk.tts.generate_sse(
                    model_id=self.model_id,
                    transcript=text,
                    voice=self._voice,
                    output_format={
                        "container": "raw",
                        "encoding": chunk_decode,
                        "sample_rate": 24000,
                    },
                    timeout=timeout,
                )
                for event in stream:
                    if ChunkEvent is not None and isinstance(event, ChunkEvent):
                        audio = event.audio
                        if audio:
                            yield audio
                    elif getattr(event, "type", None) == "chunk":
                        audio = getattr(event, "audio", None)
                        if audio:
                            yield audio
                return
            except Exception as exc:
                logger.warning("Cartesia SSE failed (%s); HTTP SSE.", exc)

        yield from _iter_synthesize_sse_httpx(
            self.api_key,
            self.model_id,
            self.voice_id,
            self.api_version,
            text,
            timeout=timeout,
            chunk_decode=chunk_decode,
        )


def _synthesize_httpx(
    api_key: str,
    model_id: str,
    voice_id: str,
    api_version: str,
    text: str,
    *,
    timeout: float,
) -> TTSResult:
    body = {
        "transcript": text,
        "model_id": model_id,
        "voice": {"mode": "id", "id": voice_id},
        "output_format": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Cartesia-Version": api_version,
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(CARTESIA_BYTES_URL, json=body, headers=headers)
        resp.raise_for_status()
        return TTSResult(
            audio_bytes=resp.content,
            content_type=resp.headers.get("content-type"),
        )


def _iter_synthesize_sse_httpx(
    api_key: str,
    model_id: str,
    voice_id: str,
    api_version: str,
    text: str,
    *,
    timeout: float,
    chunk_decode: str,
) -> Iterator[bytes]:
    body = {
        "transcript": text,
        "model_id": model_id,
        "voice": {"mode": "id", "id": voice_id},
        "output_format": {
            "container": "raw",
            "encoding": chunk_decode,
            "sample_rate": 24000,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Cartesia-Version": api_version,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", CARTESIA_SSE_URL, json=body, headers=headers) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw in ("", "[DONE]"):
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                pcm = _pcm_from_cartesia_sse_payload(obj)
                if pcm:
                    yield pcm


def _pcm_from_cartesia_sse_payload(obj: object) -> Optional[bytes]:
    if not isinstance(obj, dict):
        return None
    for key in ("data", "chunk", "audio", "pcm", "audio_data"):
        val = obj.get(key)
        if isinstance(val, str) and len(val) > 8:
            try:
                return base64.b64decode(val)
            except (binascii.Error, ValueError):
                continue
    if obj.get("type") in ("chunk", "audio", "audio_chunk") and isinstance(
        obj.get("data"), str
    ):
        try:
            return base64.b64decode(obj["data"])
        except (binascii.Error, ValueError, KeyError, TypeError):
            return None
    return None

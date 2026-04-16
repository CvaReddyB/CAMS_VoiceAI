"""Cartesia TTS over WebSocket: one connection, context_id + continue + max_buffer_delay_ms."""

from __future__ import annotations

import base64
import json
import logging
import uuid
from typing import Callable, Iterator, Optional
from urllib.parse import urlencode

from cams_voice_ai.config import Settings

logger = logging.getLogger(__name__)

try:
    import websocket
except ImportError:
    websocket = None

from cams_voice_ai.sentence_buffer import SentenceBuffer
from cams_voice_ai.text_replace import apply_replacements

WS_URL = "wss://api.cartesia.ai/tts/websocket"


class CartesiaTTSWebSocket:
    """
    JSON text frames over WebSocket, raw pcm_s16le,
    generation_config.speed, max_buffer_delay_ms, context_id + continue for streaming inputs.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_version: str,
        model_id: str,
        voice_id: str,
        sample_rate: int = 24000,
        encoding: str = "pcm_s16le",
        voice_speed: float = 1.0,
        max_buffer_delay_ms: int = 50,
        replacement_texts: Optional[dict[str, str]] = None,
    ) -> None:
        self.api_key = api_key
        self.api_version = api_version
        self.model_id = model_id
        self.voice_id = voice_id
        self.sample_rate = int(sample_rate)
        self.encoding = encoding
        self.voice_speed = float(voice_speed)
        self.max_buffer_delay_ms = int(max_buffer_delay_ms)
        self.replacement_texts = replacement_texts or {}
        self._ws: Optional[websocket.WebSocket] = None

    @classmethod
    def from_settings(cls, s: Settings) -> "CartesiaTTSWebSocket":
        return cls(
            api_key=s.cartesia_api_key or "",
            api_version=s.cartesia_api_version,
            model_id=s.cartesia_model_id,
            voice_id=s.cartesia_voice_id,
            sample_rate=s.cartesia_sample_rate,
            encoding=s.cartesia_output_encoding,
            voice_speed=s.cartesia_voice_speed,
            max_buffer_delay_ms=s.cartesia_max_buffer_delay_ms,
            replacement_texts=s.tts_replacement_texts,
        )

    def connect(self, timeout: float = 30.0) -> None:
        if websocket is None:
            raise RuntimeError("Install websocket-client for Cartesia WS TTS.")
        if self._ws is not None:
            return
        q = f"{WS_URL}?{urlencode({'api_key': self.api_key})}"
        headers = [f"Cartesia-Version: {self.api_version}"]
        self._ws = websocket.create_connection(q, header=headers, timeout=timeout)
        logger.info("Cartesia TTS WebSocket connected")

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _require_ws(self) -> websocket.WebSocket:
        if self._ws is None:
            raise RuntimeError("CartesiaTTSWebSocket.connect() was not called.")
        return self._ws

    def _build_request(
        self,
        transcript: str,
        *,
        context_id: str,
        continue_: bool,
        flush: bool,
        add_timestamps: bool,
    ) -> dict:
        body: dict = {
            "model_id": self.model_id,
            "transcript": transcript,
            "voice": {"mode": "id", "id": self.voice_id},
            "output_format": {
                "container": "raw",
                "encoding": self.encoding,
                "sample_rate": self.sample_rate,
            },
            "context_id": context_id,
            "continue": continue_,
            "max_buffer_delay_ms": self.max_buffer_delay_ms,
            "flush": flush,
            "add_timestamps": add_timestamps,
        }
        if "sonic-3" in (self.model_id or "").lower():
            body["generation_config"] = {
                "speed": max(0.6, min(1.5, self.voice_speed)),
            }
        return body

    def _read_audio_for_context(self, context_id: str) -> Iterator[bytes]:
        ws = self._require_ws()
        while True:
            try:
                raw = ws.recv()
            except Exception as exc:
                logger.warning("Cartesia WS recv ended: %s", exc)
                break
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            cid = msg.get("context_id")
            if cid and cid != context_id:
                continue
            mtype = msg.get("type")
            if mtype == "chunk":
                data = msg.get("data")
                if isinstance(data, str) and data:
                    try:
                        yield base64.b64decode(data)
                    except Exception:
                        pass
                if msg.get("done") is True:
                    break
            elif mtype == "done":
                break
            elif mtype == "error":
                err = msg.get("error") or msg
                logger.error("Cartesia TTS WS error: %s", err)
                break
            elif mtype in ("timestamps", "phoneme_timestamps"):
                continue

    def iter_pcm_plain(self, text: str) -> Iterator[bytes]:
        """Single-shot synthesis (KYC / welcome): new context, flush at end."""
        t = apply_replacements((text or "").strip(), self.replacement_texts)
        if not t:
            return
        ctx = f"cams-{uuid.uuid4().hex[:16]}"
        ws = self._require_ws()
        req = self._build_request(
            t,
            context_id=ctx,
            continue_=False,
            flush=True,
            add_timestamps=False,
        )
        ws.send(json.dumps(req))
        yield from self._read_audio_for_context(ctx)

    def iter_pcm_llm_stream(
        self,
        llm_token_iter: Iterator[str],
        *,
        on_llm_delta: Optional[Callable[[str], None]] = None,
    ) -> Iterator[bytes]:
        """
        Stream LLM tokens → sentence chunks → Cartesia WebSocket, one context per turn.
        """
        ctx = f"cams-{uuid.uuid4().hex[:16]}"
        ws = self._require_ws()
        buf = SentenceBuffer()
        sent_any = False
        for piece in llm_token_iter:
            if on_llm_delta:
                on_llm_delta(piece)
            buf.append(piece)
            for sent in buf.pop_flushed_sentences():
                sent = apply_replacements(sent, self.replacement_texts)
                if not sent:
                    continue
                req = self._build_request(
                    sent,
                    context_id=ctx,
                    continue_=sent_any,
                    flush=False,
                    add_timestamps=False,
                )
                ws.send(json.dumps(req))
                yield from self._read_audio_for_context(ctx)
                sent_any = True
        tail = apply_replacements(buf.flush_remainder(), self.replacement_texts)
        if tail:
            req = self._build_request(
                tail,
                context_id=ctx,
                continue_=sent_any,
                flush=True,
                add_timestamps=False,
            )
            ws.send(json.dumps(req))
            yield from self._read_audio_for_context(ctx)

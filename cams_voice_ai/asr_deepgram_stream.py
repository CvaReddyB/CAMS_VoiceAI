"""Deepgram live ASR over WebSocket (configurable query params + interim/final)."""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, Iterator, Optional, Tuple
from urllib.parse import urlencode

from cams_voice_ai.audio_linear16 import wav_to_linear16_mono

logger = logging.getLogger(__name__)

try:
    import websocket
except ImportError:
    websocket = None


def _parse_transcript(message: str) -> Tuple[str, bool, Optional[float]]:
    """Parse Deepgram JSON (Results); treats is_final and speech_final as finals."""
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return "", False, None

    if isinstance(data, dict) and data.get("type") == "Results":
        ch = data.get("channel") or {}
        alts = ch.get("alternatives") or []
        if not alts:
            is_final = bool(data.get("is_final") or data.get("speech_final"))
            return "", is_final, None
        best = alts[0]
        text = (best.get("transcript") or "").strip()
        conf = best.get("confidence")
        if conf is not None:
            conf = float(conf)
        is_final = bool(data.get("is_final", False) or data.get("speech_final", False))
        return text, is_final, conf

    if "channel" in data:
        ch = data["channel"]
        alts = ch.get("alternatives") or []
        if alts:
            best = alts[0]
            text = (best.get("transcript") or "").strip()
            conf = best.get("confidence")
            if conf is not None:
                conf = float(conf)
            is_final = bool(data.get("is_final", False) or data.get("speech_final", False))
            return text, is_final, conf

    return "", False, None


class DeepgramStreamASR:
    def __init__(
        self,
        api_key: str,
        model: str = "nova-2",
        *,
        listen_params: Optional[Dict[str, str]] = None,
        stream_chunk_ms: int = 20,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.listen_params = listen_params
        self.stream_chunk_ms = max(10, int(stream_chunk_ms))

    def iter_transcribe_wav(
        self,
        wav_bytes: bytes,
        *,
        chunk_ms: Optional[int] = None,
    ) -> Iterator[dict]:
        if websocket is None:
            raise RuntimeError(
                "Install websocket-client: pip install websocket-client"
            )

        pcm, sr = wav_to_linear16_mono(wav_bytes, target_sr=16000)
        params: Dict[str, str] = dict(self.listen_params or {})
        params.setdefault("model", self.model)
        params["encoding"] = "linear16"
        params["sample_rate"] = str(sr)
        params.setdefault("channels", "1")
        params.setdefault("smart_format", "true")
        params.setdefault("interim_results", "true")

        uri = f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"
        headers = [f"Authorization: Token {self.api_key}"]

        ws = websocket.create_connection(uri, header=headers, timeout=120)
        use_ms = int(chunk_ms) if chunk_ms is not None else self.stream_chunk_ms
        bytes_per_chunk = max(int(sr * 2 * (use_ms / 1000.0)), 320)
        t0 = time.perf_counter()

        try:
            for i in range(0, len(pcm), bytes_per_chunk):
                ws.send_binary(pcm[i : i + bytes_per_chunk])
            ws.send(json.dumps({"type": "CloseStream"}))

            last_final = ""
            last_partial = ""
            while True:
                try:
                    msg = ws.recv()
                except websocket.WebSocketConnectionClosedException:
                    break
                if not msg:
                    continue
                if isinstance(msg, bytes):
                    msg = msg.decode("utf-8", errors="replace")
                if not isinstance(msg, str):
                    continue
                text, is_final, conf = _parse_transcript(msg)
                if not text:
                    continue
                if is_final:
                    last_final = text
                    yield {
                        "type": "asr_final",
                        "transcript": text,
                        "confidence": conf,
                        "elapsed_ms": (time.perf_counter() - t0) * 1000,
                    }
                else:
                    last_partial = text
                    yield {
                        "type": "asr_partial",
                        "transcript": text,
                        "confidence": conf,
                        "elapsed_ms": (time.perf_counter() - t0) * 1000,
                    }
        finally:
            try:
                ws.close()
            except Exception:
                pass

        if not last_final and last_partial:
            logger.warning("Deepgram closed without final; using last partial.")
            yield {
                "type": "asr_final",
                "transcript": last_partial,
                "confidence": None,
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            }

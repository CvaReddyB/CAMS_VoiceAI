"""Environment-driven settings (no secrets in code)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


def _json_object(name: str) -> Dict[str, str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        pass
    return {}


@dataclass(frozen=True)
class Settings:
    deepgram_api_key: Optional[str]
    deepgram_model: str
    deepgram_language: str
    deepgram_endpointing_ms: int
    deepgram_no_delay: bool
    deepgram_stream_chunk_ms: int
    openai_api_key: Optional[str]
    openai_model: str
    openai_temperature: float
    openai_stream_include_usage: bool
    openai_reasoning_effort: Optional[str]
    cartesia_api_key: Optional[str]
    cartesia_model_id: str
    cartesia_voice_id: str
    cartesia_api_version: str
    cartesia_transport: str
    cartesia_voice_speed: float
    cartesia_max_buffer_delay_ms: int
    cartesia_sample_rate: int
    cartesia_output_encoding: str
    tts_replacement_texts: Dict[str, str]
    sentence_transformer_model: str
    vad_threshold: float
    vad_min_speech_ms: int
    vad_min_silence_duration_ms: int
    vad_speech_pad_ms: int
    mock_registered_caller: bool
    mock_kyc_phone_last4: str
    mock_kyc_pan_last4: str
    mock_kyc_dob: str
    mic_speech_rms_threshold: float
    log_level: str

    def deepgram_listen_params(self, *, sample_rate: int) -> Dict[str, str]:
        """Query string map for wss://api.deepgram.com/v1/listen."""
        p: Dict[str, str] = {
            "model": self.deepgram_model,
            "encoding": "linear16",
            "sample_rate": str(int(sample_rate)),
            "channels": "1",
            "smart_format": "true",
            "interim_results": "true",
        }
        if (self.deepgram_language or "").strip():
            p["language"] = self.deepgram_language.strip()
        if self.deepgram_endpointing_ms > 0:
            p["endpointing"] = str(self.deepgram_endpointing_ms)
        if self.deepgram_no_delay:
            p["no_delay"] = "true"
        return p

    @staticmethod
    def from_env() -> "Settings":
        ep = int(os.getenv("DEEPGRAM_ENDPOINTING_MS", "300") or "300")
        return Settings(
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY") or None,
            deepgram_model=os.getenv("DEEPGRAM_MODEL", "nova-2"),
            deepgram_language=os.getenv("DEEPGRAM_LANGUAGE", "en").strip(),
            deepgram_endpointing_ms=ep,
            deepgram_no_delay=_env_bool("DEEPGRAM_NO_DELAY", "true"),
            deepgram_stream_chunk_ms=int(os.getenv("DEEPGRAM_STREAM_CHUNK_MS", "20")),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.45")),
            openai_stream_include_usage=_env_bool("OPENAI_STREAM_INCLUDE_USAGE", "true"),
            openai_reasoning_effort=(os.getenv("OPENAI_REASONING_EFFORT") or "").strip()
            or None,
            cartesia_api_key=os.getenv("CARTESIA_API_KEY") or None,
            cartesia_model_id=os.getenv("CARTESIA_MODEL_ID", "sonic-2"),
            cartesia_voice_id=os.getenv(
                "CARTESIA_VOICE_ID",
                "694f9389-aac1-45b6-b726-9d9369183238",
            ),
            cartesia_api_version=os.getenv("CARTESIA_API_VERSION", "2025-04-16"),
            cartesia_transport=os.getenv("CARTESIA_TRANSPORT", "websocket")
            .strip()
            .lower(),
            cartesia_voice_speed=float(os.getenv("CARTESIA_VOICE_SPEED", "1.0")),
            cartesia_max_buffer_delay_ms=int(
                os.getenv("CARTESIA_MAX_BUFFER_DELAY_MS", "50")
            ),
            cartesia_sample_rate=int(os.getenv("CARTESIA_SAMPLE_RATE", "24000")),
            cartesia_output_encoding=os.getenv(
                "CARTESIA_OUTPUT_ENCODING", "pcm_s16le"
            ).strip(),
            tts_replacement_texts=_json_object("TTS_REPLACEMENT_JSON"),
            sentence_transformer_model=os.getenv(
                "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
            ),
            vad_threshold=float(os.getenv("SILERO_VAD_THRESHOLD", "0.8")),
            vad_min_speech_ms=int(os.getenv("SILERO_MIN_SPEECH_MS", "200")),
            vad_min_silence_duration_ms=int(
                os.getenv("SILERO_MIN_SILENCE_DURATION_MS", "400")
            ),
            vad_speech_pad_ms=int(os.getenv("SILERO_SPEECH_PAD_MS", "64")),
            mock_registered_caller=os.getenv("MOCK_REGISTERED_CALLER", "false").lower()
            in ("1", "true", "yes"),
            mock_kyc_phone_last4=os.getenv("MOCK_KYC_PHONE_LAST4", "9876"),
            mock_kyc_pan_last4=os.getenv("MOCK_KYC_PAN_LAST4", "1234"),
            mock_kyc_dob=os.getenv("MOCK_KYC_DOB", "1990-08-15"),
            mic_speech_rms_threshold=float(
                os.getenv("MIC_SPEECH_RMS_THRESHOLD", "0.012") or "0.012"
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


def missing_voice_api_env(settings: Settings) -> list[str]:
    """Return env var names that are unset or blank (ASR / LLM / TTS are always real APIs)."""
    need: list[tuple[str, Optional[str]]] = [
        ("DEEPGRAM_API_KEY", settings.deepgram_api_key),
        ("OPENAI_API_KEY", settings.openai_api_key),
        ("CARTESIA_API_KEY", settings.cartesia_api_key),
    ]
    out: list[str] = []
    for name, val in need:
        if not (val or "").strip():
            out.append(name)
    return out

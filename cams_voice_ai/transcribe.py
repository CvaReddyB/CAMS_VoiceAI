"""VAD + ASR only (KYC and utilities). Deepgram only — no mock transcripts."""

from __future__ import annotations

import logging

from cams_voice_ai.asr_deepgram import DeepgramASR
from cams_voice_ai.asr_deepgram_stream import DeepgramStreamASR
from cams_voice_ai.vad_silero import SileroVAD, VADResult

logger = logging.getLogger(__name__)


def vad_analyze(vad: SileroVAD, wav_bytes: bytes, skip_vad: bool) -> VADResult:
    if skip_vad:
        return VADResult(
            has_speech=True,
            speech_ratio=1.0,
            segments=[],
            speech_wav_bytes=wav_bytes,
            used_fallback=True,
        )
    return vad.analyze(wav_bytes)


def asr_final_from_speech(
    speech_wav: bytes,
    asr: DeepgramASR,
    asr_stream: DeepgramStreamASR,
) -> str:
    transcript = ""
    try:
        for ev in asr_stream.iter_transcribe_wav(speech_wav):
            if ev["type"] == "asr_final":
                transcript = ev.get("transcript") or ""
    except Exception as exc:
        logger.exception("Deepgram streaming ASR failed: %s", exc)
        transcript = asr.transcribe_wav(speech_wav).transcript
    return (transcript or "").strip()


def transcribe_wav(
    vad: SileroVAD,
    asr: DeepgramASR,
    asr_stream: DeepgramStreamASR,
    wav_bytes: bytes,
    *,
    skip_vad: bool,
) -> str:
    vad_res = vad_analyze(vad, wav_bytes, skip_vad)
    if not vad_res.has_speech:
        return ""
    return asr_final_from_speech(vad_res.speech_wav_bytes, asr, asr_stream)

"""Local voice activity detection using Silero VAD (torch.hub)."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import soundfile as sf
import torch
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

TARGET_SR = 16000


@dataclass
class VADSegment:
    start_sample: int
    end_sample: int


@dataclass
class VADResult:
    has_speech: bool
    speech_ratio: float
    segments: List[VADSegment]
    speech_wav_bytes: bytes
    sample_rate: int = TARGET_SR
    used_fallback: bool = False


def _to_mono_float(wav: np.ndarray) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32, copy=False)


def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    num = int(len(wav) * target_sr / orig_sr)
    return scipy_signal.resample(wav, num).astype(np.float32)


def _wav_bytes_from_array(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class SileroVAD:
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 200,
        *,
        min_silence_duration_ms: int = 400,
        speech_pad_ms: int = 64,
    ) -> None:
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_duration_ms = max(0, int(min_silence_duration_ms))
        self.speech_pad_ms = max(0, int(speech_pad_ms))
        self._model = None
        self._utils = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            self._model = model
            self._utils = utils
            logger.info("Silero VAD model loaded via torch.hub")
            return True
        except Exception as exc:
            logger.warning("Silero VAD load failed (%s); using passthrough.", exc)
            return False

    def warm(self) -> None:
        """Load Silero weights via torch.hub before the first real utterance."""
        self._ensure_model()

    def analyze(self, audio_wav_bytes: bytes) -> VADResult:
        try:
            buf = io.BytesIO(audio_wav_bytes)
            wav, sr = sf.read(buf)
        except Exception as exc:
            logger.warning("Could not parse audio as WAV (%s); passthrough.", exc)
            return VADResult(
                has_speech=True,
                speech_ratio=1.0,
                segments=[],
                speech_wav_bytes=audio_wav_bytes,
                sample_rate=TARGET_SR,
                used_fallback=True,
            )

        wav = _to_mono_float(wav)
        wav = _resample(wav, int(sr), TARGET_SR)
        sr = TARGET_SR

        if not self._ensure_model():
            return VADResult(
                has_speech=True,
                speech_ratio=1.0,
                segments=[],
                speech_wav_bytes=_wav_bytes_from_array(wav, sr),
                sample_rate=sr,
                used_fallback=True,
            )

        get_speech_timestamps, _, _, _, collect_chunks = self._utils
        wav_t = torch.from_numpy(wav)
        ts_kwargs = {
            "sampling_rate": sr,
            "threshold": self.threshold,
            "min_speech_duration_ms": self.min_speech_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
        }
        try:
            timestamps = get_speech_timestamps(wav_t, self._model, **ts_kwargs)
        except TypeError:
            timestamps = get_speech_timestamps(
                wav_t,
                self._model,
                sampling_rate=sr,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_ms,
            )
        if not timestamps:
            return VADResult(
                has_speech=False,
                speech_ratio=0.0,
                segments=[],
                speech_wav_bytes=_wav_bytes_from_array(
                    np.zeros(160, dtype=np.float32), sr
                ),
                sample_rate=sr,
                used_fallback=False,
            )

        merged_t = collect_chunks(timestamps, wav_t)
        merged = merged_t.numpy().astype(np.float32)
        speech_samples = int(merged.shape[0])
        speech_ratio = float(speech_samples / max(len(wav), 1))

        segments = [
            VADSegment(start_sample=int(t["start"]), end_sample=int(t["end"]))
            for t in timestamps
        ]

        return VADResult(
            has_speech=True,
            speech_ratio=speech_ratio,
            segments=segments,
            speech_wav_bytes=_wav_bytes_from_array(merged, sr),
            sample_rate=sr,
            used_fallback=False,
        )

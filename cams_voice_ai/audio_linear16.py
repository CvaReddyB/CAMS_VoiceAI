"""Convert WAV bytes to linear16 mono PCM for Deepgram live streaming."""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy import signal as scipy_signal


def wav_to_linear16_mono(
    wav_bytes: bytes, target_sr: int = 16000
) -> Tuple[bytes, int]:
    buf = io.BytesIO(wav_bytes)
    wav, sr = sf.read(buf, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    sr = int(sr)
    if sr != target_sr:
        num = int(len(wav) * target_sr / sr)
        wav = scipy_signal.resample(wav, num).astype(np.float32)
        sr = target_sr
    pcm = np.clip(wav * 32767.0, -32768, 32767).astype("<i2").tobytes()
    return pcm, sr

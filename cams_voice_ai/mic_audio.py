"""Microphone capture and playback (PyAudio + PortAudio)."""

from __future__ import annotations

import io
import os
import sys
import time
from typing import Iterable, Iterator, Optional

import numpy as np
import soundfile as sf
from scipy import signal as scipy_signal

try:
    import pyaudio
except ImportError as exc:
    pyaudio = None  # type: ignore[assignment]
    _PYAUDIO_IMPORT_ERROR = exc
else:
    _PYAUDIO_IMPORT_ERROR = None


def require_pyaudio() -> None:
    if pyaudio is None:
        raise RuntimeError(
            "Install PyAudio for microphone capture and playback:\n"
            "  pip install PyAudio\n"
            "macOS may need: brew install portaudio && pip install PyAudio"
        ) from _PYAUDIO_IMPORT_ERROR


def list_input_devices() -> None:
    require_pyaudio()
    pa = pyaudio.PyAudio()
    try:
        try:
            d = pa.get_default_input_device_info()
            print(
                f"Default input: [{d['index']}] {d['name']}  "
                f"defaultSR={int(d.get('defaultSampleRate', 0))}\n",
                flush=True,
            )
        except OSError:
            print("(No default input device)\n", flush=True)
        for i in range(pa.get_device_count()):
            inf = pa.get_device_info_by_index(i)
            if int(inf.get("maxInputChannels", 0)) < 1:
                continue
            print(
                f"[{i}] {inf['name']}  in_ch={inf['maxInputChannels']}  "
                f"defaultSR={int(inf.get('defaultSampleRate', 0))}",
                flush=True,
            )
        print("\nUse --mic-device <index> to select an input (PyAudio indices).")
    finally:
        pa.terminate()


def _mic_pre_record_delay_sec() -> float:
    try:
        return max(0.0, float(os.getenv("MIC_PRE_RECORD_DELAY_SEC", "0.35")))
    except ValueError:
        return 0.35


def _pa_input_device_index(pa: object, device: Optional[int]) -> Optional[int]:
    if device is not None:
        return int(device)
    try:
        return int(pa.get_default_input_device_info()["index"])
    except OSError:
        return None


def _write_output_stream(stream: object, wav: np.ndarray) -> None:
    """Write mono float32 samples in chunks (blocking)."""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = np.ascontiguousarray(wav.astype(np.float32, copy=False))
    chunk = 2048
    for i in range(0, wav.size, chunk):
        part = wav[i : i + chunk]
        stream.write(part.tobytes())


def record_wav_bytes_until_silence(
    *,
    max_duration_sec: float = 10.0,
    min_speech_duration_sec: float = 0.15,
    silence_duration_sec: float = 0.85,
    speech_rms_threshold: float = 0.012,
    sample_rate: int = 16000,
    block_samples: int = 1024,
    device: Optional[int] = None,
) -> bytes:
    require_pyaudio()
    pa = pyaudio.PyAudio()
    stream = None
    try:
        idx = _pa_input_device_index(pa, device)
        if idx is None:
            raise RuntimeError("No input device available (check mic permissions).")
        dev_info = pa.get_device_info_by_index(idx)
        dev_name = str(dev_info.get("name", "?"))
        sr_native = int(dev_info.get("defaultSampleRate") or 48000)
        if not (8000 <= sr_native <= 192000):
            sr_native = 48000

        sr_in = sr_native
        max_frames = max(block_samples, int(max_duration_sec * sr_in))
        min_frames_after_speech = int(min_speech_duration_sec * sr_in)
        silence_frames_needed = int(silence_duration_sec * sr_in)
        print(
            f"\n● Recording (silence ~{silence_duration_sec:.1f}s ends capture, "
            f"max {max_duration_sec:.1f}s).\n"
            f"● Mic: [{idx}] {dev_name}  (capture {sr_in} Hz → WAV {sample_rate} Hz, "
            f"RMS threshold {speech_rms_threshold:g})\n",
            file=sys.stderr,
        )

        delay = _mic_pre_record_delay_sec()
        if delay > 0:
            time.sleep(delay)
        print("● Listening…\n", file=sys.stderr, flush=True)

        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sr_in,
            input=True,
            input_device_index=idx,
            frames_per_buffer=block_samples,
        )

        chunks: list = []
        total_frames = 0
        speech_frames = 0
        silent_run_frames = 0
        speech_seen = False

        while total_frames < max_frames:
            n = min(block_samples, max_frames - total_frames)
            assert stream is not None
            raw = stream.read(n, exception_on_overflow=False)
            need = n * 4
            if len(raw) < need:
                raw = raw + b"\x00" * (need - len(raw))
            block = np.frombuffer(raw[:need], dtype="<f4").copy()
            chunks.append(block)
            total_frames += n
            rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
            if rms >= speech_rms_threshold:
                speech_seen = True
                speech_frames += n
                silent_run_frames = 0
            elif speech_seen:
                silent_run_frames += n
                if (
                    speech_frames >= min_frames_after_speech
                    and silent_run_frames >= silence_frames_needed
                ):
                    break

        audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        if audio.size and sr_in != sample_rate:
            num = max(1, int(len(audio) * float(sample_rate) / float(sr_in)))
            audio = scipy_signal.resample(audio, num).astype(np.float32)

        if audio.size:
            peak = float(np.max(np.abs(audio)))
            rms_all = float(np.sqrt(np.mean(np.square(audio))))
            if peak < 0.003:
                print(
                    f"● Warning: capture is very quiet (peak={peak:.4f}, RMS={rms_all:.4f}). "
                    "Check System Settings → Privacy → Microphone, try `--mic-device`, "
                    "or lower MIC_SPEECH_RMS_THRESHOLD in .env.",
                    file=sys.stderr,
                )
        if not speech_seen:
            print(
                "● Warning: volume never crossed the speech threshold during capture; "
                "the clip may be silence. Speak closer to the mic or lower MIC_SPEECH_RMS_THRESHOLD.",
                file=sys.stderr,
            )

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        print("● Done.\n", file=sys.stderr)
        return buf.getvalue()
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        pa.terminate()


def play_audio_bytes(data: bytes, *, default_pcm_sr: int = 24000) -> None:
    require_pyaudio()
    pa = pyaudio.PyAudio()
    stream = None
    try:
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            wav, sr = sf.read(io.BytesIO(data), dtype="float32")
            sr = int(sr)
        else:
            pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            if pcm.size == 0:
                print("Nothing to play.", file=sys.stderr)
                return
            wav = pcm
            sr = int(default_pcm_sr)

        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sr,
            output=True,
            frames_per_buffer=2048,
        )
        _write_output_stream(stream, wav)
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        pa.terminate()


def play_pcm_s16le_chunks(
    chunks: Iterable[bytes],
    *,
    sample_rate: int = 24000,
) -> None:
    """
    Play raw pcm_s16le mono in one continuous output stream.

    One PyAudio output stream for the whole utterance avoids gaps between chunks.
    """
    require_pyaudio()
    sr = int(sample_rate)
    if sr <= 0:
        sr = 24000

    it: Iterator[bytes] = iter(chunks)
    first = b""
    for piece in it:
        if piece:
            first = piece
            break
    if not first:
        return

    rest = it

    def _all_pieces() -> Iterator[bytes]:
        yield first
        yield from rest

    pa = pyaudio.PyAudio()
    stream = None
    try:
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sr,
            output=True,
            frames_per_buffer=2048,
        )
        for data in _all_pieces():
            if not data:
                continue
            n = len(data) & ~1
            if n <= 0:
                continue
            pcm = np.frombuffer(data[:n], dtype="<i2").astype(np.float32) / 32768.0
            if pcm.size == 0:
                continue
            _write_output_stream(stream, pcm)
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        pa.terminate()

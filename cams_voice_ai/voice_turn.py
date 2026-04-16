"""One mic turn: Silero VAD → Deepgram ASR → encoder → OpenAI stream → Cartesia (WS or SSE)."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

from cams_voice_ai.asr_deepgram import DeepgramASR
from cams_voice_ai.asr_deepgram_stream import DeepgramStreamASR
from cams_voice_ai.config import Settings
from cams_voice_ai.intent_emotion import IntentEmotionEncoder, Signals
from cams_voice_ai.llm_stream import stream_assistant_text
from cams_voice_ai.transcribe import asr_final_from_speech, vad_analyze
from cams_voice_ai.tts_cartesia import CartesiaTTS
from cams_voice_ai.tts_cartesia_ws import CartesiaTTSWebSocket
from cams_voice_ai.vad_silero import SileroVAD

logger = logging.getLogger(__name__)

TTSSink = Union[CartesiaTTS, CartesiaTTSWebSocket]


def _trace_block(title: str, body: str, *, verbose: bool) -> None:
    if not verbose:
        return
    line = "-" * 40
    print(f"\n{line}\n{title}\n{line}\n{body}\n{line}\n", file=sys.stderr, flush=True)


@dataclass
class TurnResult:
    transcript: str
    signals: Signals
    assistant_text: str
    had_speech: bool
    session_end: bool


def speak_plain(tts: TTSSink, text: str, play_audio: bool) -> None:
    from cams_voice_ai.mic_audio import play_audio_bytes, play_pcm_s16le_chunks

    if not play_audio:
        return
    if isinstance(tts, CartesiaTTSWebSocket):
        play_pcm_s16le_chunks(
            (c for c in tts.iter_pcm_plain(text) if c),
            sample_rate=tts.sample_rate,
        )
        return
    try:
        play_pcm_s16le_chunks(
            (c for c in tts.iter_synthesize_sse(text) if c),
            sample_rate=24000,
        )
    except Exception as exc:
        logger.warning("TTS SSE failed (%s); batch WAV.", exc)
        play_audio_bytes(tts.synthesize(text).audio_bytes)


def run_voice_turn(
    settings: Settings,
    vad: SileroVAD,
    asr: DeepgramASR,
    asr_stream: DeepgramStreamASR,
    encoder: IntentEmotionEncoder,
    tts: TTSSink,
    messages: List[dict],
    wav_bytes: bytes,
    *,
    skip_vad: bool = False,
    verbose: bool = False,
    show_pipeline: bool = False,
    play_audio: bool = True,
    on_llm_delta: Optional[Callable[[str], None]] = None,
) -> TurnResult:
    """If ``show_pipeline`` or ``verbose``, print ASR, intent/emotion hints, and LLM text to stderr."""
    show_blocks = verbose or show_pipeline
    t0 = time.perf_counter()
    vad_res = vad_analyze(vad, wav_bytes, skip_vad)
    if verbose:
        print(
            f"[VAD] speech={vad_res.has_speech} ratio={vad_res.speech_ratio:.3f}",
            file=sys.stderr,
        )
    if not vad_res.has_speech:
        return TurnResult(
            transcript="",
            signals=encoder.classify(""),
            assistant_text="",
            had_speech=False,
            session_end=False,
        )

    speech_wav = vad_res.speech_wav_bytes
    t_asr = time.perf_counter()
    transcript = asr_final_from_speech(speech_wav, asr, asr_stream)
    if verbose:
        print(
            f"[ASR] {(time.perf_counter() - t_asr) * 1000:.0f} ms",
            file=sys.stderr,
            flush=True,
        )
    _trace_block(
        "ASR transcript",
        transcript if transcript.strip() else "(empty)",
        verbose=show_blocks,
    )

    ctx = ""
    for m in messages[-4:]:
        if m["role"] in ("user", "assistant"):
            ctx += m["content"][:400] + "\n"
    signals = encoder.classify(transcript, context=ctx)
    if verbose:
        print(
            f"[Signals] intent={signals.intent} emo={signals.emotion}",
            file=sys.stderr,
            flush=True,
        )
    hint_lines = (
        f"intent_hint={signals.intent}\n"
        f"intent_confidence={signals.intent_confidence:.3f}\n"
        f"emotion_hint={signals.emotion}\n"
        f"emotion_confidence={signals.emotion_confidence:.3f}"
    )
    _trace_block("Intent & emotion (hints to LLM)", hint_lines, verbose=show_blocks)

    if signals.intent == "session_end":
        closing = "Thank you for calling CAMS. Have a nice day."
        _trace_block(
            "LLM output (skipped: session_end intent)",
            closing,
            verbose=show_blocks,
        )
        speak_plain(tts, closing, play_audio)
        return TurnResult(
            transcript=transcript,
            signals=signals,
            assistant_text=closing,
            had_speech=True,
            session_end=True,
        )

    user_block = (
        f"[Classifier hints — soft signals only]\n"
        f"intent_hint={signals.intent} confidence={signals.intent_confidence:.2f}\n"
        f"emotion_hint={signals.emotion} confidence={signals.emotion_confidence:.2f}\n\n"
        f"[Caller]\n{transcript}"
    )
    turn_messages = [*messages, {"role": "user", "content": user_block}]

    buf: list[str] = []

    def _delta(t: str) -> None:
        buf.append(t)
        if on_llm_delta:
            on_llm_delta(t)

    assert settings.openai_api_key, "OPENAI_API_KEY must be set"

    if isinstance(tts, CartesiaTTSWebSocket):
        from cams_voice_ai.mic_audio import play_pcm_s16le_chunks

        def _llm_gen():
            yield from stream_assistant_text(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                messages=turn_messages,
                temperature=settings.openai_temperature,
                include_usage=settings.openai_stream_include_usage,
                reasoning_effort=settings.openai_reasoning_effort,
                on_delta=None,
            )

        def _cb(t: str) -> None:
            buf.append(t)
            if on_llm_delta:
                on_llm_delta(t)

        if play_audio:
            play_pcm_s16le_chunks(
                (c for c in tts.iter_pcm_llm_stream(_llm_gen(), on_llm_delta=_cb) if c),
                sample_rate=tts.sample_rate,
            )
        else:
            for _ in tts.iter_pcm_llm_stream(_llm_gen(), on_llm_delta=_cb):
                pass
    else:
        for _ in stream_assistant_text(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            messages=turn_messages,
            temperature=settings.openai_temperature,
            include_usage=settings.openai_stream_include_usage,
            reasoning_effort=settings.openai_reasoning_effort,
            on_delta=_delta,
        ):
            pass
        assistant_text = "".join(buf).strip() or (
            "I am sorry, I did not get a response. Could you please repeat that?"
        )
        if verbose:
            print(f"[LLM] {(time.perf_counter() - t0) * 1000:.0f} ms", file=sys.stderr, flush=True)
        _trace_block("LLM output", assistant_text, verbose=show_blocks)
        speak_plain(tts, assistant_text, play_audio)
        messages.append({"role": "user", "content": user_block})
        messages.append({"role": "assistant", "content": assistant_text})
        return TurnResult(
            transcript=transcript,
            signals=signals,
            assistant_text=assistant_text,
            had_speech=True,
            session_end=False,
        )

    assistant_text = "".join(buf).strip() or (
        "I am sorry, I did not get a response. Could you please repeat that?"
    )
    if verbose:
        print(
            f"[LLM+TTS WS] {(time.perf_counter() - t0) * 1000:.0f} ms",
            file=sys.stderr,
            flush=True,
        )
    _trace_block("LLM output", assistant_text, verbose=show_blocks)

    messages.append({"role": "user", "content": user_block})
    messages.append({"role": "assistant", "content": assistant_text})

    return TurnResult(
        transcript=transcript,
        signals=signals,
        assistant_text=assistant_text,
        had_speech=True,
        session_end=False,
    )

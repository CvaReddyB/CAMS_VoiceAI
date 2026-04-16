#!/usr/bin/env python3
"""
CAMS Voice AI POC — mic demo with a single system-prompt agent.

Pipeline: Silero VAD (padding + min silence) → Deepgram WebSocket
(endpointing / no_delay / language from env) → SentenceTransformer
(intent + emotion) → OpenAI streaming (optional usage + reasoning_effort for gpt-5) → Cartesia
WebSocket (sentence-chunked, context_id + continue + max_buffer_delay_ms) or HTTP SSE fallback.

KYC (one question at a time): registered mobile last four → PAN last four → date of birth.
Set MOCK_REGISTERED_CALLER=true to skip the mobile step only. Configure MOCK_KYC_* in .env for
expected KYC answers.

Requires real APIs (no mock ASR/LLM/TTS): DEEPGRAM_API_KEY, OPENAI_API_KEY, CARTESIA_API_KEY.

Usage:
  cd CAMS_VoiceAI
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt   # PyAudio needs PortAudio on macOS: brew install portaudio
  cp .env.example .env   # add API keys
  python run_mic_demo.py
  python run_mic_demo.py --list-mic-devices
  python run_mic_demo.py --skip-vad --max-turns 5
  python run_mic_demo.py --show-pipeline   # stderr: ASR, intent/emotion hints, LLM text
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cams_voice_ai.agent_prompt import SYSTEM_PROMPT, kyc_system_preamble
from cams_voice_ai.asr_deepgram import DeepgramASR
from cams_voice_ai.asr_deepgram_stream import DeepgramStreamASR
from cams_voice_ai.config import Settings, missing_voice_api_env
from cams_voice_ai.intent_emotion import (
    dob_matches_transcript,
    extract_last_four_mobile,
    extract_pan_last_four,
)
from cams_voice_ai.intent_emotion import IntentEmotionEncoder
from cams_voice_ai.mic_audio import list_input_devices, record_wav_bytes_until_silence
from cams_voice_ai.tts_cartesia import CartesiaTTS
from cams_voice_ai.tts_cartesia_ws import CartesiaTTSWebSocket
from cams_voice_ai.vad_silero import SileroVAD
from cams_voice_ai.transcribe import transcribe_wav
from cams_voice_ai.voice_turn import _trace_block, run_voice_turn, speak_plain

WELCOME = (
    "Welcome to CAMS customer care. Calls may be recorded for quality and compliance. "
)
REGISTERED_CLI = (
    "We recognize the number you are calling from as your registered mobile on file. "
    "We will verify a few quick details for security."
)
ASK_MOBILE = (
    "For your security, please say only the last four digits of your registered mobile number."
)
ASK_PAN = "Thank you. Now please say only the last four characters of your PAN."
ASK_DOB = (
    "Thank you. Please say your date of birth as day, month, and year, "
    "for example fifteen August nineteen forty five"
)
KYC_FAIL = (
    "We are sorry, we could not verify your identity with the details provided. "
    "Please try again later from your registered number or visit a CAMS service center. "
    "Thank you for calling CAMS."
)
KYC_RETRY_MOBILE = "Those digits did not match. Please say again only the last four digits of your registered mobile."
KYC_RETRY_PAN = "That PAN ending did not match. Please say again only the last four characters of your PAN."
KYC_RETRY_DOB = "The date of birth did not match. Please say your date of birth again clearly."
NO_SPEECH = "I did not catch that. Please speak after the tone."
MAX_KYC_TRIES = 3


def _transcribe_turn(
    vad: SileroVAD,
    asr: DeepgramASR,
    asr_stream: DeepgramStreamASR,
    wav: bytes,
    *,
    skip_vad: bool,
) -> str:
    return transcribe_wav(vad, asr, asr_stream, wav, skip_vad=skip_vad)


def _trim_messages(messages: List[dict], *, max_non_system: int = 14) -> None:
    if len(messages) <= 1 + max_non_system:
        return
    sys_msg = messages[0]
    rest = messages[1:]
    if len(rest) <= max_non_system:
        return
    messages[:] = [sys_msg, *rest[-max_non_system:]]


def _run_kyc(
    settings: Settings,
    vad: SileroVAD,
    asr: DeepgramASR,
    asr_stream: DeepgramStreamASR,
    tts,
    *,
    skip_vad: bool,
    play: bool,
    mic_device: Optional[int],
    show_pipeline: bool,
) -> bool:
    """Return True if KYC passed."""
    tries = 0
    speak_plain(tts, WELCOME, play)
    if settings.mock_registered_caller:
        speak_plain(tts, REGISTERED_CLI, play)
    else:
        speak_plain(tts, ASK_MOBILE, play)
        while tries < MAX_KYC_TRIES:
            wav = record_wav_bytes_until_silence(
                max_duration_sec=12.0,
                device=mic_device,
                speech_rms_threshold=settings.mic_speech_rms_threshold,
            )
            tr = _transcribe_turn(vad, asr, asr_stream, wav, skip_vad=skip_vad)
            if show_pipeline:
                _trace_block(
                    "KYC ASR (mobile last four)",
                    tr if tr.strip() else "(empty)",
                    verbose=True,
                )
            if not tr.strip():
                speak_plain(tts, NO_SPEECH, play)
                tries += 1
                continue
            got = extract_last_four_mobile(tr)
            if got == settings.mock_kyc_phone_last4:
                break
            speak_plain(tts, KYC_RETRY_MOBILE, play)
            tries += 1
        if tries >= MAX_KYC_TRIES:
            speak_plain(tts, KYC_FAIL, play)
            return False

    tries = 0
    speak_plain(tts, ASK_PAN, play)
    while tries < MAX_KYC_TRIES:
        wav = record_wav_bytes_until_silence(
            max_duration_sec=12.0,
            device=mic_device,
            speech_rms_threshold=settings.mic_speech_rms_threshold,
        )
        tr = _transcribe_turn(vad, asr, asr_stream, wav, skip_vad=skip_vad)
        if show_pipeline:
            _trace_block(
                "KYC ASR (PAN last four)",
                tr if tr.strip() else "(empty)",
                verbose=True,
            )
        if not tr.strip():
            speak_plain(tts, NO_SPEECH, play)
            tries += 1
            continue
        got = extract_pan_last_four(tr)
        exp = (settings.mock_kyc_pan_last4 or "").upper()
        if got and got.upper() == exp:
            break
        speak_plain(tts, KYC_RETRY_PAN, play)
        tries += 1
    if tries >= MAX_KYC_TRIES:
        speak_plain(tts, KYC_FAIL, play)
        return False

    tries = 0
    speak_plain(tts, ASK_DOB, play)
    exp_dob = (settings.mock_kyc_dob or "").strip()
    while tries < MAX_KYC_TRIES:
        wav = record_wav_bytes_until_silence(
            max_duration_sec=14.0,
            device=mic_device,
            speech_rms_threshold=settings.mic_speech_rms_threshold,
        )
        tr = _transcribe_turn(vad, asr, asr_stream, wav, skip_vad=skip_vad)
        if show_pipeline:
            _trace_block(
                "KYC ASR (date of birth)",
                tr if tr.strip() else "(empty)",
                verbose=True,
            )
        if not tr.strip():
            speak_plain(tts, NO_SPEECH, play)
            tries += 1
            continue
        if dob_matches_transcript(tr, exp_dob):
            break
        speak_plain(tts, KYC_RETRY_DOB, play)
        tries += 1
    if tries >= MAX_KYC_TRIES:
        speak_plain(tts, KYC_FAIL, play)
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="CAMS Voice AI mic POC")
    parser.add_argument("--skip-vad", action="store_true", help="Send full capture to ASR")
    parser.add_argument(
        "--mic-device",
        type=int,
        default=None,
        help="PortAudio input device index (KYC and conversation). Use --list-mic-devices.",
    )
    parser.add_argument("--list-mic-devices", action="store_true")
    parser.add_argument("--no-play", action="store_true", help="Disable speaker output")
    parser.add_argument("--max-turns", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--show-pipeline",
        action="store_true",
        help="Print ASR transcript, intent/emotion hints, and LLM reply blocks to stderr",
    )
    args = parser.parse_args()
    show_pipeline = args.verbose or args.show_pipeline

    settings = Settings.from_env()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    if args.list_mic_devices:
        list_input_devices()
        return

    missing = missing_voice_api_env(settings)
    if missing:
        print(
            "The following environment variables are required (no mock ASR/LLM/TTS): "
            + ", ".join(missing),
            file=sys.stderr,
        )
        sys.exit(1)

    vad = SileroVAD(
        threshold=settings.vad_threshold,
        min_speech_ms=settings.vad_min_speech_ms,
        min_silence_duration_ms=settings.vad_min_silence_duration_ms,
        speech_pad_ms=settings.vad_speech_pad_ms,
    )
    asr = DeepgramASR(settings.deepgram_api_key, model=settings.deepgram_model)
    asr_stream = DeepgramStreamASR(
        settings.deepgram_api_key,
        model=settings.deepgram_model,
        listen_params=settings.deepgram_listen_params(sample_rate=16000),
        stream_chunk_ms=settings.deepgram_stream_chunk_ms,
    )

    use_ws = settings.cartesia_transport == "websocket"
    tts_ws: Optional[CartesiaTTSWebSocket] = None
    if use_ws:
        tts_ws = CartesiaTTSWebSocket.from_settings(settings)
        tts_ws.connect()
        tts = tts_ws
    else:
        tts = CartesiaTTS(
            api_key=settings.cartesia_api_key,
            model_id=settings.cartesia_model_id,
            voice_id=settings.cartesia_voice_id,
            api_version=settings.cartesia_api_version,
        )

    encoder = IntentEmotionEncoder(model_name=settings.sentence_transformer_model)

    print(
        "● Loading local models (Silero VAD + intent/emotion encoder)…",
        file=sys.stderr,
        flush=True,
    )
    vad.warm()
    encoder.warm()
    print("● Local models ready.\n", file=sys.stderr, flush=True)

    play = not args.no_play
    try:
        if not _run_kyc(
            settings,
            vad,
            asr,
            asr_stream,
            tts,
            skip_vad=args.skip_vad,
            play=play,
            mic_device=args.mic_device,
            show_pipeline=show_pipeline,
        ):
            return

        bridge = (
            "Thank you, your details are verified. How may I help you today? "
            "You can ask about a redemption, an account statement, or a compliance-related question."
        )
        speak_plain(tts, bridge, play)

        messages: List[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "[System] " + kyc_system_preamble(),
            },
            {
                "role": "assistant",
                "content": bridge,
            },
        ]

        turn_count = 0
        while True:
            if args.max_turns and turn_count >= args.max_turns:
                speak_plain(
                    tts,
                    "Thank you for calling CAMS. This demo session has reached its turn limit. Goodbye.",
                    play,
                )
                break
            print("\n● Your turn — speak after the tone.\n", file=sys.stderr)
            wav = record_wav_bytes_until_silence(
                max_duration_sec=18.0,
                device=args.mic_device,
                speech_rms_threshold=settings.mic_speech_rms_threshold,
            )
            res = run_voice_turn(
                settings,
                vad,
                asr,
                asr_stream,
                encoder,
                tts,
                messages,
                wav,
                skip_vad=args.skip_vad,
                verbose=args.verbose,
                show_pipeline=show_pipeline,
                play_audio=play,
                on_llm_delta=(
                    (lambda t: print(t, end="", flush=True, file=sys.stderr))
                    if args.verbose
                    else None
                ),
            )
            if args.verbose:
                print(flush=True, file=sys.stderr)
            if res.session_end:
                break
            if not res.had_speech:
                speak_plain(tts, NO_SPEECH, play)
                continue
            turn_count += 1
            _trim_messages(messages)

            if "disconnect" in (res.transcript or "").lower() or "hang up" in (
                res.transcript or ""
            ).lower():
                speak_plain(tts, "Understood. Disconnecting now. Thank you for calling CAMS.", play)
                break
    finally:
        if tts_ws is not None:
            tts_ws.close()


if __name__ == "__main__":
    main()

# CAMS Voice AI (mic demo)

Voice POC: **microphone** → Silero VAD → **Deepgram** ASR → sentence-transformer **intent/emotion** hints → **OpenAI** (streaming) → **Cartesia** TTS (WebSocket or SSE). Includes a short **KYC** flow (mobile / PAN / DOB) before free-form turns.

## Requirements

- Python 3.9+ (venv recommended)
- API keys: **Deepgram**, **OpenAI**, **Cartesia** (no mocked ASR/LLM/TTS in this demo)
- macOS: [PortAudio](https://formulae.brew.sh/formula/portaudio) for PyAudio, e.g. `brew install portaudio`

## Setup

```bash
cd CAMS_VoiceAI
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`: set `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, and `CARTESIA_API_KEY`. Adjust `MOCK_KYC_*` to match what you will say during KYC.

## Run

```bash
python run_mic_demo.py
```

Useful flags:

| Flag | Purpose |
|------|--------|
| `--list-mic-devices` | Print PyAudio input indices |
| `--mic-device N` | Select input device |
| `--show-pipeline` | Log ASR, intent/emotion hints, and LLM text to stderr |
| `--verbose` | Pipeline logs plus extra timing / token stream |
| `--no-play` | No speaker output |
| `--skip-vad` | Send full capture to ASR (debug) |

On first start the app **preloads** Silero VAD and the intent/emotion encoder so the first conversation turn is not delayed by model load.

## Configuration

See `.env.example` for variables (models, Cartesia transport, VAD thresholds, mock KYC answers, `LOG_LEVEL`). Optional tuning (defaults exist in code if omitted): `MIC_SPEECH_RMS_THRESHOLD`, `MIC_PRE_RECORD_DELAY_SEC`.

## Security

- **Never commit `.env`** (it is listed in `.gitignore`).
- Rotate any API keys that were shared or committed by mistake.

## License

Internal / demo use unless you add a project license.

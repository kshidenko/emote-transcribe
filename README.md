# Emote-Transcribe

Prosody analysis microservice — extracts **how** someone speaks, not just **what** they say.

Analyzes pitch, tempo, pauses, intensity per segment and generates human-readable descriptions for LLM context.

## Quick Start

```bash
# Install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python server.py
# → http://0.0.0.0:8200

# Test
curl -X POST http://localhost:8200/analyze -F "file=@voice.wav"
```

## API

### `POST /analyze`

Upload audio file, get full prosody analysis.

**Input:** multipart/form-data with `file` field (WAV, OGG, OPUS, MP3, M4A, WEBM)

**Output:**
```json
{
  "global": {
    "duration_sec": 5.2,
    "avg_pitch_hz": 185,
    "speech_rate": "fast",
    "total_pause_ratio": 0.15
  },
  "segments": [
    {
      "start": 0.0,
      "end": 1.8,
      "pitch_trend": "rising",
      "tempo_label": "fast",
      "intensity_label": "loud",
      "pause_label": "long"
    }
  ],
  "voice_context": "[voice analysis]\nOverall: fast pace...\n[/voice analysis]"
}
```

### `POST /context`

Same as `/analyze` but returns only the `voice_context` text string.

### `GET /health`

Health check.

## Architecture

```
Voice Message (OGG/WAV)
    ↓
┌─ ffmpeg (convert to 16kHz WAV if needed)
│
├─ Parselmouth/Praat DSP:
│   • Pitch (F0 contour, trend)
│   • Intensity (dB, loud/soft)
│   • Pauses (detection, duration)
│   • Tempo (syllable rate estimation)
│
└─ Describer:
    • Per-segment descriptions
    • Global summary
    • voice_context block for LLM
```

## Integration with OpenClaw

The `voice_context` output is designed to be prepended to Whisper transcription:

```
[voice analysis]
Overall: fast pace, mid-range pitch, loud, few pauses (energetic delivery)
Segments:
- 0.0-1.8s: speaking quickly, voice rising, speaking loudly
- 2.6-5.2s: speaking slowly, voice dropping, speaking softly, long pause (800ms)
[/voice analysis]

Привет! Как дела? ... ну ладно
```

## Tech Stack

- **Python 3.11+**
- **FastAPI** + **Uvicorn** — HTTP server
- **Parselmouth** — Praat wrapper for DSP analysis
- **ffmpeg** — audio format conversion

## License

MIT

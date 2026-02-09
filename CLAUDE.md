# CLAUDE.md

## Project Overview

Emote-Transcribe is a prosody analysis microservice that runs alongside OpenClaw on Mac Mini (M1, 16GB). It analyzes incoming voice messages to extract HOW someone speaks — pitch, tempo, pauses, intensity — and generates human-readable descriptions for LLM context.

## Architecture

- **FastAPI** server on port 8200
- **Parselmouth** (Praat wrapper) for all DSP analysis — no neural networks
- **ffmpeg** for audio format conversion (OGG/OPUS → WAV)
- Designed to run parallel to Whisper STT (port 8178)

## Key Files

- `server.py` — FastAPI app with `/analyze` and `/context` endpoints
- `analyzer.py` — Core prosody analysis (pitch, pauses, tempo, intensity)
- `describer.py` — Converts numeric data to human-readable descriptions
- `requirements.txt` — Python dependencies (parselmouth, fastapi, uvicorn)

## Data Flow

```
Voice Message → ffmpeg (→ WAV) → Parselmouth DSP → Segment Analysis → Description Generator → JSON + voice_context text
```

## API

- `POST /analyze` — Full analysis (JSON with segments + voice_context)
- `POST /context` — Only the voice_context text block
- `GET /health` — Health check

## Commands

```bash
# Run server
python server.py

# Or with uvicorn directly
uvicorn server:app --host 0.0.0.0 --port 8200 --reload

# Test
curl -X POST http://localhost:8200/analyze -F "file=@test.wav"
```

## Design Decisions

- **No ML models** — Pure DSP via Parselmouth for speed and simplicity
- **Separate service** — Python deps isolated from Node.js OpenClaw
- **Human-readable output** — `voice_context` block designed for LLM consumption
- **Segment-based** — Audio split by pauses, each segment analyzed independently

## Future: emotion2vec Integration

emotion2vec+ base (~90M params) can be added later for neural emotion classification.
Would add `emotion` and `emotion_confidence` fields to each segment.
Dependency: `funasr` + `torch` (~2GB total).

## Deployment Target

- Mac Mini M1 16GB (`nex.local`)
- SSH: `ssh nex@nex.local`
- Runs as LaunchAgent (planned)

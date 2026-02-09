# CLAUDE.md

## Project Overview

Emote-Transcribe is a two-tier prosody + emotion analysis microservice:
1. **Sync layer**: OpenAI-compatible Whisper proxy (`/v1/audio/transcriptions`) — instant response
2. **Async layer**: Background prosody (Parselmouth) + emotion (audeering wav2vec2) enrichment — writes annotated transcript logs + full analysis JSON dumps

Runs alongside OpenClaw on Mac Mini (M1, 16GB). Designed to be lightweight with auto-unload of emotion model after idle.

## Architecture

```
Audio in → Proxy → whisper-cpp (:8178) → instant text to client
        ↘ Async queue → Parselmouth (prosody) + audeering (emotion A/V/D)
                      → Annotated transcript (.jsonl)
                      → Full analysis dump (.json per message)
                      → Delete tmp WAV
```

### STT Backend
- `local` (default): Forward to whisper-cpp on port 8178
- `openai`: Forward to OpenAI API
- `auto`: Local with OpenAI fallback

### Three Modes
- `full`: Whisper + Prosody + Emotion (async)
- `whisper_only`: Pure proxy, no enrichment, emotion model unloaded
- `prosody_only`: Whisper + Parselmouth only (no emotion model)

### Emotion Model
- `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` — dimensional
- Outputs: arousal (0-1), valence (0-1), dominance (0-1)
- Language-independent (acoustic features only)
- ~800MB RAM, auto-unloads after idle timeout

## Key Files

- `server.py` — FastAPI app: OpenAI-compatible proxy + async job dispatch
- `analyzer.py` — Parselmouth DSP: pitch, pauses, tempo, intensity
- `describer.py` — Converts numeric prosody data to human-readable text
- `emotion.py` — Audeering dimensional emotion model (A/V/D)
- `worker.py` — Async enrichment worker (prosody + emotion → logs)
- `cleanup.py` — Tmp file deletion + log rotation
- `config.py` — Service configuration (mode, paths, env vars)
- `label_config.json` — Tunable thresholds for all classifiers + verbal labels + resource limits
- `label_config_loader.py` — Hot-reloadable JSON config reader
- `requirements.txt` — Python dependencies

## API Endpoints

### OpenAI-Compatible (main)
- `POST /v1/audio/transcriptions` — Drop-in Whisper replacement
  - Headers: `X-STT-Backend: local|openai` (optional override)
  - Returns: transcription text immediately (sync)
  - Response header: `X-Enrichment-Job-Id` for async result

### Service
- `GET /health` — Status, mode, model info
- `GET /v1/analysis/{job_id}` — Fetch enrichment result
- `GET /v1/analysis/latest` — Last N enriched transcripts
- `POST /v1/mode` — Switch mode: `full` | `whisper_only` | `prosody_only`
- `GET /v1/logs` — Browse annotated transcript logs

## Output Format

### Annotated Transcript (voice_context)
```
[0.7-1.9s | loud, rising | A:0.51 V:0.60 D:0.54 → neutral | exclaiming] Ну, нифига себе
[7.0-7.8s | slow, falling, pause 1670ms | A:0.15 V:0.34 D:0.23 → sad | thoughtful] Но очень странное. ...(1670ms)
```
Four layers: `[time | raw prosody | emotion A/V/D → emotion_label | auto_label] text ...(pause)`

### Analysis Dump (logs/analysis/*.json)
One JSON file per voice message with full structured data per segment:
```json
{
  "segments": [{
    "start": 0.7, "end": 1.9,
    "text": "...",
    "prosody": {"tempo": "normal", "intensity": "loud", "pitch_trend": "rising", ...},
    "emotion": {"arousal": 0.51, "valence": 0.60, "dominance": 0.54},
    "emotion_label": "neutral",
    "auto_label": "exclaiming"
  }]
}
```

## Configuration

### Environment Variables (config.py)
```
WHISPER_URL=http://127.0.0.1:8178
OPENAI_API_KEY=                              # Set if using OpenAI backend
STT_BACKEND=local
MODE=full
PORT=8200
LOG_DIR=logs/transcripts
ANALYSIS_LOG_DIR=logs/analysis
TMP_DIR=tmp
LOG_RETENTION_DAYS=30
```

### Tunable Thresholds (label_config.json)
All classifier boundaries, verbal label mappings, emotion label mappings, and resource limits. Hot-reloaded between requests — no restart needed. An agent can edit this JSON to tune detection quality.

## Commands

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run (requires whisper-cpp on :8178)
python server.py

# Test
curl -X POST http://localhost:8200/v1/audio/transcriptions \
  -F "file=@voice.ogg" -F "language=ru"

# Check enrichment
curl http://localhost:8200/v1/analysis/latest?n=1
```

## Resource Management

- Emotion model auto-unloads after idle (default 5 min)
- Worker limited to 2 CPU threads (`torch.set_num_threads`)
- Worker runs at low priority (`os.nice(10)`)
- All configurable in `label_config.json` → `resources`

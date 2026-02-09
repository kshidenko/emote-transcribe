# Emote-Transcribe Agent Skill

## What This Service Does

Emote-Transcribe is a prosody + emotion analysis microservice that:
1. Proxies audio to Whisper for transcription (instant, sync)
2. Enriches each speech segment with prosody metrics and emotion dimensions (async)
3. Writes structured analysis logs for later processing

## How to Use

### Send Audio for Analysis

```bash
curl -X POST http://localhost:8200/v1/audio/transcriptions \
  -F "file=@audio.ogg" -F "language=ru"
```

Response: transcription text (instant). Header `X-Enrichment-Job-Id` contains the async job ID.

### Get Enrichment Result

```bash
curl http://localhost:8200/v1/analysis/{job_id}
# or latest:
curl http://localhost:8200/v1/analysis/latest?n=1
```

### Read Analysis Dump

Full structured JSON per voice message is saved to `logs/analysis/`:
```
logs/analysis/2026-02-09T18-26-15_48508804.json
```

Each segment contains:
- `text` — what was said
- `prosody.tempo` — very_slow / slow / normal / fast / very_fast
- `prosody.intensity` — whisper / soft / normal / loud
- `prosody.pitch_trend` — rising / falling / flat
- `prosody.pause_after_ms` — silence duration after this segment
- `emotion.arousal` — energy level (0=calm, 1=excited)
- `emotion.valence` — sentiment (0=negative, 1=positive)
- `emotion.dominance` — assertiveness (0=submissive, 1=dominant)
- `emotion_label` — auto-derived from A/V/D (e.g. "tense", "serene", "sad")
- `auto_label` — derived from prosody (e.g. "exclaiming", "thoughtful", "curious")

## Tuning Thresholds

Edit `label_config.json` to adjust detection — changes take effect on the next request (hot-reload, no restart needed).

Key sections:
- `intensity_db` — dB boundaries for whisper/soft/normal/loud
- `tempo_syl_sec` — syllables/sec boundaries for tempo labels
- `pause_ms` — ms boundaries for short/medium/long pauses
- `pause_detection` — silence detection sensitivity
- `verbal_labels` — prosody combo → human-readable label mapping
- `emotion_labels` — A/V/D dimension → emotion word mapping
- `emotion_thresholds` — what counts as "low" and "high" for A/V/D
- `resources` — idle timeout, CPU threads, priority

## Agent Pipeline: Prosody Calibration via TTS

This is a multi-step agent workflow for calibrating and understanding prosody detection. The goal is NOT perfect replication — it's understanding the general rhythm, pauses, pitch changes, loudness, and tempo.

### Step 1: Analyze a Speech Segment

Pick a segment from the analysis log, e.g.:
```json
{
  "text": "Но очень странное.",
  "prosody": {"tempo": "slow", "intensity": "normal", "pitch_trend": "falling"},
  "emotion": {"arousal": 0.15, "valence": 0.34, "dominance": 0.23},
  "auto_label": "thoughtful"
}
```

### Step 2: Understand the Context

Look at the full transcript and surrounding segments. What emotion should the user be feeling at this point? Consider:
- What was said before and after?
- What is the topic?
- Does the prosody match the expected emotion?

### Step 3: Hypothesize Expected Prosody

Based on context, predict what the speech should sound like:
- "thoughtful" matches — the user is reflecting on something strange
- Slow pace + falling pitch + long pause = contemplation
- Low arousal + mid valence = calm but slightly negative

### Step 4: Reproduce with TTS Tags

Use TTS tags (ElevenLabs v3, OpenAI, etc.) to try reproducing the speaking style:
```
<prosody rate="slow" pitch="-10%">Но очень странное.</prosody>
<break time="1670ms"/>
```

Or with ElevenLabs voice tags:
```
[thoughtful, slow pace, falling intonation] Но очень странное.
```

### Step 5: Analyze the TTS Output

Send the generated audio back through Emote-Transcribe:
```bash
curl -X POST http://localhost:8200/v1/audio/transcriptions \
  -F "file=@tts_output.wav" -F "language=ru"
```

### Step 6: Compare

Compare the original and TTS analysis:
- Do the prosody labels roughly match? (tempo, intensity, pitch_trend)
- Are pauses in the right places?
- Is the overall rhythm similar?
- Are the A/V/D emotion dimensions in the same range?

Do NOT aim for exact match. Focus on:
- General tempo (slow vs fast)
- Pause placement and duration
- Pitch direction (rising vs falling vs flat)
- Loudness pattern (loud vs soft vs normal)

### Step 7: Iterate

If the TTS output doesn't match:
1. Adjust TTS tags (more/less pitch variation, different rate)
2. Re-send through the service
3. Compare again

This builds understanding of which TTS tags produce which prosody patterns. Over time, the agent learns the mapping between TTS controls and prosody metrics.

## Mode Switching

Two ways — via API or via config file:

### Via API
```bash
curl -X POST http://localhost:8200/v1/mode -d '{"mode": "full"}'
curl -X POST http://localhost:8200/v1/mode -d '{"mode": "prosody_only"}'
curl -X POST http://localhost:8200/v1/mode -d '{"mode": "whisper_only"}'
```

### Via label_config.json (preferred for agents)
Edit `label_config.json` — changes apply on the next request:
```json
{
  "mode": "full",
  "stt_backend": "local",
  ...
}
```

Available modes:
- `full` — Whisper + Prosody + Emotion (default)
- `prosody_only` — Whisper + Prosody, no emotion model (~800MB RAM saved)
- `whisper_only` — Pure Whisper proxy, no enrichment

Available backends:
- `local` — whisper-cpp on port 8178
- `openai` — OpenAI Whisper API
- `auto` — local with OpenAI fallback

## Health Check

```bash
curl http://localhost:8200/health
```

Returns: mode, emotion model status, queue size.

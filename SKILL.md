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

## Agent Pipeline 1: Emotional Context Analysis

This is the primary agent workflow — analyzing how user emotions evolve during a conversation by combining dialogue history with voice prosody data.

### Overview

```
Dialogue history (OpenClaw/Telegram) → understand context → predict expected emotion
                                                              ↓
Analysis dumps (emote-transcribe)    → actual voice data   → compare
                                                              ↓
                                                       emotional dynamics
```

### Step 1: Get Dialogue Context

From the conversation system (OpenClaw, Telegram, etc.), retrieve the message chain around the target voice message:
- What did the assistant say before?
- What was the user replying to?
- What is the topic / emotional context?

### Step 2: Predict Expected Emotion

Based on dialogue context, hypothesize what the user should be feeling:
- Did the assistant give good or bad news?
- Is the user asking a question, making a statement, reacting?
- Previous messages: was the user getting frustrated? excited? bored?

### Step 3: Fetch Voice Analysis

Get the analysis dump for that voice message:
```bash
# By approximate time (no need for exact match)
curl "http://localhost:8200/v1/dumps/nearest?t=2026-02-09T18:26"
```

### Step 4: Compare Predicted vs Actual

Look at the segments:
- `auto_label` (prosody) — how they actually spoke (exclaiming, thoughtful, etc.)
- `emotion` (A/V/D) — acoustic emotion dimensions
- `emotion_label` — model's emotion classification

Compare with your prediction:
- Did you expect "excited" but voice says "subdued"? Maybe the user is disappointed.
- Did you expect "neutral" but voice says "tense"? Maybe there's hidden frustration.
- Does prosody match the words? "I'm fine" spoken with low arousal + low valence = not actually fine.

### Step 5: Analyze Chains (Question → Answer → Reaction)

Track emotional dynamics across message pairs:

```
[Assistant] "Your request was denied."     → predict: user will be frustrated
[User voice] auto_label=assertive, A:0.7   → confirmed: high arousal, low valence
[Assistant] "Let me try another approach."  → predict: user calming down
[User voice] auto_label=calm, A:0.3        → confirmed: arousal dropped
```

Key patterns to detect:
- **Escalation**: arousal increasing over consecutive messages → user getting frustrated
- **De-escalation**: arousal decreasing → assistant calming the user
- **Mismatch**: words say "okay" but voice says "tense" → user is not actually okay
- **Engagement**: high arousal + high valence → user is interested/excited

### Step 6: Adjust Assistant Behavior

Based on emotional dynamics:
- If user is escalating → assistant should slow down, acknowledge, be empathetic
- If user is de-escalating → maintain current tone
- If mismatch detected → probe deeper ("Are you sure? You sound uncertain")
- If engagement high → build on the momentum

## Agent Pipeline 2: Prosody Calibration via TTS

Secondary workflow — calibrating TTS output to match natural speech patterns. Goal is NOT exact replication, but understanding the general rhythm, pauses, pitch changes, loudness, and tempo.

### Step 1: Pick a Segment from Analysis

```json
{
  "text": "Но очень странное.",
  "prosody": {"tempo": "slow", "intensity": "normal", "pitch_trend": "falling"},
  "emotion": {"arousal": 0.15, "valence": 0.34, "dominance": 0.23},
  "auto_label": "thoughtful"
}
```

### Step 2: Reproduce with TTS Tags

```
<prosody rate="slow" pitch="-10%">Но очень странное.</prosody>
<break time="1670ms"/>
```

### Step 3: Send TTS Output Back Through Service

```bash
curl -X POST http://localhost:8200/v1/audio/transcriptions \
  -F "file=@tts_output.wav" -F "language=ru"
```

### Step 4: Compare — Focus On

- General tempo (slow vs fast)
- Pause placement and duration
- Pitch direction (rising vs falling vs flat)
- Loudness pattern (loud vs soft vs normal)

Do NOT aim for exact numbers. Match the general pattern.

### Step 5: Iterate

Adjust TTS tags → re-send → compare → repeat. This builds a mapping between TTS controls and prosody metrics.

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

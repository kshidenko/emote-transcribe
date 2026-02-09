"""
Async enrichment worker for Emote-Transcribe.

Receives transcription jobs via an ``asyncio.Queue``, runs prosody analysis
(Parselmouth) and optionally emotion recognition (SER model) in a thread pool,
then writes annotated results to JSONL log files and deletes temporary audio.

The worker maintains an in-memory result store (bounded dict) so that recent
enrichment results can be queried via the API.

Usage:
    >>> import asyncio
    >>> from worker import enrichment_queue, enqueue_job, start_worker
    >>> asyncio.create_task(start_worker())
    >>> job_id = await enqueue_job("/tmp/audio.wav", "Привет!", "ru")
    >>> # Result will be available via get_result(job_id) after processing
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import settings, Mode
from analyzer import analyze_prosody
from describer import describe_segment, generate_voice_context
from cleanup import delete_tmp_file
from label_config_loader import cfg

logger = logging.getLogger("emote-transcribe.worker")

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_RESULTS = 500  # Max number of results kept in memory
WORKER_THREADS = 1  # Single thread to limit RAM usage

# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class EnrichmentJob:
    """A job enqueued for async prosody/emotion enrichment.

    Attributes:
        job_id: Unique identifier for this enrichment job.
        audio_path: Path to temporary audio WAV file.
        text: Transcription text from Whisper.
        language: Language code (e.g. 'ru', 'en').
        timestamp: ISO 8601 timestamp when the transcription was received.
        audio_duration_sec: Duration of audio in seconds (0 if unknown).
        whisper_segments: Whisper verbose_json segments with timestamps.
    """

    job_id: str
    audio_path: str
    text: str
    language: str
    timestamp: str
    audio_duration_sec: float = 0.0
    whisper_segments: list[dict] = field(default_factory=list)


@dataclass
class EnrichmentResult:
    """Result of async enrichment processing.

    Attributes:
        job_id: Matching job identifier.
        status: 'pending', 'processing', 'done', 'error'.
        timestamp: Original transcription timestamp.
        text: Original transcription text.
        prosody: Full prosody analysis dict (global + segments).
        voice_context: Human-readable voice analysis text block.
        mode: Mode that was active during processing.
        enrichment_time_ms: Total processing wall-clock time in ms.
        error: Error message if status is 'error'.
    """

    job_id: str
    status: str = "pending"
    timestamp: str = ""
    text: str = ""
    audio_duration_sec: float = 0.0
    prosody: Optional[dict[str, Any]] = None
    voice_context: str = ""
    mode: str = ""
    enrichment_time_ms: float = 0.0
    error: str = ""


# ── Module state ──────────────────────────────────────────────────────────────

enrichment_queue: asyncio.Queue[EnrichmentJob] = asyncio.Queue(maxsize=100)
_results: OrderedDict[str, EnrichmentResult] = OrderedDict()
_executor = ThreadPoolExecutor(max_workers=WORKER_THREADS)


# ── Result store ──────────────────────────────────────────────────────────────

def store_result(result: EnrichmentResult) -> None:
    """Store an enrichment result, evicting oldest if at capacity.

    Args:
        result: Enrichment result to store.
    """
    _results[result.job_id] = result
    while len(_results) > MAX_RESULTS:
        _results.popitem(last=False)


def get_result(job_id: str) -> Optional[EnrichmentResult]:
    """Retrieve an enrichment result by job ID.

    Args:
        job_id: The job identifier.

    Returns:
        EnrichmentResult if found, None otherwise.
    """
    return _results.get(job_id)


def get_latest_results(n: int = 10) -> list[EnrichmentResult]:
    """Retrieve the N most recent enrichment results.

    Args:
        n: Number of results to return (default 10).

    Returns:
        List of EnrichmentResult, newest first.
    """
    items = list(_results.values())
    return list(reversed(items[-n:]))


def get_queue_size() -> int:
    """Return current number of pending jobs in the queue."""
    return enrichment_queue.qsize()


# ── Job enqueue ───────────────────────────────────────────────────────────────

async def enqueue_job(
    audio_path: str,
    text: str,
    language: str = "ru",
    audio_duration_sec: float = 0.0,
    whisper_segments: list[dict] | None = None,
    external_timestamp: str | None = None,
) -> str:
    """Create and enqueue an enrichment job.

    Args:
        audio_path: Path to temporary audio WAV file.
        text: Transcription text.
        language: Language code.
        audio_duration_sec: Duration of audio in seconds.
        whisper_segments: Whisper verbose_json segments with timestamps.
        external_timestamp: Optional timestamp from client (e.g. Telegram message time).
            If provided, used as the primary timestamp for searching/matching.

    Returns:
        Job ID string.

    Raises:
        asyncio.QueueFull: If the enrichment queue is at capacity.
    """
    job_id = str(uuid.uuid4())
    now = external_timestamp or datetime.now(timezone.utc).isoformat()

    job = EnrichmentJob(
        job_id=job_id,
        audio_path=audio_path,
        text=text,
        language=language,
        timestamp=now,
        audio_duration_sec=audio_duration_sec,
        whisper_segments=whisper_segments or [],
    )

    # Pre-register as pending
    store_result(EnrichmentResult(
        job_id=job_id,
        status="pending",
        timestamp=now,
        text=text,
        audio_duration_sec=audio_duration_sec,
    ))

    await enrichment_queue.put(job)
    logger.debug("Enqueued job %s (queue size: %d)", job_id, enrichment_queue.qsize())
    return job_id


# ── Processing ────────────────────────────────────────────────────────────────

def _run_prosody(audio_path: str) -> dict[str, Any]:
    """Run prosody analysis in thread pool (blocking).

    Args:
        audio_path: Path to WAV file.

    Returns:
        Prosody analysis dict with 'global' and 'segments'.
    """
    return analyze_prosody(audio_path)


def _run_per_segment_emotion(
    audio_path: str,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run dimensional emotion recognition per prosody segment.

    Extracts each segment and predicts arousal/valence/dominance.

    Args:
        audio_path: Path to WAV file.
        segments: Prosody segments with 'start' and 'end' keys.

    Returns:
        Same segments list with 'emotion' dict added per segment.
    """
    from emotion import emotion_manager

    if not emotion_manager.is_loaded:
        try:
            logger.info("Emotion model not loaded — reloading for enrichment")
            emotion_manager.load()
        except Exception as exc:
            logger.error("Failed to reload emotion model: %s", exc)
            for seg in segments:
                seg["emotion"] = None
            return segments

    for seg in segments:
        try:
            result = emotion_manager.predict_segment(
                audio_path, seg["start"], seg["end"],
            )
            seg["emotion"] = result.to_dict()
        except Exception as exc:
            logger.warning("Emotion failed for segment %.1f-%.1f: %s",
                           seg["start"], seg["end"], exc)
            seg["emotion"] = None

    return segments


def _align_text_to_segments(
    prosody_segments: list[dict[str, Any]],
    whisper_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Align Whisper text to prosody segments — each whisper segment
    assigned to exactly one prosody segment (best overlap).

    Args:
        prosody_segments: Prosody segments with 'start'/'end'.
        whisper_segments: Whisper verbose_json segments with 'start'/'end'/'text'.

    Returns:
        Prosody segments with 'text' field added.
    """
    # Init empty text
    for pseg in prosody_segments:
        pseg["text"] = ""

    if not prosody_segments:
        return prosody_segments

    # Assign each whisper segment to the prosody segment with max overlap
    for wseg in whisper_segments:
        ws = wseg.get("start", 0)
        we = wseg.get("end", 0)
        wtext = wseg.get("text", "").strip()
        if not wtext:
            continue

        best_idx = -1
        best_overlap = 0
        for i, pseg in enumerate(prosody_segments):
            overlap = min(pseg["end"], we) - max(pseg["start"], ws)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        if best_idx >= 0:
            existing = prosody_segments[best_idx]["text"]
            if existing:
                prosody_segments[best_idx]["text"] = existing + " " + wtext
            else:
                prosody_segments[best_idx]["text"] = wtext

    return prosody_segments


def _enrich_segments(
    prosody: dict[str, Any],
    audio_path: str,
    whisper_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Full enrichment: per-segment emotion + text alignment + descriptions.

    Args:
        prosody: Prosody analysis dict from analyzer.
        audio_path: Path to WAV file for emotion inference.
        whisper_segments: Whisper segments with timestamps and text.

    Returns:
        Enriched prosody dict.
    """
    segments = prosody.get("segments", [])

    # Align text
    _align_text_to_segments(segments, whisper_segments)

    # Per-segment dimensional emotion
    if settings.needs_emotion:
        _run_per_segment_emotion(audio_path, segments)
    else:
        for seg in segments:
            seg["emotion"] = None

    # Add descriptions
    for seg in segments:
        seg["description"] = describe_segment(seg)

    return prosody


def _generate_annotated_transcript(prosody: dict[str, Any]) -> str:
    """Generate inline annotated transcript: ``[time | tags] text``.

    Format per line::

        [0.7-1.9s | fast, loud, rising, happy 85%] Ну, нифига себе!

    Args:
        prosody: Enriched prosody dict with segments containing text,
                 prosody labels, and emotion.

    Returns:
        Multi-line annotated transcript string.
    """
    segments = prosody.get("segments", [])

    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        time_str = f"{seg['start']:.1f}-{seg['end']:.1f}s"

        # Raw prosody tags
        raw_parts = []
        tempo = seg.get("tempo_label", "normal")
        if tempo != "normal":
            raw_parts.append(tempo)
        intensity = seg.get("intensity_label", "normal")
        if intensity != "normal":
            raw_parts.append(intensity)
        trend = seg.get("pitch_trend", "flat")
        if trend != "flat":
            raw_parts.append(trend)
        pause_ms = seg.get("pause_after_ms", 0)
        if pause_ms >= 300:
            raw_parts.append(f"pause {int(pause_ms)}ms")
        raw_str = ", ".join(raw_parts) if raw_parts else "neutral"

        # Emotion dimensions + emotion_label
        emo = seg.get("emotion")
        if emo and isinstance(emo, dict):
            emo_tag = f"A:{emo.get('arousal',0):.2f} V:{emo.get('valence',0):.2f} D:{emo.get('dominance',0):.2f}"
            el = _emotion_label(seg)
            emo_str = f"{emo_tag} → {el}"
        else:
            emo_str = ""

        # Prosody verbal label (auto_label)
        label = _verbal_label(seg)

        # Pause after text
        pause_str = f" ...({int(pause_ms)}ms)" if pause_ms >= 300 else ""

        if emo_str:
            lines.append(f"[{time_str} | {raw_str} | {emo_str} | {label}] {text}{pause_str}")
        else:
            lines.append(f"[{time_str} | {raw_str} | {label}] {text}{pause_str}")

    return "\n".join(lines)


def _verbal_label(seg: dict[str, Any]) -> str:
    """Map prosody features to a single expressive verbal label.

    Combines tempo, intensity, pitch trend, and pause into a short
    human-readable description of *how* the person is speaking.

    Args:
        seg: Prosody segment dict.

    Returns:
        Verbal label string (e.g. 'восклицая', 'шёпотом, задумчиво').
    """
    tempo = seg.get("tempo_label", "normal")
    intensity = seg.get("intensity_label", "normal")
    trend = seg.get("pitch_trend", "flat")
    pause_ms = seg.get("pause_after_ms", 0)
    emotion = seg.get("emotion")
    confidence = seg.get("emotion_confidence") or 0

    is_loud = intensity == "loud"
    is_soft = intensity in ("soft", "whisper")
    is_whisper = intensity == "whisper"
    is_fast = tempo in ("fast", "very_fast")
    is_slow = tempo in ("slow", "very_slow")
    is_rising = trend == "rising"
    is_falling = trend == "falling"
    has_long_pause = pause_ms >= 800
    has_very_long_pause = pause_ms >= 1800

    vl = cfg["verbal_labels"]
    parts = []

    # Match prosody combo to verbal label from config
    # Try most specific combo first, then fall back
    if is_whisper:
        parts.append(vl.get("whisper", "whispering"))
    elif is_soft and is_falling:
        parts.append(vl.get("soft+falling", "softly, subdued"))
    elif is_soft:
        parts.append(vl.get("soft", "softly"))
    elif is_loud and is_rising:
        parts.append(vl.get("loud+rising", "exclaiming"))
    elif is_loud and is_fast:
        parts.append(vl.get("loud+fast", "energetic"))
    elif is_loud and is_falling:
        parts.append(vl.get("loud+falling", "assertive"))
    elif is_loud:
        parts.append(vl.get("loud", "loud"))
    elif is_fast and is_rising:
        parts.append(vl.get("fast+rising", "lively"))
    elif is_fast:
        parts.append(vl.get("fast", "brisk"))
    elif is_slow and is_falling and has_long_pause:
        parts.append(vl.get("slow+falling+long_pause", "thoughtful"))
    elif is_slow and is_rising:
        parts.append(vl.get("slow+rising", "curious"))
    elif is_slow and is_falling:
        parts.append(vl.get("slow+falling", "measured"))
    elif is_slow:
        parts.append(vl.get("slow", "unhurried"))
    elif is_rising:
        parts.append(vl.get("rising", "questioning"))
    elif is_falling and has_long_pause:
        parts.append(vl.get("falling+long_pause", "trailing off"))
    elif is_falling:
        parts.append(vl.get("falling", "calm"))
    else:
        parts.append(vl.get("default", "steady"))

    # Hesitation from very long pause
    if has_very_long_pause and parts[0] not in ("thoughtful", "trailing off"):
        parts.append("long pause")

    return ", ".join(parts)


def _emotion_label(seg: dict[str, Any]) -> str:
    """Derive emotion label from arousal/valence dimensions using config thresholds.

    Args:
        seg: Segment dict with 'emotion' sub-dict containing A/V/D.

    Returns:
        Emotion label string (e.g. 'excited', 'tense', 'neutral').
    """
    emo = seg.get("emotion")
    if not emo or not isinstance(emo, dict):
        return "neutral"

    ar = emo.get("arousal", 0.5)
    va = emo.get("valence", 0.5)

    thresholds = cfg["emotion_thresholds"]
    lo = thresholds.get("low", 0.35)
    hi = thresholds.get("high", 0.65)

    labels = cfg["emotion_labels"]

    # Build key from dimensions
    ar_tag = "high_arousal" if ar > hi else ("low_arousal" if ar < lo else None)
    va_tag = "high_valence" if va > hi else ("low_valence" if va < lo else None)

    # Try combined key first, then single keys
    if ar_tag and va_tag:
        key = f"{ar_tag}+{va_tag}"
        if key in labels:
            return labels[key]
    if va_tag and va_tag in labels:
        return labels[va_tag]
    if ar_tag and ar_tag in labels:
        return labels[ar_tag]

    return labels.get("default", "neutral")


def _write_log_entry(result: EnrichmentResult) -> None:
    """Append an enrichment result to the daily JSONL log file.

    Args:
        result: Completed enrichment result.

    Notes:
        Log file path: ``{LOG_DIR}/YYYY-MM-DD.jsonl``
    """
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = log_dir / f"{date_str}.jsonl"

    entry = {
        "id": result.job_id,
        "timestamp": result.timestamp,
        "audio_duration_sec": result.audio_duration_sec,
        "text": result.text,
        "mode": result.mode,
        "prosody": result.prosody,
        "voice_context": result.voice_context,
        "enrichment_time_ms": result.enrichment_time_ms,
    }

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.debug("Wrote log entry to %s", log_path)
    except OSError as exc:
        logger.error("Failed to write log entry: %s", exc)


def _write_analysis_dump(result: EnrichmentResult) -> None:
    """Write full-dump analysis JSON to a separate file per voice message.

    Output path: ``{ANALYSIS_LOG_DIR}/{timestamp}_{id}.json``

    The JSON contains transcript text, prosody data, emotion dimensions,
    and auto-labels per segment — designed for later LLM batch analysis.

    Args:
        result: Completed enrichment result.
    """
    analysis_dir = Path(settings.analysis_log_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Build clean timestamp for filename (replace colons)
    ts = result.timestamp.replace(":", "-").replace("+", "_")[:19]
    short_id = result.job_id[:8]
    filename = f"{ts}_{short_id}.json"

    # Build full-dump JSON with clean segment structure
    segments_out = []
    for seg in (result.prosody or {}).get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments_out.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": text,
            "prosody": {
                "tempo": seg.get("tempo_label", "normal"),
                "intensity": seg.get("intensity_label", "normal"),
                "pitch_trend": seg.get("pitch_trend", "flat"),
                "pitch_mean_hz": seg.get("pitch_mean", 0),
                "pause_after_ms": seg.get("pause_after_ms", 0),
            },
            "emotion": seg.get("emotion"),
            "emotion_label": _emotion_label(seg),
            "auto_label": _verbal_label(seg),
        })

    dump = {
        "id": result.job_id,
        "timestamp": result.timestamp,
        "audio_duration_sec": result.audio_duration_sec,
        "text": result.text,
        "segments": segments_out,
    }

    dump_path = analysis_dir / filename
    try:
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)
        logger.info("Wrote analysis dump: %s", filename)
    except OSError as exc:
        logger.error("Failed to write analysis dump: %s", exc)


# ── Worker loop ───────────────────────────────────────────────────────────────

async def _process_job(job: EnrichmentJob) -> None:
    """Process a single enrichment job.

    Runs prosody + emotion analysis, stores result, writes log, deletes tmp.

    Args:
        job: The enrichment job to process.
    """
    t0 = time.perf_counter()

    # Mark as processing
    store_result(EnrichmentResult(
        job_id=job.job_id,
        status="processing",
        timestamp=job.timestamp,
        text=job.text,
        audio_duration_sec=job.audio_duration_sec,
        mode=current_mode,
    ))

    loop = asyncio.get_event_loop()

    try:
        # Hot-reload config (picks up JSON changes without restart)
        cfg.reload()

        # Apply mode/backend from config if changed
        cfg_mode = cfg.get("mode")
        if cfg_mode and cfg_mode != settings.mode.value:
            try:
                new_mode = Mode(cfg_mode)
                logger.info("Mode changed via config: %s → %s", settings.mode.value, new_mode.value)
                settings.mode = new_mode
            except ValueError:
                pass

        from config import STTBackend
        cfg_backend = cfg.get("stt_backend")
        if cfg_backend and cfg_backend != settings.stt_backend.value:
            try:
                settings.stt_backend = STTBackend(cfg_backend)
                logger.info("STT backend changed via config: %s", cfg_backend)
            except ValueError:
                pass

        # Load/unload emotion model based on mode
        from emotion import emotion_manager as _em
        if settings.needs_emotion and not _em.is_loaded:
            _em.load()
        elif not settings.needs_emotion and _em.is_loaded:
            _em.unload()

        current_mode = settings.mode.value

        # Run prosody analysis
        prosody = await loop.run_in_executor(_executor, _run_prosody, job.audio_path)

        # Enrich: per-segment emotion + text alignment + descriptions
        prosody = await loop.run_in_executor(
            _executor, _enrich_segments, prosody, job.audio_path, job.whisper_segments,
        )

        # Generate annotated transcript and voice context
        voice_context = _generate_annotated_transcript(prosody)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = EnrichmentResult(
            job_id=job.job_id,
            status="done",
            timestamp=job.timestamp,
            text=job.text,
            audio_duration_sec=job.audio_duration_sec,
            prosody=prosody,
            voice_context=voice_context,
            mode=current_mode,
            enrichment_time_ms=round(elapsed_ms, 1),
        )

        store_result(result)
        _write_log_entry(result)
        _write_analysis_dump(result)

        logger.info(
            "Enriched job %s in %.0f ms (mode=%s, segments=%d)",
            job.job_id, elapsed_ms, current_mode,
            len(prosody.get("segments", [])),
        )

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error("Enrichment failed for job %s: %s", job.job_id, exc, exc_info=True)
        store_result(EnrichmentResult(
            job_id=job.job_id,
            status="error",
            timestamp=job.timestamp,
            text=job.text,
            mode=current_mode,
            enrichment_time_ms=round(elapsed_ms, 1),
            error=str(exc),
        ))

    finally:
        # Always delete tmp audio
        delete_tmp_file(job.audio_path)


async def start_worker() -> None:
    """Main worker loop — dequeues and processes enrichment jobs.

    Should be launched via ``asyncio.create_task`` at server startup.
    Runs indefinitely until cancelled.

    Notes:
        Jobs are processed sequentially (single-threaded) to keep RAM
        usage bounded.  The thread pool is used only to avoid blocking
        the event loop during CPU-bound analysis.
    """
    # Apply resource limits from config
    res = cfg["resources"]
    cpu_threads = res.get("worker_cpu_threads", 2)
    nice_val = res.get("worker_nice", 10)

    try:
        import torch
        torch.set_num_threads(cpu_threads)
        logger.info("Worker torch threads limited to %d", cpu_threads)
    except Exception:
        pass

    try:
        import os as _os
        _os.nice(nice_val)
        logger.info("Worker priority lowered by nice=%d", nice_val)
    except Exception:
        pass

    logger.info("Enrichment worker started")
    while True:
        try:
            job = await enrichment_queue.get()
            logger.debug("Dequeued job %s", job.job_id)
            await _process_job(job)
            enrichment_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Enrichment worker stopped")
            break
        except Exception as exc:
            logger.error("Worker loop error: %s", exc, exc_info=True)

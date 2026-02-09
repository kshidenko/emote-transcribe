"""
Emote-Transcribe — Two-tier prosody analysis microservice (v2).

FastAPI server providing:
1. **Sync layer** — OpenAI-compatible Whisper proxy (``/v1/audio/transcriptions``)
   that returns transcription text instantly.
2. **Async layer** — Background prosody + emotion enrichment that writes
   annotated transcript logs (JSONL).

Supports three STT backends (local whisper-cpp, OpenAI API, auto-fallback)
and three operation modes (full, whisper_only, prosody_only).

Usage:
    $ python server.py
    # → http://0.0.0.0:8200

    # Transcription (drop-in OpenAI replacement)
    $ curl -X POST http://localhost:8200/v1/audio/transcriptions \\
        -F "file=@voice.ogg" -F "model=whisper-large-v3-turbo" -F "language=ru"

    # Switch mode
    $ curl -X POST http://localhost:8200/v1/mode -d '{"mode": "whisper_only"}'
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, File, Form, Header, Query, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from config import settings, Mode, STTBackend
from emotion import emotion_manager, start_idle_monitor
from worker import (
    enqueue_job,
    get_result,
    get_latest_results,
    get_queue_size,
    start_worker,
)
from cleanup import start_cleanup_scheduler

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("emote-transcribe")

# ── Constants ─────────────────────────────────────────────────────────────────

VERSION = "2.0.0"

# Supported audio formats
NATIVE_FORMATS = {".wav", ".aiff", ".flac"}
CONVERT_FORMATS = {".ogg", ".opus", ".mp3", ".m4a", ".webm", ".oga"}
ALL_FORMATS = NATIVE_FORMATS | CONVERT_FORMATS

OPENAI_API_BASE = "https://api.openai.com/v1/audio/transcriptions"


# ── Audio helpers ─────────────────────────────────────────────────────────────

def convert_to_wav(input_path: str) -> str:
    """Convert audio file to 16 kHz mono WAV using ffmpeg.

    Args:
        input_path: Path to input audio file.

    Returns:
        Path to converted WAV file.

    Raises:
        RuntimeError: If ffmpeg conversion fails.
    """
    output_path = input_path + ".wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", input_path,
                "-ar", "16000", "-ac", "1",
                "-y", "-loglevel", "error",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        return output_path
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg conversion failed: {exc.stderr.decode()}")


def get_audio_duration(wav_path: str) -> float:
    """Get audio duration in seconds using ffprobe.

    Args:
        wav_path: Path to audio file.

    Returns:
        Duration in seconds, or 0.0 on failure.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                wav_path,
            ],
            capture_output=True,
            text=True,
        )
        return round(float(result.stdout.strip()), 2)
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


# ── STT backends ──────────────────────────────────────────────────────────────

async def _transcribe_local(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    model: str,
    language: str,
    response_format: str,
) -> httpx.Response:
    """Forward transcription request to local whisper-cpp server.

    Args:
        audio_bytes: Raw audio file bytes.
        filename: Original filename.
        content_type: MIME type of the audio file.
        model: Model name to pass through.
        language: Language code.
        response_format: OpenAI response format (text, json, verbose_json).

    Returns:
        httpx.Response from whisper-cpp.

    Raises:
        httpx.ConnectError: If whisper-cpp is unreachable.
    """
    url = f"{settings.whisper_url}/inference"
    async with httpx.AsyncClient(timeout=60.0) as client:
        return await client.post(
            url,
            files={"file": (filename, audio_bytes, content_type)},
            data={
                "response_format": response_format,
            },
        )


async def _transcribe_openai(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    model: str,
    language: str,
    response_format: str,
) -> httpx.Response:
    """Forward transcription request to OpenAI Whisper API.

    Args:
        audio_bytes: Raw audio file bytes.
        filename: Original filename.
        content_type: MIME type of the audio file.
        model: Ignored — uses settings.openai_whisper_model.
        language: Language code.
        response_format: OpenAI response format.

    Returns:
        httpx.Response from OpenAI.

    Raises:
        HTTPException: If no OpenAI API key is configured.
    """
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    async with httpx.AsyncClient(timeout=60.0) as client:
        return await client.post(
            OPENAI_API_BASE,
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            files={"file": (filename, audio_bytes, content_type)},
            data={
                "model": settings.openai_whisper_model,
                "language": language,
                "response_format": response_format,
            },
        )


async def transcribe(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    model: str,
    language: str,
    response_format: str,
    backend_override: Optional[str] = None,
) -> httpx.Response:
    """Route transcription to the appropriate STT backend.

    Implements the backend selection logic:
    - ``local``:  whisper-cpp only
    - ``openai``: OpenAI API only
    - ``auto``:   try local first, fallback to OpenAI on connect error

    Args:
        audio_bytes: Raw audio file bytes.
        filename: Original filename.
        content_type: MIME type.
        model: Model name (passed through to local, ignored for OpenAI).
        language: Language code.
        response_format: OpenAI response format.
        backend_override: Per-request backend override (from header/param).

    Returns:
        httpx.Response from the STT backend.
    """
    backend = STTBackend(backend_override) if backend_override else settings.stt_backend
    args = (audio_bytes, filename, content_type, model, language, response_format)

    if backend == STTBackend.LOCAL:
        return await _transcribe_local(*args)

    if backend == STTBackend.OPENAI:
        return await _transcribe_openai(*args)

    # AUTO: try local, fallback to OpenAI
    try:
        return await _transcribe_local(*args)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        logger.warning("Local whisper-cpp unreachable, falling back to OpenAI: %s", exc)
        return await _transcribe_openai(*args)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown hooks.

    Startup:
        - Ensure directories exist
        - Load emotion model (if mode requires it)
        - Start enrichment worker
        - Start cleanup scheduler

    Shutdown:
        - Cancel background tasks
        - Unload emotion model
    """
    # ── Startup ──
    settings.ensure_dirs()
    logger.info(
        "Starting Emote-Transcribe v%s (mode=%s, backend=%s)",
        VERSION, settings.mode.value, settings.stt_backend.value,
    )

    # Load emotion model if needed
    if settings.needs_emotion:
        try:
            emotion_manager.load()
        except Exception as exc:
            logger.error("Failed to load emotion model: %s", exc)

    # Start background tasks
    worker_task = asyncio.create_task(start_worker())
    cleanup_task = asyncio.create_task(start_cleanup_scheduler())
    idle_task = asyncio.create_task(start_idle_monitor())

    yield

    # ── Shutdown ──
    worker_task.cancel()
    cleanup_task.cancel()
    idle_task.cancel()
    emotion_manager.unload()
    logger.info("Emote-Transcribe shut down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Emote-Transcribe",
    description=(
        "Two-tier prosody analysis microservice: OpenAI-compatible Whisper proxy "
        "with async prosody + emotion enrichment."
    ),
    version=VERSION,
    lifespan=lifespan,
)


# ── Endpoints: Transcription (OpenAI-compatible) ─────────────────────────────

@app.post("/v1/audio/transcriptions")
async def transcription_proxy(
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3-turbo"),
    language: str = Form(""),
    response_format: str = Form("text"),
    x_stt_backend: Optional[str] = Header(None, alias="X-STT-Backend"),
    backend: Optional[str] = Query(None),
):
    """OpenAI-compatible Whisper transcription proxy.

    Drop-in replacement for ``POST https://api.openai.com/v1/audio/transcriptions``.
    Returns transcription immediately (sync) and enqueues async prosody/emotion
    enrichment in the background.

    Args:
        file: Audio file (multipart/form-data).
        model: Whisper model name (passed through to backend).
        language: ISO language code (e.g. 'ru', 'en').
        response_format: Response format — 'text', 'json', or 'verbose_json'.
        x_stt_backend: Per-request backend override header.
        backend: Per-request backend override query param.

    Returns:
        Transcription in the requested format (matching OpenAI spec).

    Raises:
        HTTPException: 400 if format unsupported, 502 if backend fails.

    Notes:
        The ``X-Enrichment-Job-Id`` response header contains the async
        enrichment job ID (if enrichment is enabled).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALL_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Supported: {sorted(ALL_FORMATS)}",
        )

    # Read audio bytes
    audio_bytes = await file.read()
    file_size_kb = len(audio_bytes) / 1024
    content_type = file.content_type or "application/octet-stream"
    logger.info("Received: %s (%.1f KB, format=%s)", file.filename, file_size_kb, ext)

    # Save to tmp for enrichment (needed for prosody analysis)
    tmp_dir = Path(settings.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    job_dir = tmp_dir / str(uuid.uuid4())
    job_dir.mkdir()
    tmp_path = job_dir / f"input{ext}"
    tmp_path.write_bytes(audio_bytes)

    # Convert to WAV if needed (for enrichment; proxy forwards original format)
    wav_path = str(tmp_path)
    if ext not in NATIVE_FORMATS:
        try:
            wav_path = convert_to_wav(str(tmp_path))
        except RuntimeError as exc:
            logger.warning("WAV conversion failed for enrichment: %s", exc)
            wav_path = str(tmp_path)  # will fail in analyzer, but proxy still works

    # Get audio duration for logging
    audio_duration = get_audio_duration(wav_path)

    # ── Forward to STT backend (always verbose_json internally) ──
    backend_override = x_stt_backend or backend
    try:
        t0 = time.time()
        stt_response = await transcribe(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.wav",
            content_type=content_type,
            model=model,
            language=language,
            response_format="verbose_json",  # always get timestamps
            backend_override=backend_override,
        )
        elapsed = time.time() - t0
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="STT backend unreachable")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("STT proxy error: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"STT backend error: {str(exc)}")

    if stt_response.status_code != 200:
        raise HTTPException(
            status_code=stt_response.status_code,
            detail=f"STT backend returned {stt_response.status_code}: {stt_response.text[:500]}",
        )

    # Parse verbose_json to get text + segments with timestamps
    try:
        whisper_data = stt_response.json()
    except json.JSONDecodeError:
        whisper_data = {"text": stt_response.text.strip(), "segments": []}

    transcription_text = whisper_data.get("text", "").strip()
    whisper_segments = whisper_data.get("segments", [])

    logger.info("Transcribed in %.2fs (backend=%s, %d segments): %s",
                elapsed, backend_override or settings.stt_backend.value,
                len(whisper_segments), transcription_text[:80])

    # ── Enqueue async enrichment ──
    job_id = None
    if settings.needs_enrichment:
        try:
            job_id = await enqueue_job(
                audio_path=wav_path,
                text=transcription_text,
                language=language,
                audio_duration_sec=audio_duration,
                whisper_segments=whisper_segments,
            )
        except asyncio.QueueFull:
            logger.warning("Enrichment queue full — skipping enrichment for this request")
    else:
        # No enrichment — clean up tmp immediately
        from cleanup import delete_tmp_file
        delete_tmp_file(str(tmp_path))
        if wav_path != str(tmp_path):
            delete_tmp_file(wav_path)

    # ── Build client response in requested format ──
    headers = {}
    if job_id:
        headers["X-Enrichment-Job-Id"] = job_id

    if response_format == "text":
        return Response(
            content=transcription_text,
            media_type="text/plain",
            headers=headers,
        )
    elif response_format == "json":
        return JSONResponse(
            content={"text": transcription_text},
            headers=headers,
        )
    else:
        # verbose_json — pass through as-is
        return JSONResponse(
            content=whisper_data,
            headers=headers,
        )


def _extract_text(response: httpx.Response, response_format: str) -> str:
    """Extract plain text from an STT backend response.

    Args:
        response: httpx.Response from the STT backend.
        response_format: Requested format ('text', 'json', 'verbose_json').

    Returns:
        Transcription text string.
    """
    if response_format == "text":
        return response.text.strip()
    try:
        data = response.json()
        return data.get("text", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return response.text.strip()


# ── Endpoints: Health ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Service health check with detailed status.

    Returns:
        JSON with service status, version, mode, emotion model info,
        and enrichment queue size.
    """
    return {
        "status": "ok",
        "service": "emote-transcribe",
        "version": VERSION,
        "mode": settings.mode.value,
        "stt_backend": settings.stt_backend.value,
        "emotion_model": {
            "name": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
            "type": "dimensional (arousal/valence/dominance)",
            "is_loaded": emotion_manager.is_loaded,
        },
        "enrichment_queue_size": get_queue_size(),
    }


# ── Endpoints: Analysis results ───────────────────────────────────────────────

@app.get("/v1/analysis/latest")
async def get_latest_analysis(n: int = Query(10, ge=1, le=100)):
    """Fetch the N most recent enrichment results.

    Args:
        n: Number of results to return (default 10, max 100).

    Returns:
        JSON list of enrichment results, newest first.
    """
    results = get_latest_results(n)
    return [_result_to_dict(r) for r in results]


@app.get("/v1/analysis/{job_id}")
async def get_analysis(job_id: str):
    """Fetch enrichment result for a specific job.

    Args:
        job_id: The enrichment job ID (from ``X-Enrichment-Job-Id`` header).

    Returns:
        JSON with enrichment result.

    Raises:
        HTTPException: 404 if job not found.
    """
    result = get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return _result_to_dict(result)


def _result_to_dict(result) -> dict[str, Any]:
    """Convert EnrichmentResult to JSON-serialisable dict.

    Args:
        result: EnrichmentResult instance.

    Returns:
        Dict representation.
    """
    return {
        "id": result.job_id,
        "status": result.status,
        "timestamp": result.timestamp,
        "text": result.text,
        "audio_duration_sec": result.audio_duration_sec,
        "mode": result.mode,
        "prosody": result.prosody,
        "voice_context": result.voice_context,
        "enrichment_time_ms": result.enrichment_time_ms,
        "error": result.error if result.error else None,
    }


# ── Endpoints: Mode switching ─────────────────────────────────────────────────

@app.post("/v1/mode")
async def switch_mode(request: Request):
    """Switch service operation mode.

    Body (JSON):
        ``{"mode": "full" | "whisper_only" | "prosody_only"}``

    Optionally also switch emotion model:
        ``{"mode": "full", "emotion_model": "wav2vec2"}``

    Returns:
        JSON with new mode and model status.

    Notes:
        Mode switch may load/unload the emotion model, which takes a few seconds.
    """
    body = await request.json()
    new_mode_str = body.get("mode")

    if not new_mode_str:
        raise HTTPException(status_code=400, detail="Missing 'mode' field")

    try:
        new_mode = Mode(new_mode_str)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {new_mode_str}. Valid: {[m.value for m in Mode]}",
        )

    old_mode = settings.mode
    settings.mode = new_mode
    logger.info("Mode switched: %s → %s", old_mode.value, new_mode.value)

    # Load/unload emotion model based on new mode
    if settings.needs_emotion and not emotion_manager.is_loaded:
        try:
            emotion_manager.load()
        except Exception as exc:
            logger.error("Failed to load emotion model: %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to load model: {str(exc)}"},
            )
    elif not settings.needs_emotion and emotion_manager.is_loaded:
        emotion_manager.unload()

    return {
        "mode": settings.mode.value,
        "emotion_loaded": emotion_manager.is_loaded,
    }


# ── Endpoints: Log browsing ──────────────────────────────────────────────────

@app.get("/v1/logs")
async def browse_logs(
    date: Optional[str] = Query(None, description="Date filter YYYY-MM-DD"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Browse annotated transcript logs.

    Args:
        date: Optional date filter (YYYY-MM-DD). If omitted, returns latest.
        limit: Maximum entries per page.
        offset: Number of entries to skip.

    Returns:
        JSON with log entries, total count, and available dates.
    """
    log_dir = Path(settings.log_dir)
    if not log_dir.exists():
        return {"entries": [], "total": 0, "dates": []}

    # List available dates
    dates = sorted(
        [f.stem for f in log_dir.glob("*.jsonl")],
        reverse=True,
    )

    if date:
        log_path = log_dir / f"{date}.jsonl"
    elif dates:
        log_path = log_dir / f"{dates[0]}.jsonl"
    else:
        return {"entries": [], "total": 0, "dates": dates}

    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"No logs for date: {date}")

    # Read entries
    entries = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {exc}")

    total = len(entries)
    entries = list(reversed(entries))  # newest first
    page = entries[offset: offset + limit]

    return {
        "entries": page,
        "total": total,
        "offset": offset,
        "limit": limit,
        "date": log_path.stem,
        "dates": dates,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=settings.port,
        log_level="info",
    )

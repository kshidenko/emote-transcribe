"""
Emote-Transcribe — Prosody Analysis Microservice.

FastAPI server that accepts audio files and returns prosody analysis:
pitch, tempo, pauses, intensity per segment plus human-readable descriptions.

Designed to run alongside OpenClaw's Whisper STT server on Mac Mini.

Usage:
    Start the server:
        $ uvicorn server:app --host 0.0.0.0 --port 8200

    Analyze audio:
        $ curl -X POST http://localhost:8200/analyze -F "file=@voice.wav"

    Health check:
        $ curl http://localhost:8200/health
"""

import os
import tempfile
import subprocess
import time
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from analyzer import analyze_prosody
from describer import generate_voice_context

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("emote-transcribe")

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Emote-Transcribe",
    description="Prosody analysis microservice — pitch, tempo, pauses, intensity, emotion.",
    version="0.1.0",
)

# Supported audio formats that Parselmouth can read directly
NATIVE_FORMATS = {".wav", ".aiff", ".flac"}

# Formats that need ffmpeg conversion to WAV
CONVERT_FORMATS = {".ogg", ".opus", ".mp3", ".m4a", ".webm", ".oga"}


def convert_to_wav(input_path: str) -> str:
    """Convert audio file to 16kHz mono WAV using ffmpeg.

    Args:
        input_path: Path to input audio file.

    Returns:
        Path to converted WAV file (caller must delete).

    Raises:
        RuntimeError: If ffmpeg conversion fails.
    """
    output_path = input_path + ".converted.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", input_path,
                "-ar", "16000",
                "-ac", "1",
                "-y",
                "-loglevel", "error",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e.stderr.decode()}")


@app.get("/health")
async def health():
    """Health check endpoint.

    Returns:
        JSON with service status and version.
    """
    return {"status": "ok", "service": "emote-transcribe", "version": "0.1.0"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze prosody of an uploaded audio file.

    Accepts WAV, OGG, OPUS, MP3, M4A, WEBM audio files.
    Returns detailed prosody analysis with per-segment breakdown.

    Args:
        file: Audio file (multipart/form-data).

    Returns:
        JSON with 'global' metrics, 'segments' list, and 'voice_context' text.

    Raises:
        HTTPException: 400 if no file or unsupported format.
        HTTPException: 500 if analysis fails.

    Example:
        >>> import requests
        >>> r = requests.post("http://localhost:8200/analyze",
        ...     files={"file": open("voice.wav", "rb")})
        >>> data = r.json()
        >>> print(data["voice_context"])
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Determine format
    ext = os.path.splitext(file.filename)[1].lower()
    needs_conversion = ext in CONVERT_FORMATS
    is_native = ext in NATIVE_FORMATS

    if not needs_conversion and not is_native:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Supported: {NATIVE_FORMATS | CONVERT_FORMATS}",
        )

    # Save uploaded file to temp
    tmp_dir = tempfile.mkdtemp(prefix="emote_")
    input_path = os.path.join(tmp_dir, f"input{ext}")
    wav_path = None

    try:
        # Write uploaded file
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        file_size_kb = len(content) / 1024
        logger.info(f"Received: {file.filename} ({file_size_kb:.1f} KB, format={ext})")

        # Convert if needed
        if needs_conversion:
            wav_path = convert_to_wav(input_path)
            analyze_path = wav_path
        else:
            analyze_path = input_path

        # Run prosody analysis
        start_time = time.time()
        result = analyze_prosody(analyze_path)
        elapsed = time.time() - start_time

        # Generate human-readable context
        voice_context = generate_voice_context(result)

        logger.info(
            f"Analyzed: {result['global']['duration_sec']}s audio in {elapsed:.2f}s "
            f"({len(result['segments'])} segments)"
        )

        return JSONResponse(content={
            **result,
            "voice_context": voice_context,
            "analysis_time_sec": round(elapsed, 3),
        })

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        # Cleanup temp files
        for path in [input_path, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


@app.post("/context")
async def context_only(file: UploadFile = File(...)):
    """Return only the human-readable voice context string.

    Lightweight endpoint — returns just the text block to prepend
    to transcription, without the full JSON analysis.

    Args:
        file: Audio file (multipart/form-data).

    Returns:
        JSON with 'voice_context' string only.
    """
    result = await analyze(file)
    body = result.body.decode() if hasattr(result, "body") else "{}"
    import json
    data = json.loads(body)
    return {"voice_context": data.get("voice_context", "")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8200,
        log_level="info",
    )

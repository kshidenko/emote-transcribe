"""
Service configuration for Emote-Transcribe.

Centralises all settings: operation mode, paths, and environment variable
overrides.  Provides runtime-mutable state (mode) for hot-switching.

Environment variables
---------------------
WHISPER_URL          Local whisper-cpp URL (default http://127.0.0.1:8178)
OPENAI_API_KEY       OpenAI API key for remote STT / fallback
OPENAI_WHISPER_MODEL OpenAI model name (default whisper-1)
STT_BACKEND          local | openai | auto (default local)
MODE                 full | whisper_only | prosody_only (default full)
PORT                 Server port (default 8200)
LOG_DIR              Annotated transcript log directory (default logs/transcripts)
ANALYSIS_LOG_DIR     Full-dump analysis JSON directory (default logs/analysis)
TMP_DIR              Temporary audio file directory (default tmp)
LOG_RETENTION_DAYS   Days to keep logs (default 30)
"""

import os
from enum import Enum
from dataclasses import dataclass, field


# ── Enums ─────────────────────────────────────────────────────────────────────

class Mode(str, Enum):
    """Service operation mode."""

    FULL = "full"
    WHISPER_ONLY = "whisper_only"
    PROSODY_ONLY = "prosody_only"


class STTBackend(str, Enum):
    """Speech-to-text backend selection."""

    LOCAL = "local"
    OPENAI = "openai"
    AUTO = "auto"


# ── Settings dataclass ────────────────────────────────────────────────────────

@dataclass
class Settings:
    """Runtime-mutable service configuration.

    Attributes:
        whisper_url: Local whisper-cpp server URL.
        openai_api_key: OpenAI API key (empty = disabled).
        openai_whisper_model: Model name for OpenAI Whisper API.
        stt_backend: Which STT backend to use by default.
        port: HTTP server port.
        mode: Current operation mode.
        log_dir: Directory for annotated transcript JSONL logs.
        analysis_log_dir: Directory for full-dump analysis JSON files.
        tmp_dir: Directory for temporary audio files.
        log_retention_days: Number of days to keep logs before rotation.

    Example:
        >>> from config import settings, Mode
        >>> settings.mode = Mode.WHISPER_ONLY
    """

    whisper_url: str = field(
        default_factory=lambda: os.getenv("WHISPER_URL", "http://127.0.0.1:8178")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_whisper_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")
    )
    stt_backend: STTBackend = field(
        default_factory=lambda: STTBackend(os.getenv("STT_BACKEND", "local"))
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("PORT", "8200"))
    )
    mode: Mode = field(
        default_factory=lambda: Mode(os.getenv("MODE", "full"))
    )
    log_dir: str = field(
        default_factory=lambda: os.getenv("LOG_DIR", "logs/transcripts")
    )
    analysis_log_dir: str = field(
        default_factory=lambda: os.getenv("ANALYSIS_LOG_DIR", "logs/analysis")
    )
    tmp_dir: str = field(
        default_factory=lambda: os.getenv("TMP_DIR", "tmp")
    )
    log_retention_days: int = field(
        default_factory=lambda: int(os.getenv("LOG_RETENTION_DAYS", "30"))
    )

    # ── helpers ───────────────────────────────────────────────────────────

    @property
    def needs_emotion(self) -> bool:
        """Whether current mode requires the emotion model loaded."""
        return self.mode == Mode.FULL

    @property
    def needs_prosody(self) -> bool:
        """Whether current mode runs prosody analysis."""
        return self.mode in (Mode.FULL, Mode.PROSODY_ONLY)

    @property
    def needs_enrichment(self) -> bool:
        """Whether current mode runs any async enrichment."""
        return self.mode != Mode.WHISPER_ONLY

    def ensure_dirs(self) -> None:
        """Create log, analysis, and tmp directories if they do not exist."""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.analysis_log_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)


# ── Module-level singleton ────────────────────────────────────────────────────

settings = Settings()

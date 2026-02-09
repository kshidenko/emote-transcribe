"""
Loader for label_config.json — single source of truth for all tunable thresholds.

Reads config from ``label_config.json`` in the project root.
Config is cached and can be hot-reloaded by calling ``reload()``.

Usage:
    >>> from label_config_loader import cfg
    >>> cfg["intensity_db"]["whisper"]
    50
    >>> cfg.reload()  # re-read from disk
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("emote-transcribe.config")

_CONFIG_PATH = Path(__file__).parent / "label_config.json"


class LabelConfig:
    """Cached, hot-reloadable label configuration.

    Reads ``label_config.json`` once on first access. Call ``reload()``
    to re-read from disk (e.g. after an agent edits the file).
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.reload()

    def reload(self) -> None:
        """Re-read config from disk."""
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            self._loaded = True
            logger.info("Loaded label_config.json")
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to load label_config.json: %s — using defaults", exc)
            self._data = {}
            self._loaded = True

    def __getitem__(self, key: str) -> Any:
        self._ensure_loaded()
        return self._data.get(key, {})

    def get(self, key: str, default: Any = None) -> Any:
        """Get a top-level config key with fallback default."""
        self._ensure_loaded()
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Update a top-level key and write back to disk.

        Reads current file, updates the key, writes back — preserves
        all other fields and formatting.

        Args:
            key: Top-level config key (e.g. 'mode').
            value: New value.
        """
        self._ensure_loaded()
        # Read fresh from disk to avoid overwriting concurrent changes
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = self._data.copy()

        data[key] = value
        try:
            with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            self._data = data
            logger.info("Updated label_config.json: %s = %s", key, value)
        except OSError as exc:
            logger.error("Failed to write label_config.json: %s", exc)


cfg = LabelConfig()

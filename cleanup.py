"""
Cleanup module for temporary files and log rotation.

Handles two responsibilities:
1. Immediate deletion of temporary WAV files after async processing.
2. Periodic rotation of annotated transcript logs (default 30-day retention).

The background scheduler runs a daily check via ``asyncio.create_task``.

Usage:
    >>> import asyncio
    >>> from cleanup import delete_tmp_file, rotate_logs, start_cleanup_scheduler
    >>> delete_tmp_file("/tmp/emote_xyz/input.wav")
    >>> asyncio.run(rotate_logs())   # delete logs older than retention period
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

from config import settings

logger = logging.getLogger("emote-transcribe.cleanup")


# ── Tmp file cleanup ──────────────────────────────────────────────────────────

def delete_tmp_file(file_path: str) -> bool:
    """Delete a temporary audio file and its parent directory if empty.

    Args:
        file_path: Absolute path to the temporary file.

    Returns:
        True if file was deleted, False if it didn't exist or deletion failed.

    Example:
        >>> delete_tmp_file("/tmp/emote_abc123/input.ogg")
        True
    """
    path = Path(file_path)
    if not path.exists():
        return False

    try:
        path.unlink()
        logger.debug("Deleted tmp file: %s", file_path)

        # Remove parent dir if empty and inside our tmp_dir
        parent = path.parent
        tmp_root = Path(settings.tmp_dir).resolve()
        if parent.resolve().is_relative_to(tmp_root):
            try:
                parent.rmdir()  # only succeeds if empty
            except OSError:
                pass

        return True
    except OSError as exc:
        logger.warning("Failed to delete tmp file %s: %s", file_path, exc)
        return False


def cleanup_tmp_dir() -> int:
    """Remove all files and empty subdirectories in the tmp directory.

    Intended as a startup sweep for orphaned temp files from previous runs.

    Returns:
        Number of files deleted.

    Notes:
        Only deletes files inside ``settings.tmp_dir``.  Does not recurse
        into directories outside the configured tmp root.
    """
    tmp_root = Path(settings.tmp_dir)
    if not tmp_root.exists():
        return 0

    count = 0
    for item in tmp_root.rglob("*"):
        if item.is_file():
            try:
                item.unlink()
                count += 1
            except OSError:
                pass

    # Remove empty subdirectories bottom-up
    for item in sorted(tmp_root.rglob("*"), reverse=True):
        if item.is_dir():
            try:
                item.rmdir()
            except OSError:
                pass

    if count > 0:
        logger.info("Startup cleanup: deleted %d orphaned tmp files", count)
    return count


# ── Log rotation ──────────────────────────────────────────────────────────────

async def rotate_logs() -> int:
    """Delete transcript log files older than the configured retention period.

    Scans ``settings.log_dir`` for JSONL files with date-based names
    (``YYYY-MM-DD.jsonl``) and removes those older than
    ``settings.log_retention_days``.

    Returns:
        Number of log files deleted.

    Example:
        >>> import asyncio
        >>> deleted = asyncio.run(rotate_logs())
        >>> print(f"Rotated {deleted} old log files")
    """
    log_dir = Path(settings.log_dir)
    if not log_dir.exists():
        return 0

    cutoff = datetime.now() - timedelta(days=settings.log_retention_days)
    deleted = 0

    for log_file in log_dir.glob("*.jsonl"):
        # Parse date from filename (YYYY-MM-DD.jsonl)
        try:
            date_str = log_file.stem  # e.g. "2026-01-10"
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue  # skip non-date-named files

        if file_date < cutoff:
            try:
                log_file.unlink()
                deleted += 1
                logger.info("Rotated old log: %s", log_file.name)
            except OSError as exc:
                logger.warning("Failed to rotate %s: %s", log_file.name, exc)

    # Also rotate analysis JSON files by modification time
    analysis_dir = Path(settings.analysis_log_dir)
    if analysis_dir.exists():
        for json_file in analysis_dir.glob("*.json"):
            try:
                mtime = datetime.fromtimestamp(json_file.stat().st_mtime)
                if mtime < cutoff:
                    json_file.unlink()
                    deleted += 1
                    logger.info("Rotated old analysis: %s", json_file.name)
            except OSError:
                continue

    if deleted > 0:
        logger.info("Log rotation complete: deleted %d files (retention=%d days)",
                     deleted, settings.log_retention_days)
    return deleted


# ── Background scheduler ─────────────────────────────────────────────────────

async def start_cleanup_scheduler() -> None:
    """Background task that runs log rotation once per day.

    Should be launched via ``asyncio.create_task`` at server startup.
    Runs indefinitely until cancelled.

    Notes:
        - First rotation runs immediately on startup.
        - Subsequent rotations every 24 hours.
        - Also cleans orphaned tmp files on startup.
    """
    # Startup cleanup
    cleanup_tmp_dir()

    # Initial rotation
    try:
        await rotate_logs()
    except Exception as exc:
        logger.error("Initial log rotation failed: %s", exc)

    # Daily loop
    while True:
        await asyncio.sleep(86400)  # 24 hours
        try:
            await rotate_logs()
        except Exception as exc:
            logger.error("Scheduled log rotation failed: %s", exc)

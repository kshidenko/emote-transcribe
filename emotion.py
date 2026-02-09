"""
Dimensional Speech Emotion Recognition via audeering wav2vec2.

Uses ``audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`` to predict
three continuous dimensions per audio segment:

- **Arousal** (0-1): energy/activation (calm → excited)
- **Valence** (0-1): sentiment (negative → positive)
- **Dominance** (0-1): assertiveness (submissive → dominant)

Usage:
    >>> from emotion import emotion_manager
    >>> emotion_manager.load()
    >>> result = emotion_manager.predict("/tmp/audio.wav")
    >>> print(f"arousal={result.arousal:.2f} valence={result.valence:.2f}")
    arousal=0.72 valence=0.58
    >>> emotion_manager.unload()

Notes:
    - Thread-safe via threading.Lock.
    - Audio is expected to be 16 kHz mono WAV.
    - Model is language-independent (acoustic features only).
"""

from __future__ import annotations

import gc
import logging
import time
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("emote-transcribe.emotion")

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EmotionResult:
    """Result of dimensional emotion prediction.

    Attributes:
        arousal: Energy/activation level (0=calm, 1=excited).
        valence: Sentiment polarity (0=negative, 1=positive).
        dominance: Assertiveness (0=submissive, 1=dominant).
        inference_ms: Wall-clock inference time in milliseconds.
    """

    arousal: float
    valence: float
    dominance: float
    inference_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to JSON-serialisable dict."""
        return {
            "arousal": round(self.arousal, 3),
            "valence": round(self.valence, 3),
            "dominance": round(self.dominance, 3),
        }


# ── Audeering model classes (from model card) ────────────────────────────────

class _RegressionHead(nn.Module):
    """Regression head for dimensional emotion prediction."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def _build_model_class():
    """Dynamically build the EmotionModel class.

    Defers transformers import to load time.

    Returns:
        EmotionModel class inheriting from Wav2Vec2PreTrainedModel.
    """
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class _EmotionModel(Wav2Vec2PreTrainedModel):
        """Wav2Vec2 model with regression head for arousal/dominance/valence."""

        _tied_weights_keys = None
        all_tied_weights_keys = {}  # compat with transformers ≥5.x

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = _RegressionHead(config)
            self.init_weights()

        def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            hidden_states = outputs[0]
            hidden_states = torch.mean(hidden_states, dim=1)
            logits = self.classifier(hidden_states)
            return hidden_states, logits

    return _EmotionModel


# ── Helpers ───────────────────────────────────────────────────────────────────

def _free_memory() -> None:
    """Force garbage collection and clear accelerator caches."""
    gc.collect()
    try:
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ── Manager (singleton) ──────────────────────────────────────────────────────

class EmotionManager:
    """Thread-safe manager for the audeering dimensional emotion model.

    Loads model and processor on demand. Auto-unloads after idle timeout
    (configurable in label_config.json → resources.emotion_idle_timeout_sec).

    Attributes:
        is_loaded: Whether model is currently in memory.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._last_predict_time: float = 0.0

    @property
    def is_loaded(self) -> bool:
        """Whether model and processor are loaded."""
        return self._model is not None and self._processor is not None

    def load(self) -> None:
        """Load audeering model and processor into memory.

        Raises:
            ImportError: If transformers is not installed.
        """
        from transformers import Wav2Vec2Processor

        with self._lock:
            if self._model is not None:
                return  # already loaded

            logger.info("Loading audeering emotion model: %s", MODEL_NAME)

            EmotionModelClass = _build_model_class()
            self._processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
            self._model = EmotionModelClass.from_pretrained(MODEL_NAME)
            self._model.eval()
            self._model.to("cpu")

            logger.info("Audeering emotion model loaded")

    def unload(self) -> None:
        """Unload model, freeing memory."""
        with self._lock:
            if self._model is not None:
                self._model = None
                self._processor = None
                _free_memory()
                logger.info("Audeering emotion model unloaded")

    def _process_audio(self, audio: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
        """Run model inference on raw audio array.

        Args:
            audio: 1D float32 numpy array of audio samples.
            sampling_rate: Audio sample rate (must be 16000).

        Returns:
            Array of [arousal, dominance, valence].
        """
        y = self._processor(audio, sampling_rate=sampling_rate)
        y = y["input_values"][0]
        y = torch.from_numpy(y.reshape(1, -1)).to("cpu")

        with torch.no_grad():
            _, logits = self._model(y)

        return logits.detach().cpu().numpy()[0]

    def predict(self, audio_path: str) -> EmotionResult:
        """Predict arousal/valence/dominance for an audio file.

        Args:
            audio_path: Path to 16 kHz mono WAV file.

        Returns:
            EmotionResult with arousal, valence, dominance.

        Raises:
            RuntimeError: If model is not loaded.
        """
        import soundfile as sf

        with self._lock:
            if self._model is None:
                raise RuntimeError("Emotion model not loaded — call load() first")

            self._last_predict_time = time.time()
            t0 = time.perf_counter()

            audio, sr = sf.read(audio_path, dtype="float32")
            # Mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # Resample to 16kHz if needed
            if sr != 16000:
                import torchaudio
                waveform = torch.from_numpy(audio).unsqueeze(0)
                audio = torchaudio.transforms.Resample(sr, 16000)(waveform).squeeze().numpy()

            # Min duration ~0.5s
            if len(audio) < 8000:
                return EmotionResult(arousal=0.5, valence=0.5, dominance=0.5)

            values = self._process_audio(audio)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        # Output order: [arousal, dominance, valence]
        arousal = float(np.clip(values[0], 0, 1))
        dominance = float(np.clip(values[1], 0, 1))
        valence = float(np.clip(values[2], 0, 1))

        return EmotionResult(
            arousal=round(arousal, 3),
            valence=round(valence, 3),
            dominance=round(dominance, 3),
            inference_ms=round(elapsed_ms, 1),
        )

    def predict_segment(
        self, audio_path: str, start: float, end: float,
    ) -> EmotionResult:
        """Predict emotion for a time range within an audio file.

        Args:
            audio_path: Path to 16 kHz mono WAV file.
            start: Segment start time in seconds.
            end: Segment end time in seconds.

        Returns:
            EmotionResult for this segment.
        """
        import soundfile as sf
        import tempfile
        import os

        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        start_frame = int(start * sr)
        end_frame = int(end * sr)
        segment = audio[start_frame:end_frame]

        # Min ~0.5s for meaningful prediction
        if len(segment) < int(0.5 * sr):
            return EmotionResult(arousal=0.5, valence=0.5, dominance=0.5)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(tmp.name, segment, sr)
            result = self.predict(tmp.name)
        finally:
            os.unlink(tmp.name)
        return result


    def check_idle_unload(self) -> bool:
        """Check if model should be unloaded due to idle timeout.

        Reads timeout from label_config.json → resources.emotion_idle_timeout_sec.

        Returns:
            True if model was unloaded, False otherwise.
        """
        if not self.is_loaded or self._last_predict_time == 0:
            return False

        from label_config_loader import cfg
        timeout = cfg["resources"].get("emotion_idle_timeout_sec", 300)
        idle_sec = time.time() - self._last_predict_time

        if idle_sec > timeout:
            logger.info("Emotion model idle for %.0fs (timeout=%ds) — unloading", idle_sec, timeout)
            self.unload()
            return True
        return False


async def start_idle_monitor() -> None:
    """Background task that periodically checks if the emotion model should
    be unloaded due to idle timeout. Runs every 60 seconds.
    """
    import asyncio
    while True:
        await asyncio.sleep(60)
        try:
            emotion_manager.check_idle_unload()
        except Exception as exc:
            logger.error("Idle monitor error: %s", exc)


# ── Module-level singleton ────────────────────────────────────────────────────

emotion_manager = EmotionManager()

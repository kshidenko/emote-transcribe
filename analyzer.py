"""
Prosody analyzer module using Parselmouth (Praat wrapper).

Extracts pitch contour, intensity, pauses, and tempo from audio segments.
All DSP-based — no neural networks, extremely fast on any hardware.

Usage:
    >>> from analyzer import analyze_prosody
    >>> result = analyze_prosody("audio.wav")
    >>> print(result["global"]["speech_rate"])
    'fast'
"""

import parselmouth
from parselmouth.praat import call
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ProsodySegment:
    """Single segment of prosody analysis.

    Attributes:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        pitch_mean: Mean F0 in Hz (0 if unvoiced).
        pitch_min: Minimum F0 in Hz.
        pitch_max: Maximum F0 in Hz.
        pitch_trend: Direction of pitch change ('rising', 'falling', 'flat').
        intensity_mean: Mean intensity in dB.
        intensity_label: Human label ('loud', 'normal', 'soft', 'whisper').
        tempo_syllables_sec: Estimated syllables per second.
        tempo_label: Human label ('very_fast', 'fast', 'normal', 'slow', 'very_slow').
        pause_after_ms: Duration of silence after this segment in ms.
        pause_label: Human label ('none', 'short', 'medium', 'long', 'very_long').
    """
    start: float
    end: float
    pitch_mean: float
    pitch_min: float
    pitch_max: float
    pitch_trend: str
    intensity_mean: float
    intensity_label: str
    tempo_syllables_sec: float
    tempo_label: str
    pause_after_ms: float
    pause_label: str


@dataclass
class GlobalProsody:
    """Aggregate prosody metrics for the entire audio.

    Attributes:
        duration_sec: Total audio duration in seconds.
        avg_pitch_hz: Average F0 across voiced regions.
        pitch_range_hz: Tuple of (min, max) F0.
        avg_intensity_db: Average intensity in dB.
        avg_tempo_syllables_sec: Average estimated syllable rate.
        speech_rate: Human label for overall speech rate.
        total_pause_ratio: Ratio of silence to total duration (0.0 - 1.0).
        voiced_ratio: Ratio of voiced frames to total frames.
    """
    duration_sec: float
    avg_pitch_hz: float
    pitch_range_hz: tuple
    avg_intensity_db: float
    avg_tempo_syllables_sec: float
    speech_rate: str
    total_pause_ratio: float
    voiced_ratio: float


def load_audio(file_path: str) -> parselmouth.Sound:
    """Load audio file as Parselmouth Sound object.

    Args:
        file_path: Path to WAV file (16kHz mono recommended).

    Returns:
        Parselmouth Sound object.

    Raises:
        FileNotFoundError: If file does not exist.
        RuntimeError: If audio format is unsupported.
    """
    return parselmouth.Sound(file_path)


def detect_pauses(
    sound: parselmouth.Sound,
    min_silence_duration: float = 0.15,
    silence_threshold: float = -25.0,
) -> list[dict]:
    """Detect silent pauses in audio using intensity-based VAD.

    Args:
        sound: Parselmouth Sound object.
        min_silence_duration: Minimum silence duration to count as pause (seconds).
        silence_threshold: Intensity threshold relative to max (dB).

    Returns:
        List of dicts with 'start', 'end', 'duration_ms' for each pause.
    """
    intensity = sound.to_intensity(time_step=0.01)
    times = intensity.xs()
    values = [intensity.get_value(t) for t in times]

    if not values:
        return []

    max_intensity = max(v for v in values if v is not None and not np.isnan(v))
    threshold = max_intensity + silence_threshold

    pauses = []
    in_silence = False
    silence_start = 0.0

    for t, v in zip(times, values):
        if v is None or np.isnan(v):
            v = 0.0

        if v < threshold:
            if not in_silence:
                in_silence = True
                silence_start = t
        else:
            if in_silence:
                duration = t - silence_start
                if duration >= min_silence_duration:
                    pauses.append({
                        "start": round(silence_start, 3),
                        "end": round(t, 3),
                        "duration_ms": round(duration * 1000, 0),
                    })
                in_silence = False

    # Handle trailing silence
    if in_silence:
        duration = times[-1] - silence_start
        if duration >= min_silence_duration:
            pauses.append({
                "start": round(silence_start, 3),
                "end": round(times[-1], 3),
                "duration_ms": round(duration * 1000, 0),
            })

    return pauses


def compute_pitch_trend(pitch_values: list[float]) -> str:
    """Determine pitch direction from a sequence of F0 values.

    Args:
        pitch_values: List of F0 values (Hz), zeros excluded.

    Returns:
        'rising', 'falling', or 'flat'.
    """
    if len(pitch_values) < 2:
        return "flat"

    first_third = np.mean(pitch_values[: len(pitch_values) // 3 + 1])
    last_third = np.mean(pitch_values[-(len(pitch_values) // 3 + 1):])

    diff_ratio = (last_third - first_third) / max(first_third, 1.0)

    if diff_ratio > 0.05:
        return "rising"
    elif diff_ratio < -0.05:
        return "falling"
    return "flat"


def classify_intensity(db: float) -> str:
    """Classify intensity level into human-readable label.

    Args:
        db: Intensity in decibels.

    Returns:
        One of 'whisper', 'soft', 'normal', 'loud'.
    """
    if db < 45:
        return "whisper"
    elif db < 55:
        return "soft"
    elif db < 70:
        return "normal"
    return "loud"


def classify_tempo(syllables_per_sec: float) -> str:
    """Classify speaking tempo into human-readable label.

    Args:
        syllables_per_sec: Estimated syllable rate.

    Returns:
        One of 'very_slow', 'slow', 'normal', 'fast', 'very_fast'.

    Notes:
        Average conversational speech is ~4-5 syllables/sec for Russian.
    """
    if syllables_per_sec < 2.5:
        return "very_slow"
    elif syllables_per_sec < 3.5:
        return "slow"
    elif syllables_per_sec < 5.0:
        return "normal"
    elif syllables_per_sec < 6.5:
        return "fast"
    return "very_fast"


def classify_pause(ms: float) -> str:
    """Classify pause duration into human-readable label.

    Args:
        ms: Pause duration in milliseconds.

    Returns:
        One of 'none', 'short', 'medium', 'long', 'very_long'.
    """
    if ms < 100:
        return "none"
    elif ms < 300:
        return "short"
    elif ms < 700:
        return "medium"
    elif ms < 1500:
        return "long"
    return "very_long"


def estimate_syllable_rate(
    sound: parselmouth.Sound,
    start: float,
    end: float,
) -> float:
    """Estimate syllable rate using intensity peaks (nuclei detection).

    Uses a simplified version of the de Jong & Wempe (2009) approach:
    count intensity peaks above a threshold as syllable nuclei.

    Args:
        sound: Parselmouth Sound object.
        start: Segment start time (seconds).
        end: Segment end time (seconds).

    Returns:
        Estimated syllables per second.
    """
    duration = end - start
    if duration < 0.1:
        return 0.0

    segment = sound.extract_part(from_time=start, to_time=end)
    intensity = segment.to_intensity(time_step=0.01)
    times = intensity.xs()
    values = np.array([intensity.get_value(t) for t in times])

    # Replace NaN with 0
    values = np.nan_to_num(values, nan=0.0)

    if len(values) < 3:
        return 0.0

    # Find peaks (local maxima above mean)
    mean_val = np.mean(values)
    peak_count = 0
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1] and values[i] > mean_val:
            peak_count += 1

    return round(peak_count / duration, 2) if duration > 0 else 0.0


def segment_by_pauses(
    sound: parselmouth.Sound,
    pauses: list[dict],
    min_segment_duration: float = 0.3,
) -> list[tuple[float, float, float]]:
    """Split audio into speech segments based on detected pauses.

    Args:
        sound: Parselmouth Sound object.
        pauses: List of pause dicts from detect_pauses().
        min_segment_duration: Minimum segment length (seconds).

    Returns:
        List of (start, end, pause_after_ms) tuples.
    """
    duration = sound.get_total_duration()
    segments = []
    current_start = 0.0

    for pause in pauses:
        seg_end = pause["start"]
        if seg_end - current_start >= min_segment_duration:
            segments.append((
                round(current_start, 3),
                round(seg_end, 3),
                pause["duration_ms"],
            ))
        current_start = pause["end"]

    # Final segment after last pause
    if duration - current_start >= min_segment_duration:
        segments.append((round(current_start, 3), round(duration, 3), 0.0))

    # If no pauses detected, treat entire audio as one segment
    if not segments and duration >= min_segment_duration:
        segments.append((0.0, round(duration, 3), 0.0))

    return segments


def analyze_segment(
    sound: parselmouth.Sound,
    pitch_obj: parselmouth.Pitch,
    intensity_obj: parselmouth.Intensity,
    start: float,
    end: float,
    pause_after_ms: float,
) -> ProsodySegment:
    """Analyze prosody features for a single speech segment.

    Args:
        sound: Parselmouth Sound object.
        pitch_obj: Pre-computed Pitch object.
        intensity_obj: Pre-computed Intensity object.
        start: Segment start time (seconds).
        end: Segment end time (seconds).
        pause_after_ms: Duration of pause after this segment (ms).

    Returns:
        ProsodySegment with all computed features.
    """
    # Extract pitch values for this segment
    pitch_values = []
    time_step = 0.01
    t = start
    while t <= end:
        val = call(pitch_obj, "Get value at time", t, "Hertz", "Linear")
        if val and not np.isnan(val) and val > 0:
            pitch_values.append(val)
        t += time_step

    pitch_mean = round(np.mean(pitch_values), 1) if pitch_values else 0.0
    pitch_min = round(min(pitch_values), 1) if pitch_values else 0.0
    pitch_max = round(max(pitch_values), 1) if pitch_values else 0.0
    pitch_trend = compute_pitch_trend(pitch_values)

    # Intensity
    int_mean = call(intensity_obj, "Get mean", start, end, "energy")
    int_mean = round(int_mean, 1) if int_mean and not np.isnan(int_mean) else 0.0

    # Tempo
    syllable_rate = estimate_syllable_rate(sound, start, end)

    return ProsodySegment(
        start=start,
        end=end,
        pitch_mean=pitch_mean,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        pitch_trend=pitch_trend,
        intensity_mean=int_mean,
        intensity_label=classify_intensity(int_mean),
        tempo_syllables_sec=syllable_rate,
        tempo_label=classify_tempo(syllable_rate),
        pause_after_ms=pause_after_ms,
        pause_label=classify_pause(pause_after_ms),
    )


def analyze_prosody(file_path: str) -> dict:
    """Run full prosody analysis on an audio file.

    Detects pauses, segments the audio, and extracts pitch, intensity,
    tempo, and pause features per segment plus global aggregates.

    Args:
        file_path: Path to WAV audio file (16kHz mono recommended).

    Returns:
        Dict with 'global' (GlobalProsody) and 'segments' (list of ProsodySegment).

    Example:
        >>> result = analyze_prosody("/tmp/voice.wav")
        >>> print(result["global"]["speech_rate"])
        'normal'
        >>> for seg in result["segments"]:
        ...     print(f"{seg['start']}-{seg['end']}: {seg['pitch_trend']}")
    """
    sound = load_audio(file_path)
    duration = sound.get_total_duration()

    # Pre-compute pitch and intensity for the whole file
    pitch_obj = sound.to_pitch(time_step=0.01)
    intensity_obj = sound.to_intensity(time_step=0.01)

    # Detect pauses
    pauses = detect_pauses(sound)

    # Segment by pauses
    raw_segments = segment_by_pauses(sound, pauses)

    # Analyze each segment
    segments = []
    for start, end, pause_ms in raw_segments:
        seg = analyze_segment(sound, pitch_obj, intensity_obj, start, end, pause_ms)
        segments.append(seg)

    # Global metrics
    all_pitch = [s.pitch_mean for s in segments if s.pitch_mean > 0]
    all_intensity = [s.intensity_mean for s in segments if s.intensity_mean > 0]
    all_tempo = [s.tempo_syllables_sec for s in segments if s.tempo_syllables_sec > 0]
    total_pause_time = sum(p["duration_ms"] for p in pauses) / 1000.0

    # Voiced ratio from pitch object
    voiced_frames = 0
    total_frames = 0
    t = 0.0
    while t <= duration:
        val = call(pitch_obj, "Get value at time", t, "Hertz", "Linear")
        total_frames += 1
        if val and not np.isnan(val) and val > 0:
            voiced_frames += 1
        t += 0.01

    avg_pitch = round(np.mean(all_pitch), 1) if all_pitch else 0.0
    avg_intensity = round(np.mean(all_intensity), 1) if all_intensity else 0.0
    avg_tempo = round(np.mean(all_tempo), 2) if all_tempo else 0.0

    global_prosody = GlobalProsody(
        duration_sec=round(duration, 2),
        avg_pitch_hz=avg_pitch,
        pitch_range_hz=(
            round(min(all_pitch), 1) if all_pitch else 0.0,
            round(max(all_pitch), 1) if all_pitch else 0.0,
        ),
        avg_intensity_db=avg_intensity,
        avg_tempo_syllables_sec=avg_tempo,
        speech_rate=classify_tempo(avg_tempo),
        total_pause_ratio=round(total_pause_time / duration, 3) if duration > 0 else 0.0,
        voiced_ratio=round(voiced_frames / max(total_frames, 1), 3),
    )

    return {
        "global": asdict(global_prosody),
        "segments": [asdict(s) for s in segments],
    }

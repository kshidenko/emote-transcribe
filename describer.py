"""
Human-readable description generator for prosody analysis.

Converts numeric prosody data into natural language descriptions
suitable for LLM context and ElevenLabs v3 audio tags.

Usage:
    >>> from describer import describe_segment, describe_global, generate_voice_context
    >>> desc = describe_segment(segment_dict)
    >>> print(desc)
    'speaking fast with rising pitch, loud and energetic, followed by a long pause'
"""


def describe_pitch(pitch_mean: float, pitch_trend: str) -> str:
    """Describe pitch characteristics in natural language.

    Args:
        pitch_mean: Mean F0 in Hz.
        pitch_trend: One of 'rising', 'falling', 'flat'.

    Returns:
        Human-readable pitch description.
    """
    if pitch_mean == 0:
        return ""

    parts = []

    if pitch_mean > 250:
        parts.append("high-pitched voice")
    elif pitch_mean > 180:
        parts.append("mid-range pitch")
    elif pitch_mean > 120:
        parts.append("low pitch")
    else:
        parts.append("very low pitch")

    if pitch_trend == "rising":
        parts.append("voice rising")
    elif pitch_trend == "falling":
        parts.append("voice dropping")

    return ", ".join(parts)


def describe_tempo(tempo_label: str, syllables_sec: float) -> str:
    """Describe speaking tempo in natural language.

    Args:
        tempo_label: One of 'very_slow', 'slow', 'normal', 'fast', 'very_fast'.
        syllables_sec: Syllable rate for reference.

    Returns:
        Human-readable tempo description.
    """
    mapping = {
        "very_slow": "speaking very slowly, deliberate pace",
        "slow": "speaking slowly",
        "normal": "normal speaking pace",
        "fast": "speaking quickly",
        "very_fast": "speaking very fast, rushed",
    }
    return mapping.get(tempo_label, "")


def describe_intensity(intensity_label: str) -> str:
    """Describe voice intensity in natural language.

    Args:
        intensity_label: One of 'whisper', 'soft', 'normal', 'loud'.

    Returns:
        Human-readable intensity description.
    """
    mapping = {
        "whisper": "whispering",
        "soft": "speaking softly",
        "normal": "",
        "loud": "speaking loudly",
    }
    return mapping.get(intensity_label, "")


def describe_pause(pause_label: str, pause_ms: float) -> str:
    """Describe pause after segment in natural language.

    Args:
        pause_label: One of 'none', 'short', 'medium', 'long', 'very_long'.
        pause_ms: Pause duration in ms.

    Returns:
        Human-readable pause description.
    """
    mapping = {
        "none": "",
        "short": "brief pause",
        "medium": "noticeable pause",
        "long": f"long pause ({int(pause_ms)}ms)",
        "very_long": f"very long pause ({int(pause_ms)}ms), hesitating",
    }
    return mapping.get(pause_label, "")


def describe_emotion(emotion, confidence: float | None = None) -> str:
    """Describe detected emotion in natural language.

    Handles both categorical labels (str) and dimensional dicts
    (arousal/valence/dominance).

    Args:
        emotion: Emotion label string, dimensional dict, or None.
        confidence: Confidence score (for categorical labels only).

    Returns:
        Human-readable emotion description, or empty string.
    """
    if emotion is None:
        return ""

    # Dimensional emotion (audeering model)
    if isinstance(emotion, dict):
        ar = emotion.get("arousal", 0.5)
        va = emotion.get("valence", 0.5)
        if ar > 0.65 and va > 0.55:
            return "sounds excited"
        if ar > 0.65 and va < 0.4:
            return "sounds tense"
        if ar < 0.35 and va < 0.4:
            return "sounds sad"
        if ar < 0.35 and va > 0.55:
            return "sounds serene"
        return ""

    # Categorical emotion (string label)
    if not emotion or emotion == "neutral":
        return ""

    word = {"happy": "happy", "angry": "angry", "sad": "sad",
            "fearful": "anxious", "disgusted": "disgusted",
            "surprised": "surprised"}.get(emotion, emotion)
    if not word:
        return ""
    if confidence and confidence >= 0.7:
        return f"sounds {word}"
    elif confidence and confidence >= 0.5:
        return f"possibly {word}"
    return ""


def describe_segment(segment: dict) -> str:
    """Generate a natural language description of a prosody segment.

    Combines pitch, tempo, intensity, emotion, and pause into a single
    human-readable sentence suitable for LLM context.

    Args:
        segment: Dict from analyzer/worker with prosody + optional emotion fields.

    Returns:
        Natural language description string.

    Example:
        >>> seg = {"pitch_trend": "rising", "pitch_mean": 200,
        ...        "tempo_label": "fast", "tempo_syllables_sec": 5.5,
        ...        "intensity_label": "loud", "pause_label": "long",
        ...        "pause_after_ms": 900, "emotion": "happy",
        ...        "emotion_confidence": 0.85}
        >>> describe_segment(seg)
        'speaking quickly, mid-range pitch, voice rising, speaking loudly, sounds happy, long pause (900ms)'
    """
    parts = []

    tempo_desc = describe_tempo(seg["tempo_label"], seg["tempo_syllables_sec"]) if "tempo_label" in (seg := segment) else ""
    if tempo_desc:
        parts.append(tempo_desc)

    pitch_desc = describe_pitch(segment.get("pitch_mean", 0), segment.get("pitch_trend", "flat"))
    if pitch_desc:
        parts.append(pitch_desc)

    intensity_desc = describe_intensity(segment.get("intensity_label", "normal"))
    if intensity_desc:
        parts.append(intensity_desc)

    emotion_desc = describe_emotion(segment.get("emotion"))
    if emotion_desc:
        parts.append(emotion_desc)

    pause_desc = describe_pause(segment.get("pause_label", "none"), segment.get("pause_after_ms", 0))
    if pause_desc:
        parts.append(pause_desc)

    return ", ".join(parts) if parts else "neutral speech"


def describe_global(global_data: dict) -> str:
    """Generate a summary description of the overall voice characteristics.

    Args:
        global_data: Dict from analyzer.analyze_prosody()['global'].

    Returns:
        One-line summary of overall speaking style.

    Example:
        >>> g = {"speech_rate": "fast", "avg_pitch_hz": 190,
        ...      "avg_intensity_db": 72, "total_pause_ratio": 0.05}
        >>> describe_global(g)
        'Overall: fast pace, mid-range pitch, loud, few pauses (energetic delivery)'
    """
    parts = ["Overall:"]

    rate = global_data.get("speech_rate", "normal")
    parts.append(f"{rate} pace")

    pitch = global_data.get("avg_pitch_hz", 0)
    if pitch > 250:
        parts.append("high pitch")
    elif pitch > 180:
        parts.append("mid-range pitch")
    elif pitch > 0:
        parts.append("low pitch")

    db = global_data.get("avg_intensity_db", 0)
    if db > 70:
        parts.append("loud")
    elif db < 50:
        parts.append("quiet")

    pause_ratio = global_data.get("total_pause_ratio", 0)
    if pause_ratio > 0.3:
        parts.append("many pauses (hesitant)")
    elif pause_ratio < 0.1:
        parts.append("few pauses")

    # Infer overall energy
    if rate in ("fast", "very_fast") and db > 65:
        parts.append("(energetic delivery)")
    elif rate in ("slow", "very_slow") and db < 55:
        parts.append("(subdued delivery)")

    return " ".join(parts)


def generate_voice_context(analysis: dict) -> str:
    """Generate a complete voice analysis block for LLM context.

    This is the main function to call. It produces a text block that
    can be prepended to the transcription text, giving the LLM full
    context about how the user is speaking.

    Args:
        analysis: Full dict from analyzer.analyze_prosody().

    Returns:
        Multi-line text block with voice analysis.

    Example:
        >>> result = analyze_prosody("voice.wav")
        >>> context = generate_voice_context(result)
        >>> print(context)
        [voice analysis]
        Overall: fast pace, mid-range pitch, loud, few pauses (energetic delivery)
        Segments:
        - 0.0-1.8s: speaking quickly, voice rising, speaking loudly
        - 2.6-5.2s: speaking slowly, voice dropping, speaking softly, long pause (800ms)
        [/voice analysis]
    """
    lines = ["[voice analysis]"]

    # Global summary
    global_desc = describe_global(analysis.get("global", {}))
    lines.append(global_desc)

    # Per-segment descriptions
    segments = analysis.get("segments", [])
    if segments:
        lines.append("Segments:")
        for seg in segments:
            time_range = f"{seg['start']:.1f}-{seg['end']:.1f}s"
            desc = describe_segment(seg)
            lines.append(f"- {time_range}: {desc}")

    lines.append("[/voice analysis]")
    return "\n".join(lines)

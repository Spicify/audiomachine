from dataclasses import dataclass, asdict
from typing import Dict, Any


DEFAULT_VOICE_SETTINGS: Dict[str, Any] = {
    "stability": 0.35,           # lower -> more expressive
    "similarity_boost": 0.85,    # higher -> closer to timbre, still expressive
    "style": 0.50,               # mid -> balanced stylistic intensity
    "use_speaker_boost": True,   # presence
}


TOOLTIPS: Dict[str, str] = {
    "stability": "Lower = more expressive emotional range; Higher = more consistent/neutral.",
    "similarity_boost": "Higher = closer to reference timbre; may reduce expressive variation.",
    "style": "Strength of stylistic delivery; mid values keep variation without over-amping.",
    "use_speaker_boost": "Adds presence/volume to make the voice cut through.",
}


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def normalize_settings(d: Dict[str, Any]) -> Dict[str, Any]:
    out = DEFAULT_VOICE_SETTINGS.copy()
    try:
        incoming = {k: v for k, v in (d or {}).items() if k in out}
        out.update(incoming)
        out["stability"] = clamp01(float(out["stability"]))
        out["similarity_boost"] = clamp01(float(out["similarity_boost"]))
        out["style"] = clamp01(float(out["style"]))
        out["use_speaker_boost"] = bool(out["use_speaker_boost"])
    except Exception:
        # If anything goes wrong, return strict defaults
        return DEFAULT_VOICE_SETTINGS.copy()
    return out

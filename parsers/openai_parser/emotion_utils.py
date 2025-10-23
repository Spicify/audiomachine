from __future__ import annotations
from utils.mode import get_emotions_mode
import difflib

import os
import re
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Set, Tuple

from config_loader import EMOTION_TAGS, SPEECH_VERBS, VERB_TO_EMOTION, ADVERB_TO_EMOTION

# Canonical, ElevenLabs-friendly vocabulary (keep small and stable)
ELEVENLABS_SAFE = {
    "calm", "soft", "tense", "angry", "sad", "happy", "excited", "fearful", "surprised",
    "serious", "playful", "sarcastic", "tender", "confident", "nervous", "gentle", "warm", "cold",
}

# Frequently-seen variants/synonyms/typos → canonical
SYN_TO_CANON = {
    # calm-ish
    "serene": "calm", "relaxed": "calm", "placid": "calm", "steady": "calm",
    # soft-ish
    "soothing": "soft", "mellow": "soft", "murmured": "soft",
    # tense-ish
    "anxious": "nervous", "worried": "nervous", "edgy": "tense", "strained": "tense",
    # angry-ish
    "furious": "angry", "irritated": "angry", "annoyed": "angry", "mad": "angry",
    # sad-ish
    "melancholic": "sad", "melancholy": "sad", "somber": "sad", "downcast": "sad",
    # happy/excited-ish
    "joyful": "happy", "cheerful": "happy", "elated": "excited", "euphoric": "excited",
    # fearful-ish
    "afraid": "fearful", "scared": "fearful", "terrified": "fearful",
    # surprised-ish
    "startled": "surprised", "shocked": "surprised",
    # serious-ish
    "stern": "serious", "grave": "serious",
    # playful-ish
    "flirty": "playful", "teasing": "playful", "mischievous": "playful",
    # sarcastic-ish
    "wry": "sarcastic", "dry": "sarcastic", "ironic": "sarcastic",
    # tender-ish
    "affectionate": "tender", "loving": "tender", "caring": "tender",
    # confident-ish
    "assured": "confident", "bold": "confident", "certain": "confident",
    # nervous-ish
    "uneasy": "nervous", "restless": "nervous", "jittery": "nervous",
    # gentle-ish
    "kind": "gentle", "mild": "gentle", "tenderly": "tender",  # helps suffix cases
    # warm/cold
    "kindly": "warm", "icy": "cold",
}


def _normalize_raw_tag(t: str) -> str:
    t = (t or "").strip().lower().replace("-", " ")
    # common derivational suffixes — only strip if it helps matching adjectives
    for suf in ("ly", "ness", "ing", "ed"):
        if t.endswith(suf) and len(t) > len(suf)+2:
            t = t[: -len(suf)]
            break
    # keep first token (prefer single word)
    return t.split()[0] if t else t


def canonicalize_emotion(tag: str) -> str:
    """Map arbitrary model tag → small ElevenLabs-safe set, with fuzzy fallback."""
    raw = _normalize_raw_tag(tag)
    if not raw:
        return "calm"
    # exact safe
    if raw in ELEVENLABS_SAFE:
        return raw
    # synonym
    if raw in SYN_TO_CANON:
        mapped = SYN_TO_CANON[raw]
        return mapped if mapped in ELEVENLABS_SAFE else "calm"
    # fuzzy to SAFE
    match = difflib.get_close_matches(
        raw, list(ELEVENLABS_SAFE), n=1, cutoff=0.78)
    if match:
        return match[0]
    return "calm"


def get_allowed_emotions() -> Set[str]:
    """Build the allowed emotion tag set used in the system prompt.

    Diagnosis notes:
    - ONLY keys from EMOTION_TAGS (configs/emotion_tags.json) are aggregated here.
    - VERB_TO_EMOTION and ADVERB_TO_EMOTION are used later for normalization/derivation
      and are constrained to this allowed set.
    - SPEECH_VERBS is not included in the prompt and does not expand ALLOWED_EMOTIONS.
    """
    allowed: Set[str] = set()
    for cat in EMOTION_TAGS.values():
        for tag in cat.keys():
            allowed.add(tag.strip().lower())
    # Optional debug: print size of allowed set when DEBUG_EMOTIONS=1 is set.
    try:
        if os.getenv("DEBUG_EMOTIONS") == "1":
            print("ALLOWED_EMOTIONS size:", len(allowed))
    except Exception:
        # Never fail due to diagnostics
        pass
    return allowed


def canonicalize_emotion(tag: str) -> str:
    return (tag or "").strip().lower()


def build_emotion_kb() -> Dict[str, str]:
    """Merge verb/adverb mappings into a single kb -> canonical emotion.

    Values are canonicalized lower-case tags.
    """
    allowed = get_allowed_emotions()
    kb: Dict[str, str] = {}
    for verb, mapped in VERB_TO_EMOTION.items():
        v = (verb or "").strip().lower()
        m = (mapped or "").strip().lower()
        if m in allowed:
            kb[v] = m
    for adv, mapped in ADVERB_TO_EMOTION.items():
        a = (adv or "").strip().lower()
        m = (mapped or "").strip().lower()
        if m in allowed:
            kb[a] = m
    return kb


class EmotionMemory:
    def __init__(self, max_history_per_character: int = 8):
        self.max_history = max_history_per_character
        self._hist: Dict[str, Deque[str]] = defaultdict(
            lambda: deque(maxlen=self.max_history))

    def push(self, character: str, emotions: List[str]) -> None:
        ch = (character or "").strip().lower()
        for e in emotions:
            self._hist[ch].append(canonicalize_emotion(e))

    def last_n(self, character: str, n: int = 4) -> List[str]:
        ch = (character or "").strip().lower()
        dq = self._hist.get(ch) or deque()
        return list(list(dq)[-n:])


_WORD_RE = re.compile(r"[A-Za-z']+")


def _extract_words(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def derive_emotion_from_text(text: str, kb: Dict[str, str], allowed: Set[str]) -> Optional[str]:
    for w in _extract_words(text):
        mapped = kb.get(w)
        if mapped in allowed:
            return mapped
    return None


def diversify_emotion(second: str, recent: List[str], allowed: Set[str]) -> str:
    c = canonicalize_emotion(second)
    if c and c not in recent:
        return c
    # pick a different allowed one not in recent if possible
    for e in allowed:
        if e not in recent and e != c:
            return e
    return c or "calm"


def ensure_two_emotions(character: str, emotions, text: str, kb=None, allowed=None, memory=None):
    """
    Return exactly 2 emotions.
    - In "strict_list" mode: clamp to `allowed` if provided; fallback to safe defaults if needed.
    - In "freeform_map" mode: canonicalize to ELEVENLABS_SAFE, then top-up deterministically.
    """
    ems = [str(e or "").strip()
           for e in (emotions or []) if str(e or "").strip()]
    out = []

    mode = get_emotions_mode()
    if mode == "strict_list":
        allowed = set(allowed or [])
        for e in ems:
            if e in allowed:
                out.append(e)
            if len(out) == 2:
                break
        # top-up from allowed deterministically
        if len(out) < 2:
            for fb in ("calm", "soft", "tense", "warm", "gentle", "serious"):
                if not allowed or fb in allowed:
                    if fb not in out:
                        out.append(fb)
                if len(out) == 2:
                    break
        if not out:
            out = ["calm", "soft"]
        if len(out) == 1:
            out.append(out[0])
        return out[:2]

    # freeform_map: canonicalize + top-up
    for e in ems:
        ce = canonicalize_emotion(e)
        if ce not in out:
            out.append(ce)
        if len(out) == 2:
            break

    if len(out) < 2:
        for fb in ("calm", "soft", "tense", "warm", "gentle", "serious"):
            ce = canonicalize_emotion(fb)
            if ce not in out:
                out.append(ce)
            if len(out) == 2:
                break

    if not out:
        out = ["calm", "soft"]
    if len(out) == 1:
        out.append(out[0])
    return out[:2]

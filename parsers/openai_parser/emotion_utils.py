from __future__ import annotations

import os
import re
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Set, Tuple

from config_loader import EMOTION_TAGS, SPEECH_VERBS, VERB_TO_EMOTION, ADVERB_TO_EMOTION


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


def ensure_two_emotions(
    character: str,
    emotions: List[str],
    text: str,
    kb: Dict[str, str],
    allowed: Set[str],
    memory: EmotionMemory,
) -> List[str]:
    # Normalize and deduplicate
    norm = []
    seen: Set[str] = set()
    for e in emotions or []:
        ce = canonicalize_emotion(e)
        if ce and ce in allowed and ce not in seen:
            norm.append(ce)
            seen.add(ce)

    # Derive at least one from text if empty
    if not norm:
        d = derive_emotion_from_text(text, kb, allowed) or "calm"
        norm = [d]

    # If only one, derive a complementary one
    if len(norm) == 1:
        d2 = derive_emotion_from_text(text, kb, allowed) or "calm"
        # diversify against recent
        d2 = diversify_emotion(d2, memory.last_n(character, 4), allowed)
        if d2 == norm[0]:
            # choose any other
            for e in allowed:
                if e != norm[0]:
                    d2 = e
                    break
        norm.append(d2)

    # If more than 2, keep first two
    if len(norm) > 2:
        norm = norm[:2]

    # Guarantee length 2
    while len(norm) < 2:
        add = "calm" if "calm" not in norm else next(
            iter(allowed - set(norm)), "calm")
        norm.append(add)

    try:
        print(f"[EMO] text='{text[:60]}' emotions={norm}", flush=True)
    except Exception:
        pass
    return norm[:2]

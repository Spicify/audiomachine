from __future__ import annotations
from typing import Any, Dict, List, Tuple
from pydantic import ValidationError

from .core_types import DialogueLine, ParserState
from .emotion_utils import ensure_two_emotions, canonicalize_emotion
from utils.mode import get_emotions_mode


def validate_and_fix(
    items: List[Dict[str, Any]],
    warnings: List[str],
    state: ParserState,
    *,
    kb=None,
    allowed_emotions=None,
    memory=None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    result: List[Dict[str, Any]] = []
    base_valid = 0
    base_rejected = 0

    allowed_chars = set(state.known_characters or []) | {
        "Narrator", "Ambiguous"}

    for it in items or []:
        if str(it.get("character", "")).upper() == "REJECTED":
            result.append(it)
            base_rejected += 1
            continue

        char = (it.get("character") or "").strip() or "Ambiguous"
        txt = (it.get("text") or "").strip()
        ems = it.get("emotions") or []

        if not txt:
            base_rejected += 1
            continue

        if char not in allowed_chars:
            char = "Ambiguous"

        # ALWAYS enforce exactly two emotions
        filled = ensure_two_emotions(
            char, ems, txt, kb, allowed_emotions, memory)

        # Extra mapping safety for freeform: keep within SAFE vocab
        mode = get_emotions_mode()
        if mode != "strict_list":
            mapped = []
            for e in filled[:2]:
                ce = canonicalize_emotion(e)
                mapped.append(ce)
            filled = mapped[:2]
            # ensure 2 after mapping
            if len(filled) < 2:
                for fb in ("calm", "soft", "tense", "warm"):
                    ce = canonicalize_emotion(fb)
                    if ce not in filled:
                        filled.append(ce)
                    if len(filled) == 2:
                        break

        fixed = {"character": char, "text": txt, "emotions": filled[:2]}
        if char == "Ambiguous" and it.get("candidates"):
            cands = [str(c).strip()
                     for c in (it.get("candidates") or []) if str(c).strip()]
            if cands:
                fixed["candidates"] = cands[:5]

        try:
            DialogueLine(**fixed)
        except ValidationError as ve:
            warnings.append(
                f"Validation dropped line: {fixed} ({ve.errors()[:1]})")
            continue

        # update rolling state/memory
        if char not in ("Narrator", "Ambiguous"):
            state.known_characters.add(char)
            state.last_speaker = char
            state.last_emotions.setdefault(char, []).extend(fixed["emotions"])
            if memory is not None:
                try:
                    memory.push(char, fixed["emotions"])
                except Exception:
                    pass

        base_valid += 1
        result.append(fixed)

    try:
        print(
            f"[VALIDATE] base_valid={base_valid} base_rejected={base_rejected}", flush=True)
    except Exception:
        pass
    return result, warnings

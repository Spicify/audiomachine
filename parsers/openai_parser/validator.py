from __future__ import annotations
from typing import Any, Dict, List, Tuple
from pydantic import ValidationError
import re

from .core_types import DialogueLine, ParserState
from .emotion_utils import ensure_two_emotions, canonicalize_emotion
from utils.mode import get_emotions_mode
from .diag import diag_enabled, diag_print

_RX_QUOTE = re.compile(r'["“”‘’\']')
_RX_THIRD_PRON = re.compile(r'\b(he|she|they|his|her|their|him|them)\b', re.I)
_RX_END_PUNCT = re.compile(r'[.!?]\s*$')


def _no_quotes(s: str) -> bool:
    try:
        return not bool(_RX_QUOTE.search(s or ""))
    except Exception:
        return True


def _looks_like_narration(s: str, state) -> bool:
    """
    Heuristic: treat as narration when there are no quotes AND at least one of:
      - third-person pronouns present, OR
      - mentions a known character name (TitleCase) in text, OR
      - short declarative/fragments ending with punctuation ('.' or '!') but not a '?'
    Avoid flipping 1st-person lines (' I ') to Narrator.
    """
    try:
        t = (s or "").strip()
        if not t:
            return False
        # avoid first-person monologue being forced to Narrator
        if re.search(r'\bI\b', t):
            return False
        if _RX_THIRD_PRON.search(t):
            return True
        # known character mention
        try:
            for nm in getattr(state, "known_characters", []) or []:
                if nm and nm in t:
                    return True
        except Exception:
            pass
        # bare declaratives like "So perfect." or "It felt so good."
        if _RX_END_PUNCT.search(t) and "?" not in t:
            return True
        return False
    except Exception:
        return False


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
            try:
                print("[VALIDATE_DIAG] dropping=REJECTED sample=",
                      (it if isinstance(it, dict) else str(it)), flush=True)
            except Exception:
                pass
            if diag_enabled():
                try:
                    reason = "policy?"
                    diag_print(f"[REJECTED_FLAG] reason={reason}")
                except Exception:
                    pass
            result.append(it)
            base_rejected += 1
            continue

        char = (it.get("character") or "").strip() or "Ambiguous"
        txt = (it.get("text") or "").strip()
        ems = it.get("emotions") or []

        if not txt:
            try:
                print("[VALIDATE_DIAG] dropping=EMPTY_TEXT sample=",
                      (it if isinstance(it, dict) else str(it)), flush=True)
            except Exception:
                pass
            base_rejected += 1
            continue

        # Soft-allow new proper names output by the model (Titlecase, 1–3 words, no digits)
        def _looks_like_proper_name(s: str) -> bool:
            if not s or s in ("Narrator", "Ambiguous"):
                return False
            parts = [p for p in s.split() if p]
            if not (1 <= len(parts) <= 3):
                return False
            for p in parts:
                if any(ch.isdigit() for ch in p):
                    return False
                if not (p[0].isupper()):
                    return False
            return True

        if char not in allowed_chars and _looks_like_proper_name(char):
            try:
                state.known_characters.add(char)
                allowed_chars.add(char)
                print(
                    f"[VALIDATE] soft-allowed new character='{char}'", flush=True)
            except Exception:
                pass

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
                # Ambiguity diagnostics for quote-then-name patterns
                if diag_enabled():
                    try:
                        tail_tokens = (txt or "").split()[-4:]
                        diag_print(
                            f"[AMBIG_QUOTE_TAIL] tail={' '.join(tail_tokens)}")
                    except Exception:
                        pass

        # Coerce obvious narration: Ambiguous → Narrator (no quotes, looks like narration)
        if (fixed.get("character") == "Ambiguous") and _no_quotes(txt) and _looks_like_narration(txt, state):
            try:
                fixed["character"] = "Narrator"
                if "candidates" in fixed:
                    fixed.pop("candidates", None)
                if diag_enabled():
                    diag_print(
                        f"[AMBIG_NARR_COERCE] reason=narration_no_quotes text='{(txt[:60] + '…') if len(txt) > 60 else txt}'")
            except Exception:
                pass

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

        # Skip AI refusal or policy disclaimer lines
        refusal_text = str(it.get("text", "")).lower()
        if any(
            phrase in refusal_text
            for phrase in [
                "i'm sorry, i can’t assist",
                "i can’t assist",
                "as an ai language model",
                "as an ai model",
                "i cannot assist",
                "explicit sexual content",
                "involving familial",
                "violates policy",
                "i'm unable to help with that",
            ]
        ):
            if diag_enabled():
                diag_print(f"[VALIDATE_SKIP_REFUSAL] text='{refusal_text[:80]}...'")
            continue

        base_valid += 1
        result.append(fixed)

    try:
        print(
            f"[VALIDATE] base_valid={base_valid} base_rejected={base_rejected}", flush=True)
    except Exception:
        pass
    try:
        print(
            f"[VALIDATE_SUMMARY] base_valid={base_valid} base_rejected={base_rejected} allowed_chars={len(allowed_chars)}", flush=True)
    except Exception:
        pass
    return result, warnings

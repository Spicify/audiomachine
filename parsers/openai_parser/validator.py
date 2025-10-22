from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import ValidationError

from .core_types import DialogueLine, ParserState
from .emotion_utils import ensure_two_emotions


def validate_and_fix(items: List[Dict[str, Any]], warnings: List[str], state: ParserState, *, kb=None, allowed_emotions=None, memory=None) -> Tuple[List[Dict[str, Any]], List[str]]:
    result: List[Dict[str, Any]] = []
    base_valid = 0
    base_rejected = 0
    for it in items:
        try:
            if str(it.get("character", "")).upper() == "REJECTED":
                print("[DIAG] REJECTED line entering validator:",
                      (it.get("text", "") or "")[:60], flush=True)
                result.append(it)
                base_rejected += 1
                continue
        except Exception:
            pass
        char = (it.get("character") or "").strip()
        txt = (it.get("text") or "").strip()
        ems = it.get("emotions") or []

        allowed_chars = set(state.known_characters or []) | {
            "Narrator", "Ambiguous"}
        is_narr = (char.lower() == "narrator")

        if char not in allowed_chars:
            char = "Ambiguous"
            try:
                it["character"] = "Ambiguous"
            except Exception:
                pass

        if not txt:
            base_rejected += 1
            continue

        if not isinstance(ems, list) or len(ems) != 2:
            base_rejected += 1
            continue

        if not is_narr:
            has_quote = ('"' in txt) or (""" in txt and """ in txt)
            # Quote enforcement remains disabled as in original

        fixed = {"character": char, "text": txt, "emotions": ems}
        if str(char).lower() == "ambiguous":
            if it.get("candidates"):
                cands = [str(c).strip() for c in (
                    it.get("candidates") or []) if str(c).strip()]
                if cands:
                    fixed["candidates"] = cands[:5]
            fixed["id"] = f"amb-{abs(hash(txt))}"

        try:
            if (fixed.get("character") in ("Narrator", "Ambiguous")):
                print(
                    f"[ATTR] {fixed['character']} text='{fixed['text'][:80]}'",
                    flush=True,
                )
        except Exception:
            pass

        try:
            DialogueLine(**fixed)
        except ValidationError as ve:
            warnings.append(
                f"Validation dropped line: {fixed} ({ve.errors()[:1]})")
            continue

        if fixed["character"].lower() != "narrator" and fixed["character"].lower() != "ambiguous":
            state.known_characters.add(fixed["character"])
            state.last_speaker = fixed["character"]
            state.last_emotions.setdefault(
                fixed["character"], []).extend(fixed["emotions"])
            if memory is not None:
                try:
                    memory.push(fixed["character"], fixed["emotions"])
                except Exception:
                    pass

        result.append(fixed)
    try:
        print(
            f"[VALIDATE] base_valid={base_valid} base_rejected={base_rejected}", flush=True)
    except Exception:
        pass
    return result, warnings

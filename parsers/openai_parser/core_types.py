from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import hashlib
import json
import difflib
import io

from pydantic import BaseModel, Field


def _attr_counts(lines):
    try:
        total = len(lines or [])
        narrator = sum(1 for d in (lines or []) if str(
            d.get("character", "")).strip().lower() == "narrator")

        def src(d): return str(d.get("_src", "")).strip().lower()
        ai = sum(1 for d in (lines or []) if src(d) == "ai")
        fb = sum(1 for d in (lines or []) if src(d) == "fb")
        reinj = sum(1 for d in (lines or []) if src(d) == "reinj")
        rejected = sum(1 for d in (lines or []) if str(d.get("character", "")).upper(
        ) == "REJECTED" or str(d.get("_status", "")).upper() == "REJECTED")
        unknown = sum(1 for d in (lines or []) if (d.get("character") and d.get(
            "character") not in ("Narrator", "Ambiguous")) is False)
        return {"total": total, "narrator": narrator, "ai": ai, "fb": fb, "reinj": reinj, "rejected": rejected, "unknown": unknown}
    except Exception:
        return {"total": 0, "narrator": 0, "ai": 0, "fb": 0, "reinj": 0, "rejected": 0, "unknown": 0}


class DualLogger(io.TextIOBase):
    """Duplicates stdout writes to both terminal and an in-memory buffer."""

    def __init__(self, original_stream):
        super().__init__()
        self.original = original_stream
        self.buffer = io.StringIO()

    def write(self, data):
        try:
            self.original.write(data)
            self.original.flush()
        except Exception:
            pass
        try:
            self.buffer.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass
        try:
            self.buffer.flush()
        except Exception:
            pass

    def get_value(self):
        try:
            return self.buffer.getvalue()
        except Exception:
            return ""


@dataclass
class RawParseResult:
    formatted_text: str
    dialogues: List[Dict]
    stats: Dict[str, int]
    ambiguities: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DialogueLine(BaseModel):
    character: str
    emotions: List[str] = Field(min_length=2, max_length=2)
    text: str
    candidates: Optional[List[str]] = None


class ParserState(BaseModel):
    known_characters: Set[str] = Field(default_factory=set)
    last_speaker: Optional[str] = None
    last_emotions: Dict[str, List[str]] = Field(default_factory=dict)
    unresolved_ambiguities: List[Dict[str, Any]] = Field(default_factory=list)


def _hash_key(text: str, state: Dict[str, Any]) -> str:
    payload = json.dumps({"t": text, "s": state},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token_similarity(a: str, b: str) -> float:
    """Lightweight text similarity check between two strings (0–1 scale)."""
    try:
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    except Exception:
        return 0.0


def _fuzzy_sim(a: str, b: str) -> float:
    """SequenceMatcher-based similarity between two strings (0..1)."""
    try:
        a = (a or "").strip().lower()
        b = (b or "").strip().lower()
        return difflib.SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0

"""State adapter utilities for the CharacterDetector."""
from __future__ import annotations

from typing import Any, List, Optional


class _StateAdapter:
    """
    Lightweight adapter that wraps the user's ParserState-like object and
    provides safe defaults and small helpers without requiring a particular class.
    """

    def __init__(self, state: Optional[Any]):
        self._state = state
        # try to extract attributes defensively
        self.last_speaker = getattr(
            state, "last_speaker", None) if state else None
        self.recent_mentions = getattr(
            state, "recent_mentions", []) if state else []
        # ensure recent_mentions is a list of names (strings)
        if self.recent_mentions is None:
            self.recent_mentions = []
        self.pronoun_map = getattr(state, "pronoun_map", {}) if state else {}
        if self.pronoun_map is None:
            self.pronoun_map = {}
        # Full previous line and inferred subject (if exposed on state).
        self.last_line = getattr(state, "last_line", None) if state else None
        self.last_subject = getattr(
            state, "last_subject", None) if state else None
        self.pending_non_user_female = (
            getattr(state, "pending_non_user_female", 0) if state else 0
        )

    @property
    def state(self) -> Optional[Any]:
        """Return the underlying state object, if any."""
        return self._state

    # Helpers
    def recent_mentions_reversed(self) -> List[str]:
        # Return most recent first
        try:
            # assume list of names; make lowercase-normalized list of original casing
            rev = list(reversed(self.recent_mentions))
            return [n for n in rev if n]
        except Exception:
            return []

    # When we update the detected speaker, also push it into state's fields if available
    def update_last_speaker(self, name: str):
        # Update adapter local
        self.last_speaker = name
        # Try to update underlying state object if present
        if self._state is None:
            return
        try:
            setattr(self._state, "last_speaker", name)
            # prepend to recent_mentions if available
            if hasattr(self._state, "recent_mentions"):
                rm = getattr(self._state, "recent_mentions") or []
                if not isinstance(rm, list):
                    rm = list(rm)
                rm.append(name)
                setattr(self._state, "recent_mentions", rm)
            # optionally update pronoun_map for gender -> pronoun mapping handled by caller
        except Exception:
            # best-effort; ignore failures
            pass

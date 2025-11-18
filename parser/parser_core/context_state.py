from collections import deque
from typing import Any, Deque, Dict, List, Optional


class ParserState:
    """
    Deterministic rolling state container for the offline local parser.

    This object is intentionally lightweight and stateless across chapters:
    a new instance should be created for each chapter or story segment.

    Responsibilities
    ----------------
    * Track the last N (default 50) parsed lines in `recent_lines`. Each entry is
      a small dict with:
          {
              "text": str,
              "speaker": str | None,
              "type": "dialogue" | "narration" | "thought",
              "emotions": list[str],
          }
    * Track speaker memory:
        - `last_speaker`: most recent non-ambiguous speaker (or None).
        - `recent_speakers`: a fixed-size window (default 5) of the most recent
          speakers, stored in order of occurrence.
        - `recent_mentions`: a rolling list of up to 50 speaker names,
          representing the most recently seen speakers over the window of the
          story processed by this state object.

    Design constraints
    ------------------
    * Pure Python standard library only; deterministic behaviour.
    * No cross-talk with paragraphs, emotion trends, genders, or pronouns.
      Any higher-level reasoning remains in other components (e.g.
      CharacterDetector).
    """

    def __init__(
        self,
        max_recent_lines: int = 50,
        max_recent_speakers: int = 5,
        max_recent_mentions: int = 50,
    ) -> None:
        """
        Initialize a new ParserState instance.

        Args:
            max_recent_lines:
                Maximum number of line entries to retain in `recent_lines`.
            max_recent_speakers:
                Maximum number of speakers to retain in `recent_speakers`.
            max_recent_mentions:
                Maximum number of speaker names to keep in `recent_mentions`.
        """
        self._max_recent_lines = int(max_recent_lines)
        self._max_recent_mentions = int(max_recent_mentions)

        # Public rolling fields
        self.recent_lines: List[Dict[str, Any]] = []
        self.last_speaker: Optional[str] = None
        # Full dict for the most recently processed raw line from the splitter.
        # This is updated by the pipeline to support narration-based subject
        # extraction in the character detector.
        self.last_line: Optional[Dict[str, Any]] = None
        # Canonical user character inferred as the current speaking subject
        # from narration. Dialogue detection can consult this field before
        # applying other heuristics.
        self.last_subject: Optional[str] = None

        # Internal deques/lists for speaker memory
        self._recent_speakers: Deque[str] = deque(maxlen=int(max_recent_speakers))
        self._recent_mentions: List[str] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update(self, line_dict: Dict[str, Any], speaker: Optional[str], emotions: List[str]) -> None:
        """
        Update rolling state with a new parsed line.

        Args:
            line_dict:
                A dictionary coming directly from the sentence splitter, with at
                least:
                    {"text": str, "type": "dialogue" | "narration" | "thought"}
            speaker:
                Speaker name string from the character detector, or None.
                The special value "Ambiguous" is treated as non-deterministic
                and is NOT written into speaker memory.
            emotions:
                A list of emotion labels (typically two) from the emotion
                tagger. This list is copied to avoid external mutation.

        Behaviour:
            * Appends a new entry to `recent_lines` containing the combined
              text/speaker/type/emotions data.
            * Maintains `recent_lines` as a rolling window capped at
              `max_recent_lines` entries.
            * If `speaker` is non-empty, not None, and not "Ambiguous":
                - Updates `last_speaker`.
                - Appends to `_recent_speakers` (deque, maxlen 5).
                - Appends to `_recent_mentions`, then truncates that list to
                  at most `max_recent_mentions` entries by discarding the
                  oldest items.
        """
        text = line_dict.get("text", "")
        line_type = line_dict.get("type", "narration")

        entry = {
            "text": text,
            "speaker": speaker,
            "type": line_type,
            "emotions": list(emotions) if emotions is not None else [],
        }
        self.recent_lines.append(entry)

        # Rolling window on recent_lines
        if len(self.recent_lines) > self._max_recent_lines:
            # Remove oldest entries while we exceed capacity
            overflow = len(self.recent_lines) - self._max_recent_lines
            if overflow == 1:
                self.recent_lines.pop(0)
            else:
                # Slice off the oldest overflow entries
                self.recent_lines = self.recent_lines[overflow:]

        # Speaker memory update (skip ambiguous/unknown)
        if speaker and speaker != "Ambiguous":
            self.last_speaker = speaker
            self._recent_speakers.append(speaker)

            self._recent_mentions.append(speaker)
            # Truncate mentions to most recent N entries
            if len(self._recent_mentions) > self._max_recent_mentions:
                self._recent_mentions = self._recent_mentions[
                    -self._max_recent_mentions :
                ]

    def get_last_speaker(self) -> Optional[str]:
        """
        Return the most recently detected non-ambiguous speaker.

        The value is a simple scalar (string or None), so callers cannot mutate
        internal data through this reference.
        """
        return self.last_speaker

    def get_recent_speakers(self) -> List[str]:
        """
        Return a copy of the current window of recent speakers.

        The returned list preserves order from oldest to newest within the
        deque, but modifications to the returned list will not affect internal
        ParserState data.
        """
        return list(self._recent_speakers)

    def get_recent_mentions(self) -> List[str]:
        """
        Return a copy of the rolling list of recent speaker mentions.

        At most the last `max_recent_mentions` names are retained. The copy
        guarantees that external code cannot mutate the internal list.
        """
        return list(self._recent_mentions)

    def get_recent_lines(self) -> List[Dict[str, Any]]:
        """
        Return a shallow copy of the recent line window.

        Each entry is a dictionary describing a line. The outer list is copied
        so that callers cannot mutate the internal list structure. Individual
        dicts are not deep-copied for performance reasons; callers should treat
        them as read-only snapshots.
        """
        return list(self.recent_lines)




import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
from .character_detector_narration import NarrationMixin
from .character_detector_resolution import ResolutionMixin
from .character_detector_state import _StateAdapter
from .character_detector_user import UserCharacterMixin



class CharacterDetector(UserCharacterMixin, NarrationMixin, ResolutionMixin):
    """
    Detects the speaking character for a given line of text, using:
      - a local character voices config (configs/character_voices.json or configs/characters.json),
      - simple regex name matching,
      - pronoun resolution using parser state,
      - recent-mentions context.

    Usage:
        cd = CharacterDetector()
        result = cd.detect(line_text, state)
        # result -> {'character': 'Mikhail'} or {'character': 'Ambiguous', 'candidates': ['Mikhail', 'Lina']}
    """

    DEFAULT_CONFIG_CANDIDATES = [
        "parser/configs/character_voices.json",
        "configs/character_voices.json",
        "configs/characters.json",
        # harmless fallback (not parsed) - ignored if not JSON
        "configs/character_voices.yaml",
    ]

    SPEECH_VERB_FORMS = [
        "say",
        "says",
        "said",
        "ask",
        "asks",
        "asked",
        "reply",
        "replies",
        "replied",
        "whisper",
        "whispers",
        "whispered",
        "shout",
        "shouts",
        "shouted",
        "yell",
        "yells",
        "yelled",
        "mutter",
        "mutters",
        "muttered",
        "sigh",
        "sighs",
        "sighed",
        "cry",
        "cries",
        "cried",
        "exclaim",
        "exclaims",
        "exclaimed",
        "whimper",
        "whimpers",
        "whimpered",
        "murmur",
        "murmurs",
        "murmured",
        "growl",
        "growls",
        "growled",
        "laugh",
        "laughs",
        "laughed",
        "sob",
        "sobs",
        "sobbed",
        "note",
        "notes",
        "noted",
        "add",
        "adds",
        "added",
        "remark",
        "remarks",
        "remarked",
        "curse",
        "curses",
        "cursed",
        "command",
        "commands",
        "commanded",
        "insist",
        "insists",
        "insisted",
        "warn",
        "warns",
        "warned",
        "tease",
        "teases",
        "teased",
        "coo",
        "coos",
        "cooed",
        "breathe",
        "breathes",
        "breathed",
        "hiss",
        "hisses",
        "hissed",
        "snap",
        "snaps",
        "snapped",
        "state",
        "states",
        "stated",
        "respond",
        "responds",
        "responded",
        "answer",
        "answers",
        "answered",
        "call",
        "calls",
        "called",
        "promise",
        "promises",
        "promised",
        "plead",
        "pleads",
        "pleaded",
        "beg",
        "begs",
        "begged",
    ]
    SPEECH_VERBS = r"(?:%s)" % "|".join(SPEECH_VERB_FORMS)

    # Sentinel used when narration clearly refers to a non-user female subject
    # (e.g. "she" / "her") but no female user characters are configured.
    NON_USER_FEMALE_SUBJECT = "__NON_USER_FEMALE__"
    NON_USER_FEMALE_DIALOGUE_TTL = 3
    FEMALE_PRONOUNS = ("she", "her", "hers", "herself", "girl", "woman", "wife")
    FEMALE_NOUN_CUES = (
        "kitten",
        "princess",
        "queen",
        "mistress",
        "lover",
        "pet",
        "baby girl",
        "little one",
        "brat",
    )

    def __init__(self, config_path: Optional[str] = None):
        """
        Load known characters from a JSON config (flattened).
        If config_path is None, tries the default candidate paths.
        """
        if config_path:
            config_paths = [config_path]
        else:
            config_paths = self.DEFAULT_CONFIG_CANDIDATES

        # name_lower -> metadata dict (including 'gender' if present)
        self.characters = {}
        self.name_pattern = None
        self.gender_map = {}  # name_lower -> 'M'|'F'|'U'

        # Optional per-run preference set for story-specific characters.
        # Values are normalized via _normalize_name().
        self.preferred_characters: set[str] = set()

        # Optional user-driven character set for strict detection mode.
        # user_character_map: normalized_user_name -> canonical name for this run
        # user_character_list: canonical names in user-supplied order
        # _user_name_order: list of (original_user_name, canonical_name) tuples
        # strict_user_mode: when True, detection will only use this user set.
        self.user_character_map: Dict[str, str] = {}
        self.user_character_list: List[str] = []
        self._user_name_order: List[Tuple[str, str]] = []
        self.strict_user_mode: bool = False
        # When aggressive mode is enabled together with strict_user_mode,
        # the detector uses extended name-variant matching and deterministic
        # fallbacks to attribute speakers using only the user set.
        self.user_aggressive_mode: bool = True
        # When True (and strict/aggressive are enabled), ambiguous dialogue
        # lines may be auto-assigned to user characters using simple
        # context-based heuristics. When False, such lines remain Ambiguous.
        self.user_auto_assign_dialogue: bool = True
        # canonical_name -> list of compiled regex patterns for matching variants
        self._user_name_variant_map: Dict[str, List[re.Pattern]] = {}
        # normalized_name -> preferred display casing
        self._display_name_map: Dict[str, str] = {}
        self._load_characters(config_paths)




    def enable_strict_user_mode(self, enabled: bool = True) -> None:
        """
        Enable or disable strict user-driven detection mode.

        When strict_user_mode is True, speaker attribution is limited to
        the user-supplied character set configured via set_user_characters.
        Existing heuristics and config-driven candidates are bypassed.
        """
        self.strict_user_mode = bool(enabled)


    # ------------------ Public API ------------------ #

    def detect(
        self,
        line: Any,
        state: Optional[Any] = None,
        next_line: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Determine speaker for `line`.

        Args:
          line: either a string or a dict-like with 'text' and optional 'type'
                (dialogue/narration/thought). When 'line' is a dict originating
                from the SentenceSplitter, it may also include an optional
                'attribution' field containing a short speech-tag fragment.
          state: an object (ParserState) expected to have attributes:
                - last_speaker (str) or None
                - recent_mentions (List[str]) optional
                - pronoun_map (Dict[str, str]) optional
                - last_gender (str) optional
                - update(...) method is NOT required (we update attributes directly if present)
          next_line: optional dict representing the immediate following line
                from the splitter. When provided, short narration that looks
                like a speech attribution can be consulted as additional
                context for assigning a speaker.

        Returns:
          dict with at least 'character' key. If ambiguous, returns {'character': 'Ambiguous', 'candidates': [...]}.
        """
        text, kind = self._unpack_line(line)

        # Defensive state wrapper used throughout detection.
        s = _StateAdapter(state)

        # Narration: aggressively attribute to the most likely subject instead
        # of returning "Narrator" by default.
        if kind == "narration":
            candidate = self._detect_narration_character(text, s)
            if candidate:
                self._update_state_with_speaker(s, candidate)
                return self._finalize_character(candidate)
            return {"character": "Narrator"}

        # Attribution-aware shortcut for dialogue lines: consult attached
        # attribution text or immediate following narration before other
        # heuristics. In strict/user mode we only accept attributions that
        # resolve to one of the configured user characters (if any).
        if isinstance(line, dict) and kind == "dialogue":
            candidate = self._resolve_from_attribution(line, state, next_line)
            if candidate:
                if self.user_character_list:
                    norm_cand = self._normalize_name(candidate)
                    if not any(
                        self._normalize_name(c) == norm_cand
                        for c in self.user_character_list
                    ):
                        candidate = None
                if candidate:
                    self._update_state_with_speaker(s, candidate)
                    return self._finalize_character(candidate)

            # Rule: narration_subject is authoritative if set
            subject = getattr(s, "last_subject", None)
            if kind == "dialogue" and subject:
                if subject == self.NON_USER_FEMALE_SUBJECT:
                    self._set_pending_non_user_female(
                        s, max(self.NON_USER_FEMALE_DIALOGUE_TTL - 1, 0)
                    )
                    return {"character": "Ambiguous"}
                if self.strict_user_mode and self.user_character_list:
                    norm_subj = self._normalize_name(subject)
                    if any(
                        self._normalize_name(c) == norm_subj
                        for c in self.user_character_list
                    ):
                        self._update_state_with_speaker(s, subject)
                        return self._finalize_character(subject)
                else:
                    self._update_state_with_speaker(s, subject)
                    return self._finalize_character(subject)

        # Strict user-driven mode short-circuit:
        # when enabled, we either use aggressive matching or the simpler
        # strict routine, depending on configuration.
        if self.strict_user_mode and self._user_name_order:
            if self.user_aggressive_mode:
                return self._detect_user_aggressive(
                    text, kind, state, next_line, line if isinstance(line, dict) else None
                )
            return self._detect_strict_user(
                text, kind, state, next_line, line if isinstance(line, dict) else None
            )

        # 1) Direct mention (name appears in text)
        direct_name = self._find_direct_name(text)
        if direct_name:
            self._update_state_with_speaker(s, direct_name)
            return self._finalize_character(direct_name)

        # 1b) Speech-verb near a name pattern: try to capture "said <Name>" or "<Name> said"
        sv_name = self._search_speech_verb_patterns(text)
        if sv_name:
            self._update_state_with_speaker(s, sv_name)
            return self._finalize_character(sv_name)

        # 2) Pronoun resolution (I/he/she/they)
        pronoun_resolution = self._resolve_pronoun(text, s)
        if pronoun_resolution:
            self._update_state_with_speaker(s, pronoun_resolution)
            return self._finalize_character(pronoun_resolution)

        # 3) Recent mentions / context: look into state's recent_mentions
        recent = self._resolve_from_recent_mentions(s)
        if recent:
            self._update_state_with_speaker(s, recent)
            return self._finalize_character(recent)

        # 4) Alternation heuristic - optional: if last_speaker exists, assume alternating conversation
        alt = self._alternation_heuristic(s)
        if alt:
            self._update_state_with_speaker(s, alt)
            return self._finalize_character(alt)

        # 5) Fallback: ambiguous -> provide candidate list (recent_mentions, then known characters)
        candidates = self._candidate_list(s, text)

        # Preferred-character short-circuit:
        # If any candidate matches a preferred name, choose it deterministically
        # (first matching candidate) instead of returning 'Ambiguous'.
        preferred_choice = self._select_preferred_from_candidates(candidates)
        if preferred_choice:
            self._update_state_with_speaker(s, preferred_choice)
            return self._finalize_character(preferred_choice)

        return {"character": "Ambiguous", "candidates": candidates}

    def _detect_narration_character(
        self, text: str, state_adapter: _StateAdapter
    ) -> Optional[str]:
        subject = self._update_subject_from_narration(text, state_adapter)
        subject_candidate = (
            subject if subject and subject != self.NON_USER_FEMALE_SUBJECT else None
        )

        direct = self._find_direct_name(text)
        if direct:
            return direct

        speech = self._search_speech_verb_patterns(text)
        if speech:
            return speech

        pronoun = self._resolve_pronoun(text, state_adapter)
        if pronoun:
            return pronoun

        if subject_candidate:
            return subject_candidate

        cached_subject = getattr(state_adapter, "last_subject", None)
        if cached_subject and cached_subject != self.NON_USER_FEMALE_SUBJECT:
            return cached_subject

        last_speaker = getattr(state_adapter, "last_speaker", None)
        if last_speaker and last_speaker not in ("Ambiguous", "Narrator"):
            return last_speaker

        recent = self._resolve_from_recent_mentions(state_adapter)
        if recent and recent not in ("Ambiguous", "Narrator"):
            return recent

        alt = self._alternation_heuristic(state_adapter)
        if alt and alt not in ("Ambiguous", "Narrator"):
            return alt
        return None

    def _finalize_character(self, name: str) -> Dict[str, Any]:
        return {"character": self._display_name_for(name)}

    def _display_name_for(self, name: str) -> str:
        if not name:
            return name
        norm = self._normalize_name(name)
        display = self._display_name_map.get(norm)
        if display:
            return display
        formatted = self._format_display_name(name)
        self._display_name_map[norm] = formatted
        return formatted

    def _format_display_name(self, name: str) -> str:
        if not name:
            return name
        parts = re.split(r"(\s+|-)", str(name))
        formatted_parts: List[str] = []
        for part in parts:
            if part is None:
                continue
            if not part.strip():
                formatted_parts.append(part)
                continue
            formatted_parts.append(part[0].upper() + part[1:])
        return "".join(formatted_parts)

    # ------------------ Character loading ------------------ #

    def _load_characters(self, config_paths: List[str]):
        """Attempt to load JSON character configs from candidate paths. Flatten nested groups."""
        loaded = False
        for p in config_paths:
            pth = Path(p)
            if not pth.exists():
                continue
            try:
                raw = json.loads(pth.read_text(encoding="utf-8"))
                self._flatten_character_json(raw)
                loaded = True
                break
            except Exception:
                # not a JSON or unreadable - continue trying others
                continue

        if not loaded:
            # No config found; keep characters empty and warn
            # (Do not crash; parser proceeds and will return Ambiguous candidates later)
            print(
                "[character_detector] Warning: no character config found in", config_paths)

        # Build name matching pattern if we have names
        if self.characters:
            # Sort longer names first to prefer multi-word names in regex
            names = sorted(self.characters.keys(), key=lambda s: -len(s))
            # escape for regex and ensure word boundaries
            pattern = r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b"
            self.name_pattern = re.compile(pattern, flags=re.IGNORECASE)
            # gender map
            for n, meta in self.characters.items():
                g = meta.get("gender") if isinstance(meta, dict) else None
                gnorm = (g or "U").upper() if g else "U"
                self.gender_map[n] = gnorm

    def _flatten_character_json(self, data):
        """
        Recursively flatten nested JSON config for character definitions.
        Handles emoji-prefixed or symbol-prefixed group names safely.
        """
        import re
        for key, value in data.items():
            # Normalize the key to handle emojis and special symbols
            clean_key = re.sub(r"[^\w\s\(\)-]+", "", key).strip()
            if isinstance(value, dict):
                # Check if this dict contains character definitions (has voice_id in subdicts)
                has_character_defs = any(
                    isinstance(v, dict) and "voice_id" in v
                    for v in value.values()
                )
                if not has_character_defs:
                    # This is a group, recurse into it
                    self._flatten_character_json(value)
                else:
                    # This dict contains character definitions - extract them
                    for char_name, char_meta in value.items():
                        if isinstance(char_meta, dict) and ("voice_id" in char_meta or "gender" in char_meta):
                            # Store character with normalized name
                            norm_name = self._normalize_name(char_name)
                            self.characters[norm_name] = char_meta
                            if norm_name not in self._display_name_map:
                                self._display_name_map[norm_name] = self._format_display_name(
                                    char_name)
                            # Also store with original casing for lookup
                            self.characters[char_name.lower()] = char_meta

        # After flattening, collapse duplicates caused by normalization
        deduped = {}
        deduped_display = {}
        for name, meta in self.characters.items():
            norm = self._normalize_name(name)
            display_val = self._display_name_map.get(
                norm) or self._format_display_name(name)
            if norm not in deduped:
                deduped[norm] = meta
                deduped_display[norm] = display_val
            else:
                # Prefer the entry that has a gender or voice_id defined
                if not isinstance(deduped[norm], dict):
                    deduped[norm] = meta
                    deduped_display[norm] = display_val
                else:
                    old_meta = deduped[norm]
                    if isinstance(meta, dict) and (
                        ("voice_id" in meta and "voice_id" not in old_meta)
                        or ("gender" in meta and "gender" not in old_meta)
                    ):
                        deduped[norm] = meta
                        deduped_display[norm] = display_val
        self.characters = deduped
        self._display_name_map = deduped_display

    def _normalize_name(self, name: str) -> str:
        """
        Normalize character names for consistent matching:
        - lowercase
        - remove punctuation and parentheses
        - collapse multiple spaces
        """
        import re
        name = name.lower()
        name = re.sub(r"[\(\)\[\]\.,'\"!?\-]+", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name














    # ------------------ Heuristic helpers ------------------ #






















# ------------------------- small state adapter ------------------------- #




# ------------------------- end of file ------------------------- #

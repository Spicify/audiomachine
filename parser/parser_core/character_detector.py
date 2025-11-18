import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any


class CharacterDetector:
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

    SPEECH_VERBS = r"(?:said|asked|replied|whispered|shouted|yelled|muttered|sighed|cried|exclaimed|asked|whimpered|murmured|growled|laughed|sobbed|noted|added|remarked|cursed)"

    # Sentinel used when narration clearly refers to a non-user female subject
    # (e.g. "she" / "her") but no female user characters are configured.
    NON_USER_FEMALE_SUBJECT = "__NON_USER_FEMALE__"

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
        self._load_characters(config_paths)

    def inject_user_characters(self, user_characters: list[str]):
        """
        Allow user to provide additional or overriding character names at runtime.
        User-provided names take precedence over config-defined ones.
        """
        if not user_characters:
            return
        for name in user_characters:
            norm = self._normalize_name(name)
            # If user defines a name that already exists, override it
            self.characters[norm] = {"gender": None, "source": "user"}

    def set_preferred_characters(self, user_characters: Iterable[str]) -> None:
        """
        Configure a per-run set of "preferred" character names.

        Names are normalized using the same rules as internal matching
        (via _normalize_name). When the detector builds an ambiguous
        candidate list, any preferred name present in that list will be
        selected deterministically instead of returning 'Ambiguous'.

        If no preferred characters are provided, or none appear in the
        candidate list, the behaviour falls back to the existing heuristics.
        """
        prefs: set[str] = set()
        if user_characters:
            for name in user_characters:
                if not name:
                    continue
                prefs.add(self._normalize_name(str(name)))
        self.preferred_characters = prefs

    def set_user_characters(self, user_entries: Iterable[Tuple[Any, ...]]) -> None:
        """
        Configure a per-run, user-driven character set.

        Args:
            user_entries:
                Iterable of (user_name, exists_in_config) tuples.
                - If exists_in_config is True, the user_name will be mapped
                  to the best canonical name from self.characters using
                  normalized token matching.
                - If exists_in_config is False, the literal user_name is
                  treated as the canonical name for this run.

        This method populates:
          * user_character_map: normalized user name -> canonical name
          * user_character_list: canonical names in user order
          * _user_name_order: (original_user_name, canonical_name) in order

        Any previously configured preferred_characters are cleared, since
        strict user-mode uses this mapping instead.
        """
        self.user_character_map = {}
        self.user_character_list = []
        self._user_name_order = []

        any_entries = False
        for entry in user_entries or []:
            # Backwards-compatible handling: accept (name, exists) or
            # (name, exists, gender) tuples.
            if not entry:
                continue
            if len(entry) == 2:
                user_name, exists_in_config = entry
                gender_hint = None
            else:
                user_name, exists_in_config, gender_hint = entry[0], entry[1], entry[2]
            if not user_name:
                continue
            any_entries = True
            user_name_str = str(user_name)
            if exists_in_config:
                canonical = self._map_user_name_to_canonical(user_name_str)
                # Fallback: if we cannot map to a known config name, keep the literal
                if not canonical:
                    canonical = user_name_str
            else:
                canonical = user_name_str

            norm_user = self._normalize_name(user_name_str)
            self.user_character_map[norm_user] = canonical
            self.user_character_list.append(canonical)
            self._user_name_order.append((user_name_str, canonical))

            # If a gender hint was provided, attach/override it in the
            # characters metadata and update the gender map accordingly.
            if gender_hint:
                g = str(gender_hint).strip().upper()
                if g.startswith("M"):
                    g_norm = "M"
                elif g.startswith("F"):
                    g_norm = "F"
                else:
                    g_norm = "U"
                # Normalized canonical key used internally.
                canon_key = self._normalize_name(canonical)
                meta = self.characters.get(canon_key, {}) or {}
                if not isinstance(meta, dict):
                    meta = {}
                meta["gender"] = g_norm
                self.characters[canon_key] = meta
                self.gender_map[canon_key] = g_norm

        # Preferred-character API is independent; clear it to avoid confusion.
        self.preferred_characters = set()
        # Rebuild name variants for aggressive matching.
        if any_entries:
            self._build_user_name_variants()

    def enable_strict_user_mode(self, enabled: bool = True) -> None:
        """
        Enable or disable strict user-driven detection mode.

        When strict_user_mode is True, speaker attribution is limited to
        the user-supplied character set configured via set_user_characters.
        Existing heuristics and config-driven candidates are bypassed.
        """
        self.strict_user_mode = bool(enabled)

    def set_preferred_characters(self, names: Iterable[str]) -> None:
        """
        Configure a per-run set of "preferred" character names.

        Names are normalized using the same rules as internal matching
        (via _normalize_name). When the detector builds an ambiguous
        candidate list, any preferred name present in that list will be
        selected deterministically instead of returning 'Ambiguous'.

        If no preferred characters are provided, or none appear in the
        candidate list, the behaviour falls back to the existing heuristics.
        """
        prefs: set[str] = set()
        if names:
            for name in names:
                if not name:
                    continue
                prefs.add(self._normalize_name(str(name)))
        self.preferred_characters = prefs

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

        # Narration: infer subject (if possible) and always return Narrator.
        if kind == "narration":
            self._update_subject_from_narration(text, s)
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
                    return {"character": candidate}

            # Rule: narration_subject is authoritative if set
            subject = getattr(s, "last_subject", None)
            if kind == "dialogue" and subject:
                if self.strict_user_mode and self.user_character_list:
                    norm_subj = self._normalize_name(subject)
                    if any(
                        self._normalize_name(c) == norm_subj
                        for c in self.user_character_list
                    ):
                        self._update_state_with_speaker(s, subject)
                        return {"character": subject}
                else:
                    self._update_state_with_speaker(s, subject)
                    return {"character": subject}

        # Strict user-driven mode short-circuit:
        # when enabled, we either use aggressive matching or the simpler
        # strict routine, depending on configuration.
        if self.strict_user_mode and self._user_name_order:
            if self.user_aggressive_mode:
                return self._detect_user_aggressive(text, kind, state)
            return self._detect_strict_user(text, kind, state)

        # 1) Direct mention (name appears in text)
        direct_name = self._find_direct_name(text)
        if direct_name:
            self._update_state_with_speaker(s, direct_name)
            return {"character": direct_name}

        # 1b) Speech-verb near a name pattern: try to capture "said <Name>" or "<Name> said"
        sv_name = self._search_speech_verb_patterns(text)
        if sv_name:
            self._update_state_with_speaker(s, sv_name)
            return {"character": sv_name}

        # 2) Pronoun resolution (I/he/she/they)
        pronoun_resolution = self._resolve_pronoun(text, s)
        if pronoun_resolution:
            self._update_state_with_speaker(s, pronoun_resolution)
            return {"character": pronoun_resolution}

        # 3) Recent mentions / context: look into state's recent_mentions
        recent = self._resolve_from_recent_mentions(s)
        if recent:
            self._update_state_with_speaker(s, recent)
            return {"character": recent}

        # 4) Alternation heuristic - optional: if last_speaker exists, assume alternating conversation
        alt = self._alternation_heuristic(s)
        if alt:
            self._update_state_with_speaker(s, alt)
            return {"character": alt}

        # 5) Fallback: ambiguous -> provide candidate list (recent_mentions, then known characters)
        candidates = self._candidate_list(s, text)

        # Preferred-character short-circuit:
        # If any candidate matches a preferred name, choose it deterministically
        # (first matching candidate) instead of returning 'Ambiguous'.
        preferred_choice = self._select_preferred_from_candidates(candidates)
        if preferred_choice:
            self._update_state_with_speaker(s, preferred_choice)
            return {"character": preferred_choice}

        return {"character": "Ambiguous", "candidates": candidates}

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
                            # Also store with original casing for lookup
                            self.characters[char_name.lower()] = char_meta

        # After flattening, collapse duplicates caused by normalization
        deduped = {}
        for name, meta in self.characters.items():
            norm = self._normalize_name(name)
            if norm not in deduped:
                deduped[norm] = meta
            else:
                # Prefer the entry that has a gender or voice_id defined
                if not isinstance(deduped[norm], dict):
                    deduped[norm] = meta
                else:
                    old_meta = deduped[norm]
                    if isinstance(meta, dict) and (
                        ("voice_id" in meta and "voice_id" not in old_meta)
                        or ("gender" in meta and "gender" not in old_meta)
                    ):
                        deduped[norm] = meta
        self.characters = deduped

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

    def _map_user_name_to_canonical(self, user_name: str) -> Optional[str]:
        """
        Map a user-supplied name to the best canonical config name.

        Matching is deterministic and uses the following priority:
          1. Exact normalized equality between user and canonical name.
          2. Canonical token that exactly equals the normalized user token.
          3. Canonical token that startswith the normalized user token.
          4. Canonical name whose normalized form contains the user token.

        Among matches in the same category, prefer:
          * the longest canonical name (by normalized length), and then
          * alphabetical order as a final tiebreaker.
        """
        if not self.characters:
            return None

        user_norm = self._normalize_name(user_name)
        if not user_norm:
            return None

        matches: List[Tuple[int, str, str]] = []

        for canon_key in self.characters.keys():
            canon_norm = self._normalize_name(canon_key)
            if not canon_norm:
                continue

            tokens = canon_norm.split()
            category: Optional[int] = None

            # 1) Exact normalized equality
            if canon_norm == user_norm:
                category = 0
            # 2) Canonical token equals user token
            elif any(t == user_norm for t in tokens):
                category = 1
            # 3) Canonical token startswith user token
            elif any(t.startswith(user_norm) for t in tokens):
                category = 2
            # 4) Canonical name contains user token somewhere
            elif user_norm in canon_norm:
                category = 3

            if category is not None:
                matches.append((category, canon_norm, canon_key))

        if not matches:
            return None

        # Choose the lowest category (highest priority).
        best_category = min(m[0] for m in matches)
        best_matches = [m for m in matches if m[0] == best_category]

        # Within the same category, prefer longest canonical name, then alphabetical.
        best_matches.sort(key=lambda m: (-len(m[1]), m[1]))
        return best_matches[0][2]

    def _select_preferred_from_candidates(self, candidates: List[str]) -> Optional[str]:
        """
        Given a candidate list, return the first entry that matches the
        normalized preferred-character set, if any.

        This keeps tie-breaking deterministic by using candidate order,
        which already reflects upstream heuristics.
        """
        if not getattr(self, "preferred_characters", None):
            return None
        prefs = self.preferred_characters
        for cand in candidates:
            if self._normalize_name(cand) in prefs:
                return cand
        return None

    def _build_user_name_variants(self) -> None:
        """
        Build a deterministic set of name variants for each canonical user name.

        For each (original_user_name, canonical_name) pair, we generate variants
        in the following order:
          1. Full original user name (as given).
          2. Normalized full name (lowercase, punctuation stripped).
          3. First token (if any).
          4. Last token (if different from first).
          5. Initials (concatenated, then dotted form).

        If the canonical name differs from the user name, we append the same
        variant set derived from the canonical name after the user variants.

        Variants are stored as compiled regex patterns in
        `self._user_name_variant_map[canonical_name]` preserving this order.
        """
        variant_map: Dict[str, List[re.Pattern]] = {}

        def _variants_for(name: str) -> List[str]:
            variants: List[str] = []
            if not name:
                return variants
            norm = self._normalize_name(name)
            tokens = norm.split() if norm else []

            # 1) Full original name
            variants.append(name)
            # 2) Normalized full name
            if norm and norm.lower() != name.lower():
                variants.append(norm)
            # 3) First token
            if tokens:
                first = tokens[0]
                variants.append(first)
                # 4) Last token (if different)
                if len(tokens) > 1:
                    last = tokens[-1]
                    if last != first:
                        variants.append(last)
                # 5) Initials: concatenated and dotted
                initials = "".join(t[0] for t in tokens if t)
                if initials:
                    variants.append(initials)
                    dotted = ".".join(initials) + "."
                    variants.append(dotted)
            return variants

        for user_name, canonical in self._user_name_order:
            variants: List[str] = []
            variants.extend(_variants_for(user_name))
            if canonical != user_name:
                variants.extend(_variants_for(canonical))

            # De-duplicate while preserving order (case-insensitive)
            seen: set[str] = set()
            unique: List[str] = []
            for v in variants:
                v = v.strip()
                key = v.lower()
                if not v or key in seen:
                    continue
                seen.add(key)
                unique.append(v)

            patterns: List[re.Pattern] = []
            for v in unique:
                escaped = re.escape(v)
                # If the variant has alphanumeric characters, use word boundaries.
                if any(ch.isalnum() for ch in v):
                    pattern_text = r"\b" + escaped + r"\b"
                else:
                    pattern_text = escaped
                try:
                    patterns.append(re.compile(
                        pattern_text, flags=re.IGNORECASE))
                except re.error:
                    # Skip any pattern that cannot be compiled safely.
                    continue

            variant_map[canonical] = patterns

        self._user_name_variant_map = variant_map

    def _gender_for_canonical(self, canonical: str) -> Optional[str]:
        """
        Best-effort lookup of gender ('M' or 'F') for a canonical character
        name using the loaded character metadata.
        """
        canonical_norm = self._normalize_name(canonical)
        if not canonical_norm:
            return None
        for name, meta in self.characters.items():
            if not isinstance(meta, dict):
                continue
            if self._normalize_name(name) == canonical_norm:
                g = meta.get("gender")
                if not g:
                    return None
                g = str(g).upper()
                if g.startswith("M"):
                    return "M"
                if g.startswith("F"):
                    return "F"
                return None
        return None

    def _looks_like_attribution_text(self, text: str) -> bool:
        """
        Heuristic check for short fragments that likely serve as attributions.
        """
        if not text:
            return False
        # Simple verb and possessive checks mirroring the splitter heuristics.
        if re.search(
            r"^\s*(?:\w+[, ]+)?"
            r"(?:said|asked|whispered|replied|murmured|sighed|laughed|sobbed|shouted|"
            r"cried|whined|breathed|cursed)\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(r"\b's (?:voice|whisper)\b", text, flags=re.IGNORECASE):
            return True
        if re.search(self.SPEECH_VERBS, text, flags=re.IGNORECASE):
            return True
        return False

    def _resolve_speaker_from_text(self, text: str, state_adapter: "_StateAdapter") -> Optional[str]:
        """
        Try to resolve a speaker from a small attribution-like text fragment
        using direct-name, speech-verb, and pronoun heuristics.
        """
        if not text:
            return None
        # If the attribution fragment contains a third-person pronoun and we
        # already have an inferred subject from preceding narration, prefer
        # that subject as the speaker. Skip sentinel values that mark a
        # non-user subject (e.g. NON_USER_FEMALE_SUBJECT).
        subj = getattr(state_adapter, "last_subject", None)
        if subj and subj != self.NON_USER_FEMALE_SUBJECT:
            lower = text.lower()
            if re.search(r"\b(he|him|his|she|her|hers)\b", lower):
                return subj
        # Direct name in attribution
        direct = self._find_direct_name(text)
        if direct:
            return direct
        # If the config-based lookup failed, fall back to the user-supplied
        # character list for attribution fragments so that new, story-specific
        # names (not present in the global config) can still be resolved.
        if not direct and getattr(self, "user_character_list", None):
            norm_text = self._normalize_name(text)
            for canonical in self.user_character_list:
                cname = self._normalize_name(canonical)
                if cname and cname in norm_text:
                    return canonical
        # Speech-verb patterns near a name
        sv_name = self._search_speech_verb_patterns(text)
        if sv_name:
            return sv_name
        # Pronoun-based resolution
        pron = self._resolve_pronoun(text, state_adapter)
        if pron:
            return pron
        return None

    def _resolve_from_attribution(
        self,
        line: Any,
        state: Optional[Any],
        next_line: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Consult attached attribution (and optionally the immediate following
        short narration) to resolve a dialogue speaker before other heuristics.
        """
        if not isinstance(line, dict):
            return None

        s = _StateAdapter(state)

        # 1) Prefer inline attribution extracted from the raw line text.
        attr_text = self._extract_inline_attribution(line, next_line)
        if attr_text:
            # If this line itself has no splitter-level attribution, but the
            # *next* dialogue line does, and both attributions clearly point
            # at the same named character, assume the raw-line attribution
            # belongs to that next dialogue rather than to the current one.
            src_attr = (line.get("attribution") or "").strip()
            if not src_attr and isinstance(next_line, dict):
                next_attr = (next_line.get("attribution") or "").strip()
                if next_attr:
                    # Try to infer which name the attribution refers to.
                    name_in_attr = self._find_direct_name(attr_text) or ""
                    if name_in_attr:
                        lower_name = self._normalize_name(name_in_attr)
                        # If the same name appears in the next line's
                        # attribution or dialogue text, treat this attribution
                        # as belonging to that next dialogue.
                        nxt_text = (next_line.get("text") or "")
                        combined = (next_attr + " " + nxt_text).lower()
                        if lower_name and lower_name in self._normalize_name(combined):
                            attr_text = None

            if not attr_text:
                return None

            cand = self._resolve_speaker_from_text(attr_text, s)
            if cand:
                return cand

        return None

    def _extract_inline_attribution(
        self, line: Any, next_line: Optional[Any] = None
    ) -> Optional[str]:
        """
        Extract an attribution-like phrase for a dialogue line, preferring
        inline content from the raw line over attached or following fragments.

        Order:
          1. Raw line text (line['raw']) with inline "<Name> said"/"said <Name>".
          2. Attached line['attribution'] from the splitter.
          3. Immediate following short narration that looks like an attribution.
        """
        if not isinstance(line, dict):
            return None

        # 1) Inline patterns in the raw line, if available.
        raw = (line.get("raw") or "").strip()
        if raw:
            # Remove quoted segments to avoid matching inside dialogue text.
            tmp = re.sub(r'[\"“”‘’\'].*?[\"“”‘’\']', " ", raw)
            inline_patterns = [
                r"\b([A-Z][\w'-]{1,40})\s+"
                r"(said|asked|whispered|replied|murmured|sighed|laughed|sobbed)\b",
                r"\b(said|asked|whispered|replied|murmured|sighed|laughed|sobbed)\s+"
                r"([A-Z][\w'-]{1,40})\b",
            ]
            for pat in inline_patterns:
                m = re.search(pat, tmp)
                if m:
                    return m.group(0)

            # Fallback: if the remaining unquoted text still looks like a short
            # attribution fragment (e.g. "said the Professor", "whispered Dante"),
            # return it so downstream heuristics can resolve the speaker.
            if self._looks_like_attribution_text(tmp):
                return tmp.strip()

        # 2) Attached attribution provided by the splitter.
        attr = (line.get("attribution") or "").strip()
        if attr:
            return attr

        # 3) Immediate following line if it looks like a short attribution.
        if isinstance(next_line, dict):
            ntype = next_line.get("type")
            if ntype in ("narration", "attribution"):
                ntext = (next_line.get("text") or "").strip()
                if ntext and len(ntext.split()) <= 12 and self._looks_like_attribution_text(ntext):
                    return ntext

        return None

    def _extract_subject_from_narration(self, text: str) -> Optional[str]:
        """
        Infer the likely subject (user canonical name) from a short narration sentence.

        Matching rules (in order):

        1) Explicit name at the start of the sentence:
           - If the narration begins with a user's first token (e.g. "^Dante\\b"), choose that user.

        2) Possessive mention:
           - Match "<name>'s", "<name>’s", or "<name>s" (case-insensitive). This handles
             "Mikhail's", "Mikhail’s", and "Mikhails" forms.
           - We search each user canonical's tokens (first/last/etc) and return the first
             canonical that matches this possessive pattern.

        3) Pronoun fallback (gender-aware):
           - If narration includes male pronouns (he/him/his) and exactly one remaining male
             user (after excluding users explicitly named in the same narration), return them.
           - Analogous behavior for female pronouns (she/her).

        Returns the canonical user name or None.
        """
        if not text or not self.user_character_list:
            return None

        s = text.strip()
        if not s:
            return None

        lower_s = s.lower()

        def canonical_tokens(canonical: str) -> List[str]:
            """Tokenize a canonical name into words."""
            return [t for t in re.split(r"\\s+", canonical) if t]

        # 1) Explicit subject by name at start (first token of canonical)
        for canonical in self.user_character_list:
            tokens = canonical_tokens(canonical)
            if not tokens:
                continue
            first = re.escape(tokens[0])
            if re.match(rf"^\\s*{first}\\b", s, re.IGNORECASE):
                return canonical

        # 2) Possessive forms for user names
        for canonical in self.user_character_list:
            tokens = canonical_tokens(canonical)
            for tok in tokens:
                if not tok:
                    continue
                tok_esc = re.escape(tok)
                poss_re = re.compile(
                    rf"\\b{tok_esc}(?:['’]s|s)\\b", re.IGNORECASE)
                if poss_re.search(s):
                    return canonical

        # 3) Pronoun fallback using gender
        explicitly_mentioned: set[str] = set()
        for canonical in self.user_character_list:
            for tok in canonical_tokens(canonical):
                if re.search(rf"\\b{re.escape(tok)}\\b", s, re.IGNORECASE):
                    explicitly_mentioned.add(canonical)
                    break

        male_users: List[str] = []
        female_users: List[str] = []
        for canonical in self.user_character_list:
            if canonical in explicitly_mentioned:
                continue
            g = self._gender_for_canonical(canonical)
            if g == "M":
                male_users.append(canonical)
            elif g == "F":
                female_users.append(canonical)

        # Male pronouns
        if re.search(r"\\b(he|him|his)\\b", lower_s) and len(male_users) == 1:
            return male_users[0]

        # Female pronouns
        if re.search(r"\\b(she|her|hers)\\b", lower_s) and len(female_users) == 1:
            return female_users[0]

        return None

    def _simple_subject_from_narration(self, text: str) -> Optional[str]:
        """
        Lightweight fallback subject inference that avoids the more complex
        regex logic in _extract_subject_from_narration.

        This helper is intentionally conservative and only applies when
        user characters have been configured. It implements three rules:

          1) If the sentence begins with a user canonical name (first token),
             treat that as the subject.
          2) If the sentence contains a possessive form of a user token
             ("name's", "name’s", or "names"), treat that user as subject.
          3) If the sentence contains male/female pronouns and exactly one
             remaining male/female user (by gender), use that user.
        """
        if not text or not self.user_character_list:
            return None

        s = text.strip()
        if not s:
            return None

        lower_s = s.lower()

        def tokens_for(canonical: str) -> List[str]:
            return [t for t in canonical.split() if t]

        # 1) Explicit subject by name at start
        for canonical in self.user_character_list:
            toks = tokens_for(canonical)
            if not toks:
                continue
            first = toks[0].lower()
            stripped = s.lstrip()
            if stripped.lower().startswith(first) and (
                len(stripped) == len(first)
                or not stripped[len(first)].isalpha()
            ):
                return canonical

        # 2) Possessive mention
        for canonical in self.user_character_list:
            toks = tokens_for(canonical)
            for tok in toks:
                base = tok.lower()
                if not base:
                    continue
                if (
                    f"{base}'s" in lower_s
                    or f"{base}’s" in lower_s
                    or f"{base}s" in lower_s
                ):
                    return canonical

        # 3) Pronoun fallback using gender metadata
        explicitly_mentioned: set[str] = set()
        for canonical in self.user_character_list:
            for tok in tokens_for(canonical):
                t = tok.strip()
                if not t:
                    continue
                if re.search(rf"\b{re.escape(t.lower())}\b", lower_s):
                    explicitly_mentioned.add(canonical)
                    break

        male_users: List[str] = []
        female_users: List[str] = []
        for canonical in self.user_character_list:
            if canonical in explicitly_mentioned:
                continue
            g = self._gender_for_canonical(canonical)
            if g == "M":
                male_users.append(canonical)
            elif g == "F":
                female_users.append(canonical)

        padded = f" {lower_s} "
        # Male pronouns -> unique male user
        if any(f" {p} " in padded for p in ("he", "him", "his")) and len(male_users) == 1:
            return male_users[0]
        # Female pronouns -> unique female user
        if any(f" {p} " in padded for p in ("she", "her", "hers")) and len(female_users) == 1:
            return female_users[0]

        # Special case: narration clearly refers to a female subject ("she"/"her")
        # but there are *no* configured female user characters. In that case we
        # mark the subject as a non-user female so that subsequent dialogue is
        # *not* auto-assigned to any male user (it should remain Ambiguous
        # under strict/aggressive user modes).
        if any(f" {p} " in padded for p in ("she", "her", "hers")) and not female_users:
            return self.NON_USER_FEMALE_SUBJECT

        return None

    def _update_subject_from_narration(
        self, text: str, state_adapter: "_StateAdapter"
    ) -> Optional[str]:
        """
        Infer and persist the current speaking subject from a narration line.

        Only applies when user characters have been configured. The inferred
        subject is written to state.last_subject (if present) and returned.
        """
        if not self.user_character_list:
            return None

        subject = self._extract_subject_from_narration(text)
        if not subject:
            subject = self._simple_subject_from_narration(text)
        if not subject:
            return None

        st = state_adapter.state
        if st is not None and hasattr(st, "last_subject"):
            setattr(st, "last_subject", subject)
        return subject

    def _detect_strict_user(
        self, text: str, kind: str, state: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Strict user-driven detection path.

        Behaviour:
          * Narration lines always map to "Narrator".
          * Dialogue/thought lines are matched only against user-supplied
            names configured via set_user_characters.
          * If multiple names match a line, recency in the parser state is
            used as a deterministic tiebreaker, falling back to the original
            user order.
          * Thought lines with first-person pronouns may fall back to the
            last detected speaker (if that speaker is in the user set).
        """
        # Narration remains unaffected even in strict mode.
        if kind == "narration":
            return {"character": "Narrator"}

        # Defensive state wrapper for recency / last speaker.
        s = _StateAdapter(state)

        # 1) Collect all user-supplied names that appear in this line.
        matches: List[str] = []
        if self._user_name_order:
            for user_name, canonical in self._user_name_order:
                if not user_name:
                    continue
                try:
                    pattern = re.compile(
                        r"\b" + re.escape(user_name) + r"\b", flags=re.IGNORECASE
                    )
                    if pattern.search(text or ""):
                        matches.append(canonical)
                except re.error:
                    # If the user name cannot be safely compiled, skip it.
                    continue

        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique_matches: List[str] = []
        for name in matches:
            if name not in seen:
                seen.add(name)
                unique_matches.append(name)

        # 2) If no explicit user-name mention, maybe a thought with first-person pronoun.
        if not unique_matches:
            first_person = re.search(
                r"\b(I|I'm|I\'m|me|my|mine)\b", text or "", flags=re.IGNORECASE
            )
            if kind == "thought" or first_person:
                last = s.last_speaker
                if last and last != "Ambiguous":
                    norm_last = self._normalize_name(last)
                    # Only accept last speaker if they are part of the user set.
                    if any(
                        self._normalize_name(c) == norm_last
                        for c in self.user_character_list
                    ):
                        self._update_state_with_speaker(s, last)
                        return {"character": last}
                return {"character": "Ambiguous"}

            if kind == "dialogue":
                return {"character": "Ambiguous"}

        # 3) If we have exactly one matching user name, choose it.
        if len(unique_matches) == 1:
            chosen = unique_matches[0]
            self._update_state_with_speaker(s, chosen)
            return {"character": chosen}

        # 4) Multiple matches: tiebreak by recency, then by user order.
        chosen: Optional[str] = None
        recent_norms: List[str] = [
            self._normalize_name(n) for n in s.recent_mentions_reversed()
        ]
        candidate_norms: List[str] = [
            self._normalize_name(c) for c in unique_matches
        ]

        for rn in recent_norms:
            for cand, cn in zip(unique_matches, candidate_norms):
                if cn == rn:
                    chosen = cand
                    break
            if chosen:
                break

        # If recency did not break the tie, fall back to the earliest user order.
        if not chosen:
            chosen = unique_matches[0]

        self._update_state_with_speaker(s, chosen)
        return {"character": chosen}

    def _detect_user_aggressive(
        self, text: str, kind: str, state: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Aggressive user-driven detection path.

        Behaviour:
          * Narration lines always map to "Narrator".
          * Dialogue/thought lines are matched against user-supplied names
            using a set of variants per canonical name.
          * Multiple matches are resolved using recency in the parser state,
            then by original user order.
          * Thought and pronoun-heavy lines may fall back to the last speaker
            or simple alternation among the user set.
        """
        # Narration remains unaffected even in aggressive mode.
        if kind == "narration":
            return {"character": "Narrator"}

        s = _StateAdapter(state)
        text_str = text or ""

        # If narration has just indicated that the current subject is a
        # non-user female (e.g. "she"/"her" with no configured female
        # characters), we avoid auto-assigning dialogue in strict/aggressive
        # mode. Such lines should remain Ambiguous when no explicit user cue
        # is present.
        if kind == "dialogue" and getattr(s, "last_subject", None) == self.NON_USER_FEMALE_SUBJECT:
            return {"character": "Ambiguous"}

        # 1) Variant-based matching for each canonical user name.
        matches: List[str] = []
        for canonical in self.user_character_list:
            patterns = self._user_name_variant_map.get(canonical, [])
            for pat in patterns:
                if pat.search(text_str):
                    matches.append(canonical)
                    break

        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique_matches: List[str] = []
        for name in matches:
            key = self._normalize_name(name)
            if key in seen:
                continue
            seen.add(key)
            unique_matches.append(name)

        # 2) No explicit match → thought / pronoun or dialogue fallbacks.
        if not unique_matches:
            first_person = re.search(
                r"\b(I|I'm|I\'m|I’m|me|my|mine)\b", text_str, flags=re.IGNORECASE
            )
            if kind == "thought" or first_person:
                last = s.last_speaker
                if last and last != "Ambiguous":
                    norm_last = self._normalize_name(last)
                    if any(
                        self._normalize_name(c) == norm_last
                        for c in self.user_character_list
                    ):
                        self._update_state_with_speaker(s, last)
                        return {"character": last}
                return {"character": "Ambiguous"}

            if kind == "dialogue":
                # Dialogue fallback: optionally auto-assign using last speaker
                # or simple alternation among user-supplied speakers.
                if not self.user_auto_assign_dialogue:
                    return {"character": "Ambiguous"}

                last = s.last_speaker
                if last and last != "Ambiguous":
                    norm_last = self._normalize_name(last)
                    if any(
                        self._normalize_name(c) == norm_last
                        for c in self.user_character_list
                    ):
                        self._update_state_with_speaker(s, last)
                        return {"character": last}

                # Simple alternation: use recent mentions to pick a different
                # user speaker if possible.
                recent = s.recent_mentions_reversed()
                recent_norms = [self._normalize_name(n) for n in recent]
                user_norms = [self._normalize_name(
                    c) for c in self.user_character_list]

                for rn in recent_norms:
                    for cand, cn in zip(self.user_character_list, user_norms):
                        if cn == rn:
                            self._update_state_with_speaker(s, cand)
                            return {"character": cand}

                return {"character": "Ambiguous"}

        # 3) Single explicit match → choose directly.
        if len(unique_matches) == 1:
            chosen = unique_matches[0]
            self._update_state_with_speaker(s, chosen)
            return {"character": chosen}

        # 4) Multiple matches: tiebreak by recency, then by user order.
        recent_norms: List[str] = [
            self._normalize_name(n) for n in s.recent_mentions_reversed()
        ]
        candidate_norms: List[str] = [
            self._normalize_name(c) for c in unique_matches
        ]

        chosen: Optional[str] = None
        for rn in recent_norms:
            for cand, cn in zip(unique_matches, candidate_norms):
                if cn == rn:
                    chosen = cand
                    break
            if chosen:
                break

        if not chosen:
            # Fall back to earliest canonical in user order among the matches.
            for user_canonical in self.user_character_list:
                if user_canonical in unique_matches:
                    chosen = user_canonical
                    break
            if not chosen:
                chosen = unique_matches[0]

        self._update_state_with_speaker(s, chosen)
        return {"character": chosen}

    # ------------------ Heuristic helpers ------------------ #

    def _unpack_line(self, line: Any) -> Tuple[str, str]:
        """Accept either dict from splitter or raw string. Return (text, kind)."""
        if isinstance(line, dict):
            text = line.get("text", "")
            kind = line.get("type", "dialogue") or "dialogue"
        else:
            text = str(line or "")
            kind = "dialogue"
        return text.strip(), kind

    def _find_direct_name(self, text: str) -> Optional[str]:
        """Return known character name that appears in text, using normalized fuzzy and token-aware matching."""
        if not text or not self.characters:
            return None

        norm_text = self._normalize_name(text)
        words = set(norm_text.split())

        # 1. Try full normalized substring match first
        for stored_name in self.characters.keys():
            if stored_name in norm_text:
                return self._get_pretty_name(stored_name)

        # 2. Try token-level match (for single-word names like Dante, Lina)
        for stored_name in self.characters.keys():
            stored_tokens = set(stored_name.split())
            if len(stored_tokens) == 1 and stored_tokens & words:
                return self._get_pretty_name(stored_name)

        return None

    def _search_speech_verb_patterns(self, text: str) -> Optional[str]:
        """
        Attempt to find patterns like:
          - said <Name>
          - <Name> said
          - <Name>, she said
        Returns matched known name if present.
        """
        # look for "<verb> NAME" or "NAME <verb>" patterns using basic regex
        # pattern: (said|whispered) NAME
        if not text:
            return None

        # "<verb> <Name>"
        m = re.search(
            rf"\b{self.SPEECH_VERBS}\s+([A-Z][\w\-']{{1,40}})\b", text, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1)
            if self._is_known_name(candidate):
                return self._return_canonical(candidate)

        # "<Name> <verb>"
        m2 = re.search(
            rf"\b([A-Z][\w\-']{{1,40}})\s+{self.SPEECH_VERBS}\b", text, flags=re.IGNORECASE)
        if m2:
            candidate = m2.group(1)
            if self._is_known_name(candidate):
                return self._return_canonical(candidate)

        # Also attempt patterns like '", Name said.' where Name may come after a comma
        m3 = re.search(
            r'\b([A-Z][\w\-\'\.]{1,40}),\s*' + self.SPEECH_VERBS, text)
        if m3:
            candidate = m3.group(1)
            if self._is_known_name(candidate):
                return self._return_canonical(candidate)

        # Fallback: scan any named token adjacent to speech verbs
        tokens = re.split(r"\s+", text)
        for i, tok in enumerate(tokens):
            tt = re.sub(r'[^\w\-\']', '', tok)
            if not tt:
                continue
            if tt.lower() in self.characters:
                # look nearby for a speech verb
                window = tokens[max(0, i-3):min(len(tokens), i+4)]
                joined = " ".join(window)
                if re.search(self.SPEECH_VERBS, joined, flags=re.IGNORECASE):
                    return self._return_canonical(tt)
        return None

    def _is_known_name(self, token: str) -> bool:
        if not token:
            return False
        return token.lower() in self.characters

    def _get_pretty_name(self, stored_name: str) -> str:
        """Return properly cased version of a normalized name, if available."""
        for original in self.characters.keys():
            if self._normalize_name(original) == stored_name:
                return original
        return stored_name

    def _return_canonical(self, token: str) -> str:
        """Return the exact casing of the name as in the character config (for TTS consistency)."""
        key = token.lower()
        for proper_name in self.characters.keys():
            if proper_name.lower() == key:
                return proper_name
        return token.capitalize()

    def _resolve_pronoun(self, text: str, s: "_StateAdapter") -> Optional[str]:
        """
        Resolve pronouns using gender map from character config and parser state.
        """
        txt = text.lower()

        # First-person 'I' resolution
        if re.search(r"\bI\b", text):
            if s.last_speaker:
                return s.last_speaker
            # Try to fallback to most recent speaker in context
            recents = s.recent_mentions_reversed()
            if recents:
                return recents[0]
            return None

        # He/Him → Male
        if re.search(r"\b(he|him)\b", txt):
            # Look for recent male
            for name in s.recent_mentions_reversed():
                if self.gender_map.get(name.lower(), "U") == "M":
                    return name
            # Fallback for male pronouns
            male_candidates = [n for n, meta in self.characters.items(
            ) if isinstance(meta, dict) and meta.get("gender") == "M"]
            if len(male_candidates) == 1:
                return self._return_canonical(male_candidates[0])
            elif 1 < len(male_candidates) <= 3:
                if not s.recent_mentions:
                    return self._return_canonical(male_candidates[0])
            return None

        # She/Her → Female
        if re.search(r"\b(she|her)\b", txt):
            for name in s.recent_mentions_reversed():
                if self.gender_map.get(name.lower(), "U") == "F":
                    return name
            female_candidates = [n for n, meta in self.characters.items(
            ) if isinstance(meta, dict) and meta.get("gender") == "F"]
            if len(female_candidates) == 1:
                # Extra safety: ensure last speaker wasn't male
                last = s.last_speaker
                if last and any(self.characters.get(self._normalize_name(last), {}).get("gender") == "M" for _ in [0]):
                    return None
                return self._return_canonical(female_candidates[0])
            elif 1 < len(female_candidates) <= 3:
                if not s.recent_mentions:
                    return self._return_canonical(female_candidates[0])
            return None

        # They → Ambiguous, skip
        return None

    def _resolve_from_recent_mentions(self, s: " _StateAdapter") -> Optional[str]:
        """Return the most recent mention from state.recent_mentions if present."""
        for name in s.recent_mentions_reversed():
            if name:
                return name
        return None

    def _alternation_heuristic(self, s: " _StateAdapter") -> Optional[str]:
        """
        If last speaker exists and we have a small set of recent mentions or known characters,
        attempt a naive alternation: choose a recent different speaker if available.
        This is intentionally conservative.
        """
        if not s.last_speaker:
            return None
        # If the last speaker is known and recent_mentions contain other names, pick the most recent different one
        for name in s.recent_mentions_reversed():
            if name and name != s.last_speaker:
                return name
        return None

    def _candidate_list(self, s: " _StateAdapter", text: str = "") -> List[str]:
        """Build a short candidate list: recent mentions first, then top known characters (limit 5)."""
        candidates = []
        for nm in s.recent_mentions_reversed():
            if nm and nm not in candidates:
                candidates.append(nm)
            if len(candidates) >= 3:
                break
        # fill with top-known (alphabetical) if still empty
        if not candidates:
            candidates = [k for k in sorted(self.characters.keys())[:5]]

        # Special handling for Adam variants
        if text and "adam" not in text.lower():
            candidates = [
                c for c in candidates if not c.lower().startswith("adam")]

        return candidates

    def _update_state_with_speaker(self, state_adapter, name: str):
        """
        Internal helper to update the parser state when a character is confidently detected.
        """
        if not name:
            return
        try:
            state_adapter.update_last_speaker(name)
            # Also update subject if the underlying state exposes this field.
            st = getattr(state_adapter, "state", None)
            if st is not None and hasattr(st, "last_subject"):
                setattr(st, "last_subject", name)
        except Exception:
            pass


# ------------------------- small state adapter ------------------------- #


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


# ------------------------- end of file ------------------------- #

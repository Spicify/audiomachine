"""Dialogue resolution helpers for the CharacterDetector."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .character_detector_state import _StateAdapter


class ResolutionMixin:
    def _detect_strict_user(
        self,
        text: str,
        kind: str,
        state: Optional[Any],
        next_line: Optional[Any],
        line_dict: Optional[Any],
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
        text_str = text or ""
        if kind == "dialogue":
            if getattr(s, "last_subject", None) == self.NON_USER_FEMALE_SUBJECT:
                if not self._consume_pending_non_user_female(s):
                    self._set_pending_non_user_female(
                        s, max(self.NON_USER_FEMALE_DIALOGUE_TTL - 1, 0)
                    )
                return {"character": "Ambiguous"}
            if self._consume_pending_non_user_female(s):
                return {"character": "Ambiguous"}

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
                r"\b(I|I'm|I\'m|I\u2019m|me|my|mine)\b", text_str or "", flags=re.IGNORECASE
            )
            if kind == "thought" or first_person:
                if first_person:
                    female_candidate = self._contextual_female_candidate(s, next_line, line_dict)
                    if female_candidate:
                        self._update_state_with_speaker(s, female_candidate)
                        return self._finalize_character(female_candidate)
                    if not self._female_user_candidates() and self._context_has_feminine_cue(
                        s, next_line, line_dict
                    ):
                        self._set_pending_non_user_female(
                            s, max(self.NON_USER_FEMALE_DIALOGUE_TTL - 1, 0)
                        )
                        return {"character": "Ambiguous"}
                last = s.last_speaker
                if last and last != "Ambiguous":
                    norm_last = self._normalize_name(last)
                    if any(
                        self._normalize_name(c) == norm_last
                        for c in self.user_character_list
                    ):
                        self._update_state_with_speaker(s, last)
                        return self._finalize_character(last)
                return {"character": "Ambiguous"}

            if kind == "dialogue":
                if not self.user_auto_assign_dialogue:
                    return {"character": "Ambiguous"}
                female_candidate = self._contextual_female_candidate(s, next_line, line_dict)
                if female_candidate:
                    self._update_state_with_speaker(s, female_candidate)
                    return self._finalize_character(female_candidate)
                if not self._female_user_candidates() and self._context_has_feminine_cue(
                    s, next_line, line_dict
                ):
                    self._set_pending_non_user_female(
                        s, max(self.NON_USER_FEMALE_DIALOGUE_TTL - 1, 0)
                    )
                    return {"character": "Ambiguous"}
                last = s.last_speaker
                if last and last != "Ambiguous":
                    norm_last = self._normalize_name(last)
                    if any(
                        self._normalize_name(c) == norm_last
                        for c in self.user_character_list
                    ):
                        self._update_state_with_speaker(s, last)
                        return self._finalize_character(last)
                return {"character": "Ambiguous"}

        # 3) If we have exactly one matching user name, choose it.
        if len(unique_matches) == 1:
            chosen = unique_matches[0]
            self._update_state_with_speaker(s, chosen)
            return self._finalize_character(chosen)

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

    def _detect_user_aggressive(
        self,
        text: str,
        kind: str,
        state: Optional[Any],
        next_line: Optional[Any],
        line_dict: Optional[Any],
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
        if kind == "dialogue":
            if getattr(s, "last_subject", None) == self.NON_USER_FEMALE_SUBJECT:
                if not self._consume_pending_non_user_female(s):
                    self._set_pending_non_user_female(
                        s, max(self.NON_USER_FEMALE_DIALOGUE_TTL - 1, 0)
                    )
                return {"character": "Ambiguous"}
            if self._consume_pending_non_user_female(s):
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

        # 2) No explicit match ? thought / pronoun or dialogue fallbacks.
        if not unique_matches:
            first_person = re.search(
                r"\b(I|I'm|I\'m|I\u2019m|me|my|mine)\b", text_str or "", flags=re.IGNORECASE
            )
            if kind == "thought" or first_person:
                if first_person:
                    female_candidate = self._contextual_female_candidate(s, next_line, line_dict)
                    if female_candidate:
                        self._update_state_with_speaker(s, female_candidate)
                        return self._finalize_character(female_candidate)
                    if not self._female_user_candidates() and self._context_has_feminine_cue(
                        s, next_line, line_dict
                    ):
                        self._set_pending_non_user_female(
                            s, max(self.NON_USER_FEMALE_DIALOGUE_TTL - 1, 0)
                        )
                        return {"character": "Ambiguous"}
                last = s.last_speaker
                if last and last != "Ambiguous":
                    norm_last = self._normalize_name(last)
                    if any(
                        self._normalize_name(c) == norm_last
                        for c in self.user_character_list
                    ):
                        self._update_state_with_speaker(s, last)
                        return self._finalize_character(last)
                return {"character": "Ambiguous"}

            if kind == "dialogue":
                # Dialogue fallback: optionally auto-assign using last speaker
                # or simple alternation among user-supplied speakers.
                if not self.user_auto_assign_dialogue:
                    return {"character": "Ambiguous"}

                female_candidate = self._contextual_female_candidate(s, next_line, line_dict)
                if female_candidate:
                    self._update_state_with_speaker(s, female_candidate)
                    return self._finalize_character(female_candidate)
                if not self._female_user_candidates() and self._context_has_feminine_cue(
                    s, next_line, line_dict
                ):
                    self._set_pending_non_user_female(
                        s, max(self.NON_USER_FEMALE_DIALOGUE_TTL - 1, 0)
                    )
                    return {"character": "Ambiguous"}

                last = s.last_speaker
                if last and last != "Ambiguous":
                    norm_last = self._normalize_name(last)
                    if any(
                        self._normalize_name(c) == norm_last
                        for c in self.user_character_list
                    ):
                        self._update_state_with_speaker(s, last)
                        return self._finalize_character(last)
                return {"character": "Ambiguous"}
        # 3) Single explicit match → choose directly.
        if len(unique_matches) == 1:
            chosen = unique_matches[0]
            self._update_state_with_speaker(s, chosen)
            return self._finalize_character(chosen)

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
        return self._finalize_character(chosen)

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

        # 1. Try boundary-aware match on the normalized text so that
        #    character names never match inside other words (e.g. "Eve"
        #    matching "never" or "Mom" matching "moment").
        padded = f" {norm_text} "
        for stored_name in self.characters.keys():
            stored_norm = self._normalize_name(stored_name)
            if not stored_norm:
                continue
            target = f" {stored_norm} "
            if target in padded:
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
        if re.search(r"\b(he|him|his)\b", txt):
            # Look for recent male
            for name in s.recent_mentions_reversed():
                if self.gender_map.get(name.lower(), "U") == "M":
                    return name
            user_males = self._male_user_candidates()
            if len(user_males) == 1:
                return user_males[0]
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
        if re.search(r"\b(she|her|hers)\b", txt):
            for name in s.recent_mentions_reversed():
                if self.gender_map.get(name.lower(), "U") == "F":
                    return name
            user_females = self._female_user_candidates()
            if len(user_females) == 1:
                return user_females[0]
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

    def _male_user_candidates(self) -> List[str]:
        males: List[str] = []
        for canonical in self.user_character_list:
            if self._gender_for_canonical(canonical) == "M":
                males.append(canonical)
        return males

    def _female_user_candidates(self) -> List[str]:
        females: List[str] = []
        for canonical in self.user_character_list:
            if self._gender_for_canonical(canonical) == "F":
                females.append(canonical)
        return females

    def _text_has_feminine_cue(self, text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        padded = f" {lower} "
        if any(f" {p} " in padded for p in self.FEMALE_PRONOUNS):
            return True
        return any(cue in lower for cue in self.FEMALE_NOUN_CUES)

    def _match_name_from_text(self, text: str, candidates: List[str]) -> Optional[str]:
        if not text or not candidates:
            return None
        lower = text.lower()
        for canonical in candidates:
            tokens = [t for t in re.split(r"\s+", canonical) if t]
            for tok in tokens:
                tok_norm = tok.lower()
                if re.search(rf"\b{re.escape(tok_norm)}\b", lower):
                    return canonical
        return None

    def _contextual_female_candidate(
        self, state_adapter, next_line: Optional[Any], line_dict: Optional[Any]
    ) -> Optional[str]:
        females = self._female_user_candidates()
        if not females:
            return None
        subject = getattr(state_adapter, "last_subject", None)
        if subject and subject in females:
            return subject
        texts: List[str] = []
        prev = getattr(state_adapter, "last_line", None)
        if isinstance(prev, dict):
            texts.append(str(prev.get("text") or ""))
        elif isinstance(prev, str):
            texts.append(prev)
        if isinstance(line_dict, dict):
            texts.append(str(line_dict.get("attribution") or ""))
        if next_line is not None:
            if isinstance(next_line, dict):
                texts.append(str(next_line.get("text") or ""))
            else:
                texts.append(str(next_line))
        for snippet in texts:
            candidate = self._match_name_from_text(snippet, females)
            if candidate:
                return candidate
            if self._text_has_feminine_cue(snippet) and len(females) == 1:
                return females[0]
        return None

    def _context_has_feminine_cue(
        self, state_adapter, next_line: Optional[Any], line_dict: Optional[Any]
    ) -> bool:
        texts: List[str] = []
        prev = getattr(state_adapter, "last_line", None)
        if isinstance(prev, dict):
            texts.append(str(prev.get("text") or ""))
        elif isinstance(prev, str):
            texts.append(prev)
        if isinstance(line_dict, dict):
            texts.append(str(line_dict.get("attribution") or ""))
        if next_line is not None:
            if isinstance(next_line, dict):
                texts.append(str(next_line.get("text") or ""))
            else:
                texts.append(str(next_line))
        return any(self._text_has_feminine_cue(t) for t in texts)

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
            self._set_pending_non_user_female(state_adapter, 0)
        except Exception:
            pass

    def _pending_non_user_female(self, state_adapter) -> int:
        try:
            count = getattr(state_adapter, "pending_non_user_female", 0) or 0
            return int(count)
        except Exception:
            return 0

    def _set_pending_non_user_female(self, state_adapter, value: int) -> None:
        count = 0
        try:
            count = max(0, int(value))
        except Exception:
            count = 0
        try:
            state_adapter.pending_non_user_female = count
        except Exception:
            pass
        st = getattr(state_adapter, "state", None)
        if st is not None:
            try:
                setattr(st, "pending_non_user_female", count)
            except Exception:
                pass

    def _consume_pending_non_user_female(self, state_adapter) -> bool:
        count = self._pending_non_user_female(state_adapter)
        if count <= 0:
            return False
        self._set_pending_non_user_female(state_adapter, count - 1)
        return True

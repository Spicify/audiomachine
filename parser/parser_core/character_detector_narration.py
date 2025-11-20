"""Narration and attribution helpers for the CharacterDetector."""
from __future__ import annotations

import re
from typing import Any, List, Optional

from .character_detector_state import _StateAdapter

class NarrationMixin:
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
        # "<Name> said" style fragments (e.g. "Aleksandr murmured ...")
        if re.search(
            r"\b[A-Z][\w'-]{1,40}\s+" + self.SPEECH_VERBS + r"\b",
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
        # Direct name in attribution takes priority.
        direct = self._find_direct_name(text)
        if direct:
            return direct
        # Fall back to user-specified names so story-specific entries can match.
        if getattr(self, "user_character_list", None):
            norm_text = self._normalize_name(text)
            for canonical in self.user_character_list:
                cname = self._normalize_name(canonical)
                if cname and cname in norm_text:
                    return canonical
        # Speech-verb patterns near a name.
        sv_name = self._search_speech_verb_patterns(text)
        if sv_name:
            return sv_name
        # Pronoun-based resolution.
        pron = self._resolve_pronoun(text, state_adapter)
        if pron:
            return pron
        # Lastly, consult the previously inferred narration subject when no
        # explicit cue was found but the fragment's pronouns agree with it.
        subj = getattr(state_adapter, "last_subject", None)
        if subj and subj != self.NON_USER_FEMALE_SUBJECT:
            lower = text.lower()
            has_female = re.search(r"\b(she|her|hers)\b", lower)
            has_male = re.search(r"\b(he|him|his)\b", lower)
            gender = self._gender_for_canonical(subj)
            if has_female and not has_male:
                if gender == "F":
                    return subj
                return None
            if has_male and not has_female:
                if gender in (None, "M"):
                    return subj
                return None
            if not has_female and not has_male:
                return subj
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
            if isinstance(next_line, dict):
                next_text = (next_line.get("text") or "").strip()
                if next_text and attr_text.strip().lower() == next_text.lower():
                    if not self._looks_like_attribution_text(attr_text):
                        attr_text = None

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
        explicit splitter attributions before inline or nearby fragments.

        Order:
          1. Attached line['attribution'] from the splitter.
          2. Raw line text (line['raw']) with inline "<Name> said"/"said <Name>".
          3. Immediate following short narration that looks like an attribution.
        """
        if not isinstance(line, dict):
            return None

        attr = (line.get("attribution") or "").strip()
        if attr:
            return attr

        raw = (line.get("raw") or "").strip()
        if raw:
            tmp = re.sub(r'["\'“”‘’].*?["\'“”‘’]', " ", raw)
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

            if self._looks_like_attribution_text(tmp):
                return tmp.strip()

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
           - If the narration begins with a user's first token (e.g. "^Dante\b"), choose that user.

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
            if re.match(rf"^\\s*{first}\b", s, re.IGNORECASE):
                return canonical

        # 2) Possessive forms for user names
        for canonical in self.user_character_list:
            tokens = canonical_tokens(canonical)
            for tok in tokens:
                if not tok:
                    continue
                tok_esc = re.escape(tok)
                poss_re = re.compile(
                    rf"\b{tok_esc}(?:['’]s|s)\b", re.IGNORECASE)
                if poss_re.search(s):
                    return canonical

        # 3) Pronoun fallback using gender
        explicitly_mentioned: set[str] = set()
        for canonical in self.user_character_list:
            for tok in canonical_tokens(canonical):
                if re.search(rf"\b{re.escape(tok)}\b", s, re.IGNORECASE):
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
        if re.search(r"\b(he|him|his)\b", lower_s) and len(male_users) == 1:
            return male_users[0]

        # Female pronouns
        if re.search(r"\b(she|her|hers)\b", lower_s) and len(female_users) == 1:
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
        if subject == self.NON_USER_FEMALE_SUBJECT:
            self._set_pending_non_user_female(
                state_adapter, self.NON_USER_FEMALE_DIALOGUE_TTL
            )
        return subject

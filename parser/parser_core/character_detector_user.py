"""User character helpers for the CharacterDetector."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


class UserCharacterMixin:
    def inject_user_characters(self, user_characters: list[str]):
        """
        Allow user to provide additional or overriding character names at runtime.
        User-provided names take precedence over config-defined ones.
        """
        if not user_characters:
            return
        for name in user_characters:
            norm = self._normalize_name(name)
            existing = self.characters.get(norm)
            if isinstance(existing, dict) and existing.get("gender"):
                continue
            if norm not in self.characters:
                self.characters[norm] = {"gender": None, "source": "user"}
            else:
                meta = existing or {}
                if not isinstance(meta, dict):
                    meta = {}
                if "gender" not in meta or not meta["gender"]:
                    meta["gender"] = None
                meta["source"] = "user"
                self.characters[norm] = meta

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
            canonical = self._map_user_name_to_canonical(user_name_str)
            if not canonical:
                canonical = user_name_str

            canon_key = self._normalize_name(canonical)
            if canon_key not in self.characters:
                self.characters[canon_key] = {}
            self._display_name_map[canon_key] = self._format_display_name(canonical)

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

        user_norm_full = self._normalize_name(user_name)
        if not user_norm_full:
            return None

        variants: List[Tuple[int, str]] = [(0, user_norm_full)]
        seen_variants = {user_norm_full}
        for tok in user_norm_full.split():
            if not tok:
                continue
            if tok in seen_variants:
                continue
            seen_variants.add(tok)
            variants.append((1, tok))

        matches: List[Tuple[int, int, str, str]] = []

        for canon_key in self.characters.keys():
            canon_norm = self._normalize_name(canon_key)
            if not canon_norm:
                continue

            tokens = canon_norm.split()
            best_entry: Optional[Tuple[int, int, str, str]] = None

            for variant_priority, variant in variants:
                category: Optional[int] = None
                if canon_norm == variant:
                    category = 0
                elif any(t == variant for t in tokens):
                    category = 1
                elif any(t.startswith(variant) for t in tokens):
                    category = 2
                elif variant in canon_norm:
                    category = 3

                if category is None:
                    continue

                score = variant_priority * 10 + category
                best_entry = (score, -len(canon_norm), canon_norm, canon_key)
                break

            if best_entry:
                matches.append(best_entry)

        if not matches:
            return None

        matches.sort(key=lambda m: (m[0], m[1], m[2]))
        return matches[0][3]

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

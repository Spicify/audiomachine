import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class _EmotionEntry:
    """
    Internal representation of a single emotion label.

    Attributes:
        label:     Canonical emotion label name, e.g. "happy".
        priority:  Integer priority (0 is highest). Lower numbers win ties.
        index:     Original file order index (used as a stable tiebreaker).
        keywords:  List of keyword/phrase strings for this emotion.
        patterns:  Precompiled regex patterns corresponding to `keywords`.
    """

    label: str
    priority: int
    index: int
    keywords: List[str]
    patterns: List[re.Pattern]


class EmotionTagger:
    """
    Deterministic emotion tagger based on a simple keyword configuration.

    Configuration format
    --------------------
    Supports two JSON layouts in `configs/emotions.json`:

    1) Map form (current in this repo):
        {
          "angry": ["angry", "furious", "rage"],
          "happy": ["happy", "joyful"]
        }
       - Keys are emotion labels, values are lists of keyword strings.
       - Priority is inferred from file order: first key -> priority 0, next -> 1, etc.

    2) List-of-objects form:
        [
          {"label": "angry", "keywords": ["angry", "furious"], "priority": 1},
          {"label": "happy", "keywords": ["happy", "joyful"], "priority": 0}
        ]
       - `priority` is optional; if omitted, it is inferred from list order.

    Matching algorithm
    ------------------
    * All matching is case-insensitive using precompiled `re` patterns.
    * Each keyword is converted into a regex that:
        - Escapes literal characters.
        - Replaces spaces with `\\s+` to tolerate spacing differences.
        - Wraps the pattern in word boundaries `\\b` when the first/last
          characters are alphanumeric (so we avoid splitting on punctuation-
          heavy forms when boundaries would be misleading).
    * Keywords for each label are sorted longest-first (by token count then by
      string length) so multi-word phrases match before shorter substrings.
    * Negation handling:
        - Before scoring a keyword match, the text is tokenised via `\\S+`.
        - If one of `["not", "never", "no", "n't"]` appears within the three
          tokens immediately preceding the matched span, that match is
          considered negated and contributes no score.
    * Scoring:
        - Each single-word keyword match contributes 1 point to its label.
        - Each multi-word keyword match contributes 2 points to its label.
        - Multiple matches of the same keyword all contribute (no dedup).
    * Ranking (deterministic tie-breaking):
        1. Higher score first (descending).
        2. Lower priority value first (ascending).
        3. Earlier appearance in the config file (index, ascending).
        4. Label name alphabetical (case-insensitive).
      The sort is stable, so earlier file order is preserved when all other
      criteria are equal.

    Public API
    ----------
    * `tag(text, top_n=2) -> List[str]`
      Returns a deterministic list of up to `top_n` emotion labels. If fewer
      than `top_n` labels have a positive score, the remainder is filled with
      the literal string `"neutral"` to reach exactly `top_n` items.

    * `explain(text) -> dict`
      Returns a dict with:
        - "matches": list of matched (or negated) keyword spans.
        - "scores": per-emotion numeric scores.
        - "top": final list of emotion labels for `top_n=2`.
    """

    NEGATION_TOKENS = {"not", "never", "no", "n't"}
    AROUSAL_CUES = (
        "thigh",
        "thighs",
        "between her legs",
        "between his legs",
        "ache",
        "aching",
        "aching for",
        "heat pooled",
        "heat between",
        "smolder",
        "smoulder",
        "smoldering",
        "smouldering",
        "slick",
        "wet",
        "pulse",
        "pulsing",
        "throb",
        "throbbing",
        "squirm",
        "squirming",
        "grinding",
        "grind",
        "lap",
        "gasp",
        "gasping",
        "moan",
        "moaning",
        "needy",
        "desperate",
        "hungry",
        "yearned",
        "yearning",
        "breath hitched",
        "breathless",
        "shiver",
        "shivered",
    )
    ROMANTIC_CUES = (
        "kiss",
        "kissing",
        "mouth on",
        "lips on",
        "lips against",
        "neck",
        "throat",
        "caress",
        "caressed",
        "stroking",
        "traced",
        "lantern",
        "candle",
        "candlelight",
        "soft music",
        "slow dance",
        "romantic",
        "tender",
        "gentle",
        "embrace",
        "holding her",
        "holding him",
        "pressed a kiss",
        "wine",
        "champagne",
        "couch",
        "sofa",
    )
    DOMINANCE_CUES = (
        "belong to",
        "belongs to",
        "you belong",
        "mine",
        "you're mine",
        "claim",
        "claimed",
        "claiming",
        "punish",
        "punishment",
        "obedient",
        "obedience",
        "discipline",
        "control",
        "controlled",
        "possessive",
        "possessiveness",
        "collar",
        "leash",
        "grip tightened",
        "grip on",
        "warning glare",
        "warning growl",
        "dominant",
        "command",
        "commanded",
    )
    TENSION_CUES = (
        "tension",
        "tightened",
        "tighten",
        "warning",
        "danger",
        "threat",
        "threatened",
        "storm",
        "electric",
        "electricity",
        "pulse quickened",
        "heartbeat",
        "heart hammered",
        "edge of",
        "almost",
        "nearly",
        "could barely",
        "breath hitched",
        "nervous",
        "anxious",
    )

    def __init__(self, config_path: str = "parser/configs/emotions.json") -> None:
        self.config_path = config_path
        self._entries: List[_EmotionEntry] = []
        self._label_to_keywords: Dict[str, List[str]] = {}
        self._duplicate_keywords: Dict[str, List[str]] = {}

        config_data = self._load_json(config_path)
        entries = self._normalise_config(config_data)
        self._build_entries(entries)
        self._compute_metadata()

    # --------------------------------------------------------------------- #
    # Configuration loading and normalisation
    # --------------------------------------------------------------------- #

    def _load_json(self, path: str) -> Any:
        """Load raw JSON from `path` and return the parsed object."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _normalise_config(self, cfg: Any) -> List[Dict[str, Any]]:
        """
        Convert the config into a uniform list-of-dicts structure:

        [
          {"label": str, "keywords": List[str], "priority": int, "index": int},
          ...
        ]

        Supports both map form and list-of-objects form.
        """
        entries: List[Dict[str, Any]] = []

        # Map form: {"angry": [...], "happy": [...]}
        if isinstance(cfg, dict):
            for idx, (label, keywords) in enumerate(cfg.items()):
                if not isinstance(keywords, list):
                    continue
                cleaned_keywords = [
                    str(kw).strip()
                    for kw in keywords
                    if isinstance(kw, str) and kw.strip()
                ]
                entries.append(
                    {
                        "label": str(label),
                        "keywords": cleaned_keywords,
                        "priority": idx,
                        "index": idx,
                    }
                )
            return entries

        # List-of-objects form
        if isinstance(cfg, list):
            for idx, item in enumerate(cfg):
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", f"label_{idx}"))
                keywords = item.get("keywords", []) or []
                if not isinstance(keywords, list):
                    keywords = [keywords]
                cleaned_keywords = [
                    str(kw).strip()
                    for kw in keywords
                    if isinstance(kw, str) and kw.strip()
                ]
                priority = item.get("priority")
                if not isinstance(priority, int):
                    priority = idx
                entries.append(
                    {
                        "label": label,
                        "keywords": cleaned_keywords,
                        "priority": priority,
                        "index": idx,
                    }
                )
            return entries

        # Fallback: unsupported shape, return empty config
        return []

    def _build_entries(self, raw_entries: List[Dict[str, Any]]) -> None:
        """Compile regexes and build `_EmotionEntry` objects."""
        entries: List[_EmotionEntry] = []

        for raw in raw_entries:
            label = raw["label"]
            priority = int(raw.get("priority", 0))
            index = int(raw.get("index", 0))

            # Sort keywords longest-first:
            #   - first by token-count, then by character length.
            keywords = list(dict.fromkeys(raw.get("keywords", [])))  # dedupe, keep order
            keywords_sorted = sorted(
                keywords,
                key=lambda k: (-len(k.split()), -len(k)),
            )
            patterns: List[re.Pattern] = [
                self._compile_keyword_pattern(kw) for kw in keywords_sorted
            ]

            entries.append(
                _EmotionEntry(
                    label=label,
                    priority=priority,
                    index=index,
                    keywords=keywords_sorted,
                    patterns=patterns,
                )
            )

        self._entries = entries

    def _compile_keyword_pattern(self, keyword: str) -> re.Pattern:
        """
        Build a regex for a single keyword or phrase.

        * Spaces are converted to `\\s+` to allow flexible spacing.
        * Keywords are escaped literally.
        * If the first/last characters are alphanumeric, we wrap the whole
          pattern in `\\b` word-boundaries for token-aware matching.
        """
        escaped = re.escape(keyword)
        # Allow any whitespace between words in multi-word phrases
        pattern_body = escaped.replace(r"\ ", r"\s+")

        start_char = keyword[0] if keyword else ""
        end_char = keyword[-1] if keyword else ""

        prefix = r"\b" if start_char.isalnum() else ""
        suffix = r"\b" if end_char.isalnum() else ""

        pattern_text = f"{prefix}{pattern_body}{suffix}"
        return re.compile(pattern_text, flags=re.IGNORECASE)

    def _compute_metadata(self) -> None:
        """
        Precompute convenience metadata:
          * Mapping of labels to keyword lists.
          * Cross-label duplicate keywords (case-insensitive).
        """
        label_to_keywords: Dict[str, List[str]] = {}
        keyword_to_labels: Dict[str, List[str]] = {}

        for entry in self._entries:
            label = entry.label
            label_to_keywords[label] = list(entry.keywords)
            for kw in entry.keywords:
                key = kw.lower()
                keyword_to_labels.setdefault(key, []).append(label)

        duplicates = {
            kw: labels for kw, labels in keyword_to_labels.items() if len(labels) > 1
        }

        self._label_to_keywords = label_to_keywords
        self._duplicate_keywords = duplicates

    # --------------------------------------------------------------------- #
    # Public helpers for configuration inspection
    # --------------------------------------------------------------------- #

    @property
    def emotion_labels(self) -> List[str]:
        """Return the ordered list of configured emotion labels."""
        return [e.label for e in self._entries]

    @property
    def label_keyword_counts(self) -> Dict[str, int]:
        """Return a mapping of label -> number of configured keywords."""
        return {label: len(kws) for label, kws in self._label_to_keywords.items()}

    @property
    def duplicate_keywords(self) -> Dict[str, List[str]]:
        """
        Return a mapping of lowercased keyword -> list of labels that share it,
        only including keywords that appear under more than one label.
        """
        return dict(self._duplicate_keywords)

    # --------------------------------------------------------------------- #
    # Core tagging logic
    # --------------------------------------------------------------------- #

    def tag(
        self,
        text: str,
        top_n: int = 2,
        *,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> List[str]:
        """
        Tag the given text with up to `top_n` emotion labels.

        Behaviour:
          * Apply lightweight rule-based cues first (speech verbs, adverbs,
            strong lexical hints). If these yield a pair of emotions, they
            take precedence.
          * Otherwise, fall back to config-driven keyword scoring via
            `_analyze` and post-process the result to ensure:
                - exactly two labels,
                - labels are distinct, and
                - a non-generic fallback of ["soft", "calm"] when no cues
                  are present.
        """
        debug = os.getenv("DEBUG_EMOTIONS") == "1"
        src = text or ""
        context_chunks: List[str] = []
        for ctx in (context_before, context_after):
            if ctx:
                context_chunks.append(str(ctx))
        context_text = " ".join(context_chunks).strip()
        combined = " ".join(part for part in (src, context_text) if part)
        signals = self._contextual_signals(combined)
        fallback_labels = self._fallback_from_signals(signals)

        # 1) Rule-based cues (speech verbs, adverbs, punctuation, etc.).
        rule_labels = self._rule_based_emotions(combined)
        if debug:
            print("[EMO DEBUG] tag_input=", repr(combined))
            if rule_labels:
                print("[EMO DEBUG] rule_match=", rule_labels)

        if rule_labels:
            final = self._finalise_labels(rule_labels, allow_fallback=True, fallback=fallback_labels)
            if debug:
                print("[EMO DEBUG] final_from_rules=", final)
            return final

        context_labels = self._pick_contextual_labels(signals)
        if context_labels:
            final = self._finalise_labels(context_labels, allow_fallback=True, fallback=fallback_labels)
            if debug:
                print("[EMO DEBUG] final_from_context=", final, "signals=", signals)
            return final

        # 2) Config-driven scoring as a fallback.
        analysis = self._analyze(combined or src, top_n=top_n)
        base = list(analysis["top"])
        if debug:
            print("[EMO DEBUG] base_from_config=", base)
        final = self._finalise_labels(base, allow_fallback=True, fallback=fallback_labels)
        if debug:
            print("[EMO DEBUG] final_from_config=", final)
        return final

    def explain(self, text: str) -> Dict[str, Any]:
        """
        Explain emotion tagging for a given text.

        Returns a dict containing:
          * "matches": list of dicts with keys:
                - "label": emotion label name
                - "keyword": matched keyword string
                - "start": character offset of match start
                - "end": character offset of match end
                - "negated": whether a nearby negation cancelled the score
          * "scores": mapping of label -> numeric score
          * "top": final list of labels for `top_n=2`
        """
        return self._analyze(text, top_n=2)

    # --------------------------------------------------------------------- #
    # Internal analysis helpers
    # --------------------------------------------------------------------- #

    def _tokenize_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Tokenise the text into (token, start, end) using a simple `\\S+` regex.
        This is used exclusively for the negation window logic.
        """
        tokens: List[Tuple[str, int, int]] = []
        for match in re.finditer(r"\S+", text):
            tokens.append((match.group(0), match.start(), match.end()))
        return tokens

    def _find_token_index_for_pos(
        self, tokens: List[Tuple[str, int, int]], pos: int
    ) -> Optional[int]:
        """Return the index of the token that covers character position `pos`."""
        for i, (_, start, end) in enumerate(tokens):
            if start <= pos < end:
                return i
        return None

    def _has_negation_before(
        self, tokens: List[Tuple[str, int, int]], token_index: int
    ) -> bool:
        """
        Determine whether any negation token appears within the three tokens
        immediately preceding `token_index`.
        """
        start_idx = max(0, token_index - 3)
        for i in range(start_idx, token_index):
            tok = tokens[i][0].lower().strip(".,!?;:\"'()[]{}")
            if tok in self.NEGATION_TOKENS:
                return True
        return False

    def _analyze(self, text: str, top_n: int) -> Dict[str, Any]:
        """
        Shared analysis routine used by `tag` and `explain`.

        Computes:
          * matches (with negation flags),
          * per-label scores,
          * final top-N labels using deterministic tie-breaking.
        """
        if not text:
            # No text: no scores; caller will apply a safe fallback.
            scores: Dict[str, int] = {e.label: 0 for e in self._entries}
            return {"matches": [], "scores": scores, "top": []}

        tokens = self._tokenize_with_spans(text)
        scores: Dict[str, int] = {e.label: 0 for e in self._entries}
        matches: List[Dict[str, Any]] = []

        for entry in self._entries:
            label = entry.label
            for keyword, pattern in zip(entry.keywords, entry.patterns):
                for m in pattern.finditer(text):
                    start, end = m.start(), m.end()
                    token_idx = self._find_token_index_for_pos(tokens, start)
                    negated = False
                    if token_idx is not None and self._has_negation_before(
                        tokens, token_idx
                    ):
                        negated = True

                    # Record the match for explanation regardless of negation
                    matches.append(
                        {
                            "label": label,
                            "keyword": keyword,
                            "start": start,
                            "end": end,
                            "negated": negated,
                        }
                    )

                    if negated:
                        continue

                    # Multi-word keyword -> 2 points, otherwise 1.
                    token_count = len(keyword.split())
                    delta = 2 if token_count > 1 else 1
                    scores[label] += delta

        # Rank labels according to score and deterministic tiebreak rules.
        ranked: List[Tuple[str, int, int, int]] = []  # (label, score, priority, index)
        for entry in self._entries:
            ranked.append((entry.label, scores[entry.label], entry.priority, entry.index))

        ranked.sort(
            key=lambda item: (
                -item[1],  # score descending
                item[2],  # priority ascending
                item[3],  # index ascending
                item[0].lower(),  # label alphabetical
            )
        )

        # Select labels with positive scores.
        top_labels: List[str] = [label for (label, score, _, _) in ranked if score > 0]
        if len(top_labels) > top_n:
            top_labels = top_labels[:top_n]

        return {"matches": matches, "scores": scores, "top": top_labels}

    # --------------------------------------------------------------------- #
    # Rule-based cues and finalisation
    # --------------------------------------------------------------------- #

    def _rule_based_emotions(self, text: str) -> List[str]:
        """
        Lightweight rule-based emotion cues based on speech verbs,
        adverbs/adjectives, and a few strong lexical hints.

        Returns an empty list if no rule matches.
        """
        if not text:
            return []

        t = text.lower()

        # Ordered list of (regex, [primary, secondary]) patterns.
        # Earlier rules take precedence.
        patterns: List[Tuple[re.Pattern, List[str]]] = [
            # --- Strong negative / intense / aggressive ---
            (re.compile(r"\b(shouted|yelled|screamed|bellowed)\b"), ["screaming", "furious"]),
            (re.compile(r"\b(snarled|growled|snapped)\b"), ["aggressive", "angry"]),
            (re.compile(r"\bsaid\s+angrily\b"), ["angry", "furious"]),
            (re.compile(r"\bfurious\b"), ["furious", "angry"]),

            # --- Dominant / commanding behaviour ---
            (re.compile(r"\b(commanded|ordered|barked|snapped out)\b"), ["dominant", "commanding"]),
            (re.compile(r"\b(on your knees|be a good (?:girl|boy))\b"), ["dominant", "commanding"]),

            # --- Teasing / playful / mischievous ---
            (re.compile(r"\b(teased?|taunted|smirked)\b"), ["teasing", "mischievous"]),
            (re.compile(r"\bplayful\b"), ["playful", "teasing"]),

            # --- Soft / intimate speech ---
            (re.compile(r"\bwhispered\b"), ["soft", "breathy"]),
            (re.compile(r"\bwhispered\s+softly\b"), ["soft", "intimate"]),
            (re.compile(r"\bsaid\s+softly\b"), ["soft", "tender"]),
            (re.compile(r"\bspoke\s+softly\b"), ["soft", "gentle"]),

            # --- Arousal / romantic / tipsy-high states ---
            (re.compile(r"\baroused\b|\bturned on\b|\bhorny\b"), ["aroused", "breathy"]),
            (re.compile(r"\bache(?:d)? between\b|\bheat pooled\b"), ["aroused", "overstimulated"]),
            (re.compile(r"\bkissed\b|\bcaressed\b|\btraced (?:her|his) lips\b"), ["romantic", "soft"]),
            (re.compile(r"\btipsy\b|\bbuzzed\b"), ["tipsy", "playful"]),
            (re.compile(r"\bdrunk\b|\bwasted\b"), ["drunk", "distracted"]),
            (re.compile(r"\bhigh\b|\bstoned\b"), ["high", "distracted"]),
            (re.compile(r"\bhypnotized\b|\bin a trance\b"), ["hypnotized", "soft"]),

            # --- Sad / broken / crying ---
            (re.compile(r"\bsobbed\b|\bchoked out\b|\bbroken sob\b"), ["crying", "broken"]),
            (re.compile(r"\btears?\b|\bcrying\b|\bwept\b"), ["sad", "crying"]),
            (re.compile(r"\bbroken\b|\bshattered\b"), ["broken", "sad"]),

            # --- Happy / laughing / ecstatic ---
            (re.compile(r"\becstatic\b|\borgasm\b|\bclimaxed\b"), ["ecstatic", "climactic"]),
            (re.compile(r"\blaughed\b|\bgiggled\b"), ["laughing", "happy"]),
            (re.compile(r"\bsmiled\b|\bgrinned\b"), ["happy", "playful"]),

            # --- Nervous / scared / resisting ---
            (re.compile(r"\bnervously\b|\banxious\b|\buneasy\b"), ["shaky", "scared"]),
            (re.compile(r"\bscared\b|\bterrified\b|\bafraid\b"), ["scared", "shaky"]),
            (re.compile(r"\btrembling\b|\bshaking\b"), ["shaky", "breathless"]),
            (re.compile(r"\bresisted\b|\bpulled away\b|\bshoved\b"), ["resisting", "angry"]),

            # --- Pleading / begging ---
            (re.compile(r"\bplead(?:ed|ing)?\b"), ["pleading", "soft"]),
            (re.compile(r"\bbegged\b|\bbegging\b"), ["begging", "pleading"]),
            (re.compile(r"\bplease\b"), ["pleading", "soft"]),

            # --- Jealous / distracted / embarrassed / shy ---
            (re.compile(r"\bjealous\b|\benvy\b|\benvious\b"), ["jealous", "angry"]),
            (re.compile(r"\bdistracted\b|\bmind elsewhere\b"), ["distracted", "soft"]),
            (re.compile(r"\bblushed\b|\bcheeks (?:burned|burning)\b|\bembarrassed\b"), ["embarrassed", "shy"]),
            (re.compile(r"\binnocent\b|\bwide-eyed\b"), ["innocent", "shy"]),

            # --- Physical vocal cues ---
            (re.compile(r"\bmoaned\b|\bmoaning\b"), ["moaning", "aroused"]),
            (re.compile(r"\bgroaned\b|\bgroaning\b"), ["groaning", "overstimulated"]),
            (re.compile(r"\bwhimpered\b|\bwhimpering\b"), ["whimpering", "shaky"]),
            (re.compile(r"\bpanted\b|\bpanting\b"), ["panting", "breathless"]),
            (re.compile(r"\bbreathless\b|\bout of breath\b"), ["breathless", "soft"]),

            # --- Worship / shame / worshipping ---
            (re.compile(r"\bworship(?:ped|ping)?\b"), ["worshipping", "innocent"]),
            (re.compile(r"\bashamed\b|\bhumiliated\b"), ["ashamed", "sad"]),

            # --- Overload / overstimulated / manic ---
            (re.compile(r"\boverstimulated\b|\btoo much\b"), ["overstimulated", "shaky"]),
            (re.compile(r"\bmanic\b|\bfrantic\b"), ["manic", "intense"]),

            # --- Question / curiosity (fallback for question marks) ---
            (re.compile(r"\?"), ["curious", "soft"]),

            # --- Quiet / subdued / soft baseline ---
            (re.compile(r"\bquietly\b|\bsoftly\b"), ["soft", "calm"]),

            # --- Punctuation-driven intensity (shouty lines) ---
            (re.compile(r"!+$"), ["intense", "aggressive"]),
        ]

        for pat, labels in patterns:
            if pat.search(t):
                return labels[:2]

        return []

    @staticmethod
    def _count_cues(text: str, cues: Tuple[str, ...]) -> int:
        if not text:
            return 0
        total = 0
        for cue in cues:
            if not cue:
                continue
            if cue in text:
                total += 1
        return total

    def _contextual_signals(self, text: str) -> Dict[str, int]:
        lowered = (text or "").lower()
        return {
            "arousal": self._count_cues(lowered, self.AROUSAL_CUES),
            "romance": self._count_cues(lowered, self.ROMANTIC_CUES),
            "dominance": self._count_cues(lowered, self.DOMINANCE_CUES),
            "tension": self._count_cues(lowered, self.TENSION_CUES),
        }

    def _pick_contextual_labels(self, signals: Dict[str, int]) -> List[str]:
        arousal = signals.get("arousal", 0)
        romance = signals.get("romance", 0)
        dominance = signals.get("dominance", 0)
        tension = signals.get("tension", 0)

        if arousal >= 2 or (arousal and romance):
            return ["aroused", "breathless"]
        if romance >= 2:
            return ["romantic", "soft"]
        if dominance and (arousal or tension):
            return ["dominant", "commanding"]
        if dominance >= 2:
            return ["dominant", "aggressive"]
        if tension >= 2:
            return ["intense", "shaky"]
        return []

    def _fallback_from_signals(self, signals: Dict[str, int]) -> List[str]:
        if signals.get("arousal"):
            return ["aroused", "breathless"]
        if signals.get("romance"):
            return ["romantic", "soft"]
        if signals.get("dominance"):
            return ["dominant", "commanding"]
        if signals.get("tension"):
            return ["intense", "soft"]
        return ["soft", "calm"]

    def _finalise_labels(
        self,
        labels: List[str],
        allow_fallback: bool,
        *,
        fallback: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Ensure a 2-element list of distinct, non-generic emotion labels.

        Rules:
          * If no non-neutral labels are present and `allow_fallback` is True,
            return the deterministic default ["soft", "calm"].
          * Otherwise, prefer non-neutral labels and ensure uniqueness.
          * When exactly one non-neutral label is available, pair it with a
            sensible companion based on simple mappings.
        """
        fallback_pair = list(fallback or ["soft", "calm"])
        if not fallback_pair:
            fallback_pair = ["soft", "calm"]
        while len(fallback_pair) < 2:
            fallback_pair.append("calm")

        # Normalise and filter out empties
        norm_labels: List[str] = [str(l).strip() for l in labels if str(l).strip()]
        if not norm_labels:
            return fallback_pair[:2] if allow_fallback else []

        non_neutral = [l for l in norm_labels if l.lower() != "neutral"]

        if not non_neutral and allow_fallback:
            return fallback_pair[:2]

        companion_map: Dict[str, str] = {
            "angry": "intense",
            "happy": "playful",
            "fearful": "anxious",
            "sad": "hurt",
            "warm": "gentle",
            "soft": "calm",
        }

        # Collapse to unique non-neutral labels preserving order.
        unique_non_neutral: List[str] = []
        seen: set = set()
        for l in non_neutral:
            key = l.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_non_neutral.append(l)

        if len(unique_non_neutral) >= 2:
            return unique_non_neutral[:2]

        if not unique_non_neutral:
            return fallback_pair[:2] if allow_fallback else []

        # Exactly one non-neutral label.
        base = unique_non_neutral[0]
        base_key = base.lower()
        companion = companion_map.get(base_key, "soft")
        if companion.lower() == base_key:
            # Ensure distinctness
            companion = "calm" if base_key != "calm" else "soft"
        return [base, companion]






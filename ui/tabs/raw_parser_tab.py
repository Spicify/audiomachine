from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import streamlit as st

from audio.utils import get_flat_character_voices
from parser.local_adapter import (
    LocalParserAdapterResult,
    format_lines_to_dialogue_text,
    parse_raw_prose_to_dialogue_format,
)
from parser.parser_core.pipeline import ParserPipeline

_VOICE_DATA = get_flat_character_voices()
_VOICE_NAMES_LOWER = {name.lower() for name in _VOICE_DATA.keys()}
_GENDER_OPTIONS = [
    ("Unknown", None),
    ("Male", "M"),
    ("Female", "F"),
    ("Other", "U"),
]
_STOPWORDS = {
    "the",
    "and",
    "but",
    "not",
    "she",
    "he",
    "they",
    "his",
    "her",
    "hers",
    "him",
    "their",
    "theirs",
    "we",
    "you",
    "your",
    "yours",
    "i",
    "it",
    "its",
    "a",
    "an",
    "this",
    "that",
    "those",
    "these",
    "chapter",
    "scene",
    "moment",
    "hours",
    "minutes",
    "after",
    "before",
    "once",
    "then",
    "now",
    "today",
    "tonight",
    "yesterday",
    "tomorrow",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "summer",
    "winter",
    "spring",
    "autumn",
    "fall",
    "morning",
    "evening",
    "noon",
    "midnight",
    "dawn",
    "dusk",
    "section",
    "prologue",
    "epilogue",
    "preface",
    "mom",
    "mother",
    "dad",
    "father",
    "mama",
    "papa",
    "parents",
    "brother",
    "sister",
    "aunt",
    "uncle",
    "grandma",
    "grandpa",
    "child",
    "children",
    "just",
    "no",
    "yes",
    "every",
    "like",
    "my",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "first",
    "second",
    "third",
    "only",
    "always",
    "never",
    "maybe",
    "still",
    "again",
    "already",
    "suddenly",
    "slowly",
    "quickly",
    "instantly",
    "truly",
    "honestly",
    "finally",
    "really",
    "barely",
    "nearly",
    "almost",
    "controlling",
    "control",
    "power",
    "strength",
    "soft",
    "calm",
    "gentle",
    "angry",
    "furious",
    "dominant",
    "commanding",
    "submissive",
    "fragile",
    "cold",
    "warm",
    "dark",
    "light",
    "deep",
    "shallow",
    "long",
    "short",
    "wide",
    "narrow",
    "tired",
    "restless",
    "burning",
    "aching",
    "throbbing",
    "fading",
    "hungry",
    "thirsty",
}
_GENERIC_NAME_TOKENS = {
    "mom",
    "mother",
    "dad",
    "father",
    "mama",
    "papa",
    "parents",
    "brother",
    "sister",
    "aunt",
    "uncle",
    "grandma",
    "grandpa",
    "child",
    "baby",
    "son",
    "daughter",
    "husband",
    "wife",
    "man",
    "woman",
    "boy",
    "girl",
}
_SPEECH_VERBS_PATTERN = (
    r"(?:said|asked|replied|whispered|shouted|yelled|muttered|sighed|cried|"
    r"exclaimed|whimpered|murmured|growled|laughed|sobbed|noted|added|remarked|"
    r"cursed|commanded|insisted|warned|teased|cooed|breathed|hissed|snapped|"
    r"stated|responded|answered|called|promised|pleaded|begged)"
)
_LOWERCASE_NAME_PARTICLES = {
    "de",
    "del",
    "della",
    "di",
    "la",
    "le",
    "van",
    "von",
    "da",
    "dos",
    "las",
    "los",
}
_SENTENCE_STARTERS = {
    "after",
    "before",
    "once",
    "then",
    "so",
    "but",
    "and",
    "however",
    "meanwhile",
    "suddenly",
    "eventually",
    "later",
    "when",
    "while",
    "as",
    "for",
    "because",
    "if",
    "though",
    "yet",
    "now",
    "still",
    "in",
    "on",
    "at",
    "from",
    "toward",
    "towards",
    "with",
    "without",
    "around",
    "between",
    "inside",
    "outside",
}
_COMMON_CAPITALIZED_WORDS = {
    "After",
    "Before",
    "Once",
    "Then",
    "So",
    "But",
    "And",
    "However",
    "Meanwhile",
    "Suddenly",
    "Eventually",
    "Later",
    "When",
    "While",
    "As",
    "For",
    "Because",
    "If",
    "Though",
    "Yet",
    "Now",
    "Still",
    "In",
    "On",
    "At",
    "From",
    "Toward",
    "Towards",
    "With",
    "Without",
    "Around",
    "Between",
    "Inside",
    "Outside",
    "Chapter",
    "Section",
    "Prologue",
    "Epilogue",
    "Preface",
    "Morning",
    "Evening",
    "Night",
    "Midnight",
    "Noon",
}
_STATE_DEFAULTS: Dict[str, Any] = {
    "raw_parser_lines": [],
    "raw_parser_include_narration": True,
    "raw_parser_formatted_text": "",
    "raw_parser_choices": {},
    "raw_parser_custom_names": {},
    "raw_parser_known_chars": [],
    "raw_finalized": False,
    "raw_parser_cast": [],
    "raw_parser_cast_detected_hash": None,
}
_MAX_AUTO_CHARACTERS = 12
_DETECTION_PIPELINE: Optional[ParserPipeline] = None


def create_raw_parser_tab(get_known_characters_callable):
    _ensure_state_defaults()

    st.markdown("### Raw Text -> Dialogue Parser")
    st.markdown(
        "Paste raw prose below. The local parser will detect dialogue, speakers, "
        "and inline emotions with no API calls."
    )

    col1, col2 = st.columns(2)
    with col1:
        include_narration = st.checkbox(
            "Include Narration as [Narrator]",
            value=st.session_state.get("raw_parser_include_narration", True),
            key="raw_inc_narr",
        )
    with col2:
        st.caption("Narration lines can be excluded by unchecking the box above.")

    raw_text = st.text_area(
        "Raw Prose:",
        height=280,
        placeholder=(
            "Example:\n"
            "Dante's eyes narrowed. \"The security system is down,\" he whispered. "
            "\"This is our chance.\"\n"
            "Luca sighed. \"I still don't like this plan, Dante.\"\n"
            "\"Relax, tesoro. What could go wrong?\" Rafael said mischievously.\n"
            "Nikolai said coldly, \"Everything. That's what experience teaches you.\"\n"
            "There was a sharp gasp as the door slammed."
        ),
        key="raw_parser_input",
    )

    _render_character_configuration(raw_text)
    _render_parse_controls(
        raw_text,
        include_narration=include_narration,
        get_known_characters_callable=get_known_characters_callable,
    )

    lines: List[Dict[str, Any]] = st.session_state["raw_parser_lines"]
    if not lines:
        return

    _render_parse_summary(lines)
    _render_ambiguity_controls(lines)
    _render_finalize_and_actions()


def _ensure_state_defaults() -> None:
    for key, default in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _render_parse_controls(
    raw_text: str,
    *,
    include_narration: bool,
    get_known_characters_callable,
) -> None:
    col1, col2 = st.columns([3, 1])
    with col1:
        triggered = st.button(
            "Convert Raw -> Dialogue",
            type="primary",
            use_container_width=True,
        )
    with col2:
        if st.button(
            "Reset Parsed Output",
            type="secondary",
            use_container_width=True,
        ):
            _reset_parser_state()
            st.success("Parser state reset.")
            st.stop()

    if not triggered:
        return

    if not raw_text or not raw_text.strip():
        st.error("Please paste some raw prose first.")
        return

    user_cast = _collect_enabled_cast_entries()

    with st.spinner("Parsing locally..."):
        known_chars = _safe_get_known_characters(get_known_characters_callable)
        try:
            result: LocalParserAdapterResult = parse_raw_prose_to_dialogue_format(
                raw_text,
                include_narration=include_narration,
                user_characters=None,
                user_cast=user_cast,
                strict_mode=bool(user_cast),
                aggressive_mode=True,
                auto_assign_dialogue=True,
            )
        except Exception as exc:
            st.error(f"Parser error: {exc}")
            return

    st.session_state["raw_parser_lines"] = result.lines
    st.session_state["raw_parser_formatted_text"] = result.formatted_text
    st.session_state["raw_parser_include_narration"] = include_narration
    st.session_state["raw_parser_choices"] = {}
    st.session_state["raw_parser_custom_names"] = {}
    st.session_state["raw_parser_known_chars"] = known_chars
    st.session_state["raw_finalized"] = False

    if not _has_ambiguities(result.lines):
        _apply_ambiguity_choices(auto_finalize=True)
    else:
        st.success("Parsing complete. Resolve ambiguous speakers below before finalizing.")


def _render_parse_summary(lines: List[Dict[str, Any]]) -> None:
    total = len(lines)
    narration = sum(1 for line in lines if line.get("type") == "narration")
    ambiguous = sum(1 for line in lines if line.get("character") == "Ambiguous")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Lines", total)
    col2.metric("Narration Lines", narration)
    col3.metric("Ambiguous Lines", ambiguous)

    with st.expander("Preview Parsed Lines", expanded=False):
        previews = [_format_line_preview(line) for line in lines[:40]]
        st.code("\n".join(previews) if previews else "No lines parsed yet.", language="markdown")


def _render_ambiguity_controls(lines: List[Dict[str, Any]]) -> None:
    ambiguous_lines = [line for line in lines if line.get("character") == "Ambiguous"]
    if not ambiguous_lines:
        st.success("No ambiguous speakers detected.")
        return

    st.markdown("### Resolve Ambiguous Speakers")
    st.caption("Choose the best speaker for each ambiguous line. Add custom names if needed.")

    known_chars: List[str] = st.session_state.get("raw_parser_known_chars", [])
    add_label = "Add new character"
    for idx, line in enumerate(ambiguous_lines, start=1):
        line_id = line["id"]
        text_preview = line.get("text", "")
        candidates = _build_candidate_options(line, known_chars, add_label=add_label)
        select_key = f"amb_select_{line_id}"
        current_choice = st.session_state["raw_parser_choices"].get(line_id)
        default_index = 0
        if current_choice and current_choice in candidates:
            default_index = candidates.index(current_choice)
        elif not candidates:
            candidates = [add_label]

        selection = st.selectbox(
            f"Line {idx}: {text_preview[:140]}{'...' if len(text_preview) > 140 else ''}",
            options=candidates,
            index=min(default_index, len(candidates) - 1),
            key=select_key,
        )

        if selection == add_label:
            custom_key = f"amb_custom_{line_id}"
            custom_value = st.text_input(
                "Custom character name:",
                value=st.session_state["raw_parser_custom_names"].get(line_id, ""),
                key=custom_key,
            ).strip()
            if custom_value:
                st.session_state["raw_parser_custom_names"][line_id] = custom_value
                st.session_state["raw_parser_choices"][line_id] = custom_value
        else:
            st.session_state["raw_parser_choices"][line_id] = selection


def _render_finalize_and_actions() -> None:
    if not st.session_state.get("raw_finalized", False):
        if st.button(
            "Finalize Parsed Output",
            type="primary",
            use_container_width=True,
        ):
            _apply_ambiguity_choices(auto_finalize=False)
        return

    formatted = st.session_state.get("raw_parser_formatted_text", "")
    if not formatted:
        st.warning("Parsed text is empty after finalization.")
        return

    st.success("Parsed output ready. Review below or send to the main generator.")
    st.code(formatted, language="markdown")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(
            "Send to Main Generator",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.dialogue_text = formatted
            for key in (
                "paste_text_analysis",
                "paste_formatted_dialogue",
                "paste_parsed_dialogues",
                "paste_voice_assignments",
            ):
                st.session_state.pop(key, None)
            st.session_state.current_tab = "main"
            st.success("Parsed output sent to Main Generator.")
    with col2:
        if st.button(
            "Clear Parsed Output",
            type="secondary",
            use_container_width=True,
        ):
            _reset_parser_state()


def _apply_ambiguity_choices(*, auto_finalize: bool) -> None:
    lines: List[Dict[str, Any]] = st.session_state["raw_parser_lines"]
    choices: Dict[str, str] = st.session_state["raw_parser_choices"]
    updated: List[Dict[str, Any]] = []
    for line in lines:
        new_line = dict(line)
        if line.get("character") == "Ambiguous":
            choice = choices.get(line["id"]) or st.session_state["raw_parser_custom_names"].get(
                line["id"],
            )
            if choice:
                new_line["character"] = choice
        updated.append(new_line)

    include_narration = st.session_state.get("raw_parser_include_narration", True)
    formatted = format_lines_to_dialogue_text(
        updated,
        include_narration=include_narration,
    )

    st.session_state["raw_parser_lines"] = updated
    st.session_state["raw_parser_formatted_text"] = formatted
    st.session_state["raw_finalized"] = True

    if auto_finalize:
        st.success("No ambiguities detected. Output finalized automatically.")
    else:
        st.success("Ambiguity selections applied. Parsed output finalized.")


def _reset_parser_state() -> None:
    for key, default in _STATE_DEFAULTS.items():
        st.session_state[key] = default


def _safe_get_known_characters(get_known_characters_callable) -> List[str]:
    if not callable(get_known_characters_callable):
        return []
    try:
        known = get_known_characters_callable() or []
        return sorted({str(name) for name in known if name})
    except Exception:
        return []


def _format_line_preview(line: Dict[str, Any]) -> str:
    character = line.get("character") or "Narrator"
    emotions = line.get("emotions") or ["soft", "calm"]
    text = line.get("text", "")
    emo_text = "".join(f"({emo})" for emo in emotions if emo)
    return f"[{character}] {emo_text}: {text}"


def _build_candidate_options(line: Dict[str, Any], known_chars: List[str], *, add_label: str) -> List[str]:
    options = [c for c in (line.get("candidates") or []) if c]
    if not options:
        options = known_chars[:20]
    if add_label not in options:
        options = options + [add_label]
    return options


def _has_ambiguities(lines: List[Dict[str, Any]]) -> bool:
    return any(line.get("character") == "Ambiguous" for line in lines)


def _render_character_configuration(raw_text: str) -> None:
    st.markdown("### Characters & Pronoun Hints")
    st.caption(
        "Configure the cast to guide pronoun resolution. This mirrors the CLI flow in "
        "parser/tools/interactive_parse.py."
    )

    text_has_content = bool(raw_text and raw_text.strip())
    if text_has_content and not st.session_state.get("raw_parser_cast"):
        _populate_cast_from_text(raw_text, force=False)

    detect_col, add_col = st.columns([1, 1])
    with detect_col:
        detect_clicked = st.button(
            "Detect from text",
            key="raw_cast_detect_btn",
            disabled=not text_has_content,
            help="Auto-detect characters from the current prose.",
        )
    with add_col:
        add_clicked = st.button(
            "Add character",
            key="raw_cast_add_btn",
            help="Add a blank row to configure another character.",
        )

    if detect_clicked:
        _populate_cast_from_text(raw_text, force=True)

    cast: List[Dict[str, Any]] = list(st.session_state.get("raw_parser_cast", []))
    if add_clicked:
        cast.append({"name": "", "gender": None, "enabled": True})
        st.session_state["raw_parser_cast"] = cast
        cast = list(st.session_state.get("raw_parser_cast", []))

    if not cast:
        if text_has_content:
            st.info("No characters detected yet. Add them manually above.")
        else:
            st.info("Paste the story text or add characters manually.")
        return

    updated_rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(cast):
        cols = st.columns([3, 2, 1, 0.7])
        with cols[0]:
            name = st.text_input(
                f"Character {idx + 1}",
                value=entry.get("name", ""),
                key=f"raw_cast_name_{idx}",
            )
        with cols[1]:
            gender_index = _gender_code_to_index(entry.get("gender"))
            gender_label = st.selectbox(
                f"Gender {idx + 1}",
                options=[label for label, _ in _GENDER_OPTIONS],
                index=gender_index,
                key=f"raw_cast_gender_{idx}",
            )
        with cols[2]:
            enabled = st.checkbox(
                "Use",
                value=entry.get("enabled", True),
                key=f"raw_cast_enabled_{idx}",
            )
        with cols[3]:
            remove = st.button(
                "Remove",
                key=f"raw_cast_remove_{idx}",
                help="Remove this character row.",
            )

        updated_rows.append(
            {
                "name": name.strip(),
                "gender": _gender_label_to_code(gender_label),
                "enabled": enabled,
                "_remove": remove,
            }
        )

    final_rows: List[Dict[str, Any]] = []
    for row in updated_rows:
        if row.pop("_remove", False):
            continue
        final_rows.append(row)

    st.session_state["raw_parser_cast"] = final_rows
    st.caption("Tip: Ensure each listed character has the correct gender to improve pronoun resolution.")


def _collect_enabled_cast_entries() -> List[Dict[str, Any]]:
    cast: List[Dict[str, Any]] = st.session_state.get("raw_parser_cast", [])
    entries: List[Dict[str, Any]] = []
    for entry in cast:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        if not entry.get("enabled", True):
            continue
        entries.append(
            {
                "name": name,
                "gender": entry.get("gender"),
                "enabled": True,
                "exists_in_config": name.lower() in _VOICE_NAMES_LOWER,
            }
        )
    return entries


def _populate_cast_from_text(raw_text: str, *, force: bool) -> None:
    text_clean = (raw_text or "").strip()
    if not text_clean:
        return
    text_hash = hashlib.sha1(text_clean.encode("utf-8")).hexdigest()
    existing_hash = st.session_state.get("raw_parser_cast_detected_hash")
    cast_present = bool(st.session_state.get("raw_parser_cast"))
    if not force and cast_present:
        return
    if not force and existing_hash == text_hash:
        return

    detected = _auto_detect_characters(text_clean)
    st.session_state["raw_parser_cast_detected_hash"] = text_hash
    if not detected:
        if force:
            st.warning("No characters could be detected automatically. Add them manually.")
        return

    st.session_state["raw_parser_cast"] = [
        {
            "name": name,
            "gender": _guess_gender_from_config(name),
            "enabled": True,
        }
        for name in detected
    ]


def _get_detection_pipeline() -> Optional[ParserPipeline]:
    global _DETECTION_PIPELINE
    if _DETECTION_PIPELINE is not None:
        return _DETECTION_PIPELINE
    try:
        _DETECTION_PIPELINE = ParserPipeline(
            character_config="parser/configs/character_voices.json",
            emotion_config="parser/configs/emotions.json",
        )
    except Exception:
        _DETECTION_PIPELINE = None
    return _DETECTION_PIPELINE


def _normalize_candidate_key(name: str) -> str:
    normalized = re.sub(r"[^a-z]+", " ", name.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _looks_like_character_name(name: str) -> bool:
    if not name:
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", name.strip())
    if not tokens:
        return False
    if len(tokens) > 4:
        return False
    has_valid = False
    for token in tokens:
        lowered = token.lower()
        if lowered in _GENERIC_NAME_TOKENS:
            return False
        if lowered in _STOPWORDS:
            return False
        if token[0].islower() and lowered not in _LOWERCASE_NAME_PARTICLES:
            return False
        has_valid = True
    return has_valid


def _has_speaker_context(name: str, text: str) -> bool:
    if not name or not text:
        return False
    pattern = _SPEECH_VERBS_PATTERN
    escaped = re.escape(name)
    regexes = [
        rf"\b{escaped}\b\s+{pattern}\b",
        rf"{pattern}\s+\b{escaped}\b",
        rf"\b{escaped}'s\b",
        rf"\b{escaped},\s+{pattern}\b",
    ]
    for expr in regexes:
        if re.search(expr, text, flags=re.IGNORECASE):
            return True
    return False


def _should_skip_capitalized_token(token: str, text: str, start_idx: int) -> bool:
    if not token:
        return True
    if token in _COMMON_CAPITALIZED_WORDS:
        return True
    normalized = token.replace("’", "'").strip()
    lower_norm = normalized.lower()
    if lower_norm in _STOPWORDS:
        return True
    if lower_norm.rstrip("'s") in _STOPWORDS:
        return True

    idx = start_idx
    while idx > 0 and text[idx - 1].isspace():
        idx -= 1
    boundary = idx == 0 or text[idx - 1] in ".!?\"'\n\r;:()[]"
    if boundary and lower_norm in _SENTENCE_STARTERS:
        return True
    return False


def _heuristic_name_candidates(text: str) -> List[str]:
    names: List[str] = []
    seen_keys: set[str] = set()
    lower_text = text.lower()

    canonical_hits: List[tuple[str, int]] = []
    for display_name in _VOICE_DATA.keys():
        if not _looks_like_character_name(display_name):
            continue
        key = display_name.lower()
        count = lower_text.count(key)
        if count:
            canonical_hits.append((display_name, count))

    canonical_hits.sort(key=lambda pair: (-pair[1], pair[0]))
    for name, _ in canonical_hits:
        key = _normalize_candidate_key(name)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        names.append(name)
        if len(names) >= _MAX_AUTO_CHARACTERS:
            return names

    counts = Counter()
    for match in re.finditer(r"\b[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,2}\b", text):
        token = match.group(0).strip()
        if not token or not _looks_like_character_name(token):
            continue
        if _should_skip_capitalized_token(token, text, match.start()):
            continue
        counts[token] += 1

    context_map = {name: _has_speaker_context(name, text) for name in counts}

    for name, freq in counts.most_common(_MAX_AUTO_CHARACTERS * 2):
        key = _normalize_candidate_key(name)
        if not key or key in seen_keys:
            continue
        has_context = context_map.get(name, False)
        if freq < 2 and not has_context and name.lower() not in _VOICE_NAMES_LOWER:
            continue
        seen_keys.add(key)
        names.append(name)
        if len(names) >= _MAX_AUTO_CHARACTERS:
            break

    return names


def _auto_detect_characters(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    usable: List[str] = []
    seen_keys: set[str] = set()

    def append_candidate(raw_name: str) -> None:
        display = (raw_name or "").strip()
        if not display or not _looks_like_character_name(display):
            return
        key = _normalize_candidate_key(display)
        if not key or key in seen_keys:
            return
        seen_keys.add(key)
        usable.append(display)

    pipeline = _get_detection_pipeline()
    if pipeline:
        try:
            pipeline.reset_state()
            parsed_lines = pipeline.parse(text)
        except Exception:
            parsed_lines = []

        counts: Counter[str] = Counter()
        for line in parsed_lines:
            character = (line.get("character") or "").strip()
            if not character or character in {"Narrator", "Ambiguous"}:
                continue
            counts[character] += 1

        for name, _ in counts.most_common(_MAX_AUTO_CHARACTERS):
            append_candidate(name)
            if len(usable) >= _MAX_AUTO_CHARACTERS:
                return usable

    for fallback in _heuristic_name_candidates(text):
        append_candidate(fallback)
        if len(usable) >= _MAX_AUTO_CHARACTERS:
            break

    return usable


def _guess_gender_from_config(name: str) -> Optional[str]:
    if not name:
        return None
    info = _VOICE_DATA.get(name)
    if not info:
        info = next((data for key, data in _VOICE_DATA.items() if key.lower() == name.lower()), None)
    if not info:
        return None
    gender = str(info.get("gender") or "").strip().upper()
    if gender in {"M", "F", "U"}:
        return gender
    return None


def _gender_code_to_index(code: Optional[str]) -> int:
    normalized = (code or None)
    for idx, (_, opt_code) in enumerate(_GENDER_OPTIONS):
        if (opt_code or None) == (normalized or None):
            return idx
    return 0


def _gender_label_to_code(label: str) -> Optional[str]:
    for option_label, code in _GENDER_OPTIONS:
        if option_label == label:
            return code
    return None

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from parser.openai_character_detector import detect_characters_via_openai
from parser.openai_character_detector import detect_characters_via_openai
from parser.parser_core.pipeline import ParserPipeline

_PIPELINE: Optional[ParserPipeline] = None
_DEFAULT_EMOTIONS = ["soft", "calm"]
_HISTORY_DIR = Path("logs/parser_runs")


@dataclass
class LocalParserAdapterResult:
    lines: List[Dict[str, Any]]
    formatted_text: str
    include_narration: bool
    has_ambiguity: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def _get_pipeline() -> ParserPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = ParserPipeline(
            character_config="parser/configs/character_voices.json",
            emotion_config="parser/configs/emotions.json",
        )
    return _PIPELINE


def _sanitize_user_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s+", " ", cleaned)
    match = re.match(r"(.+?)[\s\-,:]+(?:male|female|man|woman)\s*$", cleaned, flags=re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    return cleaned


def _reset_detector(detector: Any) -> None:
    try:
        detector.set_user_characters([])
    except Exception:
        pass
    try:
        detector.set_preferred_characters([])
    except Exception:
        pass
    detector.enable_strict_user_mode(False)
    detector.user_aggressive_mode = True
    detector.user_auto_assign_dialogue = True


def _openai_gender_to_code(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    value = str(label).strip().lower()
    if value == "male":
        return "M"
    if value == "female":
        return "F"
    if value == "nonbinary":
        return "U"
    return None


def _prepare_user_entries(
    user_cast: Optional[Sequence[Dict[str, Any]]], detector: Any
) -> Tuple[List[Tuple[str, bool, Optional[str]]], List[str]]:
    entries: List[Tuple[str, bool, Optional[str]]] = []
    new_names: List[str] = []
    if not user_cast:
        return entries, new_names

    known_config_names = {k for k in getattr(detector, "characters", {}).keys()}

    for entry in user_cast:
        if not entry:
            continue
        name = _sanitize_user_name(entry.get("name") or "")
        if not name:
            continue
        gender = entry.get("gender")
        exists_flag = entry.get("exists_in_config")
        if exists_flag is None:
            exists = name.lower() in known_config_names
        else:
            exists = bool(exists_flag)
        entries.append((name, exists, gender))
        if not exists:
            new_names.append(name)
    return entries, new_names


def _assign_line_ids(lines: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for idx, original in enumerate(lines, start=1):
        line = dict(original)
        line.setdefault("id", f"line_{idx}")
        if "candidates" in line and line["candidates"] is None:
            line["candidates"] = []
        result.append(line)
    return result


def _filter_lines(
    lines: Iterable[Dict[str, Any]], include_narration: bool
) -> List[Dict[str, Any]]:
    if include_narration:
        return list(lines)
    return [line for line in lines if line.get("type") != "narration"]


def _fallback_lines(text: str) -> List[Dict[str, Any]]:
    clean = (text or "").strip()
    if not clean:
        clean = ""
    return [
        {
            "id": "line_1",
            "character": "Narrator",
            "type": "narration",
            "emotions": list(_DEFAULT_EMOTIONS),
            "text": clean,
        }
    ]


def parse_raw_prose_to_dialogue_format(
    text: str,
    *,
    include_narration: bool = True,
    user_characters: Optional[Iterable[str]] = None,
    user_cast: Optional[Sequence[Dict[str, Any]]] = None,
    strict_mode: bool = False,
    aggressive_mode: bool = True,
    auto_assign_dialogue: bool = True,
) -> LocalParserAdapterResult:
    pipeline = _get_pipeline()
    detector = pipeline.detector
    _reset_detector(detector)
    pipeline.reset_state()

    generated_cast = list(user_cast or [])
    auto_detected_cast: List[Dict[str, str]] = []
    if not generated_cast:
        try:
            auto_detected_cast = detect_characters_via_openai(text or "")
        except Exception:
            auto_detected_cast = []
        if auto_detected_cast:
            generated_cast = [
                {
                    "name": entry["name"],
                    "gender": _openai_gender_to_code(entry.get("gender")),
                    "enabled": True,
                    "exists_in_config": None,
                }
                for entry in auto_detected_cast
            ]

    if user_characters:
        detector.set_preferred_characters(user_characters)

    entries, new_names = _prepare_user_entries(generated_cast, detector)
    if entries:
        detector.set_user_characters(entries)
        detector.enable_strict_user_mode(bool(strict_mode))
        detector.user_aggressive_mode = bool(aggressive_mode)
        detector.user_auto_assign_dialogue = bool(auto_assign_dialogue)
        if new_names:
            detector.inject_user_characters(new_names)

    try:
        raw_lines = pipeline.parse(text or "")
    except Exception:
        raw_lines = _fallback_lines(text or "")
    else:
        raw_lines = _assign_line_ids(raw_lines)

    filtered_lines = _filter_lines(raw_lines, include_narration)
    formatted_text = format_lines_to_dialogue_text(
        filtered_lines, include_narration=True
    )
    has_ambiguity = any(line.get("character") == "Ambiguous" for line in filtered_lines)

    metadata = {
        "include_narration": include_narration,
        "raw_line_count": len(raw_lines),
        "line_count": len(filtered_lines),
    }
    if auto_detected_cast:
        metadata["auto_detected_characters"] = list(auto_detected_cast)
    history_path = _persist_run_history(
        text=text,
        user_cast=user_cast,
        formatted_text=formatted_text,
        lines=filtered_lines,
    )
    if history_path:
        metadata["history_file"] = str(history_path)

    return LocalParserAdapterResult(
        lines=filtered_lines,
        formatted_text=formatted_text,
        include_narration=include_narration,
        has_ambiguity=has_ambiguity,
        metadata=metadata,
    )


def format_lines_to_dialogue_text(
    lines: Iterable[Dict[str, Any]],
    *,
    include_narration: bool = True,
) -> str:
    formatted: List[str] = []
    for line in lines:
        if not include_narration and line.get("type") == "narration":
            continue
        character = line.get("character") or "Narrator"
        emotions = line.get("emotions") or _DEFAULT_EMOTIONS
        text = line.get("text") or ""
        emo = "".join(f"({e})" for e in emotions if e)
        formatted.append(f"[{character}] {emo}: {text}".rstrip())
    return "\n".join(formatted)


def _persist_run_history(
    *,
    text: str,
    user_cast: Optional[Sequence[Dict[str, Any]]],
    formatted_text: str,
    lines: Sequence[Dict[str, Any]],
) -> Optional[Path]:
    try:
        _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "user_input_story": text or "",
        "user_cast": [
            {
                "name": (entry or {}).get("name"),
                "gender": (entry or {}).get("gender"),
                "exists_in_config": (entry or {}).get("exists_in_config"),
            }
            for entry in (user_cast or [])
        ],
        "parsed_output_text": formatted_text,
        "parsed_lines": list(lines),
    }
    run_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    history_path = _HISTORY_DIR / f"parser_run_{run_id}.json"
    try:
        history_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        return None
    return history_path

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from parser.parser_core.pipeline import ParserPipeline

_PIPELINE: Optional[ParserPipeline] = None
_DEFAULT_EMOTIONS = ["soft", "calm"]


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


def _prepare_user_entries(
    user_cast: Optional[Sequence[Dict[str, Any]]],
) -> Tuple[List[Tuple[str, bool, Optional[str]]], List[str]]:
    entries: List[Tuple[str, bool, Optional[str]]] = []
    new_names: List[str] = []
    if not user_cast:
        return entries, new_names

    for entry in user_cast:
        if not entry:
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        gender = entry.get("gender")
        exists_flag = entry.get("exists_in_config")
        exists = bool(exists_flag) if exists_flag is not None else True
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

    if user_characters:
        detector.set_preferred_characters(user_characters)

    entries, new_names = _prepare_user_entries(user_cast)
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

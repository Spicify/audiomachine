# parsers/openai_parser.py
import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from openai import OpenAI
from collections import defaultdict
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import datetime
import streamlit as st
from audio.utils import get_flat_emotion_tags, get_flat_character_voices
from settings import OPENAI_API_KEY


# Load configs
_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

with open(_CONFIGS_DIR / "emotion_tags.json", "r", encoding="utf-8") as _f:
    _EMOTION_TAGS_JSON = json.load(_f)


with open(_CONFIGS_DIR / "verb_to_emotion.json", "r", encoding="utf-8") as _f:
    VERB_TO_EMOTION = json.load(_f)

with open(_CONFIGS_DIR / "adverb_to_emotion.json", "r", encoding="utf-8") as _f:
    ADVERB_TO_EMOTION = json.load(_f)

# Stopwords to prevent junk speaker labels
STOPWORDS = {
    "I", "You", "He", "She", "It", "We", "They", "My", "Mine", "Your", "Yours", "His", "Hers", "Our", "Ours",
    "Their", "Theirs", "This", "That", "These", "Those", "Here", "There", "Then", "When", "While", "Where",
    "Which", "Who", "Whom", "Because", "But", "And", "Or", "If", "So", "Not", "No", "Yes", "OK", "Okay",
    "Please", "Professor", "The", "A", "An", "On", "In", "At", "Of", "By", "For", "With", "As", "To", "From"
}

# Build allowed emotion set (flat) from all categories in emotion_tags.json
ALLOWED_EMOTIONS = set()
for _category_dict in _EMOTION_TAGS_JSON.values():
    # category dict maps tag -> "[tag]"
    for _tag in _category_dict.keys():
        ALLOWED_EMOTIONS.add(_tag)

# FX removed


def validate_line(line):
    mapped = []
    for e in line.get("emotions", []):
        e_norm = (e or "").strip().lower()
        if not e_norm:
            continue
        if e_norm in ALLOWED_EMOTIONS:
            mapped.append(e_norm)
            continue
        if e_norm in VERB_TO_EMOTION:
            mapped_val = (VERB_TO_EMOTION[e_norm] or "").strip().lower()
            if mapped_val in ALLOWED_EMOTIONS:
                mapped.append(mapped_val)
                continue
        if e_norm in ADVERB_TO_EMOTION:
            mapped_val = (ADVERB_TO_EMOTION[e_norm] or "").strip().lower()
            if mapped_val in ALLOWED_EMOTIONS:
                mapped.append(mapped_val)
    if not mapped:
        mapped = ["calm"]  # fallback default
    line["emotions"] = mapped[:2]  # max 2

    return line


load_dotenv()


@dataclass
class RawParseResult:
    formatted_text: str
    dialogues: List[Dict]
    stats: Dict[str, int]
    ambiguities: List[Dict] = field(default_factory=list)


@dataclass
class ParsedDialogue:
    character: str
    text: str
    emotions: List[str]
    line_number: int
    original_line: str


@dataclass
class ParseAnalysis:
    characters_found: Dict[str, int]
    emotions_found: Dict[str, int]
    unsupported_characters: List[str]
    unsupported_emotions: List[str]
    total_lines: int
    dialogue_lines: int


@dataclass
class ParserState:
    known_characters: Set[str] = field(default_factory=set)
    last_speaker: Optional[str] = None
    aliases: Dict[str, str] = field(default_factory=dict)  # alias -> canonical

    def canonicalize(self, name: str) -> str:
        key = (name or "").strip()
        return self.aliases.get(key, key)


class OpenAIParser:
    def __init__(self, model: str = "gpt-5-mini", include_narration: bool = True, max_chars_per_chunk: int = 8000, debug_save: bool = False):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.include_narration = include_narration
        self.max_chars_per_chunk = max_chars_per_chunk
        self.debug_save = debug_save
        self.emotion_tags = get_flat_emotion_tags()
        self.default_known_characters = set(get_flat_character_voices().keys())

    def _build_system_prompt(self, state: ParserState) -> str:
        allowed_emotions_list = ", ".join(sorted(ALLOWED_EMOTIONS))
        known_chars_snippet = ", ".join(sorted(list(state.known_characters)[
                                        :50])) if state.known_characters else "(none)"
        last_speaker = state.last_speaker or "(none)"
        lines = [
            "You are a strict audiobook dialogue parser.",
            "Output MUST be JSON Lines (JSONL), one object per line, with EXACT keys: character (string), emotions (array of 1–2 strings), text (string).",
            "Do not output anything other than JSONL. No commentary, no blank lines.",
            "",
            "Character attribution rules:",
            "- Infer the speaker ONLY if the text clearly attributes it (explicit names, clear pronouns tied to a recent speaker, or direct dialogue tags like 'said Aria').",
            "- If speaker identity is even slightly uncertain → use character 'Ambiguous'.",
            "- For 'Ambiguous', always include a `candidates` array of 2–5 possible speakers (known characters, role placeholders, or 'Unknown').",
            "- Do NOT default to the last speaker unless the pronoun is explicitly tied and unambiguous.",
            "- Do NOT invent new names. If unsupported, use 'Narrator' (for description) or 'Ambiguous' (for unclear dialogue).",
            "- Do not use junk tokens like 'My', 'When', 'And', 'If', etc. as characters.",
            "",
            "Emotion rules:",
            "- Emotions must be 1–2 tags from ALLOWED_EMOTIONS. If none applies, use 'calm'.",
            "",
            "Formatting rules:",
            "- Each JSON object must contain: character (string), emotions (array), text (string).",
            "- No extra keys except `candidates` when character == 'Ambiguous'.",
            "- No trailing commas. One JSON object per line.",
            f"ALLOWED_EMOTIONS: {allowed_emotions_list}",
            f"Known characters so far: {known_chars_snippet}",
            f"Last speaker (may be used only if explicitly clear): {last_speaker}",
        ]
        if self.include_narration:
            lines.append(
                "Include narration as 'Narrator' only for non-spoken descriptive text.")
        else:
            lines.append(
                "Do not include narration lines; only output spoken dialogue.")
        # Example JSONL lines
        lines.extend([
            '{"character": "Brad", "emotions": ["angry"], "text": "Get up!"}',
            '{"character": "Zara", "emotions": ["calm"], "text": "Hello."}',
            '{"character": "Narrator", "emotions": [], "text": "The room fell silent."}',
            '{"character": "Ambiguous", "emotions": [], "candidates": ["Aria Amato", "Luca Moretti"], "text": "You two should keep your voices down."}',
        ])
        return "\n".join(lines)

    def _build_user_prompt(self, text: str, prev_summary: Optional[str]) -> str:
        parts = []
        if prev_summary:
            parts.append(f"Previous chunk summary: {prev_summary}")
        parts.append("Text to parse:")
        parts.append(text)
        return "\n".join(parts)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return (response.output_text or "").strip()

    def _save_debug_output(self, raw_text: str, suffix: str = "") -> None:
        if not self.debug_save:
            return
        try:
            Path("debug_outputs").mkdir(exist_ok=True)
            ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            name = f"openai_parser_{ts}{suffix}.txt"
            with open(Path("debug_outputs") / name, "w", encoding="utf-8") as f:
                f.write(raw_text)
        except Exception:
            pass

    def _split_into_chunks(self, text: str, max_chars: int) -> List[str]:
        paragraphs = [p.strip() for p in re.split(
            r"\n{2,}", text) if p and p.strip()]
        if not paragraphs:
            return [text.strip()] if text.strip() else []
        chunks: List[str] = []
        current = ""
        for p in paragraphs:
            candidate = current + ("\n\n" if current else "") + p
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(p) <= max_chars:
                    current = p
                else:
                    start = 0
                    while start < len(p):
                        part = p[start:start + max_chars]
                        chunks.append(part)
                        start += max_chars
                    current = ""
        if current:
            chunks.append(current)
        return chunks

    def _parse_jsonl(self, raw_output: str) -> List[Dict[str, Any]]:
        lines = [ln.strip() for ln in raw_output.split("\n") if ln.strip()]
        objs: List[Dict[str, Any]] = []
        for ln in lines:
            if not (ln.startswith("{") and ln.endswith("}")):
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    objs.append(obj)
            except Exception:
                continue
        return objs

    def _validate_and_normalize(self, items: List[Dict[str, Any]], state: ParserState) -> Tuple[List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        result: List[Dict[str, Any]] = []
        for it in items:
            character = (it.get("character") or "").strip()
            text = (it.get("text") or "").strip()
            emotions = it.get("emotions") or []
            if not character or not text or not isinstance(emotions, list):
                warnings.append(
                    "Malformed line missing required fields; dropped.")
                continue
            interim = {"emotions": emotions}
            interim = validate_line(interim)
            emotions = interim["emotions"]
            character = state.canonicalize(character)
            entry: Dict[str, Any] = {
                "type": "speech",
                "character": character,
                "text": text,
                "emotions": emotions,
            }
            if character.lower() == "ambiguous":
                # Preserve candidates if present; do not update speaker continuity
                cand_list = it.get("candidates") or []
                if isinstance(cand_list, list):
                    canonical_cands: List[str] = []
                    seen: Set[str] = set()
                    for c in cand_list:
                        c_norm = state.canonicalize((c or "").strip())
                        if not c_norm or c_norm in seen:
                            continue
                        seen.add(c_norm)
                        canonical_cands.append(c_norm)
                    if canonical_cands:
                        entry["candidates"] = canonical_cands
            else:
                # Keep new/unseen characters as-is; add to known set for continuity
                if character.lower() != "narrator":
                    state.known_characters.add(character)
                    state.last_speaker = character
            result.append(entry)
        return result, warnings

    def convert(self, raw_text: str) -> RawParseResult:
        if not raw_text.strip():
            return RawParseResult(formatted_text="", dialogues=[], stats={}, ambiguities=[])

        chunks = self._split_into_chunks(raw_text, self.max_chars_per_chunk)
        state = ParserState(known_characters=set(
            self.default_known_characters))
        all_dialogues: List[Dict[str, Any]] = []
        all_warnings: List[str] = []
        prev_summary: Optional[str] = None
        ambiguities: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            system_prompt = self._build_system_prompt(state)
            user_prompt = self._build_user_prompt(chunk, prev_summary)
            try:
                raw_output = self._call_openai(system_prompt, user_prompt)
            except Exception as e:
                try:
                    st.error(
                        f"Parser API error on chunk {idx+1}/{len(chunks)}: {e}")
                except Exception:
                    pass
                raise

            if self.debug_save:
                self._save_debug_output(raw_output, suffix=f"_chunk{idx+1}")

            items = self._parse_jsonl(raw_output)
            if not items:
                # Retry once with stricter reminder
                raw_output = self._call_openai(
                    system_prompt + "\nIMPORTANT: Output JSONL ONLY.", user_prompt)
                if self.debug_save:
                    self._save_debug_output(
                        raw_output, suffix=f"_retry_chunk{idx+1}")
                items = self._parse_jsonl(raw_output)

            # Collect Ambiguous lines from this chunk to drive UI resolution
            try:
                for it in items:
                    ch = (it.get("character") or "").strip().lower()
                    if ch == "ambiguous":
                        txt = (it.get("text") or "").strip()
                        cands = it.get("candidates") or []
                        if not isinstance(cands, list):
                            cands = []
                        canonical_cands: List[str] = []
                        seen: Set[str] = set()
                        for c in cands:
                            c_norm = state.canonicalize((c or "").strip())
                            if not c_norm or c_norm in seen:
                                continue
                            seen.add(c_norm)
                            canonical_cands.append(c_norm)
                        if not canonical_cands:
                            canonical_cands = ["Unknown"]
                        ambiguities.append({
                            "id": f"amb-{idx+1}-{abs(hash(txt))}",
                            "text": txt,
                            "candidates": canonical_cands[:5],
                        })
            except Exception:
                # Ambiguity harvesting should not break parsing
                pass

            validated, warns = self._validate_and_normalize(items, state)
            all_warnings.extend(warns)
            all_dialogues.extend(validated)

            last_char = state.last_speaker or "Narrator"
            prev_summary = f"Last speaker: {last_char}. Emitted lines: {len(validated)}."

        # Reconcile adjacent same-speaker lines
        reconciled: List[Dict[str, Any]] = []
        for item in all_dialogues:
            if reconciled and reconciled[-1]["character"] == item["character"]:
                reconciled[-1]["text"] = f"{reconciled[-1]['text']} {item['text']}".strip()
                merged_emotions = list(dict.fromkeys(
                    reconciled[-1]["emotions"] + item["emotions"]))[:2]
                reconciled[-1]["emotions"] = merged_emotions
            else:
                reconciled.append(item)

        formatted_lines: List[str] = []
        for d in reconciled:
            if not self.include_narration and d.get("character") == "Narrator":
                continue
            emotion_text = "".join([f"({e})" for e in d.get("emotions", [])])
            formatted_lines.append(
                f"[{d['character']}] {emotion_text}: {d['text']}".strip())

        stats = {
            "quotes_found": len(reconciled),
            "lines_emitted": len(formatted_lines),
            "narration_blocks": sum(1 for d in reconciled if d.get("character") == "Narrator"),
        }

        if all_warnings:
            try:
                st.warning("\n".join(all_warnings[:10]))
            except Exception:
                pass

        return RawParseResult("\n".join(formatted_lines), reconciled, stats, ambiguities)

    def _parse_line_flexible(self, line: str) -> Optional[ParsedDialogue]:
        original_line = line
        char_match = re.match(r"\[([^\]]+)\]\s*(\([^)]+\))*\s*:", line)
        if char_match:
            character = char_match.group(1).strip()
            emotions = re.findall(r"\(([^)]+)\)", char_match.group(0))
            text_part = re.sub(
                r"^\[([^\]]+)\]\s*(\([^)]+\))*\s*:", "", line).strip()
        elif ":" in line and not line.startswith("["):
            parts = line.split(":", 1)
            if len(parts) == 2:
                character, text_part, emotions = parts[0].strip(), parts[1].strip(), [
                ]
            else:
                return None
        else:
            return None

        clean_text = re.sub(r"\*[^*]+\*", "", text_part).strip()
        if not clean_text:
            return None

        return ParsedDialogue(
            character=character,
            text=clean_text,
            emotions=[e.strip() for e in emotions],
            line_number=0,
            original_line=original_line,
        )

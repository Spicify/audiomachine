import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import streamlit as st
from audio.utils import get_flat_character_voices, get_flat_emotion_tags
from typing import List, Dict, Optional, Tuple
import time


@dataclass
class ParsedDialogue:
    """Data class for parsed dialogue line"""
    character: str
    text: str
    emotions: List[str]
    sound_effects: List[str]
    line_number: int
    original_line: str


@dataclass
class ParseAnalysis:
    """Data class for text parsing analysis"""
    characters_found: Dict[str, int]
    emotions_found: Dict[str, int]
    sound_effects_found: Dict[str, int]
    unsupported_characters: List[str]
    unsupported_emotions: List[str]
    unsupported_sound_effects: List[str]
    total_lines: int
    dialogue_lines: int

    def merge(self, other: "ParseAnalysis") -> None:
        """Merge another ParseAnalysis into this one in-place."""
        for k, v in other.characters_found.items():
            self.characters_found[k] = self.characters_found.get(k, 0) + v
        for k, v in other.emotions_found.items():
            self.emotions_found[k] = self.emotions_found.get(k, 0) + v
        for k, v in other.sound_effects_found.items():
            self.sound_effects_found[k] = self.sound_effects_found.get(
                k, 0) + v

        # Union unsupported lists
        self_uch = set(self.unsupported_characters)
        other_uch = set(other.unsupported_characters)
        self.unsupported_characters = list(self_uch.union(other_uch))

        self_uem = set(self.unsupported_emotions)
        other_uem = set(other.unsupported_emotions)
        self.unsupported_emotions = list(self_uem.union(other_uem))

        self_use = set(self.unsupported_sound_effects)
        other_use = set(other.unsupported_sound_effects)
        self.unsupported_sound_effects = list(self_use.union(other_use))

        self.total_lines += other.total_lines
        self.dialogue_lines += other.dialogue_lines

    @classmethod
    def empty(cls) -> "ParseAnalysis":
        return cls(
            characters_found={},
            emotions_found={},
            sound_effects_found={},
            unsupported_characters=[],
            unsupported_emotions=[],
            unsupported_sound_effects=[],
            total_lines=0,
            dialogue_lines=0,
        )


class TextParser:
    """Advanced text parser for various dialogue formats"""

    def __init__(self):
        self.character_voices = get_flat_character_voices()
        self.emotion_tags = get_flat_emotion_tags()
        # FX removed

    def analyze_text(self, text: str) -> ParseAnalysis:
        """Analyze text and provide comprehensive statistics (batched with progress)."""
        lines = text.strip().split('\n')
        total_lines = len(lines)

        characters_found = defaultdict(int)
        emotions_found = defaultdict(int)
        sound_effects_found = defaultdict(int)
        unsupported_characters = set()
        unsupported_emotions = set()
        unsupported_sound_effects = set()
        dialogue_lines = 0

        # Batch processing to keep UI responsive for large inputs
        batch_size = 20
        total_batches = (total_lines + batch_size -
                         1) // batch_size if total_lines > 0 else 0
        progress_bar = st.progress(0) if total_batches > 1 else None

        processed_lines = 0
        for batch_index in range(0, total_lines, batch_size):
            chunk = lines[batch_index:batch_index + batch_size]
            for idx, raw_line in enumerate(chunk, start=batch_index + 1):
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue

                parsed = self._parse_line_flexible(line)
                if parsed:
                    dialogue_lines += 1

                    # Count character usage
                    characters_found[parsed.character] += 1
                    if parsed.character not in self.character_voices:
                        unsupported_characters.add(parsed.character)

                    # Count emotions
                    for emotion in parsed.emotions:
                        emotions_found[emotion] += 1
                        if emotion not in self.emotion_tags:
                            unsupported_emotions.add(emotion)

                    # FX removed

            processed_lines += len(chunk)
            if progress_bar is not None and total_lines:
                progress_bar.progress(min(1.0, processed_lines / total_lines))

        return ParseAnalysis(
            characters_found=dict(characters_found),
            emotions_found=dict(emotions_found),
            sound_effects_found={},
            unsupported_characters=list(unsupported_characters),
            unsupported_emotions=list(unsupported_emotions),
            unsupported_sound_effects=[],
            total_lines=total_lines,
            dialogue_lines=dialogue_lines
        )

    def _parse_line_flexible(self, line: str) -> Optional[ParsedDialogue]:
        """Parse a single line with flexible format support"""
        original_line = line

        # Format 1: [Character] (emotion): Text *effect*
        char_match = re.match(r'\[([^\]]+)\]\s*(\([^)]+\))*\s*:', line)
        if char_match:
            character = char_match.group(1).strip()
            emotion_matches = re.findall(r'\(([^)]+)\)', char_match.group(0))
            text_part = re.sub(
                r'^\[([^\]]+)\]\s*(\([^)]+\))*\s*:', '', line).strip()

        # Format 2: Character: Text (no brackets)
        elif ':' in line and not line.startswith('['):
            parts = line.split(':', 1)
            if len(parts) == 2:
                character = parts[0].strip()
                text_part = parts[1].strip()
                emotion_matches = []
            else:
                return None

        # Format 3: "Character said" or Character said
        elif ' said' in line.lower():
            said_match = re.match(
                r'(["\']?)([^"\']+)\1\s+said[:\s]*(.+)', line, re.IGNORECASE)
            if said_match:
                character = said_match.group(2).strip()
                text_part = said_match.group(3).strip()
                emotion_matches = []
            else:
                return None

        # Format 4: Narrative with quoted speech
        elif '"' in line or '"' in line or '"' in line:
            # Extract character name (usually before the quote or after)
            quote_match = re.search(r'["""]([^"""]+)["""]', line)
            if quote_match:
                text_part = quote_match.group(1).strip()
                # Try to find character name in the remaining text
                remaining = line.replace(quote_match.group(0), '').strip()
                # Look for capitalized words that might be names
                name_candidates = re.findall(r'\b[A-Z][a-zA-Z]+\b', remaining)
                character = name_candidates[0] if name_candidates else "Unknown"
                emotion_matches = []
            else:
                return None
        else:
            return None

        # Clean text; strip legacy *fx* if present
        sound_effects = []
        clean_text = re.sub(r'\*[^*]+\*', '', text_part).strip()

        if not clean_text:
            return None

        return ParsedDialogue(
            character=character,
            text=clean_text,
            emotions=[e.strip() for e in emotion_matches],
            sound_effects=[e.strip() for e in sound_effects],
            line_number=0,  # Will be set later
            original_line=original_line
        )

    def parse_to_dialogue_format(self, text: str) -> Tuple[str, List[ParsedDialogue]]:
        """Parse text and convert to standard dialogue format"""
        lines = text.strip().split('\n')
        parsed_dialogues = []
        formatted_lines = []

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                formatted_lines.append(line)
                continue

            parsed = self._parse_line_flexible(line)
            if parsed:
                parsed.line_number = line_num
                parsed_dialogues.append(parsed)

                # Format to standard format
                emotion_text = ''.join(
                    [f"({emotion})" for emotion in parsed.emotions])
                effect_text = ''

                formatted_line = f"[{parsed.character}] {emotion_text}: {parsed.text} {effect_text}".strip(
                )
                formatted_lines.append(formatted_line)
            else:
                formatted_lines.append(f"# UNPARSED: {line}")

        return '\n'.join(formatted_lines), parsed_dialogues

    # Internal helper to analyze a list of lines and return a partial ParseAnalysis
    def _analyze_lines(self, lines: List[str]) -> ParseAnalysis:
        characters_found = defaultdict(int)
        emotions_found = defaultdict(int)
        sound_effects_found = defaultdict(int)
        unsupported_characters = set()
        unsupported_emotions = set()
        unsupported_sound_effects = set()
        dialogue_lines = 0

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            parsed = self._parse_line_flexible(line)
            if not parsed:
                continue

            dialogue_lines += 1
            characters_found[parsed.character] += 1
            if parsed.character not in self.character_voices:
                unsupported_characters.add(parsed.character)

            for emotion in parsed.emotions:
                emotions_found[emotion] += 1
                if emotion not in self.emotion_tags:
                    unsupported_emotions.add(emotion)

            # FX handling removed: ignore any legacy effect markers entirely

        return ParseAnalysis(
            characters_found=dict(characters_found),
            emotions_found=dict(emotions_found),
            sound_effects_found=dict(sound_effects_found),
            unsupported_characters=list(unsupported_characters),
            unsupported_emotions=list(unsupported_emotions),
            unsupported_sound_effects=list(unsupported_sound_effects),
            total_lines=len(lines),
            dialogue_lines=dialogue_lines,
        )


# Cached wrapper for analysis to avoid recomputation on identical text
@st.cache_data
def cached_analyze_text(text: str) -> ParseAnalysis:
    parser = TextParser()
    return parser.analyze_text(text)


def batched_analyze_text(text: str, batch_size: int = 20) -> ParseAnalysis:
    """Analyze text in batches with live progress updates to keep UI responsive."""
    parser = TextParser()
    lines = text.strip().split('\n')
    total = len(lines)
    if total == 0:
        return ParseAnalysis.empty()

    progress_bar = st.progress(0)
    status = st.empty()

    result = ParseAnalysis.empty()
    processed = 0

    last_update_processed = 0
    update_every = max(10, batch_size)  # throttle UI updates

    for start in range(0, total, batch_size):
        chunk = lines[start:start + batch_size]
        partial = parser._analyze_lines(chunk)
        result.merge(partial)

        processed += len(chunk)
        if total and (processed - last_update_processed) >= update_every:
            progress_bar.progress(min(1.0, processed / total))
            status.text(f"Analyzing... {processed}/{total} lines")
            last_update_processed = processed

    # Final UI update
    progress_bar.progress(1.0)
    status.text("Analysis complete")
    return result

from __future__ import annotations

import json
from typing import Dict, List, Optional, Set


def build_system_prompt(
    allowed_emotions: Set[str],
    known_characters: List[str],
    include_narration: bool,
    state_summary: Dict,
) -> str:
    ae = ", ".join(sorted(allowed_emotions))
    kc = ", ".join(sorted(known_characters)[
                   :50]) if known_characters else "(none)"
    lines: List[str] = [
        "You are a strict audiobook dialogue parser.",
        "Output MUST be JSON Lines (JSONL), one object per line, with EXACT keys: character (string), emotions (array of 2 strings), text (string).",
        "No commentary, no blank lines, no extra keys (except optional candidates for Ambiguous).",
        "",
        "Character attribution rules:",
        "- Infer the speaker ONLY when text clearly attributes it.",
        "- If uncertain about the character → use character 'Ambiguous' and include 2–5 'candidates'.",
        "- Do NOT invent names.",
        "- Narrator vs POV:",
        "  * Use 'Narrator' for objective, third-person description or scene setting that is NOT tied to any specific character’s perspective.",
        "  * Use the active POV character only when the line clearly reflects their **own** perspective (first-person pronouns like 'I', 'me', 'my' AND context showing it’s their thought/feeling).",
        "  * If the line describes actions/appearance/emotions of another character in third-person, use 'Narrator'.",
        "",
        "Emotion rules:",
        "- Provide exactly TWO emotions per line.",
        "- Emotions must be from ALLOWED_EMOTIONS. If none applies, use 'calm'.",
        "",
        "Formatting rules:",
        "- JSON object per line with keys: character, emotions, text.",
        "- candidates allowed only when character == 'Ambiguous'.",
        "",
        "Coverage requirements:",
        "- Do NOT skip or drop any input sentence. Every sentence from the provided text MUST appear as a JSONL output line.",
        "- If unsure about the speaker, still output the sentence with character 'Ambiguous' and include 2–5 'candidates'.",
        "- If unsure about emotions, still output the sentence and choose sensible defaults (e.g., 'calm' plus another allowed neutral emotion).",
        f"ALLOWED_EMOTIONS: {ae}",
        f"Known characters so far: {kc}",
        f"State summary: {json.dumps(state_summary, ensure_ascii=False)}",
    ]
    if include_narration:
        lines.append(
            "Include narration as 'Narrator' only for non-spoken descriptive text.")
    else:
        lines.append(
            "Do not include narration lines; only output spoken dialogue.")

    # Few-shot pattern examples
    lines.extend([
        '{"character": "Brad", "emotions": ["angry", "tense"], "text": "Get up!"}',
        '{"character": "Zara", "emotions": ["calm", "warm"], "text": "Hello."}',
        '{"character": "Donatello Moretti", "emotions": ["shaky", "ashamed"], "text": "I swallowed."}',
        '{"character": "Narrator", "emotions": ["neutral", "calm"], "text": "The sun was setting over the valley."}',
        '{"character": "Narrator", "emotions": ["neutral", "calm"], "text": "Donatello Moretti barely lifted his gaze from the glass of bourbon in his hand. When he did, his stare was ice."}',
        '{"character": "Luca Moretti", "emotions": ["shaky", "breathless"], "text": "I was 22. Standing in his office, heart pounding, hands cold."}',
        '{"character": "Ambiguous", "emotions": ["neutral", "calm"], "candidates": ["Aria Amato", "Luca Moretti"], "text": "You two should keep your voices down."}',
    ])
    return "\n".join(lines)


def build_user_prompt(text: str, prev_summary: Optional[str]) -> str:
    parts: List[str] = []
    if prev_summary:
        parts.append(f"Previous chunk summary: {prev_summary}")
    parts.append("Text to parse:")
    parts.append(text)
    return "\n".join(parts)

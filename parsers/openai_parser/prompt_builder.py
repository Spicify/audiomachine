from __future__ import annotations

import json
from typing import Dict, List, Optional, Set


def build_system_prompt(
    allowed_emotions: Set[str],
    known_characters: List[str],
    include_narration: bool,
    state_summary: Dict,
) -> str:
    # Derived ONLY from EMOTION_TAGS keys
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
        "  * Use the active POV character only when the line clearly reflects their own perspective (first-person pronouns like 'I', 'me', 'my' AND context showing it’s their thought/feeling).",
        "  * If the line describes actions/appearance/emotions of another character in third-person, use 'Narrator'.",
        "- Dialogue attribution (quoted speech):",
        "  * If quoted text is followed by said/asked/whispered/shouted/replied/murmured + a name or pronoun, infer that subject as the speaker.",
        "  * If a line starts with a name or pronoun and includes quoted text later, infer that subject as the speaker of the quoted dialogue.",
        "  * If narration and a quote coexist in one sentence, emit two JSONL lines: one for Narrator (non-quoted part) and one for the speaker (quoted part).",
        "  * First-person ('I', 'me', 'my') inside quotes indicates the speaking character, not Narrator.",
        "  * Never output duplicate Narrator lines repeating the same quoted text.",
        "  * When quotes include attribution verbs (e.g., said, asked, whispered, commanded, moaned, murmured, replied), exclude those verbs from the 'text' (keep only words inside quotes).",
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
        '{"character": "Narrator", "emotions": ["neutral", "calm"], "text": "The sun was setting over the valley."}',
        '{"character": "Ambiguous", "emotions": ["neutral", "calm"], "candidates": ["Aria Amato", "Luca Moretti"], "text": "You two should keep your voices down."}',
        # Concise example for attribution verbs handling
        'Input: "Keep your eyes on me," she commanded. → Output: {"character":"Maya","emotions":["commanding","dominant"],"text":"Keep your eyes on me."}',
    ])
    # REJECTED clause to prevent silent drops and enforce explicit refusal tagging
    lines.extend([
        "",
        "If you are unable to parse a sentence (for any reason such as policy, explicit content, or uncertainty),",
        "DO NOT skip or alter it.",
        "Instead, output this exact JSON line:",
        "",
        '{"character": "REJECTED", "emotions": ["neutral","calm"], "text": "<REJECTED_LINE>"}',
        "",
        "where <REJECTED_LINE> is the original sentence you refused to parse.",
        "This tag must be standalone, separate from other lines, and should not affect the rest of the output.",
    ])
    return "\n".join(lines)


def build_user_prompt(text: str, prev_summary: Optional[str]) -> str:
    parts: List[str] = []
    if prev_summary:
        parts.append(f"Previous chunk summary: {prev_summary}")
    parts.append("Text to parse:")
    parts.append(text)
    return "\n".join(parts)

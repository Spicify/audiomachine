"""OpenAI-backed character detection utility."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from openai import OpenAI

from settings import OPENAI_API_KEY

_OPENAI_CLIENT: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        if not OPENAI_API_KEY:
            raise RuntimeError('OPENAI_API_KEY is not configured.')
        _OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    return _OPENAI_CLIENT


def _extract_response_text(response) -> str:
    try:
        chunks: List[str] = []
        for item in getattr(response, 'output', []) or []:
            for piece in getattr(item, 'content', []) or []:
                text_obj = getattr(piece, 'text', None)
                if text_obj is None:
                    continue
                value = getattr(text_obj, 'value', None)
                if value:
                    chunks.append(value)
        if chunks:
            return '\n'.join(chunks)
    except Exception:
        pass
    return ''


_SYSTEM_PROMPT = (
    "You analyze fictional prose and return the unique characters who act, speak, or are referenced "
    "as agents. Respond with minified JSON exactly matching: "
    '{\"characters\": [{\"name\": \"Name\", \"gender\": \"male|female|nonbinary|unknown\"}]}. '
    "Include only real personas (proper names or recurring titles like 'The Captain'). "
    "Each gender entry must be one of: male, female, nonbinary, unknown. "
    "If gender cannot be inferred, use unknown. Normalize whitespace and avoid duplicates. Return valid JSON only."
)


def _normalize_gender_label(label: Optional[str]) -> str:
    if not label:
        return 'unknown'
    value = str(label).strip().lower()
    if value in {'male', 'man', 'masculine', 'm', 'boy'}:
        return 'male'
    if value in {'female', 'woman', 'feminine', 'f', 'girl'}:
        return 'female'
    if value in {'nonbinary', 'non-binary', 'nb', 'genderqueer', 'genderfluid'}:
        return 'nonbinary'
    return 'unknown'


def detect_characters_via_openai(story_text: str) -> List[Dict[str, str]]:
    "Returns a list of {name, gender} dicts detected in the story."
    if not story_text or not story_text.strip():
        return []

    client = _get_openai_client()
    user_prompt = (
        'Extract all character names from the following story and respond ONLY with JSON. '
        f'Story:\n{story_text}'
    )

    response = client.responses.create(
        model='gpt-5-mini',
        input=[
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ],
    )

    raw_text = (getattr(response, 'output_text', None) or '').strip()
    if not raw_text:
        raw_text = _extract_response_text(response)
    if not raw_text:
        return []

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    characters = data.get('characters') if isinstance(data, dict) else None
    if not isinstance(characters, list):
        return []

    cleaned: List[Dict[str, str]] = []
    seen: set[str] = set()
    for entry in characters:
        if isinstance(entry, dict):
            name_value = entry.get('name')
            gender_value = entry.get('gender')
        else:
            name_value = entry
            gender_value = None
        display = str(name_value or '').strip()
        if not display:
            continue
        key = display.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({'name': display, 'gender': _normalize_gender_label(gender_value)})
    return cleaned

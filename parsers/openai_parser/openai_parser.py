from __future__ import annotations
from .utils_misc import *
from .convert_streaming import convert_stream
from .convert_batch import convert_batch
from .openai_client import call_openai_safe
from .validator import validate_and_fix
from .core_types import *
from difflib import SequenceMatcher as _SM
import re as _re_quotes
import re
from difflib import SequenceMatcher

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import difflib
import time
import io
import sys

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, ValidationError

from audio.utils import get_flat_emotion_tags, get_flat_character_voices
from settings import OPENAI_API_KEY
from .chunker import build_chunks, deduplicate_lines, deduplicate_lines_exact
from .chunker import diag_consume_dedup_conflicts
from .emotion_utils import EmotionMemory, build_emotion_kb, ensure_two_emotions, get_allowed_emotions
from .prompt_builder import build_system_prompt, build_user_prompt
from .fallback_utils import (
    detect_missing_or_rejected_lines,
    call_frendli_fallback,
    replace_or_insert_lines,
    filter_fallback_lines,
)
from .fallback_utils import _sanitize_character
from utils.text_normalizer import normalize_text as _norm_for_compare
from utils.log_instrumentation import log_timed_action
from utils.session_logger import log_to_session, log_exception

# Diagnostics only; do not change behavior
DEBUG_PARSER_DIAG = True  # diagnostics only; leave True for this run


class OpenAIParser:
    # Reinjection strictness gate
    REINJECT_STRICT: bool = True

    def __init__(
        self,
        model: str = "gpt-5-mini",
        include_narration: bool = True,
        max_tokens_per_chunk: int = 1000,
        debug_save: bool = False,
    ):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.include_narration = include_narration
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.debug_save = debug_save
        self.allowed_emotions = get_allowed_emotions()
        self.default_known_characters = set(get_flat_character_voices().keys())
        self.kb = build_emotion_kb()
        self.memory = EmotionMemory()
        self._cache: Dict[str, str] = {}
        # Per-run timing metrics (seconds per OpenAI call per chunk)
        self._run_call_durations: List[float] = []
        # Per-run token counts per chunk (from chunker)
        self._run_chunk_token_counts: List[int] = []
        self._last_call_elapsed_sec: float = 0.0
        self._eod_fallback_throttled: bool = False
        self.legacy_base_parser: bool = True  # TODO: later wire via settings if needed

    def _state_summary(self, state: ParserState) -> Dict[str, Any]:
        return {
            "recent_characters": list(state.known_characters)[:20],
            "last_speaker": state.last_speaker,
            "last_emotions": {k: v[-2:] for k, v in state.last_emotions.items()},
            "unresolved": [a.get("text", "") for a in state.unresolved_ambiguities][-5:],
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    @log_timed_action("OpenAI call duration")
    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        start_time = time.monotonic()
        print(">>> Calling OpenAI…", flush=True)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        out = (response.output_text or "").strip()
        self._last_call_elapsed_sec = (time.monotonic() - start_time)
        print(
            f">>> OpenAI responded in {self._last_call_elapsed_sec*1000.0:.0f} ms (chars={len(out)})", flush=True)
        return out

    def convert(self, raw_text: str) -> RawParseResult:
        return convert_batch(self, raw_text)

    def _line_key(self, it: Dict[str, Any]) -> str:
        return hashlib.sha256(f"{(it.get('character') or '').strip().lower()}|{(it.get('text') or '').strip()}".encode("utf-8")).hexdigest()

    def convert_streaming(self, raw_text: str):
        return convert_stream(self, raw_text)

    def finalize_stream(self, dialogues: List[Dict[str, Any]], include_narration: Optional[bool] = None) -> RawParseResult:
        inc = self.include_narration if include_narration is None else include_narration
        # Preserve each dialogue line exactly as provided (no merging). Only apply narrator inclusion filter.
        reconciled: List[Dict[str, Any]] = []
        for item in dialogues:
            if not inc and str(item.get("character")).strip() == "Narrator":
                continue
            reconciled.append(item)

        formatted_lines: List[str] = []
        for d in reconciled:
            em_text = "".join([f"({e})" for e in d.get("emotions", [])])
            formatted_lines.append(
                f"[{d['character']}] {em_text}: {d['text']}".strip())

        stats = {
            "quotes_found": len(reconciled),
            "lines_emitted": len(formatted_lines),
            "narration_blocks": sum(1 for d in reconciled if d.get("character") == "Narrator"),
        }

        # collect ambiguities present in dialogues (those with id)
        ambiguities: List[Dict[str, Any]] = []
        for d in dialogues:
            if str(d.get("character", "")).lower() == "ambiguous":
                ambiguities.append({
                    "id": d.get("id") or f"amb-{abs(hash(d.get('text') or ''))}",
                    "text": d.get("text", ""),
                    "candidates": d.get("candidates", [])[:5] if isinstance(d.get("candidates"), list) else [],
                })

        return RawParseResult("\n".join(formatted_lines), reconciled, stats, ambiguities, [], [])

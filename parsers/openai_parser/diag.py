from __future__ import annotations

import os
import re
from typing import Optional


_NSFW_RE = re.compile(
    r"\b(sex|nude|naked|breast|penis|vagina|oral|anal|cum|orgasm|fuck|fucked|fucking|horny|moan|groan)\b", re.IGNORECASE)


def diag_enabled() -> bool:
    v = os.getenv("DEBUG_PARSER_DIAG")
    if v is not None:
        return v.strip().lower() in {"1", "true", "yes", "on"}
    try:
        # Fallback to module flag if available
        from .openai_parser import DEBUG_PARSER_DIAG  # type: ignore
        return bool(DEBUG_PARSER_DIAG)
    except Exception:
        return False


def diag_print(msg: str) -> None:
    if diag_enabled():
        try:
            print(msg, flush=True)
        except Exception:
            pass


def nsfw_marker_present(text: str) -> bool:
    try:
        return bool(_NSFW_RE.search(text or ""))
    except Exception:
        return False


def preview(s: Optional[str], limit: int = 200) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ")
    return s[:limit]


def approx_token_len(s: str) -> int:
    if not s:
        return 0
    # Heuristic: ~4 chars per token
    return max(1, len(s) // 4)

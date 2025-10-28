from __future__ import annotations

import datetime
import time
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from utils.log_instrumentation import log_timed_action
from .diag import diag_enabled, diag_print, nsfw_marker_present, preview, approx_token_len


def _save_debug_output(raw_text: str, suffix: str = "", debug_save: bool = False) -> None:
    if not debug_save:
        return
    try:
        Path("debug_outputs").mkdir(exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name = f"openai_parser_{ts}{suffix}.txt"
        with open(Path("debug_outputs") / name, "w", encoding="utf-8") as f:
            f.write(raw_text)
    except Exception:
        pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
@log_timed_action("OpenAI call duration")
def _call_openai(system_prompt: str, user_prompt: str, *, client, model: str, on_elapsed=None) -> str:
    start_time = time.monotonic()
    print(">>> Calling OpenAI…", flush=True)
    if diag_enabled():
        try:
            diag_print(
                f"[LLM_REQ] engine=OpenAI chunk_idx={_DIAG_CTX.get('chunk_idx','-')} chunk_first50='{preview(user_prompt,50)}' prompt_len_chars={len(user_prompt or '')} prompt_len_tokens~={approx_token_len(user_prompt or '')} params={{model:'{model}'}} nsfw_in_prompt={nsfw_marker_present(user_prompt or '')}"
            )
        except Exception:
            pass
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    out = (response.output_text or "").strip()
    elapsed = (time.monotonic() - start_time)
    if diag_enabled():
        try:
            http_status = getattr(
                getattr(response, "_response", None), "status_code", None)
        except Exception:
            http_status = None
        diag_print(
            f"[LLM_RESP] engine=OpenAI http_status={http_status or 'n/a'} elapsed_ms={elapsed*1000.0:.0f} content_len={len(out)} nsfw_in_resp={nsfw_marker_present(out)}"
        )
        if not out:
            diag_print(
                f"[LLM_EMPTY] engine=OpenAI chunk_idx={_DIAG_CTX.get('chunk_idx','-')}")
    if on_elapsed is not None:
        try:
            on_elapsed(elapsed)
        except Exception:
            pass
    print(
        f">>> OpenAI responded in {elapsed*1000.0:.0f} ms (chars={len(out)})", flush=True)
    return out


def call_openai_safe(system_prompt: str, user_prompt: str, *, client, model: str) -> str:
    """Helper wrapper that calls OpenAI and returns response text safely."""
    last_elapsed_holder = {"v": 0.0}

    def _on_elapsed(v):
        last_elapsed_holder["v"] = float(v or 0.0)

    out = _call_openai(system_prompt, user_prompt,
                       client=client, model=model, on_elapsed=_on_elapsed)
    try:
        print(
            f">>> call_openai_safe: elapsed_ms={last_elapsed_holder['v']*1000.0:.0f}", flush=True)
    except Exception:
        pass
    return out

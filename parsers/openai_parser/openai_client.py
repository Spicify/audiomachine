from __future__ import annotations

import datetime
import time
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from utils.log_instrumentation import log_timed_action


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
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    out = (response.output_text or "").strip()
    elapsed = (time.monotonic() - start_time)
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

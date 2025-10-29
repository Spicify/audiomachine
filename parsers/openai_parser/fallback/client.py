import os
from openai import OpenAI
import time
from dotenv import load_dotenv
from ..diag import diag_enabled, diag_print, nsfw_marker_present, preview, approx_token_len
from .diag_ctx import _DIAG_CTX

# Ensure .env is loaded before reading environment variables
load_dotenv()
print("[DEBUG] .env loaded for Frendli fallback")


def _build_fallback_system_prompt(known_characters: list[str] | None = None) -> str:
    """Concise strict JSONL system prompt for fallback (≈25 lines)."""
    chars_str = ", ".join(known_characters or [])
    lines = [
        "You are a strict audiobook dialogue parser.",
        "Output MUST be JSON Lines (JSONL). One JSON object per line with keys: character, emotions (2 strings), text. Optional: candidates (2–5 strings) only when character == 'Ambiguous'.",
        "No extra text, no commentary, no blank lines.",
        "Do NOT invent new character names under any circumstances.",
        "If the speaker is unclear, use 'Ambiguous' and choose candidates ONLY from Known characters.",
        "Coverage: Do NOT skip any input sentence.",
        "If speaker unclear → use 'Ambiguous' with 2–5 candidates from known characters.",
        "Narration: Use 'Narrator' for objective, third-person description (not tied to any POV).",
        "Quotes: The text field must contain ONLY the spoken words inside quotes (drop attributions like 'he growled').",
        "When quotes include attribution verbs (e.g., said, asked, whispered, commanded, moaned, murmured, replied), exclude those verbs from the 'text' (keep only the words inside quotes).",
        "Speaker inference: When a quoted dialogue appears after a character’s name or pronoun (e.g., Mark said, '…' or he whispered, '…'), infer that character as the speaker of the quoted text.",
        "Emotions: Exactly TWO per line.",
        f"Known characters: [{chars_str}]",
        "Examples:",
        '{"character":"Narrator","emotions":["soft","sad"],"text":"The rain poured outside."}',
        '{"character":"Bella","emotions":["confused","tense"],"text":"I can’t believe he said that."}',
        '{"character":"Ambiguous","emotions":["curious","soft"],"candidates":["Aria","Luca"],"text":"You two need to keep quiet."}',
        '{"character":"Luca","emotions":["angry","tense"],"text":"Get up!"}',
        # New concise example for attribution verbs inside quotes
        'Input: "Keep your eyes on me," she commanded. → Output: {"character":"Maya","emotions":["commanding","dominant"],"text":"Keep your eyes on me."}',
    ]
    return "\n".join(lines)


def call_frendli_fallback(system_prompt: str, user_prompt: str, known_characters: list[str] | None = None) -> str:
    """
    Calls Frendli serverless endpoint for fallback parsing.
    Returns the raw JSONL string.
    """
    token_ok = os.getenv("FRIENDLI_TOKEN") is not None
    print(f"[DEBUG] Frendli token loaded: {token_ok}", flush=True)
    print("[DEBUG] FRIENDLI_TOKEN found:", bool(os.getenv("FRIENDLI_TOKEN")))

    client = OpenAI(
        api_key=os.getenv("FRIENDLI_TOKEN"),
        base_url="https://api.friendli.ai/serverless/v1",
    )

    # Verify client URL/debug
    try:
        print("[DEBUG] Friendli client base_url:",
              getattr(client, "base_url", None))
        print("[DEBUG] Frendli request URL verified", flush=True)
    except Exception:
        pass

    # Construct combined system prompt (strict JSONL rules)
    strict_system_prompt = _build_fallback_system_prompt(known_characters)
    # [DIAG] show first lines of system prompt
    try:
        print("[DIAG] Friendli system prompt (first 12 lines):", flush=True)
        for i, ln in enumerate(strict_system_prompt.splitlines()[:12], 1):
            print(f"[DIAG]   {i:02d}: {ln}", flush=True)
        _kc_empty = (not known_characters) or (isinstance(
            known_characters, list) and len(known_characters) == 0)
        _has_dont_invent = ("do not invent" in strict_system_prompt.lower())
        print(
            f"[DIAG] Known characters empty: {_kc_empty}; Has 'Do NOT invent names' clause: {_has_dont_invent}", flush=True)
    except Exception:
        pass

    start = time.monotonic()
    if diag_enabled():
        try:
            _stop = None
            diag_print(
                f"[LLM_REQ] engine=Friendli chunk_idx={_DIAG_CTX.get('chunk_idx','-')} chunk_first50='{preview(user_prompt,50)}' prompt_len_chars={len(user_prompt or '')} prompt_len_tokens~={approx_token_len(user_prompt or '')} params={{model:'mistralai/Mistral-Small-3.1-24B-Instruct-2503'}} nsfw_in_prompt={nsfw_marker_present(user_prompt or '')}"
            )
        except Exception:
            pass
    print("[DEBUG] Frendli request begin", flush=True)
    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            messages=[
                {"role": "system", "content": strict_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        out = response.choices[0].message.content if response and getattr(
            response, "choices", None) else ""
        elapsed = (time.monotonic() - start)
        if diag_enabled():
            try:
                http_status = getattr(
                    getattr(response, "_response", None), "status_code", None)
            except Exception:
                http_status = None
            diag_print(
                f"[LLM_RESP] engine=Friendli http_status={http_status or 'n/a'} elapsed_ms={elapsed*1000.0:.0f} content_len={len(out)} nsfw_in_resp={nsfw_marker_present(out)}"
            )
            if not out:
                diag_print(
                    f"[LLM_EMPTY] engine=Friendli chunk_idx={_DIAG_CTX.get('chunk_idx','-')}")
        print(
            f"[DEBUG] Frendli request end ({elapsed*1000.0:.0f} ms, chars={len(out)})", flush=True)
        return out
    except Exception as e:
        if diag_enabled():
            diag_print(
                f"[LLM_ERR] engine=Friendli err={type(e).__name__}:{str(e)[:200]}")
        print(f"[ERROR] Frendli request failed: {repr(e)}", flush=True)
        raise

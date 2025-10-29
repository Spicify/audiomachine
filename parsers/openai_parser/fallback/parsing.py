import re
import json as _json
from ..diag import diag_enabled, diag_print
from .textnorm import _normalize_text, _pnorm, _tok_sim, _norm_for_compare_punct_neutral


def _extract_quotes(text: str) -> list[str]:
    """Extract normalized quoted spans. Never raises; returns [] if none.

    - Handles straight and curly quotes
    - Trims whitespace and trailing punctuation
    - Logs a brief diagnostic
    """
    out: list[str] = []
    try:
        if not text:
            print("[FB_QOUTES] extracted=0 samples=[]", flush=True)
            return out
        import re as _re
        raw = _re.findall(r'“([^”]+)”|"([^"]+)"', text)
        for g1, g2 in raw:
            q = (g1 or g2 or "").strip()
            if not q:
                continue
            q = _normalize_text(q)
            # strip terminal punctuation
            q = q.rstrip('.,;:!?"\'')
            if q:
                out.append(q)
    except Exception:
        out = []
    try:
        _samples = out[:2]
        print(
            f"[FB_QOUTES] extracted={len(out)} samples={_samples}", flush=True)
    except Exception:
        pass
    return out


def filter_fallback_lines(segment_text: str, candidate_lines: list[dict]) -> list[dict]:
    """Filter Friendli candidate lines against the specific problem segment text.

    - Narrator: keep if contained in segment or token-similarity ≥ 0.30
    - Dialogue with quotes: keep if matches any quoted span or token-similarity ≥ 0.50 against a quoted span
    - Dialogue without quotes: keep if contained in segment or token-similarity ≥ 0.45
    Logs [FB_FUZZY_KEEP]/[FB_FUZZY_DROP] per-candidate.
    """
    try:
        seg_text = (segment_text or "")
        seg_norm = _pnorm(seg_text)
        quoted_in_seg = set(_pnorm(q) for q in _extract_quotes(seg_text))

        kept: list[dict] = []
        for ln in (candidate_lines or []):
            try:
                cand = _pnorm((ln or {}).get("text", ""))
                ch = (ln or {}).get("character", "") or ""
                keep = False
                reason = ""

                if ch == "Narrator":
                    if (cand and (cand in seg_norm or seg_norm.find(cand) >= 0)) or _tok_sim(cand, seg_norm) >= 0.30:
                        keep, reason = True, "narr_contained_or_sim>=0.30"
                else:
                    if quoted_in_seg:
                        if cand in quoted_in_seg or any(_tok_sim(cand, q) >= 0.50 for q in quoted_in_seg):
                            keep, reason = True, "quoted_match"
                    else:
                        if (cand and (cand in seg_norm or seg_norm.find(cand) >= 0)) or _tok_sim(cand, seg_norm) >= 0.45:
                            keep, reason = True, "no_quotes_seg_match"

                if keep:
                    try:
                        print(
                            f"[FB_FUZZY_KEEP] ch={ch} reason={reason} text='{(ln.get('text','') or '')[:80]}'", flush=True)
                    except Exception:
                        pass
                    kept.append(ln)
                else:
                    try:
                        print(
                            f"[FB_FUZZY_DROP] ch={ch} text='{(ln.get('text','') or '')[:80]}'", flush=True)
                    except Exception:
                        pass
            except Exception:
                # on per-candidate error, drop silently
                pass

        return kept

    except Exception as e:
        import traceback as _tb
        cause = f"{type(e).__name__}: {e}"
        here = _tb.extract_tb(e.__traceback__)[-1] if e.__traceback__ else None
        loc = f"{getattr(here, 'filename', 'unknown')}:{getattr(here, 'lineno', '?')}"
        try:
            print(
                f"[EXC_TRACE] stage=fallback_validation error='{cause}' at={loc}", flush=True)
        except Exception:
            pass
        try:
            from ..openai_parser import DEBUG_PARSER_DIAG as _DBG
        except Exception:
            _DBG = False
        if _DBG:
            raise
        return list(candidate_lines or [])


def _parse_friendli_output(text: str) -> list[dict]:
    """Tolerant parser for Friendli raw output.
    Tries: code-fence strip → JSONL → JSON array → braced object stream.
    De-duplicates by identical 'text' preserving order.
    """
    raw = (text or "")

    def _dedupe_keep_order(items: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for o in items or []:
            try:
                key = ((o or {}).get("text", "") or "").strip()
            except Exception:
                key = ""
            if key and key not in seen:
                seen.add(key)
                out.append(o)
        return out

    # d) code fences/backticks: extract contents between triple backticks if present
    try:
        if "```" in raw or "~~~" in raw:
            try:
                import re as __re
                blocks = __re.findall(r"```[\w-]*\n([\s\S]*?)```", raw)
                if blocks:
                    raw = "\n".join(blocks)
            except Exception as e:
                if diag_enabled():
                    diag_print(
                        f"[FR_PARSE_ERR] codefence {type(e).__name__}: {e}")
    except Exception:
        pass

    # a) Direct JSONL
    parsed: list[dict] = []
    try:
        for ln in (raw or "").splitlines():
            s = ln.strip()
            if not (s.startswith("{") and s.endswith("}")):
                continue
            try:
                obj = _json.loads(s)
                if isinstance(obj, dict):
                    parsed.append(obj)
            except Exception:
                continue
        if parsed:
            if diag_enabled():
                diag_print(f"[FR_PARSE] mode=jsonl parsed={len(parsed)}")
            return _dedupe_keep_order(parsed)
    except Exception as e:
        if diag_enabled():
            diag_print(f"[FR_PARSE_ERR] jsonl {type(e).__name__}: {e}")

    # b) JSON array
    try:
        s = raw.strip()
        if s.startswith("["):
            arr = _json.loads(s)
            if isinstance(arr, list):
                parsed = [o for o in arr if isinstance(o, dict)]
                if parsed:
                    if diag_enabled():
                        diag_print(
                            f"[FR_PARSE] mode=array parsed={len(parsed)}")
                    return _dedupe_keep_order(parsed)
    except Exception as e:
        if diag_enabled():
            diag_print(f"[FR_PARSE_ERR] array {type(e).__name__}: {e}")

    # c) Braced objects by brace-depth
    try:
        objs: list[str] = []
        depth = 0
        cur = []
        for ch in raw:
            if ch == '{':
                depth += 1
            if depth > 0:
                cur.append(ch)
            if ch == '}':
                depth -= 1
                if depth == 0 and cur:
                    objs.append(''.join(cur))
                    cur = []
        parsed = []
        for s in objs:
            try:
                obj = _json.loads(s)
                if isinstance(obj, dict):
                    parsed.append(obj)
            except Exception:
                continue
        if parsed:
            if diag_enabled():
                diag_print(f"[FR_PARSE] mode=braced parsed={len(parsed)}")
            return _dedupe_keep_order(parsed)
    except Exception as e:
        if diag_enabled():
            diag_print(f"[FR_PARSE_ERR] braced {type(e).__name__}: {e}")

    # Final: nothing parsed
    if diag_enabled():
        diag_print("[FR_PARSE] mode=none parsed=0")
    return []


def _sanitize_character(line: dict, known_characters: list[str] | None) -> dict:
    ch = str((line or {}).get("character", "")).strip()
    allowed = set((known_characters or [])) | {"Narrator", "Ambiguous"}
    if ch and ch not in allowed:
        line["character"] = "Ambiguous"
        # keep candidates small; optional
        if not line.get("candidates"):
            line["candidates"] = list((known_characters or [])[:5])
    return line

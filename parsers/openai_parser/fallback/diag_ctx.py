from ..diag import diag_enabled

_DIAG_CTX: dict = {"chunk_idx": None, "tail_appends": {}}


def _set_diag_context(*, chunk_idx: int | None = None) -> None:
    if not diag_enabled():
        return
    try:
        if chunk_idx is not None:
            _DIAG_CTX["chunk_idx"] = int(chunk_idx)
    except Exception:
        pass


def _bump_tail_append() -> None:
    if not diag_enabled():
        return
    try:
        ci = _DIAG_CTX.get("chunk_idx")
        if ci is None:
            return
        _DIAG_CTX.setdefault("tail_appends", {})
        _DIAG_CTX["tail_appends"][ci] = 1 + \
            int(_DIAG_CTX["tail_appends"].get(ci, 0))
    except Exception:
        pass


def _get_tail_appends(chunk_idx: int) -> int:
    try:
        return int(_DIAG_CTX.get("tail_appends", {}).get(chunk_idx, 0))
    except Exception:
        return 0

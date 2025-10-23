import os
try:
    import streamlit as st  # optional; not available in all contexts
except Exception:
    st = None

# Default comes from env with freeform_map as fallback
_DEFAULT = os.getenv("PARSER_EMOTIONS_MODE", "freeform_map")


def get_emotions_mode() -> str:
    """
    Returns the current emotions mode:
    - If running inside Streamlit and session override exists, use it.
    - Else fall back to env var or default ("freeform_map").
    """
    if st is not None:
        try:
            return st.session_state.get("PARSER_EMOTIONS_MODE", _DEFAULT)
        except Exception:
            return _DEFAULT
    return _DEFAULT

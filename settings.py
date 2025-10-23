import os
import platform
import shutil
from pathlib import Path
from pydub import AudioSegment
import streamlit as st
from dotenv import load_dotenv, find_dotenv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Robust .env loading ---
_debug_settings = os.getenv("DEBUG_SETTINGS", "false").lower() == "true"
_explicit_env = os.getenv("ENV_FILE")
_found_env = None
try:
    # 1) Explicit path via ENV_FILE (if provided)
    if _explicit_env and os.path.isfile(_explicit_env):
        _found_env = _explicit_env
    else:
        # 2) Auto-discover with find_dotenv (searches upwards from CWD)
        _found_env = find_dotenv(usecwd=True)
        if not _found_env:
            # 3) Fallbacks: project root, then parent dir
            cand1 = os.path.join(PROJECT_ROOT, ".env")
            cand2 = os.path.join(PROJECT_ROOT, "..", ".env")
            _found_env = cand1 if os.path.isfile(cand1) else (
                cand2 if os.path.isfile(cand2) else "")
    if _found_env:
        load_dotenv(dotenv_path=_found_env)
    else:
        load_dotenv()  # last-resort: load from environment only
except Exception:
    # Do not fail if dotenv loading has issues in prod
    pass


def _clean_env(name: str) -> str | None:
    val = os.getenv(name)
    if val is None:
        return None
    val = val.strip()
    return val if val else None


ELEVENLABS_API_KEY = _clean_env("ELEVENLABS_API_KEY")
APP_PASSWORD_CLIENT = _clean_env("APP_PASSWORD_CLIENT")
APP_PASSWORD_TEAM = _clean_env("APP_PASSWORD_TEAM")
OPENAI_API_KEY = _clean_env("OPENAI_API_KEY")
AWS_ACCESS_KEY_ID = _clean_env("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = _clean_env("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = _clean_env(
    "AWS_DEFAULT_REGION") or _clean_env("AWS_REGION")
AWS_S3_BUCKET = _clean_env("AWS_S3_BUCKET") or _clean_env("S3_BUCKET")

# Build VALID_PASSWORDS defensively: only include non-empty keys
VALID_PASSWORDS = {}
if APP_PASSWORD_CLIENT:
    VALID_PASSWORDS[APP_PASSWORD_CLIENT] = "client"
if APP_PASSWORD_TEAM:
    VALID_PASSWORDS[APP_PASSWORD_TEAM] = "team"

if _debug_settings:
    # Safe debug: never print secrets; indicate presence only
    print(
        f"[settings] dotenv: path={'auto' if not _found_env else _found_env}; "
        f"client_pw={'set' if APP_PASSWORD_CLIENT else 'missing'}; "
        f"team_pw={'set' if APP_PASSWORD_TEAM else 'missing'}"
    )

# --- Hybrid FFmpeg Handling ---
if platform.system() == "Windows":  # Local Windows → bundled ffmpeg
    FFMPEG_DIR = os.path.join(PROJECT_ROOT, "ffmpeg", "bin")
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ["PATH"]
else:  # Linux → system ffmpeg
    AudioSegment.converter = shutil.which("ffmpeg") or "ffmpeg"
    AudioSegment.ffprobe = shutil.which("ffprobe") or "ffprobe"

PARSER_EMOTIONS_MODE = os.getenv("PARSER_EMOTIONS_MODE", "freeform_map")

import os
import platform
import shutil
from pathlib import Path
from pydub import AudioSegment
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Secrets Handling ---

# Option A: Load from Streamlit secrets (uncomment if deploying to Streamlit Cloud)
# ELEVENLABS_API_KEY = st.secrets.get("elevenlabs", {}).get("ELEVENLABS_API_KEY")
# APP_PASSWORD_CLIENT = st.secrets.get("app", {}).get("APP_PASSWORD_CLIENT")
# APP_PASSWORD_TEAM = st.secrets.get("app", {}).get("APP_PASSWORD_TEAM")
# OPENAI_API_KEY = st.secrets.get("openai", {}).get("OPENAI_API_KEY")
# AWS_ACCESS_KEY_ID = st.secrets.get("aws", {}).get("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = st.secrets.get("aws", {}).get("AWS_SECRET_ACCESS_KEY")
# AWS_DEFAULT_REGION = st.secrets.get("aws", {}).get("AWS_DEFAULT_REGION")
# AWS_S3_BUCKET = st.secrets.get("aws", {}).get("AWS_S3_BUCKET")

# Option B (default): Load from .env file (Render, local dev)
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, "..", ".env"))

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
APP_PASSWORD_CLIENT = os.getenv("APP_PASSWORD_CLIENT")
APP_PASSWORD_TEAM = os.getenv("APP_PASSWORD_TEAM")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET")

VALID_PASSWORDS = {
    APP_PASSWORD_CLIENT: "client",
    APP_PASSWORD_TEAM: "team",
}

# --- Hybrid FFmpeg Handling ---
if platform.system() == "Windows":  # Local Windows → bundled ffmpeg
    FFMPEG_DIR = os.path.join(PROJECT_ROOT, "ffmpeg", "bin")
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ["PATH"]
else:  # Linux → system ffmpeg
    AudioSegment.converter = shutil.which("ffmpeg") or "ffmpeg"
    AudioSegment.ffprobe = shutil.which("ffprobe") or "ffprobe"

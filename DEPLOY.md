# Deployment Guide (Render)

## Python Version

- Render ignores `runtime.txt` for Python services. Set an environment variable instead:
  - Key: `PYTHON_VERSION`
  - Value: `3.11.0`
- This ensures built-in `audioop` remains available for dependencies like `pydub`.

## Fallback (if using Python 3.13+)

- `requirements.txt` includes `pyaudioop` to satisfy `pydub`'s fallback when `audioop` is absent.
- No app code changes required.

## Dependencies

- `requirements.txt` lists all Python packages. Render will `pip install -r requirements.txt`.
- `packages.txt` (optional) installs system packages; ensure `ffmpeg` is present (our Dockerfile installs it directly when used).

## Using Docker (alternative)

- The included `Dockerfile` pins `python:3.11-slim` and installs `ffmpeg`.
- On Render, choose a Docker-based service and point to this `Dockerfile` to fully control the runtime.

## Streamlit Service

- App entry: `streamlit run app.py` (port `8080`). Ensure the service exposes port `8080`.
- Relevant env vars are already set in the Dockerfile for headless Streamlit.

## Summary

- Preferred: set `PYTHON_VERSION=3.11.0` on Render to match local and keep `audioop`.
- Fallback safety: `pyaudioop` is installed, so Python 3.13 also works.

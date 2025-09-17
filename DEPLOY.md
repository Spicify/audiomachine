# Deployment Guide (Render)

## Do not add `pyaudioop`

- `pyaudioop` is not published on PyPI. Do not add it to `requirements.txt`.
- The correct fix for `ModuleNotFoundError: No module named 'pyaudioop'` on Render is to run Python 3.11 where `audioop` is built in.

## Pin Python 3.11 on Render (native runtime)

We ship a `render.yaml` that pins the Python version:

- File: `render.yaml`
- Sets env var `PYTHON_VERSION=3.11.0`
- Uses `streamlit run app.py --server.port=8080`

If configuring via the dashboard instead of `render.yaml`:

1. Open your service in the Render dashboard.
2. Go to Settings → Environment.
3. Add env var: `PYTHON_VERSION=3.11.0`.
4. Save and redeploy.

This ensures `audioop` is available and `pydub` loads without falling back to `pyaudioop`.

## runtime.txt

- `runtime.txt` is present with `python-3.11.0` but is ignored by Render's native Python services.
- Keep it to document the intended local runtime, but do not rely on it for Render.

## Docker (alternative)

- The included `Dockerfile` uses `FROM python:3.11-slim` and installs `ffmpeg`.
- On Render, choose a Docker-based Web Service to guarantee the exact runtime.

## Streamlit service details

- Port: `8080` (exposed via `EXPOSE 8080` in Dockerfile and set in `render.yaml` start command).
- Headless and CORS settings handled by environment variables in Dockerfile; `render.yaml` uses default Streamlit headless flags.

## Re-deploy checklist

- [ ] `requirements.txt` contains no `pyaudioop` entry.
- [ ] `PYTHON_VERSION=3.11.0` set (via `render.yaml` or dashboard).
- [ ] Redeploy service → verify build uses Python 3.11.x and app starts.

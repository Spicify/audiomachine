from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from utils.s3_utils import s3_read_json, s3_write_json, s3_generate_presigned_url
import json


class ProjectStateManager:
    def __init__(self, project_id: str, json_key: Optional[str] = None, audio_key: Optional[str] = None):
        self.project_id = project_id
        self.json_key = json_key or f"projects/{project_id}.json"
        self.audio_key = audio_key or f"projects/{project_id}/consolidated.mp3"
        self.state: Dict = {}

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def load(self) -> Optional[Dict]:
        self.state = s3_read_json(self.json_key) or {}
        return self.state or None

    def init_state(self, chunk_ids: List[str]):
        self.state = {
            "project_id": self.project_id,
            "status": "INCOMPLETE",
            "last_updated": self._now_iso(),
            "chunks": [{"id": cid, "status": "PENDING"} for cid in chunk_ids],
            "latest_file_url": None,
            "committed_index": -1,
        }
        try:
            print(
                f"[state] init_state project_id={self.project_id} chunks={len(chunk_ids)} committed_index=-1", flush=True)
        except Exception:
            pass
        self.save()

    def save(self):
        self.state["last_updated"] = self._now_iso()
        s3_write_json(self.json_key, self.state)

    def set_latest_url(self, url: Optional[str]):
        self.state["latest_file_url"] = url
        self.save()

    def mark_chunk_done(self, chunk_id: str):
        for ch in self.state.get("chunks", []):
            if ch.get("id") == chunk_id:
                ch["status"] = "DONE"
                break
        self.save()

    def set_status(self, status: str):
        self.state["status"] = status
        self.save()


# --- Per-project voice settings helpers ---
def _project_settings_key(project_name: str, filename: str) -> str:
    # Keep path scheme aligned with existing artifacts (do not change audio/state keys)
    # Store alongside other project metadata under projects/{project}/
    safe = project_name
    return f"projects/{safe}/{filename}"


def load_project_voice_settings(project_name: str) -> dict:
    """Load per-project voice settings JSON from S3. Returns {} if missing."""
    try:
        key = _project_settings_key(project_name, "voice_settings.json")
        data = s3_read_json(key)
        return data or {}
    except Exception:
        return {}


def save_project_voice_settings(project_name: str, payload: dict) -> None:
    """Persist per-project voice settings JSON to S3."""
    key = _project_settings_key(project_name, "voice_settings.json")
    try:
        s3_write_json(key, payload or {})
    except Exception:
        # best-effort, do not raise
        pass

    def get_pending_chunks(self) -> List[str]:
        return [c["id"] for c in self.state.get("chunks", []) if c.get("status") != "DONE"]

    def presign_latest(self, expires_seconds: int = 3600) -> Optional[str]:
        return s3_generate_presigned_url(self.audio_key, expires_seconds=expires_seconds)

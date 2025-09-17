from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from utils.s3_utils import s3_read_json, s3_write_json, s3_generate_presigned_url


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
        }
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

    def get_pending_chunks(self) -> List[str]:
        return [c["id"] for c in self.state.get("chunks", []) if c.get("status") != "DONE"]

    def presign_latest(self, expires_seconds: int = 3600) -> Optional[str]:
        return s3_generate_presigned_url(self.audio_key, expires_seconds=expires_seconds)

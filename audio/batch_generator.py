from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from elevenlabs import ElevenLabs
from pydub import AudioSegment, effects
from pydub.silence import detect_nonsilent
from tenacity import retry, stop_after_attempt, wait_exponential

from utils.chunking import chunk_text
from utils.state_manager import ProjectStateManager
from utils.s3_utils import s3_upload_bytes, s3_generate_presigned_url, s3_get_bytes
from audio.utils import get_flat_character_voices
from parsers.dialogue_parser import DialogueParser
from audio.generator import DialogueAudioGenerator


@dataclass
class ChunkTask:
    id: str
    text: str
    voice_id: str
    index: int


class ResumableBatchGenerator:
    def __init__(self, project_id: str, voice_id: Optional[str] = None, model_id: str = "eleven_v3", max_workers: int = 2):
        self.project_id = project_id
        self.voice_id = voice_id  # Optional global override
        self.model_id = model_id
        self.max_workers = max(1, min(3, max_workers))
        self.client = ElevenLabs()
        self.state = ProjectStateManager(project_id)
        self.audio_key = self.state.audio_key
        self.voice_map = self._load_voice_map()
        self.default_voice_id = self._resolve_default_voice()
        self.TARGET_DBFS = -16.0
        self.PAD_MS = 250
        self.CROSSFADE_MS = 100

    def _load_voice_map(self) -> Dict[str, Dict[str, str]]:
        # Returns mapping character name (case sensitive keys) -> {voice_id, gender}
        try:
            return get_flat_character_voices()
        except Exception:
            return {}

    def _resolve_default_voice(self) -> str:
        # Prefer Narrator if present, else first mapped voice, else session override
        try:
            import streamlit as st
            dv = st.session_state.get("default_custom_voice")
            if dv:
                return dv
        except Exception:
            pass
        narrator = None
        for name, data in self.voice_map.items():
            if name.lower() == "narrator" and isinstance(data, dict) and data.get("voice_id"):
                narrator = data["voice_id"]
                break
        if narrator:
            return narrator
        for data in self.voice_map.values():
            if isinstance(data, dict) and data.get("voice_id"):
                return data["voice_id"]
            if isinstance(data, str):
                return data
        return ""

    def _tts_chunk(self, text: str, voice_id: str) -> AudioSegment:
        """Robust TTS for a single text chunk with timeout and fallback.

        Policy:
        - 40s timeout per attempt
        - 1 immediate retry on failure
        - If still failing, split text in half and try each half once, then concatenate
        - On total failure, return short silence (never raise)
        """
        if not text.strip():
            return AudioSegment.silent(duration=200)

        def _try_once(t: str) -> Optional[AudioSegment]:
            result: Dict[str, Optional[bytes]] = {"data": None}

            def _worker():
                try:
                    stream = self.client.text_to_speech.convert(
                        voice_id=voice_id or self.default_voice_id,
                        text=t,
                        model_id=self.model_id,
                    )
                    buf = io.BytesIO()
                    for part in stream:
                        if part:
                            buf.write(part)
                    result["data"] = buf.getvalue()
                except Exception:
                    result["data"] = None

            th = threading.Thread(target=_worker, daemon=True)
            th.start()
            th.join(timeout=40.0)
            if th.is_alive():
                # timed out
                return None
            data = result.get("data")
            if not data:
                return None
            try:
                audio = AudioSegment.from_file(io.BytesIO(data), format="mp3")
                # Treat extremely short audio as failure
                if len(audio) < 200:
                    return None
                return audio
            except Exception:
                return None

        # First attempt
        seg = _try_once(text)
        if seg is not None:
            return seg

        # Second attempt
        seg = _try_once(text)
        if seg is not None:
            return seg

        # Split into two halves and try once per half
        mid = max(1, len(text) // 2)
        left = text[:mid]
        right = text[mid:]
        left_seg = _try_once(left)
        right_seg = _try_once(right)
        if left_seg is None and right_seg is None:
            return AudioSegment.silent(duration=300)
        if left_seg is None:
            left_seg = AudioSegment.silent(duration=100)
        if right_seg is None:
            right_seg = AudioSegment.silent(duration=100)
        # Seamless concat (no extra pad, no crossfade)
        return self._ensure(left_seg) + self._ensure(right_seg)

    def _post(self, seg: AudioSegment) -> AudioSegment:
        """Deprecated per-objectives: keep compatibility but return as-is.
        Final mastering is applied only once at the end.
        """
        if not seg:
            return AudioSegment.silent(duration=200)
        return seg

    def _ensure(self, seg: AudioSegment) -> AudioSegment:
        return seg.set_sample_width(2).set_channels(2).set_frame_rate(44100)

    def _merge(self, base: AudioSegment, add: AudioSegment, pad_after: bool) -> AudioSegment:
        base = self._ensure(base)
        add = self._ensure(add)
        if len(base) > 0:
            merged = base.append(add, crossfade=self.CROSSFADE_MS)
        else:
            merged = base + add
        if pad_after:
            merged = merged + AudioSegment.silent(duration=self.PAD_MS)
        return merged

    def _smart_subchunks(self, text: str, limit: int = 200) -> List[str]:
        """Split text smartly around sentence boundaries near the limit.

        - Prefer splitting at '.' close to the limit going backwards
        - If none found, hard split at limit
        - Repeat until all text consumed
        """
        t = (text or "").strip()
        if len(t) <= limit:
            return [t]
        parts: List[str] = []
        start = 0
        while start < len(t):
            remaining = t[start:]
            if len(remaining) <= limit:
                parts.append(remaining)
                break
            # search backward from limit for a period
            cut = limit
            window = remaining[:limit]
            dot = window.rfind('.')
            if dot != -1 and dot >= int(limit * 0.6):
                cut = dot + 1
            piece = remaining[:cut].strip()
            if not piece:
                piece = remaining[:limit]
                cut = limit
            parts.append(piece)
            start += cut
        return parts

    def _tts_for_text(self, text: str, voice_id: str) -> AudioSegment:
        """Pre-chunk long text and synthesize sequentially; join seamlessly."""
        subtexts = self._smart_subchunks(text, limit=200)
        seg = AudioSegment.silent(duration=0)
        for stx in subtexts:
            part = self._tts_chunk(stx, voice_id)
            # Seamless concatenation (no pad, no crossfade)
            seg = self._ensure(seg) + self._ensure(part)
        return seg

    def _finalize_consolidated(self, audio: AudioSegment) -> AudioSegment:
        if not audio:
            return audio
        audio = effects.normalize(audio, headroom=1.0)
        audio = effects.compress_dynamic_range(
            audio, threshold=-18.0, ratio=2.0, attack=5.0, release=80.0)
        try:
            current = audio.dBFS
            if current != float("-inf"):
                audio = audio.apply_gain(self.TARGET_DBFS - current)
        except Exception:
            pass
        return audio

    def _export(self, audio: AudioSegment) -> bytes:
        out = io.BytesIO()
        audio.export(out, format="mp3", bitrate="128k")
        return out.getvalue()

    def _trim_silence(self, seg: AudioSegment, silence_thresh: float = -45.0, chunk_size_ms: int = 10, keep_ms: int = 50) -> AudioSegment:
        if not seg or len(seg) == 0:
            return seg
        try:
            intervals = detect_nonsilent(
                seg, min_silence_len=chunk_size_ms, silence_thresh=silence_thresh)
            if not intervals:
                return seg
            start = max(0, intervals[0][0] - keep_ms)
            end = min(len(seg), intervals[-1][1] + keep_ms)
            return seg[start:end]
        except Exception:
            return seg

    def _get_voice_assignments(self) -> Dict[str, str]:
        # Prefer user session voice assignments; fallback to flattened defaults
        try:
            import streamlit as st
            for key in ("paste_voice_assignments", "upload_voice_assignments", "vm_voice_mappings"):
                if key in st.session_state and isinstance(st.session_state[key], dict) and st.session_state[key]:
                    return st.session_state[key]
        except Exception:
            pass
        # Fallback
        flat = get_flat_character_voices()
        # Convert dict values {voice_id: ...} -> voice_id string for parser compatibility
        simple: Dict[str, str] = {}
        for name, data in flat.items():
            if isinstance(data, dict):
                vid = data.get("voice_id")
            else:
                vid = data
            if vid:
                simple[name] = vid
        return simple

    def _build_sequence(self, full_text: str) -> List[Dict]:
        parser = DialogueParser()
        assignments = self._get_voice_assignments()
        seq = parser.parse_dialogue(full_text, assignments)
        return seq or []

    def _select_voice_for_text(self, text: str) -> str:
        # Heuristics: match leading [Character] or Character: patterns
        import re
        m = re.match(r"^\s*\[(?P<name>[^\]]+)\]", text)
        if not m:
            m = re.match(
                r"^\s*(?P<name>[A-Za-z][A-Za-z\s\.'-]{0,50})\s*:\s+", text)
        if m:
            name = m.group("name").strip()
            # exact or case-insensitive match
            for key, data in self.voice_map.items():
                if key.lower() == name.lower():
                    if isinstance(data, dict):
                        return data.get("voice_id") or self.default_voice_id
                    return data
        return self.default_voice_id

    def _split_long_text(self, text: str, max_chars: int = 2000) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        parts: List[str] = []
        start = 0
        while start < len(text):
            parts.append(text[start:start + max_chars])
            start += max_chars
        return parts

    def _build_tasks(self, full_text: str) -> List[ChunkTask]:
        # Build per-line tasks preserving character voices; fall back to narrator
        tasks: List[ChunkTask] = []
        lines = [ln for ln in full_text.splitlines() if ln.strip()
                 and not ln.strip().startswith('#')]
        idx = 0
        for ln in lines:
            voice_id = self.voice_id or self._select_voice_for_text(ln)
            for piece in self._split_long_text(ln, max_chars=2000):
                cid = f"{self.project_id}_chunk{str(idx+1).zfill(3)}"
                tasks.append(ChunkTask(id=cid, text=piece,
                             voice_id=voice_id, index=idx))
                idx += 1
        if not tasks:  # fallback to whole text with default voice
            voice_id = self.voice_id or self.default_voice_id
            for i, c in enumerate(self._split_long_text(full_text, 2000), start=1):
                cid = f"{self.project_id}_chunk{str(i).zfill(3)}"
                tasks.append(ChunkTask(id=cid, text=c,
                             voice_id=voice_id, index=i-1))
        return tasks

    def _upload_and_update(self, audio: AudioSegment):
        data = self._export(audio)
        s3_upload_bytes(self.audio_key, data, content_type="audio/mpeg")
        url = s3_generate_presigned_url(self.audio_key, expires_seconds=3600)
        self.state.set_latest_url(url)

    def run(self, full_text: str, progress_cb: Optional[Callable[[int, int], None]] = None):
        sequence = self._build_sequence(full_text)
        if not sequence:
            return
        # Build deterministic IDs per entry to track progress
        ids: List[str] = [
            f"{self.project_id}_chunk{str(i+1).zfill(3)}" for i in range(len(sequence))]
        if not self.state.load():
            self.state.init_state(ids)

        # Resumability via committed_index (avoid duplicates if previous upload completed)
        try:
            last_committed = int(self.state.state.get("committed_index", -1))
        except Exception:
            last_committed = -1
        indices_to_process: List[int] = list(
            range(last_committed + 1, len(sequence)))

        total = len(ids)
        completed = total - len(indices_to_process)

        consolidated = AudioSegment.silent(duration=0)
        existing = s3_get_bytes(self.audio_key)
        if existing:
            try:
                consolidated = AudioSegment.from_file(
                    io.BytesIO(existing), format="mp3")
            except Exception:
                consolidated = AudioSegment.silent(duration=0)

        if progress_cb:
            try:
                progress_cb(completed, total)
            except Exception:
                pass

        # Prepare TTS futures for pending speech entries
        gen_fx_loader = DialogueAudioGenerator()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            speech_futures: Dict[int, any] = {}
            for i in indices_to_process:
                entry = sequence[i]
                if entry.get("type") == "speech":
                    vid = entry.get("voice_id")
                    if isinstance(vid, dict):
                        vid = vid.get("voice_id")
                    if not vid:
                        text_for_voice = f"[{entry.get('character','Narrator')}] {entry.get('text','')}"
                        vid = self._select_voice_for_text(text_for_voice)
                    # Submit pre-chunking aware task
                    speech_futures[i] = pool.submit(
                        self._tts_for_text, entry.get("text", ""), vid)

            # Sequentially merge in order for all pending entries
            for rel_idx, i in enumerate(indices_to_process):
                entry = sequence[i]
                seg: Optional[AudioSegment] = None
                if entry.get("type") == "speech":
                    try:
                        seg = speech_futures[i].result()
                    except Exception:
                        seg = AudioSegment.silent(duration=200)
                # FX removed: ignore sound_effect entries if any remain
                elif entry.get("type") == "sound_effect":
                    seg = AudioSegment.silent(duration=200)
                elif entry.get("type") == "pause":
                    seg = AudioSegment.silent(
                        duration=int(entry.get("duration", 300)))
                else:
                    seg = AudioSegment.silent(duration=50)

                # Avoid PAD if the very next entry in the full sequence is an explicit pause
                next_is_pause = (i + 1 < len(sequence)
                                 and sequence[i + 1].get("type") == "pause")
                pad_after = (not next_is_pause) and (i < len(sequence) - 1)
                consolidated = self._merge(
                    consolidated, seg, pad_after=pad_after)

                # Mark chunk done early (before upload) for visibility; resumability is guarded by committed_index
                try:
                    self.state.mark_chunk_done(ids[i])
                except Exception:
                    pass

                completed += 1
                if progress_cb:
                    try:
                        progress_cb(completed, total)
                    except Exception:
                        pass

                # Batch upload every 3 processed chunks
                if completed % 3 == 0 or (i == indices_to_process[-1]):
                    try:
                        self._upload_and_update(consolidated)
                        # Update committed index to the last processed index
                        self.state.state["committed_index"] = i
                        self.state.save()
                    except Exception:
                        # Continue; next batch will attempt again
                        pass

        # Final mastering and upload normalized audio
        try:
            consolidated = self._finalize_consolidated(consolidated)
        except Exception:
            pass
        try:
            self._upload_and_update(consolidated)
            # On final upload, commit to last sequence index
            self.state.state["committed_index"] = len(sequence) - 1
            self.state.set_status("COMPLETED")
            self.state.set_latest_url(
                s3_generate_presigned_url(self.audio_key, 3600))
            self.state.save()
        except Exception:
            # Leave state as-is; history UI may still show partial
            pass

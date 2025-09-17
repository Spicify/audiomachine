from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
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

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=0.5, min=0.5, max=6), reraise=True)
    def _tts_chunk(self, text: str, voice_id: str) -> AudioSegment:
        if not text.strip():
            return AudioSegment.silent(duration=200)
        stream = self.client.text_to_speech.convert(
            voice_id=voice_id or self.default_voice_id,
            text=text,
            model_id=self.model_id,
        )
        buf = io.BytesIO()
        for part in stream:
            if part:
                buf.write(part)
        data = buf.getvalue()
        if not data:
            raise RuntimeError("Empty audio from ElevenLabs")
        return AudioSegment.from_file(io.BytesIO(data), format="mp3")

    def _post(self, seg: AudioSegment) -> AudioSegment:
        if not seg:
            return AudioSegment.silent(duration=200)
        # Gentle dynamics, then align to target loudness
        norm = effects.normalize(seg, headroom=1.0)
        comp = effects.compress_dynamic_range(
            norm, threshold=-18.0, ratio=2.0, attack=5.0, release=50.0)
        try:
            current = comp.dBFS
            if current != float("-inf"):
                comp = comp.apply_gain(self.TARGET_DBFS - current)
        except Exception:
            pass
        return comp.fade_in(10).fade_out(30)

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

        pending_ids = self.state.get_pending_chunks()
        indices_to_process: List[int] = [
            i for i, cid in enumerate(ids) if cid in pending_ids]

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
                    speech_futures[i] = pool.submit(
                        self._tts_chunk, entry.get("text", ""), vid)

            # Sequentially merge in order for all pending entries
            for rel_idx, i in enumerate(indices_to_process):
                entry = sequence[i]
                seg: Optional[AudioSegment] = None
                if entry.get("type") == "speech":
                    seg = speech_futures[i].result()
                    seg = self._post(seg)
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

                self._upload_and_update(consolidated)
                self.state.mark_chunk_done(ids[i])

                completed += 1
                if progress_cb:
                    try:
                        progress_cb(completed, total)
                    except Exception:
                        pass

        # Final mastering and upload normalized audio
        consolidated = self._finalize_consolidated(consolidated)
        self._upload_and_update(consolidated)
        self.state.set_status("COMPLETED")
        self.state.set_latest_url(
            s3_generate_presigned_url(self.audio_key, 3600))

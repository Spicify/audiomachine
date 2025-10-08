from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import os
import datetime
import psutil

from elevenlabs import ElevenLabs
from pydub import AudioSegment, effects
from pydub.silence import detect_nonsilent
from tenacity import retry, stop_after_attempt, wait_exponential

from utils.chunking import chunk_text
from utils.state_manager import ProjectStateManager, load_project_voice_settings
from utils.s3_utils import s3_upload_bytes, s3_generate_presigned_url, s3_get_bytes
from utils.s3_utils import get_s3_client, get_bucket_defaults
from audio.utils import get_flat_character_voices
from utils.voice_settings import normalize_settings
from parsers.dialogue_parser import DialogueParser
from audio.generator import DialogueAudioGenerator
from utils.session_logger import log_to_session, log_exception


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
        # Load and cache per-project voice settings
        try:
            self._voice_settings = normalize_settings(
                load_project_voice_settings(self.project_id)
            )
        except Exception:
            self._voice_settings = None

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
        """Robust TTS for a single text chunk with timeout and recursive fallback.

        Policy:
        - 90s timeout per streaming attempt
        - Retry by recursively splitting text in half up to depth 2 (≈3 total attempts per branch)
        - Only retry on empty bytes, decode error, or <50ms audio (silence)
        - On total failure, return short silence (never raise), with explicit log
        """
        if not text.strip():
            return AudioSegment.silent(duration=200)

        def _try_once(t: str) -> Optional[AudioSegment]:
            result: Dict[str, Optional[bytes]] = {"data": None}

            def _worker():
                try:
                    print(
                        f"[TTS] start voice_id={(voice_id or self.default_voice_id)!r} model_id={self.model_id} len={len(t)} preview={t[:80]!r}", flush=True)
                    vs = self._voice_settings or {}
                    stream = self.client.text_to_speech.convert(
                        voice_id=voice_id or self.default_voice_id,
                        text=t,
                        model_id="eleven_v3",
                        voice_settings={
                            "stability": vs.get("stability", 0.35),
                            "similarity_boost": vs.get("similarity_boost", 0.85),
                            "style": vs.get("style", 0.50),
                            "use_speaker_boost": bool(vs.get("use_speaker_boost", True)),
                        },
                    )
                    try:
                        import os
                        if os.getenv("ELEVENLABS_DEBUG_REQUESTS") == "1":
                            print(
                                f"[EMO][REQ] model=eleven_v3 st={vs.get('stability', 0.35):.2f} sim={vs.get('similarity_boost', 0.85):.2f} style={vs.get('style', 0.50):.2f} boost={bool(vs.get('use_speaker_boost', True))}",
                                flush=True,
                            )
                    except Exception:
                        pass
                    buf = io.BytesIO()
                    total_parts = 0
                    total_bytes = 0
                    print("[TTS] stream begin", flush=True)
                    for part in stream:
                        if part:
                            buf.write(part)
                            total_parts += 1
                            total_bytes += len(part)
                            print(
                                f"[TTS] stream part size={len(part)} bytes", flush=True)
                    print(
                        f"[TTS] stream complete parts={total_parts} bytes={total_bytes}", flush=True)
                    result["data"] = buf.getvalue()
                except Exception as e:
                    print(
                        f"[TTS] exception during stream: {type(e).__name__}: {e}", flush=True)
                    result["data"] = None

            th = threading.Thread(target=_worker, daemon=True)
            th.start()
            th.join(timeout=90.0)
            if th.is_alive():
                # timed out
                print(
                    f"[TTS] timeout after 90s for length={len(t)}", flush=True)
                return None
            data = result.get("data")
            if not data:
                print(f"[TTS] empty data for length={len(t)}", flush=True)
                return None
            try:
                audio = AudioSegment.from_file(io.BytesIO(data), format="mp3")
                # Treat extremely short audio (<50ms) as failure; accept >=50ms as valid
                if len(audio) < 50:
                    print(
                        f"[TTS] decoded <50ms for length={len(t)}", flush=True)
                    return None
                print(f"[TTS] decode ok len_ms={len(audio)}", flush=True)
                return audio
            except Exception as e:
                print(
                    f"[TTS] decode error for length={len(t)}: {type(e).__name__}: {e}", flush=True)
                return None

        def _synthesize_recursive(t: str, depth: int) -> AudioSegment:
            seg = _try_once(t)
            if seg is not None:
                return seg
            if depth >= 2 or len(t.strip()) <= 1:
                print(
                    f"[TTS] fallback to silence at depth={depth} len={len(t)}", flush=True)
                return AudioSegment.silent(duration=300)
            # Split and recurse
            mid = max(1, len(t) // 2)
            print(
                f"[TTS] retry via split depth={depth+1} len={len(t)}", flush=True)
            left = _synthesize_recursive(t[:mid], depth + 1)
            right = _synthesize_recursive(t[mid:], depth + 1)
            return self._ensure(left) + self._ensure(right)

        return _synthesize_recursive(text, 0)

    def _post(self, seg: AudioSegment) -> AudioSegment:
        """Deprecated per-objectives: keep compatibility but return as-is.
        Final mastering is applied only once at the end.
        """
        if not seg:
            print("[post] input is empty/None -> return 200ms silence", flush=True)
            return AudioSegment.silent(duration=200)
        print(f"[post] passthrough segment len_ms={len(seg)}", flush=True)
        return seg

    def _ensure(self, seg: AudioSegment) -> AudioSegment:
        ensured = seg.set_sample_width(2).set_channels(2).set_frame_rate(44100)
        if len(seg) != len(ensured):
            print(
                f"[ensure] len changed {len(seg)}-> {len(ensured)}", flush=True)
        return ensured

    def _merge(self, base: AudioSegment, add: AudioSegment, pad_after: bool) -> AudioSegment:
        base = self._ensure(base)
        add = self._ensure(add)
        print(
            f"[merge] base_len={len(base)} add_len={len(add)} crossfade={self.CROSSFADE_MS} pad_after={pad_after}", flush=True)
        if len(base) > 0:
            merged = base.append(add, crossfade=self.CROSSFADE_MS)
        else:
            merged = base + add
        if pad_after:
            merged = merged + AudioSegment.silent(duration=self.PAD_MS)
        print(f"[merge] result_len={len(merged)}", flush=True)
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
        print(
            f"[generator] _build_sequence start project_id={self.project_id} full_text_len={len(full_text)}", flush=True)
        parser = DialogueParser()
        assignments = self._get_voice_assignments()
        base_seq = parser.parse_dialogue(full_text, assignments) or []

        # Pre-chunk long speech entries (>800 chars) into multiple speech parts.
        if not base_seq:
            return base_seq

        out: List[Dict] = []
        i = 0
        while i < len(base_seq):
            entry = base_seq[i]
            if entry.get("type") == "speech":
                text = entry.get("text", "")
                print(
                    f"[generator] entry idx={i} type=speech text_len={len(text)} preview={text[:80]!r}", flush=True)
                if len(text) > 800:
                    parts = self._smart_subchunks(text, limit=800) or [text]
                    if len(parts) > 1:
                        print(
                            f"[generator] split into {len(parts)} sub-chunks (orig_len={len(text)})", flush=True)
                        for pi, pt in enumerate(parts):
                            print(
                                f"[generator] sub-chunk[{pi}] len={len(pt)} preview={pt[:80]!r}", flush=True)
                    for part_text in parts:
                        part_entry = dict(entry)
                        part_entry["text"] = part_text
                        out.append(part_entry)
                    # Only insert pause after the last sub-chunk of this entry
                    if i + 1 < len(base_seq) and base_seq[i + 1].get("type") == "pause":
                        out.append(base_seq[i + 1])
                        i += 2
                        continue
                else:
                    print(
                        f"[generator] entry idx={i} short speech (no split)", flush=True)
                    out.append(entry)
            else:
                et = entry.get("type")
                if et == "pause":
                    print(
                        f"[generator] entry idx={i} type=pause duration={entry.get('duration')}ms", flush=True)
                else:
                    print(f"[generator] entry idx={i} type={et}", flush=True)
                out.append(entry)
            i += 1

        return out

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

    def _upload_and_update(self, audio: AudioSegment, committed_index: int):
        """Export and upload consolidated audio with committed index metadata.

        Uses temporary key then replaces final key; stores committed_index in S3 object metadata.
        """
        data = self._export(audio)
        try:
            s3 = get_s3_client()
            bucket = get_bucket_defaults()
            # Choose a temp key alongside final
            tmp_key = self.audio_key.replace(
                "/consolidated.mp3", "/consolidated_tmp.mp3")
            if tmp_key == self.audio_key:
                tmp_key = self.audio_key + ".tmp"

            # Upload to temporary key first
            s3.put_object(
                Bucket=bucket,
                Key=tmp_key,
                Body=data,
                ContentType="audio/mpeg",
                Metadata={"committed_index": str(committed_index)},
            )

            # Copy/replace into final key with same metadata
            s3.copy_object(
                Bucket=bucket,
                Key=self.audio_key,
                CopySource={"Bucket": bucket, "Key": tmp_key},
                MetadataDirective="REPLACE",
                ContentType="audio/mpeg",
                Metadata={"committed_index": str(committed_index)},
            )
        except Exception:
            # Fallback to direct upload without metadata to avoid blocking progress
            try:
                s3_upload_bytes(self.audio_key, data,
                                content_type="audio/mpeg")
            except Exception:
                pass

        url = s3_generate_presigned_url(self.audio_key, expires_seconds=3600)
        self.state.set_latest_url(url)
        # Update committed index in state after successful upload path
        try:
            self.state.state["committed_index"] = committed_index
            self.state.save()
        except Exception:
            pass

    def _get_committed_index_from_s3(self) -> int:
        try:
            s3 = get_s3_client()
            bucket = get_bucket_defaults()
            head = s3.head_object(Bucket=bucket, Key=self.audio_key)
            meta = head.get("Metadata", {}) or {}
            # S3 stores metadata keys in lower-case
            for k, v in meta.items():
                if k.lower() == "committed_index":
                    return int(v)
        except Exception:
            pass
        return -1

    def run(self, full_text: str, progress_cb: Optional[Callable[[int, int], None]] = None):
        try:
            log_to_session("INFO", "Batch generation started",
                           src="audio/batch_generator.py:run")
        except Exception:
            pass
        sequence = self._build_sequence(full_text)
        if not sequence:
            return
        # --- Heartbeat: keep UI alive during long waits ---
        stop_hb = threading.Event()
        # --- Log buffer for diagnostics ---
        try:
            import io as _io
            self._log_buffer = _io.StringIO()
            self._log_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        except Exception:
            self._log_buffer = None
            self._log_timestamp = None

        def _start_heartbeat(cb, stop_event):
            def _loop():
                while not stop_event.is_set():
                    time.sleep(30)
                    try:
                        if cb:
                            cb(-1, -1)
                    except Exception:
                        pass
            threading.Thread(target=_loop, daemon=True).start()

        _start_heartbeat(progress_cb, stop_hb)
        # Build deterministic IDs per entry to track progress
        ids: List[str] = [
            f"{self.project_id}_chunk{str(i+1).zfill(3)}" for i in range(len(sequence))]
        loaded = self.state.load()
        if not loaded:
            print(
                f"[generator] no existing state -> initializing fresh state", flush=True)
            self.state.init_state(ids)
        else:
            print(
                f"[generator] existing state loaded for project_id={self.project_id}", flush=True)

        # Resumability via committed_index (avoid duplicates if previous upload completed)
        # Determine last committed index using S3 metadata if available (authoritative),
        # otherwise fall back to state value.
        try:
            s3_committed = int(self._get_committed_index_from_s3())
        except Exception:
            s3_committed = -1
        try:
            state_committed = int(self.state.state.get("committed_index", -1))
        except Exception:
            state_committed = -1
        # Fresh run detection: no state and no S3 metadata
        if s3_committed == -1 and state_committed == -1:
            print("[generator] fresh run detected: committed_index=-1", flush=True)
            last_committed = -1
        else:
            # Resume: take the max of state and S3
            last_committed = max(s3_committed, state_committed)
        print(
            f"[generator] committed_index(state)={state_committed} committed_index(s3)={s3_committed} using={last_committed}",
            flush=True,
        )
        indices_to_process: List[int] = list(
            range(last_committed + 1, len(sequence)))
        print(
            f"[generator] indices_to_process count={len(indices_to_process)} total={len(sequence)} committed_index={last_committed} indices={indices_to_process}",
            flush=True,
        )

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
                etype = entry.get("type")
                print(f"[generator] prepare idx={i} type={etype}", flush=True)
                if etype == "speech":
                    vid = entry.get("voice_id")
                    if isinstance(vid, dict):
                        vid = vid.get("voice_id")
                    if not vid:
                        text_for_voice = f"[{entry.get('character','Narrator')}] {entry.get('text','')}"
                        vid = self._select_voice_for_text(text_for_voice)
                    # Submit pre-chunking aware task
                    text_i = entry.get("text", "")
                    subcnt = len(self._smart_subchunks(text_i, limit=200))
                    print(
                        f"[generator] submit speech idx={i} subchunks={subcnt} preview={text_i[:80]!r}", flush=True)
                    speech_futures[i] = pool.submit(
                        self._tts_for_text, text_i, vid)
                else:
                    print(
                        f"[generator] no TTS future submitted idx={i} reason=type is {etype}", flush=True)

            # Sequentially merge in order for all pending entries
            for rel_idx, i in enumerate(indices_to_process):
                entry = sequence[i]
                seg: Optional[AudioSegment] = None
                # periodic diagnostics (every 5th item)
                try:
                    if (i % 5) == 0:
                        rss = psutil.Process(
                            os.getpid()).memory_info().rss / 1e6
                        now = datetime.datetime.now().strftime("%H:%M:%S")
                        _msg = f"[DIAG] {now} – mem {rss:.1f} MB – phase: TTS wait {i+1}/{total}"
                        print(_msg, flush=True)
                        try:
                            if getattr(self, "_log_buffer", None):
                                self._log_buffer.write(_msg + "\n")
                        except Exception:
                            pass
                except Exception:
                    pass
                if entry.get("type") == "speech":
                    try:
                        print(
                            f"[generator] collect TTS result idx={i}", flush=True)
                        seg = speech_futures[i].result()
                        print(
                            f"[generator] speech done idx={i} seg_len_ms={len(seg)}", flush=True)
                        try:
                            log_to_session(
                                "INFO", f"TTS collected idx={i} len_ms={len(seg)}", src="audio/batch_generator.py:run")
                        except Exception:
                            pass
                    except Exception as e:
                        print(
                            f"[generator] speech error idx={i}: {type(e).__name__}: {e}", flush=True)
                        seg = AudioSegment.silent(duration=200)
                        try:
                            log_exception("audio/batch_generator.py:run", e)
                        except Exception:
                            pass
                # FX removed: ignore sound_effect entries if any remain
                elif entry.get("type") == "sound_effect":
                    print(
                        f"[generator] collect idx={i} non-speech(type=sound_effect) -> skip TTS", flush=True)
                    seg = AudioSegment.silent(duration=200)
                elif entry.get("type") == "pause":
                    print(
                        f"[generator] collect idx={i} non-speech(type=pause) -> skip TTS", flush=True)
                    seg = AudioSegment.silent(
                        duration=int(entry.get("duration", 300)))
                else:
                    print(
                        f"[generator] collect idx={i} non-speech(type=other) -> skip TTS", flush=True)
                    seg = AudioSegment.silent(duration=50)

                # Avoid PAD if the very next entry in the full sequence is an explicit pause
                next_is_pause = (i + 1 < len(sequence)
                                 and sequence[i + 1].get("type") == "pause")
                pad_after = (not next_is_pause) and (i < len(sequence) - 1)
                try:
                    rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
                    now = datetime.datetime.now().strftime("%H:%M:%S")
                    if (i % 5) == 0:
                        _msg = f"[DIAG] {now} – mem {rss:.1f} MB – phase: merge {i+1}/{total}"
                        print(_msg, flush=True)
                        try:
                            if getattr(self, "_log_buffer", None):
                                self._log_buffer.write(_msg + "\n")
                        except Exception:
                            pass
                except Exception:
                    pass
                consolidated = self._merge(
                    consolidated, seg, pad_after=pad_after)
                print(
                    f"[generator] consolidated_len_ms={len(consolidated)} after idx={i}", flush=True)
                try:
                    if (i % 5) == 0:
                        log_to_session(
                            "DIAG", f"consolidated_len_ms={len(consolidated)} after idx={i}", src="audio/batch_generator.py:run")
                except Exception:
                    pass

                # Mark chunk done early (before upload) for visibility; resumability is guarded by committed_index
                try:
                    self.state.mark_chunk_done(ids[i])
                    print(
                        f"[generator] mark DONE id={ids[i]} idx={i}", flush=True)
                except Exception as e:
                    print(
                        f"[generator] mark DONE error id={ids[i]} idx={i}: {type(e).__name__}: {e}", flush=True)

                completed += 1
                if progress_cb:
                    try:
                        progress_cb(completed, total)
                    except Exception:
                        pass

                # Batch upload every 5 processed chunks
                if completed % 5 == 0 or (i == indices_to_process[-1]):
                    try:
                        try:
                            rss = psutil.Process(
                                os.getpid()).memory_info().rss / 1e6
                            now = datetime.datetime.now().strftime("%H:%M:%S")
                            _msg = f"[DIAG] {now} – mem {rss:.1f} MB – phase: upload start idx={i}"
                            print(_msg, flush=True)
                            try:
                                if getattr(self, "_log_buffer", None):
                                    self._log_buffer.write(_msg + "\n")
                            except Exception:
                                pass
                        except Exception:
                            pass
                        print(
                            f"[upload] batch upload idx={i} completed={completed} total={total} len_ms={len(consolidated)} key={self.audio_key}", flush=True)
                        self._upload_and_update(consolidated, i)
                        try:
                            log_to_session(
                                "INFO", f"batch upload complete idx={i} len_ms={len(consolidated)}", src="audio/batch_generator.py:run")
                        except Exception:
                            pass
                        print(
                            f"[state] committed_index updated -> {i}", flush=True)
                    except Exception as e:
                        print(
                            f"[upload] batch upload error idx={i}: {type(e).__name__}: {e}", flush=True)
                        # Continue; next batch will attempt again
                        try:
                            log_exception("audio/batch_generator.py:run", e)
                        except Exception:
                            pass
                        pass

        # Final mastering and upload normalized audio
        try:
            consolidated = self._finalize_consolidated(consolidated)
        except Exception as e:
            print(
                f"[generator] finalize error: {type(e).__name__}: {e}", flush=True)
            try:
                log_exception("audio/batch_generator.py:run", e)
            except Exception:
                pass
            pass
        try:
            print(
                f"[upload] final upload entries={len(sequence)} len_ms={len(consolidated)} key={self.audio_key}", flush=True)
            try:
                rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
                now = datetime.datetime.now().strftime("%H:%M:%S")
                _msg = f"[DIAG] {now} – mem {rss:.1f} MB – phase: final upload"
                print(_msg, flush=True)
                try:
                    if getattr(self, "_log_buffer", None):
                        self._log_buffer.write(_msg + "\n")
                except Exception:
                    pass
            except Exception:
                pass
            self._upload_and_update(consolidated, len(sequence) - 1)
            print(
                f"[state] committed_index updated -> {len(sequence) - 1}", flush=True)
            # On final upload, mark status complete
            self.state.set_status("COMPLETED")
            self.state.set_latest_url(
                s3_generate_presigned_url(self.audio_key, 3600))
            self.state.save()
            try:
                log_to_session("INFO", "final upload complete",
                               src="audio/batch_generator.py:run")
            except Exception:
                pass
        except Exception as e:
            print(
                f"[upload] final upload error: {type(e).__name__}: {e}", flush=True)
            # Leave state as-is; history UI may still show partial
            try:
                log_exception("audio/batch_generator.py:run", e)
            except Exception:
                pass
            pass
        finally:
            # stop heartbeat on exit
            try:
                stop_hb.set()
            except Exception:
                pass
            # Upload log buffer to S3 and prune to latest 10
            try:
                if getattr(self, "_log_buffer", None) and getattr(self, "_log_timestamp", None):
                    from utils.s3_utils import s3_upload_bytes, s3_list_objects, s3_delete_object
                    log_key = f"projects/{self.project_id}/logs/debug_{self._log_timestamp}.log"
                    data = self._log_buffer.getvalue().encode("utf-8", "ignore")
                    s3_upload_bytes(log_key, data, content_type="text/plain")
                    s3_upload_bytes(
                        f"projects/{self.project_id}/logs/latest.log", data, content_type="text/plain")
                    keys = s3_list_objects(
                        prefix=f"projects/{self.project_id}/logs/")
                    log_keys = sorted(
                        [k for k in keys if k.endswith('.log')], reverse=True)
                    for old in log_keys[10:]:
                        try:
                            s3_delete_object(old)
                        except Exception:
                            pass
            except Exception:
                pass

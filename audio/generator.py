from pydub import AudioSegment
from elevenlabs import ElevenLabs
from settings import ELEVENLABS_API_KEY
from pathlib import Path
import re
import tempfile
import streamlit as st
import io
import time
from datetime import datetime
from pydub import AudioSegment, effects
from audio.utils import get_flat_character_voices
from utils.voice_settings import normalize_settings
from utils.state_manager import load_project_voice_settings
from utils.downloads import create_output_folders
import os
import time
import traceback

TTS_DEBUG = os.getenv("TTS_DEBUG", "0") == "1"


def _ttsdbg(msg: str):
    if TTS_DEBUG:
        print(f"[TTSDBG] {msg}", flush=True)


ALLOWED_STABILITIES = (0.0, 0.5, 1.0)


def _sanitize_voice_settings(model_id: str, vs: dict | None) -> dict:
    """Clamp/clean voice_settings for ElevenLabs v3 rules to avoid 400s."""
    if not vs:
        return {}
    out = {}
    stab = vs.get("stability", None)
    if isinstance(stab, (int, float)):
        nearest = min(ALLOWED_STABILITIES, key=lambda x: abs(x - float(stab)))
        out["stability"] = nearest
    # Optionally keep similarity_boost only if numeric in [0,1]
    sb = vs.get("similarity_boost", None)
    if isinstance(sb, (int, float)):
        out["similarity_boost"] = max(0.0, min(1.0, float(sb)))
    _ttsdbg(f"voice_settings sanitized for model={model_id}: {out}")
    return out


class DialogueAudioGenerator:
    def __init__(self, api_key=ELEVENLABS_API_KEY, fx_library_path="./fx_library/"):
        """Initialize the dialogue audio generator"""
        self.client = ElevenLabs(api_key=api_key)
        self.fx_library_path = Path(fx_library_path)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.character_voices = get_flat_character_voices()
        # FX removed
        self.output_folders = create_output_folders()
        self._voice_settings = None  # lazy-init per project

    def apply_post_processing(self, audio: AudioSegment) -> AudioSegment:
        """Apply mastering chain: normalization, compression, fades, EQ/gain matching"""
        if not audio or len(audio) == 0:
            return audio

        try:
            # 1. Normalize loudness (target RMS ~ -20 dBFS)
            normalized = effects.normalize(audio, headroom=1.0)

            # 2. Light compression (tames peaks but keeps natural range)
            compressed = effects.compress_dynamic_range(
                normalized,
                threshold=-18.0,   # start compressing above -18 dBFS
                ratio=2.0,         # gentle compression
                attack=5.0,        # ms
                release=50.0       # ms
            )

            # 3. Add fades at edges (avoid clicks/pops)
            faded = compressed.fade_in(20).fade_out(50)

            # 4. Gain match fallback voices if needed (tweakable constant)
            # adjust if non-ElevenLabs sounds are quieter/louder
            final = faded.apply_gain(0.0)

            return final

        except Exception as e:
            st.warning(f"Post-processing error: {e}")
            return audio

    def generate_speech(self, voice_id, text, model_id="eleven_v3", voice_settings=None):
        """Generate TTS audio and return as AudioSegment"""
        try:
            # Ensure text is not empty
            if not text or not text.strip():
                _ttsdbg("generate_speech: empty text, returning silence 500ms")
                return AudioSegment.silent(duration=500)

            # Ensure we pass a plain string voice id to the API
            if isinstance(voice_id, dict):
                voice_id = voice_id.get("voice_id") or voice_id.get(
                    "id") or voice_id.get("voiceId")

            # Merge effective voice settings (if provided)
            vs = voice_settings or {}
            text_preview = (repr(text[:120]) +
                            "…") if len(text) > 120 else repr(text)
            _ttsdbg(
                f"generate_speech: start voice_id={voice_id} model_id={model_id} text_len={len(text)} text_preview={text_preview}"
            )
            vs_preview = {k: voice_settings.get(k) for k in (
                voice_settings or {})} if voice_settings else None
            _ttsdbg(
                f"generate_speech: voice_settings={vs_preview}"
            )
            start_ts = time.time()

            # Create a filesystem-safe filename component for voice id
            safe_voice = re.sub(r"[^a-zA-Z0-9_-]", "", str(voice_id))
            temp_file = self.temp_dir / f"temp_{safe_voice}_{hash(text)}.mp3"

            total_bytes = 0
            chunk_idx = 0
            try:
                safe_vs = _sanitize_voice_settings(
                    model_id, voice_settings or {})
                try:
                    audio_generator = self.client.text_to_speech.convert(
                        voice_id=voice_id,
                        text=text,
                        model_id=model_id,  # pass-through
                        voice_settings=safe_vs,
                    )
                except Exception as e:
                    msg = str(e)
                    _ttsdbg(f"generate_speech: ApiError on convert: {msg}")
                    if "invalid_ttd_stability" in msg.lower():
                        _ttsdbg(
                            "generate_speech: retrying once with stability=0.5")
                        audio_generator = self.client.text_to_speech.convert(
                            voice_id=voice_id,
                            text=text,
                            model_id=model_id,
                            voice_settings={"stability": 0.5},
                        )
                    else:
                        raise
                _ttsdbg("generate_speech: ElevenLabs convert() returned generator")

                with open(temp_file, "wb") as f:
                    for chunk in audio_generator:
                        chunk_idx += 1
                        if not chunk:
                            _ttsdbg(
                                f"generate_speech: chunk {chunk_idx} is empty/None")
                            continue
                        size = len(chunk)
                        total_bytes += size
                        if chunk_idx <= 3:
                            _ttsdbg(
                                f"generate_speech: chunk {chunk_idx} size={size} bytes (showing first 3 only)"
                            )
                        f.write(chunk)

                _ttsdbg(
                    f"generate_speech: finished streaming, chunks={chunk_idx}, total_bytes={total_bytes}, elapsed_ms={int((time.time()-start_ts)*1000)}"
                )
            except Exception as e:
                _ttsdbg(f"generate_speech: EXCEPTION during streaming: {e!r}")
                _ttsdbg(traceback.format_exc())
                raise  # preserve behavior
            try:
                import os
                if os.getenv("ELEVENLABS_DEBUG_REQUESTS") == "1":
                    print(
                        f"[EMO][REQ] model=eleven_v3 st={vs.get('stability', 0.35):.2f} sim={vs.get('similarity_boost', 0.85):.2f} style={vs.get('style', 0.50):.2f} boost={bool(vs.get('use_speaker_boost', True))}",
                        flush=True,
                    )
            except Exception:
                pass

            # Optional debug artifact: save the first successful raw stream once
            if TTS_DEBUG and total_bytes > 0:
                try:
                    os.makedirs("debug", exist_ok=True)
                    first_path = os.path.join("debug", "first_stream.mp3")
                    if not os.path.exists(first_path):
                        with open(first_path, "wb") as df, open(temp_file, "rb") as sf:
                            df.write(sf.read())
                        _ttsdbg("debug artifact: wrote debug/first_stream.mp3")
                except Exception as _e:
                    _ttsdbg(
                        f"debug artifact: failed to write raw stream: {_e!r}")

            # Verify file was created and has content
            if temp_file.exists() and temp_file.stat().st_size > 0:
                try:
                    fs = os.path.getsize(temp_file)
                    _ttsdbg(
                        f"generate_speech: temp file size on disk={fs} bytes at {temp_file}"
                    )
                except Exception as e:
                    _ttsdbg(
                        f"generate_speech: could not stat temp file: {e!r}")
                audio = AudioSegment.from_mp3(temp_file)
                _ttsdbg(
                    f"generate_speech: decoded segment duration_ms={len(audio)} channels={getattr(audio, 'channels', '?')} frame_rate={getattr(audio, 'frame_rate', '?')}"
                )
                if len(audio) < 50:
                    _ttsdbg(
                        "generate_speech: WARNING very short (<50ms) decoded audio — likely failure/silence path"
                    )
                # Ensure audio has minimum duration
                if len(audio) < 100:  # If less than 100ms, add some silence
                    audio = audio + AudioSegment.silent(duration=100)
                return audio
            else:
                st.warning(f"Failed to generate audio for: {text[:30]}...")
                _ttsdbg(
                    "generate_speech: streaming produced no file or zero bytes — returning 1000ms silence")
                return AudioSegment.silent(duration=1000)

        except Exception as e:
            st.error(f"Error generating speech for '{text[:30]}...': {e}")
            _ttsdbg("generate_speech: exception path — returning 1000ms silence")
            return AudioSegment.silent(duration=1000)

    def generate_preview(self, voice_id: str, text: str, project_name: str = "default"):
        """
        Generate a short voice preview using per-project voice settings.
        Mirrors generate_speech() but automatically loads the project's saved settings.
        """
        try:
            from utils.voice_settings import normalize_settings
            from utils.state_manager import load_project_voice_settings
            try:
                voice_settings = normalize_settings(
                    load_project_voice_settings(project_name))
            except Exception:
                voice_settings = {}
            return self.generate_speech(
                voice_id=voice_id,
                text=text,
                voice_settings=voice_settings,
            )
        except Exception:
            # Fallback: call with defaults if anything goes wrong
            return self.generate_speech(
                voice_id=voice_id,
                text=text,
                voice_settings={},
            )

    def load_sound_effect(self, effect_name):
        """FX removed: return short silence for compatibility."""
        return AudioSegment.silent(duration=300)

    def _save_to_organized_folder(self, audio_data: bytes, output_type: str, project_name: str):
        """Save audio to organized folder structure"""
        try:
            # Clean project name for filename
            clean_name = re.sub(r'[^\w\s-]', '', project_name).strip()
            clean_name = re.sub(r'[-\s]+', '_', clean_name)

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Determine output folder and filename
            if output_type == "teaser":
                folder = self.output_folders["teasers"]
                filename = f"{clean_name}_teaser_{timestamp}.mp3"
            elif output_type == "voice_test":
                folder = self.output_folders["voice_tests"]
                filename = f"voice_test_{timestamp}.mp3"
            elif output_type == "chapter":
                folder = self.output_folders["chapters"]
                filename = f"{clean_name}_chapter_{timestamp}.mp3"
            else:
                folder = self.output_folders["books"]
                filename = f"{clean_name}_{timestamp}.mp3"

            # Save file
            output_path = folder / filename
            with open(output_path, 'wb') as f:
                f.write(audio_data)

            st.success(f"💾 Audio saved to: {output_path}")

        except Exception as e:
            st.warning(f"Could not save to organized folder: {e}")

    def _ensure_compatible(self, segment: AudioSegment) -> AudioSegment:
        """Force audio segment to 16-bit PCM, stereo, 44.1kHz for safe merging"""
        if not segment:
            return AudioSegment.silent(duration=100)
        return segment.set_sample_width(2).set_channels(2).set_frame_rate(44100)

    def process_dialogue(self, dialogue_data, voice_assignments=None, output_type="chapter", project_name="project", batch_size: int = 20, progress_callback=None):
        """Process the entire dialogue with TTS and sound effects in batches to keep UI responsive.

        progress_callback (optional): callable accepting (completed_batches, total_batches)
        """
        try:
            _ttsdbg(f"process_dialogue: lines={len(dialogue_data)}")
        except Exception:
            pass
        # Load per-project voice settings once
        try:
            self._voice_settings = normalize_settings(
                load_project_voice_settings(project_name)
            )
        except Exception:
            self._voice_settings = None
        # Use custom voice assignments if provided
        if voice_assignments:
            working_voices = voice_assignments.copy()
        else:
            working_voices = self.character_voices

        total_items = len(dialogue_data)
        total_batches = (total_items + batch_size -
                         1) // batch_size if total_items > 0 else 0

        # Streamlit progress elements
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        # Prepare incremental final audio (start silent)
        final_audio = AudioSegment.silent(duration=0)

        def _append_segment(target: AudioSegment, segment: AudioSegment) -> AudioSegment:
            if segment and len(segment) > 0:
                return target + self._ensure_compatible(segment)
            return target

        completed_items = 0
        completed_batches = 0

        last_progress_items = 0
        progress_step = max(5, batch_size // 2)  # throttle UI updates

        for batch_start in range(0, total_items, batch_size):
            batch = dialogue_data[batch_start:batch_start + batch_size]
            batch_segments = []

            for entry in batch:
                if entry["type"] == "speech":
                    char = entry.get("character", "Unknown")
                    text_preview = entry["text"][:50] + \
                        ("..." if len(entry["text"]) > 50 else "")
                    status_text.text(
                        f"🎤 Generating speech for {char}: {text_preview}")
                    try:
                        _ttsdbg(
                            f"process_dialogue: line speaker={char} text_len={len(entry.get('text',''))}"
                        )
                    except Exception:
                        pass

                    voice_info = working_voices.get(char) or working_voices.get(
                        char.lower()) or entry.get("voice_id")
                    if isinstance(voice_info, dict):
                        voice_id = voice_info.get("voice_id") or voice_info.get(
                            "id") or voice_info.get("voiceId")
                    else:
                        voice_id = voice_info
                    if not voice_id:
                        voice_id = st.session_state.get(
                            "default_custom_voice", "default_voice_id_here")

                    # Retry TTS a few times
                    max_retries = 3
                    audio = None
                    for attempt in range(max_retries):
                        try:
                            audio = self.generate_speech(
                                voice_id=voice_id,
                                text=entry["text"],
                                model_id="eleven_v3",
                                voice_settings=self._voice_settings or {},
                            )
                            if len(audio) > 0:
                                break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                st.warning(
                                    f"Failed to generate speech after {max_retries} attempts: {e}")
                                audio = AudioSegment.silent(duration=1000)
                            else:
                                time.sleep(0.2)

                    if audio:
                        try:
                            _ttsdbg(
                                f"process_dialogue: generated segment duration_ms={len(audio)}"
                            )
                        except Exception:
                            pass
                        batch_segments.append(audio)

                # FX entries removed from sequence; keep branch for safety
                elif entry["type"] == "sound_effect":
                    seg = AudioSegment.silent(duration=300)
                    _ttsdbg("tts fallback: returning generated silence segment")
                    batch_segments.append(seg)

                elif entry["type"] == "pause":
                    duration = entry.get("duration", 500)
                    seg = AudioSegment.silent(duration=duration)
                    _ttsdbg("tts fallback: returning generated silence segment")
                    batch_segments.append(seg)

            # Incrementally merge this batch into final audio to avoid RAM spikes
            status_text.text("🔄 Merging batch audio segments...")
            try:
                for seg in batch_segments:
                    final_audio = _append_segment(final_audio, seg)
            except Exception as e:
                st.error(f"Error merging batch: {e}")

            # Update counters and progress (throttled)
            completed_items += len(batch)
            completed_batches += 1
            if (completed_items - last_progress_items) >= progress_step or completed_items == total_items:
                progress = completed_items / total_items if total_items else 1.0
                progress_bar.progress(min(1.0, progress))
                if progress_callback:
                    try:
                        progress_callback(completed_batches, total_batches)
                    except Exception:
                        pass
                last_progress_items = completed_items

            # Slightly longer sleep to reduce heartbeat pressure
            time.sleep(0.05)

        status_text.text("✨ Finalizing audio...")

        # Post-process final audio
        try:
            final_audio = self.apply_post_processing(final_audio)
        except Exception as e:
            st.warning(f"Skipping post-processing due to error: {e}")

        # Export to buffer with error handling
        try:
            try:
                _ttsdbg(
                    f"process_dialogue: final merged duration_ms={len(final_audio)}")
            except Exception:
                pass
            audio_buffer = io.BytesIO()
            final_audio.export(audio_buffer, format="mp3", bitrate="128k")
            audio_data = audio_buffer.getvalue()
            self._save_to_organized_folder(
                audio_data, output_type, project_name)
        except Exception as e:
            st.error(f"Error exporting audio: {e}")
            audio_buffer = io.BytesIO()
            AudioSegment.silent(duration=1000).export(
                audio_buffer, format="mp3")
            audio_data = audio_buffer.getvalue()

        progress_bar.progress(1.0)
        status_text.text("✅ Audio generation complete!")

        return audio_data

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

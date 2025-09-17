import streamlit as st
from parsers.openai_parser import OpenAIParser
from parsers.dialogue_parser import DialogueParser
from audio.generator import DialogueAudioGenerator
import io
from utils.downloads import get_audio_download_link
from audio.utils import get_flat_character_voices, get_flat_emotion_tags, CHARACTER_VOICES, EMOTION_TAGS
from audio.voice_manager import save_voice_mappings, load_voice_mappings, get_saved_voice_projects


def generate_audio_for_tab(text_content: str, tab_prefix: str, analysis_key: str, voice_assignments_key: str, output_type: str = "chapter", project_name: str = "project"):
    """Generate audio for a specific tab's content"""
    if not text_content.strip():
        st.error("Please enter some dialogue text!")
        return

    try:
        # Get analysis and voice assignments for this tab
        current_analysis = st.session_state.get(analysis_key)
        current_voice_assignments = st.session_state.get(voice_assignments_key)

        # Initialize generator and parser
        generator = DialogueAudioGenerator()
        parser = DialogueParser()

        # Parse dialogue
        dialogue_sequence = parser.parse_dialogue(
            text_content,
            current_voice_assignments
        )

        if not dialogue_sequence:
            st.error("No valid dialogue found!")
            return

        # Display summary before generation
        speech_count = sum(
            1 for entry in dialogue_sequence if entry["type"] == "speech")
        effect_count = sum(
            1 for entry in dialogue_sequence if entry["type"] == "sound_effect")
        st.info(
            f"📊 **Processing:** {speech_count} speech segments, {effect_count} sound effects")

        # Generate audio with batch progress tracking to keep UI responsive
        batch_progress = st.progress(0)
        batch_info = st.empty()

        def _on_batch_update(done, total):
            if total:
                batch_progress.progress(min(1.0, done / total))
            batch_info.write(f"Completed {done}/{total} batches")

        with st.spinner("Generating audio..."):
            audio_data = generator.process_dialogue(
                dialogue_sequence,
                current_voice_assignments,
                output_type=output_type,
                project_name=project_name,
                batch_size=20,
                progress_callback=_on_batch_update,
            )

        # Display success and download link
        st.success("✅ Audio generated successfully!")

        # Audio player
        st.audio(audio_data, format="audio/mp3")

        # Download button
        filename = f"{project_name}_{output_type}_audio.mp3"
        st.markdown(
            get_audio_download_link(audio_data, filename),
            unsafe_allow_html=True
        )

        # Cleanup
        generator.cleanup()

    except Exception as e:
        st.error(f"Error generating audio: {e}")
        st.error("Check your network connection and file paths.")


def create_teaser_generator_tab():
    """Create teaser generator interface for TikTok/Shorts content"""
    st.markdown("### 🎬 Teaser Line Generator")
    st.markdown(
        "Create **TikTok/Shorts-ready** teaser content (1-5 lines) for marketing purposes")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Project name for teaser
        project_name = st.text_input(
            "Project/Book Name:",
            value=st.session_state.get('teaser_project_name', ''),
            placeholder="Enter book or project name...",
            key="teaser_project_name"
        )

    with col2:
        if st.button("🗑️ Reset Teaser", type="secondary", use_container_width=True, key="reset_teaser_btn"):
            # Clear teaser-related session state
            keys_to_clear = [
                'teaser_text', 'teaser_project_name'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Reset completed.")

    # Handle emotion/effect additions from sidebar for teaser tab
    if 'emotion_to_add' in st.session_state and st.session_state.get('teaser_text', ''):
        st.session_state.teaser_text += f" {st.session_state.emotion_to_add}"
        del st.session_state.emotion_to_add
        st.success("Emotion added successfully.")
    # FX insert removed

    # Teaser text input
    teaser_text = st.text_area(
        "Enter 1-5 teaser lines:",
        value=st.session_state.get('teaser_text', ''),
        height=200,
        placeholder="""[Dante] (whispers)(excited): The security system is down. This is our chance.
[Luca] (frustrated): I still don't like this plan, Dante.
[Rafael] (mischievously): Relax, tesoro. What could go wrong?""",
        help="Perfect for TikTok/Shorts! Keep it short and punchy (1-5 lines max)",
        key="teaser_text_input"
    )

    # Update session state
    if teaser_text != st.session_state.get('teaser_text', ''):
        st.session_state.teaser_text = teaser_text

    # Quick stats
    if teaser_text.strip():
        lines = [line.strip() for line in teaser_text.split(
            '\n') if line.strip() and not line.startswith('#')]
        line_count = len(lines)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Lines", line_count)
        with col2:
            color = "🟢" if line_count <= 5 else "🔴"
            st.metric("📱 TikTok Ready",
                      f"{color} {'Yes' if line_count <= 5 else 'Too long'}")
        with col3:
            est_duration = line_count * 3  # Rough estimate: 3 seconds per line
            st.metric("⏱️ Est. Duration", f"~{est_duration}s")

        if line_count > 5:
            st.warning(
                "⚠️ Recommended: Keep teasers to 5 lines or less for social media")
        elif line_count == 0:
            st.info("💡 Add some dialogue lines to generate a teaser")

    # Generate teaser button
    if st.button("🎬 Generate Teaser Audio", type="primary", use_container_width=True, key="generate_teaser_btn"):
        if not teaser_text.strip():
            st.error("Please enter some teaser dialogue!")
        elif not project_name.strip():
            st.error("Please enter a project name!")
        else:
            try:
                # Initialize generator and parser
                generator = DialogueAudioGenerator()
                parser = DialogueParser()

                # Parse dialogue
                dialogue_sequence = parser.parse_dialogue(teaser_text)

                if not dialogue_sequence:
                    st.error("No valid dialogue found!")
                else:
                    # Generate audio
                    with st.spinner("Generating teaser audio..."):
                        audio_data = generator.process_dialogue(
                            dialogue_sequence,
                            output_type="teaser",
                            project_name=project_name
                        )

                    # Display success
                    st.success("✅ Teaser audio generated!")

                    # Audio player
                    st.audio(audio_data, format="audio/mp3")

                    # Download button
                    filename = f"{project_name}_teaser.mp3"
                    st.markdown(
                        get_audio_download_link(audio_data, filename),
                        unsafe_allow_html=True
                    )

                    # Cleanup
                    generator.cleanup()

            except Exception as e:
                st.error(f"Error generating teaser: {e}")


def create_emotion_preview_tab():
    """Create emotion preview interface for testing voices with emotions"""
    st.markdown("### 😊 Emotion Preview")
    st.markdown("Test how different voices sound with various emotions")

    # Voice selection
    all_voices = {}
    for category, chars in CHARACTER_VOICES.items():
        for char, vid in chars.items():
            all_voices[char] = {
                "voice_id": vid,
                "character": char
            }

    col1, col2 = st.columns(2)

    with col1:
        selected_voice_label = st.selectbox(
            "Select Voice to Test:",
            options=list(all_voices.keys()),
            key="emotion_preview_voice"
        )

        selected_voice = all_voices[selected_voice_label]

        # Show voice info
        st.info(f"**{selected_voice['character']}**")

    with col2:
        # Emotion selection
        emotion_options = ["None"] + list(get_flat_emotion_tags().keys())
        selected_emotion = st.selectbox(
            "Select Emotion to Test:",
            options=emotion_options,
            key="emotion_preview_emotion"
        )

    # Test text input
    test_text = st.text_input(
        "Test Text:",
        value="Hello there, this is a voice test with the selected emotion.",
        key="emotion_preview_text"
    )

    # Build preview text with emotion
    if selected_emotion != "None":
        preview_text = f"[{selected_emotion}] {test_text}"
    else:
        preview_text = test_text

    st.text(f"Preview: {preview_text}")

    # Generate preview button
    if st.button("🎤 Generate Voice Preview", type="primary", use_container_width=True, key="generate_emotion_preview"):
        if not test_text.strip():
            st.error("Please enter some test text!")
        else:
            try:
                # Initialize generator
                generator = DialogueAudioGenerator()

                # Generate audio
                with st.spinner("Generating voice preview..."):
                    audio = generator.generate_speech(
                        voice_id=selected_voice["voice_id"],
                        text=preview_text
                    )

                    # Convert to bytes
                    audio_buffer = io.BytesIO()
                    audio.export(audio_buffer, format="mp3", bitrate="128k")
                    audio_data = audio_buffer.getvalue()

                    # Save to voice tests folder
                    generator._save_to_organized_folder(
                        audio_data,
                        "voice_test",
                        f"{selected_voice['character']}_{selected_emotion}"
                    )

                st.success("✅ Voice preview generated!")

                # Audio player
                st.audio(audio_data, format="audio/mp3")

                # Download button
                filename = f"{selected_voice['character']}_{selected_emotion}_preview.mp3"
                st.markdown(
                    get_audio_download_link(audio_data, filename),
                    unsafe_allow_html=True
                )

                # Cleanup
                generator.cleanup()

            except Exception as e:
                st.error(f"Error generating preview: {e}")

    # Emotion reference guide
    with st.expander("📚 Available Emotions Reference"):
        for category, emotions in EMOTION_TAGS.items():
            st.markdown(f"**{category.replace('_', ' ').title()}:**")
            emotion_list = list(emotions.keys())
            cols = st.columns(3)
            for i, emotion in enumerate(emotion_list):
                with cols[i % 3]:
                    st.markdown(f"• {emotion}")
            st.divider()


def create_voice_manager_tab():
    """Create comprehensive voice manager interface"""
    st.markdown("### 🎭 Voice Manager")
    st.markdown("Assign AI voices to characters and save mappings for projects")

    # Project management section
    st.markdown("#### 📁 Project Management")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        project_name = st.text_input(
            "Project Name:",
            value=st.session_state.get('vm_project_name', ''),
            placeholder="Enter project/book name...",
            key="vm_project_name"
        )

    with col2:
        # Load existing project
        saved_projects = get_saved_voice_projects()
        if saved_projects:
            selected_project = st.selectbox(
                "Load Saved Project:",
                options=[""] + saved_projects,
                key="vm_load_project"
            )

            if st.button("📂 Load", type="secondary", use_container_width=True):
                if selected_project:
                    loaded_mappings = load_voice_mappings(selected_project)
                    if loaded_mappings:
                        st.session_state.vm_voice_mappings = loaded_mappings
                        st.session_state.vm_project_name = selected_project
                        st.info("Project loaded successfully.")

    with col3:
        # Save current project
        if st.button("💾 Save Project", type="secondary", use_container_width=True):
            if project_name.strip() and 'vm_voice_mappings' in st.session_state:
                save_file = save_voice_mappings(
                    st.session_state.vm_voice_mappings,
                    project_name
                )
                st.success(f"✅ Project saved: {save_file.name}")
            else:
                st.error(
                    "Please enter project name and create voice mappings first!")

    st.markdown("---")

    # Character management
    st.markdown("#### 👥 Character Voice Assignments")

    # Initialize voice mappings if not exists
    if 'vm_voice_mappings' not in st.session_state:
        st.session_state.vm_voice_mappings = get_flat_character_voices().copy()

    # Add new character section
    with st.expander("➕ Add New Character", expanded=False):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            new_char_name = st.text_input(
                "Character Name:",
                placeholder="Enter new character name...",
                key="vm_new_char_name"
            )

        with col2:
            # All available voices for dropdown
            all_voice_options = {}
            for category, chars in CHARACTER_VOICES.items():
                for char, vid in chars.items():
                    all_voice_options[char] = vid

            selected_voice_for_new = st.selectbox(
                "Assign Voice:",
                options=list(all_voice_options.keys()),
                key="vm_new_char_voice"
            )

        with col3:
            if st.button("➕ Add", type="secondary", use_container_width=True, key="vm_add_char"):
                if new_char_name.strip():
                    voice_id = all_voice_options[selected_voice_for_new]
                    st.session_state.vm_voice_mappings[new_char_name.strip(
                    )] = voice_id
                    st.success(f"✅ Added {new_char_name} with voice!")
                else:
                    st.error("Please enter a character name!")

    # Display current character assignments
    if st.session_state.vm_voice_mappings:
        st.markdown("#### 🎯 Current Assignments")

        # Get all available voices for dropdowns
        all_voice_options = {}
        for category, chars in CHARACTER_VOICES.items():
            for char, vid in chars.items():
                all_voice_options[char] = vid

        # Create assignment interface
        characters_to_remove = []

        for char_name, current_voice_id in st.session_state.vm_voice_mappings.items():
            col1, col2, col3 = st.columns([2, 3, 1])

            with col1:
                st.markdown(f"**{char_name}**")

            with col2:
                # Find current selection
                current_selection = None
                for label, voice_id in all_voice_options.items():
                    if voice_id == current_voice_id:
                        current_selection = label
                        break

                if current_selection is None:
                    current_selection = list(all_voice_options.keys())[0]

                # Voice selection dropdown
                new_voice_selection = st.selectbox(
                    f"Voice for {char_name}",
                    options=list(all_voice_options.keys()),
                    index=list(all_voice_options.keys()).index(
                        current_selection),
                    key=f"vm_voice_select_{char_name}",
                    label_visibility="collapsed"
                )

                # Update if changed
                new_voice_id = all_voice_options[new_voice_selection]
                if new_voice_id != current_voice_id:
                    st.session_state.vm_voice_mappings[char_name] = new_voice_id
                    st.info("Voice updated successfully.")

            with col3:
                if st.button("🗑️", key=f"vm_remove_{char_name}", help=f"Remove {char_name}"):
                    characters_to_remove.append(char_name)

        # Remove characters if requested
        for char_to_remove in characters_to_remove:
            del st.session_state.vm_voice_mappings[char_to_remove]
            st.success("Character removed.")

    else:
        st.info(
            "No character voice assignments yet. Add characters above or load a saved project.")

    # Available voices reference
    st.markdown("---")
    st.markdown("#### 🎤 Available Voices")

    for category, characters in CHARACTER_VOICES.items():
        with st.expander(f"{category} ({len(characters)} voices)"):
            for char, vid in characters.items():
                col1, col2, col3 = st.columns([2, 3, 1])

                with col1:
                    st.markdown(f"**{char}**")

                with col2:
                    st.code(vid, language=None)

                with col3:
                    # Quick test button
                    if st.button("🔊 Test", key=f"vm_test_{char}", help=f"Quick voice test for {char}"):
                        try:
                            generator = DialogueAudioGenerator()
                            test_audio = generator.generate_speech(
                                voice_id=vid,
                                text=f"Hello, I am {char}. This is how my voice sounds."
                            )
                            audio_buffer = io.BytesIO()
                            test_audio.export(audio_buffer, format="mp3")
                            audio_data = audio_buffer.getvalue()
                            st.audio(audio_data, format="audio/mp3")
                            generator.cleanup()
                        except Exception as e:
                            st.error(f"Error testing voice: {e}")


def create_raw_parser_tab(get_known_characters_callable):
    import streamlit as st

    st.markdown("### 📚 Raw Text → Dialogue Parser")
    st.markdown(
        "Paste raw book text below. The parser will detect quotes, infer speakers from narration like _\"…\" said Dante_, assign basic emotions (e.g., whispered → (whispers)), and optionally add narration lines as [Narrator]."
    )

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        include_narration = st.checkbox(
            "Include Narration as [Narrator]", value=True, key="raw_inc_narr")
    with col2:
        attach_fx = False

    # Friendly notice about character handling
    st.info(
        """
        🤖 AI Character Detection is enabled. The system automatically recognizes and attributes speakers
        using long-range context (names, pronouns, and narration cues). No manual character setup required.
        """
    )

    # Character input removed: parser will use predefined characters from configuration only

    raw_text = st.text_area(
        "Raw Prose:",
        height=280,
        placeholder=(
            "Example:\n"
            "Dante’s eyes narrowed. \"The security system is down,\" he whispered. \"This is our chance.\"\n"
            "Luca sighed. \"I still don't like this plan, Dante.\"\n"
            "\"Relax, tesoro. What could go wrong?\" Rafael said mischievously.\n"
            "Nikolai said coldly, \"Everything. That’s what experience teaches you.\"\n"
            "There was a sharp gasp as the door slammed."
        ),
        key="raw_parser_input",
    )

    # --- Convert action without forced reruns
    if st.button("🔍 Convert Raw → Dialogue", type="primary", use_container_width=True, key="raw_convert_btn"):
        if not raw_text.strip():
            st.error("Please paste some raw prose first.")
        else:
            with st.spinner("Generating..."):
                parser = OpenAIParser(
                    include_narration=include_narration
                )
                result = parser.convert(raw_text)

                st.session_state["raw_last_formatted_text"] = result.formatted_text
                st.session_state["raw_last_dialogues"] = result.dialogues
                st.session_state["raw_last_stats"] = result.stats
                st.session_state["raw_last_ambiguities"] = getattr(
                    result, "ambiguities", [])
                if "ambiguity_resolutions" not in st.session_state:
                    st.session_state["ambiguity_resolutions"] = {}
                st.session_state["raw_parsed_ready"] = True
            st.success("✅ Parsed successfully.")

    # --- Results area: rendered whenever we have a parsed result in session
    if st.session_state.get("raw_parsed_ready") and st.session_state.get("raw_last_formatted_text"):
        st.success("✅ Parsed successfully.")

        stats = st.session_state.get("raw_last_stats", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Quotes",   stats.get("quotes_found", 0))
        with c2:
            st.metric("Lines",    stats.get("lines_emitted", 0))
        with c3:
            st.metric("From after", stats.get("speaker_from_after", 0))
        with c4:
            st.metric("From before", stats.get("speaker_from_before", 0))
        with c5:
            st.metric("Narration", stats.get("narration_blocks", 0))

        # --- Ambiguity resolution panel ---
        ambiguities = st.session_state.get("raw_last_ambiguities", [])
        if ambiguities:
            st.markdown(f"#### 🧩 Resolve Ambiguities ({len(ambiguities)})")
            apply_all_map = {}
            for amb in ambiguities:
                line_id = amb.get("id")
                quote = amb.get("text", "")
                candidates = amb.get("candidates", [])
                st.markdown(f"**Quote:** \"{quote}\"")
                # Allow adding a new candidate
                options = candidates + ["➕ New character…"]
                sel = st.selectbox(
                    f"Select speaker for {line_id}", options=options,
                    key=f"amb_sel_{line_id}"
                )
                new_name = None
                if sel == "➕ New character…":
                    new_name = st.text_input(
                        f"Enter new character for {line_id}", key=f"amb_new_{line_id}")
                apply_all = st.checkbox(
                    "Apply to all similar", value=False, key=f"amb_all_{line_id}")
                if st.button("Apply", key=f"amb_apply_{line_id}"):
                    final_choice = (new_name.strip() if new_name else sel)
                    if final_choice and final_choice != "➕ New character…":
                        st.session_state["ambiguity_resolutions"][line_id] = final_choice
                        # Apply to dialogues and formatted text
                        resolved = 0
                        for d in st.session_state.get("raw_last_dialogues", []):
                            if d.get("character") == "Ambiguous" and d.get("text") == quote:
                                d["character"] = final_choice
                                resolved += 1
                                if not apply_all:
                                    break
                        # Rebuild formatted text without [Ambiguous]
                        lines = []
                        for d in st.session_state.get("raw_last_dialogues", []):
                            em = "".join(
                                [f"({e})" for e in d.get("emotions", [])])
                            lines.append(
                                f"[{d.get('character')}] {em}: {d.get('text')}".strip())
                        st.session_state["raw_last_formatted_text"] = "\n".join(
                            lines)
                        # Remove resolved ambiguity entries
                        st.session_state["raw_last_ambiguities"] = [
                            a for a in st.session_state["raw_last_ambiguities"]
                            if not (a.get("text") == quote and (apply_all or a.get("id") == line_id))
                        ]
                        st.success(
                            f"Applied '{final_choice}' to {resolved} line(s).")
                        st.rerun()

        st.markdown("#### ▶ Standardized Output")
        st.code(
            st.session_state["raw_last_formatted_text"], language="markdown")

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("→ Send to Main Generator", key="raw_send_to_main", type="primary", use_container_width=True):
                # 1) Hand the parsed text to the Main tab
                st.session_state.dialogue_text = st.session_state["raw_last_formatted_text"]
                # 2) Clear Main analysis so user re-parses there (optional)
                for k in ("paste_text_analysis", "paste_formatted_dialogue", "paste_parsed_dialogues", "paste_voice_assignments"):
                    st.session_state.pop(k, None)
                # 3) Switch tabs logically
                st.session_state.current_tab = "main"
                st.info("Parsed output sent to Main Generator.")

        with colB:
            if st.button("🗑 Reset Parsed Output", key="raw_reset", type="secondary", use_container_width=True):
                for k in ("raw_last_formatted_text", "raw_last_dialogues", "raw_last_stats", "raw_parsed_ready"):
                    st.session_state.pop(k, None)
                st.success("Reset completed.")

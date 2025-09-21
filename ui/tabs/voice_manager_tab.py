import streamlit as st
import io
from audio.generator import DialogueAudioGenerator
from audio.utils import get_flat_character_voices, CHARACTER_VOICES
from audio.voice_manager import save_voice_mappings, load_voice_mappings, get_saved_voice_projects


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

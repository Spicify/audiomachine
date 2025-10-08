import streamlit as st
import io
from audio.generator import DialogueAudioGenerator
from audio.utils import get_flat_emotion_tags, CHARACTER_VOICES, EMOTION_TAGS
from utils.downloads import get_audio_download_link


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
                    project_name = st.session_state.get(
                        "current_project") or "default"
                    audio = generator.generate_preview(
                        voice_id=selected_voice["voice_id"],
                        text=preview_text,
                        project_name=project_name,
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

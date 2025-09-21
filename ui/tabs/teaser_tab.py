import streamlit as st
import io
from audio.generator import DialogueAudioGenerator
from parsers.dialogue_parser import DialogueParser
from utils.downloads import get_audio_download_link


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

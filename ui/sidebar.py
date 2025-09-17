import streamlit as st
from audio.utils import CHARACTER_VOICES, EMOTION_TAGS


def create_navigation_sidebar():
    """Create enhanced sidebar with navigation and resources"""
    with st.sidebar:
        # User info
        if st.session_state.get('authenticated'):
            user_type = st.session_state.get('user_type', 'unknown')
            st.success(f"✅ Logged in as: **{user_type.title()}**")

            if st.button("🚪 Logout", type="secondary", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.user_type = None
                st.rerun()

            st.markdown("---")

        # Navigation buttons
        st.markdown("### 🧭 Navigation")

        # Initialize current tab if not set
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = "main"

        nav_buttons = [
            ("main", "📝 Main Generator", "Main dialogue generation"),
            ("teaser", "🎬 Teaser Generator", "TikTok/Shorts teasers"),
            ("emotion", "😊 Emotion Preview", "Test voices with emotions"),
            ("voice_manager", "🎭 Voice Manager", "Manage character voices"),
            ("raw", "📚 Raw Parser", "Convert raw prose to dialogue"),
            ("history", "🕓 History", "Completed and in-progress projects")
        ]

        for tab_key, tab_label, tab_help in nav_buttons:
            if st.button(
                tab_label,
                key=f"nav_{tab_key}",
                use_container_width=True,
                type="primary" if st.session_state.current_tab == tab_key else "secondary",
                help=tab_help
            ):
                st.session_state.current_tab = tab_key
                st.rerun()

        st.markdown("---")

        # Resources section
        st.title("🎭 Resources")

        # Emotion Tags Accordion
        with st.expander("😊 Emotion Tags", expanded=False):
            for category, emotions in EMOTION_TAGS.items():
                st.markdown(f"**{category.replace('_', ' ').title()}:**")
                cols = st.columns(2)
                for i, (emotion, tag) in enumerate(emotions.items()):
                    with cols[i % 2]:
                        if st.button(f"({emotion})", key=f"emotion_{emotion}", use_container_width=True):
                            # Store emotion to add to the current active tab
                            st.session_state.emotion_to_add = f"({emotion})"
                st.divider()

        # Characters Accordion
        with st.expander("👥 Available Characters", expanded=False):
            for category, characters in CHARACTER_VOICES.items():
                st.markdown(f"**{category}**")

                for char, vid in characters.items():
                    with st.container():
                        st.markdown(f"**{char}**")
                        st.code(vid, language=None)
                st.divider()

        # Sound Effects removed from application

        # Output folders info
        with st.expander("📁 Output Folders", expanded=False):
            st.markdown("""
            **Organized Output Structure:**
            - 📁 `audio_output/teasers/` - TikTok/Shorts clips
            - 📁 `audio_output/chapters/` - Full chapters
            - 📁 `audio_output/voice_tests/` - Voice previews
            - 📁 `audio_output/books/` - Complete books
            - 📁 `voice_mappings/` - Saved voice assignments
            """)

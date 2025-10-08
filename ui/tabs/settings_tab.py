import streamlit as st
from utils.voice_settings import DEFAULT_VOICE_SETTINGS, TOOLTIPS, normalize_settings
from utils.state_manager import load_project_voice_settings, save_project_voice_settings


def render(project_name: str):
    # Title & unified info
    st.header("Voice Settings")
    st.info(
        "Tune voice delivery for richer emotion. These settings apply to all new generations for this project. "
        "These settings complement inline audio tags like [sad], [whispering], [laughing]."
    )

    saved = load_project_voice_settings(project_name or "default")
    eff = normalize_settings(saved)

    # Controls with larger, bold labels and tighter proximity for first three
    st.markdown("<div style='font-size:1.1rem; font-weight:600; margin-bottom:6px;'>Stability</div>",
                unsafe_allow_html=True)
    stability = st.slider(
        "Stability",
        0.0,
        1.0,
        eff["stability"],
        help="Controls how consistent or expressive the voice is. Lower values produce more emotional variation, higher values make speech more stable and neutral.",
    )
    st.divider()

    st.markdown("<div style='font-size:1.1rem; font-weight:600; margin-bottom:6px;'>Clarity + Similarity</div>",
                unsafe_allow_html=True)
    sim = st.slider(
        "Clarity + Similarity",
        0.0,
        1.0,
        eff["similarity_boost"],
        help="Balances clarity and similarity to the reference voice. Higher values sound clearer and closer to the original voice; lower values allow more variation and texture.",
    )
    st.divider()

    st.markdown("<div style='font-size:1.1rem; font-weight:600; margin-bottom:6px;'>Style</div>",
                unsafe_allow_html=True)
    style = st.slider(
        "Style",
        0.0,
        1.0,
        eff["style"],
        help="Adjusts stylistic intensity — how dynamic and performative the speech feels. Higher values make the voice sound more dramatic; lower values keep delivery flatter.",
    )
    st.divider()

    # Speaker Boost: increase title size/weight, keep existing spacing
    st.markdown("<div style='font-size:1.1rem; font-weight:600;'>Speaker Boost</div>",
                unsafe_allow_html=True)
    boost = st.toggle(
        "    ",
        eff["use_speaker_boost"],
        help="Enhances presence and volume, making the voice sound closer and more prominent. Useful for dialogue-heavy or soft-spoken scenes.",
    )

    # Buttons separated visually
    st.markdown(" ")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Save"):
            payload = {
                "stability": float(stability),
                "similarity_boost": float(sim),
                "style": float(style),
                "use_speaker_boost": bool(boost),
            }
            save_project_voice_settings(project_name or "default", payload)
            st.success("Saved. New generations will use these settings.")
    with col2:
        if st.button("Reset to Default"):
            save_project_voice_settings(
                project_name or "default", DEFAULT_VOICE_SETTINGS)
            st.success(
                "Reset to defaults. Reload the page if controls didn’t update.")

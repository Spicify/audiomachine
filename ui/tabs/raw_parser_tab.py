import streamlit as st
from parsers.openai_parser.openai_parser import OpenAIParser


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

    # --- Ensure session state keys exist and persist across reruns (baseline set) ---
    if "stream_dialogues" not in st.session_state:
        st.session_state["stream_dialogues"] = []
    if "stream_lines" not in st.session_state:
        st.session_state["stream_lines"] = []
    if "stream_ambiguities" not in st.session_state:
        st.session_state["stream_ambiguities"] = {}
    if "stream_progress" not in st.session_state:
        st.session_state["stream_progress"] = {"idx": 0, "total": 1}
    if "ambiguity_resolutions" not in st.session_state:
        st.session_state["ambiguity_resolutions"] = {}

    # --- Convert action (synchronous streaming in UI thread) ---
    if st.button("🔍 Convert Raw → Dialogue", type="primary", use_container_width=True, key="raw_convert_btn"):
        if not raw_text.strip():
            st.error("Please paste some raw prose first.")
        else:
            # Reset buffers
            st.session_state["stream_dialogues"] = []
            st.session_state["stream_lines"] = []
            st.session_state["stream_ambiguities"] = {}
            st.session_state["stream_progress"] = {"idx": 0, "total": 1}

            parser = OpenAIParser(include_narration=include_narration)
            progress_bar = st.progress(0)
            lines_area = st.container()

            with st.spinner("Generating..."):
                for chunk in parser.convert_streaming(raw_text):
                    total = max(1, chunk.get("total_chunks") or 1)
                    idx = max(1, chunk.get("chunk_index") or 1)

                    # Append dialogues and render immediately
                    with lines_area:
                        for d in (chunk.get("dialogues") or []):
                            st.session_state["stream_dialogues"].append(d)
                            em = "".join(
                                [f"({e})" for e in d.get("emotions", [])])
                            st.session_state["stream_lines"].append(
                                f"[{d.get('character')}] {em}: {d.get('text')}")
                            st.write(st.session_state["stream_lines"][-1])

                    # Merge ambiguities (dedup by id)
                    for amb in (chunk.get("ambiguities") or []):
                        lid = amb.get("id")
                        if lid and lid not in st.session_state["stream_ambiguities"]:
                            st.session_state["stream_ambiguities"][lid] = amb

                    # Progress update
                    st.session_state["stream_progress"] = {
                        "idx": idx, "total": total}
                    progress_bar.progress(min(1.0, idx / total))

            # Finalize
            result = parser.finalize_stream(
                st.session_state.get("stream_dialogues") or [], include_narration=include_narration)
            st.session_state["raw_last_formatted_text"] = result.formatted_text
            st.session_state["raw_last_dialogues"] = result.dialogues
            st.session_state["raw_last_stats"] = result.stats
            st.session_state["raw_last_ambiguities"] = getattr(
                result, "ambiguities", [])
            st.session_state["raw_parsed_ready"] = True

    # --- Render streaming view from session buffers ---
    progress_holder = st.empty()
    lines_area = st.empty()
    amb_area = st.empty()

    prog = st.session_state.get("stream_progress", {"idx": 0, "total": 1})
    idx = max(0, prog.get("idx", 0))
    total = max(1, prog.get("total", 1))
    frac = min(1.0, (idx / total) if total else 0.0)
    percent = int(frac * 100)
    with progress_holder.container():
        prog_widget = st.progress(0)
        prog_widget.progress(
            min(1.0, (idx / total) if total else 0.0), text=f"Progress: {percent}%")

    # Show streamed lines (already written incrementally during parse); nothing else to do here

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

        # Final consolidated output
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

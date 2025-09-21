import streamlit as st
from parsers.openai_parser.openai_parser import OpenAIParser
from parsers.openai_parser.chunker import build_chunks
from queue import Queue, Empty
import threading
import time


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

    # Persistent placeholders for progress and status (always present)
    progress_text_placeholder = st.empty()
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

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
    # Per-chunk storage and finalization flag
    if "stream_chunks" not in st.session_state:
        st.session_state["stream_chunks"] = []
    if "raw_finalized" not in st.session_state:
        st.session_state["raw_finalized"] = False
    if "parsing_in_progress" not in st.session_state:
        st.session_state["parsing_in_progress"] = False
    if "ambiguity_custom" not in st.session_state:
        # per-ambiguity custom entered names {amb_id: custom_name}
        st.session_state["ambiguity_custom"] = {}
    if "ambiguity_choices" not in st.session_state:
        # stores user selections per ambiguity keyed by "{chunk_index}:{amb_id}"
        st.session_state["ambiguity_choices"] = {}

    # --- Convert button (always visible) ---
    clicked_convert = st.button(
        "🔍 Convert Raw → Dialogue", type="primary", use_container_width=True, key="raw_convert_btn")

    if clicked_convert:
        if not raw_text.strip():
            st.error("Please paste some raw prose first.")
        else:
            # Mark parsing phase and reset buffers for a fresh run
            st.session_state["parsing_in_progress"] = True
            st.session_state["stream_dialogues"] = []
            st.session_state["stream_lines"] = []
            st.session_state["stream_ambiguities"] = {}
            st.session_state["stream_progress"] = {"idx": 0, "total": 1}
            st.session_state["stream_chunks"] = []
            st.session_state["raw_finalized"] = False
            print("[raw_parser_tab] Convert clicked → starting parse")

    # --- Phase 1: run parsing with live progress ---
    if st.session_state.get("parsing_in_progress"):
        parser = OpenAIParser(include_narration=include_narration)
        # Precompute estimated progress by tokens for smooth animation
        preview_chunks = build_chunks(
            raw_text, max_tokens=parser.max_tokens_per_chunk, model=parser.model, overlap_sentences=2)
        token_counts = [max(1, int(getattr(c, "token_count", 0) or 0))
                        for c in preview_chunks]
        total_tokens = max(1, sum(token_counts))
        cumulative_targets = []
        running = 0
        for tc in token_counts:
            running += tc
            cumulative_targets.append(int(running * 100 / total_tokens))

        current_percent = 0

        def render_progress(pct: int):
            pct = max(0, min(100, int(pct)))
            progress_text_placeholder.markdown(f"**Progress: {pct}%**")
            progress_placeholder.progress(pct / 100.0)

        # Initialize progress UI at 0% before any chunk is processed
        progress_text_placeholder.markdown("**Progress: 0%**")
        progress_placeholder.progress(0.0)
        status_placeholder.caption(
            "Please wait while all chunks are processed.")
        print("[raw_parser_tab] Parsing loop starting")
        # Background worker to fetch chunks while we animate progress
        q: Queue = Queue()
        chunks_by_idx = {}

        def _worker():
            try:
                for ch in parser.convert_streaming(raw_text):
                    q.put(ch)
            finally:
                q.put({"_done": True})

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        with st.spinner("Generating..."):
            expected = len(token_counts) or 1
            for idx in range(1, expected + 1):
                # Determine target percent for this chunk using token-based fraction
                target_percent = cumulative_targets[idx - 1] if (idx - 1) < len(
                    cumulative_targets) else int(idx * 100 / max(1, expected))
                # Estimate duration: ~120s per ~1000 tokens; overestimate for small chunks
                tokens_for_chunk = token_counts[idx -
                                                1] if (idx - 1) < len(token_counts) else 1000
                est_seconds = 120.0 * (tokens_for_chunk / 1000.0)
                if tokens_for_chunk < 600:
                    est_seconds = max(est_seconds, 100.0)

                # Animate illusion progress while waiting for actual chunk result
                start = current_percent
                end = max(start, target_percent)
                steps = max(1, end - start)
                # Use token-based estimate directly; gentle lower bound only
                per_step = max(0.05, est_seconds / max(1, steps))
                print(
                    f"[progress] chunk={idx}, tokens={tokens_for_chunk}, est_seconds={est_seconds:.1f}, steps={steps}, per_step={per_step:.3f}, start={start}, target={end}")

                received = False
                while current_percent < end:
                    # Drain any available results without blocking
                    try:
                        while True:
                            ch = q.get_nowait()
                            if ch.get("_done"):
                                break
                            ci = max(1, ch.get("chunk_index") or 1)
                            chunks_by_idx[ci] = ch
                            if ci == idx:
                                received = True
                                break
                    except Empty:
                        pass

                    if received:
                        current_percent = end
                        render_progress(current_percent)
                        break

                    current_percent += 1
                    render_progress(current_percent)
                    time.sleep(per_step)

                # Ensure we have the chunk result; block minimally if still not arrived
                while idx not in chunks_by_idx:
                    try:
                        ch = q.get(timeout=0.2)
                        if ch.get("_done"):
                            break
                        ci = max(1, ch.get("chunk_index") or 1)
                        chunks_by_idx[ci] = ch
                    except Empty:
                        if current_percent < end:
                            current_percent += 1
                            render_progress(current_percent)

                # Process the chunk data now
                ch = chunks_by_idx.get(idx)
                if ch:
                    st.session_state["stream_chunks"].append({
                        "index": idx,
                        "dialogues": list(ch.get("dialogues") or []),
                        "ambiguities": list(ch.get("ambiguities") or []),
                    })
                    for d in (ch.get("dialogues") or []):
                        st.session_state["stream_dialogues"].append(d)
                    for amb in (ch.get("ambiguities") or []):
                        lid = amb.get("id")
                        if lid and lid not in st.session_state["stream_ambiguities"]:
                            st.session_state["stream_ambiguities"][lid] = amb
                    st.session_state["stream_progress"] = {
                        "idx": idx, "total": expected}
                    print(
                        f"[raw_parser_tab] progress {idx}/{expected} → {current_percent}%")

        # Fill to 100%
        for p in range(current_percent + 1, 101):
            progress_text_placeholder.markdown(f"**Progress: {p}%**")
            progress_placeholder.progress(p / 100.0)
            time.sleep(0.01)

        # Done: sort and clear spinner
        st.session_state["stream_chunks"].sort(key=lambda c: c.get("index", 0))
        status_placeholder.empty()
        st.session_state["parsing_in_progress"] = False
        print("[raw_parser_tab] Parsing completed")

    # --- Phase 2: After parsing finishes, show chunked outputs with ambiguity resolution ---
    if st.session_state.get("stream_chunks") and not st.session_state.get("raw_finalized"):
        print("[raw_parser_tab] Phase 2 rendering with", len(
            st.session_state.get("stream_chunks", [])), "chunks")
        for i, ch in enumerate(st.session_state["stream_chunks"], start=1):
            with st.container():
                st.markdown(f"**Chunk {i} Output**")
                # Pretty-format dialogues for display
                lines = []
                for d in (ch.get("dialogues") or []):
                    em = "".join(
                        [f"({e})" for e in (d.get("emotions") or [])])
                    lines.append(
                        f"[{d.get('character')}] {em}: {d.get('text')}")
                if lines:
                    st.code("\n".join(lines), language="markdown")

                    # Ambiguities for this chunk
                    ambs = list(ch.get("ambiguities") or [])
                    st.markdown(f"{len(ambs)} Ambiguities for Chunk {i}")
                    if ambs:
                        # Light shading/indent via container nesting
                        with st.container():
                            for amb in ambs:
                                amb_id = amb.get("id")
                                # Allowed options: characters that appeared in this chunk + optional custom
                                chunk_chars = []
                                for _d in (ch.get("dialogues") or []):
                                    nm = str(_d.get("character") or "").strip()
                                    if not nm:
                                        continue
                                    if nm.lower() in ("ambiguous",):
                                        continue
                                    # We keep Narrator as option in case narration attribution is desired
                                    if nm not in chunk_chars:
                                        chunk_chars.append(nm)

                                # If user previously entered a custom name for this ambiguity, include it first
                                custom_name = (st.session_state.get(
                                    "ambiguity_custom") or {}).get(amb_id)
                                options: list[str] = []
                                if custom_name and custom_name not in chunk_chars:
                                    options.append(custom_name)
                                options.extend(chunk_chars)
                                add_new_label = "➕ Add New Character"
                                if add_new_label not in options:
                                    options.append(add_new_label)

                                # Determine current selection from local choices
                                choice_key = f"{i}:{amb_id}"
                                stored_choice = st.session_state["ambiguity_choices"].get(
                                    choice_key)
                                if stored_choice and stored_choice in options:
                                    current_value = stored_choice
                                elif custom_name:
                                    current_value = custom_name
                                elif options:
                                    # default to first non-add option if available
                                    current_value = options[0]
                                else:
                                    current_value = ""

                                label = f"Select character for: {amb.get('text','')[:60]}{'…' if len(amb.get('text',''))>60 else ''}"
                                try:
                                    idx = options.index(
                                        current_value) if current_value in options else 0
                                except Exception:
                                    idx = 0
                                selection = st.selectbox(
                                    label,
                                    options=options if options else [""],
                                    index=idx if options else 0,
                                    key=f"amb_sel_{i}_{amb_id}",
                                    help="ℹ️ All Ambiguities will be updated once parsing is complete."
                                )

                                # If user selects add-new, show input and persist custom
                                if selection == add_new_label:
                                    new_val = st.text_input(
                                        "Enter new character name:",
                                        value=(custom_name or ""),
                                        key=f"amb_new_{i}_{amb_id}"
                                    ).strip()
                                    if new_val:
                                        st.session_state["ambiguity_custom"][amb_id] = new_val
                                        st.session_state["ambiguity_choices"][choice_key] = new_val
                                else:
                                    # Persist chosen existing option
                                    st.session_state["ambiguity_choices"][choice_key] = selection

        # Consolidation action
        st.markdown("---")
        if st.button("Update Ambiguities", type="primary", use_container_width=True, key="apply_ambiguities"):
            # Apply user selections and build final consolidated output
            updated_dialogues = []
            for ch in st.session_state.get("stream_chunks", []):
                chunk_index = ch.get("index")
                for d in (ch.get("dialogues") or []):
                    if str(d.get("character", "")).lower() == "ambiguous":
                        amb_id = d.get("id")
                        choice_key = f"{chunk_index}:{amb_id}"
                        chosen = st.session_state["ambiguity_choices"].get(
                            choice_key)
                        if chosen and chosen.strip():
                            nd = dict(d)
                            nd["character"] = chosen.strip()
                            updated_dialogues.append(nd)
                            continue
                    updated_dialogues.append(d)
            # Finalize consolidated output
            parser = OpenAIParser(include_narration=include_narration)
            result = parser.finalize_stream(
                updated_dialogues, include_narration=include_narration)
            st.session_state["raw_last_formatted_text"] = result.formatted_text
            st.session_state["raw_last_dialogues"] = result.dialogues
            st.session_state["raw_finalized"] = True

    # --- Final consolidated output view ---
    if st.session_state.get("raw_finalized") and st.session_state.get("raw_last_formatted_text"):
        print("[raw_parser_tab] Final consolidated output rendering")
        st.success("✅ Parsed successfully.")
        st.markdown("#### ▶ Standardized Output")
        st.code(
            st.session_state["raw_last_formatted_text"], language="markdown")
        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("→ Send to Main Generator", key="raw_send_to_main", type="primary", use_container_width=True):
                st.session_state.dialogue_text = st.session_state["raw_last_formatted_text"]
                for k in ("paste_text_analysis", "paste_formatted_dialogue", "paste_parsed_dialogues", "paste_voice_assignments"):
                    st.session_state.pop(k, None)
                st.session_state.current_tab = "main"
                st.info("Parsed output sent to Main Generator.")
        with colB:
            if st.button("🗑 Reset Parsed Output", key="raw_reset", type="secondary", use_container_width=True):
                for k in ("raw_last_formatted_text", "raw_last_dialogues", "raw_finalized"):
                    st.session_state.pop(k, None)
                st.session_state["stream_chunks"] = []
                st.success("Reset completed.")

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

from utils.s3_utils import s3_list_json, s3_read_json, s3_generate_presigned_url, s3_get_bytes
from utils.s3_utils import s3_list_objects_page, s3_list_recent_json, s3_object_exists, s3_list_projects_page
from utils.session_logger import log_to_session


def _parse_iso(ts: str) -> datetime:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.min


def _format_dt(dt: datetime, tz_label: str = "UTC") -> str:
    try:
        return dt.strftime("%b %d, %Y, %I:%M %p") + f" ({tz_label})"
    except Exception:
        return "-"


def create_history_tab():
    try:
        log_to_session("INFO", "History tab opened",
                       src="ui/history.py:create_history_tab")
    except Exception:
        pass

    tab1, tab2 = st.tabs(["📁 Generations", "📜 Logs"])

    with tab1:
        # Early loading indicator with animated spinner
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown("""
                <div style='display:flex; align-items:center; justify-content:center; gap:10px; margin:20px 0;'>
                    <div class='loader'></div>
                    <div style='font-weight:600; font-size:16px; color:#495057;'>Loading history from S3...</div>
                </div>
                <style>
                .loader {
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #3498db;
                    border-radius: 50%;
                    width: 20px;
                    height: 20px;
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                </style>
            """, unsafe_allow_html=True)

        # Styles and scroll container
        st.markdown(
            """
            <style>
            .history-scroll { max-height: 540px; overflow-y: auto; padding-right: 8px; }
            .history-card { border: 1px solid #eee; border-radius: 8px; padding: 12px; margin-bottom: 10px; background: #fff; }
            .history-title { font-weight: 600; font-size: 0.95rem; color: #111; }
            .history-meta { color: #666; font-size: 0.85rem; }
            .chip-ok { display:inline-block; padding:2px 8px; border-radius:12px; background:#e6ffed; color:#1a7f37; font-size:0.8rem; margin-left:8px; }
            .chip-bad { display:inline-block; padding:2px 8px; border-radius:12px; background:#ffecec; color:#b30000; font-size:0.8rem; margin-left:8px; }
            .btn { display:inline-block; padding:8px 14px; background:linear-gradient(45deg,#667eea,#764ba2); color:#fff; border-radius:6px; text-decoration:none; font-weight:600; }
            .btn.disabled { background:#ccc; pointer-events:none; color:#666; }
            .btn.secondary { background:#6c757d; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Initialize pagination state and cache
        if "history_page" not in st.session_state:
            st.session_state["history_page"] = 1
        if "history_page_token" not in st.session_state:
            st.session_state["history_page_token"] = None
        if "history_cache" not in st.session_state:
            st.session_state["history_cache"] = {}

        current_page = st.session_state["history_page"]

        # Check if this page is already cached
        if current_page in st.session_state["history_cache"]:
            project_keys = st.session_state["history_cache"][current_page]
            next_token = st.session_state.get(
                f"history_next_token_{current_page}", None)
            loading_placeholder.empty()
        else:
            try:
                page_token = st.session_state["history_page_token"]
                page_result = s3_list_projects_page(
                    prefix="projects/", max_keys=10, continuation_token=page_token)
                project_keys = page_result.get("keys", [])
                next_token = page_result.get("next_token")
                st.session_state["history_cache"][current_page] = project_keys
                st.session_state[f"history_next_token_{current_page}"] = next_token
                st.session_state["next_history_page_token"] = next_token
                loading_placeholder.empty()
            except Exception as e:
                loading_placeholder.empty()
                st.error(f"Cannot list S3 projects: {e}")
                return

        if not project_keys:
            loading_placeholder.empty()
            st.info("No history yet. Generate audio to see entries here.")
            return

        # Build entries list
        entries: List[Dict[str, Any]] = []
        for key in project_keys:
            data = s3_read_json(key) or {}
            project_id = data.get("project_id") or key.split(
                "/")[-1].replace(".json", "")
            status = data.get("status", "INCOMPLETE")
            last_updated = data.get("last_updated") or "1970-01-01T00:00:00Z"
            dt = _parse_iso(last_updated)
            audio_key = f"projects/{project_id}/consolidated.mp3"
            url = None
            if status == "COMPLETED":
                url = s3_generate_presigned_url(
                    audio_key, expires_seconds=3600)
            else:
                try:
                    if s3_object_exists(audio_key):
                        url = s3_generate_presigned_url(
                            audio_key, expires_seconds=3600)
                except Exception:
                    url = None
            entries.append({"project_id": project_id, "status": status,
                           "last_updated": last_updated, "dt": dt, "url": url})

        entries.sort(key=lambda e: e["dt"], reverse=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("◀ Previous Page", disabled=st.session_state["history_page"] <= 1, key="hist_prev_btn"):
                try:
                    log_to_session("UI", "History pagination: Prev",
                                   src="ui/history.py:create_history_tab")
                except Exception:
                    pass
                st.session_state["history_page"] = max(
                    1, st.session_state["history_page"] - 1)
                st.rerun()
        with col2:
            if next_token and st.button("Next ▶", key="hist_next_btn"):
                try:
                    log_to_session("UI", "History pagination: Next",
                                   src="ui/history.py:create_history_tab")
                except Exception:
                    pass
                st.session_state["history_page"] += 1
                st.session_state["history_page_token"] = next_token
                st.rerun()
            elif not next_token:
                st.caption("No more generations in S3.")

        st.markdown('<div class="history-scroll">', unsafe_allow_html=True)
        for e in entries:
            status_ok = e["status"] == "COMPLETED"
            chip = '<span class="chip-ok">✅ Success</span>' if status_ok else '<span class="chip-bad">❌ Failed</span>'
            pretty_time = _format_dt(e["dt"], "UTC")
            if e.get("url"):
                if status_ok:
                    download_html = f'<a class="btn" href="{e["url"]}" target="_blank">Download</a>'
                else:
                    download_html = f'<a class="btn secondary" href="{e["url"]}" target="_blank">Download Partial File</a>'
            else:
                download_html = '<span class="btn disabled">No file</span>'
            card_html = f"""
            <div class=\"history-card\">\
                <div class=\"history-title\">{e["project_id"]} {chip}</div>\
                <div class=\"history-meta\">Updated: {pretty_time}</div>\
                <div style=\\"margin-top:8px;\\">{download_html}</div>\
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        create_logs_sub_tab()


def create_logs_sub_tab():
    try:
        log_to_session("INFO", "Logs sub-tab opened",
                       src="ui/history.py:create_logs_sub_tab")
    except Exception:
        pass

    # Init pagination state and cache
    if "logs_page" not in st.session_state:
        st.session_state["logs_page"] = 1
    if "logs_cache" not in st.session_state:
        st.session_state["logs_cache"] = {}

    page = st.session_state["logs_page"]
    cache = st.session_state["logs_cache"]

    project_id = st.session_state.get("main_project_name") or st.session_state.get(
        "upload_project_name") or "global"
    prefix = f"projects/{project_id}/session_logs/"

    # Fetch logs (cached or new)
    if page in cache:
        logs = cache[page]
    else:
        res = s3_list_objects_page(
            prefix=prefix, max_keys=10, continuation_token=st.session_state.get(f"logs_token_{page}"))
        logs = sorted(res.get("keys", []), reverse=True)
        cache[page] = logs
        if res.get("next_token"):
            st.session_state[f"logs_token_{page+1}"] = res["next_token"]

    # Optional filter (date or simple substring)
    colf1, colf2 = st.columns([2, 3])
    with colf1:
        filter_text = st.text_input("Filter by date (YYYY-MM-DD)", "")
    with colf2:
        filter_sub = st.text_input("Search within filenames (substring)", "")
    if filter_text:
        logs = [k for k in logs if filter_text in k]
    if filter_sub:
        logs = [k for k in logs if filter_sub.lower() in k.lower()]

    if not logs:
        st.info("No session logs found yet.")
        return

    st.caption(f"Showing {len(logs)} logs (newest first)")
    for key in logs:
        filename = key.split("/")[-1]
        ts = filename.replace(".log", "")
        human_time = ts.replace("T", " ").replace("Z", " UTC")

        col1, col2, col3 = st.columns([4, 2, 3])
        with col1:
            st.markdown(f"**{filename}**")
        with col2:
            st.markdown(human_time)
        with col3:
            data = s3_get_bytes(key)
            if st.download_button("⬇ Download", data=data, file_name=filename, mime="text/plain", key=f"dl_{filename}_{page}"):
                try:
                    log_to_session(
                        "UI", f"Downloaded {filename}", src="ui/history.py:create_logs_sub_tab")
                except Exception:
                    pass
        st.divider()

    cols = st.columns([1, 1, 6])
    if cols[0].button("⬅ Prev", disabled=page == 1, key="logs_prev"):
        try:
            log_to_session(
                "UI", f"Logs page {page-1}", src="ui/history.py:create_logs_sub_tab")
        except Exception:
            pass
        st.session_state["logs_page"] -= 1
        st.rerun()
    if cols[1].button("Next ➡", disabled=f"logs_token_{page+1}" not in st.session_state, key="logs_next"):
        try:
            log_to_session(
                "UI", f"Logs page {page+1}", src="ui/history.py:create_logs_sub_tab")
        except Exception:
            pass
        st.session_state["logs_page"] += 1
        st.rerun()

    # Analytics expander
    with st.expander("📊 Analyze Logs"):
        try:
            log_to_session("UI", "Opened log analytics panel",
                           src="ui/history.py")
        except Exception:
            pass
        selected = st.selectbox("Select log to analyze",
                                logs, key="analytics_log_sel")
        if selected:
            try:
                from utils.log_analysis import load_log_summary, bundle_logs_as_zip
                summary = load_log_summary(selected)
                st.write("### Summary")
                st.json(summary.get("levels", {}))
                if summary.get("avg_mem") is not None:
                    st.metric("Avg Memory (MB)", summary["avg_mem"])
                errs = summary.get("errors") or []
                if errs:
                    st.warning(f"{len(errs)} errors found")
                    if st.checkbox("Show error lines", key="show_err_lines"):
                        st.text("\n".join(errs))
                if st.button("⬇ Download Raw Log", key="analytics_download_btn"):
                    data = s3_get_bytes(selected) or b""
                    st.download_button(
                        label="Download",
                        data=data,
                        file_name=selected.split("/")[-1],
                        mime="text/plain",
                        key="analytics_download_file",
                    )
                if st.button("📦 Download All Logs (ZIP)", key="analytics_zip_btn"):
                    data = bundle_logs_as_zip(logs)
                    st.download_button(
                        "Download ZIP", data=data, file_name="all_logs.zip", key="analytics_zip_download")
            except Exception as e:
                st.error(f"Analytics error: {e}")

    # Manual purge
    if st.button("🧹 Purge Old Logs (keep latest 25)", use_container_width=True, type="secondary"):
        try:
            from utils.session_logger import cleanup_old_logs
            log_to_session("UI", "Manual purge triggered", src="ui/history.py")
            cleanup_old_logs(project_id)
            st.success("Old logs purged.")
        except Exception as e:
            st.error(f"Purge error: {e}")

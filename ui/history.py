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
    # Minimal UI instrumentation
    try:
        log_to_session("INFO", "History tab opened",
                       src="ui/history.py:create_history_tab")
    except Exception:
        pass
    # Main header layout
    header_col1, header_col2 = st.columns([0.7, 0.3])

    with header_col1:
        st.markdown("### 🕓 Project History")
        st.caption("View past generations")

    with header_col2:
        # Logs card container
        st.markdown("""
        <div style='border: 1px solid rgba(255,255,255,0.15);
                    border-radius:8px; padding:12px 16px;'>
            <h4 style='margin:0 0 8px 0;'>📜 Logs</h4>
        </div>
        """, unsafe_allow_html=True)

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
        # Use cached results - no S3 call needed
        project_keys = st.session_state["history_cache"][current_page]
        next_token = st.session_state.get(
            f"history_next_token_{current_page}", None)
        # Skip spinner for cached pages
        loading_placeholder.empty()
    else:
        # Fetch from S3 for new pages
        try:
            page_token = st.session_state["history_page_token"]
            page_result = s3_list_projects_page(
                prefix="projects/", max_keys=10, continuation_token=page_token)
            project_keys = page_result.get("keys", [])
            next_token = page_result.get("next_token")

            # Cache results for this page
            st.session_state["history_cache"][current_page] = project_keys
            st.session_state[f"history_next_token_{current_page}"] = next_token
            st.session_state["next_history_page_token"] = next_token

            # Clear spinner after S3 load
            loading_placeholder.empty()
        except Exception as e:
            loading_placeholder.empty()
            st.error(f"Cannot list S3 projects: {e}")
            return

    if not project_keys:
        loading_placeholder.empty()
        st.info("No history yet. Generate audio to see entries here.")
        return

    # Build entries and find latest project in single pass
    entries: List[Dict[str, Any]] = []
    latest_project_id = None
    latest_dt = None

    for key in project_keys:
        data = s3_read_json(key) or {}
        project_id = data.get("project_id") or key.split(
            "/")[-1].replace(".json", "")
        status = data.get("status", "INCOMPLETE")
        last_updated = data.get("last_updated") or "1970-01-01T00:00:00Z"
        dt = _parse_iso(last_updated)

        # Track latest project
        if (latest_dt is None) or (dt > latest_dt):
            latest_dt = dt
            latest_project_id = project_id

        audio_key = f"projects/{project_id}/consolidated.mp3"
        url = None
        if status == "COMPLETED":
            url = s3_generate_presigned_url(audio_key, expires_seconds=3600)
        else:
            # Use fast HEAD check instead of downloading entire file
            try:
                if s3_object_exists(audio_key):
                    url = s3_generate_presigned_url(
                        audio_key, expires_seconds=3600)
            except Exception:
                url = None

        entries.append({
            "project_id": project_id,
            "status": status,
            "last_updated": last_updated,
            "dt": dt,
            "url": url,
        })

    # Clear loading indicator now that data is loaded
    loading_placeholder.empty()

    # Logs section UI in the top-right card
    with header_col2:
        if st.button("📄 Download Latest Logs", key="download_latest_logs", use_container_width=True):
            try:
                log_to_session("UI", "Clicked Download Latest Logs",
                               src="ui/history.py:create_history_tab")
            except Exception:
                pass
            try:
                if latest_project_id:
                    log_key = f"projects/{latest_project_id}/logs/latest.log"
                    data = s3_get_bytes(log_key)
                    if data:
                        st.download_button(
                            label="Download Logs File",
                            data=data,
                            file_name=f"{latest_project_id}_latest_logs.txt",
                            mime="text/plain",
                            key="download_logs_file"
                        )
                    else:
                        st.warning("No logs found yet.")
                else:
                    st.warning("No projects found.")
            except Exception as e:
                st.error(f"Failed to download logs: {e}")

        # Simple logs pagination (only if we have a latest project)
        if latest_project_id:
            # Maintain page state for logs
            logs_page_key = f"logs_page_{latest_project_id}"
            if logs_page_key not in st.session_state:
                st.session_state[logs_page_key] = 1

            # Fetch current page of logs
            try:
                page_num = st.session_state[logs_page_key]
                continuation_token = None
                # For simplicity, we'll fetch page 1 initially, then use continuation tokens
                if page_num > 1:
                    # In a real implementation, we'd store continuation tokens per page
                    # For now, we'll just show page 1 and disable pagination beyond that
                    st.caption(
                        "Logs pagination: Page 1 (showing latest 10 logs)")
                    page_num = 1
                    st.session_state[logs_page_key] = 1

                page = s3_list_objects_page(
                    prefix=f"projects/{latest_project_id}/logs/", max_keys=10, continuation_token=continuation_token)
                log_keys = [k for k in page.get(
                    "keys", []) if k.endswith('.log')]

                if log_keys:
                    st.caption(
                        f"**Page {page_num}** - Latest {len(log_keys)} logs:")
                    for lk in log_keys:
                        name = lk.split('/')[-1]
                        st.write(f"• {name}")
                else:
                    st.caption("No logs found.")

            except Exception as e:
                st.error(f"Failed to load logs: {e}")
        else:
            st.caption("No logs found.")

    # Sort by most recent
    entries.sort(key=lambda e: e["dt"], reverse=True)

    # Pagination controls
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
            # For previous pages, we don't need to update tokens since we use cached data
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

    # Render current page
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
        <div class="history-card">
            <div class="history-title">{e["project_id"]} {chip}</div>
            <div class="history-meta">Updated: {pretty_time}</div>
            <div style=\"margin-top:8px;\">{download_html}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

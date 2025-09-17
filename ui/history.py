import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

from utils.s3_utils import s3_list_json, s3_read_json, s3_generate_presigned_url, s3_get_bytes


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
    st.markdown("### 🕓 Project History")
    st.caption("View past generations")

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

    with st.spinner("Loading history from S3..."):
        try:
            keys = s3_list_json(prefix="projects/")
        except Exception as e:
            st.error(f"Cannot list S3 projects: {e}")
            return

        if not keys:
            st.info("No history yet. Generate audio to see entries here.")
            return

        # Build entries
        entries: List[Dict[str, Any]] = []
        for key in keys:
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
                # Try to expose partial result if present in S3
                try:
                    exists = s3_get_bytes(audio_key)
                    if exists is not None:
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

    # Sort by most recent
    entries.sort(key=lambda e: e["dt"], reverse=True)

    # --- Pagination (10 per page) ---
    per_page = 10
    total_items = len(entries)
    total_pages = max(1, (total_items + per_page - 1) // per_page)

    # Initialize current page in session state
    if "history_page" not in st.session_state:
        st.session_state["history_page"] = 1

    # Clamp page within bounds
    current_page = max(1, min(st.session_state["history_page"], total_pages))
    st.session_state["history_page"] = current_page

    # Compute slice indices
    start_idx = (current_page - 1) * per_page
    end_idx = min(start_idx + per_page, total_items)
    page_entries = entries[start_idx:end_idx]

    # Page header and controls
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_a:
        prev_disabled = current_page <= 1
        if st.button("◀ Previous", disabled=prev_disabled, use_container_width=True, key="hist_prev_btn"):
            if st.session_state["history_page"] > 1:
                st.session_state["history_page"] -= 1
                st.rerun()
    with col_b:
        st.markdown(
            f"<div style='text-align:center; font-weight:600;'>Page {current_page} of {total_pages}</div>", unsafe_allow_html=True)
    with col_c:
        next_disabled = current_page >= total_pages
        if st.button("Next ▶", disabled=next_disabled, use_container_width=True, key="hist_next_btn"):
            if st.session_state["history_page"] < total_pages:
                st.session_state["history_page"] += 1
                st.rerun()

    # Render current page
    st.markdown('<div class="history-scroll">', unsafe_allow_html=True)
    for e in page_entries:
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

import io
import threading
import time
import datetime
import traceback
from typing import Optional

import psutil

from utils.s3_utils import s3_upload_bytes


# Module-level state (independent of Streamlit context)
_log_buffer = io.StringIO()
_log_lock = threading.Lock()
_session_log_key: Optional[str] = None
_flush_thread_started = False
_mem_thread_started = False
_project_id = "global"
_logger_initialized = False


def init_session_logger(project_id: Optional[str] = None) -> None:
    global _project_id, _session_log_key, _logger_initialized
    _project_id = (project_id or "global").strip() or "global"
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    _session_log_key = f"projects/{_project_id}/session_logs/{ts}.log"
    _logger_initialized = True
    log_to_session("INFO", f"Session logger initialized for {_project_id}")


def get_session_log_key() -> Optional[str]:
    return _session_log_key


def log_to_session(level: str, msg: str, src: Optional[str] = None) -> None:
    if not _logger_initialized:
        return
    try:
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        src_tag = f"[{src}]" if src else ""
        entry = f"[{ts} UTC] [{(level or '').upper()}]{src_tag} {msg}\n"
        with _log_lock:
            _log_buffer.write(entry)
    except Exception:
        pass


def log_exception(src: str, exc: BaseException) -> None:
    try:
        log_to_session("ERROR", f"{exc.__class__.__name__}: {exc}", src)
        tb = "".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__))
        with _log_lock:
            _log_buffer.write(tb + "\n")
    except Exception:
        pass


def _flush_loop() -> None:
    global _session_log_key
    while True:
        time.sleep(120)
        try:
            with _log_lock:
                data = _log_buffer.getvalue().encode("utf-8", "ignore")
            if data and _session_log_key:
                s3_upload_bytes(_session_log_key, data,
                                content_type="text/plain")
                print(
                    f"[session_logger] Uploaded {_session_log_key} ({len(data)} bytes)")
        except Exception as e:
            print(f"[session_logger] flush error: {e}")


def _mem_loop() -> None:
    proc = psutil.Process()
    while True:
        time.sleep(10)
        try:
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            log_to_session(
                "DIAG", f"Memory usage: {rss_mb:.2f} MB", src="session_logger")
        except Exception as e:
            print(f"[session_logger] mem error: {e}")


def ensure_background_tasks() -> None:
    global _flush_thread_started, _mem_thread_started
    if not _flush_thread_started:
        threading.Thread(target=_flush_loop,
                         name="_flush_loop", daemon=True).start()
        _flush_thread_started = True
        print("[session_logger] Started flush loop (120 s)")
    if not _mem_thread_started:
        threading.Thread(target=_mem_loop, name="_mem_loop",
                         daemon=True).start()
        _mem_thread_started = True
        print("[session_logger] Started memory loop (10 s)")


def final_flush() -> None:
    try:
        with _log_lock:
            data = _log_buffer.getvalue().encode("utf-8", "ignore")
        if data and _session_log_key:
            s3_upload_bytes(_session_log_key, data, content_type="text/plain")
            print(
                f"[session_logger] Final flush complete → {_session_log_key}")
    except Exception as e:
        print(f"[session_logger] final flush error: {e}")

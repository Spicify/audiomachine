import io
import threading
import time
import datetime
import traceback
from typing import Optional, Dict, Any

import psutil

from utils.s3_utils import s3_upload_bytes, s3_list_objects_with_meta, s3_delete_object, s3_get_bytes
import gzip


# Module-level state (independent of Streamlit context)
_log_buffer = io.StringIO()
_log_lock = threading.Lock()
_session_log_key: Optional[str] = None
_flush_thread_started = False
_mem_thread_started = False
_sys_thread_started = False
_project_id = "global"
_logger_initialized = False
_flush_counter = 0
MAX_LOG_FILES = 25
COMPRESS_AFTER_DAYS = 3


def init_session_logger(project_id: Optional[str] = None) -> None:
    global _project_id, _session_log_key, _logger_initialized
    _project_id = (project_id or "global").strip() or "global"
    # Try to resume an existing recent session log (<10 minutes old)
    try:
        prefix = f"projects/{_project_id}/session_logs/"
        objs = sorted(s3_list_objects_with_meta(prefix),
                      key=lambda o: o["LastModified"], reverse=True)
        now = datetime.datetime.utcnow()
        resumed = False
        for obj in objs[:3]:  # only check a few recent
            lm = obj["LastModified"]
            try:
                age_min = (now - lm.replace(tzinfo=None)
                           ).total_seconds() / 60.0
            except Exception:
                continue
            if age_min <= 10 and not obj["Key"].endswith(".gz"):
                _session_log_key = obj["Key"]
                _logger_initialized = True
                log_to_session(
                    "INFO", "Resuming previous session log", src="session_logger")
                resumed = True
                break
        if not resumed:
            ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            _session_log_key = f"projects/{_project_id}/session_logs/{ts}.log"
            _logger_initialized = True
            log_to_session(
                "INFO", f"Session logger initialized for {_project_id}")
    except Exception:
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


def log_json(level: str, cat: str, event: str, msg: str, kv: Optional[Dict[str, Any]] = None,
             session_id: Optional[str] = None, project_id: Optional[str] = None, user_id: Optional[str] = None,
             route: Optional[str] = None, control_id: Optional[str] = None, correlation_id: Optional[str] = None,
             src: Optional[str] = None) -> None:
    """Structured JSON log line appended to the same session buffer (alongside human-readable lines).

    Fields: ts, level, cat, event, session_id, project_id, user_id, route, control_id, correlation_id, msg, kv
    """
    if not _logger_initialized:
        return
    try:
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        payload: Dict[str, Any] = {
            "ts": ts,
            "level": (level or "").upper(),
            "cat": cat,
            "event": event,
            "session_id": session_id,
            "project_id": project_id or _project_id,
            "user_id": user_id,
            "route": route,
            "control_id": control_id,
            "correlation_id": correlation_id,
            "msg": msg,
            "kv": kv or {},
        }
        line = f"[JSON] {payload}\n"
        with _log_lock:
            _log_buffer.write(line)
        if src:
            # also emit a short human-readable line for quick scanning
            log_to_session(level, f"{cat}/{event}: {msg}", src=src)
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
            _maybe_maintenance()
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


def _sys_metrics_loop() -> None:
    proc = psutil.Process()
    while True:
        time.sleep(60)
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            try:
                open_files = proc.open_files()
                open_fd = len(open_files)
            except Exception:
                open_fd = 0
            log_to_session(
                "DIAG", f"CPU={cpu_pct:.1f}%, Mem={rss_mb:.0f} MB, OpenFD={open_fd}", src="session_logger")
        except Exception as e:
            print(f"[session_logger] sys metrics error: {e}")


def ensure_background_tasks() -> None:
    global _flush_thread_started, _mem_thread_started, _sys_thread_started
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
    if not _sys_thread_started:
        threading.Thread(target=_sys_metrics_loop,
                         name="_sys_metrics_loop", daemon=True).start()
        _sys_thread_started = True
        print("[session_logger] Started system metrics loop (60 s)")


def final_flush() -> None:
    try:
        with _log_lock:
            data = _log_buffer.getvalue().encode("utf-8", "ignore")
        if data and _session_log_key:
            s3_upload_bytes(_session_log_key, data, content_type="text/plain")
            print(
                f"[session_logger] Final flush complete → {_session_log_key}")
        _maintenance(force=True)
    except Exception as e:
        print(f"[session_logger] final flush error: {e}")


def _maintenance(force: bool = False) -> None:
    """Run compression and cleanup for current project logs."""
    try:
        _compress_old_logs(_project_id)
        cleanup_old_logs(_project_id)
    except Exception as e:
        print(f"[session_logger] maintenance error: {e}")


def _maybe_maintenance() -> None:
    global _flush_counter
    _flush_counter += 1
    if _flush_counter % 5 == 0:
        _maintenance()


def _compress_old_logs(project_id: str = "global") -> None:
    prefix = f"projects/{project_id}/session_logs/"
    objs = s3_list_objects_with_meta(prefix)
    now = datetime.datetime.utcnow()
    for obj in objs:
        key = obj["Key"]
        lm = obj["LastModified"]
        try:
            age_days = (now - lm.replace(tzinfo=None)).days
        except Exception:
            continue
        if age_days >= COMPRESS_AFTER_DAYS and not key.endswith(".gz"):
            try:
                data = s3_get_bytes(key)
                if not data:
                    continue
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                    gz.write(data)
                gz_key = key + ".gz"
                s3_upload_bytes(gz_key, buf.getvalue(),
                                content_type="application/gzip")
                s3_delete_object(key)
                log_to_session(
                    "INFO", f"Compressed old log → {gz_key}", src="session_logger")
            except Exception as e:
                log_exception("session_logger.compress", e)


def cleanup_old_logs(project_id: str = "global") -> None:
    prefix = f"projects/{project_id}/session_logs/"
    objs = sorted(s3_list_objects_with_meta(prefix),
                  key=lambda o: o["LastModified"], reverse=True)
    for obj in objs[MAX_LOG_FILES:]:
        try:
            s3_delete_object(obj["Key"])
            log_to_session(
                "INFO", f"Deleted old log {obj['Key']}", src="session_logger")
        except Exception as e:
            log_exception("session_logger.cleanup", e)

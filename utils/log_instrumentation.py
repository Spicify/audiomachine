import time
from functools import wraps
from typing import Callable, TypeVar, Any

from utils.session_logger import log_to_session, log_exception


F = TypeVar("F", bound=Callable[..., Any])


def log_timed_action(label: str) -> Callable[[F], F]:
    """Decorator to measure function duration and log start/end.

    Usage:
        @log_timed_action("Generation Run")
        def run_generation(...):
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[misc]
            log_to_session("INFO", f"[START] {label}")
            t0 = time.time()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_exception(label, e)
                raise
            finally:
                dt = time.time() - t0
                log_to_session("INFO", f"[END] {label} — {dt:.2f}s elapsed")

        return wrapper  # type: ignore[return-value]

    return decorator


def log_duration(label: str, start_time: float) -> None:
    """Helper to log duration since start_time under a label."""
    try:
        dt = time.time() - start_time
        log_to_session("INFO", f"{label} — {dt:.2f}s elapsed")
    except Exception:
        pass

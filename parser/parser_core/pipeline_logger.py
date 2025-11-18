import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, Optional


class PipelineLogger:
    """
    Minimal, deterministic JSONL logger for the parser pipeline.

    This logger is *optional* and designed so that failures never break the
    main parsing pipeline. All file-system operations and writes are wrapped
    in internal try/except blocks; if anything goes wrong, the logger simply
    becomes a no-op.

    Usage
    -----
        logger = PipelineLogger("logs/output/story.jsonl")
        logger.log_story(list_of_line_dicts)
        logger.close()

    The output file path is automatically timestamped so that multiple runs do
    not overwrite each other, e.g.:
        "story.jsonl" -> "story_2025-01-01_12-30-00.jsonl"
    """

    def __init__(self, base_path: str) -> None:
        """
        Initialize the logger, resolving and opening the JSONL file.

        Args:
            base_path:
                User-requested log file path, e.g. "logs/output/story.jsonl".
                If the filename ends with ".jsonl", a timestamp of the form
                "YYYY-MM-DD_HH-MM-SS" is inserted before the extension.

        Behaviour:
            * Preserves any directory portion of `base_path`.
            * Ensures parent directories exist.
            * Opens the file in UTF-8 append mode.
            * On any failure, internal file handle is set to None and all log
              methods become no-ops.
        """
        self.path: Optional[str] = None
        self._fh: Optional[Any] = None

        try:
            dir_name, file_name = os.path.split(base_path)
            if not file_name:
                # Degenerate case: treat the directory name as the file stem.
                file_name = "pipeline.jsonl"

            name, ext = os.path.splitext(file_name)
            if ext.lower() == ".jsonl":
                ts = self._timestamp()
                file_name = f"{name}_{ts}{ext}"

            resolved_dir = dir_name or "."
            os.makedirs(resolved_dir, exist_ok=True)

            resolved_path = os.path.join(resolved_dir, file_name)
            # Store path even if open fails; useful for debugging.
            self.path = resolved_path

            self._fh = open(resolved_path, mode="a", encoding="utf-8", newline="\n")
        except Exception:
            # Swallow all exceptions; logger degrades to no-op.
            self._fh = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _timestamp(self) -> str:
        """
        Return a filesystem-safe timestamp string.

        Format: "YYYY-MM-DD_HH-MM-SS"
        """
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ------------------------------------------------------------------ #
    # Public logging API
    # ------------------------------------------------------------------ #

    def log_line(self, line_dict: Dict[str, Any]) -> None:
        """
        Write a single JSON object as one line to the log file.

        The line is encoded via json.dumps(line_dict, ensure_ascii=False),
        followed by a newline. If the logger is inactive (file handle is
        None) or a write error occurs, the method simply returns without
        raising.
        """
        if self._fh is None:
            return
        try:
            line = json.dumps(line_dict, ensure_ascii=False)
            self._fh.write(line + "\n")
            self._fh.flush()
        except Exception:
            # Do not propagate logging errors.
            return

    def log_story(self, lines: Iterable[Dict[str, Any]]) -> None:
        """
        Convenience method to write multiple line dicts as JSONL entries.

        Args:
            lines:
                Iterable of dict objects, each representing one formatted
                parser line (e.g. from ParserPipeline.parse()).
        """
        for entry in lines:
            self.log_line(entry)

    def close(self) -> None:
        """
        Close the underlying log file handle, if open.

        Safe to call multiple times; errors are swallowed to avoid affecting
        callers.
        """
        if self._fh is None:
            return
        try:
            self._fh.close()
        except Exception:
            pass
        finally:
            self._fh = None






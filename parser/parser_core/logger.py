import os
from datetime import datetime
from typing import Optional


class ParserLogger:
    """
    Lightweight logger for test outputs.
    Handles writing to main log file and detailed per-sample logs.
    """

    def __init__(self, log_file: str = "logs/splitter_test_results.txt", echo: bool = True):
        """
        Initialize logger.
        
        Args:
            log_file: Path to main log file (relative to project root)
            echo: Whether to print to console in addition to logging
        """
        self.log_file = log_file
        self.echo = echo
        self.is_first_write = True
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        """Ensure log directories exist."""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        detailed_dir = "logs/detailed"
        if not os.path.exists(detailed_dir):
            os.makedirs(detailed_dir, exist_ok=True)

    def _write(self, message: str, to_file: bool = True):
        """Internal write method."""
        if self.echo:
            print(message, end="")
        
        if to_file:
            mode = "w" if self.is_first_write else "a"
            with open(self.log_file, mode, encoding="utf-8") as f:
                f.write(message)
            self.is_first_write = False

    def log_sample_start(self, sample_name: str):
        """Log the start of processing a sample file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"\n[{timestamp}] {sample_name}\n"
        message += "-" * 40 + "\n"
        self._write(message)

    def log_line(self, line_number: int, line_type: str, text: str):
        """
        Log a single parsed line.
        
        Args:
            line_number: 1-based line number
            line_type: "dialogue", "narration", or "thought"
            text: The line text
        """
        type_upper = line_type.upper()
        message = f"{line_number:02d}. [{type_upper}] {text}\n"
        self._write(message)

    def log_summary(self, dialogue: int, narration: int, thought: int):
        """
        Log summary counts for a sample.
        
        Args:
            dialogue: Number of dialogue lines
            narration: Number of narration lines
            thought: Number of thought lines
        """
        total = dialogue + narration + thought
        message = f"Total lines: {total} (DIALOGUE: {dialogue} | NARRATION: {narration} | THOUGHT: {thought})\n"
        self._write(message)

    def write_to_detailed_log(self, sample_name: str, content: str):
        """
        Write detailed log content to a per-sample file.
        
        Args:
            sample_name: Name of the sample file (e.g., "sample_basic.txt")
            content: Full content to write
        """
        # Remove .txt extension and add .log
        base_name = os.path.splitext(sample_name)[0]
        detailed_path = f"logs/detailed/{base_name}.log"
        
        with open(detailed_path, "w", encoding="utf-8") as f:
            f.write(content)

    def log_raw(self, message: str, to_file: bool = True):
        """
        Log raw message without any formatting.
        
        Args:
            message: Message to log
            to_file: Whether to write to file
        """
        self._write(message, to_file=to_file)






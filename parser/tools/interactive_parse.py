import json
import os
import sys

# Ensure project root is on the import path so `parser_core` can be imported
# when this script is run directly (e.g., `python tools/interactive_parse.py`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser_core.pipeline import ParserPipeline
from parser_core.pipeline_logger import PipelineLogger


def prompt_story_path(max_attempts: int = 3) -> str:
    """
    Prompt the user for a story text file path, retrying up to `max_attempts`.

    Returns:
        The validated file path.

    Raises:
        SystemExit with code 1 if the user fails to provide an existing file
        within the allowed number of attempts.
    """
    attempts = 0
    while attempts < max_attempts:
        path = input("Enter path to story text file: ").strip()
        if path and os.path.isfile(path):
            return path
        print("File not found. Please try again.")
        attempts += 1

    print("Fatal error: file not found after 3 attempts.")
    sys.exit(1)


def read_text_file(path: str) -> str:
    """Read a UTF-8 text file and return its contents."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        print(f"Fatal error while reading file: {exc}")
        sys.exit(1)


def load_existing_new_characters(path: str) -> set[str]:
    """
    Load existing entries from data/new_characters.txt (if present).

    Comparison is case-insensitive. Returns a set of lowercased names.
    """
    existing: set[str] = set()
    if not os.path.exists(path):
        return existing
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    existing.add(name.lower())
    except Exception:
        # Failure to read this optional file should not be fatal.
        return existing
    return existing


def append_new_character(path: str, name: str, seen: set[str]) -> None:
    """
    Append a new character name to data/new_characters.txt if not already present.

    The check is case-insensitive and idempotent across runs.
    """
    key = name.lower()
    if key in seen:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(name + "\n")
        seen.add(key)
    except Exception as exc:
        # Do not crash; just report the issue.
        print(f"Warning: could not append '{name}' to {path}: {exc}")


def summarize_and_print(result: list[dict]) -> None:
    """
    Print a compact summary of parsed lines and then each formatted line.
    """
    total = len(result)
    counts = {"dialogue": 0, "narration": 0, "thought": 0}
    for line in result:
        t = line.get("type", "")
        if t in counts:
            counts[t] += 1

    print("\n=== Parse Summary ===")
    print(f"Total lines: {total}")
    for t in ("dialogue", "narration", "thought"):
        print(f"  {t.capitalize()}: {counts.get(t, 0)}")

    print("\n=== Parsed Lines ===")
    for idx, line in enumerate(result, start=1):
        character = line.get("character", "Unknown")
        ltype = line.get("type", "dialogue")
        emotions = line.get("emotions") or []
        text = line.get("text", "")
        emo_str = ", ".join(emotions)
        print(f"{idx:02d}. [{ltype.upper()}] ({character}) [{emo_str}] {text}")


def main() -> None:
    """
    Interactive CLI for the local parser pipeline.

    The user is prompted for:
      * Story text file path.
      * Character names present in the story.
      * Whether each name is new or already present in configs.

    New characters are appended to data/new_characters.txt (idempotent),
    preferred characters are injected into CharacterDetector for this run,
    and the full ParserPipeline is executed. The user may also choose to
    save the JSONL output via PipelineLogger.
    """
    if len(sys.argv) == 1:
        print(
            "Interactive parser: you'll be prompted for the file and characters. "
            "New characters will be added to data/new_characters.txt if you mark them as new."
        )

    story_path = prompt_story_path()
    text = read_text_file(story_path)

    # Prompt for character names present in this story.
    raw_names = input(
        "Enter character names present in this story (comma-separated): "
    ).strip()
    input_names = [n.strip() for n in raw_names.split(",") if n.strip()]

    existing_names: list[str] = []
    new_names: list[str] = []
    # Each entry: (name, exists_in_config, gender_hint)
    user_entries: list[tuple[str, bool, str | None]] = []

    new_chars_path = os.path.join("data", "new_characters.txt")
    existing_file_names = load_existing_new_characters(new_chars_path)

    for name in input_names:
        prompt = (
            f"Is '{name}' already present in configs/character_voices.json? "
            "(y/n) [default: y] "
        )
        answer = input(prompt).strip().lower()
        if answer in ("", "y", "yes"):
            existing_names.append(name)
            exists_flag = True
        else:
            # Treat anything else as "new" for simplicity.
            new_names.append(name)
            append_new_character(new_chars_path, name, existing_file_names)
            exists_flag = False

        gender_raw = input(
            f"Enter gender for '{name}' (m/f/other, optional): "
        ).strip().lower()
        gender_hint: str | None
        if gender_raw.startswith("m"):
            gender_hint = "M"
        elif gender_raw.startswith("f"):
            gender_hint = "F"
        elif gender_raw:
            gender_hint = "U"
        else:
            gender_hint = None

        user_entries.append((name, exists_flag, gender_hint))

    # Instantiate the pipeline after collecting user preferences.
    pipeline = ParserPipeline()

    # Register story-specific user character mapping and enable strict mode.
    if user_entries:
        try:
            pipeline.detector.set_user_characters(user_entries)
            pipeline.detector.enable_strict_user_mode(True)
            print(
                "Strict user-driven detection enabled: only your supplied "
                "characters will be used for speaker attribution."
            )
            # Prompt for aggressive matching behaviour
            aggressive_ans = input(
                "Enable aggressive user matching? (y/n) [default: y] "
            ).strip().lower()
            if aggressive_ans in ("", "y", "yes"):
                pipeline.detector.user_aggressive_mode = True
                print("Aggressive user matching: ON")
            else:
                pipeline.detector.user_aggressive_mode = False
                print("Aggressive user matching: OFF")

            # Prompt for dialogue auto-assignment behaviour
            auto_ans = input(
                "Enable auto-assignment of speakers for untagged dialogues? "
                "(y/n) [default: y] "
            ).strip().lower()
            if auto_ans in ("", "y", "yes"):
                pipeline.detector.user_auto_assign_dialogue = True
                print("Dialogue auto-assignment: ON")
            else:
                pipeline.detector.user_auto_assign_dialogue = False
                print("Dialogue auto-assignment: OFF")
        except Exception as exc:
            print(f"Warning: failed to configure strict user-driven detection: {exc}")

    # Inject any truly new user characters for this run.
    if new_names:
        try:
            pipeline.detector.inject_user_characters(new_names)
        except Exception as exc:
            print(f"Warning: failed to inject new characters: {exc}")

    # Execute the pipeline.
    try:
        result = pipeline.parse(text)
    except Exception as exc:
        print(f"Fatal error while parsing story: {exc}")
        sys.exit(1)

    summarize_and_print(result)

    # Optional JSONL save via PipelineLogger.
    save_answer = input(
        "Save parsed output to logs/manual_from_cli/story.jsonl? (y/n) [default: y] "
    ).strip().lower()

    if save_answer in ("", "y", "yes"):
        try:
            os.makedirs("logs/manual_from_cli", exist_ok=True)
            logger = PipelineLogger("logs/manual_from_cli/story.jsonl")
            logger.log_story(result)
            logger.close()
            print(f"Saved JSONL to: {logger.path}")
        except Exception as exc:
            print(f"Warning: failed to save JSONL output: {exc}")
    else:
        print("Skipping JSONL save.")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except SystemExit:
        # Respect explicit exit codes from main helpers.
        raise
    except Exception as e:
        # Safe failure: print error and exit non-zero.
        print(f"Unexpected fatal error: {e}")
        sys.exit(1)


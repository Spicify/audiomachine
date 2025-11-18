from typing import Any, Dict, List

from .sentence_splitter import SentenceSplitter
from .character_detector import CharacterDetector
from .emotion_tagger import EmotionTagger
from .context_state import ParserState
from .line_formatter import LineFormatter


class ParserPipeline:
    """
    Deterministic offline parsing pipeline for local, non-LLM processing.

    Components:
      - SentenceSplitter   → splits raw text into typed line dicts
      - CharacterDetector  → assigns a speaking character per line
      - EmotionTagger      → assigns top-2 emotion labels per line
      - ParserState        → maintains rolling context (recent lines/speakers)
      - LineFormatter      → produces canonical TTS-ready line dictionaries

    This pipeline is intentionally side-effect free except for its internal
    ParserState mutations. It uses only Python's standard library and does
    not perform any logging or external API calls.
    """

    def __init__(
        self,
        *,
        character_config: str = "parser/configs/character_voices.json",
        emotion_config: str = "parser/configs/emotions.json",
    ) -> None:
        """
        Initialize a new ParserPipeline with configurable character/emotion paths.

        Args:
            character_config:
                Path to the character voices configuration JSON used by
                CharacterDetector.
            emotion_config:
                Path to the emotions configuration JSON used by EmotionTagger.
        """
        self.splitter = SentenceSplitter()
        self.detector = CharacterDetector(config_path=character_config)
        self.emotion_tagger = EmotionTagger(config_path=emotion_config)
        self.state = ParserState()
        self.formatter = LineFormatter()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def parse(self, text: Any) -> List[Dict[str, Any]]:
        """
        Run the full deterministic pipeline on the given text.

        Steps:
          1. Normalise input `text` into a string.
          2. Use SentenceSplitter to build a list of `line_dict` entries.
          3. Pass those through LineFormatter.format_all(...) using the
             CharacterDetector, EmotionTagger, and ParserState.

        Behaviour:
          * If the normalised text is empty or whitespace-only, returns [].
          * Never raises due to malformed input; if an internal error occurs,
            returns a single best-effort line with:
                character = "Narrator"
                type      = "narration"
                emotions  = ["neutral", "neutral"]
                text      = stripped input text
        """
        # 1. Normalise incoming text to string
        text_str = "" if text is None else str(text)
        if not text_str or not text_str.strip():
            return []

        text_str = text_str.strip()

        # 2. Sentence splitting
        split_lines = self.splitter.split(text_str)

        # 3. Formatting via LineFormatter (with fail-safe fallback)
        try:
            # format_all does not mutate the input list, it only reads from it
            formatted = self.formatter.format_all(
                split_lines, self.detector, self.emotion_tagger, self.state
            )
            return formatted
        except Exception:
            # Fail-safe default: a single narrator line with neutral emotions
            return [
                {
                    "character": "Narrator",
                    "type": "narration",
                    "emotions": ["neutral", "neutral"],
                    "text": text_str,
                }
            ]

    def reset_state(self) -> None:
        """
        Reset the rolling ParserState to a fresh instance.

        This should be called between chapters or story segments when running
        multi-chapter stories, so that state such as recent speakers and
        recent lines does not leak across segments.
        """
        self.state = ParserState()






import os
from typing import Any, Dict, Iterable, List, Optional


class LineFormatter:
    """
    Final line formatter for the deterministic offline parser pipeline.

    This class converts the intermediate outputs from:
      * SentenceSplitter  (line_dict: {"text", "type"})
      * CharacterDetector (speaker string or dict with "character")
      * EmotionTagger     (list of two emotion labels)
      * ParserState       (rolling context, updated but not mutated here)

    into a canonical structure suitable for the TTS pipeline.

    The formatter is intentionally simple, fully deterministic, and uses only
    Python's standard library.
    """

    # ------------------------------------------------------------------ #
    # Public single-line formatter
    # ------------------------------------------------------------------ #

    def format_line(
        self, line_dict: Dict[str, Any], speaker: Any, emotions: Iterable[str]
    ) -> Dict[str, Any]:
        """
        Format a single parsed line into the canonical dict structure.

        Args:
            line_dict:
                A dictionary from SentenceSplitter with at least:
                    {"text": str, "type": "dialogue"|"narration"|"thought"}
            speaker:
                Speaker information from CharacterDetector. May be:
                  * a string, or
                  * a dict containing at least the key "character".
            emotions:
                Iterable of emotion labels (typically a list of two strings)
                produced by EmotionTagger.tag().

        Behaviour:
            * Validates presence of required keys in line_dict.
            * Normalises `speaker` to a non-empty string:
                  - If speaker is falsy or None → "Narrator".
            * Ensures `emotions` is converted to a list of exactly two strings;
              if this is not possible a ValueError is raised.
            * Preserves `line_dict["type"]` exactly.
            * Preserves `text` value from the splitter except for .strip().

        Returns:
            dict with the canonical structure:

                {
                  "character": <speaker>,
                  "type": <line type>,
                  "emotions": <list of two strings>,
                  "text": <raw text>,
                }
        """
        if not isinstance(line_dict, dict):
            raise ValueError("line_dict must be a dict from SentenceSplitter")
        if "text" not in line_dict or "type" not in line_dict:
            raise ValueError("line_dict must contain 'text' and 'type' keys")

        raw_text = line_dict.get("text", "")
        line_type = line_dict["type"]

        # Normalise speaker to a string
        if isinstance(speaker, dict):
            speaker_value = speaker.get("character")
        else:
            speaker_value = speaker
        if not speaker_value:
            speaker_str = "Narrator"
        else:
            speaker_str = str(speaker_value)

        # Normalise emotions into a list of exactly two strings
        emo_list = list(emotions) if emotions is not None else []
        if len(emo_list) != 2:
            raise ValueError(
                f"emotions must be an iterable of exactly two strings, got {emo_list!r}"
            )
        emo_list = [str(e) for e in emo_list]

        return {
            "character": speaker_str,
            "type": line_type,
            "emotions": emo_list,
            "text": str(raw_text).strip(),
        }

    # ------------------------------------------------------------------ #
    # Public batch formatter
    # ------------------------------------------------------------------ #

    def format_all(
        self,
        split_lines: List[Dict[str, Any]],
        detector: Any,
        emotion_tagger: Any,
        state: Any,
    ) -> List[Dict[str, Any]]:
        """
        Run the full formatting pipeline over a list of splitter results.

        Args:
            split_lines:
                List of line_dicts from SentenceSplitter.
            detector:
                CharacterDetector instance providing `detect(line_dict, state)`.
                May return either:
                    - a string speaker name, or
                    - a dict with key "character".
            emotion_tagger:
                EmotionTagger instance providing `tag(text, top_n=2)`.
            state:
                ParserState instance providing `update(line_dict, speaker, emotions)`.

        Behaviour:
            For each line_dict:
              * Compute speaker via detector.detect(line_dict, state).
              * Compute emotions via emotion_tagger.tag(line_dict["text"]).
              * Build canonical dict via format_line(...).
              * Update the rolling ParserState with state.update(...).
              * Append formatted dict to the output list.

        Returns:
            List of canonical formatted line dicts, one per input line.
        """
        formatted: List[Dict[str, Any]] = []

        prev_raw_line: Optional[Dict[str, Any]] = None
        debug_emotions = os.getenv("DEBUG_EMOTIONS") == "1"

        for idx, line in enumerate(split_lines):
            text = line.get("text", "")

            # Update state with the previous raw line before detecting the
            # current line's speaker. This allows the detector to consult
            # immediately preceding narration/attribution when resolving
            # dialogue subjects.
            if hasattr(state, "last_line"):
                setattr(state, "last_line", prev_raw_line)

            # Speaker detection (pass next_line for attribution-aware rules)
            next_line = split_lines[idx + 1] if idx + 1 < len(split_lines) else None
            detection_result = detector.detect(line, state, next_line=next_line)
            if isinstance(detection_result, dict):
                speaker = detection_result.get("character")
            else:
                speaker = detection_result

            # Emotion tagging (top_n defaults to 2 in EmotionTagger).
            # For richer cues, include any attached attribution text when
            # present so that speech verbs/adverbs influence the result.
            emotion_source = text
            attr = None
            if isinstance(line, dict):
                attr = line.get("attribution")
                if attr:
                    emotion_source = f"{text} {attr}"

            if debug_emotions:
                print(
                    "[EMO DEBUG] source=",
                    repr(emotion_source),
                    "text=",
                    repr(text),
                    "attr=",
                    repr(attr),
                )

            prev_text = (
                prev_raw_line.get("text") if isinstance(prev_raw_line, dict) else None
            )
            next_text = next_line.get("text") if isinstance(next_line, dict) else None

            emotions = emotion_tagger.tag(
                emotion_source,
                context_before=prev_text,
                context_after=next_text,
            )

            # Canonical formatting
            formatted_line = self.format_line(line, speaker, emotions)

            # State update
            if hasattr(state, "update"):
                state.update(line, speaker, emotions)

            formatted.append(formatted_line)
            prev_raw_line = line

        return formatted

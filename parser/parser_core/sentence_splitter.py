import re
from typing import List, Dict, Any


# =============================
# SAFE DIALOGUE DELIMITERS
# =============================
# Double quotes ONLY.
# Single quotes are EXCLUDED because they are apostrophes 99% of the time.
DQUOTES = "\u201C\u201D\""      # “ ” "
SQUOTES = ""                   # Disable single-quote toggles completely
QUOTE_CHARS = f"{DQUOTES}{SQUOTES}"
QUOTE_CLASS = "[" + re.escape(QUOTE_CHARS) + "]"

# Debug output flag
DEBUG_SPLIT = False


class SentenceSplitter:
    """
    New stable splitter with:
    - Soft linebreak merging (paragraph reconstruction)
    - Unicode double-quote dialogue detection
    - No single-quote dialogue toggling
    - Proper sentence boundary inference
    """

    SENTENCE_RE = re.compile(r"([^.!?]*[.!?])", re.MULTILINE)

    SPEECH_VERBS = re.compile(
        r"\b(said|asked|whispered|murmured|shouted|replied|called|answered|cried|laughed|sobbed|muttered)\b",
        re.IGNORECASE,
    )

    def __init__(self):
        pass

    # ------------------------------------------------------------
    # 1. Soft linebreak normalization
    # ------------------------------------------------------------
    def merge_soft_linebreaks(self, text: str) -> List[str]:
        """
        Turns soft-wrapped lines into real paragraphs.

        RULE:
        - If a line does NOT end in punctuation and the next line is not blank,
          merge with a space.
        """
        raw_lines = text.splitlines()
        paragraphs = []
        buffer = ""

        def flush():
            nonlocal buffer, paragraphs
            if buffer.strip():
                paragraphs.append(buffer.strip())
            buffer = ""

        for line in raw_lines:
            stripped = line.rstrip()

            if not stripped:  # blank line → paragraph break
                flush()
                continue

            if not buffer:  # start new buffer
                buffer = stripped
                continue

            # Should we merge soft wrap?
            if (
                # previous line lacks ending punctuation
                not re.search(r'[.!?"”]$', buffer)
                and stripped[0].islower() or not stripped[0].isalpha()
            ):
                # definitely continue the sentence
                buffer += " " + stripped
            else:
                # otherwise it's a new paragraph/sentence chunk
                flush()
                buffer = stripped

        flush()
        return paragraphs

    # ------------------------------------------------------------
    # 2. Dialogue extraction helpers
    # ------------------------------------------------------------
    def normalize_text(self, text: str) -> str:
        # Normalize curly quotes to straight ASCII for simpler processing
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("‘", "'").replace("’", "'")
        # Remove double spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _looks_like_attribution_fragment(self, text: str) -> bool:
        if self.SPEECH_VERBS.search(text):
            return True
        if re.search(r"\b(he|she)\b\s+\w+", text, re.IGNORECASE):
            return True
        return False

    # ------------------------------------------------------------
    # 3. Core line splitter
    # ------------------------------------------------------------
    def _split_line_into_results(
        self, line: str, results: List[Dict[str, Any]], raw_line: str
    ):
        segments = re.split(f"({QUOTE_CLASS})", line)
        inside_quote = False
        buffer = ""

        for seg in segments:
            if re.fullmatch(QUOTE_CLASS, seg):
                # We are hitting a quote boundary: flush any accumulated
                # text from the *previous* region before toggling.
                if buffer.strip():
                    if inside_quote:
                        # Closing a dialogue span
                        results.append(
                            {
                                "text": buffer.strip(),
                                "type": "dialogue",
                                "raw": raw_line,
                            }
                        )
                    else:
                        # Closing a narration span before entering dialogue
                        narr = buffer.strip()
                        for m in self.SENTENCE_RE.finditer(narr):
                            sent = m.group(1).strip()
                            if not sent:
                                continue
                            # Attach short attribution-like fragments to the
                            # immediately preceding dialogue line when present.
                            if (
                                self._looks_like_attribution_fragment(sent)
                                and results
                                and results[-1].get("type") == "dialogue"
                                and "attribution" not in results[-1]
                            ):
                                results[-1]["attribution"] = sent
                            results.append(
                                {
                                    "text": sent,
                                    "type": "narration",
                                    "raw": raw_line,
                                }
                            )
                    buffer = ""
                # Toggle dialogue/narration region
                inside_quote = not inside_quote
                continue

            # Accumulate plain text into the current region buffer
            buffer += seg

        # Flush any trailing narration outside quotes
        if buffer.strip():
            if inside_quote:
                # Unclosed quote; treat as dialogue to avoid losing text
                results.append(
                    {"text": buffer.strip(), "type": "dialogue", "raw": raw_line}
                )
            else:
                narr = buffer.strip()
                for m in self.SENTENCE_RE.finditer(narr):
                    sent = m.group(1).strip()
                    if not sent:
                        continue
                    if (
                        self._looks_like_attribution_fragment(sent)
                        and results
                        and results[-1].get("type") == "dialogue"
                        and "attribution" not in results[-1]
                    ):
                        results[-1]["attribution"] = sent
                    results.append(
                        {"text": sent, "type": "narration", "raw": raw_line}
                    )

    # ------------------------------------------------------------
    # 4. Public split() API
    # ------------------------------------------------------------
    def split(self, text: str, debug: bool = False) -> List[Dict[str, Any]]:
        paragraphs = self.merge_soft_linebreaks(text)
        results: List[Dict[str, Any]] = []

        for para in paragraphs:
            raw_line = para  # preserve original merged paragraph text
            line = self.normalize_text(para)
            if not line:
                continue
            self._split_line_into_results(line, results, raw_line)

        # Debug print
        if debug or DEBUG_SPLIT:
            for i, r in enumerate(results, 1):
                print(f"{i:02d}. [{r['type'].upper()}] {r['text']}")

        return results

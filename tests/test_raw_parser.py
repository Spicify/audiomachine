import unittest

from parser.local_adapter import parse_raw_prose_to_dialogue_format


def _parse(text: str, include_narration: bool = True):
    result = parse_raw_prose_to_dialogue_format(
        text,
        include_narration=include_narration,
        user_cast=[
            {"name": "Brad", "gender": "M", "enabled": True},
            {"name": "Zara", "gender": "F", "enabled": True},
            {"name": "Nathaniel", "gender": "M", "enabled": True},
        ],
        strict_mode=True,
    )
    return result.formatted_text.splitlines()


class TestLocalParser(unittest.TestCase):
    def test_post_quote_attribution(self):
        lines = _parse('"Hi," said Brad. "I\'m home," he said.')
        self.assertIn("brad", lines[0].lower())
        self.assertIn("brad", lines[1].lower())

    def test_narration_based_resolution(self):
        lines = _parse('Brad entered. "I\'m home," he said.')
        self.assertTrue(any(line.lower().startswith("[brad]") for line in lines))

    def test_fallback_to_narrator_when_unknown(self):
        lines = _parse('"Hello there."')
        line = lines[0].lower()
        self.assertTrue(line.startswith("[narrator]") or line.startswith("[ambiguous]"))

    def test_question_curiosity_emotion(self):
        lines = _parse('"How are you?" he asked.')
        combined = " ".join(lines).lower()
        self.assertTrue("(curious)" in combined or "(soft)" in combined)

    def test_exclude_narration_when_requested(self):
        lines = _parse('"Hello," said Brad. Nathaniel waved. "Hi," he said.', include_narration=False)
        self.assertTrue(all("narrator" not in line.lower() for line in lines))


if __name__ == "__main__":
    unittest.main()

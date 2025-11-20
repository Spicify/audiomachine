import unittest

from parser.parser_core.character_detector import CharacterDetector
from parser.parser_core.context_state import ParserState


class TestCharacterDetectorMapping(unittest.TestCase):
    def setUp(self):
        self.detector = CharacterDetector()
        self.detector.set_user_characters(
            [
                ("Mikhail", False, "M"),
                ("isabelle", False, "F"),
                ("Aleksandr", False, "M"),
            ]
        )

    def test_user_name_with_suffix_maps_to_canonical(self):
        result = self.detector._map_user_name_to_canonical("Mikhail - Male")
        self.assertIsNotNone(result)
        self.assertEqual(self.detector._normalize_name(result), "mikhail")

    def test_narration_prefers_unique_female_subject(self):
        state = ParserState()
        line = {"text": "She wore a sleek black dress.", "type": "narration"}
        result = self.detector.detect(line, state)
        self.assertEqual(result["character"], "Isabelle")

    def test_narration_carries_subject_without_cues(self):
        state = ParserState()
        first = {"text": "She felt the weight of their protectiveness.", "type": "narration"}
        self.detector.detect(first, state)
        follow = {"text": "Dinner was luxurious — fine wine, soft music.", "type": "narration"}
        result = self.detector.detect(follow, state)
        self.assertEqual(result["character"], "Isabelle")

    def test_direct_name_in_narration_assigned(self):
        state = ParserState()
        line = {"text": "mikhail smirked and glanced over.", "type": "narration"}
        result = self.detector.detect(line, state)
        self.assertEqual(result["character"], "Mikhail")

    def test_no_partial_name_match_from_common_words(self):
        text = "The moment the doors shut, tension snapped."
        self.assertIsNone(self.detector._find_direct_name(text))

    def test_named_attribution_overrides_subject_bias(self):
        state = ParserState()
        self.detector.detect({"text": "She leaned in.", "type": "narration"}, state)
        dialogue = {
            "text": "You want to misbehave in public now?",
            "type": "dialogue",
            "raw": "\"You want to misbehave in public now?\" Aleksandr murmured at her ear.",
            "attribution": "Aleksandr murmured at her ear.",
        }
        next_line = {"text": "Aleksandr murmured at her ear.", "type": "narration"}
        result = self.detector.detect(dialogue, state, next_line=next_line)
        self.assertEqual(result["character"], "Aleksandr")

    def test_following_named_narration_assigns_dialogue(self):
        state = ParserState()
        dialogue = {
            "text": "Oh, is she now?",
            "type": "dialogue",
            "raw": "\"Oh, is she now?\" Mikhail adds.",
        }
        next_line = {"text": "Mikhail adds.", "type": "narration"}
        result = self.detector.detect(dialogue, state, next_line=next_line)
        self.assertEqual(result["character"], "Mikhail")

    def test_following_pronoun_attribution_assigns_dialogue(self):
        state = ParserState()
        state.recent_mentions = ["Mikhail", "Isabelle"]
        dialogue = {
            "text": "I wanted you,",
            "type": "dialogue",
            "raw": "\"I wanted you,\" she breathed.",
            "attribution": "she breathed.",
        }
        next_line = {"text": "she breathed.", "type": "narration"}
        result = self.detector.detect(dialogue, state, next_line=next_line)
        self.assertEqual(result["character"], "Isabelle")

    def test_pronoun_resolution_handles_his(self):
        state = ParserState()
        state.recent_mentions = ["Isabelle", "Mikhail"]
        line = {"text": "His jaw tightened as her hand moved.", "type": "narration"}
        result = self.detector.detect(line, state)
        self.assertEqual(result["character"], "Mikhail")


if __name__ == "__main__":
    unittest.main()

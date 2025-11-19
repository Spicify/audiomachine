import unittest

from parser.parser_core.character_detector import CharacterDetector


class TestCharacterDetectorMapping(unittest.TestCase):
    def setUp(self):
        self.detector = CharacterDetector()

    def test_user_name_with_suffix_maps_to_canonical(self):
        result = self.detector._map_user_name_to_canonical("Mikhail - Male")
        self.assertIsNotNone(result)
        self.assertEqual(self.detector._normalize_name(result), "mikhail")


if __name__ == "__main__":
    unittest.main()

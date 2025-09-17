import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "configs"

with open(CONFIG_DIR / "character_voices.json", "r", encoding="utf-8") as f:
    CHARACTER_VOICES = json.load(f)

with open(CONFIG_DIR / "emotion_tags.json", "r", encoding="utf-8") as f:
    EMOTION_TAGS = json.load(f)

with open(CONFIG_DIR / "speech_verbs.json", "r", encoding="utf-8") as f:
    SPEECH_VERBS = json.load(f)

with open(CONFIG_DIR / "adverb_to_emotion.json", "r", encoding="utf-8") as f:
    ADVERB_TO_EMOTION = json.load(f)

with open(CONFIG_DIR / "verb_to_emotion.json", "r", encoding="utf-8") as f:
    VERB_TO_EMOTION = json.load(f)

# FX effects removed from the application

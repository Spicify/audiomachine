from config_loader import CHARACTER_VOICES, EMOTION_TAGS


def get_flat_character_voices():
    flat = {}
    for category, chars in CHARACTER_VOICES.items():
        for name, data in chars.items():
            if isinstance(data, dict):  # new format
                flat[name] = {
                    "voice_id": data.get("voice_id"),
                    "gender": data.get("gender", "U")  # U = Unknown
                }
            else:  # fallback for legacy format
                flat[name] = {
                    "voice_id": data,
                    "gender": "U"
                }
    return flat


def get_flat_emotion_tags():
    flat_emotions = {}
    for category in EMOTION_TAGS.values():
        flat_emotions.update(category)
    return flat_emotions

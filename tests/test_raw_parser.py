from parsers.openai_parser.openai_parser import OpenAIParser


def run_tests():
    tests = [
        # --- Basic Attribution ---
        {
            "desc": "Post-quote attribution",
            "input": '"Hi," said Brad. "I’m home," he said.',
            "expected": ["[Brad]: Hi,", "[Brad]: I’m home,"]
        },
        {
            "desc": "Narration-based last seen",
            "input": "Brad entered. \"I’m home,\" he said.",
            "expected": ["[Brad]: I’m home,"]
        },
        {
            "desc": "Simple pronoun resolution",
            "input": '"Hi," said Zara. Nathaniel looked at him and said "Hey". "What\'s up?" she replied.',
            "expected": ["[Zara]: Hi,", "[Nathaniel]: Hey", "[Zara]: What's up?"]
        },
        {
            "desc": "Fallback to narrator",
            "input": "\"Hello there.\"",
            "expected": ["[Narrator]: Hello there."]
        },

        # --- Multiple Characters ---
        {
            "desc": "Conversation back and forth",
            "input": '"Hello," said Brad. "Hi," said Zara. "How are you?" he asked.',
            "expected": ["[Brad]: Hello,", "[Zara]: Hi,", "[Brad]: How are you?"]
        },
        {
            "desc": "Ambiguous multiple males → fallback",
            "input": '"Hey," said Brad. Nathaniel walked in. "Yo," he said.',
            "expected": ["[Brad]: Hey,", "[Narrator]: Yo,"]
        },

        # --- Punctuation Handling ---
        {
            "desc": "British punctuation",
            "input": '"Hello there." said Brad.',
            "expected": ["[Brad]: Hello there."]
        },
        {
            "desc": "US punctuation",
            "input": '"Hello there", said Brad.',
            # If parser strips comma → update expected
            "expected": ["[Brad]: Hello there,"]
        },

        # --- Emotion/Adverb Inference ---
        {
            "desc": "Verb to emotion mapping",
            "input": '"Go away!" shouted Brad.',
            "expected": ["[Brad](angry): Go away!"]
        },
        {
            "desc": "Adverb to emotion mapping",
            "input": '"I’m fine," Brad said softly.',
            "expected": ["[Brad](gentle): I’m fine,"]
        },

        # --- Complex Narration ---
        {
            "desc": "Inline narration updates last seen",
            "input": '"Hi," said Zara. Nathaniel smiled. "Hey there," he said.',
            "expected": ["[Zara]: Hi,", "[Nathaniel]: Hey there,"]
        },
        {
            "desc": "Pronoun chain with missing female (→ fallback)",
            "input": 'Brad walked in. "Hello," he said. "How are you?" she asked. "I’m fine," he replied.',
            "expected": ["[Brad]: Hello,", "[Narrator]: How are you?", "[Brad]: I’m fine,"]
        },
    ]

    for include_narration in (True, False):
        print("\n===============================")
        print(f"Testing with include_narration={include_narration}")
        print("===============================")

        parser = OpenAIParser(
            include_narration=include_narration)

        for t in tests:
            result = parser.convert(t["input"])
            print(f"\n--- {t['desc']} ---")
            for line in result.formatted_text.split("\n"):
                print(line)
            print("Expected:", t["expected"])
            if not include_narration:
                print("(Narration lines should be suppressed here)")


if __name__ == "__main__":
    run_tests()

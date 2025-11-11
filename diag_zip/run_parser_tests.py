from parsers.openai_parser.openai_parser import OpenAIParser


class _Resp:
    def __init__(self, text: str):
        self.output_text = text


def _set_stub_response(parser: OpenAIParser, text: str) -> None:
    class _R:
        def create(self, **kwargs):
            return _Resp(text)

    class _C:
        def __init__(self):
            self.responses = _R()

    # Replace only the responses interface to avoid network
    try:
        parser.client.responses = _R()
    except Exception:
        parser.client = _C()


cases = [
    (
        "Case 1",
        "She whispered, \"Stay with me.\" The room felt serene, then turned melancholy.",
        # JSONL: Ambiguous quote + Narrator sentence
        """
{"character":"Ambiguous","emotions":["soft","tender"],"candidates":["She"],"text":"Stay with me."}
{"character":"Narrator","emotions":["serene","sad"],"text":"The room felt serene, then turned melancholy."}
""".strip(),
    ),
    (
        "Case 2",
        "“You want to misbehave in public now?” Aleksandr murmured, lips at her ear.",
        # JSONL: Aleksandr quote + Narrator clause (no attribution tail)
        """
{"character":"Aleksandr","emotions":["warm","soft"],"text":"You want to misbehave in public now?"}
{"character":"Narrator","emotions":["tender","intimate"],"text":"His lips were at her ear."}
""".strip(),
    ),
    (
        "Case 3",
        "He paused. \"We move now.\" The hallway lights flickered, humming nervously.",
        # JSONL: Narrator pause + Ambiguous quote + Narrator hallway
        """
{"character":"Narrator","emotions":["calm","neutral"],"text":"He paused."}
{"character":"Ambiguous","emotions":["serious","tense"],"candidates":["He"],"text":"We move now."}
{"character":"Narrator","emotions":["nervous","tense"],"text":"The hallway lights flickered, humming nervously."}
""".strip(),
    ),
    (
        "Case 4",
        "Mikhail raised a slow brow. “What do you think you’re doing, kitten?”",
        # JSONL: Action as Mikhail + Mikhail quote
        """
{"character":"Mikhail","emotions":["serious","calm"],"text":"Mikhail raised a slow brow."}
{"character":"Mikhail","emotions":["stern","warm"],"text":"What do you think you're doing, kitten?"}
""".strip(),
    ),
]


def run():
    parser = OpenAIParser(debug_save=False)
    for name, raw, stub in cases:
        print(f"\n===== RUN {name} =====", flush=True)
        try:
            _set_stub_response(parser, stub)
            # Drain generator to surface all logs
            for _ in parser.convert_streaming(raw):
                pass
        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            print(f"[ERROR] {name}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    run()

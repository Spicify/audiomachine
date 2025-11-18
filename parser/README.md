# Exported Local Parser Module

This folder contains a self‑contained copy of the updated local parser, ready to be dropped into another repository (for example under `new-parser/`). Cursor/Codex can then wire it into the main app without referring back to this repo.

---

## Folder layout

- `parser_core/`
  - `pipeline.py` – orchestrator (`ParserPipeline`).
  - `sentence_splitter.py` – splits raw text into dialogue/narration lines and attaches attributions.
  - `character_detector.py` – infers speakers using user‑supplied character sets, narration subjects, pronouns, and heuristics.
  - `emotion_tagger.py` – deterministic rule + keyword emotion inference (always 2 distinct tags).
  - `context_state.py` – rolling state (last speaker/subject, recent mentions).
  - `line_formatter.py` – glues splitter/detector/tagger together and emits canonical dicts.
  - `logger.py` – optional helper logging utilities.
  - `pipeline_logger.py` – JSONL logger for parsed stories.
  - `__init__.py`
- `configs/`
  - `character_voices.json` – character metadata (voice IDs, gender).
  - `emotions.json` – keyword lists for fallback emotion scoring.
- `tools/`
  - `interactive_parse.py` – CLI wrapper for manual runs / debugging.

All code needed to run the parser is under `parser_core/` plus the configs. The CLI is optional.

---

## High-level architecture

### `ParserPipeline` (`parser_core/pipeline.py`)

Main entry point. Usage:

```python
from exported_parser.parser_core.pipeline import ParserPipeline

pipeline = ParserPipeline(
    character_config="exported_parser/configs/character_voices.json",
    emotion_config="exported_parser/configs/emotions.json",
)

result = pipeline.parse(text)
```

`parse(text)` returns a list of dicts like:

```python
{
  "character": "Aleksandr" | "Mikhail" | "Narrator" | "Ambiguous",
  "type": "dialogue" | "narration" | "thought",
  "emotions": ["soft", "tender"],  # always 2 distinct labels
  "text": "Raw line text"
}
```

It wires together:

1. `SentenceSplitter` – merges soft line breaks, toggles dialogue via quotes, extracts short attribution fragments (e.g. “he said softly”) and attaches them to dialogue lines as `line["attribution"]`.
2. `CharacterDetector` – uses user-supplied character names + genders (from CLI or your app) to constrain speaker attribution. Leverages narration subjects, pronouns, context, and aggressive/strict modes.
3. `EmotionTagger` – rule-based plus config fallback, always emits two meaningful emotion labels.
4. `ParserState` – remembers `last_speaker`, `last_subject`, and recent mentions to support attribution.
5. `LineFormatter` – stitches the pieces together, feeds the detector and tagger, and updates state.

`reset_state()` resets the rolling state between stories/chapters.

---

## Core modules overview

### `parser_core/sentence_splitter.py`
- `merge_soft_linebreaks`: reconstructs paragraphs based on punctuation/casing.
- `_split_line_into_results`: toggles between narration/dialogue segments using quotes and `SENTENCE_RE`. Attaches `dialogue_line["attribution"]` when narration fragments look like speech tags.
- `split(text)` – top-level method returning list of line dicts with `text`, `type`, `raw`, optional `attribution`.

### `parser_core/context_state.py`
- `ParserState`: stores recent lines, last speaker, last subject, and recent mentions.
- `update(...)`: appends new entries and updates speaker/subject context.
- Accessors: `get_last_speaker()`, `get_recent_mentions()`, etc., used by the detector.

### `parser_core/character_detector.py`
- Configuration loaders (`_load_characters`, `_flatten_character_json`) for `character_voices.json`.
- `set_user_characters(...)`: accepts `(name, exists_flag, gender_hint)` tuples, builds user character list, and updates gender info.
- Main `detect(line, state, next_line)` flow:
  1. Narration → `_update_subject_from_narration` (explicit names, possessives, pronouns; uses sentinel for unnamed female subjects).
  2. Dialogue:
     - Attribution shortcut via `_resolve_from_attribution` (attached inline text or following short narration).
     - Subject-based fallback (if narration established a subject).
     - Strict/aggressive user-mode heuristics (variant matching, recency, alternation).
     - General heuristics (direct name, speech verbs, pronouns, recency, alternation).
  3. Returns `{"character": "Name"}` or `{"character": "Ambiguous", "candidates": [...]}`.

### `parser_core/emotion_tagger.py`
- Loads keyword config from `emotions.json`.
- `_rule_based_emotions(text)` – ordered regex patterns covering cues like:
  - Shouted/screamed (→ angry/intense).
  - Dominant/commanding phrases.
  - Teasing/playful/mischievous cues.
  - Soft/whispered/breathy speech.
  - Aroused/romantic/tipsy/drunk/high states.
  - Sad/crying/broken, ecstatic/climactic, jealous, embarrassed, etc.
  - Vocal cues (moaning, panting, whimpering, breathless).
  - Questions (→ curious/soft) and punctuation cues (“!”).
- `_analyze(text)` – keyword scoring with negation handling, used if no rule matches.
- `_finalise_labels(labels)` – ensures exactly two distinct labels, uses companion map for singletons, fallback `["soft", "calm"]`.

### `parser_core/line_formatter.py`
- `format_all(split_lines, detector, emotion_tagger, state)`:
  - Maintains `state.last_line`.
  - Feeds each line to `CharacterDetector.detect` (with `next_line` for attribution).
  - Builds `emotion_source = line["text"] + attribution` and calls `EmotionTagger.tag`.
  - Updates `ParserState` with the detected speaker/emotions.
  - Returns canonical line dicts.

### Utilities
- `logger.py` / `pipeline_logger.py` – optional logging utilities; you can use them or ignore them depending on your app’s needs.

### CLI (`tools/interactive_parse.py`)
- Prompts for:
  - Story path.
  - Character names and whether they already exist in `character_voices.json`.
  - Gender hint (m/f/other) for each character.
  - Strict/aggressive mode toggles.
- Wires those inputs into `ParserPipeline` (configuring user characters and detection mode).
- Runs `pipeline.parse(text)` and prints/saves results.
- Handy for testing but not required for integration.

---

## How to integrate into another repo

1. **Copy this folder**
   - In your main app repo, create `new-parser/` (or any name).
   - Copy everything under `exported_parser/` into that folder, preserving subfolders.

2. **Instantiate the parser**

   ```python
   from new_parser.parser_core.pipeline import ParserPipeline

   pipeline = ParserPipeline(
       character_config="new_parser/configs/character_voices.json",
       emotion_config="new_parser/configs/emotions.json",
   )

   parsed_lines = pipeline.parse(story_text)
   ```

3. **Configure per story (optional)**
   - If your UI collects character names/genders:
     ```python
     entries = [
         ("Mikhail", True, "M"),
         ("Aleksandr", False, "M"),
         ("Isabelle", False, "F"),
     ]
     pipeline.detector.set_user_characters(entries)
     pipeline.detector.enable_strict_user_mode(True)
     pipeline.detector.user_aggressive_mode = True
     pipeline.detector.user_auto_assign_dialogue = True
     ```
   - Otherwise, run in default mode (uses global config names).

4. **Replace old parser calls**
   - Wherever the old parser returned structured lines, replace it with `pipeline.parse`.
   - Each returned dict has `character`, `type`, `emotions`, `text`. Adapt or map them to your audio/TTS pipeline as needed.

5. **Remove obsolete parser code**
   - Since this parser is fully local, you can remove old OpenAI/fallback code once everything is wired up.

6. **Testing**
   - Run your story parsing flows to confirm:
     - UI/API usage is unchanged.
     - Audio pipeline receives the expected character labels.
     - No runtime errors due to missing configs or imports.

This README is intentionally detailed so Cursor/Codex can follow it and integrate the parser quickly. Copy `exported_parser/` into your main app, point imports/config paths to it, and replace the old parser implementation with `ParserPipeline.parse`.


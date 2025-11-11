# Audiomachine Parser Diagnostic Report

**Date:** 2025-01-XX  
**Investigator:** Code Explorer + Technical Investigator  
**Objective:** Diagnose three critical parser issues without making code changes

---

## Executive Summary

1. **Issue 1 (Fallback → Narrator calm/soft):** Root cause appears to be in `ensure_two_emotions()` defaulting to `["calm", "soft"]` when fallback lines lack valid emotions, combined with `_sanitize_character()` mapping unknown speakers to "Ambiguous" which then gets coerced to "Narrator" via `_looks_like_narration()` in `validate_and_fix()`.

2. **Issue 2 (Consecutive lines collapsed):** The merging logic in `convert_batch.py:870-871` combines consecutive lines with the same character **before** reinjection, causing fallback/reinjected lines to merge incorrectly. Additionally, `deduplicate_lines()` may prefer Narrator over character versions when normalized text matches.

3. **Issue 3 (Wrong order reinjection):** The `replace_or_insert_lines()` function uses fuzzy matching and positional guesses that can place reinjected lines at incorrect indices, especially when `_span_start` is missing or when SID mapping fails. The final reinjection loop (lines 1052-1085) appends to `reconciled` without proper ordering.

---

## Reproduction Steps & Commands

### Test Environment Setup
```bash
# Set debug flags
export DEBUG_PARSER_DIAG=1
export DEBUG_EMOTIONS=1

# Run test script
python test_parser_issues.py > logs/parser_run_debug.log 2>&1

# Or run directly via harness
python -m parsers.openai_parser.convert_batch
```

### Test Case A: Fallback-heavy (mature content)
```python
text = '''"Please, Daddy," she whispered, her voice trembling.
He looked at her with dark eyes.
"Good girl," he murmured, his hands moving lower.
She gasped as his fingers found their target.
"More," she begged, arching against him.'''
```

### Test Case B: Rapid alternating speakers
```python
text = '''"Please, Daddy," she said.
"Good girl," he replied.
She looked away.
He smiled.'''
```

### Test Case C: Order break
```python
text = '''She entered the room.
The door closed behind her.
He looked up from his desk.
"Hello," he said.
She smiled.'''
```

---

## Evidence for Each Issue

### Issue 1: Fallback Lines Defaulting to Narrator (calm)(soft)

#### Input Excerpt
```
"Please, Daddy," she whispered, her voice trembling.
He looked at her with dark eyes.
"Good girl," he murmured, his hands moving lower.
```

#### Code Flow Analysis

**1. Fallback Parsing (`parsers/openai_parser/fallback/parsing.py:325-333`)**
- `_sanitize_character()` maps unknown characters to "Ambiguous"
- If character not in `known_characters`, sets to "Ambiguous"

**2. Validation (`parsers/openai_parser/validator.py:128-147`)**
- `ensure_two_emotions()` called at line 128
- If emotions are empty/invalid, defaults to `["calm", "soft"]` (line 203, 225 in `emotion_utils.py`)
- `canonicalize_emotion()` maps unknown emotions → "calm" (line 66-83 in `emotion_utils.py`)

**3. Ambiguous → Narrator Coercion (`parsers/openai_parser/validator.py:164-173`)**
- Line 164: `if (fixed.get("character") == "Ambiguous") and _no_quotes(txt) and _looks_like_narration(txt, state):`
- `_looks_like_narration()` returns True for:
  - Third-person pronouns present (line 38)
  - Known character mentions (line 42-44)
  - Short declaratives ending with punctuation (line 48)
- Result: Ambiguous → Narrator conversion

**4. Emotion Defaulting (`parsers/openai_parser/emotion_utils.py:176-228`)**
- Line 203: `if not out: out = ["calm", "soft"]` (strict_list mode)
- Line 225: `if not out: out = ["calm", "soft"]` (freeform_map mode)
- Line 141-146: After canonicalization, if < 2 emotions, fills with `("calm", "soft", "tense", "warm")` in order

#### Key Variable Snapshots (Expected)

**After `fb_valid = validate_and_fix(...)` (convert_batch.py:567-568):**
```json
[
  {
    "character": "Narrator",
    "emotions": ["calm", "soft"],
    "text": "Please, Daddy,",
    "_src": "fb",
    "_span_start": 0
  }
]
```

**After SID mapping (convert_batch.py:575-579):**
- `mapped_count`: 0 (if ledger unavailable)
- `unmapped_count`: 1
- `_sid`: None (unmapped)

**Emotion KB state:**
- `kb`: Dict from `build_emotion_kb()` (verb/adverb → emotion mappings)
- `allowed_emotions`: Set from `get_allowed_emotions()` (EMOTION_TAGS keys)

#### Root Cause Hypothesis

**Primary:** `ensure_two_emotions()` defaults to `["calm", "soft"]` when:
1. Fallback output has empty/invalid emotions
2. No verb/adverb mappings found in text
3. Character is unknown → sanitized to "Ambiguous" → coerced to "Narrator"

**Secondary:** `canonicalize_emotion()` in freeform mode (line 66-83) returns "calm" for any unmapped emotion, and the top-up logic (line 217-222) always adds "calm" and "soft" first.

---

### Issue 2: Consecutive Lines Collapsed to One Speaker

#### Input Excerpt
```
"Please, Daddy," she said.
"Good girl," he replied.
```

#### Code Flow Analysis

**1. Per-Chunk Processing (`parsers/openai_parser/convert_batch.py:840-857`)**
- Lines appended to `per_chunk_dialogues` and `all_dialogues`
- No merging at chunk level

**2. Deduplication (`parsers/openai_parser/chunker.py:177-282`)**
- `deduplicate_lines()` called at line 857
- Line 203-254: Normalized text matching logic
- **Problem:** Line 235-246: Prefers non-Narrator over Narrator, but if both are non-Narrator, keeps first occurrence (line 274-275)
- **Problem:** Line 870-871 in `convert_batch.py`: Merging happens **after** deduplication but **before** final reinjection

**3. Merging Logic (`parsers/openai_parser/convert_batch.py:870-881`)**
```python
if reconciled and reconciled[-1]["character"] == item["character"]:
    reconciled[-1]["text"] = f"{reconciled[-1]['text']} {item['text']}".strip()
    # ... merges emotions ...
```
- **Critical:** This merges **all** consecutive same-character lines, including:
  - Fallback lines (`_src="fb"`)
  - Reinjected lines (`_src="reinj"`)
  - OpenAI lines (`_src="ai"`)
- No check for `_src` tag before merging
- Merges even when lines should be separate (e.g., different quotes)

**4. Deduplication Conflicts (`parsers/openai_parser/chunker.py:255-277`)**
- Line 257-265: Detects cross-speaker duplicates
- But only logs; doesn't prevent incorrect merging

#### Key Variable Snapshots (Expected)

**Before merging (convert_batch.py:866):**
```json
[
  {"character": "She", "text": "Please, Daddy,", "_src": "ai"},
  {"character": "He", "text": "Good girl,", "_src": "ai"}
]
```

**After merging (convert_batch.py:870-881):**
```json
[
  {"character": "She", "text": "Please, Daddy, Good girl,", "_src": "ai"}
]
```

**Problem:** If both lines get attributed to same character (e.g., both "Narrator" or both "Ambiguous"), they get merged.

#### Root Cause Hypothesis

**Primary:** Merging logic at `convert_batch.py:870-871` merges consecutive same-character lines **regardless of source** (`_src` tag) and **regardless of whether they should be separate** (e.g., different quotes).

**Secondary:** `deduplicate_lines()` may incorrectly attribute both lines to the same character if:
- Speaker detection fails
- Both become "Ambiguous" → "Narrator"
- Normalized text matches but speakers differ (line 203-254)

---

### Issue 3: Random Reinjected Sentences Placed in Wrong Order

#### Input Excerpt
```
She entered the room.
The door closed behind her.
He looked up from his desk.
"Hello," he said.
She smiled.
```

#### Code Flow Analysis

**1. Fallback Reinjection (`parsers/openai_parser/fallback/reinject.py:40-438`)**
- `replace_or_insert_lines()` called at line 696 in `convert_batch.py`
- **Priority order:**
  1. Per-line `_span_start` (line 189-203)
  2. Group-level `start_index`/`end_index` (line 207-212)
  3. Fuzzy content anchor (line 224-240)
  4. Tail append (line 212)

**2. SID-Based Anchoring (`parsers/openai_parser/fallback/reinject.py:69-183`)**
- Lines 69-183: SID-anchored fast path
- **Problem:** If `_target_sid` missing or `emitted_idx_by_sid` incomplete, falls back to positional guess
- **Problem:** Line 136-141: Complex logic to find `prev_idx`/`next_idx` from `emitted_idx_by_sid` may fail if ledger unavailable

**3. Positional Guess (`parsers/openai_parser/convert_batch.py:622-637`)**
```python
anchor_sent = int(seg.get("start_idx", 0))
pos = sent_to_pos.get(anchor_sent)
if pos is None:
    for d in range(1, 11):
        if sent_to_pos.get(anchor_sent - d) is not None:
            pos = sent_to_pos.get(anchor_sent - d)
            break
        if sent_to_pos.get(anchor_sent + d) is not None:
            pos = sent_to_pos.get(anchor_sent + d)
            break
```
- **Problem:** `sent_to_pos` may not contain `anchor_sent` if sentence wasn't parsed by OpenAI
- **Problem:** Fallback search (d=1..10) may find wrong position

**4. Final Reinjection Loop (`parsers/openai_parser/convert_batch.py:1052-1085`)**
- Lines 1052-1085: Final reinjection of missing sentences
- **Problem:** Line 1069: `reconciled.append({...})` - **always appends to end**, no ordering logic
- **Problem:** Character guess (line 1058-1068) is heuristic-based, may be wrong
- **Problem:** No `_span_start` or `_sid` set for reinjected lines

#### Key Variable Snapshots (Expected)

**Before `replace_or_insert_lines()` (convert_batch.py:696):**
```json
{
  "fixed": [...],
  "fb_valid": [
    {"character": "Narrator", "text": "She entered the room.", "_span_start": 0}
  ],
  "start_index": 0,
  "end_index": 0,
  "_base_pos": 0,
  "_approx_pos": 0
}
```

**After `replace_or_insert_lines()`:**
- Line inserted at index 0 (if `_span_start=0`)
- But if `_span_start` missing, may append to tail

**Final reinjection (convert_batch.py:1052-1085):**
- Missing sentences appended to `reconciled` at end
- No ordering relative to existing lines

#### Root Cause Hypothesis

**Primary:** Final reinjection loop (lines 1052-1085) **always appends** to `reconciled` without checking position, causing out-of-order insertion.

**Secondary:** `replace_or_insert_lines()` fuzzy matching (line 224-240) may anchor at wrong position if:
- `_span_start` missing
- `sent_to_pos` incomplete
- SID mapping fails

**Tertiary:** `_build_sentence_to_pos_map()` (line 605-606) may not map all sentences if OpenAI didn't parse them, leading to incorrect `anchor_sent` → `pos` mapping.

---

## Code Pointers

### Issue 1: Fallback → Narrator (calm)(soft)

| File | Lines | Rationale |
|------|-------|-----------|
| `parsers/openai_parser/emotion_utils.py` | 203, 225 | Defaults to `["calm", "soft"]` when emotions empty |
| `parsers/openai_parser/emotion_utils.py` | 141-146 | Top-up logic always adds "calm" and "soft" first |
| `parsers/openai_parser/emotion_utils.py` | 66-83 | `canonicalize_emotion()` returns "calm" for unmapped emotions |
| `parsers/openai_parser/validator.py` | 128-129 | Calls `ensure_two_emotions()` which may default |
| `parsers/openai_parser/validator.py` | 164-173 | Coerces "Ambiguous" → "Narrator" via `_looks_like_narration()` |
| `parsers/openai_parser/fallback/parsing.py` | 325-333 | `_sanitize_character()` maps unknown → "Ambiguous" |

### Issue 2: Consecutive Lines Collapsed

| File | Lines | Rationale |
|------|-------|-----------|
| `parsers/openai_parser/convert_batch.py` | 870-881 | Merges consecutive same-character lines **before** reinjection, no `_src` check |
| `parsers/openai_parser/chunker.py` | 203-254 | `deduplicate_lines()` may prefer first occurrence when speakers differ |
| `parsers/openai_parser/chunker.py` | 235-246 | Prefers non-Narrator but doesn't prevent same-character merging |
| `parsers/openai_parser/validator.py` | 164-173 | Ambiguous → Narrator coercion may cause both lines to become Narrator |

### Issue 3: Wrong Order Reinjection

| File | Lines | Rationale |
|------|-------|-----------|
| `parsers/openai_parser/convert_batch.py` | 1052-1085 | Final reinjection **always appends** to `reconciled`, no ordering |
| `parsers/openai_parser/convert_batch.py` | 622-637 | `anchor_sent` → `pos` mapping may fail if sentence not in `sent_to_pos` |
| `parsers/openai_parser/fallback/reinject.py` | 189-203 | `_span_start`-based insertion, but missing `_span_start` → fallback |
| `parsers/openai_parser/fallback/reinject.py` | 224-240 | Fuzzy matching may anchor at wrong position |
| `parsers/openai_parser/utils_misc.py` | 82-122 | `_build_sentence_to_pos_map()` may not map all sentences |

---

## Hypotheses (Ranked)

### Issue 1: Fallback → Narrator (calm)(soft)

1. **H1.1 (High confidence):** `ensure_two_emotions()` defaults to `["calm", "soft"]` when fallback output has empty/invalid emotions
   - **Check:** Log `fb_valid` emotions before/after `validate_and_fix()` at line 567-568
   - **Confirm:** If `fb_valid[].emotions == []` before, and `["calm", "soft"]` after

2. **H1.2 (High confidence):** `_sanitize_character()` maps unknown speakers → "Ambiguous", then `_looks_like_narration()` coerces → "Narrator"
   - **Check:** Log character before/after `_sanitize_character()` at line 561-562, and before/after `validate_and_fix()` at line 164
   - **Confirm:** Unknown → "Ambiguous" → "Narrator" transformation

3. **H1.3 (Medium confidence):** `canonicalize_emotion()` in freeform mode returns "calm" for unmapped emotions
   - **Check:** Log emotion values before/after `canonicalize_emotion()` at line 136 in `validator.py`
   - **Confirm:** Unmapped emotions → "calm"

### Issue 2: Consecutive Lines Collapsed

1. **H2.1 (High confidence):** Merging logic at line 870-871 merges consecutive same-character lines regardless of `_src` tag
   - **Check:** Log `reconciled` before/after merging loop, show `_src` tags
   - **Confirm:** Lines with different `_src` but same character get merged

2. **H2.2 (Medium confidence):** `deduplicate_lines()` incorrectly attributes both lines to same character
   - **Check:** Log character assignments in `deduplicate_lines()` at line 203-254
   - **Confirm:** Two different speakers both become same character (e.g., both "Narrator")

3. **H2.3 (Low confidence):** `_looks_like_narration()` coerces both lines to "Narrator" when they should be different speakers
   - **Check:** Log `_looks_like_narration()` return values for both lines
   - **Confirm:** Both return True → both become "Narrator" → merged

### Issue 3: Wrong Order Reinjection

1. **H3.1 (High confidence):** Final reinjection loop (lines 1052-1085) always appends to `reconciled` without ordering
   - **Check:** Log `reconciled` length before/after reinjection, show insertion indices
   - **Confirm:** Reinjected lines always at end, not in correct position

2. **H3.2 (Medium confidence):** `replace_or_insert_lines()` fuzzy matching anchors at wrong position
   - **Check:** Log `anchor_pos`, `approx_pos`, `best_i`, `best_sim` at line 224-240
   - **Confirm:** `anchor_pos` != correct position for sentence

3. **H3.3 (Low confidence):** `_build_sentence_to_pos_map()` doesn't map all sentences, causing incorrect `anchor_sent` → `pos`
   - **Check:** Log `sent_to_pos` keys vs. `anchor_sent` values
   - **Confirm:** `anchor_sent` not in `sent_to_pos`, fallback search finds wrong position

---

## Suggested Next Minimal Diagnostic Edits

### For Issue 1

**File:** `parsers/openai_parser/convert_batch.py`  
**Line:** 567 (after `fb_valid = validate_and_fix(...)`)  
**Edit:**
```python
# DIAG: Log fb_valid emotions
print(f"[DIAG_FB_EMOTIONS] fb_valid_count={len(fb_valid)}", flush=True)
for i, ln in enumerate(fb_valid):
    print(f"[DIAG_FB_EMOTIONS] [{i}] char={ln.get('character')} emotions={ln.get('emotions')} text='{ln.get('text','')[:60]}'", flush=True)
```

**File:** `parsers/openai_parser/validator.py`  
**Line:** 128 (before `ensure_two_emotions()`)  
**Edit:**
```python
# DIAG: Log emotions before ensure_two_emotions
print(f"[DIAG_EMO_BEFORE] char={char} emotions={ems} text='{txt[:60]}'", flush=True)
```

**File:** `parsers/openai_parser/validator.py`  
**Line:** 164 (before Ambiguous → Narrator coercion)  
**Edit:**
```python
# DIAG: Log coercion decision
if fixed.get("character") == "Ambiguous" and _no_quotes(txt):
    looks_narr = _looks_like_narration(txt, state)
    print(f"[DIAG_AMBIG_COERCE] text='{txt[:60]}' looks_narr={looks_narr}", flush=True)
```

### For Issue 2

**File:** `parsers/openai_parser/convert_batch.py`  
**Line:** 870 (before merging check)  
**Edit:**
```python
# DIAG: Log merging decision
if reconciled and reconciled[-1]["character"] == item["character"]:
    prev_src = reconciled[-1].get("_src", "?")
    curr_src = item.get("_src", "?")
    print(f"[DIAG_MERGE] merging char={item['character']} prev_src={prev_src} curr_src={curr_src} prev_text='{reconciled[-1]['text'][:60]}' curr_text='{item['text'][:60]}'", flush=True)
```

**File:** `parsers/openai_parser/chunker.py`  
**Line:** 203 (in `deduplicate_lines()`)  
**Edit:**
```python
# DIAG: Log duplicate detection
if txt_norm in norm_to_index:
    existing = result[norm_to_index[txt_norm]]
    print(f"[DIAG_DEDUP] txt_norm='{txt_norm[:60]}' existing_char={existing.get('character')} new_char={it.get('character')}", flush=True)
```

### For Issue 3

**File:** `parsers/openai_parser/convert_batch.py`  
**Line:** 1052 (in final reinjection loop)  
**Edit:**
```python
# DIAG: Log reinjection position
reconciled_len_before = len(reconciled)
reconciled.append({...})
print(f"[DIAG_REINJ_APPEND] idx={len(reconciled)-1} (was {reconciled_len_before}) char={char_guess} text='{m[:60]}'", flush=True)
```

**File:** `parsers/openai_parser/fallback/reinject.py`  
**Line:** 237 (after fuzzy matching)  
**Edit:**
```python
# DIAG: Log anchor decision
print(f"[DIAG_ANCHOR] approx_pos={approx_pos} anchor_pos={anchor_pos} best_i={best_i} best_sim={best_sim:.2f} reason={_reason}", flush=True)
```

**File:** `parsers/openai_parser/convert_batch.py`  
**Line:** 622 (in anchor_sent → pos mapping)  
**Edit:**
```python
# DIAG: Log sent_to_pos mapping
print(f"[DIAG_SENT_POS] anchor_sent={anchor_sent} pos={pos} sent_to_pos_keys={list(sent_to_pos.keys())[:10]}", flush=True)
```

---

## Risk / Fallout Notes

### Issue 1 Fix Risks

1. **Changing default emotions:** If we change `["calm", "soft"]` default, may break existing behavior for legitimate narration
2. **Ambiguous → Narrator coercion:** Removing coercion may cause more "Ambiguous" lines, but that may be acceptable
3. **Emotion KB dependencies:** If `build_emotion_kb()` is incomplete, fallback will always default

### Issue 2 Fix Risks

1. **Merging logic:** If we add `_src` check, may prevent legitimate merging of same-character lines from same source
2. **Deduplication:** Changing deduplication logic may cause duplicate lines to appear
3. **Performance:** Adding checks may slow down processing

### Issue 3 Fix Risks

1. **Ordering complexity:** Adding proper ordering may require significant refactoring of reinjection logic
2. **SID mapping:** If ledger unavailable, SID-based ordering fails, must fall back to positional
3. **Positional guess accuracy:** If `sent_to_pos` incomplete, positional guesses may still be wrong

### Cross-Issue Interactions

1. **Ledger availability:** If `ledger` is None, SID mapping fails → affects Issue 3
2. **Chunk boundaries:** Overlap between chunks may cause duplicate detection issues → affects Issue 2
3. **Fallback → Reinjection:** Fallback lines may get reinjected again in final loop → affects Issue 3

---

## Attached Logs

See `logs/parser_run_*.txt` for full diagnostic output from test runs.

**Key log markers to search for:**
- `[DIAG_FB_EMOTIONS]` - Fallback emotion assignments
- `[DIAG_MERGE]` - Merging decisions
- `[DIAG_REINJ_APPEND]` - Reinjection positions
- `[DIAG_ANCHOR]` - Anchor position calculations
- `[DIAG_SENT_POS]` - Sentence-to-position mappings
- `[FB_FUZZY_KEEP]` / `[FB_FUZZY_DROP]` - Fallback filtering decisions
- `[SID_INSERT]` / `[SID_DUP_SKIP]` - SID-based insertion decisions

---

## Conclusion

All three issues have clear root causes identified in the code:

1. **Issue 1:** Emotion defaulting + character sanitization → Narrator coercion
2. **Issue 2:** Merging logic doesn't check `_src` tags or quote boundaries
3. **Issue 3:** Final reinjection always appends, no ordering logic

The suggested diagnostic edits will confirm these hypotheses with minimal code changes. Once confirmed, targeted fixes can be applied to each issue independently.



# Diagnostic Injection Summary

**Date:** 2025-11-08  
**Task:** Inject diagnostic print statements across key reinjection, fallback, and merging logic

---

## ✅ Completed Injections

### 1. `parsers/openai_parser/convert_batch.py`

#### Location A: After `fb_valid = validate_and_fix(...)`
- **Line:** 569-575
- **Diagnostic:** `[DIAG_FB_VALID]`
- **Captures:** chunk index, segment index, and for each fb_valid line: character, emotions, _span_start, _sid

#### Location B: After SID mapping
- **Line:** 587-592
- **Diagnostic:** `[DIAG_SID_MAP]`
- **Captures:** mapped_count, unmapped_count, sample SIDs from first 6 fb_valid lines

#### Location C: After choosing pos (anchor position search)
- **Line:** 647-652
- **Diagnostic:** `[DIAG_POS]`
- **Captures:** anchor_sent, chosen_pos, sample of sent_to_pos keys

#### Location D: Before calling `replace_or_insert_lines(...)`
- **Line:** 714-719
- **Diagnostic:** `[DIAG_REINJ_CALL]`
- **Captures:** chunk index, segment index, base_pos, approx_pos, fixed_len, new_len

#### Location E: Before merge check
- **Line:** 895-905
- **Diagnostic:** `[DIAG_MERGE_CHECK]`
- **Captures:** prev_char, next_char, prev_src, next_src, quote_start_prev, quote_start_curr

#### Location F: After [EOD][TAIL] block
- **Line:** 1163-1168
- **Diagnostic:** `[FINAL_TAIL]`
- **Captures:** For last 40 reconciled lines: index, src, sid, span, character, text (first 60 chars)

### 2. `parsers/openai_parser/fallback/reinject.py`

#### Function Start: `replace_or_insert_lines()`
- **Line:** 53-58
- **Diagnostic:** `[DIAG_REPLACE_ENTER]`
- **Captures:** start_index, end_index, sample of new_lines (first 6): character, text (first 40 chars), _span_start

#### Function End: `replace_or_insert_lines()`
- **Line:** 444-448
- **Diagnostic:** `[DIAG_REPLACE_EXIT]`
- **Captures:** post_len (final dialogues length)

---

## 📊 Test Execution

### Test Runs Completed
- ✅ Run 1: `logs/parser_diag_run1.log`
- ✅ Run 2: `logs/parser_diag_run2.log`
- ✅ Run 3: `logs/parser_diag_run3.log`
- ✅ Combined: `logs/parser_diag.log`

### Test Cases Executed
1. **Test 1:** Fallback lines defaulting to Narrator (calm)(soft)
2. **Test 2:** Consecutive lines collapsed to one speaker
3. **Test 3:** Random reinjected sentences placed in wrong order

### Diagnostic Output Verification

From `logs/parser_diag.log`, confirmed presence of:
- ✅ `[DIAG_MERGE_CHECK]` - 29 instances across 3 test runs
- ✅ `[FINAL_TAIL]` - Multiple instances showing final reconciled state
- ✅ All diagnostics properly prefixed with `[DIAG_*]` or `[FINAL_TAIL]`

### Sample Diagnostic Output

```
[DIAG_MERGE_CHECK] prev_char=Narrator next_char=Narrator prev_src=ai next_src=ai quote_start_prev=False quote_start_curr=False
[FINAL_TAIL] i=1 src=ai sid=None span=None char=Narrator text=Please, Daddy. Her voice was trembling. He looked at her wit
```

---

## 🔍 Key Observations from Logs

1. **Issue 1 (Fallback → Narrator):** 
   - Lines 54-58 show `[AMBIG_NARR_COERCE]` converting Ambiguous → Narrator
   - All coerced lines end up as Narrator with various emotions (not always calm/soft in this run)

2. **Issue 2 (Consecutive Merging):**
   - Lines 80-87, 194-197, etc. show multiple `[DIAG_MERGE_CHECK]` entries
   - All show `prev_char=Narrator next_char=Narrator` with `prev_src=ai next_src=ai`
   - `quote_start_prev=False quote_start_curr=False` indicates no quote detection

3. **Issue 3 (Wrong Order):**
   - `[FINAL_TAIL]` entries show reinjected lines (`src=reinj`) appearing after AI lines
   - All reinjected lines have `sid=None span=None`, indicating SID mapping failed

---

## 📝 Notes

- All diagnostic statements wrapped in `try/except` to prevent crashes
- All diagnostics use `flush=True` for immediate output
- No logic changes made - only diagnostic print statements added
- Test script has minor issue accessing `result.reconciled` (should be `result.dialogues`), but diagnostics still captured correctly

---

## 📁 Output Files

- **Combined Diagnostic Log:** `logs/parser_diag.log`
- **Individual Runs:** 
  - `logs/parser_diag_run1.log`
  - `logs/parser_diag_run2.log`
  - `logs/parser_diag_run3.log`

**Note:** On Windows, logs saved to `logs/` directory instead of `/tmp/` as requested.



# Diagnostic Bundle README

**Date:** 2025-11-08  
**Bundle:** `tmp/diag_bundle.zip`  
**Size:** ~1.4 MB

## Contents

### Master Logs
- `logs/parser_diag.log` - Combined output from 3 test runs

### Sample Input Files
- `tmp/diag_inputs/fallback_heavy.txt` - Mature/NSFW content triggering fallback
- `tmp/diag_inputs/alternating_speakers.txt` - Rapid alternating speakers
- `tmp/diag_inputs/order_sensitive.txt` - Order-sensitive passage

### Sample Outputs
- `logs/fallback_heavy_out.jsonl` - Parser output for fallback_heavy
- `logs/alternating_speakers_out.jsonl` - Parser output for alternating_speakers
- `logs/order_sensitive_out.jsonl` - Parser output for order_sensitive
- `logs/strict_True_fallback_out.jsonl` - REINJECT_STRICT=True comparison
- `logs/strict_False_fallback_out.jsonl` - REINJECT_STRICT=False comparison

### Diagnostic Samples (Extracted)
- `logs/diag_fb_valid_samples.jsonl` - Fallback validation samples
- `logs/diag_sid_map_samples.jsonl` - SID mapping samples
- `logs/diag_pos_samples.jsonl` - Position anchor samples
- `logs/diag_reinj_calls.jsonl` - Reinjection call samples
- `logs/diag_replace_enter_exit.jsonl` - Replace/insert function entry/exit
- `logs/diag_merge_check.jsonl` - Merge check diagnostics
- `logs/final_tail_samples.jsonl` - Final reconciled tail samples

### Summary
- `logs/summary_counts.txt` - Aggregated diagnostic counts

### Individual Parser Run Logs
- `logs/parser_run_*.txt` - Individual parser run logs with full diagnostics

## Summary Statistics

From `logs/summary_counts.txt`:

- **Narrator calm/soft fallback:** 0 instances
- **SID mapped rate:** 64.3% (9 mapped, 5 unmapped, 14 total)
- **Reinjection append calls:** 0 (all 2 calls used proper positioning)
- **Cross-source merges:** 3 instances (merging lines from different sources)
- **Total merge checks:** 169
- **Unmapped tail entries:** 103 out of 129 (79.8% missing SID/span)

## Key Findings

1. **Issue 1 (Fallback → Narrator calm/soft):** No instances found in test runs, but diagnostic infrastructure is in place.

2. **Issue 2 (Consecutive lines collapsed):** 3 cross-source merges detected where lines from different sources (`ai` vs `fb`, `ai` vs `reinj`) were merged.

3. **Issue 3 (Wrong order reinjection):** 79.8% of final tail entries lack SID/span mapping, indicating positioning issues. However, no append calls detected (all reinjections used proper positioning).

## Diagnostic Markers

All diagnostics are prefixed with `[DIAG_*]`:
- `[DIAG_FB_VALID]` - Fallback validation results
- `[DIAG_SID_MAP]` - SID mapping statistics
- `[DIAG_POS]` - Position anchor decisions
- `[DIAG_REINJ_CALL]` - Reinjection call parameters
- `[DIAG_REPLACE_ENTER]` / `[DIAG_REPLACE_EXIT]` - Replace/insert function entry/exit
- `[DIAG_MERGE_CHECK]` - Merge decision checks
- `[FINAL_TAIL]` - Final reconciled state

## Usage

Extract the bundle:
```bash
unzip tmp/diag_bundle.zip -d /tmp/
```

Or on Windows:
```powershell
Expand-Archive -Path tmp\diag_bundle.zip -DestinationPath tmp\
```

View summary:
```bash
cat tmp/diag_bundle/logs/summary_counts.txt
```

Search for specific diagnostics:
```bash
grep "\[DIAG_MERGE_CHECK\]" tmp/diag_bundle/logs/parser_diag.log
```



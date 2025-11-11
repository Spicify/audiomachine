#!/usr/bin/env python3
"""Extract diagnostic samples from logs and create summary counts."""
import re
from pathlib import Path
import json

logs_dir = Path("logs")
master_log = logs_dir / "parser_diag.log"

# Diagnostic patterns
patterns = {
    "diag_fb_valid_samples": r"\[DIAG_FB_VALID\].*",
    "diag_sid_map_samples": r"\[DIAG_SID_MAP\].*",
    "diag_pos_samples": r"\[DIAG_POS\].*",
    "diag_reinj_calls": r"\[DIAG_REINJ_CALL\].*",
    "diag_replace_enter_exit": r"\[DIAG_REPLACE_(ENTER|EXIT)\].*",
    "diag_merge_check": r"\[DIAG_MERGE_CHECK\].*",
    "final_tail_samples": r"\[FINAL_TAIL\].*",
}

# Extract samples
if master_log.exists():
    try:
        content = master_log.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Try with error handling
        content = master_log.read_text(encoding='utf-8', errors='replace')
    
    for name, pattern in patterns.items():
        matches = re.findall(pattern, content, re.MULTILINE)
        out_file = logs_dir / f"{name}.jsonl"
        with out_file.open('w', encoding='utf-8') as f:
            for match in matches:
                f.write(match + "\n")
        print(f"Extracted {len(matches)} lines to {out_file}")

# Create summary counts
summary = {}

# Count DIAG_FB_VALID with Narrator and ['calm','soft']
fb_valid_lines = re.findall(r"\[DIAG_FB_VALID\].*", content, re.MULTILINE)
narrator_calm_soft = 0
for line in fb_valid_lines:
    if "'Narrator'" in line and "'calm'" in line and "'soft'" in line:
        narrator_calm_soft += 1
summary["narrator_calm_soft_fallback"] = narrator_calm_soft

# SID mapped rate
sid_map_lines = re.findall(r"\[DIAG_SID_MAP\] mapped=(\d+) unmapped=(\d+)", content, re.MULTILINE)
total_mapped = sum(int(m) for m, _ in sid_map_lines)
total_unmapped = sum(int(u) for _, u in sid_map_lines)
total_attempted = total_mapped + total_unmapped
sid_rate = (total_mapped / total_attempted * 100) if total_attempted > 0 else 0.0
summary["sid_mapped_rate_pct"] = f"{sid_rate:.1f}%"
summary["sid_total_mapped"] = total_mapped
summary["sid_total_unmapped"] = total_unmapped
summary["sid_total_attempted"] = total_attempted

# Reinjection calls with approx_pos >= len(fixed)
reinj_lines = re.findall(r"\[DIAG_REINJ_CALL\].*fixed_len=(\d+).*approx_pos=(\d+)", content, re.MULTILINE)
append_calls = 0
for fixed_len_str, approx_pos_str in reinj_lines:
    try:
        fixed_len = int(fixed_len_str)
        approx_pos = int(approx_pos_str)
        if approx_pos >= fixed_len:
            append_calls += 1
    except ValueError:
        pass
summary["reinj_append_calls"] = append_calls
summary["reinj_total_calls"] = len(reinj_lines)

# DIAG_MERGE_CHECK with prev_src != next_src
merge_lines = re.findall(r"\[DIAG_MERGE_CHECK\].*prev_src=(\w+).*next_src=(\w+)", content, re.MULTILINE)
cross_source_merges = sum(1 for prev, next in merge_lines if prev != next and prev != "None" and next != "None")
summary["cross_source_merges"] = cross_source_merges
summary["total_merge_checks"] = len(merge_lines)

# Final tail entries with _sid=None and _span_start=None
tail_lines = re.findall(r"\[FINAL_TAIL\].*sid=(\w+).*span=(\w+)", content, re.MULTILINE)
unmapped_tail = sum(1 for sid, span in tail_lines if sid == "None" and span == "None")
summary["unmapped_tail_entries"] = unmapped_tail
summary["total_tail_entries"] = len(tail_lines)

# Write summary
summary_file = logs_dir / "summary_counts.txt"
with summary_file.open('w', encoding='utf-8') as f:
    f.write("Parser Diagnostic Summary Counts\n")
    f.write("=" * 50 + "\n\n")
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")

print(f"\nSummary written to {summary_file}")
print("\nSummary:")
for key, value in summary.items():
    print(f"  {key}: {value}")


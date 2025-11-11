#!/usr/bin/env python3
"""Extract diagnostics from all parser run logs."""
import re
from pathlib import Path
import json

logs_dir = Path("logs")

# Find all parser run logs
parser_logs = list(logs_dir.glob("parser_run_*.txt"))

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

# Collect all matches
all_matches = {name: [] for name in patterns.keys()}

for log_file in parser_logs:
    try:
        content = log_file.read_text(encoding='utf-8', errors='replace')
        for name, pattern in patterns.items():
            matches = re.findall(pattern, content, re.MULTILINE)
            all_matches[name].extend(matches)
    except Exception as e:
        print(f"Error reading {log_file}: {e}")

# Write extracted samples
for name, matches in all_matches.items():
    out_file = logs_dir / f"{name}.jsonl"
    with out_file.open('w', encoding='utf-8') as f:
        for match in matches:
            f.write(match + "\n")
    print(f"Extracted {len(matches)} lines to {out_file}")

# Create summary counts from all matches
summary = {}

# Count DIAG_FB_VALID with Narrator and ['calm','soft']
fb_valid_lines = all_matches["diag_fb_valid_samples"]
narrator_calm_soft = 0
for line in fb_valid_lines:
    if "'Narrator'" in line or '"Narrator"' in line:
        if "'calm'" in line and "'soft'" in line:
            narrator_calm_soft += 1
summary["narrator_calm_soft_fallback"] = narrator_calm_soft

# SID mapped rate
sid_map_lines = all_matches["diag_sid_map_samples"]
total_mapped = 0
total_unmapped = 0
for line in sid_map_lines:
    m = re.search(r"mapped=(\d+)", line)
    u = re.search(r"unmapped=(\d+)", line)
    if m:
        total_mapped += int(m.group(1))
    if u:
        total_unmapped += int(u.group(1))
total_attempted = total_mapped + total_unmapped
sid_rate = (total_mapped / total_attempted * 100) if total_attempted > 0 else 0.0
summary["sid_mapped_rate_pct"] = f"{sid_rate:.1f}%"
summary["sid_total_mapped"] = total_mapped
summary["sid_total_unmapped"] = total_unmapped
summary["sid_total_attempted"] = total_attempted

# Reinjection calls with approx_pos >= len(fixed)
reinj_lines = all_matches["diag_reinj_calls"]
append_calls = 0
for line in reinj_lines:
    m1 = re.search(r"fixed_len=(\d+)", line)
    m2 = re.search(r"approx_pos=(\d+)", line)
    if m1 and m2:
        try:
            fixed_len = int(m1.group(1))
            approx_pos = int(m2.group(1))
            if approx_pos >= fixed_len:
                append_calls += 1
        except ValueError:
            pass
summary["reinj_append_calls"] = append_calls
summary["reinj_total_calls"] = len(reinj_lines)

# DIAG_MERGE_CHECK with prev_src != next_src
merge_lines = all_matches["diag_merge_check"]
cross_source_merges = 0
for line in merge_lines:
    m1 = re.search(r"prev_src=(\w+)", line)
    m2 = re.search(r"next_src=(\w+)", line)
    if m1 and m2:
        prev = m1.group(1)
        next_src = m2.group(1)
        if prev != next_src and prev != "None" and next_src != "None":
            cross_source_merges += 1
summary["cross_source_merges"] = cross_source_merges
summary["total_merge_checks"] = len(merge_lines)

# Final tail entries with _sid=None and _span_start=None
tail_lines = all_matches["final_tail_samples"]
unmapped_tail = 0
for line in tail_lines:
    if "sid=None" in line and "span=None" in line:
        unmapped_tail += 1
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



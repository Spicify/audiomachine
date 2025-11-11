#!/usr/bin/env python3
"""Run parser on diagnostic sample inputs and extract diagnostics."""
import os
import sys
from pathlib import Path
import json
import re

# Set debug flags
os.environ["DEBUG_PARSER_DIAG"] = "1"
os.environ["DEBUG_EMOTIONS"] = "1"

from parsers.openai_parser.openai_parser import OpenAIParser

# Create output directories
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
inputs_dir = Path("tmp/diag_inputs")

# Run parser on each sample input
samples = ['fallback_heavy', 'alternating_speakers', 'order_sensitive']

for name in samples:
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")
    
    inp_file = inputs_dir / f"{name}.txt"
    if not inp_file.exists():
        print(f"ERROR: {inp_file} not found")
        continue
    
    try:
        parser = OpenAIParser()
        text = inp_file.read_text(encoding='utf-8')
        result = parser.convert(text)
        
        # Save formatted output
        out_file = logs_dir / f"{name}_out.jsonl"
        out_file.write_text(result.formatted_text, encoding='utf-8')
        print(f"Saved output to {out_file}")
        
        # Save full result as JSON
        result_json = {
            "formatted_text": result.formatted_text,
            "dialogues": result.dialogues,
            "stats": result.stats,
            "warnings": result.warnings,
            "errors": result.errors,
            "ambiguities": result.ambiguities
        }
        json_file = logs_dir / f"{name}_out.json"
        json_file.write_text(json.dumps(result_json, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"Saved JSON to {json_file}")
        
    except Exception as e:
        print(f"ERROR processing {name}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Running REINJECT_STRICT comparison runs")
print("="*60)

# Run comparison runs with REINJECT_STRICT toggle
inp_file = inputs_dir / "fallback_heavy.txt"
if inp_file.exists():
    text = inp_file.read_text(encoding='utf-8')
    
    for strict in (True, False):
        print(f"\nRunning with REINJECT_STRICT={strict}")
        try:
            parser = OpenAIParser()
            parser.REINJECT_STRICT = strict
            parser.legacy_base_parser = False
            result = parser.convert(text)
            
            out_file = logs_dir / f"strict_{strict}_fallback_out.jsonl"
            out_file.write_text(result.formatted_text, encoding='utf-8')
            print(f"Saved to {out_file}")
            
        except Exception as e:
            print(f"ERROR with strict={strict}: {e}")
            import traceback
            traceback.print_exc()

print("\nDone!")



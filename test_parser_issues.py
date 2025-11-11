#!/usr/bin/env python3
"""
Test script to reproduce three parser issues:
1. Fallback lines defaulting to Narrator (calm)(soft)
2. Consecutive lines collapsed to one speaker
3. Random reinjected sentences placed in wrong order
"""
import os
import sys
import json
from pathlib import Path

# Set debug flags
os.environ["DEBUG_PARSER_DIAG"] = "1"
os.environ["DEBUG_EMOTIONS"] = "1"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from parsers.openai_parser.openai_parser import OpenAIParser

def test_issue_1_fallback_narrator():
    """Test: Fallback lines defaulting to Narrator (calm)(soft)"""
    print("\n" + "="*80)
    print("TEST 1: Fallback lines defaulting to Narrator (calm)(soft)")
    print("="*80)
    
    # Mature content that might trigger fallback
    text = '''"Please, Daddy," she whispered, her voice trembling.
He looked at her with dark eyes.
"Good girl," he murmured, his hands moving lower.
She gasped as his fingers found their target.
"More," she begged, arching against him.'''
    
    parser = OpenAIParser()
    result = parser.convert(text)
    
    print("\n--- Final Output (last 10 lines) ---")
    for i, line in enumerate(result.reconciled[-10:], start=len(result.reconciled)-9):
        char = line.get('character', '?')
        emotions = line.get('emotions', [])
        txt = line.get('text', '')[:80]
        src = line.get('_src', '?')
        print(f"{i:03d}. [{src}] {char} {emotions}: {txt}")
    
    # Check for Narrator(calm)(soft) fallback lines
    narrator_calm_soft = [
        d for d in result.reconciled 
        if d.get('character') == 'Narrator' 
        and d.get('emotions') == ['calm', 'soft']
        and d.get('_src') in ('fb', 'reinj')
    ]
    
    print(f"\n--- Issue 1 Analysis ---")
    print(f"Total Narrator(calm)(soft) fallback lines: {len(narrator_calm_soft)}")
    for d in narrator_calm_soft[:5]:
        print(f"  - {d.get('text', '')[:80]}")
    
    return result, narrator_calm_soft

def test_issue_2_consecutive_collapsed():
    """Test: Consecutive lines collapsed to one speaker"""
    print("\n" + "="*80)
    print("TEST 2: Consecutive lines collapsed to one speaker")
    print("="*80)
    
    # Two characters speaking in succession
    text = '''"Please, Daddy," she said.
"Good girl," he replied.
She looked away.
He smiled.'''
    
    parser = OpenAIParser()
    result = parser.convert(text)
    
    print("\n--- Final Output ---")
    for i, line in enumerate(result.reconciled, start=1):
        char = line.get('character', '?')
        emotions = line.get('emotions', [])
        txt = line.get('text', '')[:80]
        src = line.get('_src', '?')
        print(f"{i:03d}. [{src}] {char} {emotions}: {txt}")
    
    # Check for consecutive same-character lines that should be different
    print(f"\n--- Issue 2 Analysis ---")
    consecutive_same = []
    for i in range(len(result.reconciled) - 1):
        curr = result.reconciled[i]
        next_line = result.reconciled[i + 1]
        if (curr.get('character') == next_line.get('character') 
            and curr.get('character') not in ('Narrator', 'Ambiguous')
            and '"' in curr.get('text', '') and '"' in next_line.get('text', '')):
            consecutive_same.append((i, curr, next_line))
    
    print(f"Consecutive same-character dialogue pairs: {len(consecutive_same)}")
    for idx, (i, curr, next_line) in enumerate(consecutive_same[:3], 1):
        print(f"  {idx}. Lines {i+1}-{i+2}:")
        print(f"     '{curr.get('text', '')[:60]}'")
        print(f"     '{next_line.get('text', '')[:60]}'")
    
    return result, consecutive_same

def test_issue_3_wrong_order():
    """Test: Random reinjected sentences placed in wrong order"""
    print("\n" + "="*80)
    print("TEST 3: Random reinjected sentences placed in wrong order")
    print("="*80)
    
    # Text with specific sentence order
    text = '''She entered the room.
The door closed behind her.
He looked up from his desk.
"Hello," he said.
She smiled.'''
    
    parser = OpenAIParser()
    result = parser.convert(text)
    
    print("\n--- Final Output with _src markers ---")
    for i, line in enumerate(result.reconciled, start=1):
        char = line.get('character', '?')
        txt = line.get('text', '')[:80]
        src = line.get('_src', '?')
        sid = line.get('_sid', '-')
        print(f"{i:03d}. [{src}] {char}: {txt} (sid={sid})")
    
    # Check for reinjected lines
    reinjected = [d for d in result.reconciled if d.get('_src') == 'reinj']
    print(f"\n--- Issue 3 Analysis ---")
    print(f"Total reinjected lines: {len(reinjected)}")
    for d in reinjected:
        print(f"  - {d.get('text', '')[:80]}")
    
    return result, reinjected

if __name__ == "__main__":
    print("Starting parser diagnostic tests...")
    print(f"DEBUG_PARSER_DIAG={os.getenv('DEBUG_PARSER_DIAG')}")
    print(f"DEBUG_EMOTIONS={os.getenv('DEBUG_EMOTIONS')}")
    
    results = {}
    
    try:
        result1, narrator_issues = test_issue_1_fallback_narrator()
        results['issue1'] = {
            'result': result1,
            'narrator_calm_soft_count': len(narrator_issues)
        }
    except Exception as e:
        print(f"ERROR in test 1: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result2, collapsed_issues = test_issue_2_consecutive_collapsed()
        results['issue2'] = {
            'result': result2,
            'collapsed_count': len(collapsed_issues)
        }
    except Exception as e:
        print(f"ERROR in test 2: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result3, reinjected_issues = test_issue_3_wrong_order()
        results['issue3'] = {
            'result': result3,
            'reinjected_count': len(reinjected_issues)
        }
    except Exception as e:
        print(f"ERROR in test 3: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Issue 1 (Narrator calm/soft): {results.get('issue1', {}).get('narrator_calm_soft_count', 0)} instances")
    print(f"Issue 2 (Collapsed speakers): {results.get('issue2', {}).get('collapsed_count', 0)} instances")
    print(f"Issue 3 (Wrong order reinject): {results.get('issue3', {}).get('reinjected_count', 0)} instances")
    
    # Save results to file
    output_file = Path("logs") / "test_results.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        # Convert results to serializable format
        serializable = {}
        for key, value in results.items():
            if 'result' in value:
                result_obj = value['result']
                serializable[key] = {
                    'stats': result_obj.stats,
                    'warnings': result_obj.warnings,
                    'errors': result_obj.errors,
                    'reconciled_count': len(result_obj.reconciled),
                    'narrator_calm_soft_count': value.get('narrator_calm_soft_count', 0),
                    'collapsed_count': value.get('collapsed_count', 0),
                    'reinjected_count': value.get('reinjected_count', 0),
                }
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")



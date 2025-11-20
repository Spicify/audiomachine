from pathlib import Path
path = Path('ui/tabs/raw_parser_tab.py')
text = path.read_text()
old = '        if st.button(\n            \" Finalize Parsed "Output\,\n            type=\primary\,\n            use_container_width=True,\n        ):\n            _apply_ambiguity_choices(auto_finalize=False)\n        return\n'
new = '        if st.button(\n            \" Finalize Parsed "Output\,\n            type=\primary\,\n            use_container_width=True,\n        ):\n            _apply_ambiguity_choices(auto_finalize=False)\n            st.experimental_rerun()\n        return\n'
if old not in text:
    raise SystemExit('pattern not found for finalize button')
text = text.replace(old, new, 1)
path.write_text(text)

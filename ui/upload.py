import streamlit as st
from utils.file_extractor import FileExtractor


def create_file_upload_interface():
    """Create interface for file upload processing"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 📄 Upload Text File")

    with col2:
        # Reset button for upload tab
        if st.button("🗑️ Reset Upload", type="secondary", use_container_width=True, key="reset_upload_btn"):
            # Clear all upload-related session state
            keys_to_clear = [
                'upload_dialogue_text',
                'upload_text_analysis',
                'upload_formatted_dialogue',
                'upload_parsed_dialogues',
                'upload_voice_assignments',
                'upload_voice_manager'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

            st.session_state.upload_dialogue_text = ""
            st.success("Upload tab reset successfully!")
            st.rerun()

    uploaded_file = st.file_uploader(
        "Upload a text file:",
        type=['txt', 'docx', 'pdf'],
        help="Upload a .txt, .docx, or .pdf file containing dialogue text",
        key="file_uploader"
    )

    if uploaded_file:
        try:
            # Show processing status
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Read file content as bytes
                file_content = uploaded_file.read()

                # Extract text based on file type
                extractor = FileExtractor()

                if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
                    content = extractor.extract_from_txt(file_content)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or uploaded_file.name.endswith('.docx'):
                    content = extractor.extract_from_docx(file_content)
                elif uploaded_file.type == "application/pdf" or uploaded_file.name.endswith('.pdf'):
                    content = extractor.extract_from_pdf(file_content)
                else:
                    # Fallback to text extraction
                    content = extractor.extract_from_txt(file_content)

            if not content or not content.strip():
                st.error("No text content found in the uploaded file.")
                return

            # Limit content size for performance
            if len(content) > 100000:  # 100k characters limit
                st.warning(
                    "⚠️ File is very large. Truncating to first 100,000 characters for performance.")
                content = content[:100000] + \
                    "\n\n# [Content truncated for performance]"

            word_count = len(content.split())
            st.success(f"✅ **File loaded:** {uploaded_file.name}")
            st.info(f"📊 **Word count:** {word_count:,} words")

            if word_count > 5000:
                st.warning(
                    "⚠️ Large file detected (>5k words). Processing may take longer.")
            elif word_count < 10:
                st.warning(
                    "⚠️ Very short file (<10 words). Please check file content.")

            # Preview first few lines
            lines = [line for line in content.split('\n')[:15] if line.strip()]
            if lines:
                with st.expander("👀 File Preview (first 15 non-empty lines)"):
                    for i, line in enumerate(lines, 1):
                        st.markdown(
                            f"**{i}.** {line[:120]}{'...' if len(line) > 120 else ''}")

            # Auto-load text into upload tab input field
            if 'upload_dialogue_text' not in st.session_state or not st.session_state.upload_dialogue_text:
                st.session_state.upload_dialogue_text = content
                st.success(
                    "File content automatically loaded into text input field below!")
            else:
                # Ask user if they want to replace existing content
                if st.button("🔄 Replace current text with uploaded file", type="secondary"):
                    st.session_state.upload_dialogue_text = content
                    st.success("Text replaced with uploaded file content!")
                    st.rerun()

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.error(
                "Please ensure the file is a valid text, Word, or PDF document.")

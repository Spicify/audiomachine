import streamlit as st
import re


class FileExtractor:
    """Extract text from various file formats"""

    @staticmethod
    def extract_from_txt(file_content: bytes) -> str:
        """Extract text from .txt file"""
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return file_content.decode('latin-1')
            except:
                return file_content.decode('utf-8', errors='ignore')

    @staticmethod
    def extract_from_docx(file_content: bytes) -> str:
        """Extract text from .docx file"""
        try:
            # Try using python-docx if available
            import docx
            from io import BytesIO

            doc = docx.Document(BytesIO(file_content))
            text_parts = []

            for paragraph in doc.paragraphs:
                text_parts.append(paragraph.text)

            return '\n'.join(text_parts)

        except ImportError:
            st.warning(
                "python-docx not available. Attempting basic text extraction...")
            # Fallback: try to extract readable text
            text = file_content.decode('utf-8', errors='ignore')
            # Remove common docx artifacts
            text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
            return text
        except Exception as e:
            st.error(f"Error extracting from DOCX: {e}")
            return file_content.decode('utf-8', errors='ignore')

    @staticmethod
    def extract_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            from io import BytesIO

            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text_parts = []

            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())

            return '\n'.join(text_parts)

        except ImportError:
            st.warning("PyPDF2 not available. Cannot extract PDF text.")
            return "Error: PDF extraction requires PyPDF2 library"
        except Exception as e:
            st.error(f"Error extracting from PDF: {e}")
            return f"Error extracting PDF: {e}"

import streamlit as st
from settings import VALID_PASSWORDS


def check_password():
    """Check if user has entered correct password"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_type = None

    if not st.session_state.authenticated:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin: 2rem 0;
        ">
            <h1 style="color: white; margin-bottom: 1rem;">🎭 AUDIOMACHINE</h1>
            <p style="color: white; opacity: 0.9;">Professional Audiobook Production Suite</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔐 Enter Access Password")
        password = st.text_input(
            "Password:", type="password", key="password_input")

        if st.button("🚀 Access System", type="primary", use_container_width=True):
            if password in VALID_PASSWORDS:
                st.session_state.authenticated = True
                st.session_state.user_type = VALID_PASSWORDS[password]
                st.success("✅ Access granted! Redirecting...")
                st.rerun()
            else:
                st.error(
                    "❌ Invalid password. Please contact the administrator for access.")

        st.markdown("---")
        st.info("🔒 This system is password-protected for authorized users only.")
        return False

    return True

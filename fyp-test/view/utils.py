import streamlit as st

def logout():
    """Handle user logout and redirect to landing page."""
    st.session_state.logged_in = False
    st.session_state.page = "landing"
    st.info("You have been logged out.")
    st.rerun()
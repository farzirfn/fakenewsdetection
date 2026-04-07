import streamlit as st
from view.login import login_page
from view.upload import upload_page
from view.admin import stats_page
from view.user import user_home
from view.model import model_page
from view.augment import augment_page
from view.validationresult import main as validation_result

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# SESSION STATE INITIALIZATION
# ================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "user_home"

# ================================
# HEADER
# ================================
col1, col2 = st.columns([9, 2])
with col1:
    st.markdown("## 📰 Fake News Detection System via Augmented")
    st.markdown("---")

with col2:
    if not st.session_state.logged_in:
        if st.button("🔑 Admin Login"):
            st.session_state.page = "login"
            st.rerun()

# ================================
# SIDEBAR - ONLY IF LOGGED IN
# ================================
def sidebar_button(label, page_name):
    """Helper for sidebar buttons with active highlight"""
    if st.session_state.page == page_name:
        st.button(f"✅ {label}", use_container_width=True, key=label)
    else:
        if st.button(label, use_container_width=True, key=label):
            st.session_state.page = page_name
            st.rerun()

if st.session_state.logged_in:
    with st.sidebar:
        st.markdown("## 🛠️ Admin Panel")
        st.markdown("---")

        sidebar_button("🏠 Dashboard", "stats")
        sidebar_button("📤 Upload Data", "upload")
        sidebar_button("⚙️ Train Model", "model")
        sidebar_button("🔄 Augment Data", "augment")
        sidebar_button("📊 Validation Results", "validationresult")

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.page = "user_home"
            st.rerun()

# ================================
# PAGE ROUTING
# ================================
if st.session_state.page == "user_home":
    user_home()
elif st.session_state.page == "login":
    # LOGIN PAGE
    login_page()
    # If login is successful inside login_page(), make sure to set:
    # st.session_state.logged_in = True
    # st.session_state.page = "stats"
elif st.session_state.logged_in:
    if st.session_state.page == "stats":
        stats_page()
    elif st.session_state.page == "upload":
        upload_page()
    elif st.session_state.page == "model":
        model_page()
    elif st.session_state.page == "augment":
        augment_page()
    elif st.session_state.page == "validationresult":
        validation_result()
    else:
        st.session_state.page = "stats"
        st.rerun()
else:
    st.session_state.page = "user_home"
    st.rerun()

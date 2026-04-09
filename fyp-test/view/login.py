import streamlit as st
import mysql.connector

# -------------------------------
# DB Connection
# -------------------------------
def create_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        port=st.secrets["mysql"]["port"],
        user=st.secrets["mysql"]["username"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"]
    )

# -------------------------------
# Login Page
# -------------------------------
def login_page():
    st.markdown("""
    <style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 3rem; padding-bottom: 2rem; }

    .login-wrap {
        max-width: 400px;
        margin: 0 auto;
        padding: 2.5rem 2rem;
        background: #f7f6f2;
        border-radius: 16px;
        border: 0.5px solid #e0ded8;
    }
    .login-title {
        font-size: 22px;
        font-weight: 500;
        color: #2c2c2a;
        text-align: center;
        margin-bottom: 4px;
    }
    .login-sub {
        font-size: 13px;
        color: #888780;
        text-align: center;
        margin-bottom: 1.75rem;
    }
    .thin-divider {
        border: none;
        border-top: 0.5px solid #e0ded8;
        margin: 1.25rem 0;
    }
    .help-text {
        font-size: 12px;
        color: #aaa;
        text-align: center;
        margin-top: 1rem;
        line-height: 1.6;
    }

    @media (prefers-color-scheme: dark) {
        .login-wrap  { background: #1e1e1c; border-color: #333; }
        .login-title { color: #e0ded8; }
        .login-sub   { color: #666; }
        .thin-divider{ border-color: #333; }
        .help-text   { color: #555; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Back button
    if st.button("← Back", key="back_home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Center card using columns
    _, col, _ = st.columns([1, 2, 1])

    with col:
        st.markdown("""
        <div class='login-wrap'>
            <div class='login-title'>Admin login</div>
            <div class='login-sub'>Secure access to dashboard</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        username = st.text_input(
            "Username",
            placeholder="Enter username",
            key="username_input",
            label_visibility="collapsed"
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password",
            key="password_input",
            label_visibility="collapsed"
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        if st.button("Login →", key="admin_login", use_container_width=True):
            if not username or not password:
                st.warning("Please enter both username and password.")
            else:
                with st.spinner("Authenticating..."):
                    try:
                        conn = create_connection()
                        cursor = conn.cursor(dictionary=True)
                        cursor.execute("SELECT password FROM user WHERE username = %s", (username,))
                        result = cursor.fetchone()
                        cursor.close()
                        conn.close()

                        if result and password == result["password"]:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.page = "admin_home"
                            st.success("Login successful.")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")

                    except mysql.connector.Error as e:
                        st.error(f"Database error: {str(e)}")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

        st.markdown("""
        <div class='help-text'>
            Forgot your password? Contact your system administrator.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    login_page()
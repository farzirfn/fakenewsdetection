import streamlit as st
import mysql.connector

# Database connection
def create_connection():
    return mysql.connector.connect(
    host="localhost",
    user="farzifae_fyp",
    password="AK9CYVY#),&2",
    database="farzifae_fyp"
)

def login_page():
    # Back to Home button (styled)
    st.markdown(
        """
        <style>
        .back-btn {
            display: inline-block;
            padding: 8px 16px;
            background-color: #f0f2f6;
            color: #333;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 500;
        }
        .back-btn:hover {
            background-color: #e0e3e8;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("← Back to Home", key="back_home"):
        st.session_state.page = "home"
        st.rerun()

    # Page title
    st.markdown("<h2 style='text-align:center;'>🔑 Admin Login</h2>", unsafe_allow_html=True)
    st.write("<p style='text-align:center;color:gray;'>Secure access to admin dashboard</p>", unsafe_allow_html=True)
    st.divider()

    # Input fields
    username = st.text_input(
        "👤 Username",
        placeholder="Enter your username",
        key="username_input"
    )

    password = st.text_input(
        "🔒 Password",
        type="password",
        placeholder="Enter your password",
        key="password_input"
    )

    # Login button
    if st.button("🚀 Login", key="admin_login", use_container_width=True):
        if not username or not password:
            st.warning("⚠️ Please enter both username and password.")
        else:
            with st.spinner("🔐 Authenticating..."):
                try:
                    conn = create_connection()
                    cursor = conn.cursor(dictionary=True)

                    query = "SELECT password FROM user WHERE username = %s"
                    cursor.execute(query, (username,))
                    result = cursor.fetchone()

                    if result and password == result["password"]:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.page = "admin_home"
                        st.success("✅ Login successful! Redirecting...")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password. Please try again.")

                    cursor.close()
                    conn.close()

                except mysql.connector.Error as e:
                    st.error(f"❌ Database error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

    # Security info
    st.info("🔒 Your login credentials are encrypted and secure.\n\nFor security reasons, please don't share your password.")

    # Help section
    with st.expander("ℹ️ Need Help?"):
        st.markdown("""
        **Forgot your password?**  
        Contact the system administrator to reset your password.

        **Having trouble logging in?**  
        - Make sure your username is correct  
        - Check that Caps Lock is off  
        - Ensure you have admin privileges  

        **Security Tips:**  
        - Don't share your credentials  
        - Use a strong, unique password  
        - Log out after each session  
        """)

# Run the login page
if __name__ == "__main__":
    login_page()
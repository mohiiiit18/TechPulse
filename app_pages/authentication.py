import streamlit as st

from utils.theme import apply_base_css, page_header
from utils.auth import (
    register_user,
    login_user,
    logout_user,
    is_authenticated,
    get_user_history,
    save_profile,
    get_profile,
    submit_feedback,
)
from utils.db import is_database_available, is_database_configured

apply_base_css()

page_header(
    ":material/account_circle:",
    "TechPulse Account & Preferences",
    "Manage your account, career preferences, analysis history, and platform feedback."
)

# Database connectivity banner if offline
if not is_database_available():
    st.warning(
        ":material/database_off: Database connection is currently offline. "
        "Account authentication, saved profiles, and history require an active PostgreSQL connection. "
        "Check your `DATABASE_URL` in `.env` or Streamlit secrets.",
        icon=":material/warning:"
    )

if is_authenticated():

    user = st.session_state.get("user", {})
    user_name = user.get("name", "User")
    user_email = user.get("email", "")
    user_id = user.get("id")

    # Account Information
    with st.container(border=True):
        st.subheader(
            f"Welcome, {user_name}!",
            icon=":material/badge:"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Email:** `{user_email}`")

        with col2:
            st.markdown(f"**User ID:** `{user_id}`")

        st.space("small")

        if st.button(
            "Log Out",
            icon=":material/logout:",
            type="primary"
        ):
            logout_user()
            st.rerun()

    # Career Profile Section
    st.space("small")

    st.subheader(
        "Your Career Profile",
        icon=":material/person:"
    )
    st.caption("These preferences automatically personalize Career Matcher and Job Recommendations.")

    profile = get_profile(user_id) if user_id else None

    if profile:
        default_skills = ", ".join(profile.get("skills") or [])
        default_role = profile.get("target_role") or ""
        default_location = profile.get("preferred_location") or ""
        default_experience = profile.get("experience_level") or ""
    else:
        default_skills = ""
        default_role = ""
        default_location = ""
        default_experience = ""

    with st.container(border=True):

        with st.form("career_profile_form"):

            skills = st.text_input(
                "Your Skills (comma-separated)",
                value=default_skills,
                placeholder="Python, SQL, Pandas, Machine Learning"
            )

            target_role = st.text_input(
                "Target Job Role",
                value=default_role,
                placeholder="Data Scientist"
            )

            location = st.text_input(
                "Preferred Job Location",
                value=default_location,
                placeholder="Delhi, Bangalore, Remote, United States"
            )

            exp_options = [
                "Fresher",
                "0-1 Years",
                "1-3 Years",
                "3-5 Years",
                "5+ Years"
            ]
            exp_index = exp_options.index(default_experience) if default_experience in exp_options else 0

            experience = st.selectbox(
                "Experience Level",
                options=exp_options,
                index=exp_index
            )

            submitted = st.form_submit_button(
                "Save Career Profile",
                icon=":material/save:",
                type="primary",
                use_container_width=True
            )

            if submitted:
                skills_list = [
                    skill.strip()
                    for skill in skills.split(",")
                    if skill.strip()
                ]

                ok, msg = save_profile(
                    user_id=user_id,
                    skills=skills_list,
                    target_role=target_role.strip(),
                    location=location.strip(),
                    experience=experience
                )

                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    # Feedback Section
    st.space("small")
    with st.container(border=True):
        st.subheader("Platform Feedback", icon=":material/rate_review:")
        st.caption("How has your experience been with TechPulse? Your rating helps improve recommendations.")

        with st.form("feedback_form"):
            rating = st.slider("Rating (1 = Needs Work, 5 = Excellent)", min_value=1, max_value=5, value=5, step=1)
            comments = st.text_area("Comments & Suggestions", placeholder="Tell us what you like or what features you'd love to see...")
            fb_submit = st.form_submit_button("Submit Feedback", icon=":material/send:", type="primary")

            if fb_submit:
                ok, fb_msg = submit_feedback(user_id=user_id, rating=rating, comments=comments)
                if ok:
                    st.success(fb_msg)
                else:
                    st.error(fb_msg)

else:

    tab_login, tab_register, tab_feedback = st.tabs(
        ["Login", "Create Account", "Feedback"]
    )

    # Login
    with tab_login:

        st.subheader(
            "Welcome Back",
            icon=":material/login:"
        )

        login_email = st.text_input(
            "Email",
            key="login_email"
        )

        login_password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button(
            "Sign In",
            icon=":material/login:",
            type="primary",
            key="btn_login"
        ):

            if not login_email or not login_password:
                st.warning("Please enter both email and password.")

            elif login_user(login_email, login_password):
                st.success("Login successful!")
                st.rerun()

            else:
                st.error("Invalid email or password. Please try again.")

    # Registration
    with tab_register:

        st.subheader(
            "Register New Account",
            icon=":material/person_add:"
        )

        reg_name = st.text_input(
            "Full Name",
            key="register_name"
        )

        reg_email = st.text_input(
            "Email Address",
            key="register_email"
        )

        reg_password = st.text_input(
            "Password (minimum 8 characters)",
            type="password",
            key="register_password"
        )

        if st.button(
            "Create Account",
            icon=":material/person_add:",
            type="primary",
            key="btn_register"
        ):

            success, message = register_user(
                reg_name,
                reg_email,
                reg_password
            )

            if success:
                st.success(message)
                if login_user(reg_email, reg_password):
                    st.rerun()
            else:
                st.error(message)

    # Anonymous Feedback Tab
    with tab_feedback:
        st.subheader("Community Feedback", icon=":material/rate_review:")
        st.caption("You can leave feedback even without signing in.")
        
        with st.form("anon_feedback_form"):
            anon_rating = st.slider("Rating (1-5 stars)", min_value=1, max_value=5, value=5, step=1)
            anon_comments = st.text_area("Your thoughts", placeholder="Feedback, questions or bug reports...")
            anon_submit = st.form_submit_button("Send Feedback", icon=":material/send:", type="primary")

            if anon_submit:
                ok, fb_msg = submit_feedback(user_id=None, rating=anon_rating, comments=anon_comments)
                if ok:
                    st.success(fb_msg)
                else:
                    st.error(fb_msg)

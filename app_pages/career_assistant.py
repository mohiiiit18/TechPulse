import streamlit as st
from utils.theme import apply_base_css, page_header
from utils.data_loader import load_data, get_dataset_date_range
from src.career_assistant import CareerAssistantApp, ChatSession, SUGGESTED_QUESTIONS
from src import llm_provider

df = load_data()
_latest = get_dataset_date_range()["latest"]
_latest_str = f"{_latest:%B %Y}" if _latest is not None else "unknown"

apply_base_css()
page_header(":material/forum:", "AI Career Assistant",
            f"Ask about roles, skills, salary or interviews. Answers are grounded in the "
            f"cleaned dataset (latest records {_latest_str}) and TechPulse's trained models — "
            "nothing about the market or your qualifications is invented.")

# ── Session memory ───────────────────────────────────────────────────────────
if "ca_session" not in st.session_state:
    st.session_state["ca_session"] = ChatSession()
if "ca_resume_consent" not in st.session_state:
    st.session_state["ca_resume_consent"] = False
if "ca_use_llm" not in st.session_state:
    st.session_state["ca_use_llm"] = False

session: ChatSession = st.session_state["ca_session"]
app = CareerAssistantApp(df)

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### :material/forum: Assistant context")
    st.caption("Only information you provide here is used. Nothing is inferred "
               "or invented about you.")

    skills_in = st.text_input("Your skills (comma-separated)",
                              value=", ".join(session.profile.get("skills", [])),
                              placeholder="python, sql, machine learning")
    role_in = st.text_input("Target role",
                            value=session.profile.get("target_role", ""),
                            placeholder="e.g. Data Scientist")
    loc_in = st.text_input("Preferred location",
                           value=session.profile.get("preferred_location", ""),
                           placeholder="e.g. United States")
    exp_in = st.selectbox("Experience level (optional)",
                          ["", "junior", "mid", "senior"],
                          index=0,
                          help="Used only by the salary model when supported.")
    resume_btn = st.button("Use skills from Resume analyzer",
                           icon=":material/history_edu:",
                           disabled=not st.session_state.get("resume_skills"))

    st.divider()
    st.markdown("#### :material/key: LLM enhancement (optional)")
    if llm_provider.llm_available():
        st.caption("LLM key detected — answers can be rephrased by the LLM "
                   "using dataset facts only.")
        use_llm = st.checkbox("Enhance answers with LLM", value=session.profile.get("use_llm", False))
    else:
        st.caption("No LLM key configured — the assistant works fully offline "
                   "with grounded templates. Add TECHPULSE_LLM_API_KEY (env) or "
                   "LLM_API_KEY (st.secrets) to enable.")
        use_llm = False

    resume_consent = False
    if use_llm:
        resume_consent = st.checkbox(
            "Share resume text with the LLM (requires consent)",
            value=False,
            help="Off by default. Resume text is only sent when you opt in here.")

    st.divider()
    if st.button("Clear conversation", icon=":material/delete_history:"):
        session.clear()
        st.rerun()

# Apply sidebar context to the session profile
session.profile["skills"] = [s.strip().lower() for s in (skills_in or "").split(",") if s.strip()]
session.profile["target_role"] = (role_in or "").strip() or None
session.profile["preferred_location"] = (loc_in or "").strip() or None
session.profile["seniority"] = exp_in or None
session.profile["use_llm"] = use_llm
if resume_btn:
    session.profile["resume_skills"] = st.session_state.get("resume_skills", [])

# ── Suggested questions ──────────────────────────────────────────────────────
with st.container():
    st.markdown("**Suggested questions**")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state["ca_pending"] = q

# ── Chat history ─────────────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    for turn in session.history:
        with st.chat_message(turn.role):
            st.markdown(turn.text)

# ── Input ────────────────────────────────────────────────────────────────────
user_msg = st.chat_input("Ask a career question...")
pending = st.session_state.pop("ca_pending", None) or user_msg
if pending:
    session.add("user", pending)
    with chat_container:
        with st.chat_message("user"):
            st.markdown(pending)
    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("Checking the dataset and models..."):
                turn = app.handle(session, pending,
                                  use_llm=use_llm,
                                  resume_consent=resume_consent,
                                  resume_text=str(st.session_state.get("resume_raw_text", ""))
                                  if resume_consent else "")
            st.markdown(turn.text)

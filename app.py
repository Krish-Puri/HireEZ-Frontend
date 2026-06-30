"""
HireEZ — AI-Powered Recruitment Platform
Dark Theme · Port 8502
Pipeline: Upload Dataset → Job Description → Parse Resumes → AI Evaluation
          → GitHub Analysis → Rank → Send Test Links → Upload Test Results
          → Shortlist → Schedule Interviews → Google Meet
"""

import streamlit as st
st.set_page_config(
    page_title="HireEZ — AI Recruitment Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

import requests
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
import pytz

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

import os
API_BASE = st.secrets.get("API_BASE", os.environ.get("API_BASE", "http://localhost:8002"))

# DEBUG: Remove this line after debugging
st.write(f"DEBUG: API_BASE = {API_BASE}")

# ─────────────────────────────────────────────────────────────────────────────
# Session State Defaults
# ─────────────────────────────────────────────────────────────────────────────

if "active_job_id" not in st.session_state:
    st.session_state["active_job_id"] = None
if "active_job_label" not in st.session_state:
    st.session_state["active_job_label"] = "All Jobs"
if "test_upload_done" not in st.session_state:
    st.session_state["test_upload_done"] = False
if "rank_done" not in st.session_state:
    st.session_state["rank_done"] = False

# ─────────────────────────────────────────────────────────────────────────────
# Dark Theme CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
* { box-sizing: border-box; }
html, body, .stApp {
    background-color: #0e1117 !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio span {
    color: #c9d1d9 !important;
}
[data-testid="stMetric"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] {
    color: #c9d1d9 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] td {
    background-color: #161b22 !important;
    color: #c9d1d9 !important;
    border-radius: 12px !important;
    border: 1px solid #30363d !important;
}
[data-testid="stDataFrame"] th {
    background-color: #21262d !important;
    color: #8b949e !important;
}
details[data-testid="stExpander"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
}
.stTabs [data-baseweb="tab-list"] {
    background-color: #21262d !important;
    border-radius: 10px !important;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    color: #8b949e !important;
    font-weight: 500 !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSlider"] label {
    color: #8b949e !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploader"] {
    border: 2px dashed #30363d !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    background-color: #161b22 !important;
}
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
hr { border-color: #30363d !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0e1117 !important; }
::-webkit-scrollbar-thumb { background: #30363d !important; border-radius: 3px; }
.stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline step definitions
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_STEPS = [
    ("📤", "Upload Dataset",         "1"),
    ("💼", "Job Description",        "2"),
    ("📄", "Parse Resumes",          "3"),
    ("🤖", "AI Evaluation",         "4"),
    ("💻", "GitHub Analysis",        "5"),
    ("🏆", "Score & Rank",           "6"),
    ("✉️",  "Send Test Links",        "7"),
    ("📊", "Upload Test Results",    "8"),
    ("✅",  "Shortlist",             "9"),
    ("📅",  "Schedule Interviews",   "10"),
    ("🎥",  "Google Meet Sent",      "11"),
]

def pipeline_progress(candidates: list) -> dict:
    """Derive pipeline stage completion from candidate statuses."""
    total = len(candidates)
    if total == 0:
        return {s[1]: False for s in PIPELINE_STEPS}
    df = pd.DataFrame(candidates)
    stages = {
        "Upload Dataset":       True,
        "Job Description":     True,
        "Parse Resumes":       bool((df["status"] != "Uploaded").sum()),
        "AI Evaluation":       bool(df["final_score"].notna().sum()),
        "GitHub Analysis":     bool(df["github_score"].notna().sum()),
        "Score & Rank":        bool(df["candidate_rank"].notna().sum()),
        "Send Test Links":     bool((df["status"] == "Test Link Sent").sum()),
        "Upload Test Results": bool(df["test_la"].notna().sum()),
        "Shortlist":          bool(((df["test_la"] + df["test_code"]) >= 80).sum()) if "test_la" in df.columns else False,
        "Schedule Interviews": bool((df["status"] == "Interview Scheduled").sum()),
        "Google Meet Sent":    bool((df["status"] == "Interview Scheduled").sum()),
    }
    return stages

# ─────────────────────────────────────────────────────────────────────────────
# API Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=15)
def get_candidates():
    try:
        return requests.get(f"{API_BASE}/candidates/", timeout=10).json()
    except Exception:
        return []

@st.cache_data(ttl=15)
def get_jobs():
    try:
        return requests.get(f"{API_BASE}/jobs/", timeout=10).json()
    except Exception:
        return []

@st.cache_data(ttl=15)
def get_interviews():
    try:
        return requests.get(f"{API_BASE}/interviews/", timeout=10).json()
    except Exception:
        return []

def post_candidates_csv(file_bytes, filename, job_id=None):
    return requests.post(f"{API_BASE}/candidates/upload",
        files={"file": (filename, file_bytes, "text/csv")},
        data={"job_id": str(job_id) if job_id else ""},
        timeout=60).json()

def post_test_results(file_bytes, filename):
    return requests.post(f"{API_BASE}/tests/upload-results",
        files={"file": (filename, file_bytes, "text/csv")}, timeout=60).json()

@st.cache_data(ttl=15)
def get_shortlisted(threshold=50.0):
    try:
        return requests.get(f"{API_BASE}/tests/shortlisted",
            params={"threshold": threshold}, timeout=10).json()
    except Exception:
        return []

@st.cache_data(ttl=15)
def get_shortlisted_after_test(min_total=50.0):
    """Shortlist only candidates who received test links and have test scores."""
    try:
        return requests.get(f"{API_BASE}/tests/shortlisted-after-test",
            params={"min_total": min_total}, timeout=10).json()
    except Exception:
        return []

def send_test_links(threshold: float = 50.0):
    return requests.post(f"{API_BASE}/tests/send-test-links", params={"threshold": threshold}, timeout=60).json()

def rank_all():
    return requests.post(f"{API_BASE}/candidates/rank-all", timeout=60).json()

def schedule_interviews(**kwargs):
    return requests.post(f"{API_BASE}/interviews/schedule", params=kwargs, timeout=60).json()

def create_job(payload):
    return requests.post(f"{API_BASE}/jobs/", json=payload, timeout=10)

def clear_cache():
    get_candidates.clear()
    get_jobs.clear()
    get_interviews.clear()
    get_shortlisted.clear()
    get_shortlisted_after_test.clear()


def get_filtered_candidates():
    """Return candidates filtered by the active job selection."""
    all_candidates = get_candidates()
    job_id = st.session_state.get("active_job_id")
    if job_id is None:
        return all_candidates
    return [c for c in all_candidates if c.get("job_id") == job_id]

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation — exact pipeline order
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🚀 **HireEZ**")
    st.caption("AI Recruitment Platform")
    st.markdown("---")

    PAGES = [
        "📤  Upload Dataset",
        "💼  Job Description",
        "📄  Parse Resumes",
        "🤖  AI Evaluation",
        "💻  GitHub Analysis",
        "🏆  Score & Rank",
        "✉️  Send Test Links",
        "📊  Upload Test Results",
        "✅  Shortlist",
        "📅  Schedule Interviews",
        "🎥  Interview Status",
    ]
    page = st.radio("Pipeline Steps", PAGES, index=0)

    st.markdown("---")

    # ── Active Job Filter ───────────────────────────────────────────────────────
    st.markdown("**💼  Active Job**")
    jobs = get_jobs()
    job_options = ["All Jobs"] + [f"ID {j['id']}: {j['title']}" for j in jobs]

    selected_job_label = st.selectbox(
        "Filter candidates by job:",
        job_options,
        index=0,
        key="active_job_label",
    )
    if selected_job_label == "All Jobs":
        st.session_state["active_job_id"] = None
    else:
        st.session_state["active_job_id"] = int(selected_job_label.split(":")[0].replace("ID ", "").strip())

    st.markdown("---")

    # ── Manage Data ──────────────────────────────────────────────────────────────
    st.markdown("**🗑️  Manage Data**")
    with st.expander("Clear Data", expanded=False):
        clear_candidates = st.button("🗑️  Clear All Candidates", type="secondary", width="stretch")
        if clear_candidates:
            try:
                requests.post(f"{API_BASE}/admin/clear-candidates", timeout=10)
                st.success("✅ Candidates cleared")
                st.rerun()
            except Exception:
                st.error("❌ Failed to clear candidates")

        clear_tests = st.button("🗑️  Clear Test Results", type="secondary", width="stretch")
        if clear_tests:
            try:
                requests.post(f"{API_BASE}/admin/clear-test-results", timeout=10)
                st.success("✅ Test results cleared")
                st.rerun()
            except Exception:
                st.error("❌ Failed to clear test results")

        clear_all = st.button("🗑️  Clear Everything", type="secondary", width="stretch")
        if clear_all:
            try:
                requests.post(f"{API_BASE}/admin/clear-all", timeout=10)
                st.success("✅ All data cleared")
                st.rerun()
            except Exception:
                st.error("❌ Failed to clear all data")

    st.markdown("---")
    st.caption(f"HireEZ v1.0.0  •  {datetime.now().strftime('%b %d, %Y')}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 1 — Upload Dataset
# ─────────────────────────────────────────────────────────────────────────────

if page == "📤  Upload Dataset":
    st.title("📤  Upload Candidate Dataset")
    st.markdown("Upload a CSV or Excel file containing candidate information. The AI will process each candidate automatically.")

    col_info, col_fmt = st.columns([1, 1])
    with col_info:
        st.markdown("""
        **Accepted formats:** `.csv` and `.xlsx` (Excel)
        - Duplicate candidates (by email) are skipped automatically
        - After upload, candidates enter the pipeline for evaluation
        - Supports all standard candidate fields
        """)
    with col_fmt:
        st.markdown("**📋 Expected Columns**")
        sample = pd.DataFrame([
            {"Name":"Priya Sharma","Email":"priya@example.com","College":"IIT Delhi","Branch":"CS","CGPA":"8.9","Best AI Project":"Image Classifier using CNN","Research Work":"NLP Paper","GitHub Profile":"github.com/priya","Resume Link":"https://..."},
            {"Name":"Rahul Verma","Email":"rahul@example.com","College":"MIT","Branch":"EE","CGPA":"9.1","Best AI Project":"Chatbot with RL","Research Work":"","GitHub Profile":"github.com/rahul","Resume Link":"https://..."},
        ])
        st.dataframe(sample, hide_index=True, height=130, width="stretch")

    st.markdown("---")

    # Job selector — required before upload
    jobs = get_jobs()
    if not jobs:
        st.warning("⚠️  No job created yet. Please go to **Job Description** first and create a job before uploading candidates.")
    else:
        selected_job_label = st.selectbox(
            "🏢  Select Job to match candidates against:",
            [f"ID {j['id']}: {j['title']}" for j in jobs],
            key="upload_job_select",
        )
        selected_job_id = int(selected_job_label.split(":")[0].replace("ID ", "").strip())
        st.session_state["upload_job_id"] = selected_job_id

    uploaded = st.file_uploader(
        "Drop your candidate dataset here (CSV or Excel)",
        type=["csv", "xlsx"],
    )

    if uploaded:
        fname = uploaded.name.lower()
        if st.button("🚀  Upload & Process Dataset", type="primary", width="stretch"):
            with st.spinner("Uploading candidates — pipeline runs in background..."):
                file_bytes = uploaded.getvalue()
                job_id = st.session_state.get("upload_job_id")
                result = post_candidates_csv(file_bytes, uploaded.name, job_id=job_id)
            if result.get("success"):
                st.success(f"✅  Dataset uploaded! {result['summary']['inserted']} new candidates added to the pipeline.")
                s = result["summary"]
                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("Total Rows",   s["total_rows"])
                r2.metric("Inserted",     s["inserted"])
                r3.metric("Duplicates",   s["duplicates"])
                r4.metric("Valid",        s["valid_candidates"])
                r5.metric("Invalid",     s["invalid_rows"])
                if result.get("errors"):
                    with st.expander("⚠️  Errors"):
                        for err in result["errors"][:20]:
                            st.write(err)
                st.info("👉  Move to the next step: **Job Description**")
                clear_cache()
            else:
                st.error(f"❌  Upload failed: {result}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 2 — Job Description
# ─────────────────────────────────────────────────────────────────────────────

elif page == "💼  Job Description":
    st.title("💼  Job Description")
    st.markdown("Create a job posting. The AI uses the job description and required skills to evaluate candidates.")

    tab_create, tab_list = st.tabs(["✏️  Create Job", "📋  Existing Jobs"])

    with tab_create:
        with st.form("create_job", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                title   = st.text_input("Job Title *", placeholder="e.g. Founding AI Engineer")
                company = st.text_input("Company *", placeholder="e.g. Visl AI Labs")
                dept    = st.text_input("Department", placeholder="e.g. Engineering / AI Research")
            with c2:
                req_sk  = st.text_input("Required Skills *", placeholder="Python, Machine Learning, Deep Learning, NLP")
                pref_sk = st.text_input("Preferred Skills", placeholder="PyTorch, GCP, LLMs, Research publications")
                min_cgpa = st.number_input("Minimum CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            description = st.text_area(
                "Job Description *",
                placeholder="We are looking for an AI engineer to build and deploy ML models at scale. Responsibilities include designing neural architectures, fine-tuning LLMs, and integrating AI into our product pipeline...",
                height=120,
            )
            submitted = st.form_submit_button("🚀  Create Job Posting", width="stretch")
            if submitted:
                if not title or not company or not req_sk:
                    st.error("Title, Company, and Required Skills are required.")
                else:
                    r = create_job({
                        "title":           title,
                        "company":         company,
                        "department":      dept or None,
                        "description":     description or None,
                        "required_skills": req_sk,
                        "preferred_skills": pref_sk or None,
                        "minimum_cgpa":    min_cgpa,
                    })
                    if r.status_code == 200:
                        st.success("✅  Job created! Candidates in the pipeline will now be evaluated against this job.")
                        clear_cache()
                    else:
                        st.error(f"❌  Error: {r.text}")

    with tab_list:
        jobs = get_jobs()
        if jobs:
            df_j = pd.DataFrame(jobs)
            df_j["created_at"] = pd.to_datetime(df_j.get("created_at", pd.Series()), errors="coerce")
            df_j = df_j.sort_values("created_at", ascending=False)
            st.dataframe(df_j[["id","title","company","department","required_skills"]], width="stretch", hide_index=True)

            # Delete job
            st.markdown("---")
            job_to_delete = st.selectbox(
                "Select a job to delete:",
                ["(Select a job)"] + [f"ID {j['id']}: {j['title']}" for j in jobs],
                key="delete_job_select",
            )
            if job_to_delete != "(Select a job)":
                job_id = int(job_to_delete.split(":")[0].replace("ID ", "").strip())
                col_confirm, col_btn = st.columns([1, 3])
                with col_confirm:
                    st.write(f"Delete job ID {job_id}?")
                with col_btn:
                    if st.button("🗑️  Delete Job", type="secondary"):
                        try:
                            requests.delete(f"{API_BASE}/jobs/{job_id}", timeout=10)
                            st.success(f"✅  Job {job_id} deleted")
                            clear_cache()
                            st.rerun()
                        except Exception:
                            st.error("❌  Failed to delete job")
        else:
            st.info("No jobs created yet.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 3 — Parse Resumes
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📄  Parse Resumes":
    st.title("📄  Parse Resumes")
    st.markdown("Candidate resumes are downloaded and parsed to extract text for AI evaluation. This step runs automatically when candidates are uploaded.")

    candidates = get_filtered_candidates()
    if not candidates:
        st.info("No candidates yet. Go to **Upload Dataset** first.")
    else:
        df = pd.DataFrame(candidates)
        parsed = df[df["resume_text"].notna()] if "resume_text" in df.columns else df[df["status"] != "Uploaded"]
        st.metric("Candidates with Parsed Resumes", len(parsed))

        # Show candidates still pending resume parsing
        pending = df[df["status"] == "Uploaded"] if "status" in df.columns else pd.DataFrame()
        if not pending.empty:
            st.warning(f"⏳  {len(pending)} candidate(s) still pending resume parsing.")
            st.caption("Resume parsing runs automatically. Ensure resume URLs are valid.")

        st.markdown("---")
        st.markdown("**📋 Resume Parsing Status**")
        status_counts = df["status"].value_counts() if "status" in df.columns else pd.Series()
        for status, count in status_counts.items():
            st.write(f"  • **{status}**: {count} candidate(s)")

        st.markdown("---")
        st.info("✅  Resume parsing is complete. Move to **AI Evaluation** to start candidate scoring.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 4 — AI Evaluation
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🤖  AI Evaluation":
    st.title("🤖  AI Evaluation")
    st.markdown("Candidates are evaluated by **Google Gemini** against the job description, required skills, and evaluation rubric. Each candidate receives scores for technical skills, projects, education, and research.")

    candidates = get_filtered_candidates()
    jobs = get_jobs()

    if not candidates:
        st.info("No candidates yet. Go to **Upload Dataset** first.")
    elif not jobs:
        st.warning("⚠️  No job created yet. Go to **Job Description** and create a job first — AI evaluation requires a job context.")
    else:
        df = pd.DataFrame(candidates)
        evaluated = df[df["final_score"].notna()] if "final_score" in df.columns else pd.DataFrame()
        pending   = df[df["final_score"].isna()] if "final_score" in df.columns else df

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Candidates",   len(df))
        m2.metric("AI Evaluated",     len(evaluated))
        m3.metric("Pending",           len(pending))

        if not evaluated.empty:
            st.markdown("&nbsp;")
            st.markdown("**📊 AI Score Distribution**")
            chart_df = evaluated[["final_score","name"]].copy()
            chart_df["bin"] = pd.cut(
                chart_df["final_score"], bins=[0,25,50,75,90,101],
                labels=["0–25","26–50","51–75","76–90","91–100"], right=False
            ).astype(str)
            bar_df = chart_df["bin"].value_counts().reset_index()
            bar_df.columns = ["bin","count"]
            if not bar_df.empty:
                bar = alt.Chart(bar_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("bin:N", title="Score Range"),
                    y=alt.Y("count:Q", title="Candidates"),
                    color=alt.Color("bin:N", legend=None,
                        scale=alt.Scale(domain=["0–25","26–50","51–75","76–90","91–100"],
                            range=["#f85149","#d29922","#58a6ff","#3fb950","#1a7f37"])),
                    tooltip=["bin","count"]
                ).properties(height=200)
                st.altair_chart(bar, width="stretch")

        st.markdown("&nbsp;")
        st.markdown("**🤖 Top Evaluated Candidates**")
        top_eval = df.dropna(subset=["final_score"]).sort_values("final_score", ascending=False).head(15)
        if not top_eval.empty:
            cols = ["id","name","email","college","final_score","status"]
            avail = [c for c in cols if c in top_eval.columns]
            disp = top_eval[avail].copy()
            disp.columns = ["ID","Name","Email","College","AI Score","Status"]
            st.dataframe(disp, width="stretch", hide_index=True)

        if not pending.empty:
            st.warning(f"⏳  {len(pending)} candidate(s) still pending AI evaluation.")

        st.markdown("---")
        st.info("✅  Move to **GitHub Analysis** next.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 5 — GitHub Analysis
# ─────────────────────────────────────────────────────────────────────────────

elif page == "💻  GitHub Analysis":
    st.title("💻  GitHub Analysis")
    st.markdown("Each candidate's GitHub profile is analyzed for AI/ML projects, top programming languages, and technical contributions.")

    candidates = get_filtered_candidates()
    if not candidates:
        st.info("No candidates yet. Go to **Upload Dataset** first.")
    else:
        df = pd.DataFrame(candidates)
        github_done = df[df["github_score"].notna()] if "github_score" in df.columns else pd.DataFrame()
        pending = df[df["github_score"].isna()] if "github_score" in df.columns else df

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Candidates",  len(df))
        m2.metric("GitHub Analyzed", len(github_done))
        m3.metric("Pending",         len(pending))

        if not github_done.empty:
            st.markdown("&nbsp;")
            st.markdown("**💻 GitHub Score Distribution**")
            g_df = github_done[["github_score","name"]].copy()
            g_df["bucket"] = pd.cut(g_df["github_score"], bins=[0,20,40,60,80,101],
                labels=["0–20","21–40","41–60","61–80","81–100"], right=False).astype(str)
            bar_df = g_df["bucket"].value_counts().reset_index()
            bar_df.columns = ["bucket","count"]
            if not bar_df.empty:
                gb = alt.Chart(bar_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("bucket:N", title="GitHub Score Range"),
                    y=alt.Y("count:Q", title="Candidates"),
                    color=alt.Color("bucket:N", legend=None,
                        scale=alt.Scale(range=["#f85149","#d29922","#58a6ff","#3fb950","#1a7f37"])),
                    tooltip=["bucket","count"]
                ).properties(height=200)
                st.altair_chart(gb, width="stretch")

        st.markdown("&nbsp;")
        st.markdown("**🏆 Top GitHub Scores**")
        top_gh = df.dropna(subset=["github_score"]).sort_values("github_score", ascending=False).head(15)
        if not top_gh.empty:
            cols = ["id","name","github_url","github_score","top_languages","best_ai_project"]
            avail = [c for c in cols if c in top_gh.columns]
            disp = top_gh[avail].copy()
            disp.columns = ["ID","Name","GitHub","GitHub Score","Top Languages","Best Project"]
            st.dataframe(disp, width="stretch", hide_index=True)

        st.markdown("---")
        st.info("✅  Move to **Score & Rank** to compute combined rankings.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 6 — Score & Rank
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🏆  Score & Rank":
    st.title("🏆  Score & Rank Candidates")
    st.markdown("Candidates are ranked using a weighted combination of AI evaluation (60%), GitHub analysis (25%), CGPA (10%), and research work (5%).")

    # Auto-rank on page load (once per session)
    if not st.session_state.get("ranked_done"):
        result = rank_all()
        st.session_state["ranked_done"] = True
        if result.get("success"):
            clear_cache()

    if st.button("🔄  Re-rank Now", type="secondary"):
        result = rank_all()
        if result.get("success"):
            st.success(f"✅  Re-ranked {result.get('ranked', 0)} candidates!")
            clear_cache()
        else:
            st.error("❌  Ranking failed")

    candidates = get_filtered_candidates()

    candidates = get_filtered_candidates()
    if not candidates:
        st.info("No candidates yet. Go to **Upload Dataset** first.")
    else:
        df = pd.DataFrame(candidates)
        ranked = df.dropna(subset=["candidate_rank"]).sort_values("candidate_rank")
        unranked = df[df["candidate_rank"].isna()] if "candidate_rank" in df.columns else pd.DataFrame()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total",   len(df))
        m1.metric("Ranked",  len(ranked))
        m2.metric("Avg AI Score", f"{df['final_score'].mean():.1f}" if "final_score" in df.columns else "—")
        m3.metric("Avg GitHub",   f"{df['github_score'].mean():.1f}" if "github_score" in df.columns else "—")
        m4.metric("Pending",      len(unranked))

        st.markdown("&nbsp;")
        st.markdown("**🏆 Full Rankings**")
        if not ranked.empty:
            cols = ["id","name","email","college","branch","final_score","github_score","candidate_rank","test_la","test_code"]
            avail = [c for c in cols if c in ranked.columns]
            disp = ranked[avail].copy()
            disp.columns = ["ID","Name","Email","College","Branch","AI Score","GitHub Score","Rank","Test LA","Test Code"]
            disp["Rank"] = disp["Rank"].astype(int)
            st.dataframe(disp, width="stretch", hide_index=True)
        else:
            st.info("No rankings yet. Complete AI Evaluation and GitHub Analysis first.")

        st.markdown("---")
        st.info("✅  Move to **Send Test Links** to start shortlisting candidates by AI score.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 7 — Send Test Links
# ─────────────────────────────────────────────────────────────────────────────

elif page == "✉️  Send Test Links":
    st.title("✉️  Send Test Links")
    st.markdown("Shortlist candidates based on their AI evaluation score. Those meeting the threshold receive an **automated email** with a link to complete the assessment test.")

    threshold = st.slider("Minimum AI Score to Shortlist", 0.0, 100.0, 50.0, step=1.0)
    st.session_state["test_threshold"] = threshold
    shortlisted = get_shortlisted(threshold)

    if shortlisted:
        st.success(f"**{len(shortlisted)} candidates** meet the {threshold}+ AI score threshold")
        df_s = pd.DataFrame(shortlisted)
        cols = ["id","name","email","college","final_score","candidate_rank"]
        avail = [c for c in cols if c in df_s.columns]
        sub = df_s[avail].copy()
        sub.columns = ["ID","Name","Email","College","AI Score","Rank"]
        st.dataframe(sub, width="stretch", hide_index=True)
    else:
        st.warning(f"No candidates meet the {threshold}+ AI score threshold.")

    st.markdown("---")
    st.markdown("""
    **📧 Email Dispatch**
    Clicking **Send Test Links** will email all shortlisted candidates with a unique assessment link.
    Requires SMTP credentials configured in your `.env` file.
    """)
    if st.button("✉️  Send Test Links to All Shortlisted", type="primary", width="stretch"):
        with st.spinner("Sending emails to shortlisted candidates..."):
            result = send_test_links(threshold=st.session_state.get("test_threshold", 50.0))
        if result.get("success"):
            st.success(f"✅  Emails sent to {result['shortlisted_count']} candidates!")
            results_df = pd.DataFrame(result.get("results", []))
            if not results_df.empty:
                st.dataframe(results_df[["name","email","test_link_sent"]], width="stretch", hide_index=True)
            clear_cache()
        else:
            st.error(f"❌  Error: {result}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 8 — Upload Test Results
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊  Upload Test Results":
    st.title("📊  Upload Test Results")
    st.markdown("Upload the test results CSV after candidates complete their assessments. Required columns: `Email`, `test_la` (Logical Aptitude), `test_code` (Coding Test).")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**📋 CSV Format**")
        sample = pd.DataFrame([
            {"Email":"priya@example.com","test_la":"85","test_code":"78"},
            {"Email":"rahul@example.com","test_la":"72","test_code":"90"},
            {"Email":"anita@example.com","test_la":"60","test_code":"65"},
        ])
        st.dataframe(sample, hide_index=True, height=130, width="stretch")
    with c2:
        st.markdown("""
        **ℹ️  Notes**
        - Scores should be **0–100** for each test
        - Matching is by **email** (case-insensitive)
        - Unknown emails are skipped
        - After upload, move to **Shortlist** to rank by test performance
        """)
        st.markdown("---")
        st.markdown("**⚠️  Column names must be exactly:** `Email`, `test_la`, `test_code`")

    st.markdown("---")
    uploaded = st.file_uploader("Drop test results CSV", type=["csv"])

    if uploaded and st.button("🚀  Upload Results & Update Rankings", type="primary", disabled=st.session_state.get("test_upload_done", False)):
        if not st.session_state.get("test_upload_done"):
            st.session_state["test_upload_done"] = True
            result = post_test_results(uploaded.getvalue(), uploaded.name)
            if result.get("success"):
                st.success("✅  Test results uploaded!")
                st.session_state["test_upload_done"] = True
            else:
                st.session_state["test_upload_done"] = False
                st.error("❌  Upload failed")
            s = result["summary"]
            r1, r2, r3 = st.columns(3)
            r1.metric("Total Rows",  s["total_rows"])
            r2.metric("Updated",     s["updated"])
            r3.metric("Not Found",  s["not_found"])
            if result.get("errors"):
                with st.expander("⚠️  Errors"):
                    for err in result["errors"][:20]:
                        st.write(err)
            clear_cache()
        else:
            st.error(f"❌  Failed: {result}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 9 — Shortlist (by Test Performance)
# ─────────────────────────────────────────────────────────────────────────────

elif page == "✅  Shortlist":
    st.title("✅  Shortlist Based on Test Performance")
    st.markdown("Only candidates who received test links and have completed their test are considered. Combined score = Logical Aptitude + Coding Test.")

    min_combined = st.slider(
        "Minimum combined score (test_la + test_code)",
        0.0, 200.0, 80.0, step=5.0
    )
    shortlisted = get_shortlisted_after_test(min_combined)

    if not shortlisted:
        st.warning(f"No candidates have a combined score ≥ {min_combined} yet. Make sure to: 1) Send test links from the previous step, 2) Upload test results.")
    else:
        st.success(f"**{len(shortlisted)} candidates** meet the {min_combined}+ combined score threshold")

        cols = ["id","name","email","college","branch","test_la","test_code","total_test_score"]
        avail = [c for c in cols if c in shortlisted[0].keys()]
        disp = pd.DataFrame(shortlisted)[avail].copy()
        disp.columns = ["ID","Name","Email","College","Branch","Test LA","Test Code","Combined"]
        st.dataframe(disp, width="stretch", hide_index=True)

        # Score breakdown chart
        st.markdown("&nbsp;")
        st.markdown("**📊 Combined Score Breakdown**")
        chart_df = pd.DataFrame(shortlisted[:15])[["name","test_la","test_code","total_test_score"]].copy()
        chart_df.columns = ["Name","Logical Aptitude","Coding Test","Combined"]
        melted = chart_df.melt(id_vars="Name", var_name="Component", value_name="Score")
        bc = alt.Chart(melted).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Name:N", title=None),
            y=alt.Y("Score:Q", title="Score"),
            color=alt.Color("Component:N", title="Test Component"),
            tooltip=["Name","Component","Score"]
        ).properties(height=260)
        st.altair_chart(bc, width="stretch")

    st.markdown("---")
    st.info("✅  Move to **Schedule Interviews** to auto-schedule Google Meet interviews.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 10 — Schedule Interviews
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📅  Schedule Interviews":
    st.title("📅  Schedule Interviews Automatically")
    st.markdown("Automatically schedule Google Meet interviews for shortlisted candidates.")

    # Google OAuth status check
    try:
        resp = requests.get(f"{API_BASE}/google/oauth/login", timeout=5).json()
        auth_url = resp.get("auth_url", "")
        if "your-client-id" in auth_url:
            st.warning("⚠️  **Google OAuth not configured.** Meet links are demo IDs. To get real Meet links: set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in `.env`. See setup instructions below.")
    except Exception:
        pass

    c1, c2 = st.columns(2)
    with c1:
        min_combined = st.number_input(
            "Minimum combined test score (test_la + test_code)",
            min_value=0.0, max_value=200.0, value=100.0, step=5.0,
        )
    with c2:
        duration = st.selectbox("Interview duration", [30, 45, 60, 90, 120], index=2)

    use_custom = st.checkbox("Set custom interview start time")
    start_str = None
    if use_custom:
        default_dt = datetime.now(pytz.UTC).replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=1)
        start_str = st.text_input(
            "Start time (ISO UTC — e.g. 2026-07-01T10:00:00Z)",
            value=default_dt.isoformat().replace("+00:00", "Z"),
        )

    cand_ids = st.text_input(
        "Candidate IDs to schedule (optional, blank = all eligible)",
        placeholder="1, 3, 5",
    )

    # Preview eligible
    all_c = get_candidates()
    if all_c:
        df_c = pd.DataFrame(all_c)
        eligible = df_c[df_c["test_la"].notna() & df_c["test_code"].notna()].copy()
        eligible["total"] = eligible["test_la"] + eligible["test_code"]
        eligible = eligible[eligible["total"] >= min_combined]
        st.info(f"**{len(eligible)} candidates** are eligible (test_la + test_code ≥ {min_combined})")

    st.markdown("---")
    st.markdown("""
    **🎥 What happens when you click Schedule:**
    1. A Google Meet link is automatically generated via Google Calendar API
    2. A calendar event is created for each candidate
    3. An email invitation is sent with the Meet link and interview time
    """)
    if st.button("📅  Schedule Interviews & Send Invites", type="primary", width="stretch"):
        kwargs = {"min_test_score": min_combined, "interview_duration_minutes": duration}
        if start_str:
            kwargs["start_time_str"] = start_str
        if cand_ids:
            kwargs["candidate_ids"] = cand_ids

        with st.spinner("Creating Google Meet events and sending email invites..."):
            result = schedule_interviews(**kwargs)

        if result.get("success"):
            st.success(f"✅  **{result['scheduled_count']} interviews scheduled**, **{result['failed_count']} failed**")
            for r in result.get("results", []):
                icon = "🟢" if r.get("status") == "Scheduled" else "🔴"
                with st.expander(f"{icon}  {r.get('name')} — {r.get('email')}"):
                    st.write(f"**Time:**      {r.get('interview_time', 'N/A')}")
                    st.write(f"**Meet Link:** {r.get('meet_link', 'N/A')}")
                    st.write(f"**Email Sent:** {'✅' if r.get('email_sent') else '❌'}")
                    if r.get("error"):
                        st.error(f"Error: {r['error']}")
            clear_cache()
        else:
            st.error(f"❌  Error: {result}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 11 — Interview Status (Google Meet Sent)
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🎥  Interview Status":
    st.title("🎥  Interview Status")
    st.markdown("Track all scheduled interviews, their Google Meet links, and candidate attendance status.")

    interviews = get_interviews()
    candidates = get_filtered_candidates()

    if interviews:
        st.dataframe(pd.DataFrame(interviews), width="stretch", hide_index=True)
    else:
        st.info("No interviews scheduled yet.")

    st.markdown("---")
    st.markdown("**📊 Pipeline Summary**")

    if candidates:
        df_c = pd.DataFrame(candidates)
        total  = len(df_c)
        scored = int(df_c["final_score"].notna().sum()) if "final_score" in df_c.columns else 0
        ranked = int(df_c["candidate_rank"].notna().sum()) if "candidate_rank" in df_c.columns else 0
        tested = int(df_c["test_la"].notna().sum()) if "test_la" in df_c.columns else 0
        int_sc = int((df_c["status"] == "Interview Scheduled").sum()) if "status" in df_c.columns else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total",      total)
        m2.metric("AI Scored", scored)
        m3.metric("Ranked",    ranked)
        m4.metric("Tested",    tested)
        m5.metric("Interviews", int_sc)

        if total > 0:
            st.markdown("&nbsp;")
            st.markdown("**📈 Pipeline Funnel**")
            prog = pd.DataFrame({
                "Stage":  ["Uploaded","AI Scored","Ranked","Tested","Interview"],
                "Count":  [total, scored, ranked, tested, int_sc],
            })
            bar = alt.Chart(prog).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=35).encode(
                x=alt.X("Stage:N", title=None, axis=alt.Axis(labelAngle=-20)),
                y=alt.Y("Count:Q", title="Candidates"),
                color=alt.Color("Stage:N", legend=None),
                tooltip=["Stage","Count"]
            ).properties(height=250)
            st.altair_chart(bar, width="stretch")

    st.markdown("---")
    st.markdown("### 🎥  Google Meet — Interview Invitation Email Preview")
    st.info("""
    Each scheduled candidate receives an email with:
    - 📅 **Date & Time** of the interview
    - ⏱ **Duration**
    - 🔗 **Google Meet Link** (auto-generated)
    - 📝 Brief description of the role
    """)

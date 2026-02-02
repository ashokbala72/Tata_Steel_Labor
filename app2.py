import streamlit as st
import pandas as pd
import os
import requests
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import re
from io import StringIO

def extract_pipe_table(ai_text: str) -> pd.DataFrame | None:
    lines = [
        line for line in ai_text.splitlines()
        if line.strip().startswith("|")
    ]

    if not lines:
        return None

    df = pd.read_csv(
        StringIO("\n".join(lines)),
        sep="|",
        engine="python"
    )

    df = df.dropna(axis=1, how="all")
    df.columns = [c.strip() for c in df.columns]
    df = df[~df.iloc[:, 0].astype(str).str.contains("SECTION", na=False)]
    df = df.reset_index(drop=True)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return df


# =================================================
# ENV
# =================================================
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# =================================================
# AZURE OPENAI CALL
# =================================================
def call_genai(prompt: str) -> str:
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json"
    }

    body = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a procurement approval intelligence assistant for Tata Steel. "
                    "Explain pricing changes, recommend approval paths, "
                    "and generate audit-ready justifications."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1500
    }

    r = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
        f"{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}",
        headers=headers,
        json=body,
        timeout=30
    )

    if r.status_code != 200:
        return "AI service unavailable."

    return r.json()["choices"][0]["message"]["content"]

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Tata Steel Workflow-Based Approval with Pricing Transparency Dashboard",
    layout="wide"
)

# =================================================
# STYLES
# =================================================
st.markdown("""
<style>
.stApp { background-color: #eef1f7; }
.center { text-align: center; }

.ai-output {
    font-size: 16px;
    font-weight: 500;
    line-height: 1.8;
    color: #111827;
    white-space: pre-wrap;
}

.textarea {
    font-size: 10px !important;
    font-weight: 500 !important;
    color: #111827 !important;
    line-height: 1.6 !important;
}

.ai-summary {
    font-size: 8px;
    line-height: 1.3;
    color: #1f2937;
    white-space: pre-wrap;
}

.section-box {
    background: #f8f9fc;
    padding: 16px;
    border-radius: 14px;
    margin-bottom: 16px;
}

.metric {
    padding: 14px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-weight: bold;
}

.blue { background: linear-gradient(135deg,#1f77b4,#4fa3e3); }
.green { background: linear-gradient(135deg,#2ca02c,#6fdc8c); }
.orange { background: linear-gradient(135deg,#ff7f0e,#ffb347); }
.purple { background: linear-gradient(135deg,#7b61c9,#a88beb); }
</style>
""", unsafe_allow_html=True)

# =================================================
# TITLE
# =================================================
st.markdown(
    "<h1 class='center'>Tata Steel Workflow-Based Approval with Pricing Transparency Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 class='center'>Powered by TCS Gen AI and Azure Open AI</h4>",
    unsafe_allow_html=True
)
st.markdown("---")

# =================================================
# FILE UPLOADS
# =================================================
u1, u2, u3, u4, u5, u6 = st.columns(6)

with u1:
    quotes_file = st.file_uploader("quotes.csv", type="csv")
with u2:
    approval_history_file = st.file_uploader("approval_history.csv", type="csv")
with u3:
    historical_benchmarks_file = st.file_uploader("historical_benchmarks.csv", type="csv")
with u4:
    approval_outcomes_file = st.file_uploader("approval_outcomes.csv", type="csv")
with u5:
    approval_rules_file = st.file_uploader("approval_rules.csv", type="csv")
with u6:
    audit_trail_file = st.file_uploader("audit_trail.csv", type="csv")

if not all([quotes_file, approval_history_file, historical_benchmarks_file, approval_rules_file]):
    st.info("Upload required files to continue.")
    st.stop()

# =================================================
# LOAD DATA
# =================================================
quotes = pd.read_csv(quotes_file)
approval_history = pd.read_csv(approval_history_file)
benchmarks = pd.read_csv(historical_benchmarks_file)
approval_rules = pd.read_csv(approval_rules_file)
approval_rules.columns = approval_rules.columns.str.strip()

# =================================================
# KEY METRICS (GBP)
# =================================================
avg_initial = quotes["initial_quoted_rate"].mean()
avg_change = quotes["change_percentage"].mean()

approval_success = 0.0
if approval_outcomes_file:
    try:
        outcomes = pd.read_csv(approval_outcomes_file)
        if not outcomes.empty:
            approval_success = (
                outcomes[outcomes["approval_status"].astype(str).str.upper() == "APPROVED"]
                .shape[0] / outcomes.shape[0]
            ) * 100
    except pd.errors.EmptyDataError:
        approval_success = 0.0

m1, m2, m3, m4 = st.columns(4)

m1.markdown(f"<div class='metric blue'>£ {avg_initial:,.0f}<br/>Avg Initial Quote</div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric orange'>{avg_change:.1f}%<br/>Avg Change</div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric green'>{approval_success:.0f}%<br/>Approval Success</div>", unsafe_allow_html=True)
m4.markdown(f"<div class='metric purple'>{len(quotes)}<br/>Quotes</div>", unsafe_allow_html=True)

# =================================================
# MAIN LAYOUT
# =================================================
left, right = st.columns([3, 2])

# =================================================
# LEFT — WORKFLOW
# =================================================
with left:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("### Quote Overview")

    st.dataframe(
        quotes[
            [
                "quote_id",
                "vendor_id",
                "initial_quoted_rate",
                "revised_quoted_rate",
                "change_percentage",
                "approval_status",
                "approval_path"
            ]
        ],
        use_container_width=True,
        height=220
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("### Approval History")

    st.dataframe(
        approval_history[
            [
                "quote_id",
                "decision_made",
                "approver_role",
                "approval_date",
                "reason_for_decision"
            ]
        ],
        use_container_width=True,
        height=200
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("### Benchmark Comparison")

    merged = quotes.merge(benchmarks, on="work_id", how="left")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        merged["work_id"],
        merged["revised_quoted_rate"],
        label="Revised Quote",
        color="#4fa3e3"
    )
    ax.plot(
        merged["work_id"],
        merged["avg_rate"],
        label="Historical Benchmark",
        color="red",
        linewidth=2
    )
    ax.set_ylabel("£ / hr")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# =================================================
# RIGHT — AI JUSTIFICATION
# =================================================
with right:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("### AI-Generated Justification & Recommendations")

    prompt = f"""
You are a procurement approval intelligence assistant for Tata Steel.

INPUT DATA:
Quotes:
{quotes[['quote_id','initial_quoted_rate','revised_quoted_rate','change_percentage','reason_for_change']].to_string(index=False)}

Approval History:
{approval_history[['quote_id','decision_made','approver_role','reason_for_decision']].to_string(index=False)}

Decision Rules:
{approval_rules[['change_threshold','action','description']].to_string(index=False)}

OBJECTIVES:
1. Explain pricing changes clearly and factually
2. Recommend the appropriate approval path
3. Provide an audit-ready justification

OUTPUT STRUCTURE (MANDATORY):

SECTION: Summary
Provide a compact factual summary (max 6 bullet points).

SECTION: Recommendation Table
Provide a compact pipe table with columns:
| Quote ID | Price Change (%) | Key Reason | Recommended Approval |

SECTION: Audit Justification
Provide a short audit-ready paragraph.

FORMATTING RULES (STRICT — MUST FOLLOW):
- Do NOT insert blank lines between text and tables
- Use at most ONE newline unless separating sections
- Tables must start immediately after headers or preceding sentences
- Do NOT add empty lines before or after tables
- Do NOT add decorative spacing or visual padding
- Keep output dense, compact, and audit-style
- Do NOT repeat input data verbatim
- Do NOT add headings other than the specified SECTION headers

TONE & STYLE:
- Professional, formal, and objective
- No conversational language
- No emojis
- No markdown styling beyond plain text and pipe tables
- Suitable for senior management and auditors

IMPORTANT:
Any violation of formatting rules is considered an error.
"""

    with st.spinner("Generating approval intelligence..."):
        if "ai_output" not in st.session_state:
            st.session_state.ai_output = call_genai(prompt)

    ai_output = st.session_state.ai_output

    summary_part = ai_output.split("SECTION: Recommendation Table")[0]
    st.markdown(
        f"<div class='ai-output'>{summary_part}</div>",
        unsafe_allow_html=True
    )

    df = extract_pipe_table(ai_output)
    if df is not None:
        st.markdown("#### Recommendation Table")
        st.table(df)

    if "SECTION: Audit Justification" in ai_output:
        audit_part = ai_output.split("SECTION: Audit Justification")[1]
        st.markdown(
            f"<div class='ai-output'><b>Audit Justification</b><br>{audit_part}</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

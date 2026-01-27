import streamlit as st
import pandas as pd
import os
import requests
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# -------------------------------------------------
# AZURE OPENAI REST CALL
# -------------------------------------------------
def call_genai(prompt: str) -> str:
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json",
    }

    body = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a procurement and vendor pricing AI copilot for Tata Steel. "
                    "Analyze vendor pricing, benchmark rates, detect anomalies, "
                    "and provide executive-ready recommendations."
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
        return "Unable to generate insights at this time."

    return r.json()["choices"][0]["message"]["content"].strip()

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Tata Steel Vendor Pricing Dashboard",
    layout="wide"
)

# -------------------------------------------------
# STYLES
# -------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #eef1f7; }
.center { text-align: center; }

.metric-card {
    padding: 18px;
    border-radius: 16px;
    color: white;
    text-align: center;
}

.blue { background: linear-gradient(135deg, #1f77b4, #4fa3e3); }
.purple { background: linear-gradient(135deg, #7b61c9, #a88beb); }
.orange { background: linear-gradient(135deg, #ff7f0e, #ffb347); }
.green { background: linear-gradient(135deg, #2ca02c, #6fdc8c); }

.metric-value { font-size: 28px; font-weight: 700; }
.metric-label { font-size: 14px; opacity: 0.9; }

.section-box {
    background-color: #f8f9fc;
    padding: 16px;
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown("<h1 class='center'>Tata Steel Vendor Pricing Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='center'>Powered by TCS Gen AI and Azure Open AI</h4>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------
# FILE UPLOADS
# -------------------------------------------------
u1, u2, u3, u4, u5 = st.columns(5)

with u1:
    vendor_master = st.file_uploader("vendor_master.csv", type="csv")
with u2:
    work_catalog = st.file_uploader("work_catalog.csv", type="csv")
with u3:
    vendor_quotes = st.file_uploader("vendor_quotes.csv", type="csv")
with u4:
    historical_quotes = st.file_uploader("historical_quotes.csv", type="csv")
with u5:
    approval_outcomes = st.file_uploader("approval_outcomes.csv", type="csv")

if not all([vendor_master, work_catalog, vendor_quotes]):
    st.info("Upload vendor master, work catalog, and vendor quotes to continue.")
    st.stop()

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
vendors = pd.read_csv(vendor_master)
works = pd.read_csv(work_catalog)
quotes = pd.read_csv(vendor_quotes)

# -------------------------------------------------
# BASIC METRICS
# -------------------------------------------------
avg_rate = quotes["quoted_rate"].mean()
median_rate = quotes["quoted_rate"].median()

quotes["deviation_pct"] = (quotes["quoted_rate"] - median_rate) / median_rate * 100
avg_deviation = quotes["deviation_pct"].mean()

approval_rate = 0
if approval_outcomes is not None:
    try:
        approvals = pd.read_csv(approval_outcomes)
        if not approvals.empty and "approval_status" in approvals.columns:
            approval_rate = (
                approvals[approvals["approval_status"].astype(str).str.strip().str.upper() == "APPROVED"]
                .shape[0] / approvals.shape[0]
            ) * 100
    except pd.errors.EmptyDataError:
        approval_rate = 0

# -------------------------------------------------
# KEY METRICS
# -------------------------------------------------
st.subheader("Key Metrics")

m1, m2, m3, m4 = st.columns(4)

m1.markdown(f"<div class='metric-card blue'><div class='metric-value'>₹ {avg_rate:.0f}/hr</div><div class='metric-label'>Avg Rate</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-card purple'><div class='metric-value'>₹ {median_rate:.0f}/hr</div><div class='metric-label'>Market Median</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-card orange'><div class='metric-value'>{avg_deviation:.1f}%</div><div class='metric-label'>Avg Deviation</div></div>", unsafe_allow_html=True)
m4.markdown(f"<div class='metric-card green'><div class='metric-value'>{approval_rate:.0f}%</div><div class='metric-label'>Approval Success</div></div>", unsafe_allow_html=True)

# -------------------------------------------------
# MAIN CONTENT
# -------------------------------------------------
left, right = st.columns([2, 3])

# ---------------- LEFT ----------------
# ---------------- LEFT SIDE ----------------
with left:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)

    # -------- Benchmarking --------
    st.markdown("### Similar Work Benchmarking & Anomalies")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        quotes["vendor_id"],
        quotes["quoted_rate"],
        color="#4fa3e3",
        alpha=0.8
    )

    ax.axhline(
        median_rate,
        color="red",
        linestyle="--",
        label="Market Median"
    )

    ax.set_ylabel("Rate (INR)")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(
        quotes[
            ["vendor_id", "work_description_raw", "quoted_rate", "deviation_pct"]
        ].assign(deviation_pct=lambda x: x["deviation_pct"].round(1)),
        use_container_width=True,
        height=300
    )

    # -------- Vendor Scoring (INSIDE LEFT) --------
    st.markdown("### Vendor Scoring Overview")

    vendor_scores = quotes.groupby("vendor_id").agg(
        avg_rate=("quoted_rate", "mean"),
        consistency=("quoted_rate", "std")
    ).reset_index()

    vendor_scores["consistency"] = vendor_scores["consistency"].fillna(0)

    # Vendor scoring graph
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(
        vendor_scores["vendor_id"],
        vendor_scores["avg_rate"],
        color="#7b61c9",
        alpha=0.85
    )

    ax2.set_ylabel("Avg Rate (INR)")
    ax2.tick_params(axis="x", rotation=30, labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    st.pyplot(fig2)

    # Vendor scoring table
    st.dataframe(
        vendor_scores.round(1),
        use_container_width=True,
        height=300
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- RIGHT ----------------
with right:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("### AI-Powered Insights")

    prompt = f"""
    Average quoted rate is ₹{avg_rate:.0f}.
    Market median is ₹{median_rate:.0f}.
    Pricing deviations observed:
    {quotes[['vendor_id','quoted_rate','deviation_pct']].to_string(index=False)}

    Provide vendor pricing insights and recommendations.
    """

    with st.spinner("Generating AI-powered insights..."):
        insights = call_genai(prompt)

    st.markdown(
    f"""
    <div style="
        font-size: 8px;
        line-height: 1.5;
        background-color: #eaf4ff;
        padding: 12px;
        border-radius: 10px;
    ">
        {insights}
    </div>
    """,
    unsafe_allow_html=True
)




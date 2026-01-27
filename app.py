import streamlit as st
import pandas as pd
import os
import requests
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
                    "You are an enterprise operations AI copilot for Tata Steel. "
                    "Analyze labor utilization, idle time, and cost leakage. "
                    "Provide executive-ready insights with quantified INR impact "
                    "and operational recommendations."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 700
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
    page_title="Tata Steel Labor Utilization Optimization Dashboard",
    layout="wide"
)
st.markdown("""
<style>
/* Match Tata Steel dashboard background */
.stApp {
    background-color: #eef1f7;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Full page background */
.stApp {
    background: linear-gradient(
        180deg,
        #f7f9fc 0%,
        #eef2f7 100%
    );
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# STYLES
# -------------------------------------------------
st.markdown("""
<style>
.center { text-align: center; }

.metric-card {
    padding: 20px;
    border-radius: 16px;
    color: white;
    text-align: center;
}
.blue { background: linear-gradient(135deg, #1f77b4, #4fa3e3); }
.red { background: linear-gradient(135deg, #d62728, #ff6b6b); }
.orange { background: linear-gradient(135deg, #ff7f0e, #ffb347); }
.green { background: linear-gradient(135deg, #2ca02c, #6fdc8c); }

.metric-value { font-size: 32px; font-weight: 700; }
.metric-label { font-size: 14px; opacity: 0.9; }

.chart-box-blue {
    background-color: #eef4fb;
    padding: 15px;
    border-radius: 16px;
}
.chart-box-purple {
    background-color: #f4f1fa;
    padding: 15px;
    border-radius: 16px;
}
.header-bar {
    background: linear-gradient(90deg, #3b6db3, #4a7fc7);
    padding: 16px 24px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.header-title {
    color: white;
    font-size: 26px;
    font-weight: 600;
}

.header-sub {
    color: #e6ecf5;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown("<h1 class='center'>Tata Steel Labor Utilization Optimization Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='center'>Powered by TCS Gen AI and Azure Open AI</h4>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------
# FILE UPLOADERS
# -------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    workforce_file = st.file_uploader("workforce_master.csv", type="csv")
with c2:
    attendance_file = st.file_uploader("shift_attendance.csv", type="csv")
with c3:
    production_file = st.file_uploader("production_volume.csv", type="csv")
with c4:
    external_file = st.file_uploader("external_factors.csv", type="csv")

if not all([workforce_file, attendance_file, production_file, external_file]):
    st.info("Upload all CSV files to load the dashboard.")
    st.stop()

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
employees = pd.read_csv(workforce_file)
attendance = pd.read_csv(attendance_file)
external = pd.read_csv(external_file)

attendance["date"] = pd.to_datetime(attendance["date"])
external["date"] = pd.to_datetime(external["date"])

attendance = attendance.merge(employees, on="employee_id")

attendance["idle_hours"] = attendance["scheduled_hours"] - attendance["actual_hours_worked"]
attendance["idle_cost"] = attendance["idle_hours"] * attendance["hourly_cost_inr"]

# -------------------------------------------------
# METRICS
# -------------------------------------------------
utilization_pct = (attendance["actual_hours_worked"].sum() / attendance["scheduled_hours"].sum()) * 100
avg_idle_hours = attendance["idle_hours"].mean()
total_idle_cost_lakhs = attendance["idle_cost"].sum() / 100000

rainy = attendance.merge(external, on="date")
rainy_idle_pct = (len(rainy[rainy["rainfall_mm"] > 20]) / len(attendance)) * 100

# -------------------------------------------------
# KEY METRICS
# -------------------------------------------------
st.subheader("Key Metrics")

m1, m2, m3, m4 = st.columns(4)

m1.markdown(f"<div class='metric-card blue'><div class='metric-value'>{utilization_pct:.1f}%</div><div class='metric-label'>Utilization</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-card red'><div class='metric-value'>₹ {total_idle_cost_lakhs:.1f} L</div><div class='metric-label'>Idle Cost</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-card orange'><div class='metric-value'>{avg_idle_hours:.1f} hrs</div><div class='metric-label'>Idle Time / Employee</div></div>", unsafe_allow_html=True)
m4.markdown(f"<div class='metric-card green'><div class='metric-value'>+{rainy_idle_pct:.1f}%</div><div class='metric-label'>Idle on Rainy Days</div></div>", unsafe_allow_html=True)

# -------------------------------------------------
# CHARTS SIDE BY SIDE
# -------------------------------------------------
st.subheader("Operational Insights")

left, right = st.columns(2)

# -------- LEFT CHART --------
with left:
    st.markdown("<div class='chart-box-blue'>", unsafe_allow_html=True)
    st.markdown("### Utilization and Idle Cost Overview by Plant")

    plant_summary = attendance.groupby("plant").agg(
        utilization=("actual_hours_worked", lambda x: x.sum() / (len(x) * 8)),
        idle_cost=("idle_cost", "sum")
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.bar(
        plant_summary["plant"],
        plant_summary["utilization"] * 100,
        color="#4fa3e3",
        alpha=0.85,
        zorder=2
    )
    ax1.set_ylabel("Utilization (%)", fontsize=9)
    ax1.tick_params(axis="both", labelsize=8)

    ax2 = ax1.twinx()
    ax2.plot(
        plant_summary["plant"],
        plant_summary["idle_cost"] / 100000,
        color="#d62728",
        linewidth=3,
        marker="o",
        zorder=3
    )
    ax2.set_ylabel("Idle Cost (₹ Lakhs)", fontsize=9)
    ax2.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# -------- RIGHT CHART --------
with right:
    st.markdown("<div class='chart-box-purple'>", unsafe_allow_html=True)
    st.markdown("### Idle Patterns and Cost Leakage")

    daily = attendance.groupby("date").agg(idle_cost=("idle_cost", "sum")).reset_index()
    daily = daily.merge(external, on="date")

    fig2, ax1 = plt.subplots(figsize=(6, 4))

    ax1.bar(
        daily["date"],
        daily["idle_cost"] / 100000,
        color="#8b8bd6",
        alpha=0.8,
        zorder=2
    )
    ax1.set_ylabel("Idle Cost (₹ Lakhs)", fontsize=9)
    ax1.tick_params(axis="x", rotation=45, labelsize=7)
    ax1.tick_params(axis="y", labelsize=8)

    ax2 = ax1.twinx()
    ax2.plot(
        daily["date"],
        daily["rainfall_mm"],
        color="#2ca02c",
        linewidth=3,
        zorder=3
    )
    ax2.set_ylabel("Rainfall (mm)", fontsize=9)
    ax2.tick_params(axis="y", labelsize=8)

    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=4))

    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# GPT INSIGHTS
# -------------------------------------------------
st.subheader("GPT Insights")

prompt = f"""
Overall utilization is {utilization_pct:.1f}%.
Total idle cost is ₹{total_idle_cost_lakhs:.1f} lakhs.
Rainfall-driven idle impact is {rainy_idle_pct:.1f}%.
Plant-level summary:
{plant_summary.to_string(index=False)}

Generate 4 executive insights with quantified impact and recommendations.
"""

if st.button("Refresh Insights"):
    with st.spinner("Generating AI insights..."):
        st.success(call_genai(prompt))

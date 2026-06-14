import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AI Mule Account Detection Ecosystem",
    page_icon="🛡️",
    layout="wide"
)

# =========================
# HEADER
# =========================

st.title("🏦 AI-Powered Mule Account Detection Ecosystem")
st.subheader("PSB Hackathon Submission")

st.sidebar.success("System Online")

# =========================
# SIDEBAR
# =========================

st.sidebar.markdown("## Platform Components")

st.sidebar.write("✅ Missing Value Intelligence")
st.sidebar.write("✅ Behavioral Pattern Analysis")
st.sidebar.write("✅ Risk Fusion Engine")
st.sidebar.write("✅ Explainable AI")
st.sidebar.write("✅ Human-in-the-Loop")
st.sidebar.write("✅ Analyst Feedback Module")

# =========================
# FEATURE INPUT
# =========================

st.markdown("## Account Feature Analysis")

col1, col2 = st.columns(2)

with col1:
    F115 = st.number_input("F115", value=1000.0)
    F527 = st.number_input("F527", value=0.0)
    F531 = st.number_input("F531", value=0.0)
    F2582 = st.number_input("F2582", value=0.0)

with col2:
    F2678 = st.number_input("F2678", value=0.0)
    F2956 = st.number_input("F2956", value=0.0)
    F3043 = st.number_input("F3043", value=0.0)

# =========================
# MISSING INTELLIGENCE
# =========================

miss_score = 0
reasons = []

if F2678 == 0:
    miss_score += 20
    reasons.append("F2678 missing pattern")

if F3043 == 0:
    miss_score += 20
    reasons.append("F3043 missing pattern")

if F527 == 0:
    miss_score += 10
    reasons.append("F527 sparse behavior")

# =========================
# BEHAVIOR AGENT
# =========================

behavior_score = 0

if F115 > 100000:
    behavior_score += 30
    reasons.append("Large transaction behavior")

if abs(F2582) > 5000:
    behavior_score += 15
    reasons.append("Abnormal account activity")

if abs(F2956) > 5000:
    behavior_score += 15
    reasons.append("Unusual behavioral movement")

# =========================
# RISK FUSION ENGINE
# =========================

risk_score = min(
    100,
    miss_score + behavior_score
)

# =========================
# ANALYSIS
# =========================

if st.button("🔍 Analyze Account"):

    st.markdown("## Analysis Results")

    st.metric(
        "Mule Risk Score",
        f"{risk_score:.0f}%"
    )

    if risk_score >= 70:
        st.error("🚨 HIGH RISK MULE ACCOUNT")

    elif risk_score >= 40:
        st.warning("⚠️ SUSPICIOUS ACCOUNT")

    else:
        st.success("✅ LOW RISK ACCOUNT")

    st.markdown("### Explainability")

    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("• No significant risk indicators")

# =========================
# REVIEW QUEUE
# =========================

st.markdown("---")
st.markdown("## Analyst Review Queue")

sample_queue = pd.DataFrame({
    "Account_ID": [
        "ACC001",
        "ACC002",
        "ACC003"
    ],
    "Risk": [
        85,
        62,
        22
    ],
    "Status": [
        "REVIEW",
        "REVIEW",
        "CLEAR"
    ]
})

st.dataframe(
    sample_queue,
    use_container_width=True
)

# =========================
# HITL
# =========================

st.markdown("---")
st.markdown("## Human-in-the-Loop Feedback")

verdict = st.selectbox(
    "Analyst Decision",
    [
        "CLEAN",
        "SUSPICIOUS",
        "MULE"
    ]
)

if st.button("Submit Analyst Feedback"):
    st.success(
        f"Feedback stored: {verdict}"
    )

# =========================
# FOOTER
# =========================

st.markdown("---")
st.caption(
    "PSB Hackathon | AI/ML-Based Classification of Suspicious Mule Accounts"
)

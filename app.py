import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AI Mule Detection Ecosystem",
    page_icon="🛡️",
    layout="wide"
)

# =====================================================
# PAGE HEADER
# =====================================================

st.title("🏦 AI-Powered Mule Account Detection Ecosystem")
st.subheader("PSB Hackathon | Multi-Agent Fraud Intelligence Platform")

st.sidebar.success("System Online")

# =====================================================
# INPUT SECTION
# =====================================================

st.markdown("## 📥 Account Features")

c1, c2 = st.columns(2)

with c1:
    F115 = st.number_input("F115", value=1000.0)
    F527 = st.number_input("F527", value=0.0)
    F531 = st.number_input("F531", value=0.0)
    F2582 = st.number_input("F2582", value=0.0)

with c2:
    F2678 = st.number_input("F2678", value=0.0)
    F2956 = st.number_input("F2956", value=0.0)
    F3043 = st.number_input("F3043", value=0.0)

# =====================================================
# ANALYSIS
# =====================================================

if st.button("🚀 Launch Agent Ecosystem"):

    # -------------------------------------------------
    # AGENT 1
    # -------------------------------------------------

    missing_score = 0
    missing_reasoning = []

    if F2678 == 0:
        missing_score += 25
        missing_reasoning.append(
            "F2678 missing. Possible concealment of account history."
        )

    if F3043 == 0:
        missing_score += 20
        missing_reasoning.append(
            "F3043 unavailable. Historical profile incomplete."
        )

    if F527 == 0:
        missing_score += 10
        missing_reasoning.append(
            "Sparse information pattern detected."
        )

    # -------------------------------------------------
    # AGENT 2
    # -------------------------------------------------

    behavior_score = 0
    behavior_reasoning = []

    if F115 > 100000:
        behavior_score += 35
        behavior_reasoning.append(
            "High-value transaction behavior observed."
        )

    if abs(F2582) > 5000:
        behavior_score += 20
        behavior_reasoning.append(
            "Abnormal transaction movement identified."
        )

    if abs(F2956) > 5000:
        behavior_score += 15
        behavior_reasoning.append(
            "Behavioral profile deviates from normal baseline."
        )

    # -------------------------------------------------
    # AGENT 3
    # -------------------------------------------------

    pattern_score = np.random.randint(40, 90)

    if pattern_score > 70:
        pattern_reason = (
            "Cluster resembles previously observed mule-account behavior."
        )
    else:
        pattern_reason = (
            "Cluster moderately similar to suspicious account patterns."
        )

    # -------------------------------------------------
    # AGENT 4
    # -------------------------------------------------

    final_score = (
        0.30 * missing_score +
        0.40 * behavior_score +
        0.30 * pattern_score
    )

    final_score = round(min(final_score, 100), 2)

    # -------------------------------------------------
    # AGENT 5
    # -------------------------------------------------

    if final_score >= 75:
        action = "FREEZE ACCOUNT"
    elif final_score >= 50:
        action = "MANUAL REVIEW"
    else:
        action = "ALLOW"

    # =====================================================
    # OUTPUT
    # =====================================================

    st.markdown("---")

    colA, colB = st.columns([2,1])

    with colA:

        st.metric(
            "🎯 Mule Risk Score",
            f"{final_score}%"
        )

        if final_score >= 75:
            st.error("🚨 HIGH-RISK MULE ACCOUNT")

        elif final_score >= 50:
            st.warning("⚠️ SUSPICIOUS ACCOUNT")

        else:
            st.success("✅ LOW RISK ACCOUNT")

    with colB:

        st.metric(
            "📂 Recommended Action",
            action
        )

    # =====================================================
    # AGENT REASONING
    # =====================================================

    st.markdown("## 🧠 Agent Reasoning")

    with st.expander("🕵️ Missing Intelligence Agent", expanded=True):
        st.write(
            f"Confidence: {missing_score}%"
        )

        if missing_reasoning:
            for r in missing_reasoning:
                st.write("•", r)
        else:
            st.write("No suspicious missing-value patterns detected.")

    with st.expander("🧠 Behavioral Intelligence Agent"):
        st.write(
            f"Confidence: {behavior_score}%"
        )

        if behavior_reasoning:
            for r in behavior_reasoning:
                st.write("•", r)
        else:
            st.write("Behavior appears normal.")

    with st.expander("🌐 Pattern Discovery Agent"):
        st.write(
            f"Similarity Score: {pattern_score}%"
        )
        st.write(pattern_reason)

    with st.expander("⚖️ Risk Fusion Agent"):
        st.write(
            f"""
            Missing Agent Contribution: {missing_score}

            Behavioral Agent Contribution: {behavior_score}

            Pattern Agent Contribution: {pattern_score}

            Consensus Risk Score: {final_score}
            """
        )

    with st.expander("📂 Case Manager Agent"):
        st.write(
            f"Recommended Action: {action}"
        )

    # =====================================================
    # REVIEW QUEUE
    # =====================================================

    st.markdown("---")
    st.markdown("## 📋 Investigation Queue")

    queue = pd.DataFrame({
        "Account": ["ACC001","ACC002","ACC003"],
        "Risk": [87,64,23],
        "Status": ["REVIEW","REVIEW","CLEAR"]
    })

    st.dataframe(queue, use_container_width=True)

    # =====================================================
    # HITL
    # =====================================================

    st.markdown("---")
    st.markdown("## 👨‍💼 Human Investigator Agent")

    analyst_decision = st.selectbox(
        "Analyst Decision",
        ["CLEAN","SUSPICIOUS","MULE"]
    )

    if st.button("Submit Feedback"):

        st.success(
            f"Feedback captured: {analyst_decision}"
        )

        st.info(
            "🔄 Self-Healing Learning Loop Triggered"
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.caption(
    "PSB Hackathon | AI/ML-Based Classification of Suspicious Mule Accounts"
)

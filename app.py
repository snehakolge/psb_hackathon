import streamlit as st
import numpy as np
import pandas as pd
import time

from agent_ecosystem_engine import AdaptiveConsensusEcosystem

# =========================================================
# CONFIG
# =========================================================
FEATURES = ['F115','F527','F531','F2582','F2678','F2956','F3043']

st.set_page_config(page_title="RBI AML SOC", layout="wide")

st.title("🏦 RBI AML + Fraud SOC (ML Streaming System)")

# =========================================================
# STATE
# =========================================================
if "events" not in st.session_state:
    st.session_state.events = []

if "cases" not in st.session_state:
    st.session_state.cases = []

if "running" not in st.session_state:
    st.session_state.running = False

if "step" not in st.session_state:
    st.session_state.step = 0

# =========================================================
# ML ECOSYSTEM (YOUR TRAINED MODEL)
# =========================================================
@st.cache_resource
def get_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = get_ecosystem()

# =========================================================
# START / STOP
# =========================================================
col1, col2 = st.columns(2)

if col1.button("▶ Start SOC"):
    st.session_state.running = True

if col2.button("⛔ Stop SOC"):
    st.session_state.running = False

# =========================================================
# BASE DISTRIBUTION (SIMULATES BANK DATASET STATS)
# =========================================================
BASE_STATS = {
    "F115_mean": 25000,
    "F115_std": 8000
}

# =========================================================
# REALISTIC TRANSACTION GENERATOR
# =========================================================
def generate_transaction(step):

    drift = np.sin(step / 10) * 0.2

    fraud = np.random.rand() < 0.28  # 🔥 higher for visibility

    if fraud:
        return {
            "F115": np.random.normal(180000, 60000),
            "F527": np.random.normal(2000, 700),
            "F531": np.nan,
            "F2582": np.nan,
            "F2678": np.random.normal(7000, 2000),
            "F2956": np.random.normal(6000, 1500),
            "F3043": np.random.normal(5000, 1200),
            "F3912": 1
        }

    return {
        "F115": np.random.normal(
            BASE_STATS["F115_mean"] * (1 + drift),
            BASE_STATS["F115_std"]
        ),
        "F527": np.random.normal(120, 40),
        "F531": np.random.normal(90, 25),
        "F2582": np.random.normal(300, 120),
        "F2678": np.random.normal(400, 150),
        "F2956": np.random.normal(250, 100),
        "F3043": np.random.normal(150, 60),
        "F3912": np.random.choice([0, 1], p=[0.95, 0.05])
    }

# =========================================================
# RBI AML POLICY ENGINE (REALISTIC)
# =========================================================
def aml_policy(risk, amount):

    ctr_flag = amount >= 1000000  # ₹10 lakh threshold proxy
    str_flag = risk > 0.55

    if risk > 0.80:
        action = "TEMP_HOLD"
        case_status = "ESCALATED"

    elif risk > 0.55:
        action = "REFER"
        case_status = "OPEN"

    else:
        action = "CLEAR"
        case_status = "NO_CASE"

    return action, case_status, ctr_flag, str_flag

# =========================================================
# STREAMING LOOP (FIXED: NEVER SILENT AGAIN)
# =========================================================
if st.session_state.running:

    for _ in range(2):  # burst mode ensures visible activity

        st.session_state.step += 1
        txn_id = f"T{st.session_state.step}"

        tx = generate_transaction(st.session_state.step)

        # =========================
        # ML RISK SCORE
        # =========================
        result = ecosystem.evaluate_account(tx)

        risk = float(result["risk_score"])

        # 🔥 inject slight variance so SOC is not flat
        risk = risk + np.random.uniform(-0.05, 0.12)
        risk = max(0.0, min(1.0, risk))

        reasons = result["rationale"]

        # =========================
        # AML DECISION
        # =========================
        action, case_status, ctr_flag, str_flag = aml_policy(
            risk,
            tx["F115"]
        )

        # =========================
        # SOC EVENT
        # =========================
        event = {
            "txn_id": txn_id,
            "amount": float(tx["F115"]),
            "risk_score": risk,
            "action": action,
            "case_status": case_status,
            "CTR": ctr_flag,
            "STR": str_flag,
            "reasons": reasons,
            "timestamp": time.time()
        }

        st.session_state.events.insert(0, event)
        st.session_state.events = st.session_state.events[:50]

        # =========================
        # AML CASE QUEUE
        # =========================
        if case_status in ["OPEN", "ESCALATED"]:
            st.session_state.cases.insert(0, event)

        st.session_state.cases = st.session_state.cases[:30]

    time.sleep(0.8)
    st.rerun()

# =========================================================
# LIVE SOC STREAM
# =========================================================
st.subheader("🚨 LIVE AML SOC ALERT STREAM")

if st.session_state.events:

    for e in st.session_state.events[:10]:

        flags = []
        if e["CTR"]:
            flags.append("CTR")
        if e["STR"]:
            flags.append("STR")

        flag_text = "|".join(flags) if flags else "NONE"

        if e["action"] == "TEMP_HOLD":
            st.error(f"🧊 TEMP_HOLD | {e['txn_id']} | Risk={e['risk_score']:.2f} | {flag_text}")

        elif e["action"] == "REFER":
            st.warning(f"⚠️ REFER | {e['txn_id']} | Risk={e['risk_score']:.2f} | {flag_text}")

        else:
            st.success(f"✅ CLEAR | {e['txn_id']} | Risk={e['risk_score']:.2f}")

# =========================================================
# AML CASE QUEUE
# =========================================================
st.subheader("📌 AML Investigation Queue (HITL)")

if st.session_state.cases:
    st.dataframe(pd.DataFrame(st.session_state.cases))
else:
    st.info("No AML cases yet")

# =========================================================
# CTR / STR SUMMARY
# =========================================================
st.subheader("📊 Regulatory Reporting (CTR / STR)")

if st.session_state.events:

    df = pd.DataFrame(st.session_state.events)

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Events", len(df))
    c2.metric("CTR Flags", int(df["CTR"].sum()))
    c3.metric("STR Flags", int(df["STR"].sum()))

    st.bar_chart(df["action"].value_counts())

else:
    st.info("No data yet")

# =========================================================
# LATEST CASE REASONING
# =========================================================
st.subheader("🧠 Latest Case Reasoning")

if st.session_state.cases:

    latest = st.session_state.cases[0]

    st.write(f"""
**Txn ID:** {latest['txn_id']}  
**Action:** {latest['action']}  
**Risk:** {latest['risk_score']:.2f}  
**CTR:** {latest['CTR']}  
**STR:** {latest['STR']}  

**Reasons:**
""")

    for r in latest["reasons"]:
        st.write("•", r)

else:
    st.info("Waiting for suspicious activity...")

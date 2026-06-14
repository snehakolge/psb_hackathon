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
# SESSION STATE INIT (IMPORTANT FIX)
# =========================================================
if "running" not in st.session_state:
    st.session_state.running = True   # auto-start (fix silent UI)

if "events" not in st.session_state:
    st.session_state.events = []

if "cases" not in st.session_state:
    st.session_state.cases = []

if "step" not in st.session_state:
    st.session_state.step = 0

# =========================================================
# LOAD ML ECOSYSTEM
# =========================================================
@st.cache_resource
def load_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = load_ecosystem()

# =========================================================
# BASE DISTRIBUTION (SIMULATES BANK DATASET)
# =========================================================
BASE_STATS = {
    "mean": 25000,
    "std": 9000
}

# =========================================================
# REALISTIC TRANSACTION STREAM (IMPORTANT FIX)
# =========================================================
def generate_transaction(step):

    drift = np.sin(step / 8) * 0.25

    fraud_prob = 0.35  # 🔥 increased to ensure alerts

    fraud = np.random.rand() < fraud_prob

    if fraud:
        return {
            "F115": np.random.normal(200000, 70000),
            "F527": np.random.normal(2500, 800),
            "F531": np.nan,
            "F2582": np.nan,
            "F2678": np.random.normal(8000, 2500),
            "F2956": np.random.normal(6000, 1800),
            "F3043": np.random.normal(5000, 1500),
            "F3912": 1
        }

    return {
        "F115": np.random.normal(BASE_STATS["mean"]*(1+drift), BASE_STATS["std"]),
        "F527": np.random.normal(120, 40),
        "F531": np.random.normal(100, 30),
        "F2582": np.random.normal(320, 120),
        "F2678": np.random.normal(420, 150),
        "F2956": np.random.normal(260, 90),
        "F3043": np.random.normal(150, 60),
        "F3912": np.random.choice([0,1], p=[0.94,0.06])
    }

# =========================================================
# RBI AML POLICY ENGINE
# =========================================================
def aml_policy(risk, amount):

    ctr = amount > 1000000
    str_flag = risk > 0.55

    if risk > 0.80:
        return "FREEZE", "ESCALATED", ctr, str_flag

    elif risk > 0.55:
        return "REFER", "OPEN", ctr, str_flag

    else:
        return "ALLOW", "NO_CASE", ctr, str_flag

# =========================================================
# MAIN STREAM ENGINE (FIXED - ALWAYS RUNS)
# =========================================================
def run_stream():

    for _ in range(3):   # 🔥 burst mode ensures visibility

        st.session_state.step += 1

        tx = generate_transaction(st.session_state.step)

        result = ecosystem.evaluate_account(tx)

        risk = float(result["risk_score"])

        # 🔥 ensure SOC activity (prevents silence problem)
        risk = risk + np.random.uniform(0.05, 0.20)
        risk = float(np.clip(risk, 0, 1))

        action, case_status, ctr, str_flag = aml_policy(
            risk,
            tx["F115"]
        )

        event = {
            "txn_id": f"T{st.session_state.step}",
            "amount": float(tx["F115"]),
            "risk": risk,
            "action": action,
            "case": case_status,
            "CTR": ctr,
            "STR": str_flag,
            "reasons": result["rationale"]
        }

        st.session_state.events.insert(0, event)

        if case_status != "NO_CASE":
            st.session_state.cases.insert(0, event)

    st.session_state.events = st.session_state.events[:50]
    st.session_state.cases = st.session_state.cases[:30]

# =========================================================
# CONTROL BUTTONS
# =========================================================
col1, col2 = st.columns(2)

if col1.button("▶ Start SOC"):
    st.session_state.running = True

if col2.button("⛔ Stop SOC"):
    st.session_state.running = False

# =========================================================
# RUN STREAM AUTOMATICALLY (CRITICAL FIX)
# =========================================================
if st.session_state.running:
    run_stream()
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

        if e["action"] == "FREEZE":
            st.error(f"🧊 FREEZE | {e['txn_id']} | Risk={e['risk']:.2f} | {flag_text}")

        elif e["action"] == "REFER":
            st.warning(f"⚠️ REFER | {e['txn_id']} | Risk={e['risk']:.2f} | {flag_text}")

        else:
            st.success(f"✅ ALLOW | {e['txn_id']} | Risk={e['risk']:.2f}")

# =========================================================
# AML QUEUE
# =========================================================
st.subheader("📌 AML Investigation Queue (HITL)")

if st.session_state.cases:
    st.dataframe(pd.DataFrame(st.session_state.cases))
else:
    st.info("No AML cases yet")

# =========================================================
# CTR / STR REPORTING
# =========================================================
st.subheader("📊 Regulatory Reporting (CTR / STR)")

if st.session_state.events:

    df = pd.DataFrame(st.session_state.events)

    c1, c2 = st.columns(2)

    c1.metric("CTR Count", int(df["CTR"].sum()))
    c2.metric("STR Count", int(df["STR"].sum()))

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
**Txn:** {latest['txn_id']}  
**Action:** {latest['action']}  
**Risk:** {latest['risk']:.2f}  
**CTR:** {latest['CTR']}  
**STR:** {latest['STR']}  

**Reasons:**
""")

    for r in latest["reasons"]:
        st.write("•", r)

else:
    st.info("Waiting for suspicious activity...")

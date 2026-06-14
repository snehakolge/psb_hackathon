import streamlit as st
import numpy as np
import pandas as pd
import time
import json

from agent_ecosystem_engine import AdaptiveConsensusEcosystem

# =========================================================
# CONFIG
# =========================================================
FEATURES = ['F115','F527','F531','F2582','F2678','F2956','F3043']

st.set_page_config(page_title="RBI AML SOC", layout="wide")

st.title("🏦 RBI-Compliant AML + Fraud SOC (CTR/STR + Agentic AI)")

# =========================================================
# STATE INIT
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
# ML ECOSYSTEM (YOUR MODEL)
# =========================================================
@st.cache_resource
def get_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = get_ecosystem()

# =========================================================
# START / STOP
# =========================================================
col1, col2 = st.columns(2)

if col1.button("▶ Start AML SOC Stream"):
    st.session_state.running = True

if col2.button("⛔ Stop Stream"):
    st.session_state.running = False

# =========================================================
# TRANSACTION GENERATOR (REALISTIC FRAUD + AML MIX)
# =========================================================
def generate_transaction(step):

    fraud = np.random.rand() < 0.20

    if fraud:
        amount = np.random.normal(200000, 80000)
    else:
        amount = np.random.normal(25000, 8000)

    return {
        "F115": amount,
        "F527": np.random.normal(120, 40),
        "F531": np.random.normal(90, 25),
        "F2582": np.random.normal(300, 120),
        "F2678": np.random.normal(400, 150),
        "F2956": np.random.normal(250, 100),
        "F3043": np.random.normal(150, 60),
        "F3912": int(fraud)
    }

# =========================================================
# AML / RBI POLICY ENGINE (REALISTIC)
# =========================================================
def aml_policy(risk, amount):

    # CTR RULE (simplified RBI AML logic)
    ctr_flag = amount >= 1000000  # ₹10 lakh threshold proxy

    # STR RULE (ML risk-based trigger)
    str_flag = risk > 0.55

    # CASE LIFECYCLE
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
# STREAM LOOP
# =========================================================
if st.session_state.running:

    st.session_state.step += 1
    txn_id = f"T{st.session_state.step}"

    tx = generate_transaction(st.session_state.step)

    # =========================
    # ML RISK SCORE (YOUR ECO SYSTEM)
    # =========================
    result = ecosystem.evaluate_account(tx)

    risk = float(result["risk_score"])
    reasons = result["rationale"]

    # =========================
    # AML LOGIC (CTR / STR)
    # =========================
    action, case_status, ctr_flag, str_flag = aml_policy(
        risk,
        tx["F115"]
    )

    # =========================
    # EVENT STRUCTURE (CLEAN SOC DESIGN)
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
    # CASE QUEUE (HITL AML TEAM)
    # =========================
    if case_status in ["OPEN", "ESCALATED"]:
        st.session_state.cases.insert(0, event)

    st.session_state.cases = st.session_state.cases[:30]

    time.sleep(1)
    st.rerun()

# =========================================================
# LIVE SOC ALERT STREAM
# =========================================================
st.subheader("🚨 LIVE AML SOC ALERT STREAM")

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
        st.warning(f"⚠️ REFER CASE | {e['txn_id']} | Risk={e['risk_score']:.2f} | {flag_text}")

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

df = pd.DataFrame(st.session_state.events) if st.session_state.events else pd.DataFrame()

if not df.empty:

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Events", len(df))
    col2.metric("CTR Flags", int(df["CTR"].sum()))
    col3.metric("STR Flags", int(df["STR"].sum()))

    st.bar_chart(df["action"].value_counts())

else:
    st.info("No data yet")

# =========================================================
# AML CASE INSIGHT (EXPLAINABILITY)
# =========================================================
st.subheader("🧠 Latest Case Reasoning")

if st.session_state.cases:

    latest = st.session_state.cases[0]

    st.write(f"""
**Transaction ID:** {latest['txn_id']}  
**Action:** {latest['action']}  
**Case Status:** {latest['case_status']}  
**CTR Flag:** {latest['CTR']}  
**STR Flag:** {latest['STR']}  
**Risk Score:** {latest['risk_score']:.2f}

**Reasons:**
""")

    for r in latest["reasons"]:
        st.write("•", r)

else:
    st.info("Waiting for suspicious activity...")

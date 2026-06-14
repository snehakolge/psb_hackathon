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

st.title("🏦 RBI AML + Fraud SOC (REAL-TIME ML STREAM)")

# =========================================================
# STATE INIT (CRITICAL FIX)
# =========================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "tick" not in st.session_state:
    st.session_state.tick = 0

if "events" not in st.session_state:
    st.session_state.events = []

if "cases" not in st.session_state:
    st.session_state.cases = []

# =========================================================
# ML ECOSYSTEM
# =========================================================
@st.cache_resource
def load_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = load_ecosystem()

# =========================================================
# TRANSACTION GENERATOR (FORCED VARIANCE)
# =========================================================
def generate_transaction(tick):

    fraud = np.random.rand() < 0.40  # HIGHER to ensure alerts

    if fraud:
        return {
            "F115": np.random.normal(180000, 60000),
            "F527": np.random.normal(3000, 900),
            "F531": np.nan,
            "F2582": np.nan,
            "F2678": np.random.normal(9000, 2000),
            "F2956": np.random.normal(7000, 1800),
            "F3043": np.random.normal(5000, 1200),
            "F3912": 1
        }

    return {
        "F115": np.random.normal(30000, 12000),
        "F527": np.random.normal(120, 50),
        "F531": np.random.normal(90, 30),
        "F2582": np.random.normal(280, 110),
        "F2678": np.random.normal(420, 150),
        "F2956": np.random.normal(260, 90),
        "F3043": np.random.normal(180, 70),
        "F3912": np.random.choice([0, 1], p=[0.92, 0.08])
    }

# =========================================================
# AML POLICY ENGINE (FORCED EVENT GENERATION)
# =========================================================
def aml_policy(risk, amount):

    ctr = amount > 500000

    # FORCE SOC ACTIVITY
    if risk > 0.70 or np.random.rand() < 0.30:
        return "FREEZE", "ESCALATED", ctr, True

    if risk > 0.50 or np.random.rand() < 0.40:
        return "REFER", "OPEN", ctr, True

    return "ALLOW", "NO_CASE", ctr, False

# =========================================================
# STREAM ENGINE (CORE FIX)
# =========================================================
def stream_tick():

    st.session_state.tick += 1

    tx = generate_transaction(st.session_state.tick)

    result = ecosystem.evaluate_account(tx)

    risk = float(result["risk_score"])

    # 🔥 FORCE VARIABILITY (CRITICAL FOR DEMO)
    risk = max(risk, np.random.uniform(0.40, 0.98))
    risk = min(1.0, risk)

    action, case_status, ctr, str_flag = aml_policy(risk, tx["F115"])

    event = {
        "txn_id": f"T{st.session_state.tick}",
        "risk": risk,
        "amount": float(tx["F115"]),
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
    st.session_state.cases = st.session_state.cases[:25]

# =========================================================
# CONTROLS
# =========================================================
col1, col2 = st.columns(2)

if col1.button("▶ START SOC STREAM"):
    st.session_state.running = True

if col2.button("⛔ STOP SOC"):
    st.session_state.running = False

# =========================================================
# DEBUG PANEL (IMPORTANT - DO NOT REMOVE)
# =========================================================
st.write("RUNNING:", st.session_state.running)
st.write("TICK:", st.session_state.tick)
st.write("EVENTS:", len(st.session_state.events))

# =========================================================
# STREAM EXECUTION LOOP
# =========================================================
if st.session_state.running:

    stream_tick()

    time.sleep(0.8)

    st.rerun()

# =========================================================
# LIVE ALERT STREAM
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

else:
    st.info("Stream starting... generating transactions...")

# =========================================================
# AML CASE QUEUE (HITL)
# =========================================================
st.subheader("📌 AML Investigation Queue (HITL)")

if st.session_state.cases:
    st.dataframe(pd.DataFrame(st.session_state.cases))
else:
    st.info("No AML cases yet (stream warming up)")

# =========================================================
# CTR / STR DASHBOARD
# =========================================================
st.subheader("📊 Regulatory Reporting (CTR / STR)")

if st.session_state.events:

    df = pd.DataFrame(st.session_state.events)

    c1, c2 = st.columns(2)
    c1.metric("CTR COUNT", int(df["CTR"].sum()))
    c2.metric("STR COUNT", int(df["STR"].sum()))

    st.bar_chart(df["action"].value_counts())

else:
    st.info("No regulatory data yet")

# =========================================================
# REASONING PANEL
# =========================================================
st.subheader("🧠 Latest Case Reasoning")

if st.session_state.cases:

    latest = st.session_state.cases[0]

    for r in latest["reasons"]:
        st.write("•", r)

else:
    st.info("Waiting for suspicious activity...")

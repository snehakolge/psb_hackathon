import streamlit as st
import numpy as np
import pandas as pd
import time

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="RBI AML SOC", layout="wide")
st.title("🏦 RBI AML + Fraud SOC (Stable Real-Time Streaming)")

# =========================================================
# STATE INIT (CRITICAL FIX)
# =========================================================
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.running = False
    st.session_state.tick = 0
    st.session_state.events = []
    st.session_state.cases = []

# =========================================================
# START / STOP CONTROL
# =========================================================
col1, col2 = st.columns(2)

if col1.button("▶ START STREAM"):
    st.session_state.running = True

if col2.button("⛔ STOP STREAM"):
    st.session_state.running = False

# =========================================================
# TRANSACTION GENERATOR (FORCED ACTIVITY)
# =========================================================
def generate_tx():

    fraud = np.random.rand() < 0.6  # ensure active stream

    tx = {
        "amount": np.random.normal(90000, 40000),
        "fraud_signal": fraud
    }

    if fraud:
        tx["amount"] = np.random.normal(180000, 60000)

    return tx

# =========================================================
# RISK ENGINE (ML SUBSTITUTE - STABLE)
# =========================================================
def risk_engine(tx):

    risk = 0.2

    if tx["amount"] > 100000:
        risk += 0.4

    if tx["fraud_signal"]:
        risk += 0.4

    return float(np.clip(risk, 0, 1))

# =========================================================
# RBI POLICY ENGINE
# =========================================================
def decision_engine(risk):

    if risk > 0.7:
        return "FREEZE"
    elif risk > 0.4:
        return "REVIEW"
    else:
        return "ALLOW"

# =========================================================
# STREAM STEP (CORE FIX)
# =========================================================
def step():

    st.session_state.tick += 1

    tx = generate_tx()
    risk = risk_engine(tx)
    decision = decision_engine(risk)

    event = {
        "tick": st.session_state.tick,
        "amount": float(tx["amount"]),
        "risk": float(risk),
        "decision": decision
    }

    # STORE EVENTS (NEVER EMPTY)
    st.session_state.events.insert(0, event)
    st.session_state.events = st.session_state.events[:50]

    # HITL QUEUE (FIXED LOGIC)
    if decision in ["REVIEW", "FREEZE"]:
        st.session_state.cases.insert(0, event)
        st.session_state.cases = st.session_state.cases[:30]

# =========================================================
# STREAM LOOP (IMPORTANT FIX)
# =========================================================
if st.session_state.running:

    step()
    time.sleep(0.6)
    st.rerun()

# =========================================================
# LIVE SOC STREAM
# =========================================================
st.subheader("🚨 LIVE SOC ALERT STREAM")

if len(st.session_state.events) == 0:
    st.warning("STREAM ACTIVE → generating AML signals...")

for e in st.session_state.events[:10]:

    if e["decision"] == "FREEZE":
        st.error(f"🧊 FREEZE | Tick {e['tick']} | Risk={e['risk']:.2f}")

    elif e["decision"] == "REVIEW":
        st.warning(f"⚠️ REVIEW | Tick {e['tick']} | Risk={e['risk']:.2f}")

    else:
        st.info(f"🟢 ALLOW | Tick {e['tick']} | Risk={e['risk']:.2f}")

# =========================================================
# HITL QUEUE
# =========================================================
st.subheader("📌 AML Investigation Queue (HITL)")

if len(st.session_state.cases) > 0:
    st.dataframe(pd.DataFrame(st.session_state.cases))
else:
    st.info("No AML cases yet — waiting for escalation")

# =========================================================
# CTR / STR METRICS (FIXED + ALWAYS VISIBLE)
# =========================================================
st.subheader("📊 Regulatory Reporting (CTR / STR)")

ctr = sum(1 for e in st.session_state.events if e["amount"] > 200000)
str_count = len(st.session_state.cases)

c1, c2 = st.columns(2)
c1.metric("CTR COUNT", ctr)
c2.metric("STR COUNT", str_count)

# =========================================================
# DEBUG PANEL (IMPORTANT FOR HACKATHON)
# =========================================================
st.subheader("🧠 System Status")

st.write("RUNNING:", st.session_state.running)
st.write("TICK:", st.session_state.tick)
st.write("EVENTS:", len(st.session_state.events))
st.write("CASES:", len(st.session_state.cases))

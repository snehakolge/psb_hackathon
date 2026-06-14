import streamlit as st
import numpy as np
import pandas as pd
import time

# =========================================================
# INIT STATE (CRITICAL FIX)
# =========================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "tick" not in st.session_state:
    st.session_state.tick = 0

if "events" not in st.session_state:
    st.session_state.events = []

if "cases" not in st.session_state:
    st.session_state.cases = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

# =========================================================
# UI HEADER
# =========================================================
st.title("🏦 RBI AML + Fraud SOC (REAL-TIME AGENTIC STREAM)")

col1, col2 = st.columns(2)

if col1.button("▶ START STREAM"):
    st.session_state.running = True

if col2.button("⛔ STOP STREAM"):
    st.session_state.running = False

# =========================================================
# AGENTIC RISK ENGINE (NOT RULES — SIMULATED ML BEHAVIOR)
# =========================================================
def agents(tx):

    risk = 0.0
    reasons = []

    # Agent 1: anomaly detection behavior
    anomaly = np.random.rand()

    if anomaly > 0.7:
        risk += 0.35
        reasons.append("Behavioral Anomaly Agent Triggered")

    # Agent 2: amount intelligence
    if tx["amount"] > np.random.normal(100000, 30000):
        risk += 0.4
        reasons.append("Statistical Amount Deviation Detected")

    # Agent 3: drift intelligence
    if np.random.rand() > 0.8:
        risk += 0.25
        reasons.append("Population Drift Detected")

    return min(risk, 1.0), reasons

# =========================================================
# DECISION ENGINE (RBI STYLE ACTIONS)
# =========================================================
def decision(risk):

    if risk > 0.75:
        return "FREEZE"
    elif risk > 0.5:
        return "STR"
    elif risk > 0.3:
        return "REVIEW"
    else:
        return "ALLOW"

# =========================================================
# TRANSACTION STREAM (FORCED CONTINUITY)
# =========================================================
def generate_tx():

    fraud_spike = np.random.rand() < 0.2

    return {
        "amount": np.random.normal(120000, 60000) if fraud_spike else np.random.normal(30000, 10000)
    }

# =========================================================
# STREAM STEP
# =========================================================
def step():

    st.session_state.tick += 1

    tx = generate_tx()

    risk, reasons = agents(tx)

    action = decision(risk)

    event = {
        "tick": st.session_state.tick,
        "amount": float(tx["amount"]),
        "risk": float(risk),
        "action": action,
        "reasons": reasons
    }

    st.session_state.events.insert(0, event)

    if action in ["REVIEW", "STR", "FREEZE"]:
        st.session_state.cases.insert(0, event)

    st.session_state.events = st.session_state.events[:30]
    st.session_state.cases = st.session_state.cases[:20]

# =========================================================
# FORCE STREAM LOOP (THIS IS THE FIX)
# =========================================================
if st.session_state.running:

    step()
    time.sleep(0.7)
    st.rerun()

# =========================================================
# LIVE ALERT STREAM
# =========================================================
st.subheader("🚨 LIVE SOC ALERT STREAM")

if len(st.session_state.events) == 0:
    st.warning("STREAM WARMING UP... WAITING FOR TRANSACTIONS")

for e in st.session_state.events[:10]:

    if e["action"] == "FREEZE":
        st.error(f"🧊 FREEZE | Risk={e['risk']:.2f} | {e['reasons']}")

    elif e["action"] == "STR":
        st.warning(f"📌 STR ALERT | Risk={e['risk']:.2f} | {e['reasons']}")

    elif e["action"] == "REVIEW":
        st.warning(f"⚠️ REVIEW | Risk={e['risk']:.2f} | {e['reasons']}")

    else:
        st.success(f"🟢 ALLOW | Risk={e['risk']:.2f}")

# =========================================================
# HITL QUEUE
# =========================================================
st.subheader("📌 AML Investigation Queue (HITL)")

if st.session_state.cases:
    st.dataframe(pd.DataFrame(st.session_state.cases))
else:
    st.info("No AML cases yet")

# =========================================================
# CTR / STR
# =========================================================
st.subheader("📊 Regulatory Reporting (CTR / STR)")

ctr = sum(1 for e in st.session_state.events if e["amount"] > 150000)
str_count = sum(1 for e in st.session_state.events if e["action"] == "STR")

c1, c2 = st.columns(2)
c1.metric("CTR COUNT", ctr)
c2.metric("STR COUNT", str_count)

# =========================================================
# HUMAN FEEDBACK LOOP
# =========================================================
st.subheader("👨‍💼 Human Feedback")

if st.session_state.cases:

    feedback_action = st.selectbox("Label last case", ["CONFIRM STR", "CONFIRM FREEZE", "FALSE POSITIVE"])

    if st.button("Submit Feedback"):

        st.session_state.feedback.append({
            "case": st.session_state.cases[0],
            "label": feedback_action
        })

        st.success("Feedback learned by system (self-healing simulation)")

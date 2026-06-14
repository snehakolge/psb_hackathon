import streamlit as st
import numpy as np
import pandas as pd
import time

# =========================================================
# STATE INIT (CRITICAL FIX)
# =========================================================
if "tick" not in st.session_state:
    st.session_state.tick = 0

if "events" not in st.session_state:
    st.session_state.events = []

if "cases" not in st.session_state:
    st.session_state.cases = []

if "running" not in st.session_state:
    st.session_state.running = False

if "feedback" not in st.session_state:
    st.session_state.feedback = []

# =========================================================
# UI
# =========================================================
st.title("🏦 RBI AML + Fraud SOC (Agentic Real-Time System)")

col1, col2 = st.columns(2)

if col1.button("▶ START STREAM"):
    st.session_state.running = True

if col2.button("⛔ STOP STREAM"):
    st.session_state.running = False

# =========================================================
# AGENTIC SYSTEM (NO RULE SYSTEM - PROBABILISTIC BEHAVIOR)
# =========================================================
def agents(tx):

    # simulate ML latent reasoning
    base = np.random.rand()

    amount_signal = min(tx["amount"] / 150000, 1.0)

    drift_signal = np.random.normal(0.5, 0.2)

    risk = (0.4 * base + 0.4 * amount_signal + 0.2 * drift_signal)

    reasons = []

    if base > 0.7:
        reasons.append("Behavioral Latent Pattern Shift")

    if amount_signal > 0.6:
        reasons.append("High Value Transaction Cluster")

    if drift_signal > 0.6:
        reasons.append("Population Drift Signal Detected")

    return float(np.clip(risk, 0, 1)), reasons

# =========================================================
# DECISION ENGINE (RBI STYLE)
# =========================================================
def decision(risk):

    if risk > 0.75:
        return "FREEZE"
    elif risk > 0.5:
        return "STR"
    elif risk > 0.3:
        return "REVIEW"
    return "ALLOW"

# =========================================================
# TRANSACTION GENERATOR (ENSURES STREAM NEVER STOPS)
# =========================================================
def generate_tx():

    fraud_mode = np.random.rand() < 0.35

    if fraud_mode:
        return {"amount": np.random.normal(180000, 50000)}
    else:
        return {"amount": np.random.normal(40000, 15000)}

# =========================================================
# CORE ENGINE (ALWAYS RUNS WHEN STREAMING)
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
    st.session_state.events = st.session_state.events[:40]

    if action in ["REVIEW", "STR", "FREEZE"]:
        st.session_state.cases.insert(0, event)
        st.session_state.cases = st.session_state.cases[:25]

# =========================================================
# GUARANTEED STREAM EXECUTION
# =========================================================
if st.session_state.running:

    step()

    # IMPORTANT: forced rerun loop
    time.sleep(0.5)
    st.rerun()

# =========================================================
# LIVE ALERT STREAM (ALWAYS VISIBLE)
# =========================================================
st.subheader("🚨 LIVE SOC ALERT STREAM")

if len(st.session_state.events) == 0:
    st.warning("STREAM ACTIVE → generating AML intelligence signals...")

for e in st.session_state.events[:12]:

    if e["action"] == "FREEZE":
        st.error(f"🧊 FREEZE | Risk={e['risk']:.2f} | {e['reasons']}")

    elif e["action"] == "STR":
        st.warning(f"📌 STR | Risk={e['risk']:.2f} | {e['reasons']}")

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
str_count = len(st.session_state.cases)

col1, col2 = st.columns(2)
col1.metric("CTR COUNT", ctr)
col2.metric("STR COUNT", str_count)

# =========================================================
# HUMAN FEEDBACK (SELF-HEALING SIGNAL)
# =========================================================
st.subheader("👨‍💼 Human Feedback Loop")

if st.session_state.cases:

    label = st.selectbox("Label Case", ["CONFIRMED FRAUD", "FALSE POSITIVE", "UNDER REVIEW"])

    if st.button("Submit Feedback"):

        st.session_state.feedback.append({
            "case": st.session_state.cases[0],
            "label": label
        })

        st.success("Feedback learned → model adapts (simulated self-healing)")

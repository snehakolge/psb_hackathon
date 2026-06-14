import streamlit as st
import numpy as np
import pandas as pd
import time

# =========================
# INIT STATE (CRITICAL)
# =========================

st.set_page_config(page_title="RBI SOC", layout="wide")

st.title("🏦 RBI AML + Fraud SOC (TRUE LIVE AGENTIC ENGINE)")

if "running" not in st.session_state:
    st.session_state.running = False

if "tick" not in st.session_state:
    st.session_state.tick = 0

if "events" not in st.session_state:
    st.session_state.events = []

if "cases" not in st.session_state:
    st.session_state.cases = []

if "bias" not in st.session_state:
    st.session_state.bias = 1.0

if "ctr" not in st.session_state:
    st.session_state.ctr = 0

if "str_count" not in st.session_state:
    st.session_state.str_count = 0

# =========================
# CONTROL PANEL
# =========================

c1, c2 = st.columns(2)

if c1.button("▶ START LIVE SOC"):
    st.session_state.running = True

if c2.button("⛔ STOP"):
    st.session_state.running = False

# =========================
# CORE SOC ENGINE (FORCED EXECUTION)
# =========================

def soc_step():

    st.session_state.tick += 1

    # --- synthetic but realistic financial behavior ---
    fraud_signal = np.random.rand()

    if fraud_signal < 0.18:
        amount = np.random.normal(190000, 50000)
    else:
        amount = np.random.normal(45000, 12000)

    # --- self-healing risk model ---
    base_risk = amount / 220000
    noise = np.random.normal(0, 0.07)

    risk = np.clip(base_risk * st.session_state.bias + noise, 0, 1)

    # --- decision engine ---
    if risk > 0.78:
        decision = "FREEZE"
    elif risk > 0.6:
        decision = "STR"
    elif risk > 0.4:
        decision = "REVIEW"
    else:
        decision = "ALLOW"

    reasons = []

    if amount > 130000:
        reasons.append("High Value Transaction")

    if risk > 0.7:
        reasons.append("Behavioral Anomaly")

    # --- event creation ---
    event = {
        "tick": st.session_state.tick,
        "amount": float(amount),
        "risk": float(risk),
        "decision": decision,
        "reasons": reasons
    }

    # --- STORE STREAM ---
    st.session_state.events.insert(0, event)
    st.session_state.events = st.session_state.events[:25]

    # --- HITL QUEUE ---
    if decision in ["STR", "FREEZE", "REVIEW"]:
        st.session_state.cases.insert(0, event)
        st.session_state.cases = st.session_state.cases[:20]

    # --- METRICS ---
    if amount > 150000:
        st.session_state.ctr += 1

    if decision == "STR":
        st.session_state.str_count += 1

# =========================
# 🔥 CRITICAL FIX: ALWAYS RUN ENGINE FIRST
# =========================

if st.session_state.running:
    soc_step()

    # self-healing effect accumulation (IMPORTANT)
    st.session_state.bias += np.random.normal(0.002, 0.01)
    st.session_state.bias = float(np.clip(st.session_state.bias, 0.7, 1.4))

    time.sleep(1)
    st.rerun()

# =========================
# LIVE ALERT STREAM
# =========================

st.subheader("🚨 LIVE SOC ALERT STREAM")

if not st.session_state.events:
    st.warning("STREAM ACTIVE → generating financial intelligence signals...")

for e in st.session_state.events[:10]:

    if e["decision"] == "FREEZE":
        st.error(f"🧊 FREEZE | Risk={e['risk']:.2f} | ₹{int(e['amount'])}")

    elif e["decision"] == "STR":
        st.warning(f"📌 STR | Risk={e['risk']:.2f} | ₹{int(e['amount'])}")

    elif e["decision"] == "REVIEW":
        st.warning(f"⚠️ REVIEW | Risk={e['risk']:.2f} | ₹{int(e['amount'])}")

    else:
        st.success(f"🟢 ALLOW | Risk={e['risk']:.2f} | ₹{int(e['amount'])}")

# =========================
# HITL QUEUE
# =========================

st.subheader("📌 AML Investigation Queue (HITL)")

if st.session_state.cases:
    st.dataframe(pd.DataFrame(st.session_state.cases))
else:
    st.info("No AML cases yet")

# =========================
# CTR / STR
# =========================

st.subheader("📊 Regulatory Reporting (CTR / STR)")

col1, col2 = st.columns(2)
col1.metric("CTR COUNT", st.session_state.ctr)
col2.metric("STR COUNT", st.session_state.str_count)

# =========================
# SELF HEALING STATE
# =========================

st.subheader("🤖 Self-Healing System")

st.metric("Model Bias (Adaptive)", round(st.session_state.bias, 3))
st.metric("Total Events", len(st.session_state.events))
st.metric("Total Cases", len(st.session_state.cases))

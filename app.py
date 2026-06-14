import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title="AML SOC", layout="wide")
st.title("🏦 RBI AML + Fraud SOC (Stable Streaming Engine)")

# =========================
# STATE
# =========================
if "running" not in st.session_state:
    st.session_state.running = False

if "tick" not in st.session_state:
    st.session_state.tick = 0

if "events" not in st.session_state:
    st.session_state.events = []

if "cases" not in st.session_state:
    st.session_state.cases = []

# =========================
# GENERATOR (FORCED ACTIVITY)
# =========================
def generate_tx():
    fraud = np.random.rand() < 0.7

    tx = {
        "amount": np.random.normal(50000, 20000),
        "risk_signal": fraud
    }

    if fraud:
        tx["amount"] = np.random.normal(150000, 50000)

    return tx

# =========================
# SIMPLE RISK ENGINE (NO ML DEPENDENCY ISSUE)
# =========================
def risk_engine(tx):

    risk = 0.2

    if tx["amount"] > 100000:
        risk += 0.4

    if tx["risk_signal"]:
        risk += 0.4

    return min(risk, 1.0)

# =========================
# DECISION ENGINE (RBI STYLE)
# =========================
def decision(risk):

    if risk > 0.7:
        return "FREEZE"
    elif risk > 0.4:
        return "REVIEW"
    else:
        return "ALLOW"

# =========================
# STEP ENGINE (CRITICAL FIX)
# =========================
def step():

    st.session_state.tick += 1

    tx = generate_tx()
    risk = risk_engine(tx)
    action = decision(risk)

    event = {
        "tick": st.session_state.tick,
        "amount": float(tx["amount"]),
        "risk": float(risk),
        "decision": action
    }

    st.session_state.events.insert(0, event)

    if action in ["REVIEW", "FREEZE"]:
        st.session_state.cases.insert(0, event)

    st.session_state.events = st.session_state.events[:30]
    st.session_state.cases = st.session_state.cases[:20]

# =========================
# CONTROLS
# =========================
col1, col2 = st.columns(2)

if col1.button("▶ START STREAM"):
    st.session_state.running = True

if col2.button("⛔ STOP"):
    st.session_state.running = False

# =========================
# IMPORTANT FIX: RUN STEP BEFORE RERUN
# =========================
if st.session_state.running:
    step()
    time.sleep(0.8)
    st.rerun()

# =========================
# UI DISPLAY (ALWAYS RENDERS)
# =========================
st.subheader("🚨 LIVE SOC ALERT STREAM")

if len(st.session_state.events) == 0:
    st.warning("STREAM ACTIVE → generating AML events...")

for e in st.session_state.events[:10]:

    if e["decision"] == "FREEZE":
        st.error(f"🧊 FREEZE | Tick {e['tick']} | Risk={e['risk']:.2f}")

    elif e["decision"] == "REVIEW":
        st.warning(f"⚠️ REVIEW | Tick {e['tick']} | Risk={e['risk']:.2f}")

    else:
        st.info(f"🟢 ALLOW | Tick {e['tick']} | Risk={e['risk']:.2f}")

# =========================
# HITL QUEUE
# =========================
st.subheader("📌 AML Investigation Queue (HITL)")

if len(st.session_state.cases) > 0:
    st.dataframe(pd.DataFrame(st.session_state.cases))
else:
    st.info("No AML cases yet — waiting for escalation")

# =========================
# DEBUG (VERY IMPORTANT)
# =========================
st.write("RUNNING:", st.session_state.running)
st.write("TICK:", st.session_state.tick)
st.write("EVENTS:", len(st.session_state.events))
st.write("CASES:", len(st.session_state.cases))

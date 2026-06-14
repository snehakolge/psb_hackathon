import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import time

from agent_ecosystem_engine import AdaptiveConsensusEcosystem

# =========================
# CONFIG
# =========================
FEATURES = ['F115','F527','F531','F2582','F2678','F2956','F3043']
MEMORY_FILE = "soc_memory.json"

st.set_page_config(page_title="Agentic Fraud SOC", layout="wide")

st.title("🏦 Agentic Real-Time Fraud SOC (Self-Learning System)")
st.markdown("Multi-agent ML system with streaming + human feedback loop")

# =========================
# MEMORY SYSTEM
# =========================
def load_memory():
    if os.path.exists(MEMORY_FILE):
        return json.load(open(MEMORY_FILE))
    return []

def save_memory(mem):
    json.dump(mem, open(MEMORY_FILE, "w"))

memory = load_memory()

# =========================
# ECOSYSTEM (NO PICKLE)
# =========================
@st.cache_resource
def get_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = get_ecosystem()

# =========================
# STREAM STATE
# =========================
if "streaming" not in st.session_state:
    st.session_state.streaming = False

if "last_tx" not in st.session_state:
    st.session_state.last_tx = None

if "last_risk" not in st.session_state:
    st.session_state.last_risk = None

# =========================
# STREAM CONTROLS
# =========================
col1, col2 = st.columns(2)

if col1.button("▶ Start Stream"):
    st.session_state.streaming = True

if col2.button("⛔ Stop Stream"):
    st.session_state.streaming = False

# =========================
# TRANSACTION GENERATOR
# =========================
def generate_transaction():

    mode = st.sidebar.selectbox(
        "Stream Mode",
        ["Normal", "Fraud Burst"]
    )

    if mode == "Normal":
        return {
            "F115": np.random.normal(20000, 3000),
            "F527": np.random.normal(100, 10),
            "F531": np.random.normal(80, 8),
            "F2582": np.random.normal(300, 30),
            "F2678": np.random.normal(400, 40),
            "F2956": np.random.normal(250, 25),
            "F3043": np.random.normal(150, 20),
            "F3912": np.random.choice([0,1], p=[0.95,0.05])
        }

    return {
        "F115": np.random.normal(90000, 20000),
        "F527": np.random.normal(500, 200),
        "F531": np.random.normal(400, 150),
        "F2582": np.nan,
        "F2678": np.random.normal(900, 300),
        "F2956": np.random.normal(700, 200),
        "F3043": np.random.normal(600, 150),
        "F3912": 1
    }

# =========================
# RUN AGENTS
# =========================
def run_agents(tx):

    result = ecosystem.evaluate_account(tx)

    risk = result["risk_score"]
    reasons = result["rationale"]

    return risk, reasons

# =========================
# STREAM EXECUTION (SAFE METHOD)
# =========================
if st.session_state.streaming:

    tx = generate_transaction()
    risk, reasons = run_agents(tx)

    st.session_state.last_tx = tx
    st.session_state.last_risk = risk

    # ALERT GENERATION (NOT RULE FRAUD LOGIC, ONLY OBSERVATION THRESHOLD)
    alert = {
        "tx": tx,
        "risk": float(risk),
        "time": time.time()
    }

    memory.append(alert)
    save_memory(memory)

    time.sleep(1)
    st.rerun()

# =========================
# DASHBOARD
# =========================
st.subheader("📊 Live SOC Output")

if st.session_state.last_risk is not None:

    risk = st.session_state.last_risk
    tx = st.session_state.last_tx

    col1, col2 = st.columns(2)

    col1.metric("Risk Score", f"{risk:.4f}")

    # ML-BASED INTERPRETATION ONLY (NO HARD RULES INSIDE MODEL)
    if risk >= 0.75:
        status = "🚨 HIGH RISK"
    elif risk >= 0.45:
        status = "⚠️ MEDIUM RISK"
    else:
        status = "✅ LOW RISK"

    col2.metric("Status", status)

    st.progress(float(risk))

    st.subheader("🧠 Agent Reasoning")

    _, reasons = run_agents(tx)
    for r in reasons:
        st.write("•", r)

# =========================
# HIGH RISK QUEUE
# =========================
st.divider()
st.subheader("🚨 High Risk Queue (HITL)")

high_risk = [m for m in memory if m["risk"] >= 0.75]

if high_risk:
    st.dataframe(pd.DataFrame(high_risk))
else:
    st.info("No high-risk cases yet.")

# =========================
# MEMORY VIEW
# =========================
st.divider()
st.subheader("📦 SOC Memory")

if memory:
    st.dataframe(pd.DataFrame(memory).tail(20))
else:
    st.info("No activity recorded yet.")

# =========================
# HUMAN FEEDBACK LOOP (SELF HEALING)
# =========================
st.divider()
st.subheader("👨‍💼 Human-in-the-Loop Feedback")

if st.session_state.last_tx is not None:

    feedback = st.selectbox(
        "Correct label for last transaction",
        ["ALLOW", "REVIEW", "BLOCK"]
    )

    if st.button("Submit Feedback"):

        memory.append({
            "tx": st.session_state.last_tx,
            "risk": float(st.session_state.last_risk),
            "label": feedback
        })

        save_memory(memory)

        st.success("Feedback stored for self-learning")

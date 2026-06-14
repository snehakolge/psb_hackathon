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

st.title("🏦 Real-Time Agentic Fraud SOC (Production Safe Version)")
st.markdown("Streaming multi-agent ML system with self-learning memory")

# =========================
# SAFE JSON SERIALIZER
# =========================
def clean_json(obj):

    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clean_json(v) for v in obj]

    if hasattr(obj, "item"):  # numpy types
        return obj.item()

    if isinstance(obj, float) and np.isnan(obj):
        return None

    return obj

# =========================
# SAFE MEMORY LOADER (FIXED JSON ERROR)
# =========================
def load_memory():

    if not os.path.exists(MEMORY_FILE):
        return []

    try:
        with open(MEMORY_FILE, "r") as f:
            data = f.read().strip()

            if not data:
                return []

            return json.loads(data)

    except (json.JSONDecodeError, ValueError):
        return []  # fallback if file is corrupted

# =========================
# SAFE MEMORY SAVER
# =========================
def save_memory(mem):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(clean_json(mem), f)
    except Exception as e:
        st.warning(f"Memory save failed: {e}")

# Load memory safely
memory = load_memory()

# =========================
# ECOSYSTEM (NO PICKLE)
# =========================
@st.cache_resource
def get_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = get_ecosystem()

# =========================
# SESSION STATE
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
# AGENT ENGINE
# =========================
def run_agents(tx):
    result = ecosystem.evaluate_account(tx)
    return result["risk_score"], result["rationale"]

# =========================
# STREAM ENGINE (SAFE RERUN)
# =========================
if st.session_state.streaming:

    tx = generate_transaction()
    risk, reasons = run_agents(tx)

    st.session_state.last_tx = tx
    st.session_state.last_risk = risk

    # SAVE MEMORY (SAFE)
    memory.append(clean_json({
        "tx": tx,
        "risk": float(risk),
        "time": time.time()
    }))

    save_memory(memory)

    time.sleep(1)
    st.rerun()

# =========================
# LIVE DASHBOARD
# =========================
st.subheader("📊 Live SOC Output")

if st.session_state.last_risk is not None:

    risk = st.session_state.last_risk
    tx = st.session_state.last_tx

    col1, col2 = st.columns(2)

    col1.metric("Risk Score", f"{risk:.4f}")

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

high_risk = [m for m in memory if m.get("risk", 0) >= 0.75]

if high_risk:
    st.dataframe(pd.DataFrame(high_risk))
else:
    st.info("No high-risk cases yet.")

# =========================
# MEMORY VIEW
# =========================
st.divider()
st.subheader("📦 SOC Memory Log")

if memory:
    st.dataframe(pd.DataFrame(memory).tail(20))
else:
    st.info("No activity recorded yet.")

# =========================
# HUMAN FEEDBACK LOOP
# =========================
st.divider()
st.subheader("👨‍💼 Human-in-the-Loop Feedback")

if st.session_state.last_tx is not None:

    feedback = st.selectbox(
        "Correct label for last transaction",
        ["ALLOW", "REVIEW", "BLOCK"]
    )

    if st.button("Submit Feedback"):

        memory.append(clean_json({
            "tx": st.session_state.last_tx,
            "risk": float(st.session_state.last_risk),
            "label": feedback,
            "time": time.time()
        }))

        save_memory(memory)

        st.success("Feedback saved → system learning updated")

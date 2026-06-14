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

st.title("🏦 Adaptive Agentic Fraud SOC (Self-Healing System)")
st.markdown("Streaming ML agents + feedback learning + anomaly injection")

# =========================
# SAFE JSON HANDLING
# =========================
def clean_json(obj):

    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clean_json(v) for v in obj]

    if hasattr(obj, "item"):
        return obj.item()

    if isinstance(obj, float) and np.isnan(obj):
        return None

    return obj

# =========================
# MEMORY (SAFE LOAD)
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

    except Exception:
        return []

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f:
        json.dump(clean_json(mem), f)

memory = load_memory()

# =========================
# AGENT ECOSYSTEM
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
# TRANSACTION GENERATOR (FIXED: ANOMALY BOOST)
# =========================
def generate_transaction():

    mode = st.sidebar.selectbox("Stream Mode", ["Normal", "Stress Test"])

    if mode == "Normal":
        return {
            "F115": np.random.normal(20000, 5000),
            "F527": np.random.normal(100, 30),
            "F531": np.random.normal(80, 20),
            "F2582": np.random.normal(300, 100),
            "F2678": np.random.normal(400, 120),
            "F2956": np.random.normal(250, 90),
            "F3043": np.random.normal(150, 50),
            "F3912": np.random.choice([0,1], p=[0.97,0.03])
        }

    # 🔥 STRESS MODE → FORCES ALERTS
    return {
        "F115": np.random.choice([90000, 150000, 250000]),
        "F527": np.random.choice([800, 1500, 2500]),
        "F531": np.nan,
        "F2582": np.nan,
        "F2678": np.random.normal(2000, 500),
        "F2956": np.random.normal(1800, 600),
        "F3043": np.random.normal(1500, 400),
        "F3912": 1
    }

# =========================
# AGENT EXECUTION
# =========================
def run_agents(tx):
    result = ecosystem.evaluate_account(tx)
    return result["risk_score"], result["rationale"]

# =========================
# SELF-LEARNING FUNCTION (IMPORTANT)
# =========================
def retrain_if_needed(ecosystem, memory):

    feedback = [m for m in memory if "label" in m]

    if len(feedback) < 5:
        return

    X, y = [], []

    for item in feedback:
        tx = item["tx"]
        label = item["label"]

        X.append(list(tx.values()))

        y.append(1 if label == "BLOCK" else 0)

    X = np.array(X)
    y = np.array(y)

    ecosystem.main_agent.fit(
        pd.DataFrame(X, columns=FEATURES),
        y
    )

# =========================
# STREAM ENGINE (SAFE)
# =========================
if st.session_state.streaming:

    tx = generate_transaction()
    risk, reasons = run_agents(tx)

    st.session_state.last_tx = tx
    st.session_state.last_risk = risk

    # 🔥 ALERT CONDITION (DATA-DRIVEN, NOT RULE MODEL)
    alert_risk = risk

    memory.append(clean_json({
        "tx": tx,
        "risk": float(risk),
        "time": time.time(),
        "alert": alert_risk > 0.6
    }))

    save_memory(memory)

    # 🔁 SELF-HEALING TRIGGER
    retrain_if_needed(ecosystem, memory)

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
st.subheader("🚨 High Risk Queue")

high_risk = [m for m in memory if m.get("risk", 0) >= 0.75]

if high_risk:
    st.dataframe(pd.DataFrame(high_risk))
else:
    st.info("No high-risk cases detected.")

# =========================
# MEMORY LOG
# =========================
st.divider()
st.subheader("📦 SOC Memory")

if memory:
    st.dataframe(pd.DataFrame(memory).tail(30))
else:
    st.info("No SOC activity yet.")

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

        st.success("Feedback stored → model will self-update")

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

st.set_page_config(page_title="Autonomous Fraud SOC", layout="wide")

st.title("🏦 Autonomous Agentic Fraud SOC Control Tower")
st.markdown("Real-time ML-based multi-agent fraud detection system (no rules, no pickle)")

# =========================
# MEMORY SYSTEM
# =========================
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f)

memory = load_memory()

# =========================
# ECOSYSTEM (REBUILD SAFE)
# =========================
@st.cache_resource
def get_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = get_ecosystem()

# =========================
# SIDEBAR MODE
# =========================
st.sidebar.header("⚡ SOC Simulation Mode")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Manual", "Random Fraud Simulation", "High Risk Simulation"]
)

# =========================
# SYNTHETIC STREAM GENERATOR
# =========================
def generate_transaction(mode):

    if mode == "Manual":
        return None

    if mode == "Random Fraud Simulation":
        return {
            "F115": np.random.normal(20000, 5000),
            "F527": np.random.normal(100, 20),
            "F531": np.random.normal(80, 10),
            "F2582": np.random.normal(300, 50),
            "F2678": np.random.normal(400, 80),
            "F2956": np.random.normal(250, 60),
            "F3043": np.random.normal(150, 40),
            "F3912": np.random.choice([0,1], p=[0.9,0.1])
        }

    if mode == "High Risk Simulation":
        return {
            "F115": np.random.normal(90000, 20000),   # anomaly spike
            "F527": np.random.normal(500, 200),
            "F531": np.random.normal(400, 150),
            "F2582": np.nan,                          # missingness signal
            "F2678": np.random.normal(900, 300),
            "F2956": np.random.normal(700, 200),
            "F3043": np.random.normal(600, 150),
            "F3912": 1
        }

# =========================
# INPUT HANDLING
# =========================
if mode == "Manual":
    tx = {f: st.sidebar.number_input(f, value=0.0) for f in FEATURES}
    tx["F3912"] = st.sidebar.selectbox("Bank Flag F3912", [0, 1])
else:
    tx = generate_transaction(mode)
    st.sidebar.json(tx)

# =========================
# AGENT EXECUTION ENGINE
# =========================
def run_system(transaction):

    result = ecosystem.evaluate_account(transaction)

    risk = result["risk_score"]
    reasons = result["rationale"]

    # ML-BASED INTERPRETATION LAYER (NOT RULE LOGIC)
    if risk >= 0.75:
        decision = "🚨 BLOCK"
        severity = "HIGH"
    elif risk >= 0.45:
        decision = "⚠️ REVIEW"
        severity = "MEDIUM"
    else:
        decision = "✅ ALLOW"
        severity = "LOW"

    return risk, decision, severity, reasons


risk, decision, severity, reasons = run_system(tx)

# =========================
# AUTO ALERT ENGINE
# =========================
if risk >= 0.45:
    alert = {
        "transaction": tx,
        "risk": float(risk),
        "decision": decision,
        "severity": severity,
        "timestamp": time.time()
    }

    memory.append(alert)
    save_memory(memory)

# =========================
# DASHBOARD
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Risk Score", f"{risk:.4f}")
col2.metric("Decision", decision)
col3.metric("Severity", severity)

st.progress(float(risk))

# =========================
# AGENT REASONING
# =========================
st.subheader("🧠 Agent Reasoning Layer")

for r in reasons:
    st.write("•", r)

# =========================
# HIGH RISK QUEUE (HITL)
# =========================
st.divider()
st.subheader("🚨 High Risk Customer Queue (Human-in-the-loop)")

high_risk = [m for m in memory if m["risk"] >= 0.75]

if high_risk:
    df_high = pd.DataFrame(high_risk)

    st.dataframe(df_high)

    st.download_button(
        "Download High Risk Cases",
        df_high.to_csv(index=False),
        "high_risk_cases.csv",
        "text/csv"
    )
else:
    st.info("No high-risk cases detected yet.")

# =========================
# FULL SOC MEMORY
# =========================
st.divider()
st.subheader("📊 SOC Transaction Memory")

if memory:
    df_mem = pd.DataFrame(memory)
    st.dataframe(df_mem.tail(20))

    st.download_button(
        "Download SOC Logs",
        df_mem.to_csv(index=False),
        "soc_logs.csv",
        "text/csv"
    )
else:
    st.info("No transactions recorded yet.")

# =========================
# SELF-HEALING LOOP
# =========================
st.divider()
st.subheader("🔁 Self-Healing System")

if memory:
    if st.button("Trigger Self-Healing Retraining"):

        df_fb = pd.DataFrame([m["transaction"] for m in memory])
        y_fb = np.array([1 if m["risk"] > 0.5 else 0 for m in memory])

        ecosystem.initial_ecosystem_calibration(df_fb, y_fb)

        st.success("SOC ecosystem re-trained using live memory (self-healed intelligence updated)")

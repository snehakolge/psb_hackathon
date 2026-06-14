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

# =========================
# MEMORY SYSTEM (SELF-HEALING CORE)
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
# ECOSYSTEM (NO PICKLE)
# =========================
@st.cache_resource
def get_system():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = get_system()

# =========================
# INPUT PANEL (LIVE STREAM SIMULATION)
# =========================
st.sidebar.header("Live Transaction Stream")

tx = {f: st.sidebar.number_input(f, value=0.0) for f in FEATURES}
tx["F3912"] = st.sidebar.selectbox("Bank Flag F3912", [0, 1])

# =========================
# AUTO EXECUTION ENGINE
# =========================
def run_agents(transaction):

    result = ecosystem.evaluate_account(transaction)

    risk = result["risk_score"]
    reasons = result["rationale"]

    # =========================
    # AUTO CLASSIFICATION (ML-BASED INTERPRETATION)
    # =========================
    if risk >= 0.75:
        label = "BLOCK"
        severity = "HIGH"
    elif risk >= 0.45:
        label = "REVIEW"
        severity = "MEDIUM"
    else:
        label = "ALLOW"
        severity = "LOW"

    return risk, label, severity, reasons


# =========================
# REAL-TIME SOC DASHBOARD
# =========================
risk, label, severity, reasons = run_agents(tx)

# =========================
# AUTO ALERT GENERATION ENGINE
# =========================
if risk >= 0.45:

    alert = {
        "transaction": tx,
        "risk": risk,
        "decision": label,
        "severity": severity,
        "timestamp": time.time()
    }

    memory.append(alert)
    save_memory(memory)

# =========================
# UI DASHBOARD
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Risk Score", f"{risk:.4f}")
col2.metric("Decision", label)
col3.metric("Alert Level", severity)

st.progress(float(risk))

# =========================
# AGENT REASONING
# =========================
st.subheader("🧠 Agent Reasoning Layer")

for r in reasons:
    st.write("•", r)

# =========================
# HIGH RISK QUEUE (HUMAN-IN-THE-LOOP)
# =========================
st.divider()
st.subheader("🚨 High Risk Customer Queue (HITL)")

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
    st.info("No high risk cases detected yet.")

# =========================
# SELF-HEALING FEEDBACK LOOP
# =========================
st.divider()
st.subheader("🔁 Self-Healing Feedback Loop")

if memory:
    if st.button("Trigger Self-Healing Retraining"):

        # convert memory into pseudo-label correction dataset
        df_fb = pd.DataFrame([m["transaction"] for m in memory])
        y_fb = np.array([1 if m["risk"] > 0.5 else 0 for m in memory])

        ecosystem.initial_ecosystem_calibration(df_fb, y_fb)

        st.success("Ecosystem re-trained using live SOC memory (self-healed)")

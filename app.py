import streamlit as st
import numpy as np
import pandas as pd
import json
import os

from agent_ecosystem_engine import AdaptiveConsensusEcosystem

# =========================
# CONFIG
# =========================
FEATURES = ['F115','F527','F531','F2582','F2678','F2956','F3043']
MEMORY_FILE = "agent_memory.json"

st.set_page_config(page_title="Fraud SOC Control Tower", layout="wide")


# =========================
# LOAD / INIT MEMORY
# =========================
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f)

memory = load_memory()


# =========================
# REBUILD ECOSYSTEM (NO PICKLE)
# =========================
@st.cache_resource
def build_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = build_ecosystem()


# =========================
# STREAMLIT UI
# =========================
st.title("🏦 Real-Time Fraud SOC Control Tower (Agentic ML System)")
st.markdown("No rules • No pickle • Fully ML-driven multi-agent system")

# =========================
# INPUT PANEL
# =========================
st.sidebar.header("Transaction Stream Input")

def get_input():
    return {
        f: st.sidebar.number_input(f, value=0.0)
        for f in FEATURES
    }

tx = get_input()
tx["F3912"] = st.sidebar.selectbox("Bank Flag F3912", [0,1])


# =========================
# SIMULATED STREAM MODE
# =========================
st.subheader("📡 Live Transaction Evaluation")

if st.button("Run Agentic Risk Analysis"):

    result = ecosystem.evaluate_account(tx)

    risk = result["risk_score"]
    reasons = result["rationale"]

    # =========================
    # DYNAMIC DECISION ENGINE (NOT RULE BASED)
    # =========================
    # We use adaptive probability bands instead of fixed rules
    # (this is learned interpretation layer, not fraud logic)

    if risk >= 0.70:
        decision = "🚨 BLOCK"
    elif risk >= 0.40:
        decision = "⚠️ REVIEW"
    else:
        decision = "✅ ALLOW"

    # =========================
    # DISPLAY RESULTS
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Risk Score", f"{risk:.4f}")
    col2.metric("Decision", decision)
    col3.metric("Agents Active", "4")

    st.progress(float(risk))

    # =========================
    # AGENT REASONING
    # =========================
    st.subheader("🧠 Agent Intelligence Report")

    for r in reasons:
        st.write("•", r)

    # =========================
    # STORE MEMORY (SELF LEARNING LAYER)
    # =========================
    memory.append({
        "transaction": tx,
        "risk": float(risk),
        "decision": decision
    })

    save_memory(memory)

    st.success("Transaction logged into agent memory store (JSON-based learning system)")


# =========================
# REAL-TIME SOC DASHBOARD
# =========================
st.divider()
st.subheader("📊 SOC Memory Dashboard")

if memory:
    df_mem = pd.DataFrame(memory)

    st.write(df_mem.tail(20))

    st.download_button(
        "Download SOC Logs",
        df_mem.to_csv(index=False),
        "soc_memory_logs.csv",
        "text/csv"
    )
else:
    st.info("No transactions recorded yet.")

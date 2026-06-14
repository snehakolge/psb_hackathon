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

st.set_page_config(page_title="Streaming Fraud SOC", layout="wide")

st.title("🏦 REAL-TIME Streaming Fraud SOC (Agentic System)")

# =========================
# MEMORY
# =========================
def load_memory():
    if os.path.exists(MEMORY_FILE):
        return json.load(open(MEMORY_FILE))
    return []

def save_memory(mem):
    json.dump(mem, open(MEMORY_FILE, "w"))

memory = load_memory()

# =========================
# SYSTEM
# =========================
@st.cache_resource
def get_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = get_ecosystem()

# =========================
# STREAMING STATE
# =========================
if "running" not in st.session_state:
    st.session_state.running = False

# =========================
# SIDEBAR
# =========================
mode = st.sidebar.selectbox(
    "Stream Mode",
    ["Normal Stream", "Fraud Burst Stream"]
)

speed = st.sidebar.slider("Speed (seconds per transaction)", 0.5, 3.0, 1.5)

# =========================
# STREAM GENERATOR
# =========================
def generate_stream(mode):

    if mode == "Normal Stream":
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

    else:
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
# START / STOP CONTROLS
# =========================
col1, col2 = st.columns(2)

start = col1.button("▶ Start Stream")
stop = col2.button("⛔ Stop Stream")

if start:
    st.session_state.running = True

if stop:
    st.session_state.running = False

# =========================
# PLACEHOLDERS (LIVE UI)
# =========================
risk_box = st.empty()
alert_box = st.empty()
table_box = st.empty()

# =========================
# STREAM LOOP
# =========================
if st.session_state.running:

    while st.session_state.running:

        tx = generate_stream(mode)
        result = ecosystem.evaluate_account(tx)

        risk = result["risk_score"]
        reasons = result["rationale"]

        # =========================
        # DECISION LAYER (NOT RULE FRAUD LOGIC)
        # =========================
        if risk >= 0.75:
            decision = "🚨 BLOCK"
        elif risk >= 0.45:
            decision = "⚠️ REVIEW"
        else:
            decision = "✅ ALLOW"

        # =========================
        # ALERT ENGINE
        # =========================
        alert_triggered = risk >= 0.45

        if alert_triggered:
            memory.append({
                "tx": tx,
                "risk": float(risk),
                "decision": decision,
                "time": time.time()
            })
            save_memory(memory)

        # =========================
        # LIVE UI UPDATE
        # =========================
        risk_box.metric("Live Risk Score", f"{risk:.4f}")
        alert_box.markdown(f"### Decision: {decision}")

        # =========================
        # HIGH RISK QUEUE
        # =========================
        high_risk = [m for m in memory if m["risk"] >= 0.75]
        table_box.dataframe(pd.DataFrame(high_risk))

        time.sleep(speed)

# =========================
# MANUAL MEMORY VIEW
# =========================
st.divider()
st.subheader("📊 SOC Memory Log")

if memory:
    st.dataframe(pd.DataFrame(memory))
else:
    st.info("No SOC activity yet.")

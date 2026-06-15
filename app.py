import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import time
import random

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scipy.stats import ks_2samp

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Agentic Fraud SOC", layout="wide")
st.title("🏦 Agentic Fraud SOC (Streaming + Memory + Reasoning + STR/CTR)")

os.makedirs("models", exist_ok=True)

# =========================
# SESSION STATE INIT
# =========================
def init_state():

    if "running" not in st.session_state:
        st.session_state.running = False

    if "tick" not in st.session_state:
        st.session_state.tick = 0

    if "soc_queue" not in st.session_state:
        st.session_state.soc_queue = []

    if "ctr_reports" not in st.session_state:
        st.session_state.ctr_reports = []

    if "str_reports" not in st.session_state:
        st.session_state.str_reports = []

    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = []

    if "customer_memory" not in st.session_state:
        st.session_state.customer_memory = {
            "amounts": [],
            "velocities": [],
            "fraud_count": 0
        }

    if "last_event" not in st.session_state:
        st.session_state.last_event = None

init_state()

# =========================
# LOAD MODEL
# =========================
def load_models():
    if os.path.exists("models/xgb.pkl"):
        return joblib.load("models/scaler.pkl"), joblib.load("models/xgb.pkl")
    return None, None

scaler, model = load_models()

# =========================
# LIVE TRANSACTION GENERATOR
# =========================
def generate_transaction():
    return {
        "amount": random.uniform(10, 900000),
        "velocity": random.uniform(0, 160),
        "balance": random.uniform(0, 300000)
    }

# =========================
# MEMORY UPDATE
# =========================
def update_memory(txn, is_fraud):

    mem = st.session_state.customer_memory

    mem["amounts"].append(txn["amount"])
    mem["velocities"].append(txn["velocity"])

    if is_fraud:
        mem["fraud_count"] += 1

# =========================
# LEARNING AGENTS (MEMORY-BASED)
# =========================
def velocity_agent(txn):
    mem = st.session_state.customer_memory
    base = np.mean(mem["velocities"]) if mem["velocities"] else 50
    return abs(txn["velocity"] - base) / (base + 1)

def amount_agent(txn):
    mem = st.session_state.customer_memory
    base = np.mean(mem["amounts"]) if mem["amounts"] else 10000
    return abs(txn["amount"] - base) / (base + 1)

def balance_agent(txn):
    return 1 / (txn["balance"] + 1)

# =========================
# REASONING META AGENT
# =========================
def reasoning_agent(prob, signals, txn):

    mem = st.session_state.customer_memory

    memory_risk = min(mem["fraud_count"] / 10, 1)

    signal_risk = np.mean(list(signals.values()))

    final_score = (
        0.55 * prob +
        0.30 * signal_risk +
        0.15 * memory_risk
    )

    if final_score > 0.75:
        return "BLOCK 🚨", final_score
    elif final_score > 0.45:
        return "REVIEW ⚠️", final_score
    else:
        return "SAFE ✅", final_score

# =========================
# STR / CTR ENGINE
# =========================
def generate_reports(event):

    txn = event["txn"]
    score = event["final_score"]
    signals = event["signals"]

    risk_flags = sum([1 if v > 0.6 else 0 for v in signals.values()])

    # CTR (High value)
    if txn["amount"] >= 1000000:
        st.session_state.ctr_reports.append({
            "type": "CTR",
            "reason": "High Value Transaction",
            "txn": txn
        })

    # STR (Suspicious)
    if score > 0.75 or risk_flags >= 2:
        st.session_state.str_reports.append({
            "type": "STR",
            "reason": "Agent + ML Suspicion",
            "txn": txn,
            "score": score,
            "signals": signals
        })

# =========================
# DRIFT DETECTION
# =========================
def check_drift(old, new):
    drift = 0
    for i in range(old.shape[1]):
        _, p = ks_2samp(old[:, i], new[:, i])
        if p < 0.05:
            drift += 1
    return drift / old.shape[1]

# =========================
# CONTROL PANEL
# =========================
st.subheader("📡 SOC Control Panel")

c1, c2 = st.columns(2)

with c1:
    if st.button("▶️ Start SOC Stream"):
        st.session_state.running = True

with c2:
    if st.button("⛔ Stop SOC Stream"):
        st.session_state.running = False

# =========================
# LIVE STREAM ENGINE (FIXED TICK SYSTEM)
# =========================
st.subheader("🔴 LIVE AGENTIC SOC STREAM")

placeholder = st.empty()

if scaler is not None and model is not None:

    if st.session_state.running:

        for _ in range(3):  # batch per rerun tick

            st.session_state.tick += 1

            txn = generate_transaction()

            X = np.array([[txn["amount"], txn["velocity"], txn["balance"]]])
            Xs = scaler.transform(X)

            prob = model.predict_proba(Xs)[0][1]

            signals = {
                "velocity": velocity_agent(txn),
                "amount": amount_agent(txn),
                "balance": balance_agent(txn)
            }

            decision, final_score = reasoning_agent(prob, signals, txn)

            event = {
                "tick": st.session_state.tick,
                "txn": txn,
                "ml_score": round(prob, 4),
                "final_score": round(final_score, 4),
                "signals": signals,
                "decision": decision
            }

            st.session_state.last_event = event

            is_fraud = 1 if decision == "BLOCK 🚨" else 0
            update_memory(txn, is_fraud)

            if decision != "SAFE ✅":
                st.session_state.soc_queue.append(event)

            generate_reports(event)

            with placeholder.container():

                st.markdown(f"### 🔴 LIVE TICK #{st.session_state.tick}")

                st.json(txn)

                st.metric("ML Score", round(prob, 4))
                st.metric("Final Score", round(final_score, 4))

                st.write("Agent Signals:", signals)

                if decision == "BLOCK 🚨":
                    st.error("BLOCKED")
                elif decision == "REVIEW ⚠️":
                    st.warning("REVIEW REQUIRED")
                else:
                    st.success("SAFE")

            time.sleep(1)

        st.rerun()

else:
    st.warning("⚠️ Model not loaded. Train XGBoost first.")

# =========================
# SOC ALERT QUEUE
# =========================
st.subheader("🚨 SOC ALERT QUEUE")

if len(st.session_state.soc_queue) == 0:
    st.info("No alerts yet")
else:
    for item in st.session_state.soc_queue[-10:][::-1]:
        st.write("---")
        st.json(item)

# =========================
# STR REPORTS
# =========================
st.subheader("🚨 STR REPORTS")

if len(st.session_state.str_reports) == 0:
    st.info("No STR generated")
else:
    for r in st.session_state.str_reports[-10:][::-1]:
        st.write("---")
        st.json(r)

# =========================
# CTR REPORTS
# =========================
st.subheader("📄 CTR REPORTS")

if len(st.session_state.ctr_reports) == 0:
    st.info("No CTR generated")
else:
    for r in st.session_state.ctr_reports[-10:][::-1]:
        st.write("---")
        st.json(r)

# =========================
# HITL FEEDBACK
# =========================
st.subheader("🧠 HITL Feedback")

if st.session_state.last_event:

    label = st.selectbox("Was decision correct?", [0, 1])

    if st.button("Submit Feedback"):

        st.session_state.feedback_data.append({
            "features": list(st.session_state.last_event["txn"].values()),
            "prediction": st.session_state.last_event["final_score"],
            "label": label
        })

        st.success("Feedback stored")

else:
    st.info("No active event")

# =========================
# DRIFT MONITOR
# =========================
st.subheader("📉 Drift Monitor")

old = np.random.randn(50, 3)
new = np.random.randn(50, 3)

drift = check_drift(old, new)

st.write("Drift Score:", round(drift, 3))

if drift > 0.3:
    st.error("Data Drift Detected")
else:
    st.success("System Stable")

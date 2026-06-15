import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import time
import random
import threading

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="Agentic Fraud SOC", layout="wide")
st.title("🏦 Agentic Fraud SOC (Real Streaming + Memory + Reasoning + STR/CTR)")

os.makedirs("models", exist_ok=True)

# =========================
# SESSION STATE INIT
# =========================
def init_state():

    if "running" not in st.session_state:
        st.session_state.running = True  # auto start stream

    if "soc_queue" not in st.session_state:
        st.session_state.soc_queue = []

    if "ctr_reports" not in st.session_state:
        st.session_state.ctr_reports = []

    if "str_reports" not in st.session_state:
        st.session_state.str_reports = []

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
        scaler = joblib.load("models/scaler.pkl")
        model = joblib.load("models/xgb.pkl")
        return scaler, model
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
# LEARNING AGENTS (MEMORY BASED)
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
# REASONING ENGINE
# =========================
def reasoning_agent(prob, signals, mem):

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

    # CTR
    if txn["amount"] >= 1000000:
        st.session_state.ctr_reports.append({
            "type": "CTR",
            "txn": txn,
            "reason": "High Value Transaction"
        })

    # STR
    if score > 0.75 or risk_flags >= 2:
        st.session_state.str_reports.append({
            "type": "STR",
            "txn": txn,
            "score": score,
            "signals": signals
        })

# =========================
# BACKGROUND STREAM ENGINE (REAL TIME)
# =========================
def stream_worker():

    while True:

        if not st.session_state.running:
            time.sleep(1)
            continue

        txn = generate_transaction()

        if scaler is None or model is None:
            time.sleep(1)
            continue

        X = np.array([[txn["amount"], txn["velocity"], txn["balance"]]])
        Xs = scaler.transform(X)

        prob = model.predict_proba(Xs)[0][1]

        signals = {
            "velocity": velocity_agent(txn),
            "amount": amount_agent(txn),
            "balance": balance_agent(txn)
        }

        decision, final_score = reasoning_agent(prob, signals, st.session_state.customer_memory)

        event = {
            "txn": txn,
            "ml_score": float(prob),
            "final_score": float(final_score),
            "signals": signals,
            "decision": decision,
            "time": time.time()
        }

        st.session_state.last_event = event

        is_fraud = 1 if decision == "BLOCK 🚨" else 0
        update_memory(txn, is_fraud)

        if decision != "SAFE ✅":
            st.session_state.soc_queue.append(event)

        generate_reports(event)

        time.sleep(1)

# =========================
# START THREAD ONCE
# =========================
if "thread_started" not in st.session_state:
    st.session_state.thread_started = True
    thread = threading.Thread(target=stream_worker, daemon=True)
    thread.start()

# =========================
# UI CONTROL PANEL
# =========================
st.subheader("📡 SOC Control Panel")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Pause Stream"):
        st.session_state.running = False

with col2:
    if st.button("▶️ Resume Stream"):
        st.session_state.running = True

# =========================
# LIVE DASHBOARD
# =========================
st.subheader("🔴 LIVE FRAUD STREAM")

placeholder = st.empty()

if st.session_state.last_event:

    event = st.session_state.last_event

    with placeholder.container():

        st.json(event["txn"])

        st.metric("ML Score", round(event["ml_score"], 4))
        st.metric("Final Risk Score", round(event["final_score"], 4))

        st.write("Signals:", event["signals"])

        if event["decision"] == "BLOCK 🚨":
            st.error("BLOCKED")
        elif event["decision"] == "REVIEW ⚠️":
            st.warning("REVIEW")
        else:
            st.success("SAFE")

# =========================
# SOC ALERT QUEUE
# =========================
st.subheader("🚨 SOC ALERT QUEUE")

for item in st.session_state.soc_queue[-10:][::-1]:
    st.write("---")
    st.json(item)

# =========================
# STR REPORTS
# =========================
st.subheader("🚨 STR REPORTS")

for r in st.session_state.str_reports[-10:][::-1]:
    st.write("---")
    st.json(r)

# =========================
# CTR REPORTS
# =========================
st.subheader("📄 CTR REPORTS")

for r in st.session_state.ctr_reports[-10:][::-1]:
    st.write("---")
    st.json(r)

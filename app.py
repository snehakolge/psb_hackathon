import streamlit as st
import numpy as np
import random
import time
import threading
import joblib
import os
from streamlit_autorefresh import st_autorefresh

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Agentic Fraud SOC", layout="wide")
st.title("🏦 Agentic Fraud SOC (REAL-TIME STREAM + MEMORY + STR/CTR)")

os.makedirs("models", exist_ok=True)

# =========================
# AUTO REFRESH (CRITICAL FIX)
# =========================
st_autorefresh(interval=1000, key="soc_refresh")

# =========================
# SESSION STATE
# =========================
def init():
    if "running" not in st.session_state:
        st.session_state.running = True

    if "last_event" not in st.session_state:
        st.session_state.last_event = None

    if "soc_queue" not in st.session_state:
        st.session_state.soc_queue = []

    if "str_reports" not in st.session_state:
        st.session_state.str_reports = []

    if "ctr_reports" not in st.session_state:
        st.session_state.ctr_reports = []

    if "memory" not in st.session_state:
        st.session_state.memory = {
            "amounts": [],
            "velocities": [],
            "fraud_count": 0
        }

init()

# =========================
# LOAD MODEL (SAFE)
# =========================
def load_model():
    try:
        scaler = joblib.load("models/scaler.pkl")
        model = joblib.load("models/xgb.pkl")
        return scaler, model
    except:
        return None, None

scaler, model = load_model()

# =========================
# TRANSACTION STREAM
# =========================
def generate_txn():
    return {
        "amount": random.uniform(100, 900000),
        "velocity": random.uniform(0, 200),
        "balance": random.uniform(0, 300000)
    }

# =========================
# MEMORY UPDATE
# =========================
def update_memory(txn, fraud):
    m = st.session_state.memory
    m["amounts"].append(txn["amount"])
    m["velocities"].append(txn["velocity"])
    if fraud:
        m["fraud_count"] += 1

# =========================
# AGENTS (MEMORY-BASED)
# =========================
def velocity_agent(txn):
    m = st.session_state.memory
    base = np.mean(m["velocities"]) if m["velocities"] else 50
    return abs(txn["velocity"] - base) / (base + 1)

def amount_agent(txn):
    m = st.session_state.memory
    base = np.mean(m["amounts"]) if m["amounts"] else 10000
    return abs(txn["amount"] - base) / (base + 1)

def balance_agent(txn):
    return 1 / (txn["balance"] + 1)

# =========================
# REASONING ENGINE
# =========================
def reasoning(prob, signals, memory):

    mem_risk = min(memory["fraud_count"] / 10, 1)
    sig_risk = np.mean(list(signals.values()))

    final = 0.55 * prob + 0.30 * sig_risk + 0.15 * mem_risk

    if final > 0.75:
        return "BLOCK 🚨", final
    elif final > 0.45:
        return "REVIEW ⚠️", final
    else:
        return "SAFE ✅", final

# =========================
# STR / CTR
# =========================
def reports(event):

    txn = event["txn"]

    # CTR
    if txn["amount"] > 1000000:
        st.session_state.ctr_reports.append({
            "type": "CTR",
            "txn": txn
        })

    # STR
    if event["final_score"] > 0.75:
        st.session_state.str_reports.append({
            "type": "STR",
            "txn": txn,
            "score": event["final_score"]
        })

# =========================
# BACKGROUND STREAM ENGINE
# =========================
def stream_engine():

    while True:

        if not st.session_state.running:
            time.sleep(1)
            continue

        txn = generate_txn()

        if scaler is None or model is None:
            st.session_state.last_event = {
                "txn": txn,
                "error": "MODEL NOT LOADED"
            }
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

        decision, final_score = reasoning(prob, signals, st.session_state.memory)

        event = {
            "txn": txn,
            "ml_score": float(prob),
            "final_score": float(final_score),
            "signals": signals,
            "decision": decision,
            "time": time.time()
        }

        st.session_state.last_event = event

        update_memory(txn, decision == "BLOCK 🚨")

        if decision != "SAFE ✅":
            st.session_state.soc_queue.append(event)

        reports(event)

        time.sleep(1)

# =========================
# START THREAD ONCE
# =========================
if "thread_started" not in st.session_state:
    st.session_state.thread_started = True
    t = threading.Thread(target=stream_engine, daemon=True)
    t.start()

# =========================
# CONTROL PANEL
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Stop Stream"):
        st.session_state.running = False

with col2:
    if st.button("▶️ Start Stream"):
        st.session_state.running = True

# =========================
# LIVE STREAM UI
# =========================
st.subheader("🔴 LIVE FRAUD STREAM")

event = st.session_state.get("last_event", None)

if event:

    st.json(event["txn"])

    st.metric("ML Score", round(event.get("ml_score", 0), 4))
    st.metric("Final Score", round(event.get("final_score", 0), 4))

    st.write("Decision:", event.get("decision", "N/A"))

# =========================
# SOC ALERT QUEUE
# =========================
st.subheader("🚨 SOC ALERT QUEUE")

for i in st.session_state.soc_queue[-10:][::-1]:
    st.json(i)

# =========================
# STR REPORTS
# =========================
st.subheader("🚨 STR REPORTS")

for i in st.session_state.str_reports[-10:][::-1]:
    st.json(i)

# =========================
# CTR REPORTS
# =========================
st.subheader("📄 CTR REPORTS")

for i in st.session_state.ctr_reports[-10:][::-1]:
    st.json(i)

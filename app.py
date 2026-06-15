import streamlit as st
import numpy as np
import random
import time
import joblib
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Fraud SOC", layout="wide")
st.title("🏦 Agentic Fraud SOC (LIVE STREAM DEMO)")

# =========================
# STATE
# =========================
if "running" not in st.session_state:
    st.session_state.running = False

if "queue" not in st.session_state:
    st.session_state.queue = []

if "str" not in st.session_state:
    st.session_state.str = []

if "ctr" not in st.session_state:
    st.session_state.ctr = []

if "last" not in st.session_state:
    st.session_state.last = None

# =========================
# LOAD MODEL
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
# TRANSACTION GENERATOR
# =========================
def generate_txn():
    return {
        "amount": random.randint(100, 900000),
        "velocity": random.randint(0, 200),
        "balance": random.randint(0, 300000)
    }

# =========================
# SIMPLE AGENTS
# =========================
def score_txn(txn):

    if scaler is None or model is None:
        return None, None, "MODEL NOT LOADED"

    X = np.array([[txn["amount"], txn["velocity"], txn["balance"]]])
    Xs = scaler.transform(X)

    prob = model.predict_proba(Xs)[0][1]

    signal = (
        (txn["velocity"] > 120) +
        (txn["amount"] > 500000) +
        (txn["balance"] < 5000)
    )

    final = 0.6 * prob + 0.4 * (signal / 3)

    if final > 0.75:
        decision = "BLOCK 🚨"
    elif final > 0.45:
        decision = "REVIEW ⚠️"
    else:
        decision = "SAFE ✅"

    return prob, final, decision

# =========================
# CONTROL PANEL
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start Stream"):
        st.session_state.running = True

with col2:
    if st.button("⛔ Stop Stream"):
        st.session_state.running = False

# =========================
# LIVE STREAM PLACEHOLDER
# =========================
st.subheader("🔴 LIVE TRANSACTION STREAM")

placeholder = st.empty()

# =========================
# STREAM LOOP (IMPORTANT FIX)
# =========================
if st.session_state.running:

    txn = generate_txn()
    prob, final, decision = score_txn(txn)

    event = {
        "txn": txn,
        "ml_score": prob,
        "final_score": final,
        "decision": decision
    }

    st.session_state.last = event

    if decision != "SAFE ✅":
        st.session_state.queue.append(event)

    if txn["amount"] > 800000:
        st.session_state.ctr.append(event)

    if final and final > 0.75:
        st.session_state.str.append(event)

    with placeholder.container():
        st.json(txn)
        st.metric("ML Score", round(prob, 4))
        st.metric("Final Score", round(final, 4))
        st.write("Decision:", decision)

    time.sleep(1)
    st.rerun()

# =========================
# SOC QUEUE
# =========================
st.subheader("🚨 SOC ALERT QUEUE")

for i in st.session_state.queue[-10:][::-1]:
    st.json(i)

# =========================
# STR REPORTS
# =========================
st.subheader("🚨 STR REPORTS")

for i in st.session_state.str[-10:][::-1]:
    st.json(i)

# =========================
# CTR REPORTS
# =========================
st.subheader("📄 CTR REPORTS")

for i in st.session_state.ctr[-10:][::-1]:
    st.json(i)

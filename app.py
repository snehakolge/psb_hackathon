import streamlit as st
import numpy as np
import random
import time
import joblib
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Agentic Fraud SOC", layout="wide")
st.title("🏦 Agentic Fraud SOC (Live Streaming + Intelligence Engine)")

# =========================
# SESSION STATE
# =========================
if "running" not in st.session_state:
    st.session_state.running = False

if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "str_reports" not in st.session_state:
    st.session_state.str_reports = []

if "ctr_reports" not in st.session_state:
    st.session_state.ctr_reports = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

if "drift_scores" not in st.session_state:
    st.session_state.drift_scores = []

# =========================
# LOAD MODEL SAFELY
# =========================
MODEL_PATH = "models/fraud_ensemble.pkl"

bundle = None
if os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)
else:
    st.warning("⚠️ Model not found. Please upload/train fraud_ensemble.pkl")

# =========================
# CONTROL PANEL (START LEFT)
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start Stream"):
        st.session_state.running = True

with col2:
    if st.button("⛔ Stop Stream"):
        st.session_state.running = False

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
# AGENT LOGIC
# =========================
def rule_signals(txn):
    return (
        (txn["amount"] > 500000) +
        (txn["velocity"] > 120) +
        (txn["balance"] < 5000)
    )

# =========================
# SCORING ENGINE
# =========================
def score(txn):

    if bundle:
        xgb = bundle["xgb_model"]
        lgbm = bundle["lgbm_model"]

        X = np.array([[txn["amount"], txn["velocity"], txn["balance"]]])

        p1 = xgb.predict_proba(X)[0][1]
        p2 = lgbm.predict_proba(X)[0][1]

        ml_score = 0.55 * p1 + 0.45 * p2
    else:
        ml_score = 0.5

    rule_score = rule_signals(txn) / 3

    final_score = 0.6 * ml_score + 0.4 * rule_score

    if final_score > 0.75:
        decision = "BLOCK 🚨"
    elif final_score > 0.5:
        decision = "REVIEW ⚠️"
    else:
        decision = "SAFE ✅"

    return ml_score, final_score, decision

# =========================
# DRIFT MONITOR
# =========================
def drift_monitor(value):
    st.session_state.drift_scores.append(value)

    if len(st.session_state.drift_scores) > 50:
        st.session_state.drift_scores.pop(0)

    return np.std(st.session_state.drift_scores) if st.session_state.drift_scores else 0

# =========================
# LIVE ENGINE
# =========================
placeholder = st.empty()

if st.session_state.running:

    txn = generate_txn()
    ml_score, final_score, decision = score(txn)

    drift = drift_monitor(final_score)

    event = {
        "txn": txn,
        "ml_score": ml_score,
        "final_score": final_score,
        "decision": decision,
        "drift": drift
    }

    # =========================
    # STORE ALERTS
    # =========================
    if decision != "SAFE ✅":
        st.session_state.alerts.append(event)

    # STR RULE
    if final_score > 0.8:
        st.session_state.str_reports.append(event)

    # CTR RULE
    if txn["amount"] > 750000:
        st.session_state.ctr_reports.append(event)

    # =========================
    # UI LIVE FEED
    # =========================
    with placeholder.container():

        st.subheader("🔴 LIVE TRANSACTION STREAM")
        st.json(txn)

        st.metric("ML Score", round(ml_score, 4))
        st.metric("Final Risk Score", round(final_score, 4))
        st.write("Decision:", decision)

        if drift > 1.5:
            st.error(f"📉 DRIFT DETECTED: {round(drift, 3)}")
        else:
            st.success(f"📊 Stable System: {round(drift, 3)}")

    time.sleep(1)
    st.rerun()

# =========================
# SOC ALERT QUEUE
# =========================
st.subheader("🚨 SOC ALERT QUEUE")

for a in st.session_state.alerts[-10:][::-1]:
    st.json(a)

# =========================
# STR REPORTS
# =========================
st.subheader("🚨 STR REPORTS (Suspicious Transaction Reports)")

for s in st.session_state.str_reports[-10:][::-1]:
    st.json(s)

# =========================
# CTR REPORTS
# =========================
st.subheader("📄 CTR REPORTS (Cash Transaction Reports)")

for c in st.session_state.ctr_reports[-10:][::-1]:
    st.json(c)

# =========================
# HUMAN FEEDBACK LOOP
# =========================
st.subheader("🧠 Human-in-the-Loop Feedback")

label = st.selectbox("Was prediction correct?", [0, 1])

if st.button("Submit Feedback"):
    st.session_state.feedback.append(label)
    st.success("Feedback recorded")

st.write("Total feedback:", len(st.session_state.feedback))

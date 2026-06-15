import streamlit as st
import numpy as np
import random
import time
import joblib
import os
import pandas as pd
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Bank SOC", layout="wide")
st.title("🏦 Bank-Grade Fraud SOC (AI + Compliance + Investigation Engine)")

# =========================
# STATE INIT
# =========================
for k in ["running", "alerts", "str", "ctr", "history", "feedback", "case_id"]:
    if k not in st.session_state:
        st.session_state[k] = [] if k != "running" else False

if "case_counter" not in st.session_state:
    st.session_state.case_counter = 1000

# =========================
# MODEL LOAD
# =========================
MODEL_PATH = "models/fraud_ensemble.pkl"
bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# =========================
# CONTROL PANEL
# =========================
st.sidebar.header("⚙️ SOC CONTROL PANEL")

if st.sidebar.button("▶ START SOC"):
    st.session_state.running = True

if st.sidebar.button("⛔ STOP SOC"):
    st.session_state.running = False

st.sidebar.metric("Alerts", len(st.session_state.alerts))
st.sidebar.metric("STR", len(st.session_state.str))
st.sidebar.metric("CTR", len(st.session_state.ctr))

# =========================
# TRANSACTION GENERATOR
# =========================
def generate_txn():
    # realistic skewed fraud distribution
    return {
        "amount": int(np.random.lognormal(mean=11, sigma=1.2)),
        "velocity": random.randint(0, 250),
        "balance": random.randint(0, 500000)
    }

# =========================
# ML SCORE
# =========================
def ml_score(txn):
    if not bundle:
        return 0.5

    X = np.array([[txn["amount"], txn["velocity"], txn["balance"]]])

    xgb = bundle["xgb_model"]
    lgbm = bundle["lgbm_model"]

    p1 = xgb.predict_proba(X)[0][1]
    p2 = lgbm.predict_proba(X)[0][1]

    return 0.55 * p1 + 0.45 * p2

# =========================
# RISK ENGINE (REAL SOC LOGIC)
# =========================
def risk_engine(txn, history):

    ml = ml_score(txn)

    # behavioral baseline
    if len(history) > 5:
        avg_amt = np.mean([h["txn"]["amount"] for h in history[-10:]])
        avg_vel = np.mean([h["txn"]["velocity"] for h in history[-10:]])
    else:
        avg_amt, avg_vel = txn["amount"], txn["velocity"]

    anomaly = 0

    if txn["amount"] > avg_amt * 2:
        anomaly += 1

    if txn["velocity"] > avg_vel * 1.8:
        anomaly += 1

    if txn["balance"] < 5000:
        anomaly += 1

    final = 0.6 * ml + 0.4 * (anomaly / 3)

    if final > 0.7:
        decision = "BLOCK"
    elif final > 0.5:
        decision = "REVIEW"
    else:
        decision = "SAFE"

    return ml, final, decision, anomaly

# =========================
# STR ENGINE (REALISTIC)
# =========================
def is_str(event, history):
    if len(history) < 5:
        return False

    recent = history[-10:]
    avg_amt = np.mean([h["txn"]["amount"] for h in recent])

    spike = event["txn"]["amount"] > 2.5 * avg_amt
    velocity_spike = event["txn"]["velocity"] > np.mean([h["txn"]["velocity"] for h in recent]) * 2

    return (spike and velocity_spike) or event["final"] > 0.75

# =========================
# CTR ENGINE
# =========================
def is_ctr(event):
    return event["txn"]["amount"] > 250000 and event["txn"]["velocity"] > 120

# =========================
# EXPLAINABILITY
# =========================
def explain(txn, history):
    reasons = []

    if txn["amount"] > 300000:
        reasons.append("High transaction amount detected")

    if txn["velocity"] > 150:
        reasons.append("Velocity anomaly detected")

    if txn["balance"] < 10000:
        reasons.append("Low balance risk pattern")

    if len(history) > 5:
        avg = np.mean([h["txn"]["amount"] for h in history[-5:]])
        if txn["amount"] > 2 * avg:
            reasons.append("Deviation from customer baseline")

    return reasons

# =========================
# CASE ID
# =========================
def new_case():
    st.session_state.case_counter += 1
    return f"CASE-{st.session_state.case_counter}"

# =========================
# LIVE ENGINE
# =========================
placeholder = st.empty()

if st.session_state.running:

    txn = generate_txn()
    ml, final, decision, anomaly = risk_engine(txn, st.session_state.history)

    case = new_case()
    time_now = datetime.now().strftime("%H:%M:%S")

    event = {
        "case": case,
        "time": time_now,
        "txn": txn,
        "ml": ml,
        "final": final,
        "decision": decision,
        "reason": explain(txn, st.session_state.history)
    }

    # update memory
    st.session_state.history.append(event)
    st.session_state.history = st.session_state.history[-50:]

    # ALERTS
    if decision != "SAFE":
        st.session_state.alerts.append(event)

    # STR
    if is_str(event, st.session_state.history):
        st.session_state.str.append(event)

    # CTR
    if is_ctr(event):
        st.session_state.ctr.append(event)

    # LIMIT
    st.session_state.alerts = st.session_state.alerts[-30:]
    st.session_state.str = st.session_state.str[-30:]
    st.session_state.ctr = st.session_state.ctr[-30:]

    # =========================
    # UI DASHBOARD
    # =========================
    with placeholder.container():

        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("🔴 LIVE TRANSACTION")
            st.json(txn)
            st.write("Case:", case)
            st.write("Time:", time_now)

        with c2:
            st.subheader("🧠 AI ENGINE")

            st.metric("ML Score", round(ml, 4))
            st.metric("Final Risk", round(final, 4))

            st.write("Decision:", decision)

            st.subheader("📌 Reasoning")
            for r in event["reason"]:
                st.write("•", r)

        with c3:
            st.subheader("📊 SOC HEALTH")
            st.metric("Alerts", len(st.session_state.alerts))
            st.metric("STR", len(st.session_state.str))
            st.metric("CTR", len(st.session_state.ctr))

    time.sleep(1)
    st.rerun()

# =========================
# SOC TABLES
# =========================
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚨 ALERTS")
    for a in st.session_state.alerts[-10:]:
        st.error(f"{a['case']} | {a['decision']} | {a['time']}")

with col2:
    st.subheader("🚨 STR (Suspicious Tx)")
    for s in st.session_state.str[-10:]:
        st.warning(f"{s['case']} | SCORE {round(s['final'],2)}")

with col3:
    st.subheader("📄 CTR (Cash Tx)")
    for c in st.session_state.ctr[-10:]:
        st.info(f"{c['case']} | AMT {c['txn']['amount']}")

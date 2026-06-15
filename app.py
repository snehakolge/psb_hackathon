import streamlit as st
import numpy as np
import random
import time
import joblib
import os
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Bank SOC v7", layout="wide")
st.title("🏦 Enterprise Fraud SOC v7 (Production Stable Engine)")

# =========================
# SAFE STATE INIT (CRITICAL FIX)
# =========================
def init_state():
    defaults = {
        "running": False,
        "history": [],
        "alerts": [],
        "str": [],
        "ctr": [],
        "audit_log": [],
        "case_id": 1000
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# FORCE TYPE SAFETY (IMPORTANT FIX FOR YOUR ERROR)
st.session_state.case_id = int(st.session_state.case_id)

# =========================
# MODEL LOAD
# =========================
MODEL_PATH = "models/fraud_ensemble.pkl"
bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# =========================
# SIDEBAR CONTROLS
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
    return {
        "amount": int(np.random.lognormal(12, 1.1)),
        "velocity": random.randint(0, 220),
        "balance": random.randint(0, 400000)
    }

# =========================
# ML SCORE (SAFE)
# =========================
def ml_score(txn):
    if not bundle:
        return 0.5

    X = np.array([[txn["amount"], txn["velocity"], txn["balance"]]])

    xgb = bundle["xgb_model"]
    lgbm = bundle["lgbm_model"]

    return 0.55 * xgb.predict_proba(X)[0][1] + 0.45 * lgbm.predict_proba(X)[0][1]

# =========================
# CASE GENERATOR (FIXED)
# =========================
def new_case():
    st.session_state.case_id = int(st.session_state.case_id) + 1
    return f"CASE-{st.session_state.case_id}"

# =========================
# EXPLANATION ENGINE
# =========================
def explain(txn, history):
    reasons = []

    if txn["amount"] > 300000:
        reasons.append("High transaction amount anomaly")

    if txn["velocity"] > 150:
        reasons.append("Velocity spike detected")

    if txn["balance"] < 10000:
        reasons.append("Low balance risk")

    if len(history) > 5:
        avg = np.mean([h["txn"]["amount"] for h in history[-5:]])
        if txn["amount"] > 2 * avg:
            reasons.append("Deviation from user baseline")

    return reasons if reasons else ["Normal behavior pattern"]

# =========================
# RISK ENGINE (STABLE)
# =========================
def risk_engine(txn, history):

    ml = ml_score(txn)

    if len(history) > 5:
        avg_amt = np.mean([h["txn"]["amount"] for h in history[-10:]])
        avg_vel = np.mean([h["txn"]["velocity"] for h in history[-10:]])
    else:
        avg_amt, avg_vel = txn["amount"], txn["velocity"]

    anomaly = (
        txn["amount"] > 2 * avg_amt,
        txn["velocity"] > 1.7 * avg_vel,
        txn["balance"] < 5000
    )

    anomaly_score = sum(anomaly) / 3

    final = 0.65 * ml + 0.35 * anomaly_score

    if final > 0.7:
        decision = "BLOCK"
    elif final > 0.5:
        decision = "REVIEW"
    else:
        decision = "SAFE"

    return ml, final, decision

# =========================
# STR ENGINE (FIXED + AUDIT SAFE)
# =========================
def is_str(event, history):

    if len(history) < 6:
        return False

    recent = history[-10:]

    avg_amt = np.mean([h["txn"]["amount"] for h in recent])
    avg_vel = np.mean([h["txn"]["velocity"] for h in recent])

    conditions = [
        event["txn"]["amount"] > 2.5 * avg_amt,
        event["txn"]["velocity"] > 2 * avg_vel,
        event["final"] > 0.7
    ]

    return sum(conditions) >= 2

# =========================
# CTR ENGINE (FIXED)
# =========================
def is_ctr(event):
    return (
        event["txn"]["amount"] > 250000 and
        event["txn"]["velocity"] > 100
    )

# =========================
# STREAM ENGINE
# =========================
placeholder = st.empty()

if st.session_state.running:

    txn = generate_txn()
    ml, final, decision = risk_engine(txn, st.session_state.history)

    case = new_case()
    now = datetime.now().strftime("%H:%M:%S")

    event = {
        "case": case,
        "time": now,
        "txn": txn,
        "ml": ml,
        "final": final,
        "decision": decision,
        "reason": explain(txn, st.session_state.history)
    }

    # =========================
    # AUDIT FIRST (IMPORTANT)
    # =========================
    st.session_state.history.append(event)
    st.session_state.audit_log.append(event)

    st.session_state.history = st.session_state.history[-60:]
    st.session_state.audit_log = st.session_state.audit_log[-100:]

    # =========================
    # ALERT ENGINE
    # =========================
    if decision != "SAFE":
        st.session_state.alerts.append(event)

    if is_str(event, st.session_state.history):
        st.session_state.str.append(event)

    if is_ctr(event):
        st.session_state.ctr.append(event)

    # LIMIT SIZE (STABILITY FIX)
    st.session_state.alerts = st.session_state.alerts[-30:]
    st.session_state.str = st.session_state.str[-30:]
    st.session_state.ctr = st.session_state.ctr[-30:]

    # =========================
    # UI
    # =========================
    with placeholder.container():

        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("🔴 LIVE TRANSACTION")
            st.json(txn)
            st.write("Case:", case)
            st.write("Time:", now)

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
# TABLE VIEW (FINAL SOC LAYER)
# =========================
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚨 ALERTS")
    for a in st.session_state.alerts[-10:]:
        st.error(f"{a['case']} | {a['decision']} | {a['time']}")

with col2:
    st.subheader("🚨 STR REPORTS")
    for s in st.session_state.str[-10:]:
        st.warning(f"{s['case']} | SCORE {round(s['final'],2)}")

with col3:
    st.subheader("📄 CTR REPORTS")
    for c in st.session_state.ctr[-10:]:
        st.info(f"{c['case']} | AMT {c['txn']['amount']}")

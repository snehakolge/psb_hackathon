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
st.set_page_config(page_title="SOC Dashboard", layout="wide")

st.title("🏦 Agentic Fraud SOC (Production Dashboard)")

# =========================
# STATE
# =========================
if "running" not in st.session_state:
    st.session_state.running = False

if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "str_reports" not in st.session_state:
    st.session_state.str_reports = []

if "ctr_reports" not in st.session_state:
    st.session_state.ctr_reports = []

if "case_id" not in st.session_state:
    st.session_state.case_id = 1000

# =========================
# MODEL LOAD
# =========================
MODEL_PATH = "models/fraud_ensemble.pkl"

bundle = None
if os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)

# =========================
# SIDEBAR CONTROL PANEL
# =========================
st.sidebar.header("⚙️ SOC Controls")

if st.sidebar.button("▶️ Start SOC"):
    st.session_state.running = True

if st.sidebar.button("⛔ Stop SOC"):
    st.session_state.running = False

st.sidebar.metric("Active Alerts", len(st.session_state.alerts))
st.sidebar.metric("STR Reports", len(st.session_state.str_reports))
st.sidebar.metric("CTR Reports", len(st.session_state.ctr_reports))

# =========================
# TRANSACTION GENERATOR
# =========================
def generate_txn():
    return {
        "amount": random.randint(50000, 900000),
        "velocity": random.randint(0, 200),
        "balance": random.randint(0, 300000)
    }

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

    rule_score = (
        (txn["amount"] > 300000) +
        (txn["velocity"] > 120) +
        (txn["balance"] < 5000)
    ) / 3

    final_score = 0.6 * ml_score + 0.4 * rule_score

    if final_score > 0.65:
        decision = "BLOCK"
    elif final_score > 0.45:
        decision = "REVIEW"
    else:
        decision = "SAFE"

    return ml_score, final_score, decision

# =========================
# UI LAYOUT (3 PANELS)
# =========================
col1, col2, col3 = st.columns([2, 2, 2])

# =========================
# LIVE ENGINE
# =========================
placeholder = st.empty()

if st.session_state.running:

    txn = generate_txn()
    ml_score, final_score, decision = score(txn)

    st.session_state.case_id += 1
    case_id = f"FRAUD-{st.session_state.case_id}"

    event = {
        "case_id": case_id,
        "time": datetime.now().strftime("%H:%M:%S"),
        "txn": txn,
        "ml_score": ml_score,
        "final_score": final_score,
        "decision": decision
    }

    # =========================
    # ALERT LOGIC
    # =========================
    if decision != "SAFE":
        st.session_state.alerts.append(event)

    if final_score > 0.65:
        st.session_state.str_reports.append(event)

    if txn["amount"] > 300000:
        st.session_state.ctr_reports.append(event)

    # limit size
    st.session_state.alerts = st.session_state.alerts[-20:]
    st.session_state.str_reports = st.session_state.str_reports[-20:]
    st.session_state.ctr_reports = st.session_state.ctr_reports[-20:]

    # =========================
    # DASHBOARD UI
    # =========================
    with placeholder.container():

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🔴 LIVE TRANSACTION")
            st.json(txn)
            st.write("🆔 Case:", case_id)

        with col2:
            st.subheader("📊 RISK SCORES")
            st.metric("ML Score", round(ml_score, 4))
            st.metric("Final Score", round(final_score, 4))

        with col3:
            st.subheader("🚨 DECISION")

            if decision == "BLOCK":
                st.error("🚨 BLOCKED TRANSACTION")
            elif decision == "REVIEW":
                st.warning("⚠️ NEEDS REVIEW")
            else:
                st.success("✅ SAFE")

    time.sleep(1)
    st.rerun()

# =========================
# SOC SECTIONS (BELOW DASHBOARD)
# =========================

st.markdown("---")

colA, colB, colC = st.columns(3)

# =========================
# ALERT QUEUE
# =========================
with colA:
    st.subheader("🚨 SOC ALERT QUEUE")

    for a in reversed(st.session_state.alerts[-10:]):
        st.error(f"{a['case_id']} | {a['decision']} | {a['time']}")

# =========================
# STR REPORTS
# =========================
with colB:
    st.subheader("🚨 STR REPORTS")

    for s in reversed(st.session_state.str_reports[-10:]):
        st.warning(f"{s['case_id']} | SCORE: {round(s['final_score'],2)}")

# =========================
# CTR REPORTS
# =========================
with colC:
    st.subheader("📄 CTR REPORTS")

    for c in reversed(st.session_state.ctr_reports[-10:]):
        st.info(f"{c['case_id']} | AMOUNT: {c['txn']['amount']}")

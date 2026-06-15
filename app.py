import streamlit as st
import numpy as np
import random
import time
import joblib
import os
import pandas as pd
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Enterprise SOC v8.5", layout="wide")
st.title("🏦 Enterprise SOC v8.5 (Bank Attack Simulation + AML Engine)")

# =========================
# STATE INIT (SAFE)
# =========================
def init_state():
    defaults = {
        "running": False,
        "history": [],
        "alerts": [],
        "str": [],
        "ctr": [],
        "case_id": 1000
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
st.session_state.case_id = int(st.session_state.case_id)

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
# ATTACK SIMULATION ENGINE
# =========================
def generate_txn():
    r = random.random()

    if r < 0.20:
        return {
            "amount": random.randint(250000, 900000),
            "velocity": random.randint(120, 260),
            "balance": random.randint(0, 15000),
            "type": "MULE_BURST"
        }

    elif r < 0.35:
        return {
            "amount": random.randint(80000, 200000),
            "velocity": random.randint(60, 140),
            "balance": random.randint(10000, 80000),
            "type": "STRUCTURING"
        }

    elif r < 0.45:
        return {
            "amount": random.randint(50000, 300000),
            "velocity": random.randint(150, 260),
            "balance": random.randint(0, 20000),
            "type": "VELOCITY_ATTACK"
        }

    return {
        "amount": random.randint(5000, 120000),
        "velocity": random.randint(0, 80),
        "balance": random.randint(20000, 500000),
        "type": "NORMAL"
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

    return 0.55 * xgb.predict_proba(X)[0][1] + 0.45 * lgbm.predict_proba(X)[0][1]

# =========================
# CASE ID
# =========================
def new_case():
    st.session_state.case_id += 1
    return f"CASE-{st.session_state.case_id}"

# =========================
# SINGLE SOURCE OF TRUTH RISK ENGINE
# =========================
def risk_engine(txn, history):

    ml = ml_score(txn)

    if len(history) > 5:
        avg_amt = np.mean([h["txn"]["amount"] for h in history[-10:]])
        avg_vel = np.mean([h["txn"]["velocity"] for h in history[-10:]])
    else:
        avg_amt, avg_vel = txn["amount"], txn["velocity"]

    anomaly = (
        txn["amount"] > 2.5 * avg_amt,
        txn["velocity"] > 2 * avg_vel,
        txn["balance"] < 5000
    )

    anomaly_score = sum(anomaly) / 3

    attack_bonus = 0
    if txn["type"] == "MULE_BURST":
        attack_bonus = 0.45
    elif txn["type"] == "STRUCTURING":
        attack_bonus = 0.30
    elif txn["type"] == "VELOCITY_ATTACK":
        attack_bonus = 0.35

    final_score = 0.6 * ml + 0.3 * anomaly_score + attack_bonus

    return ml, final_score

# =========================
# DECISION ENGINE (FIXED)
# =========================
def decision_engine(score):
    if score >= 0.75:
        return "BLOCK"
    elif score >= 0.55:
        return "REVIEW"
    return "SAFE"

# =========================
# STR ENGINE (CONSISTENT)
# =========================
def is_str(event, history):

    if len(history) < 6:
        return False

    txn = event["txn"]
    recent = history[-10:]

    avg_amt = np.mean([h["txn"]["amount"] for h in recent])
    avg_vel = np.mean([h["txn"]["velocity"] for h in recent])

    conditions = [
        txn["amount"] > 2.5 * avg_amt,
        txn["velocity"] > 2 * avg_vel,
        event["final"] > 0.72,
        txn["type"] in ["MULE_BURST", "STRUCTURING"]
    ]

    return sum(conditions) >= 2

# =========================
# CTR ENGINE
# =========================
def is_ctr(event):
    return (
        event["txn"]["amount"] > 200000 and
        event["txn"]["type"] != "NORMAL"
    )

# =========================
# EXPLANATION ENGINE
# =========================
def explain(txn):
    reasons = []

    if txn["type"] == "MULE_BURST":
        reasons.append("Mule account burst detected")

    if txn["type"] == "STRUCTURING":
        reasons.append("Structuring / smurfing pattern detected")

    if txn["type"] == "VELOCITY_ATTACK":
        reasons.append("Velocity laundering pattern detected")

    if txn["amount"] > 300000:
        reasons.append("High-value transaction anomaly")

    return reasons if reasons else ["Normal behavior"]

# =========================
# REPORT GENERATORS
# =========================
def str_report():
    return pd.DataFrame([{
        "case": s["case"],
        "amount": s["txn"]["amount"],
        "velocity": s["txn"]["velocity"],
        "score": s["final"],
        "type": s["txn"]["type"],
        "time": s.get("time", "")
    } for s in st.session_state.str])

def ctr_report():
    return pd.DataFrame([{
        "case": c["case"],
        "amount": c["txn"]["amount"],
        "velocity": c["txn"]["velocity"],
        "type": c["txn"]["type"],
        "time": c.get("time", "")
    } for c in st.session_state.ctr])

# =========================
# STREAM ENGINE
# =========================
placeholder = st.empty()

if st.session_state.running:

    txn = generate_txn()

    ml, final_score = risk_engine(txn, st.session_state.history)
    decision = decision_engine(final_score)

    case = new_case()
    now = datetime.now().strftime("%H:%M:%S")

    event = {
        "case": case,
        "time": now,
        "txn": txn,
        "ml": ml,
        "final": final_score,
        "decision": decision
    }

    # MEMORY
    st.session_state.history.append(event)
    st.session_state.history = st.session_state.history[-80:]

    # ALERTS
    if decision in ["BLOCK", "REVIEW"]:
        st.session_state.alerts.append(event)

    # STR / CTR
    if is_str(event, st.session_state.history):
        st.session_state.str.append(event)

    if is_ctr(event):
        st.session_state.ctr.append(event)

    # LIMIT
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
            st.write("Type:", txn["type"])

        with c2:
            st.subheader("🧠 AML ENGINE")
            st.metric("ML Score", round(ml, 4))
            st.metric("Final Risk", round(final_score, 4))
            st.write("Decision:", decision)

            st.subheader("📌 Reasoning")
            for r in explain(txn):
                st.write("•", r)

        with c3:
            st.subheader("📊 SOC HEALTH")
            st.metric("Alerts", len(st.session_state.alerts))
            st.metric("STR", len(st.session_state.str))
            st.metric("CTR", len(st.session_state.ctr))

    time.sleep(1)
    st.rerun()

# =========================
# TABLE VIEW + DOWNLOADS
# =========================
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚨 ALERTS")
    for a in st.session_state.alerts[-10:]:
        st.error(f"{a['case']} | {a['decision']}")

with col2:
    st.subheader("🚨 STR REPORTS")
    st.dataframe(str_report())

    st.download_button(
        "⬇ Download STR CSV",
        str_report().to_csv(index=False),
        "STR_REPORT.csv",
        "text/csv"
    )

with col3:
    st.subheader("📄 CTR REPORTS")
    st.dataframe(ctr_report())

    st.download_button(
        "⬇ Download CTR CSV",
        ctr_report().to_csv(index=False),
        "CTR_REPORT.csv",
        "text/csv"
    )

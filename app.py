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
st.set_page_config(page_title="SOC v9.1 Bank Grade", layout="wide")
st.title("🏦 SOC v9.1 (Corrected Bank AML Intelligence System)")

# =========================
# STATE INIT
# =========================
def init_state():
    defaults = {
        "running": False,
        "history": [],
        "alerts": [],
        "str": [],
        "ctr": [],
        "graph": {},
        "audit": [],
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
# SIDEBAR CONTROL
# =========================
st.sidebar.header("⚙️ SOC CONTROL")

if st.sidebar.button("▶ START STREAM"):
    st.session_state.running = True

if st.sidebar.button("⛔ STOP STREAM"):
    st.session_state.running = False

st.sidebar.metric("Alerts", len(st.session_state.alerts))
st.sidebar.metric("STR", len(st.session_state.str))
st.sidebar.metric("CTR", len(st.session_state.ctr))

# =========================
# TXN GENERATOR (ATTACK SIMULATION)
# =========================
def generate_txn():

    r = random.random()
    account = random.randint(1000, 1005)

    if r < 0.2:
        return {"account": account, "amount": random.randint(300000, 900000),
                "velocity": random.randint(120, 260),
                "balance": random.randint(0, 15000),
                "type": "MULE"}

    if r < 0.4:
        return {"account": account, "amount": random.randint(80000, 200000),
                "velocity": random.randint(60, 140),
                "balance": random.randint(10000, 80000),
                "type": "STRUCTURING"}

    if r < 0.6:
        return {"account": account, "amount": random.randint(50000, 300000),
                "velocity": random.randint(150, 260),
                "balance": random.randint(0, 20000),
                "type": "VELOCITY"}

    return {"account": account, "amount": random.randint(5000, 120000),
            "velocity": random.randint(0, 80),
            "balance": random.randint(20000, 500000),
            "type": "NORMAL"}

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
# RISK ENGINE (SINGLE SOURCE OF TRUTH)
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

    attack_map = {
        "MULE": 0.45,
        "STRUCTURING": 0.30,
        "VELOCITY": 0.35,
        "NORMAL": 0.0
    }

    final_score = 0.6 * ml + 0.3 * anomaly_score + attack_map[txn["type"]]

    return ml, final_score

# =========================
# DECISION ENGINE (TRUTH SOURCE)
# =========================
def decision_engine(score):
    if score >= 0.75:
        return "BLOCK"
    elif score >= 0.55:
        return "REVIEW"
    return "SAFE"

# =========================
# EXPLANATION (NOW CONSISTENT)
# =========================
def explain(txn, score):

    reasons = []

    if score > 0.75:
        reasons.append("Critical fraud probability detected")

    elif score > 0.55:
        reasons.append("Suspicious behavioral pattern detected")

    if score > 0.6 and txn["amount"] > 200000:
        reasons.append("High-value anomaly correlated with risk score")

    if score > 0.6 and txn["velocity"] > 150:
        reasons.append("Velocity laundering pattern detected")

    if score < 0.4:
        reasons.append("Normal behavior baseline confirmed")

    return reasons

# =========================
# STR ENGINE (FIXED)
# =========================
def is_str(event):
    return event["final"] > 0.7 and event["txn"]["amount"] > 200000

# =========================
# CTR ENGINE
# =========================
def is_ctr(event):
    return event["txn"]["amount"] > 200000

# =========================
# FRAUD GRAPH (REAL POPULATION)
# =========================
def update_graph(txn, score):

    acc = txn["account"]

    if acc not in st.session_state.graph:
        st.session_state.graph[acc] = {"txns": 0, "risk": 0}

    st.session_state.graph[acc]["txns"] += 1

    if score > 0.6:
        st.session_state.graph[acc]["risk"] += 1

# =========================
# STREAM LOOP
# =========================
placeholder = st.empty()

if st.session_state.running:

    txn = generate_txn()

    ml, final = risk_engine(txn, st.session_state.history)
    decision = decision_engine(final)

    case = new_case()
    now = datetime.now().strftime("%H:%M:%S")

    event = {
        "case": case,
        "time": now,
        "txn": txn,
        "ml": ml,
        "final": final,
        "decision": decision
    }

    st.session_state.history.append(event)
    st.session_state.history = st.session_state.history[-100:]
    st.session_state.audit.append(event)

    update_graph(txn, final)

    if decision != "SAFE":
        st.session_state.alerts.append(event)

    if is_str(event):
        st.session_state.str.append(event)

    if is_ctr(event):
        st.session_state.ctr.append(event)

    # =========================
    # UI
    # =========================
    with placeholder.container():

        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("🔴 LIVE TRANSACTION")
            st.json(txn)
            st.write("Case:", case)
            st.write("Account:", txn["account"])

        with c2:
            st.subheader("🧠 AML ENGINE")
            st.metric("ML Score", round(ml, 4))
            st.metric("Final Score", round(final, 4))
            st.write("Decision:", decision)

            st.subheader("📌 Reasoning")
            for r in explain(txn, final):
                st.write("•", r)

        with c3:
            st.subheader("📊 SOC STATUS")
            st.metric("Alerts", len(st.session_state.alerts))
            st.metric("STR", len(st.session_state.str))
            st.metric("CTR", len(st.session_state.ctr))

    time.sleep(1)
    st.rerun()

# =========================
# TABLES
# =========================
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚨 ALERTS")
    st.dataframe(st.session_state.alerts[-10:])

with col2:
    st.subheader("🚨 STR REPORTS")
    st.dataframe(st.session_state.str)

with col3:
    st.subheader("📄 CTR REPORTS")
    st.dataframe(st.session_state.ctr)

# =========================
# FRAUD GRAPH VIEW
# =========================
st.markdown("### 🕸️ Fraud Network Graph (Accounts vs Risk Hits)")
st.write(st.session_state.graph)

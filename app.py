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
st.set_page_config(page_title="SOC v9 Bank Deployment", layout="wide")
st.title("🏦 SOC v9 (Bank Deployment Simulation - AML + Fraud Intelligence System)")

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
        "cases": {},
        "audit": [],
        "case_id": 1000,
        "graph": {}   # fraud network graph
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
# FRAUD GRAPH (MULE NETWORK)
# =========================
def update_graph(account, txn_type):
    if account not in st.session_state.graph:
        st.session_state.graph[account] = []

    st.session_state.graph[account].append(txn_type)

# =========================
# ATTACK SIMULATION ENGINE
# =========================
def generate_txn():

    r = random.random()
    account_id = random.randint(1000, 1010)

    if r < 0.20:
        t = "MULE_BURST"
        amt = random.randint(300000, 900000)
        vel = random.randint(120, 260)
        bal = random.randint(0, 15000)

    elif r < 0.35:
        t = "STRUCTURING"
        amt = random.randint(80000, 200000)
        vel = random.randint(60, 140)
        bal = random.randint(10000, 80000)

    elif r < 0.45:
        t = "VELOCITY_ATTACK"
        amt = random.randint(50000, 300000)
        vel = random.randint(150, 260)
        bal = random.randint(0, 20000)

    else:
        t = "NORMAL"
        amt = random.randint(5000, 120000)
        vel = random.randint(0, 80)
        bal = random.randint(20000, 500000)

    update_graph(account_id, t)

    return {
        "account": account_id,
        "amount": amt,
        "velocity": vel,
        "balance": bal,
        "type": t
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
# CASE SYSTEM
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

    attack_boost = {
        "MULE_BURST": 0.45,
        "STRUCTURING": 0.30,
        "VELOCITY_ATTACK": 0.35,
        "NORMAL": 0.0
    }[txn["type"]]

    final_score = 0.6 * ml + 0.3 * anomaly_score + attack_boost

    return ml, final_score

# =========================
# DECISION ENGINE
# =========================
def decision_engine(score):
    if score >= 0.75:
        return "BLOCK"
    elif score >= 0.55:
        return "REVIEW"
    return "SAFE"

# =========================
# STR ENGINE (AML COMPLIANT)
# =========================
def is_str(event, history):

    if len(history) < 6:
        return False

    txn = event["txn"]
    recent = history[-10:]

    avg_amt = np.mean([h["txn"]["amount"] for h in recent])
    avg_vel = np.mean([h["txn"]["velocity"] for h in recent])

    return (
        txn["amount"] > 2.5 * avg_amt and
        txn["velocity"] > 2 * avg_vel and
        event["final"] > 0.7 and
        txn["type"] in ["MULE_BURST", "STRUCTURING"]
    )

# =========================
# CTR ENGINE
# =========================
def is_ctr(event):
    return event["txn"]["amount"] > 200000

# =========================
# EXPLANATION ENGINE
# =========================
def explain(txn, final):

    reasons = []

    if final > 0.75:
        reasons.append("Critical fraud probability detected")

    if txn["type"] == "MULE_BURST":
        reasons.append("Mule account network activity detected")

    if txn["type"] == "STRUCTURING":
        reasons.append("Structuring (smurfing) pattern detected")

    if txn["velocity"] > 150:
        reasons.append("Velocity laundering detected")

    if txn["amount"] > 300000:
        reasons.append("High-value suspicious transfer")

    return reasons

# =========================
# REPORTS (BANK READY)
# =========================
def str_report():
    return pd.DataFrame(st.session_state.str)

def ctr_report():
    return pd.DataFrame(st.session_state.ctr)

# =========================
# STREAM ENGINE
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

    if decision != "SAFE":
        st.session_state.alerts.append(event)

    if is_str(event, st.session_state.history):
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
            st.write("Account:", txn["account"])
            st.write("Type:", txn["type"])

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
# DASHBOARD TABLES
# =========================
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚨 ALERTS")
    st.dataframe(st.session_state.alerts[-10:])

with col2:
    st.subheader("🚨 STR REPORTS")
    st.dataframe(str_report())

with col3:
    st.subheader("📄 CTR REPORTS")
    st.dataframe(ctr_report())

# =========================
# FRAUD GRAPH VIEW (OPTIONAL INSIGHT)
# =========================
st.markdown("### 🕸️ Fraud Network Graph (Mule Links)")
st.write(st.session_state.graph)

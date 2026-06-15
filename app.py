import streamlit as st
import numpy as np
import random
import time
import joblib
import os
import pandas as pd
from datetime import datetime

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="SOC v10 Enterprise", layout="wide")
st.title("🏦 SOC v10 (Enterprise Fraud + AML Intelligence Platform)")

# =========================
# STATE INIT (CRITICAL FIX)
# =========================
def init_state():
    defaults = {
        "running": False,
        "history": [],
        "alerts": [],
        "str": [],
        "ctr": [],
        "audit": [],
        "graph": {},
        "cases": {},
        "case_id": 1000
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

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
# TXN GENERATOR
# =========================
def generate_txn():
    r = random.random()
    acc = random.randint(1000, 1010)

    if r < 0.25:
        t = "MULE"
        amt = random.randint(300000, 900000)
        vel = random.randint(120, 260)
        bal = random.randint(0, 15000)

    elif r < 0.5:
        t = "STRUCTURING"
        amt = random.randint(80000, 250000)
        vel = random.randint(60, 160)
        bal = random.randint(5000, 80000)

    elif r < 0.7:
        t = "VELOCITY"
        amt = random.randint(50000, 300000)
        vel = random.randint(150, 280)
        bal = random.randint(0, 30000)

    else:
        t = "NORMAL"
        amt = random.randint(5000, 120000)
        vel = random.randint(0, 80)
        bal = random.randint(50000, 600000)

    return {
        "account": acc,
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
# CASE ENGINE
# =========================
def new_case():
    st.session_state.case_id += 1
    return f"CASE-{st.session_state.case_id}"

# =========================
# RISK ENGINE (SINGLE TRUTH)
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
        "MULE": 0.5,
        "STRUCTURING": 0.35,
        "VELOCITY": 0.4,
        "NORMAL": 0.0
    }

    final = 0.6 * ml + 0.3 * anomaly_score + attack_map[txn["type"]]

    return ml, final

# =========================
# DECISION ENGINE
# =========================
def decision(score):
    if score >= 0.75:
        return "BLOCK"
    elif score >= 0.55:
        return "REVIEW"
    return "SAFE"

# =========================
# EXPLANATION ENGINE (CONSISTENT)
# =========================
def explain(txn, score):
    r = []

    if score > 0.75:
        r.append("Critical fraud detected")

    if score > 0.6:
        r.append("Behavioral anomaly detected")

    if txn["type"] != "NORMAL":
        r.append(f"{txn['type']} pattern detected")

    if txn["amount"] > 300000:
        r.append("High value transaction risk")

    return r if r else ["Normal behavior"]

# =========================
# FRAUD GRAPH ENGINE
# =========================
def update_graph(txn, score):

    acc = txn["account"]

    if acc not in st.session_state.graph:
        st.session_state.graph[acc] = {
            "txns": 0,
            "risk_hits": 0
        }

    st.session_state.graph[acc]["txns"] += 1

    if score > 0.6:
        st.session_state.graph[acc]["risk_hits"] += 1

# =========================
# STR / CTR ENGINE
# =========================
def is_str(event):
    return event["final"] > 0.7 and event["txn"]["amount"] > 200000

def is_ctr(event):
    return event["txn"]["amount"] > 200000

# =========================
# STREAM ENGINE
# =========================
placeholder = st.empty()

if st.session_state.running:

    txn = generate_txn()

    ml, final = risk_engine(txn, st.session_state.history)
    dec = decision(final)

    case = new_case()
    now = datetime.now().strftime("%H:%M:%S")

    event = {
        "case": case,
        "time": now,
        "txn": txn,
        "ml": ml,
        "final": final,
        "decision": dec
    }

    st.session_state.history.append(event)
    st.session_state.history = st.session_state.history[-100:]
    st.session_state.audit.append(event)

    update_graph(txn, final)

    if dec != "SAFE":
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
            st.subheader("🔴 LIVE TXN")
            st.json(txn)
            st.write("Case:", case)

        with c2:
            st.subheader("🧠 AML ENGINE")
            st.metric("ML Score", round(ml, 4))
            st.metric("Final Score", round(final, 4))
            st.write("Decision:", dec)

            st.subheader("📌 Reasoning")
            for x in explain(txn, final):
                st.write("•", x)

        with c3:
            st.subheader("📊 SOC METRICS")
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
    st.dataframe(st.session_state.str)

with col3:
    st.subheader("📄 CTR REPORTS")
    st.dataframe(st.session_state.ctr)

# =========================
# FRAUD GRAPH VIEW
# =========================
st.subheader("🕸️ Fraud Network Graph")
st.write(st.session_state.graph)

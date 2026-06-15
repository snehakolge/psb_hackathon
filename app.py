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
st.set_page_config(page_title="SOC v10.3 Stable Bank Engine", layout="wide")
st.title("🏦 SOC v10.3 (Stable AML + Fraud Intelligence System)")

# =========================
# SAFE STATE INIT
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

if not isinstance(st.session_state.graph, dict):
    st.session_state.graph = {}

# =========================
# SAFE APPEND FUNCTION (CRITICAL FIX)
# =========================
def safe_append(key, value):
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].append(value)

# =========================
# MODEL LOAD
# =========================
MODEL_PATH = "models/fraud_ensemble.pkl"
bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ SOC CONTROL PANEL")

if st.sidebar.button("▶ START STREAM"):
    st.session_state.running = True

if st.sidebar.button("⛔ STOP STREAM"):
    st.session_state.running = False

st.sidebar.metric("Alerts", len(st.session_state.alerts))
st.sidebar.metric("STR", len(st.session_state.str))
st.sidebar.metric("CTR", len(st.session_state.ctr))

# =========================
# TXN GENERATOR
# =========================
def generate_txn():
    r = random.random()
    acc = random.randint(1000, 1012)

    if r < 0.25:
        return {"account": acc, "amount": random.randint(300000, 900000),
                "velocity": random.randint(120, 260),
                "balance": random.randint(0, 15000),
                "type": "MULE"}

    elif r < 0.5:
        return {"account": acc, "amount": random.randint(80000, 250000),
                "velocity": random.randint(60, 160),
                "balance": random.randint(5000, 80000),
                "type": "STRUCTURING"}

    elif r < 0.7:
        return {"account": acc, "amount": random.randint(50000, 300000),
                "velocity": random.randint(150, 280),
                "balance": random.randint(0, 30000),
                "type": "VELOCITY"}

    return {"account": acc, "amount": random.randint(5000, 120000),
            "velocity": random.randint(0, 80),
            "balance": random.randint(50000, 600000),
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
# CASE ENGINE
# =========================
def new_case():
    st.session_state.case_id += 1
    return f"CASE-{st.session_state.case_id}"

# =========================
# RISK ENGINE
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
        "MULE": 0.55,
        "STRUCTURING": 0.35,
        "VELOCITY": 0.40,
        "NORMAL": 0.0
    }

    final_score = 0.6 * ml + 0.3 * anomaly_score + attack_map[txn["type"]]

    return ml, final_score

# =========================
# DECISION ENGINE
# =========================
def decision_engine(score):
    if score >= 0.80:
        return "BLOCK"
    elif score >= 0.60:
        return "REVIEW"
    return "SAFE"

# =========================
# EXPLANATION ENGINE (CONSISTENT)
# =========================
def explain(txn, score):

    if score >= 0.80:
        return ["CRITICAL FRAUD DETECTED"]

    elif score >= 0.60:
        return ["HIGH RISK BEHAVIOR", f"{txn['type']} pattern detected"]

    return ["NORMAL BEHAVIOR"]

# =========================
# STR RULES
# =========================
def is_str(event):
    return (
        event["final"] >= 0.75 and
        event["txn"]["amount"] >= 200000 and
        event["txn"]["velocity"] >= 100
    )

# =========================
# CTR RULES
# =========================
def is_ctr(event):
    return event["txn"]["amount"] >= 200000

# =========================
# FRAUD GRAPH
# =========================
def update_graph(txn, score):

    acc = str(txn["account"])

    if acc not in st.session_state.graph:
        st.session_state.graph[acc] = {
            "txns": 0,
            "risk_hits": 0
        }

    st.session_state.graph[acc]["txns"] += 1

    if score >= 0.6:
        st.session_state.graph[acc]["risk_hits"] += 1

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

    # HISTORY
    st.session_state.history.append(event)
    st.session_state.history = st.session_state.history[-100:]
    st.session_state.audit.append(event)

    # GRAPH
    update_graph(txn, final)

    # ALERTS
    if decision != "SAFE":
        st.session_state.alerts.append(event)

    # STR / CTR (SAFE PERSISTENCE)
    if is_str(event):
        safe_append("str", {
            "case": case,
            "time": now,
            "amount": txn["amount"],
            "velocity": txn["velocity"],
            "score": final,
            "type": txn["type"]
        })

    if is_ctr(event):
        safe_append("ctr", {
            "case": case,
            "time": now,
            "amount": txn["amount"]
        })

    # =========================
    # UI
    # =========================
    with placeholder.container():

        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("🔴 LIVE TRANSACTION")
            st.json(txn)
            st.write("Case:", case)

        with c2:
            st.subheader("🧠 AML ENGINE")
            st.metric("ML Score", round(ml, 4))
            st.metric("Final Score", round(final, 4))
            st.write("Decision:", decision)

            st.subheader("📌 Reasoning")
            for r in explain(txn, final):
                st.write("•", r)

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
    st.dataframe(pd.DataFrame(st.session_state.alerts[-10:]))

with col2:
    st.subheader("🚨 STR REPORTS")
    st.dataframe(pd.DataFrame(st.session_state.str))

with col3:
    st.subheader("📄 CTR REPORTS")
    st.dataframe(pd.DataFrame(st.session_state.ctr))

# =========================
# FRAUD GRAPH
# =========================
st.subheader("🕸️ FRAUD NETWORK GRAPH")
st.json(st.session_state.graph)

# =========================
# RBI EXPORT
# =========================
st.subheader("📥 RBI COMPLIANCE EXPORT")

def to_csv(data):
    if len(data) == 0:
        return None
    return pd.DataFrame(data).to_csv(index=False).encode("utf-8")

col1, col2 = st.columns(2)

with col1:
    ctr_csv = to_csv(st.session_state.ctr)
    if ctr_csv:
        st.download_button("Download CTR (RBI Format)", ctr_csv, "ctr.csv", "text/csv")

with col2:
    str_csv = to_csv(st.session_state.str)
    if str_csv:
        st.download_button("Download STR (RBI Format)", str_csv, "str.csv", "text/csv")

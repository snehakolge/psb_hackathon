import streamlit as st
import numpy as np
import random
import joblib
import os
import pandas as pd
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="SOC v11 Stable Engine", layout="wide")
st.title("🏦 SOC v11 (Production Fraud + AML Intelligence System)")

# =========================
# STATE INIT (HARD SAFE)
# =========================
def init_state():
    defaults = {
        "history": [],
        "alerts": [],
        "str": [],
        "ctr": [],
        "graph": {},
        "case_id": 1000,
        "last_event": None
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =========================
# SAFE APPEND
# =========================
def safe_append(key, value):
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].append(value)

# =========================
# MODEL LOAD (optional)
# =========================
MODEL_PATH = "models/fraud_ensemble.pkl"
bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# =========================
# TRANSACTION GENERATOR
# =========================
def generate_txn():
    r = random.random()
    acc = random.randint(1000, 1015)

    if r < 0.25:
        return {"account": acc, "amount": random.randint(300000, 900000),
                "velocity": random.randint(120, 260),
                "balance": random.randint(0, 15000),
                "type": "MULE"}

    if r < 0.5:
        return {"account": acc, "amount": random.randint(80000, 250000),
                "velocity": random.randint(60, 160),
                "balance": random.randint(5000, 80000),
                "type": "STRUCTURING"}

    if r < 0.7:
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
# RISK ENGINE (single truth)
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

    final = 0.6 * ml + 0.3 * anomaly_score + attack_map[txn["type"]]

    return ml, final

# =========================
# DECISION ENGINE
# =========================
def decision(score):
    if score >= 0.80:
        return "BLOCK"
    if score >= 0.60:
        return "REVIEW"
    return "SAFE"

# =========================
# EXPLAIN (NO CONTRADICTION)
# =========================
def explain(txn, score):

    if score >= 0.80:
        return ["CRITICAL FRAUD DETECTED"]

    if score >= 0.60:
        return ["HIGH RISK TRANSACTION"]

    if txn["type"] != "NORMAL":
        return [f"{txn['type']} pattern observed"]

    return ["NORMAL TRANSACTION"]

# =========================
# STR RULE
# =========================
def is_str(event):
    return event["final"] >= 0.75 and event["txn"]["amount"] >= 200000

# =========================
# CTR RULE
# =========================
def is_ctr(event):
    return event["txn"]["amount"] >= 200000

# =========================
# GRAPH UPDATE (ROBUST)
# =========================
def update_graph(txn, score):

    acc = str(txn["account"])

    if acc not in st.session_state.graph:
        st.session_state.graph[acc] = {"txns": 0, "risk_hits": 0}

    st.session_state.graph[acc]["txns"] += 1

    if score >= 0.60:
        st.session_state.graph[acc]["risk_hits"] += 1

# =========================
# UI CONTROLS
# =========================
colA, colB = st.columns(2)

with colA:
    if st.button("▶ GENERATE NEXT TRANSACTION"):
        txn = generate_txn()

        ml, final = risk_engine(txn, st.session_state.history)
        dec = decision(final)

        case = st.session_state.case_id
        st.session_state.case_id += 1

        event = {
            "case": f"CASE-{case}",
            "txn": txn,
            "ml": ml,
            "final": final,
            "decision": dec,
            "time": datetime.now().strftime("%H:%M:%S")
        }

        st.session_state.last_event = event
        st.session_state.history.append(event)

        update_graph(txn, final)

        if dec != "SAFE":
            safe_append("alerts", event)

        if is_str(event):
            safe_append("str", event)

        if is_ctr(event):
            safe_append("ctr", event)

# =========================
# DASHBOARD
# =========================
st.markdown("---")

if st.session_state.last_event:

    e = st.session_state.last_event

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("🔴 LIVE TXN")
        st.json(e["txn"])
        st.write(e["case"])

    with c2:
        st.subheader("🧠 AML ENGINE")
        st.metric("ML Score", round(e["ml"], 4))
        st.metric("Final Score", round(e["final"], 4))
        st.write("Decision:", e["decision"])

        st.subheader("📌 Reasoning")
        for r in explain(e["txn"], e["final"]):
            st.write("•", r)

    with c3:
        st.subheader("📊 SOC METRICS")
        st.metric("Alerts", len(st.session_state.alerts))
        st.metric("STR", len(st.session_state.str))
        st.metric("CTR", len(st.session_state.ctr))

# =========================
# TABLES
# =========================
st.markdown("## 🚨 ALERTS")
st.dataframe(pd.DataFrame(st.session_state.alerts))

st.markdown("## 🚨 STR REPORTS")
st.dataframe(pd.DataFrame(st.session_state.str))

st.markdown("## 📄 CTR REPORTS")
st.dataframe(pd.DataFrame(st.session_state.ctr))

# =========================
# GRAPH (FIXED)
# =========================
st.markdown("## 🕸️ FRAUD GRAPH")
st.dataframe(pd.DataFrame.from_dict(st.session_state.graph, orient="index"))

# =========================
# EXPORT
# =========================
st.markdown("## 📥 RBI EXPORT")

def export(data):
    if len(data) == 0:
        return None
    return pd.DataFrame(data).to_csv(index=False).encode("utf-8")

col1, col2 = st.columns(2)

with col1:
    ctr = export(st.session_state.ctr)
    if ctr:
        st.download_button("Download CTR", ctr, "ctr.csv")

with col2:
    strr = export(st.session_state.str)
    if strr:
        st.download_button("Download STR", strr, "str.csv")

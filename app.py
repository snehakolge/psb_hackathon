import streamlit as st
import numpy as np
import random
import pandas as pd
import joblib
import os
from datetime import datetime
import time

st.set_page_config(page_title="SOC v12 Stable Engine", layout="wide")
st.title("🏦 SOC v12 (Production Fraud + AML Intelligence System)")

# =========================
# SAFE INIT (CRITICAL FIX)
# =========================
def init():
    defaults = {
        "running": False,
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

init()

# =========================
# MODEL LOAD
# =========================
MODEL_PATH = "models/fraud_ensemble.pkl"
bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# =========================
# TRANSACTION STREAM
# =========================
def txn_generator():
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
# RISK ENGINE
# =========================
def risk(txn):
    ml = ml_score(txn)

    attack_map = {
        "MULE": 0.55,
        "STRUCTURING": 0.35,
        "VELOCITY": 0.40,
        "NORMAL": 0.0
    }

    final = 0.7 * ml + attack_map[txn["type"]]
    return ml, final

# =========================
# DECISION
# =========================
def decision(score):
    if score >= 0.80:
        return "BLOCK"
    if score >= 0.60:
        return "REVIEW"
    return "SAFE"

# =========================
# STR / CTR RULES
# =========================
def is_str(e):
    return e["final"] >= 0.75 and e["txn"]["amount"] >= 200000

def is_ctr(e):
    return e["txn"]["amount"] >= 200000

# =========================
# GRAPH UPDATE
# =========================
def update_graph(txn, score):
    acc = str(txn["account"])

    if acc not in st.session_state.graph:
        st.session_state.graph[acc] = {"txns": 0, "risk_hits": 0}

    st.session_state.graph[acc]["txns"] += 1
    if score > 0.6:
        st.session_state.graph[acc]["risk_hits"] += 1

# =========================
# SAFE APPEND
# =========================
def add(list_name, item):
    if list_name not in st.session_state:
        st.session_state[list_name] = []
    st.session_state[list_name].append(item)

# =========================
# CONTROL PANEL
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ START LIVE SOC STREAM"):
        st.session_state.running = True

with col2:
    if st.button("⛔ STOP STREAM"):
        st.session_state.running = False

# =========================
# STREAM ENGINE (SAFE LOOP)
# =========================
if st.session_state.running:

    txn = txn_generator()
    ml, final = risk(txn)
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
        add("alerts", event)

    if is_str(event):
        add("str", event)

    if is_ctr(event):
        add("ctr", event)

    time.sleep(1)
    st.rerun()

# =========================
# DASHBOARD
# =========================
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

    with c3:
        st.subheader("📊 SOC METRICS")
        st.metric("Alerts", len(st.session_state.alerts))
        st.metric("STR", len(st.session_state.str))
        st.metric("CTR", len(st.session_state.ctr))

# =========================
# TABLES (FORCE NON EMPTY SAFE VIEW)
# =========================
st.markdown("## 🚨 ALERTS")
st.dataframe(pd.DataFrame(st.session_state.alerts))

st.markdown("## 🚨 STR")
st.dataframe(pd.DataFrame(st.session_state.str))

st.markdown("## 📄 CTR")
st.dataframe(pd.DataFrame(st.session_state.ctr))

# =========================
# GRAPH FIX
# =========================
st.markdown("## 🕸️ FRAUD GRAPH")
st.dataframe(pd.DataFrame.from_dict(st.session_state.graph, orient="index"))

# =========================
# RBI EXPORT FIX (CRITICAL)
# =========================
def export(data):
    if not data or len(data) == 0:
        return None
    return pd.DataFrame(data).to_csv(index=False).encode("utf-8")

st.markdown("## 📥 RBI EXPORT")

ctr_file = export(st.session_state.ctr)
str_file = export(st.session_state.str)

col1, col2 = st.columns(2)

with col1:
    if ctr_file:
        st.download_button("Download CTR", ctr_file, "ctr.csv")

with col2:
    if str_file:
        st.download_button("Download STR", str_file, "str.csv")

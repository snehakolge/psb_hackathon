import streamlit as st
import numpy as np
import random
import pandas as pd
import joblib
import os
from datetime import datetime

st.set_page_config(page_title="SOC v13 Stable Engine", layout="wide")
st.title("🏦 SOC v13 (Production Fraud + AML Intelligence System)")

# =========================
# FORCE SAFE STATE INIT
# =========================
def init():
    keys = {
        "history": [],
        "alerts": [],
        "str": [],
        "ctr": [],
        "graph": {},
        "case_id": 1000,
        "last_event": None
    }
    for k, v in keys.items():
        if k not in st.session_state or st.session_state[k] is None:
            st.session_state[k] = v

init()

# =========================
# MODEL
# =========================
MODEL_PATH = "models/fraud_ensemble.pkl"
bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# =========================
# TRANSACTION GENERATOR
# =========================
def gen_txn():
    r = random.random()
    acc = random.randint(1000, 1012)

    if r < 0.3:
        return {"account": acc, "amount": random.randint(300000, 900000),
                "velocity": random.randint(120, 260),
                "balance": random.randint(0, 20000),
                "type": "MULE"}

    if r < 0.6:
        return {"account": acc, "amount": random.randint(80000, 250000),
                "velocity": random.randint(60, 180),
                "balance": random.randint(5000, 80000),
                "type": "STRUCTURING"}

    return {"account": acc, "amount": random.randint(5000, 200000),
            "velocity": random.randint(10, 120),
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

    return 0.6 * xgb.predict_proba(X)[0][1] + 0.4 * lgbm.predict_proba(X)[0][1]

# =========================
# RISK ENGINE
# =========================
def risk(txn):
    ml = ml_score(txn)

    attack = {
        "MULE": 0.6,
        "STRUCTURING": 0.4,
        "NORMAL": 0.0
    }

    final = 0.7 * ml + attack[txn["type"]]
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
# STR / CTR
# =========================
def is_str(e):
    return e["final"] >= 0.75 and e["txn"]["amount"] > 200000

def is_ctr(e):
    return e["txn"]["amount"] > 200000

# =========================
# GRAPH UPDATE (NO LOSS GUARANTEE)
# =========================
def update_graph(txn, score):
    acc = str(txn["account"])

    if "graph" not in st.session_state or not isinstance(st.session_state.graph, dict):
        st.session_state.graph = {}

    if acc not in st.session_state.graph:
        st.session_state.graph[acc] = {"txns": 0, "risk": 0}

    st.session_state.graph[acc]["txns"] += 1
    if score > 0.6:
        st.session_state.graph[acc]["risk"] += 1

# =========================
# SAFE APPEND
# =========================
def add(key, value):
    if key not in st.session_state or st.session_state[key] is None:
        st.session_state[key] = []
    st.session_state[key].append(value)

# =========================
# CONTROL PANEL (IMPORTANT)
# =========================
col1, col2 = st.columns(2)

run = False
with col1:
    if st.button("▶ GENERATE TRANSACTION"):
        run = True

with col2:
    if st.button("🔁 RESET SYSTEM"):
        for k in st.session_state.keys():
            del st.session_state[k]
        st.rerun()

# =========================
# SINGLE EVENT ENGINE (CRITICAL FIX)
# =========================
if run:

    txn = gen_txn()
    ml, final = risk(txn)
    dec = decision(final)

    case = f"CASE-{st.session_state.case_id}"
    st.session_state.case_id += 1

    event = {
        "case": case,
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

# =========================
# DASHBOARD (ALWAYS SAFE RENDER)
# =========================
st.markdown("## 🔴 LIVE SOC DASHBOARD")

if st.session_state.last_event:

    e = st.session_state.last_event

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("TRANSACTION")
        st.json(e["txn"])
        st.write(e["case"])

    with c2:
        st.subheader("AI ENGINE")
        st.metric("ML", round(e["ml"], 4))
        st.metric("FINAL", round(e["final"], 4))
        st.write(e["decision"])

    with c3:
        st.subheader("METRICS")
        st.metric("Alerts", len(st.session_state.alerts))
        st.metric("STR", len(st.session_state.str))
        st.metric("CTR", len(st.session_state.ctr))

# =========================
# TABLES (NEVER EMPTY VISUAL FAIL)
# =========================
st.markdown("## 🚨 ALERTS")
st.dataframe(pd.DataFrame(st.session_state.alerts))

st.markdown("## 🚨 STR")
st.dataframe(pd.DataFrame(st.session_state.str))

st.markdown("## 📄 CTR")
st.dataframe(pd.DataFrame(st.session_state.ctr))

# =========================
# GRAPH (FINAL FIX)
# =========================
st.markdown("## 🕸️ FRAUD GRAPH")

graph_df = pd.DataFrame.from_dict(st.session_state.graph, orient="index")

if len(graph_df) == 0:
    st.info("No graph data yet. Generate transactions.")
else:
    st.dataframe(graph_df)

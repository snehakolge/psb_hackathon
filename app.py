import streamlit as st
import numpy as np
import pandas as pd
import random
import joblib
import os
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="SOC AML Engine", layout="wide")
st.title("🏦 SOC (Fraud + AML Intelligence Engine)")

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
        "graph": [],
        "case_id": 1000,
        "last_event": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =========================
# MODEL LOAD (SAFE)
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

    if r < 0.75:
        return {"account": acc, "amount": random.randint(50000, 300000),
                "velocity": random.randint(150, 280),
                "balance": random.randint(0, 30000),
                "type": "VELOCITY"}

    return {"account": acc, "amount": random.randint(5000, 120000),
            "velocity": random.randint(10, 80),
            "balance": random.randint(50000, 600000),
            "type": "NORMAL"}

# =========================
# ML SCORE (SAFE FALLBACK)
# =========================
def ml_score(txn):
    if not bundle:
        return 0.5

    try:
        X = np.array([[txn["amount"], txn["velocity"], txn["balance"]]])
        xgb = bundle["xgb_model"]
        lgbm = bundle["lgbm_model"]

        return 0.6 * xgb.predict_proba(X)[0][1] + 0.4 * lgbm.predict_proba(X)[0][1]

    except:
        return 0.5

# =========================
# RISK ENGINE
# =========================
def risk(txn):
    ml = ml_score(txn)

    weights = {
        "MULE": 0.6,
        "STRUCTURING": 0.4,
        "VELOCITY": 0.45,
        "NORMAL": 0.0
    }

    final = 0.7 * ml + weights[txn["type"]]
    return ml, final

# =========================
# DECISION ENGINE
# =========================
def decision(score):
    if score >= 0.80:
        return "BLOCK"
    elif score >= 0.60:
        return "REVIEW"
    return "SAFE"

# =========================
# STR / CTR RULES
# =========================
def is_str(event):
    return event["final"] >= 0.75 and event["txn"]["amount"] > 200000

def is_ctr(event):
    return event["txn"]["amount"] > 200000

# =========================
# SAFE APPEND
# =========================
def add(key, value):
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].append(value)

# =========================
# TRANSACTION STEP ENGINE
# =========================
def step():
    txn = generate_txn()
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

    # GRAPH DATA (SAFE SIMPLE FORMAT)
    st.session_state.graph.append({
        "account": txn["account"],
        "risk": final
    })

    if dec != "SAFE":
        add("alerts", event)

    if is_str(event):
        add("str", event)

    if is_ctr(event):
        add("ctr", event)

# =========================
# CONTROLS
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ GENERATE TRANSACTION"):
        step()

with col2:
    if st.button("▶ RUN 5-STEP STREAM"):
        for _ in range(5):
            step()

# =========================
# DASHBOARD
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
        st.metric("ML Score", round(e["ml"], 4))
        st.metric("Final Score", round(e["final"], 4))
        st.write(e["decision"])

    with c3:
        st.subheader("METRICS")
        st.metric("Alerts", len(st.session_state.alerts))
        st.metric("STR", len(st.session_state.str))
        st.metric("CTR", len(st.session_state.ctr))

# =========================
# TABLES (ALWAYS VISIBLE)
# =========================
st.markdown("## 🚨 ALERTS")
st.dataframe(pd.DataFrame(st.session_state.alerts))

st.markdown("## 🚨 STR REPORTS")
st.dataframe(pd.DataFrame(st.session_state.str))

st.markdown("## 📄 CTR REPORTS")
st.dataframe(pd.DataFrame(st.session_state.ctr))

# =========================
# FRAUD GRAPH (NO LIBRARIES)
# =========================
st.markdown("## 🕸️ FRAUD RISK GRAPH")

if len(st.session_state.graph) > 0:
    df = pd.DataFrame(st.session_state.graph)

    st.bar_chart(df.groupby("account")["risk"].mean())
else:
    st.info("No graph data yet")

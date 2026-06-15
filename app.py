import streamlit as st
import numpy as np
import random
import pandas as pd
import joblib
import os
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="SOC v14 AML Engine", layout="wide")
st.title("🏦 SOC v14 (Bank-Grade AML + Fraud Intelligence System)")

# =========================
# INIT STATE (CRITICAL FIX)
# =========================
def init():
    defaults = {
        "running": False,
        "history": [],
        "alerts": [],
        "str": [],
        "ctr": [],
        "graph_edges": [],
        "case_id": 1000,
        "last_event": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()

# =========================
# MODEL (optional)
# =========================
MODEL_PATH = "models/fraud_ensemble.pkl"
bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# =========================
# TRANSACTION STREAM
# =========================
def gen_txn():
    r = random.random()
    acc = random.randint(1000, 1015)

    if r < 0.3:
        return {"account": acc, "amount": random.randint(300000, 900000),
                "velocity": random.randint(120, 260),
                "balance": random.randint(0, 15000),
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
# DECISION ENGINE
# =========================
def decision(score):
    if score > 0.80:
        return "BLOCK"
    if score > 0.60:
        return "REVIEW"
    return "SAFE"

# =========================
# STR / CTR LOGIC
# =========================
def is_str(e):
    return e["final"] > 0.75 and e["txn"]["amount"] > 200000

def is_ctr(e):
    return e["txn"]["amount"] > 200000

# =========================
# GRAPH UPDATE (REAL NETWORK)
# =========================
def update_graph(txn, score):
    acc = txn["account"]

    st.session_state.graph_edges.append((
        acc,
        "RISK" if score > 0.6 else "SAFE"
    ))

# =========================
# MULTI-STREAM GENERATOR (KEY FIX)
# =========================
def generate_batch(n=3):
    for _ in range(n):

        txn = gen_txn()
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

        st.session_state.history.append(event)
        st.session_state.last_event = event

        if dec != "SAFE":
            st.session_state.alerts.append(event)

        if is_str(event):
            st.session_state.str.append(event)

        if is_ctr(event):
            st.session_state.ctr.append(event)

        update_graph(txn, final)

# =========================
# CONTROL PANEL
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ RUN SOC STREAM (BATCH 5 EVENTS)"):
        generate_batch(5)

with col2:
    if st.button("🔄 RESET SYSTEM"):
        for k in st.session_state.keys():
            del st.session_state[k]
        st.rerun()

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
        st.metric("ML", round(e["ml"], 4))
        st.metric("FINAL", round(e["final"], 4))
        st.write(e["decision"])

    with c3:
        st.subheader("METRICS")
        st.metric("Alerts", len(st.session_state.alerts))
        st.metric("STR", len(st.session_state.str))
        st.metric("CTR", len(st.session_state.ctr))

# =========================
# TABLES (FORCE VISIBILITY)
# =========================
st.markdown("## 🚨 ALERTS")
st.dataframe(pd.DataFrame(st.session_state.alerts))

st.markdown("## 🚨 STR")
st.dataframe(pd.DataFrame(st.session_state.str))

st.markdown("## 📄 CTR")
st.dataframe(pd.DataFrame(st.session_state.ctr))

# =========================
# GRAPH (VISUAL FIX - NETWORKX)
# =========================
st.markdown("## 🕸️ FRAUD GRAPH (VISUAL)")

G = nx.Graph()
for edge in st.session_state.graph_edges:
    G.add_edge(edge[0], edge[1])

fig, ax = plt.subplots()
nx.draw(G, with_labels=True, node_color="lightblue", node_size=1500, ax=ax)

st.pyplot(fig)

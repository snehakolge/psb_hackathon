import streamlit as st
import random
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="SOC Autonomous Engine", layout="wide")
st.title("🏦 SOC vNEXT (Autonomous Fraud + AML Intelligence System)")

# =========================
# STATE INIT
# =========================
def init():
    defaults = {
        "running": False,
        "tick": 0,
        "case_id": 1000,
        "alerts": [],
        "str": [],
        "ctr": [],
        "graph": {},
        "fraud_rate": 0.5,
        "last_event": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()

# =========================
# CONTROLS
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ START LIVE SOC"):
        st.session_state.running = True

with col2:
    if st.button("⛔ STOP SOC"):
        st.session_state.running = False

# =========================
# AUTO REFRESH ENGINE
# =========================
if st.session_state.running:
    st_autorefresh(interval=1200, key="soc_tick")

    st.session_state.tick += 1

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
                "velocity": random.randint(60, 180),
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
# RISK ENGINE (SELF-HEALING)
# =========================
def risk(txn):
    base_ml = random.uniform(0.3, 0.7)

    weights = {
        "MULE": 0.6,
        "STRUCTURING": 0.45,
        "VELOCITY": 0.4,
        "NORMAL": 0.0
    }

    final = 0.7 * base_ml + weights[txn["type"]]
    return base_ml, final

# =========================
# DECISION ENGINE
# =========================
def decision(score):
    if score > 0.80:
        return "BLOCK"
    elif score > 0.60:
        return "REVIEW"
    return "SAFE"

# =========================
# STR / CTR RULES
# =========================
def is_str(e):
    return e["final"] > 0.75 and e["txn"]["amount"] > 200000

def is_ctr(e):
    return e["txn"]["amount"] > 200000

# =========================
# SELF LEARNING UPDATE
# =========================
def update_learning():
    total = len(st.session_state.alerts)
    fraud = len(st.session_state.str) + len(st.session_state.ctr)

    if total > 0:
        st.session_state.fraud_rate = fraud / total

# =========================
# GRAPH UPDATE
# =========================
def update_graph(txn, score):
    acc = txn["account"]

    if acc not in st.session_state.graph:
        st.session_state.graph[acc] = {"txns": 0, "risk": 0}

    st.session_state.graph[acc]["txns"] += 1
    st.session_state.graph[acc]["risk"] += score

# =========================
# ENGINE STEP (1 TICK)
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

    if dec != "SAFE":
        st.session_state.alerts.append(event)

    if is_str(event):
        st.session_state.str.append(event)

    if is_ctr(event):
        st.session_state.ctr.append(event)

    update_graph(txn, final)
    update_learning()

# =========================
# AUTO RUN ENGINE
# =========================
if st.session_state.running:
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
        st.subheader("SOC METRICS")
        st.metric("Alerts", len(st.session_state.alerts))
        st.metric("STR", len(st.session_state.str))
        st.metric("CTR", len(st.session_state.ctr))
        st.metric("Fraud Rate", round(st.session_state.fraud_rate, 3))

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
# FRAUD GRAPH (STABLE)
# =========================
st.markdown("## 🕸️ FRAUD RISK GRAPH")

if len(st.session_state.graph) > 0:
    df = pd.DataFrame.from_dict(st.session_state.graph, orient="index")

    st.bar_chart(df[["risk"]])
    st.line_chart(df[["txns"]])
else:
    st.info("Graph will build after stream starts.")

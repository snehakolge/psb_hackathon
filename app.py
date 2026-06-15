import streamlit as st
import random
import numpy as np
import pandas as pd
from datetime import datetime
import time

# =========================
# PAGE
# =========================
st.set_page_config(page_title="SOC AI Engine", layout="wide")
st.title("🏦 SOC vNEXT (Autonomous AML + Fraud Intelligence)")

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
        "last_event": None,
        "feedback": []
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()

# =========================
# CONTROL PANEL
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ START SOC STREAM"):
        st.session_state.running = True

with col2:
    if st.button("⛔ STOP SOC"):
        st.session_state.running = False

# =========================
# TRANSACTION GENERATOR
# =========================
def gen_txn():
    r = random.random()
    acc = random.randint(1000, 1015)

    if r < 0.3:
        return {"account": acc, "amount": random.randint(250000, 900000),
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
# RISK ENGINE
# =========================
def risk(txn):
    ml = random.uniform(0.3, 0.8)

    weight = {
        "MULE": 0.6,
        "STRUCTURING": 0.4,
        "NORMAL": 0.0
    }

    final = 0.7 * ml + weight[txn["type"]]
    return ml, final

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
# RULES
# =========================
def is_str(e):
    return e["final"] > 0.75 and e["txn"]["amount"] > 200000

def is_ctr(e):
    return e["txn"]["amount"] > 200000

# =========================
# SELF HEALING
# =========================
def update_learning():
    total = len(st.session_state.alerts)
    fraud = len(st.session_state.str) + len(st.session_state.ctr)

    if total > 0:
        st.session_state.fraud_rate = fraud / total

# =========================
# STEP ENGINE
# =========================
def step():
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

    st.session_state.last_event = event

    if dec != "SAFE":
        st.session_state.alerts.append(event)

    if is_str(event):
        st.session_state.str.append(event)

    if is_ctr(event):
        st.session_state.ctr.append(event)

    # graph update
    acc = txn["account"]
    if acc not in st.session_state.graph:
        st.session_state.graph[acc] = {"risk": 0, "txns": 0}

    st.session_state.graph[acc]["risk"] += final
    st.session_state.graph[acc]["txns"] += 1

    update_learning()

# =========================
# AUTO LOOP (SAFE STREAMLIT WAY)
# =========================
if st.session_state.running:
    step()
    time.sleep(1)
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

st.markdown("## 🚨 STR")
st.dataframe(pd.DataFrame(st.session_state.str))

st.markdown("## 📄 CTR")
st.dataframe(pd.DataFrame(st.session_state.ctr))

# =========================
# GRAPH
# =========================
st.markdown("## 🕸️ FRAUD GRAPH")

df = pd.DataFrame.from_dict(st.session_state.graph, orient="index")

if len(df) > 0:
    st.bar_chart(df["risk"])
    st.line_chart(df["txns"])
else:
    st.info("Graph building after stream starts.")

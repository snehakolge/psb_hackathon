import streamlit as st
import random
import numpy as np
import pandas as pd
from datetime import datetime
import time

st.set_page_config(page_title="SOC Stable Engine", layout="wide")
st.title("🏦 SOC vNEXT (Stable Autonomous AML System)")

# =========================
# SAFE INIT
# =========================
def init():
    defaults = {
        "running": False,
        "case_id": 1000,
        "alerts": [],
        "str": [],
        "ctr": [],
        "graph": {},   # FIXED STRUCTURE
        "fraud_rate": 0.5,
        "last_event": None,
        "tick": 0
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
    if st.button("▶ START SOC"):
        st.session_state.running = True

with col2:
    if st.button("⛔ STOP SOC"):
        st.session_state.running = False

# =========================
# TXN GENERATOR (VARIATION FIX)
# =========================
def gen_txn():
    seed = st.session_state.tick + random.randint(1, 1000)

    random.seed(seed)

    acc = random.randint(1000, 1015)
    r = random.random()

    if r < 0.3:
        return {"account": acc, "amount": random.randint(200000, 900000),
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

    weights = {
        "MULE": 0.6,
        "STRUCTURING": 0.4,
        "NORMAL": 0.0
    }

    final = 0.7 * ml + weights[txn["type"]]
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
# RULES
# =========================
def is_str(e):
    return e["final"] > 0.75 and e["txn"]["amount"] > 200000

def is_ctr(e):
    return e["txn"]["amount"] > 200000

# =========================
# SAFE GRAPH UPDATE (FIXED)
# =========================
def update_graph(txn, score):
    acc = str(txn["account"])

    if acc not in st.session_state.graph:
        st.session_state.graph[acc] = {"risk": 0.0, "txns": 0}

    st.session_state.graph[acc]["risk"] += float(score)
    st.session_state.graph[acc]["txns"] += 1

# =========================
# ENGINE STEP
# =========================
def step():

    st.session_state.tick += 1

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

    update_graph(txn, final)

# =========================
# AUTO LOOP (SAFE)
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
        st.subheader("METRICS")
        st.metric("Alerts", len(st.session_state.alerts))
        st.metric("STR", len(st.session_state.str))
        st.metric("CTR", len(st.session_state.ctr))

# =========================
# TABLES (ALWAYS STABLE)
# =========================
st.markdown("## 🚨 ALERTS")
st.dataframe(pd.DataFrame(st.session_state.alerts))

st.markdown("## 🚨 STR")
st.dataframe(pd.DataFrame(st.session_state.str))

st.markdown("## 📄 CTR")
st.dataframe(pd.DataFrame(st.session_state.ctr))

# =========================
# GRAPH FIX (NO from_dict)
# =========================
st.markdown("## 🕸️ FRAUD GRAPH")

if len(st.session_state.graph) > 0:

    df = pd.DataFrame([
        {"account": k, "risk": v["risk"], "txns": v["txns"]}
        for k, v in st.session_state.graph.items()
    ])

    st.bar_chart(df.set_index("account")[["risk"]])
    st.line_chart(df.set_index("account")[["txns"]])

else:
    st.info("Graph building after stream starts")

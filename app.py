import streamlit as st
import numpy as np
import random
import time
import joblib
import os
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="SOC v8 Attack Simulation", layout="wide")
st.title("🏦 Enterprise SOC v8 (Attack Simulation + AML Intelligence Engine)")

# =========================
# STATE INIT (SAFE)
# =========================
def init():
    defaults = {
        "running": False,
        "history": [],
        "alerts": [],
        "str": [],
        "ctr": [],
        "case_id": 1000
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()
st.session_state.case_id = int(st.session_state.case_id)

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "models/fraud_ensemble.pkl"
bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# =========================
# CONTROL PANEL
# =========================
st.sidebar.header("⚙️ SOC CONTROL PANEL")

if st.sidebar.button("▶ START ATTACK SIMULATION"):
    st.session_state.running = True

if st.sidebar.button("⛔ STOP"):
    st.session_state.running = False

st.sidebar.metric("Alerts", len(st.session_state.alerts))
st.sidebar.metric("STR", len(st.session_state.str))
st.sidebar.metric("CTR", len(st.session_state.ctr))

# =========================
# ATTACK SIMULATION ENGINE (KEY UPGRADE)
# =========================
def generate_txn():

    mode = random.random()

    # 🔴 FRAUD ATTACK MODES (30%)
    if mode < 0.15:
        # Mule / laundering burst
        return {
            "amount": random.randint(200000, 900000),
            "velocity": random.randint(120, 250),
            "balance": random.randint(0, 15000),
            "type": "MULE_BURST"
        }

    elif mode < 0.25:
        # Structuring (smurfing)
        return {
            "amount": random.randint(90000, 199000),
            "velocity": random.randint(60, 140),
            "balance": random.randint(10000, 80000),
            "type": "STRUCTURING"
        }

    elif mode < 0.3:
        # Rapid velocity laundering
        return {
            "amount": random.randint(50000, 300000),
            "velocity": random.randint(160, 260),
            "balance": random.randint(0, 20000),
            "type": "VELOCITY_ATTACK"
        }

    # 🟢 NORMAL
    return {
        "amount": random.randint(5000, 120000),
        "velocity": random.randint(0, 80),
        "balance": random.randint(20000, 500000),
        "type": "NORMAL"
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
# CASE ID
# =========================
def new_case():
    st.session_state.case_id += 1
    return f"CASE-{st.session_state.case_id}"

# =========================
# RISK ENGINE (ATTACK-AWARE)
# =========================
def risk_engine(txn, history):

    ml = ml_score(txn)

    if len(history) > 5:
        avg_amt = np.mean([h["txn"]["amount"] for h in history[-10:]])
        avg_vel = np.mean([h["txn"]["velocity"] for h in history[-10:]])
    else:
        avg_amt, avg_vel = txn["amount"], txn["velocity"]

    anomaly = 0
    anomaly += txn["amount"] > 2 * avg_amt
    anomaly += txn["velocity"] > 1.8 * avg_vel
    anomaly += txn["balance"] < 5000

    attack_bonus = 0
    if txn["type"] == "MULE_BURST":
        attack_bonus = 0.4
    elif txn["type"] == "STRUCTURING":
        attack_bonus = 0.25
    elif txn["type"] == "VELOCITY_ATTACK":
        attack_bonus = 0.35

    final = 0.6 * ml + 0.3 * (anomaly / 3) + attack_bonus

    if final > 0.72:
        decision = "BLOCK"
    elif final > 0.55:
        decision = "REVIEW"
    else:
        decision = "SAFE"

    return ml, final, decision

# =========================
# STR ENGINE (REAL AML LOGIC)
# =========================
def is_str(event, history):

    if len(history) < 6:
        return False

    txn = event["txn"]
    recent = history[-10:]

    avg_amt = np.mean([h["txn"]["amount"] for h in recent])
    avg_vel = np.mean([h["txn"]["velocity"] for h in recent])

    # AML STR triggers
    conditions = [
        txn["amount"] > 2.5 * avg_amt,
        txn["velocity"] > 2 * avg_vel,
        event["final"] > 0.7,
        txn["type"] in ["MULE_BURST", "STRUCTURING"]
    ]

    return sum(conditions) >= 2

# =========================
# CTR ENGINE (COMPLIANCE)
# =========================
def is_ctr(event):
    txn = event["txn"]
    return txn["amount"] > 200000 and txn["type"] != "NORMAL"

# =========================
# EXPLANATION ENGINE
# =========================
def explain(txn):

    reasons = []

    if txn["type"] == "MULE_BURST":
        reasons.append("Possible mule account burst detected")

    if txn["type"] == "STRUCTURING":
        reasons.append("Structuring pattern (smurfing) detected")

    if txn["velocity"] > 150:
        reasons.append("High velocity laundering pattern")

    if txn["amount"] > 300000:
        reasons.append("High-value suspicious transfer")

    return reasons

# =========================
# STREAM ENGINE
# =========================
placeholder = st.empty()

if st.session_state.running:

    txn = generate_txn()
    ml = ml_score(txn)
    final, decision = risk_engine(txn, st.session_state.history)[1:3]

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

    # MEMORY
    st.session_state.history.append(event)
    st.session_state.history = st.session_state.history[-80:]

    # ALERT ENGINE
    if decision != "SAFE":
        st.session_state.alerts.append(event)

    if is_str(event, st.session_state.history):
        st.session_state.str.append(event)

    if is_ctr(event):
        st.session_state.ctr.append(event)

    # LIMIT
    st.session_state.alerts = st.session_state.alerts[-30:]
    st.session_state.str = st.session_state.str[-30:]
    st.session_state.ctr = st.session_state.ctr[-30:]

    # =========================
    # UI DASHBOARD
    # =========================
    with placeholder.container():

        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("🔴 LIVE ATTACK STREAM")
            st.json(txn)
            st.write("Case:", case)
            st.write("Type:", txn["type"])

        with c2:
            st.subheader("🧠 AML ENGINE")
            st.metric("ML Score", round(ml, 4))
            st.metric("Risk Score", round(final, 4))
            st.write("Decision:", decision)

            st.subheader("📌 Reasoning")
            for r in explain(txn):
                st.write("•", r)

        with c3:
            st.subheader("📊 SOC HEALTH")
            st.metric("Alerts", len(st.session_state.alerts))
            st.metric("STR", len(st.session_state.str))
            st.metric("CTR", len(st.session_state.ctr))

    time.sleep(1)
    st.rerun()

# =========================
# SOC TABLES
# =========================
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚨 ALERTS")
    for a in st.session_state.alerts[-10:]:
        st.error(f"{a['case']} | {a['decision']} | {a['time']}")

with col2:
    st.subheader("🚨 STR (AML REPORTS)")
    for s in st.session_state.str[-10:]:
        st.warning(f"{s['case']} | SCORE {round(s['final'],2)}")

with col3:
    st.subheader("📄 CTR (COMPLIANCE)")
    for c in st.session_state.ctr[-10:]:
        st.info(f"{c['case']} | TYPE {c['txn']['type']}")

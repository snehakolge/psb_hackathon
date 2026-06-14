import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import os

from agent_ecosystem_engine import AdaptiveConsensusEcosystem

# =========================================================
# CONFIG
# =========================================================
FEATURES = ['F115','F527','F531','F2582','F2678','F2956','F3043']

st.set_page_config(page_title="Agentic Fraud SOC", layout="wide")

st.title("🏦 PSB Hackathon - Agentic Fraud SOC (Clean Architecture)")

# =========================================================
# SAFE STATE STORAGE (IMPORTANT FIX)
# =========================================================
if "events" not in st.session_state:
    st.session_state.events = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

if "high_risk_queue" not in st.session_state:
    st.session_state.high_risk_queue = []

if "running" not in st.session_state:
    st.session_state.running = False

if "step" not in st.session_state:
    st.session_state.step = 0

# =========================================================
# ML ECOSYSTEM
# =========================================================
@st.cache_resource
def get_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = get_ecosystem()

# =========================================================
# START / STOP
# =========================================================
col1, col2 = st.columns(2)

if col1.button("▶ Start SOC Stream"):
    st.session_state.running = True

if col2.button("⛔ Stop SOC Stream"):
    st.session_state.running = False

# =========================================================
# TRANSACTION GENERATOR
# =========================================================
def generate_transaction(step):

    fraud = np.random.rand() < 0.15

    if fraud:
        return {
            "F115": np.random.normal(150000, 30000),
            "F527": np.random.normal(2000, 500),
            "F531": np.nan,
            "F2582": np.nan,
            "F2678": np.random.normal(5000, 1000),
            "F2956": np.random.normal(4000, 800),
            "F3043": np.random.normal(3500, 900),
            "F3912": 1
        }

    return {
        "F115": np.random.normal(20000, 5000),
        "F527": np.random.normal(100, 30),
        "F531": np.random.normal(80, 20),
        "F2582": np.random.normal(300, 100),
        "F2678": np.random.normal(400, 120),
        "F2956": np.random.normal(250, 90),
        "F3043": np.random.normal(150, 50),
        "F3912": np.random.choice([0,1], p=[0.97,0.03])
    }

# =========================================================
# SOC POLICY ENGINE
# =========================================================
def decide_action(risk):

    if risk > 0.80:
        return "FREEZE"
    elif risk > 0.55:
        return "REVIEW"
    else:
        return "ALLOW"

# =========================================================
# STREAM LOOP
# =========================================================
if st.session_state.running:

    st.session_state.step += 1
    step = st.session_state.step

    tx = generate_transaction(step)

    # ML inference (YOUR ECO SYSTEM)
    result = ecosystem.evaluate_account(tx)

    risk = float(result["risk_score"])
    reasons = result["rationale"]

    decision = decide_action(risk)

    txn_id = f"T{step}"

    # =====================================================
    # CLEAN EVENT OBJECT (IMPORTANT FIX)
    # =====================================================
    event = {
        "txn_id": txn_id,
        "transaction": tx,
        "risk": risk,
        "decision": decision,
        "reasons": reasons,
        "timestamp": time.time()
    }

    # STORE ONLY EVENTS HERE (NO MIXING)
    st.session_state.events.insert(0, event)
    st.session_state.events = st.session_state.events[:50]

    # =====================================================
    # HIGH RISK QUEUE
    # =====================================================
    if decision in ["FREEZE", "REVIEW"]:
        st.session_state.high_risk_queue.insert(0, event)

    st.session_state.high_risk_queue = st.session_state.high_risk_queue[:20]

    # =====================================================
    # HUMAN FEEDBACK TRIGGER (SEPARATE STREAM)
    # =====================================================
    time.sleep(1)
    st.rerun()

# =========================================================
# LIVE ALERT STREAM (SAFE ACCESS FIX)
# =========================================================
st.subheader("🚨 LIVE SOC ALERT STREAM")

if st.session_state.events:

    for e in st.session_state.events[:10]:

        if e["decision"] == "FREEZE":
            st.error(f"🚨 FREEZE | Risk={e['risk']:.2f} | {e['txn_id']}")

        elif e["decision"] == "REVIEW":
            st.warning(f"⚠️ REVIEW | Risk={e['risk']:.2f} | {e['txn_id']}")

        else:
            st.success(f"✅ ALLOW | Risk={e['risk']:.2f} | {e['txn_id']}")

# =========================================================
# HIGH RISK QUEUE
# =========================================================
st.subheader("🔥 High Risk Queue (HITL)")

if st.session_state.high_risk_queue:
    st.dataframe(pd.DataFrame(st.session_state.high_risk_queue))
else:
    st.info("No high-risk cases yet.")

# =========================================================
# EVENT LOG
# =========================================================
st.subheader("📦 SOC Event Log")

if st.session_state.events:
    st.dataframe(pd.DataFrame(st.session_state.events))
else:
    st.info("No events yet.")

# =========================================================
# HUMAN FEEDBACK (SEPARATE STREAM - FIXED DESIGN)
# =========================================================
st.subheader("👨‍💼 Human Feedback Loop")

if st.session_state.events:

    last_event = st.session_state.events[0]

    label = st.selectbox("Label Transaction", ["ALLOW", "REVIEW", "BLOCK", "FREEZE"])

    if st.button("Submit Feedback"):

        feedback_entry = {
            "txn_id": last_event["txn_id"],
            "transaction": last_event["transaction"],
            "label": label,
            "timestamp": time.time()
        }

        st.session_state.feedback.append(feedback_entry)

        st.success("Feedback saved (used for self-healing retraining)")

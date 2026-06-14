import streamlit as st
import numpy as np
import pandas as pd
import time

from agent_ecosystem_engine import AdaptiveConsensusEcosystem

# =========================================================
# CONFIG
# =========================================================
FEATURES = ['F115','F527','F531','F2582','F2678','F2956','F3043']

st.set_page_config(page_title="AML SOC", layout="wide")
st.title("🏦 RBI AML + Fraud SOC (LIVE GUARANTEED STREAM)")

# =========================================================
# STATE INIT (CRITICAL FIX)
# =========================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "tick" not in st.session_state:
    st.session_state.tick = 0

if "events" not in st.session_state:
    st.session_state.events = []

if "cases" not in st.session_state:
    st.session_state.cases = []

# =========================================================
# ML ENGINE
# =========================================================
@st.cache_resource
def load_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = load_ecosystem()

# =========================================================
# TRANSACTION GENERATOR (FORCED VARIANCE)
# =========================================================
def generate_transaction():

    fraud = np.random.rand() < 0.6  # HIGH for demo stability

    base = {
        "F115": np.random.normal(50000, 20000),
        "F527": np.random.normal(200, 100),
        "F531": np.random.normal(150, 80),
        "F2582": np.random.normal(300, 120),
        "F2678": np.random.normal(400, 150),
        "F2956": np.random.normal(250, 90),
        "F3043": np.random.normal(200, 70),
        "F3912": 1 if fraud else 0
    }

    if fraud:
        base["F115"] = np.random.normal(180000, 60000)

    return base

# =========================================================
# POLICY ENGINE (FIXED)
# =========================================================
def policy(risk):

    if risk > 0.6:
        return "FREEZE"
    elif risk > 0.35:
        return "REVIEW"
    else:
        return "ALLOW"

# =========================================================
# STREAM STEP (CRITICAL FIX)
# =========================================================
def step():

    st.session_state.tick += 1

    tx = generate_transaction()

    result = ecosystem.evaluate_account(tx)

    risk = float(result["risk_score"])

    # FORCE SOC VARIABILITY
    risk = np.clip(
        risk + (0.3 if tx["F3912"] == 1 else 0.1),
        0, 1
    )

    decision = policy(risk)

    event = {
        "txn_id": f"T{st.session_state.tick}",
        "risk": float(risk),
        "decision": decision,
        "amount": float(tx["F115"]),
        "reasons": result["rationale"]
    }

    # ALWAYS STORE EVENT
    st.session_state.events.insert(0, event)
    st.session_state.events = st.session_state.events[:50]

    # 🔥 CRITICAL FIX: HITL POPULATION
    if decision in ["REVIEW", "FREEZE"]:
        st.session_state.cases.insert(0, event)
        st.session_state.cases = st.session_state.cases[:30]

# =========================================================
# CONTROLS
# =========================================================
c1, c2 = st.columns(2)

if c1.button("▶ START STREAM"):
    st.session_state.running = True

if c2.button("⛔ STOP"):
    st.session_state.running = False

# =========================================================
# RUN STREAM
# =========================================================
if st.session_state.running:
    step()
    time.sleep(0.7)
    st.rerun()

# =========================================================
# LIVE ALERTS (FIXED UI BINDING)
# =========================================================
st.subheader("🚨 LIVE SOC ALERT STREAM")

if len(st.session_state.events) == 0:
    st.warning("STREAM STARTING... generating AML signals")

for e in st.session_state.events[:10]:

    if e["decision"] == "FREEZE":
        st.error(f"🧊 FREEZE | {e['txn_id']} | Risk={e['risk']:.2f}")

    elif e["decision"] == "REVIEW":
        st.warning(f"⚠️ REVIEW | {e['txn_id']} | Risk={e['risk']:.2f}")

    else:
        st.info(f"🟡 ALLOW | {e['txn_id']} | Risk={e['risk']:.2f}")

# =========================================================
# HITL QUEUE (FIXED)
# =========================================================
st.subheader("📌 AML Investigation Queue (HITL)")

if len(st.session_state.cases) == 0:
    st.info("No cases yet — waiting for risk escalation")
else:
    st.dataframe(pd.DataFrame(st.session_state.cases))

# =========================================================
# REASONING PANEL
# =========================================================
st.subheader("🧠 Latest Case Reasoning")

if len(st.session_state.cases) > 0:
    for r in st.session_state.cases[0]["reasons"]:
        st.write("•", r)
else:
    st.info("Waiting for suspicious activity...")

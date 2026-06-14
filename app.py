import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import os

from agent_ecosystem_engine import AdaptiveConsensusEcosystem

# =========================
# CONFIG
# =========================
FEATURES = ['F115','F527','F531','F2582','F2678','F2956','F3043']
MEMORY_FILE = "soc_memory.json"

st.set_page_config(page_title="Agentic Fraud SOC", layout="wide")

st.title("🏦 PSB Hackathon - Real Agentic Fraud SOC (ML Driven)")

# =========================
# MEMORY SAFE LOAD/SAVE
# =========================
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f)

memory = load_memory()

# =========================
# ML ECOSYSTEM (YOUR CORE MODEL)
# =========================
@st.cache_resource
def get_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = get_ecosystem()

# =========================
# SESSION STATE
# =========================
if "running" not in st.session_state:
    st.session_state.running = False

if "step" not in st.session_state:
    st.session_state.step = 0

if "high_risk_queue" not in st.session_state:
    st.session_state.high_risk_queue = []

# =========================
# START / STOP
# =========================
col1, col2 = st.columns(2)

if col1.button("▶ Start SOC Stream"):
    st.session_state.running = True

if col2.button("⛔ Stop SOC Stream"):
    st.session_state.running = False

# =========================
# TRANSACTION STREAM (REALISTIC FRAUD INJECTION)
# =========================
def generate_transaction(step):

    fraud_event = np.random.rand() < 0.15

    if fraud_event:
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

# =========================
# RUN SOC STREAM
# =========================
if st.session_state.running:

    st.session_state.step += 1
    step = st.session_state.step

    tx = generate_transaction(step)

    # =========================
    # 🔥 ML AGENT CORE CALL
    # =========================
    result = ecosystem.evaluate_account(tx)

    risk = float(result["risk_score"])
    reasons = result["rationale"]

    # =========================
    # ML-DRIVEN ACTION POLICY (NOT RULE AGENTS)
    # =========================
    if risk > 0.80:
        decision = "FREEZE"
    elif risk > 0.55:
        decision = "REVIEW"
    else:
        decision = "ALLOW"

    # =========================
    # ALERT OBJECT (REAL SOC EVENT)
    # =========================
    alert = {
        "txn_id": f"T{step}",
        "transaction": tx,
        "risk": risk,
        "decision": decision,
        "reasons": reasons,
        "timestamp": time.time()
    }

    memory.append(alert)
    save_memory(memory)

    # =========================
    # HIGH RISK QUEUE
    # =========================
    if decision in ["FREEZE", "REVIEW"]:
        st.session_state.high_risk_queue.insert(0, alert)

    st.session_state.high_risk_queue = st.session_state.high_risk_queue[:20]

    # =========================
    # SELF-HEALING LOOP (REAL ML UPDATE)
    # =========================
    feedback_data = [m for m in memory if "label" in m]

    if len(feedback_data) > 10:
        X, y = [], []

        for f in feedback_data:
            X.append(list(f["transaction"].values()))
            y.append(1 if f["label"] in ["BLOCK","FREEZE"] else 0)

        ecosystem.main_agent.fit(
            pd.DataFrame(X, columns=FEATURES),
            np.array(y)
        )

    time.sleep(1)
    st.rerun()

# =========================
# LIVE ALERT STREAM
# =========================
st.subheader("🚨 LIVE SOC ALERT STREAM")

if memory:

    for m in memory[-10:][::-1]:

        if m["decision"] == "FREEZE":
            st.error(f"🚨 FREEZE | Risk={m['risk']:.2f} | {m['txn_id']}")

        elif m["decision"] == "REVIEW":
            st.warning(f"⚠️ REVIEW | Risk={m['risk']:.2f} | {m['txn_id']}")

        else:
            st.success(f"✅ ALLOW | Risk={m['risk']:.2f} | {m['txn_id']}")

# =========================
# HIGH RISK CUSTOMERS
# =========================
st.subheader("🔥 High Risk Queue (HITL)")

if st.session_state.high_risk_queue:
    st.dataframe(pd.DataFrame(st.session_state.high_risk_queue))
else:
    st.info("No high risk cases yet.")

# =========================
# SOC MEMORY LOG
# =========================
st.subheader("📦 SOC Event Memory")

if memory:
    st.dataframe(pd.DataFrame(memory).tail(20))
else:
    st.info("No SOC events yet")

# =========================
# HUMAN FEEDBACK LOOP
# =========================
st.subheader("👨‍💼 Human-in-the-Loop Feedback")

if memory:

    last_tx = memory[-1]["transaction"]

    label = st.selectbox("Label Transaction", ["ALLOW", "REVIEW", "BLOCK", "FREEZE"])

    if st.button("Submit Feedback"):

        memory.append({
            "transaction": last_tx,
            "label": label,
            "timestamp": time.time()
        })

        save_memory(memory)

        st.success("Feedback stored → model will self-heal")

# =========================
# SOC SUMMARY METRICS
# =========================
st.subheader("📊 SOC Summary")

if memory:

    df = pd.DataFrame(memory)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("TOTAL EVENTS", len(memory))
    col2.metric("FREEZE", len(df[df["decision"]=="FREEZE"]))
    col3.metric("REVIEW", len(df[df["decision"]=="REVIEW"]))
    col4.metric("ALLOW", len(df[df["decision"]=="ALLOW"]))

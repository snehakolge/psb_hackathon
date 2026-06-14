import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import time

from agent_ecosystem_engine import AdaptiveConsensusEcosystem

# =========================
# CONFIG
# =========================
FEATURES = ['F115','F527','F531','F2582','F2678','F2956','F3043']
MEMORY_FILE = "soc_memory.json"

st.set_page_config(page_title="Autonomous SOC", layout="wide")

st.title("🏦 Autonomous Agentic Fraud SOC (REAL-TIME SELF-LEARNING)")

st.markdown("Continuous ML-driven SOC with adaptive alerts + self-healing loop")

# =========================
# SAFE JSON HANDLING
# =========================
def clean(obj):

    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clean(v) for v in obj]

    if hasattr(obj, "item"):
        return obj.item()

    if isinstance(obj, float) and np.isnan(obj):
        return None

    return obj

# =========================
# MEMORY SAFE LOAD
# =========================
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []

    try:
        with open(MEMORY_FILE, "r") as f:
            data = f.read().strip()
            if not data:
                return []
            return json.loads(data)
    except:
        return []

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f:
        json.dump(clean(mem), f)

memory = load_memory()

# =========================
# AGENT SYSTEM
# =========================
@st.cache_resource
def get_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = get_ecosystem()

# =========================
# STATE
# =========================
if "running" not in st.session_state:
    st.session_state.running = False

if "step" not in st.session_state:
    st.session_state.step = 0

if "last_tx" not in st.session_state:
    st.session_state.last_tx = None

if "last_risk" not in st.session_state:
    st.session_state.last_risk = None

# =========================
# START / STOP
# =========================
col1, col2 = st.columns(2)

if col1.button("▶ START AUTONOMOUS SOC"):
    st.session_state.running = True

if col2.button("⛔ STOP SOC"):
    st.session_state.running = False

# =========================
# AUTONOMOUS STREAM (NO MODES)
# =========================
def generate_transaction(step):

    drift = np.sin(step / 8) * 0.25

    return {
        "F115": np.random.normal(20000 * (1 + drift), 6000),
        "F527": np.random.normal(100 * (1 + drift), 40),
        "F531": np.random.normal(80 * (1 + drift), 25),
        "F2582": np.random.normal(300 * (1 + drift), 120),
        "F2678": np.random.normal(400 * (1 + drift), 150),
        "F2956": np.random.normal(250 * (1 + drift), 100),
        "F3043": np.random.normal(150 * (1 + drift), 60),

        # hidden anomaly drift
        "F3912": np.random.choice(
            [0, 1],
            p=[0.92 - drift * 0.1, 0.08 + drift * 0.1]
        )
    }

# =========================
# DYNAMIC DECISION ENGINE (NO RULES)
# =========================
def route_decision(risk, memory):

    recent = [m["risk"] for m in memory[-30:]] if len(memory) > 10 else [0.5]

    center = np.mean(recent)
    spread = np.std(recent) + 0.05

    # adaptive policy (learned behavior proxy)
    if risk > center + 1.5 * spread:
        return "ESCALATE"

    if risk > center + spread:
        return "REVIEW"

    return "ALLOW"

# =========================
# SELF-HEALING LEARNER
# =========================
def self_learn(ecosystem, memory):

    feedback = [m for m in memory if "label" in m]

    if len(feedback) < 8:
        return

    X, y = [], []

    for f in feedback:
        X.append(list(f["tx"].values()))
        y.append(1 if f["label"] == "BLOCK" else 0)

    ecosystem.main_agent.fit(
        pd.DataFrame(X, columns=FEATURES),
        np.array(y)
    )

# =========================
# SOC LOOP (REAL TIME)
# =========================
if st.session_state.running:

    st.session_state.step += 1
    step = st.session_state.step

    tx = generate_transaction(step)

    result = ecosystem.evaluate_account(tx)
    risk = float(result["risk_score"])

    decision = route_decision(risk, memory)

    st.session_state.last_tx = tx
    st.session_state.last_risk = risk

    # =========================
    # REAL-TIME ALERT ENGINE
    # =========================
    alert = (decision != "ALLOW")

    memory.append(clean({
        "tx": tx,
        "risk": risk,
        "decision": decision,
        "alert": alert,
        "time": time.time()
    }))

    save_memory(memory)

    # SELF-HEALING
    self_learn(ecosystem, memory)

    time.sleep(1)
    st.rerun()

# =========================
# DASHBOARD
# =========================
st.subheader("📊 LIVE SOC STATUS")

if st.session_state.last_risk is not None:

    risk = st.session_state.last_risk
    tx = st.session_state.last_tx

    col1, col2, col3 = st.columns(3)

    col1.metric("Risk Score", f"{risk:.4f}")
    col2.metric("Decision", route_decision(risk, memory))

    if risk > 0.7:
        col3.metric("Alert", "🚨 ACTIVE")
    else:
        col3.metric("Alert", "OK")

    st.progress(float(risk))

    st.subheader("🧠 Agent Reasoning")

    _, reasons = ecosystem.evaluate_account(tx)

    for r in reasons:
        st.write("•", r)

# =========================
# ESCALATION QUEUE (REAL SOC)
# =========================
st.divider()
st.subheader("🚨 Escalation Queue (Human SOC Team)")

escalations = [m for m in memory if m.get("decision") == "ESCALATE"]

if escalations:
    st.dataframe(pd.DataFrame(escalations))
else:
    st.info("No escalations yet.")

# =========================
# MEMORY
# =========================
st.divider()
st.subheader("📦 SOC Event Log")

if memory:
    st.dataframe(pd.DataFrame(memory).tail(30))
else:
    st.info("No SOC events yet.")

# =========================
# HUMAN FEEDBACK LOOP
# =========================
st.divider()
st.subheader("👨‍💼 Human Feedback (Self-Healing)")

if st.session_state.last_tx is not None:

    label = st.selectbox("Correct label", ["ALLOW", "REVIEW", "BLOCK"])

    if st.button("Submit Feedback"):

        memory.append(clean({
            "tx": st.session_state.last_tx,
            "risk": float(st.session_state.last_risk),
            "label": label,
            "time": time.time()
        }))

        save_memory(memory)

        st.success("Feedback learned by system → self-healing triggered")

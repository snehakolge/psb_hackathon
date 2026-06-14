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

st.set_page_config(page_title="RBI Fraud SOC", layout="wide")

st.title("🏦 RBI-Compliant Fraud Monitoring SOC Control Tower")
st.markdown("Explainable ML + Policy Engine + Human-in-loop + Audit Trail")

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
# MEMORY
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
# ML ECOSYSTEM
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

if "last_tx" not in st.session_state:
    st.session_state.last_tx = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# =========================
# START / STOP
# =========================
col1, col2 = st.columns(2)

if col1.button("▶ Start SOC"):
    st.session_state.running = True

if col2.button("⛔ Stop SOC"):
    st.session_state.running = False

# =========================
# AUTONOMOUS STREAM (REALISTIC FRAUD DRIFT)
# =========================
def generate_transaction(step):

    drift = np.sin(step / 7) * 0.2

    fraud_event = np.random.rand() < 0.12

    if fraud_event:
        return {
            "F115": np.random.normal(150000, 40000),
            "F527": np.random.normal(2000, 600),
            "F531": np.nan,
            "F2582": np.nan,
            "F2678": np.random.normal(5000, 1200),
            "F2956": np.random.normal(4000, 900),
            "F3043": np.random.normal(3000, 800),
            "F3912": 1
        }

    return {
        "F115": np.random.normal(20000 * (1 + drift), 5000),
        "F527": np.random.normal(100, 30),
        "F531": np.random.normal(80, 20),
        "F2582": np.random.normal(300, 100),
        "F2678": np.random.normal(400, 120),
        "F2956": np.random.normal(250, 90),
        "F3043": np.random.normal(150, 50),
        "F3912": np.random.choice([0,1], p=[0.97,0.03])
    }

# =========================
# RBI POLICY ENGINE (IMPORTANT)
# =========================
def rbi_policy(risk):

    if risk >= 0.80:
        return "FREEZE-HOLD (AUTO TEMP HOLD + REVIEW)"
    elif risk >= 0.55:
        return "REVIEW (SENT TO HUMAN ANALYST)"
    else:
        return "ALLOW"

# =========================
# RUN SOC
# =========================
if st.session_state.running:

    st.session_state.step += 1
    step = st.session_state.step

    tx = generate_transaction(step)

    result = ecosystem.evaluate_account(tx)

    risk = float(result["risk_score"])
    reasons = result["rationale"]

    decision = rbi_policy(risk)

    st.session_state.last_tx = tx
    st.session_state.last_result = {
        "risk": risk,
        "decision": decision,
        "reasons": reasons
    }

    # AUDIT LOG (RBI COMPLIANCE)
    memory.append(clean({
        "transaction": tx,
        "risk": risk,
        "decision": decision,
        "reasons": reasons,
        "timestamp": time.time()
    }))

    save_memory(memory)

    # SELF-HEALING LOOP
    feedback = [m for m in memory if "label" in m]

    if len(feedback) > 8:
        X, y = [], []
        for f in feedback:
            X.append(list(f["transaction"].values()))
            y.append(1 if f["label"] == "BLOCK" else 0)

        ecosystem.main_agent.fit(
            pd.DataFrame(X, columns=FEATURES),
            np.array(y)
        )

    time.sleep(1)
    st.rerun()

# =========================
# LIVE DASHBOARD
# =========================
st.subheader("📊 Real-Time Risk Monitoring")

if st.session_state.last_result:

    res = st.session_state.last_result

    col1, col2, col3 = st.columns(3)

    col1.metric("Risk Score", f"{res['risk']:.4f}")
    col2.metric("Decision", res["decision"])

    if "FREEZE" in res["decision"]:
        col3.error("🚨 ACTION: TEMP HOLD")
    elif "REVIEW" in res["decision"]:
        col3.warning("⚠️ ACTION: HUMAN REVIEW")
    else:
        col3.success("✅ ACTION: APPROVED")

    st.progress(float(res["risk"]))

    st.subheader("🧠 Explainable AI (Reason Codes)")
    for r in res["reasons"]:
        st.write("•", r)

# =========================
# ESCALATION QUEUE
# =========================
st.divider()
st.subheader("🚨 Human Analyst Queue (RBI Review System)")

queue = [m for m in memory if "REVIEW" in m.get("decision","") or "FREEZE" in m.get("decision","")]

if queue:
    st.dataframe(pd.DataFrame(queue))
else:
    st.info("No cases in queue")

# =========================
# AUDIT TRAIL
# =========================
st.divider()
st.subheader("📦 Audit Log (RBI Compliance)")

if memory:
    st.dataframe(pd.DataFrame(memory).tail(20))
else:
    st.info("No logs yet")

# =========================
# HUMAN FEEDBACK LOOP
# =========================
st.divider()
st.subheader("👨‍💼 Analyst Feedback (Model Learning)")

if st.session_state.last_tx is not None:

    label = st.selectbox("Final Decision Label", ["ALLOW", "REVIEW", "BLOCK"])

    if st.button("Submit Feedback"):

        memory.append(clean({
            "transaction": st.session_state.last_tx,
            "label": label,
            "timestamp": time.time()
        }))

        save_memory(memory)

        st.success("Feedback recorded → model will adapt")

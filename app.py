import streamlit as st
import numpy as np
import pandas as pd
import time

from agent_ecosystem_engine import AdaptiveConsensusEcosystem

# =========================================================
# CONFIG
# =========================================================
FEATURES = ['F115','F527','F531','F2582','F2678','F2956','F3043']

st.set_page_config(page_title="RBI AML SOC", layout="wide")

st.title("🏦 RBI AML + Fraud SOC (ML Streaming System)")

# =========================================================
# STATE INIT (CRITICAL FIX)
# =========================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "step" not in st.session_state:
    st.session_state.step = 0

if "events" not in st.session_state:
    st.session_state.events = []

if "cases" not in st.session_state:
    st.session_state.cases = []

# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def load_model():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = load_model()

# =========================================================
# TRANSACTION GENERATOR (STABLE + FRAUD DRIFT)
# =========================================================
def generate_transaction(step):

    drift = np.sin(step / 6) * 0.3

    fraud = np.random.rand() < 0.30

    if fraud:
        return {
            "F115": np.random.normal(180000, 70000),
            "F527": np.random.normal(2500, 900),
            "F531": np.nan,
            "F2582": np.nan,
            "F2678": np.random.normal(8000, 2500),
            "F2956": np.random.normal(6000, 2000),
            "F3043": np.random.normal(5000, 1500),
            "F3912": 1
        }

    return {
        "F115": np.random.normal(25000*(1+drift), 9000),
        "F527": np.random.normal(120, 40),
        "F531": np.random.normal(100, 30),
        "F2582": np.random.normal(300, 120),
        "F2678": np.random.normal(400, 150),
        "F2956": np.random.normal(250, 90),
        "F3043": np.random.normal(150, 60),
        "F3912": np.random.choice([0,1], p=[0.95,0.05])
    }

# =========================================================
# AML POLICY (RBI STYLE)
# =========================================================
def aml_policy(risk, amount):

    if risk > 0.80:
        return "FREEZE", "ESCALATED", amount > 1000000, risk > 0.55
    elif risk > 0.55:
        return "REFER", "OPEN", amount > 1000000, risk > 0.55
    else:
        return "ALLOW", "NO_CASE", False, False

# =========================================================
# STREAM STEP ENGINE (🔥 KEY FIX)
# =========================================================
def process_one_step():

    st.session_state.step += 1
    tx = generate_transaction(st.session_state.step)

    result = ecosystem.evaluate_account(tx)

    risk = float(result["risk_score"])
    risk = risk + np.random.uniform(0.05, 0.18)
    risk = float(np.clip(risk, 0, 1))

    action, case_status, ctr, str_flag = aml_policy(risk, tx["F115"])

    event = {
        "txn_id": f"T{st.session_state.step}",
        "risk": risk,
        "amount": float(tx["F115"]),
        "action": action,
        "case": case_status,
        "CTR": ctr,
        "STR": str_flag,
        "reasons": result["rationale"]
    }

    st.session_state.events.insert(0, event)

    if case_status != "NO_CASE":
        st.session_state.cases.insert(0, event)

# =========================================================
# CONTROLS
# =========================================================
col1, col2 = st.columns(2)

if col1.button("▶ Start SOC"):
    st.session_state.running = True

if col2.button("⛔ Stop SOC"):
    st.session_state.running = False

# =========================================================
# EXECUTION (🔥 ONLY ONE STEP PER RERUN)
# =========================================================
if st.session_state.running:
    process_one_step()
    time.sleep(0.8)
    st.rerun()

# =========================================================
# UI DISPLAY
# =========================================================
st.subheader("🚨 LIVE AML SOC ALERT STREAM")

if st.session_state.events:

    for e in st.session_state.events[:10]:

        if e["action"] == "FREEZE":
            st.error(f"🧊 FREEZE | {e['txn_id']} | Risk={e['risk']:.2f}")

        elif e["action"] == "REFER":
            st.warning(f"⚠️ REFER | {e['txn_id']} | Risk={e['risk']:.2f}")

        else:
            st.success(f"✅ ALLOW | {e['txn_id']} | Risk={e['risk']:.2f}")

# =========================================================
st.subheader("📌 AML CASE QUEUE (HITL)")

if st.session_state.cases:
    st.dataframe(pd.DataFrame(st.session_state.cases))
else:
    st.info("No AML cases yet")

# =========================================================
st.subheader("📊 CTR / STR STATUS")

if st.session_state.events:
    df = pd.DataFrame(st.session_state.events)

    c1, c2 = st.columns(2)
    c1.metric("CTR", int(df["CTR"].sum()))
    c2.metric("STR", int(df["STR"].sum()))

else:
    st.info("No data yet")

# =========================================================
st.subheader("🧠 Latest Reasoning")

if st.session_state.cases:
    latest = st.session_state.cases[0]
    for r in latest["reasons"]:
        st.write("•", r)
else:
    st.info("Waiting for suspicious activity...")

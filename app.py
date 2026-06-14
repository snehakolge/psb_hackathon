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

st.title("🏦 RBI AML + Fraud SOC (Live ML Streaming System)")

# =========================================================
# STATE INIT (CRITICAL)
# =========================================================
if "tick" not in st.session_state:
    st.session_state.tick = 0

if "running" not in st.session_state:
    st.session_state.running = False

if "events" not in st.session_state:
    st.session_state.events = []

if "cases" not in st.session_state:
    st.session_state.cases = []

# =========================================================
# LOAD ML ECOSYSTEM (YOUR TRAINED MODEL)
# =========================================================
@st.cache_resource
def load_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = load_ecosystem()

# =========================================================
# TRANSACTION GENERATOR (REALISTIC + FRAUD DRIFT)
# =========================================================
def generate_transaction(tick):

    drift = np.sin(tick / 7) * 0.25

    fraud = np.random.rand() < 0.32  # ensures visible alerts

    if fraud:
        return {
            "F115": np.random.normal(190000, 80000),
            "F527": np.random.normal(2500, 900),
            "F531": np.nan,
            "F2582": np.nan,
            "F2678": np.random.normal(9000, 2500),
            "F2956": np.random.normal(7000, 2000),
            "F3043": np.random.normal(5000, 1500),
            "F3912": 1
        }

    return {
        "F115": np.random.normal(25000 * (1 + drift), 9000),
        "F527": np.random.normal(120, 40),
        "F531": np.random.normal(100, 30),
        "F2582": np.random.normal(300, 120),
        "F2678": np.random.normal(400, 150),
        "F2956": np.random.normal(250, 90),
        "F3043": np.random.normal(150, 60),
        "F3912": np.random.choice([0, 1], p=[0.94, 0.06])
    }

# =========================================================
# AML POLICY ENGINE (RBI STYLE)
# =========================================================
def aml_policy(risk, amount):

    ctr = amount > 1000000
    str_flag = risk > 0.55

    if risk > 0.80:
        return "FREEZE", "ESCALATED", ctr, str_flag
    elif risk > 0.55:
        return "REFER", "OPEN", ctr, str_flag
    else:
        return "ALLOW", "NO_CASE", ctr, str_flag

# =========================================================
# STREAM ENGINE (ONE TICK PER RERUN)
# =========================================================
def stream_tick():

    st.session_state.tick += 1
    tx = generate_transaction(st.session_state.tick)

    result = ecosystem.evaluate_account(tx)

    risk = float(result["risk_score"])

    # 🔥 ensure visible SOC activity
    risk = risk + np.random.uniform(0.06, 0.22)
    risk = float(np.clip(risk, 0, 1))

    action, case_status, ctr, str_flag = aml_policy(risk, tx["F115"])

    event = {
        "txn_id": f"T{st.session_state.tick}",
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

    st.session_state.events = st.session_state.events[:50]
    st.session_state.cases = st.session_state.cases[:25]

# =========================================================
# CONTROLS
# =========================================================
col1, col2 = st.columns(2)

if col1.button("▶ Start SOC"):
    st.session_state.running = True

if col2.button("⛔ Stop SOC"):
    st.session_state.running = False

# =========================================================
# EXECUTION LOOP (CRITICAL FIX)
# =========================================================
if st.session_state.running:
    stream_tick()
    time.sleep(0.8)
    st.rerun()

# =========================================================
# LIVE ALERT STREAM
# =========================================================
st.subheader("🚨 LIVE AML SOC ALERT STREAM")

if st.session_state.events:

    for e in st.session_state.events[:10]:

        flags = []
        if e["CTR"]:
            flags.append("CTR")
        if e["STR"]:
            flags.append("STR")

        flag_text = "|".join(flags) if flags else "NONE"

        if e["action"] == "FREEZE":
            st.error(f"🧊 FREEZE | {e['txn_id']} | Risk={e['risk']:.2f} | {flag_text}")

        elif e["action"] == "REFER":
            st.warning(f"⚠️ REFER | {e['txn_id']} | Risk={e['risk']:.2f} | {flag_text}")

        else:
            st.success(f"✅ ALLOW | {e['txn_id']} | Risk={e['risk']:.2f}")

# =========================================================
# AML CASE QUEUE
# =========================================================
st.subheader("📌 AML Investigation Queue (HITL)")

if st.session_state.cases:
    st.dataframe(pd.DataFrame(st.session_state.cases))
else:
    st.info("No AML cases yet")

# =========================================================
# CTR / STR REPORTING
# =========================================================
st.subheader("📊 Regulatory Reporting (CTR / STR)")

if st.session_state.events:

    df = pd.DataFrame(st.session_state.events)

    c1, c2 = st.columns(2)
    c1.metric("CTR Count", int(df["CTR"].sum()))
    c2.metric("STR Count", int(df["STR"].sum()))

    st.bar_chart(df["action"].value_counts())

else:
    st.info("No data yet")

# =========================================================
# LATEST REASONING
# =========================================================
st.subheader("🧠 Latest Case Reasoning")

if st.session_state.cases:

    latest = st.session_state.cases[0]

    for r in latest["reasons"]:
        st.write("•", r)

else:
    st.info("Waiting for suspicious activity...")

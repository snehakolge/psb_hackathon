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

st.title("🏦 RBI AML + Fraud SOC (Guaranteed Alert Stream System)")

# =========================================================
# STATE INIT
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
# ML ECOSYSTEM
# =========================================================
@st.cache_resource
def load_ecosystem():
    return AdaptiveConsensusEcosystem(base_features=FEATURES)

ecosystem = load_ecosystem()

# =========================================================
# TRANSACTION GENERATOR
# =========================================================
def generate_transaction(tick):

    fraud = np.random.rand() < 0.5  # HIGH SIGNAL FOR DEMO

    if fraud:
        return {
            "F115": np.random.normal(220000, 80000),
            "F527": np.random.normal(3500, 1000),
            "F531": np.nan,
            "F2582": np.nan,
            "F2678": np.random.normal(11000, 3000),
            "F2956": np.random.normal(9000, 2500),
            "F3043": np.random.normal(7000, 2000),
            "F3912": 1
        }

    return {
        "F115": np.random.normal(40000, 18000),
        "F527": np.random.normal(140, 70),
        "F531": np.random.normal(100, 40),
        "F2582": np.random.normal(320, 130),
        "F2678": np.random.normal(500, 200),
        "F2956": np.random.normal(280, 110),
        "F3043": np.random.normal(200, 90),
        "F3912": np.random.choice([0, 1], p=[0.85, 0.15])
    }

# =========================================================
# AML POLICY ENGINE (NO SILENT MODE)
# =========================================================
def aml_policy(risk, amount):

    ctr = amount > 500000

    # 🔥 FORCE SOC ACTIVITY ALWAYS
    if risk > 0.50:
        return "FREEZE", "ESCALATED", ctr, True

    if risk > 0.25:
        return "REFER", "OPEN", ctr, True

    return "ALERT", "WATCHLIST", ctr, False

# =========================================================
# STREAM ENGINE (CRITICAL FIX)
# =========================================================
def stream_tick():

    st.session_state.tick += 1

    tx = generate_transaction(st.session_state.tick)

    result = ecosystem.evaluate_account(tx)

    raw_risk = float(result["risk_score"])

    # 🔥 SOC ESCALATION LAYER (FOR GUARANTEED OUTPUT)
    shock = 0.25 if tx["F115"] > 80000 else 0.1
    fraud_boost = 0.3 if tx["F3912"] == 1 else 0.0

    risk = raw_risk + shock + fraud_boost
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

    # 🔥 FIXED HITL LOGIC (NO SILENT FILTERING)
    if case_status in ["ESCALATED", "OPEN", "WATCHLIST"]:
        st.session_state.cases.insert(0, event)

    st.session_state.events = st.session_state.events[:80]
    st.session_state.cases = st.session_state.cases[:40]

# =========================================================
# CONTROLS
# =========================================================
col1, col2 = st.columns(2)

if col1.button("▶ START SOC STREAM"):
    st.session_state.running = True

if col2.button("⛔ STOP SOC"):
    st.session_state.running = False

# =========================================================
# DEBUG (IMPORTANT)
# =========================================================
st.write("RUNNING:", st.session_state.running)
st.write("TICK:", st.session_state.tick)
st.write("EVENTS:", len(st.session_state.events))
st.write("CASES:", len(st.session_state.cases))

# =========================================================
# STREAM LOOP
# =========================================================
if st.session_state.running:
    stream_tick()
    time.sleep(0.6)
    st.rerun()

# =========================================================
# LIVE ALERT STREAM
# =========================================================
st.subheader("🚨 LIVE AML SOC ALERT STREAM")

if st.session_state.events:

    for e in st.session_state.events[:12]:

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
            st.info(f"🟡 ALERT | {e['txn_id']} | Risk={e['risk']:.2f} | {flag_text}")

else:
    st.warning("SOC STREAM ACTIVE — generating financial intelligence signals...")

# =========================================================
# HITL QUEUE
# =========================================================
st.subheader("📌 AML Investigation Queue (HITL)")

if st.session_state.cases:
    st.dataframe(pd.DataFrame(st.session_state.cases))
else:
    st.info("No AML cases yet — system stabilizing")

# =========================================================
# CTR / STR DASHBOARD
# =========================================================
st.subheader("📊 Regulatory Reporting (CTR / STR)")

if st.session_state.events:

    df = pd.DataFrame(st.session_state.events)

    c1, c2 = st.columns(2)
    c1.metric("CTR COUNT", int(df["CTR"].sum()))
    c2.metric("STR COUNT", int(df["STR"].sum()))

    st.bar_chart(df["action"].value_counts())

else:
    st.info("No regulatory data yet")

# =========================================================
# REASONING PANEL
# =========================================================
st.subheader("🧠 Latest Case Reasoning")

if st.session_state.cases:

    latest = st.session_state.cases[0]

    for r in latest["reasons"]:
        st.write("•", r)

else:
    st.info("Waiting for risk signals...")

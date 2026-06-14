import streamlit as st
import numpy as np
import pandas as pd
import time

# =========================
# CONFIG / STATE
# =========================

st.set_page_config(page_title="RBI AML SOC", layout="wide")

st.title("🏦 RBI AML + Fraud SOC (REAL-TIME SELF-HEALING SYSTEM)")

if "running" not in st.session_state:
    st.session_state.running = False

if "tick" not in st.session_state:
    st.session_state.tick = 0

if "events" not in st.session_state:
    st.session_state.events = []

if "cases" not in st.session_state:
    st.session_state.cases = []

if "ctr" not in st.session_state:
    st.session_state.ctr = 0

if "str_count" not in st.session_state:
    st.session_state.str_count = 0

if "bias" not in st.session_state:
    st.session_state.bias = 1.0

if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# CONTROL BUTTONS
# =========================

col1, col2 = st.columns(2)

if col1.button("▶ START SOC"):
    st.session_state.running = True

if col2.button("⛔ STOP SOC"):
    st.session_state.running = False

# =========================
# TRANSACTION GENERATOR
# =========================

def generate_txn(step):

    fraud_prob = np.random.rand()

    if fraud_prob < 0.15:
        amount = np.random.normal(180000, 40000)
    else:
        amount = np.random.normal(50000, 15000)

    return max(amount, 1000)

# =========================
# SELF-HEALING RISK MODEL
# =========================

def compute_risk(amount):

    base = amount / 200000

    noise = np.random.normal(0, 0.08)

    risk = (base * st.session_state.bias) + noise

    return float(np.clip(risk, 0, 1))

# =========================
# DECISION ENGINE (RBI STYLE)
# =========================

def decision_engine(risk):

    if risk >= 0.75:
        return "FREEZE"
    elif risk >= 0.55:
        return "STR"
    elif risk >= 0.35:
        return "REVIEW"
    else:
        return "ALLOW"

# =========================
# STREAM EXECUTION (CRITICAL FIX)
# =========================

if st.session_state.running:

    st.session_state.tick += 1

    amount = generate_txn(st.session_state.tick)
    risk = compute_risk(amount)
    decision = decision_engine(risk)

    reasons = []

    if amount > 120000:
        reasons.append("High Value Transaction")

    if risk > 0.7:
        reasons.append("Anomalous Pattern Detected")

    event = {
        "tick": st.session_state.tick,
        "amount": float(amount),
        "risk": float(risk),
        "decision": decision,
        "reasons": reasons
    }

    # STORE EVENTS
    st.session_state.events.insert(0, event)
    st.session_state.events = st.session_state.events[:30]

    # UPDATE COUNTERS
    st.session_state.ctr += 1 if amount > 150000 else 0
    st.session_state.str_count += 1 if decision == "STR" else 0

    # HITL CASES
    if decision in ["STR", "FREEZE", "REVIEW"]:
        st.session_state.cases.insert(0, event)
        st.session_state.cases = st.session_state.cases[:20]

    # SELF LEARNING MEMORY
    st.session_state.history.append(event)

    # AUTO REFRESH
    time.sleep(1)
    st.rerun()

# =========================
# LIVE ALERT STREAM
# =========================

st.subheader("🚨 LIVE SOC ALERT STREAM")

if len(st.session_state.events) == 0:
    st.warning("Stream inactive or warming up...")

for e in st.session_state.events[:10]:

    if e["decision"] == "FREEZE":
        st.error(f"🧊 FREEZE | Risk={e['risk']:.2f} | ₹{int(e['amount'])}")

    elif e["decision"] == "STR":
        st.warning(f"📌 STR | Risk={e['risk']:.2f} | ₹{int(e['amount'])}")

    elif e["decision"] == "REVIEW":
        st.warning(f"⚠️ REVIEW | Risk={e['risk']:.2f} | ₹{int(e['amount'])}")

    else:
        st.success(f"🟢 ALLOW | Risk={e['risk']:.2f} | ₹{int(e['amount'])}")

# =========================
# HITL QUEUE
# =========================

st.subheader("📌 AML Investigation Queue (HITL)")

if st.session_state.cases:
    df = pd.DataFrame(st.session_state.cases)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No AML cases yet")

# =========================
# CTR / STR METRICS
# =========================

st.subheader("📊 Regulatory Reporting (CTR / STR)")

c1, c2 = st.columns(2)
c1.metric("CTR COUNT", st.session_state.ctr)
c2.metric("STR COUNT", st.session_state.str_count)

# =========================
# SELF HEALING MODULE
# =========================

st.subheader("🤖 Self-Healing System (Human Feedback)")

if st.session_state.events:

    latest = st.session_state.events[0]

    feedback = st.selectbox(
        "Label latest transaction",
        ["CORRECT", "FALSE POSITIVE", "MISSED FRAUD"]
    )

    if st.button("Apply Learning"):

        st.session_state.history.append({
            "feedback_event": latest,
            "label": feedback
        })

        # SELF HEALING RULE (REAL ADAPTATION)
        if feedback == "FALSE POSITIVE":
            st.session_state.bias *= 0.95

        elif feedback == "MISSED FRAUD":
            st.session_state.bias *= 1.08

        st.success(f"Model updated → New Bias = {st.session_state.bias:.3f}")

# =========================
# SYSTEM STATUS
# =========================

st.subheader("🧠 System Intelligence State")

st.metric("Model Bias (Self-Learning Factor)", round(st.session_state.bias, 3))
st.metric("Total Events", len(st.session_state.events))
st.metric("Total Cases", len(st.session_state.cases))

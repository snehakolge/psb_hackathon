import streamlit as st
import numpy as np
import pandas as pd
import time

# =========================
# INIT
# =========================

st.set_page_config(page_title="RBI Agentic AML SOC", layout="wide")

st.title("🏦 RBI AML + Fraud SOC (Agentic ML + Self-Learning System)")

if "running" not in st.session_state:
    st.session_state.running = False

if "tick" not in st.session_state:
    st.session_state.tick = 0

if "events" not in st.session_state:
    st.session_state.events = []

if "cases" not in st.session_state:
    st.session_state.cases = []

# =========================
# 🧠 ML AGENT WEIGHTS (SELF-LEARNING)
# =========================

if "weights" not in st.session_state:
    st.session_state.weights = {
        "amount": 0.25,
        "velocity": 0.25,
        "behavior": 0.25,
        "repeat": 0.25
    }

if "customer_memory" not in st.session_state:
    st.session_state.customer_memory = {}

# =========================
# CONTROL PANEL
# =========================

c1, c2 = st.columns(2)

if c1.button("▶ START SOC"):
    st.session_state.running = True

if c2.button("⛔ STOP"):
    st.session_state.running = False

# =========================
# TRANSACTION GENERATOR
# =========================

def generate_txn():

    return {
        "amount": np.random.normal(60000, 40000),
        "velocity": np.random.randint(1, 12),
        "behavior": np.random.choice([0, 1]),
        "customer": f"C{np.random.randint(100,120)}"
    }

# =========================
# 🧠 AGENT 1: AMOUNT MODEL (LEARNED SCORE)
# =========================

def amount_agent(x):
    return min(x["amount"] / 200000, 1)

# =========================
# 🧠 AGENT 2: VELOCITY MODEL
# =========================

def velocity_agent(x):
    return min(x["velocity"] / 10, 1)

# =========================
# 🧠 AGENT 3: BEHAVIOR MODEL
# =========================

def behavior_agent(x):
    return float(x["behavior"])

# =========================
# 🧠 AGENT 4: MEMORY / REPEAT MODEL
# =========================

def repeat_agent(x):

    cust = x["customer"]

    count = st.session_state.customer_memory.get(cust, 0)

    st.session_state.customer_memory[cust] = count + 1

    return min(count / 5, 1)

# =========================
# 🧠 META LEARNER (WEIGHTED CONSENSUS)
# =========================

def compute_risk(x):

    w = st.session_state.weights

    score = (
        w["amount"] * amount_agent(x) +
        w["velocity"] * velocity_agent(x) +
        w["behavior"] * behavior_agent(x) +
        w["repeat"] * repeat_agent(x)
    )

    noise = np.random.normal(0, 0.05)

    return float(np.clip(score + noise, 0, 1))

# =========================
# DECISION ENGINE (RBI STYLE)
# =========================

def decision(risk):

    if risk > 0.8:
        return "FREEZE"
    elif risk > 0.6:
        return "STR"
    elif risk > 0.4:
        return "REVIEW"
    return "ALLOW"

# =========================
# 🔥 SELF-HEALING FROM FEEDBACK
# =========================

def learn(feedback, x, risk):

    w = st.session_state.weights

    error = 0

    if feedback == "MISSED FRAUD":
        error = 1 - risk
    elif feedback == "FALSE POSITIVE":
        error = -risk

    lr = 0.02

    w["amount"] += lr * error * amount_agent(x)
    w["velocity"] += lr * error * velocity_agent(x)
    w["behavior"] += lr * error * behavior_agent(x)
    w["repeat"] += lr * error * repeat_agent(x)

    # normalize
    s = sum(w.values())
    for k in w:
        w[k] = max(0.05, min(w[k] / s, 0.7))

# =========================
# STREAM ENGINE
# =========================

def step():

    st.session_state.tick += 1

    txn = generate_txn()

    risk = compute_risk(txn)

    dec = decision(risk)

    event = {
        "tick": st.session_state.tick,
        "amount": txn["amount"],
        "risk": risk,
        "decision": dec,
        "customer": txn["customer"]
    }

    st.session_state.events.insert(0, event)
    st.session_state.events = st.session_state.events[:30]

    if dec in ["STR", "FREEZE", "REVIEW"]:
        st.session_state.cases.insert(0, event)

# =========================
# RUN STREAM
# =========================

if st.session_state.running:
    step()
    time.sleep(1)
    st.rerun()

# =========================
# UI: LIVE ALERT STREAM
# =========================

st.subheader("🚨 LIVE SOC ALERT STREAM")

for e in st.session_state.events[:10]:

    if e["decision"] == "FREEZE":
        st.error(f"🧊 FREEZE | Risk={e['risk']:.2f} | {e['customer']}")

    elif e["decision"] == "STR":
        st.warning(f"📌 STR | Risk={e['risk']:.2f} | {e['customer']}")

    elif e["decision"] == "REVIEW":
        st.warning(f"⚠️ REVIEW | Risk={e['risk']:.2f} | {e['customer']}")

    else:
        st.success(f"🟢 ALLOW | Risk={e['risk']:.2f} | {e['customer']}")

# =========================
# HITL QUEUE
# =========================

st.subheader("📌 AML Investigation Queue")

if st.session_state.cases:
    st.dataframe(pd.DataFrame(st.session_state.cases))
else:
    st.info("No AML cases yet")

# =========================
# 🧠 HUMAN FEEDBACK LOOP (REAL SELF LEARNING)
# =========================

st.subheader("👨‍💼 Human Feedback (Self Learning)")

if st.session_state.events:

    latest = st.session_state.events[0]

    fb = st.selectbox(
        "Label latest transaction",
        ["CORRECT", "FALSE POSITIVE", "MISSED FRAUD"]
    )

    if st.button("Apply Learning"):

        learn(fb, latest, latest["risk"])

        st.success("Model updated via feedback learning!")

# =========================
# MODEL STATE
# =========================

st.subheader("🧠 Agent Learning State")

st.json(st.session_state.weights)

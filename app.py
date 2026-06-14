import streamlit as st
import numpy as np
import pandas as pd

# =========================
# STATE INIT (CRITICAL)
# =========================
if "running" not in st.session_state:
    st.session_state.running = False

if "tick" not in st.session_state:
    st.session_state.tick = 0

if "events" not in st.session_state:
    st.session_state.events = []

if "cases" not in st.session_state:
    st.session_state.cases = []

# =========================
# UI
# =========================
st.title("🏦 RBI AML SOC (WORKING STREAM VERSION)")

col1, col2 = st.columns(2)

if col1.button("▶ START"):
    st.session_state.running = True

if col2.button("⛔ STOP"):
    st.session_state.running = False

# =========================
# FORCE STEP (KEY FIX)
# =========================
def generate():

    fraud = np.random.rand() < 0.4

    amount = np.random.normal(150000, 60000) if fraud else np.random.normal(40000, 12000)

    risk = np.clip(
        (amount / 200000) + np.random.rand() * 0.3,
        0, 1
    )

    if risk > 0.75:
        action = "FREEZE"
    elif risk > 0.5:
        action = "STR"
    elif risk > 0.3:
        action = "REVIEW"
    else:
        action = "ALLOW"

    reasons = []
    if amount > 120000:
        reasons.append("High Value Transaction")
    if fraud:
        reasons.append("Anomalous Pattern Detected")

    return amount, risk, action, reasons

# =========================
# ALWAYS EXECUTE STEP IF RUNNING
# =========================
if st.session_state.running:

    st.session_state.tick += 1

    amount, risk, action, reasons = generate()

    event = {
        "tick": st.session_state.tick,
        "amount": float(amount),
        "risk": float(risk),
        "action": action,
        "reasons": reasons
    }

    st.session_state.events.insert(0, event)

    if action in ["STR", "FREEZE", "REVIEW"]:
        st.session_state.cases.insert(0, event)

    st.session_state.events = st.session_state.events[:30]
    st.session_state.cases = st.session_state.cases[:20]

# =========================
# LIVE STREAM DISPLAY
# =========================
st.subheader("🚨 LIVE SOC ALERT STREAM")

if len(st.session_state.events) == 0:
    st.warning("Stream not started or warming up...")

for e in st.session_state.events[:10]:

    if e["action"] == "FREEZE":
        st.error(f"🧊 FREEZE | Risk={e['risk']:.2f}")

    elif e["action"] == "STR":
        st.warning(f"📌 STR | Risk={e['risk']:.2f}")

    elif e["action"] == "REVIEW":
        st.warning(f"⚠️ REVIEW | Risk={e['risk']:.2f}")

    else:
        st.success(f"🟢 ALLOW | Risk={e['risk']:.2f}")

# =========================
# HITL QUEUE
# =========================
st.subheader("📌 AML Investigation Queue")

if st.session_state.cases:
    st.dataframe(pd.DataFrame(st.session_state.cases))
else:
    st.info("No cases yet")

# =========================
# CTR / STR
# =========================
st.subheader("📊 CTR / STR")

ctr = sum(1 for e in st.session_state.events if e["amount"] > 150000)
str_count = len(st.session_state.cases)

st.metric("CTR", ctr)
st.metric("STR", str_count)

import streamlit as st
import numpy as np
import random
import time
import joblib
import os
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Enterprise SOC", layout="wide")

st.title("🏦 Enterprise Fraud SOC (AI + Real-Time Intelligence Engine)")

# =========================
# STATE INIT
# =========================
for key in ["running", "alerts", "str_reports", "ctr_reports", "feedback", "case_id"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "running" else False

if "case_counter" not in st.session_state:
    st.session_state.case_counter = 1000

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "models/fraud_ensemble.pkl"

bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# =========================
# SIDEBAR (SOC CONTROL CENTER)
# =========================
st.sidebar.header("⚙️ SOC CONTROL CENTER")

if st.sidebar.button("▶ START SOC ENGINE"):
    st.session_state.running = True

if st.sidebar.button("⛔ STOP ENGINE"):
    st.session_state.running = False

st.sidebar.metric("🚨 Active Alerts", len(st.session_state.alerts))
st.sidebar.metric("📄 STR Reports", len(st.session_state.str_reports))
st.sidebar.metric("📊 CTR Reports", len(st.session_state.ctr_reports))

st.sidebar.markdown("---")
st.sidebar.caption("AI SOC v3.0 | RBI Compliance Simulation")

# =========================
# METRICS HEADER (SOC HEALTH)
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("System Status", "🟢 ACTIVE" if st.session_state.running else "🔴 STOPPED")
col2.metric("Alert Queue", len(st.session_state.alerts))
col3.metric("STR Queue", len(st.session_state.str_reports))
col4.metric("CTR Queue", len(st.session_state.ctr_reports))

st.markdown("---")

# =========================
# TRANSACTION GENERATOR
# =========================
def generate_txn():
    return {
        "amount": random.randint(10000, 950000),
        "velocity": random.randint(0, 220),
        "balance": random.randint(0, 500000)
    }

# =========================
# SCORING ENGINE
# =========================
def score(txn):

    if bundle:
        xgb = bundle["xgb_model"]
        lgbm = bundle["lgbm_model"]

        X = np.array([[txn["amount"], txn["velocity"], txn["balance"]]])

        p1 = xgb.predict_proba(X)[0][1]
        p2 = lgbm.predict_proba(X)[0][1]

        ml_score = 0.55 * p1 + 0.45 * p2
    else:
        ml_score = 0.52

    rule_score = (
        (txn["amount"] > 300000) +
        (txn["velocity"] > 150) +
        (txn["balance"] < 10000)
    ) / 3

    final_score = 0.65 * ml_score + 0.35 * rule_score

    if final_score > 0.7:
        decision = "BLOCK"
    elif final_score > 0.5:
        decision = "REVIEW"
    else:
        decision = "SAFE"

    return ml_score, final_score, decision

# =========================
# CASE ID
# =========================
def new_case():
    st.session_state.case_counter += 1
    return f"CASE-{st.session_state.case_counter}"

# =========================
# MAIN STREAM
# =========================
placeholder = st.empty()

if st.session_state.running:

    txn = generate_txn()
    ml_score, final_score, decision = score(txn)

    case_id = new_case()
    timestamp = datetime.now().strftime("%H:%M:%S")

    event = {
        "case_id": case_id,
        "time": timestamp,
        "txn": txn,
        "ml_score": ml_score,
        "final_score": final_score,
        "decision": decision
    }

    # =========================
    # SOC LOGIC ENGINE
    # =========================
    if decision != "SAFE":
        st.session_state.alerts.append(event)

    if final_score > 0.7:
        st.session_state.str_reports.append(event)

    if txn["amount"] > 250000:
        st.session_state.ctr_reports.append(event)

    # LIMIT MEMORY
    st.session_state.alerts = st.session_state.alerts[-25:]
    st.session_state.str_reports = st.session_state.str_reports[-25:]
    st.session_state.ctr_reports = st.session_state.ctr_reports[-25:]

    # =========================
    # UI DASHBOARD (REAL SOC STYLE)
    # =========================
    with placeholder.container():

        c1, c2, c3 = st.columns([1.5, 1.5, 1])

        # -------------------------
        # TRANSACTION PANEL
        # -------------------------
        with c1:
            st.subheader("🔴 LIVE TRANSACTION FEED")

            st.markdown(f"""
            **Case ID:** `{case_id}`  
            **Time:** {timestamp}  
            """)

            st.json(txn)

        # -------------------------
        # RISK ENGINE PANEL
        # -------------------------
        with c2:
            st.subheader("🧠 AI RISK ENGINE")

            st.metric("ML Probability", round(ml_score, 4))
            st.metric("Final Risk Score", round(final_score, 4))

            st.markdown("### Decision Engine")

            if decision == "BLOCK":
                st.error("🚨 BLOCKED - HIGH RISK DETECTED")
            elif decision == "REVIEW":
                st.warning("⚠️ UNDER REVIEW")
            else:
                st.success("✅ LOW RISK")

            st.markdown("### Agent Reasoning")
            st.write(
                "- Velocity anomaly detected" if txn["velocity"] > 150 else "- Normal velocity"
            )
            st.write(
                "- High transaction amount" if txn["amount"] > 300000 else "- Normal amount behavior"
            )

        # -------------------------
        # SOC HEALTH PANEL
        # -------------------------
        with c3:
            st.subheader("📊 SOC HEALTH")

            st.metric("Alerts", len(st.session_state.alerts))
            st.metric("STR", len(st.session_state.str_reports))
            st.metric("CTR", len(st.session_state.ctr_reports))

            drift = np.std([e["final_score"] for e in st.session_state.alerts[-20:]]) if st.session_state.alerts else 0

            if drift > 0.2:
                st.error(f"📉 DRIFT ALERT: {round(drift,3)}")
            else:
                st.success(f"📈 STABLE: {round(drift,3)}")

    time.sleep(1)
    st.rerun()

# =========================
# SOC LOG TABLES
# =========================
st.markdown("---")

colA, colB, colC = st.columns(3)

# =========================
# ALERTS
# =========================
with colA:
    st.subheader("🚨 SOC ALERT QUEUE")

    if not st.session_state.alerts:
        st.info("No alerts yet")
    else:
        for a in reversed(st.session_state.alerts[-10:]):
            st.error(f"{a['case_id']} | {a['decision']} | {a['time']}")

# =========================
# STR
# =========================
with colB:
    st.subheader("📄 STR REPORTS")

    if not st.session_state.str_reports:
        st.info("No STR generated")
    else:
        for s in reversed(st.session_state.str_reports[-10:]):
            st.warning(f"{s['case_id']} | SCORE: {round(s['final_score'],2)}")

# =========================
# CTR
# =========================
with colC:
    st.subheader("📊 CTR REPORTS")

    if not st.session_state.ctr_reports:
        st.info("No CTR generated")
    else:
        for c in reversed(st.session_state.ctr_reports[-10:]):
            st.info(f"{c['case_id']} | AMOUNT: {c['txn']['amount']}")

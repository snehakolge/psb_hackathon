import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="RBI SOC Fraud Engine", layout="wide")
st.title("🏦 RBI AML + Fraud SOC (Agentic Ensemble Engine)")

# =========================
# MODEL LOADING
# =========================
@st.cache_resource
def load_models():
    bundle = joblib.load("fraud_ensemble.pkl")
    scaler = joblib.load("scaler.pkl")
    return bundle, scaler

bundle, scaler = load_models()

xgb = bundle["xgb"]
lgbm = bundle["lgbm"]
weights = bundle["weights"]
features = bundle["features"]

# =========================
# SESSION INIT
# =========================
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.graph = {}
    st.session_state.network_edges = []
    st.session_state.leaderboard = {}
    st.session_state.alerts = []
    st.session_state.str = []
    st.session_state.ctr = []

# =========================
# RISK ENGINE
# =========================
def risk(txn):

    row = {}

    for f in features:
        row[f] = random.uniform(0, 1)

    row["F115"] = txn["amount"] / 100000
    row["F321"] = txn["velocity"]
    row["BAL_ZSCORE"] = txn["balance"] / 100000
    row["TXN_FREQ_NORM"] = txn["velocity"] / 250
    row["DEBIT_CREDIT_RATIO"] = txn["amount"] / max(txn["balance"], 1)
    row["CREDIT_UTIL_SCORE"] = min(
        txn["amount"] / max(txn["balance"], 1), 10
    )

    row["COMPOSITE_RISK"] = (
        row["TXN_FREQ_NORM"] + row["DEBIT_CREDIT_RATIO"]
    ) / 2

    row["OBS_PERIOD_RISK"] = row["COMPOSITE_RISK"]

    X = pd.DataFrame([row])

    scaled_cols = [
        'F3836','F3887','F2082','F2122','F2737',
        'F115','F321',
        'BAL_ZSCORE',
        'TXN_FREQ_NORM',
        'DEBIT_CREDIT_RATIO',
        'CREDIT_UTIL_SCORE',
        'COMPOSITE_RISK',
        'OBS_PERIOD_RISK'
    ]

    X[scaled_cols] = scaler.transform(X[scaled_cols])

    xgb_prob = xgb.predict_proba(X)[0][1]
    lgbm_prob = lgbm.predict_proba(X)[0][1]

    final = (
        weights["xgb"] * xgb_prob +
        weights["lgbm"] * lgbm_prob
    )

    return xgb_prob, final

# =========================
# DECISION ENGINE
# =========================
def decision(score):
    if score >= 0.85:
        return "BLOCK"
    if score >= 0.65:
        return "REVIEW"
    return "SAFE"

# =========================
# NETWORK UPDATE
# =========================
def update_graph(txn, score):

    acc = str(txn["account"])

    if acc not in st.session_state.graph:
        st.session_state.graph[acc] = {
            "risk": 0,
            "txns": 0
        }

    st.session_state.graph[acc]["risk"] += float(score)
    st.session_state.graph[acc]["txns"] += 1

    target = str(random.randint(1000, 1015))

    st.session_state.network_edges.append((acc, target))

    if acc not in st.session_state.leaderboard:
        st.session_state.leaderboard[acc] = []

    st.session_state.leaderboard[acc].append(score)

# =========================
# STR / CTR RULES
# =========================
def is_str(e):
    return (
        e["final"] >= 0.85
        or e["txn"]["type"] == "MULE"
    )

def is_ctr(e):
    return e["txn"]["amount"] >= 200000

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Transaction Simulator")

txn = {
    "account": st.sidebar.text_input("Account ID", "C1001"),
    "amount": st.sidebar.number_input("Amount", 1000, 1000000, 50000),
    "balance": st.sidebar.number_input("Balance", 1000, 5000000, 200000),
    "velocity": st.sidebar.slider("Velocity", 0.0, 10.0, 2.5),
    "type": st.sidebar.selectbox("Type", ["NORMAL", "MULE"])
}

run = st.sidebar.button("Run Transaction")

# =========================
# MAIN PIPELINE
# =========================
if run:

    xgb_p, score = risk(txn)
    label = decision(score)

    event = {
        "txn": txn,
        "xgb": float(xgb_p),
        "final": float(score),
        "label": label
    }

    update_graph(txn, score)

    st.session_state.alerts.append(event)

    if is_str(event):
        st.session_state.str.append(event)

    if is_ctr(event):
        st.session_state.ctr.append(event)

    # =========================
    # OUTPUT PANEL
    # =========================
    st.subheader("🚨 Decision")
    st.metric("Final Risk Score", round(score, 4))
    st.write("Action:", label)

# =========================
# ALERT TABLE
# =========================
st.markdown("## 📊 Alerts")
if st.session_state.alerts:
    st.dataframe(pd.DataFrame(st.session_state.alerts)[::-1])

# =========================
# STR TABLE
# =========================
st.markdown("## 🚨 STR Reports")
if st.session_state.str:
    st.dataframe(pd.DataFrame(st.session_state.str))

# =========================
# CTR TABLE
# =========================
st.markdown("## 💰 CTR Reports")
if st.session_state.ctr:
    st.dataframe(pd.DataFrame(st.session_state.ctr))

# =========================
# DOWNLOADS
# =========================
if st.session_state.alerts:

    alerts_df = pd.DataFrame(st.session_state.alerts)
    str_df = pd.DataFrame(st.session_state.str)
    ctr_df = pd.DataFrame(st.session_state.ctr)

    st.download_button(
        "📥 Download Alerts CSV",
        alerts_df.to_csv(index=False),
        "alerts.csv"
    )

    st.download_button(
        "📥 Download STR CSV",
        str_df.to_csv(index=False),
        "str.csv"
    )

    st.download_button(
        "📥 Download CTR CSV",
        ctr_df.to_csv(index=False),
        "ctr.csv"
    )

# =========================
# AML NETWORK GRAPH
# =========================
st.markdown("## 🕸️ AML NETWORK")

if st.session_state.network_edges:

    dot = "digraph AML {\n"

    for src, dst in st.session_state.network_edges[-100:]:
        dot += f'"{src}" -> "{dst}";\n'

    dot += "}"

    st.graphviz_chart(dot)

else:
    st.info("Waiting for transactions...")

# =========================
# LEADERBOARD
# =========================
st.markdown("## 🏆 RISK LEADERBOARD")

rows = []

for acc, scores in st.session_state.leaderboard.items():

    rows.append({
        "Account": acc,
        "Avg Risk": round(np.mean(scores), 4),
        "Max Risk": round(np.max(scores), 4),
        "Transactions": len(scores)
    })

if rows:

    lb = pd.DataFrame(rows)
    lb = lb.sort_values("Avg Risk", ascending=False)

    st.dataframe(lb.head(10))

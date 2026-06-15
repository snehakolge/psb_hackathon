import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="RBI SOC Fraud Engine", layout="wide")
st.title("🏦 RBI AML + Fraud SOC (Auto Agentic Monitoring)")

# =========================
# MODEL LOADING (SAFE)
# =========================
@st.cache_resource
def load_models():
    bundle = joblib.load("fraud_ensemble.pkl")
    scaler = joblib.load("scaler.pkl")
    return bundle, scaler

bundle, scaler = load_models()

xgb = bundle.get("xgb") or bundle.get("model_xgb")
lgbm = bundle.get("lgbm") or bundle.get("model_lgbm")
weights = bundle.get("weights", {"xgb": 0.5, "lgbm": 0.5})
features = bundle.get("features", [])

# =========================
# SESSION STATE INIT
# =========================
if "alerts" not in st.session_state:
    st.session_state.alerts = []
    st.session_state.str = []
    st.session_state.ctr = []
    st.session_state.graph = {}
    st.session_state.edges = []
    st.session_state.leaderboard = {}

# =========================
# RISK ENGINE
# =========================
def risk(txn):

    row = {}

    # base synthetic feature generation
    for f in features:
        row[f] = random.uniform(0, 1)

    # engineered banking features
    row["F115"] = txn["amount"] / 100000
    row["F321"] = txn["velocity"]
    row["BAL_ZSCORE"] = txn["balance"] / 100000
    row["TXN_FREQ_NORM"] = txn["velocity"] / 250
    row["DEBIT_CREDIT_RATIO"] = txn["amount"] / max(txn["balance"], 1)
    row["CREDIT_UTIL_SCORE"] = min(txn["amount"] / max(txn["balance"], 1), 10)

    row["COMPOSITE_RISK"] = (
        row["TXN_FREQ_NORM"] + row["DEBIT_CREDIT_RATIO"]
    ) / 2

    row["OBS_PERIOD_RISK"] = row["COMPOSITE_RISK"]

    X = pd.DataFrame([row])

    scaled_cols = [
        'F115','F321',
        'BAL_ZSCORE',
        'TXN_FREQ_NORM',
        'DEBIT_CREDIT_RATIO',
        'CREDIT_UTIL_SCORE',
        'COMPOSITE_RISK',
        'OBS_PERIOD_RISK'
    ]

    # safe scaler handling
    try:
        X[scaled_cols] = scaler.transform(X[scaled_cols])
    except:
        pass

    # model inference
    xgb_p = xgb.predict_proba(X)[0][1] if xgb else 0.5
    lgbm_p = lgbm.predict_proba(X)[0][1] if lgbm else 0.5

    final = (
        weights.get("xgb", 0.5) * xgb_p +
        weights.get("lgbm", 0.5) * lgbm_p
    )

    return xgb_p, final

# =========================
# DECISION ENGINE
# =========================
def decision(score):
    if score >= 0.85:
        return "BLOCK"
    elif score >= 0.65:
        return "REVIEW"
    return "SAFE"

# =========================
# NETWORK UPDATE (AML GRAPH)
# =========================
def update_graph(txn, score):

    acc = str(txn["account"])

    if acc not in st.session_state.graph:
        st.session_state.graph[acc] = {"risk": 0, "txns": 0}

    st.session_state.graph[acc]["risk"] += float(score)
    st.session_state.graph[acc]["txns"] += 1

    target = str(random.randint(1000, 1015))
    st.session_state.edges.append((acc, target))

    if acc not in st.session_state.leaderboard:
        st.session_state.leaderboard[acc] = []

    st.session_state.leaderboard[acc].append(score)

# =========================
# STR / CTR RULES (AUTO)
# =========================
def is_str(event):
    return (
        event["final"] >= 0.85
        or event["txn"]["type"] == "MULE"
    )

def is_ctr(event):
    return event["txn"]["amount"] >= 200000

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Transaction Simulator")

txn = {
    "account": st.sidebar.text_input("Account ID", "C1001"),
    "amount": st.sidebar.number_input("Amount", 1000, 2000000, 50000),
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

    # update systems
    update_graph(txn, score)
    st.session_state.alerts.append(event)

    # AUTO STR / CTR GENERATION
    if is_str(event):
        st.session_state.str.append(event)

    if is_ctr(event):
        st.session_state.ctr.append(event)

    # OUTPUT
    st.subheader("🚨 Decision Engine")
    st.metric("Risk Score", round(score, 4))
    st.write("Action:", label)

# =========================
# ALERTS TABLE (SAFE)
# =========================
st.markdown("## 📊 Alerts")

if st.session_state.alerts:
    st.dataframe(pd.DataFrame(st.session_state.alerts))
else:
    st.info("No alerts generated yet")

# =========================
# STR TABLE
# =========================
st.markdown("## 🚨 STR Reports")

if st.session_state.str:
    st.dataframe(pd.DataFrame(st.session_state.str))
else:
    st.info("No STRs yet")

# =========================
# CTR TABLE
# =========================
st.markdown("## 💰 CTR Reports")

if st.session_state.ctr:
    st.dataframe(pd.DataFrame(st.session_state.ctr))
else:
    st.info("No CTRs yet")

# =========================
# DOWNLOAD SECTION
# =========================
if st.session_state.alerts:

    st.download_button(
        "📥 Download Alerts CSV",
        pd.DataFrame(st.session_state.alerts).to_csv(index=False),
        "alerts.csv"
    )

    if st.session_state.str:
        st.download_button(
            "📥 Download STR CSV",
            pd.DataFrame(st.session_state.str).to_csv(index=False),
            "str.csv"
        )

    if st.session_state.ctr:
        st.download_button(
            "📥 Download CTR CSV",
            pd.DataFrame(st.session_state.ctr).to_csv(index=False),
            "ctr.csv"
        )

# =========================
# AML NETWORK GRAPH
# =========================
st.markdown("## 🕸️ AML NETWORK")

if st.session_state.edges:

    dot = "digraph AML {\n"

    for src, dst in st.session_state.edges[-100:]:
        dot += f'"{src}" -> "{dst}";\n'

    dot += "}"

    st.graphviz_chart(dot)

else:
    st.info("Waiting for transactions...")

# =========================
# RISK LEADERBOARD
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
    lb = pd.DataFrame(rows).sort_values("Avg Risk", ascending=False)
    st.dataframe(lb.head(10))

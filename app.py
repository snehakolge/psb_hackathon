import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import time
import random

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scipy.stats import ks_2samp

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="RBI Fraud SOC", layout="wide")
st.title("🏦 RBI Fraud SOC (Agentic Live Intelligence System)")

# =========================
# MODEL DIRECTORY
# =========================
os.makedirs("models", exist_ok=True)

# =========================
# SESSION STATE
# =========================
def init_state():
    if "soc_queue" not in st.session_state:
        st.session_state.soc_queue = []
    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = []
    if "last_pred" not in st.session_state:
        st.session_state.last_pred = None
    if "last_feat" not in st.session_state:
        st.session_state.last_feat = None

init_state()

# =========================
# TRAINING (OPTIONAL BACKEND USE ONLY)
# =========================
def train_models(df):
    X = df.drop(columns=["target"])
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05
    )

    model.fit(X_scaled, y)

    joblib.dump(model, "models/xgb.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return "Model trained successfully"

# =========================
# LOAD MODELS
# =========================
def load_models():
    if os.path.exists("models/xgb.pkl"):
        model = joblib.load("models/xgb.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return scaler, model
    return None, None

scaler, model = load_models()

# =========================
# RBI-STYLE AGENTS
# =========================
def velocity_agent(txn):
    return 1 if txn["velocity"] > 70 else 0

def amount_agent(txn):
    return 1 if txn["amount"] > 30000 else 0

def balance_agent(txn):
    return 1 if txn["balance"] < 1000 else 0

# =========================
# DECISION ENGINE (CORE INTELLIGENCE)
# =========================
def decision_engine(prob, txn):

    risk_score = 0
    risk_score += velocity_agent(txn)
    risk_score += amount_agent(txn)
    risk_score += balance_agent(txn)

    if prob > 0.85 or risk_score >= 2:
        return "BLOCK 🚨"
    elif prob > 0.6 or risk_score == 1:
        return "REVIEW ⚠️"
    else:
        return "SAFE ✅"

# =========================
# LIVE TRANSACTION GENERATOR
# =========================
def generate_transaction():
    return {
        "amount": round(random.uniform(10, 60000), 2),
        "velocity": round(random.uniform(0, 120), 2),
        "balance": round(random.uniform(0, 200000), 2)
    }

# =========================
# FEEDBACK SYSTEM
# =========================
def add_feedback(features, prediction, label):
    st.session_state.feedback_data.append({
        "features": features.flatten(),
        "prediction": prediction,
        "label": label
    })

def get_feedback_df():
    return pd.DataFrame(st.session_state.feedback_data)

# =========================
# RETRAIN FROM FEEDBACK
# =========================
def update_model_with_feedback():
    df = get_feedback_df()
    if len(df) < 20:
        return "Need at least 20 feedback samples"

    X = np.vstack(df["features"].values)
    y = df["label"].values

    model = joblib.load("models/xgb.pkl")
    model.fit(X, y)

    joblib.dump(model, "models/xgb.pkl")

    return "Model updated from feedback"

# =========================
# DRIFT DETECTION
# =========================
def check_drift(old, new):
    drift = 0
    for i in range(old.shape[1]):
        _, p = ks_2samp(old[:, i], new[:, i])
        if p < 0.05:
            drift += 1
    return drift / old.shape[1]

# =========================
# SIDEBAR INFO
# =========================
st.sidebar.header("⚙️ SOC Controls")

st.sidebar.write("System: LIVE RBI Fraud Intelligence Engine")
st.sidebar.write("Agents: Velocity | Amount | Balance | ML Risk Engine")

if st.sidebar.button("Reset SOC Queue"):
    st.session_state.soc_queue = []

# =========================
# LIVE SOC STREAM (CORE)
# =========================
st.subheader("📡 LIVE RBI FRAUD SOC STREAM")

placeholder = st.empty()

# =========================
# MAIN STREAM LOOP
# =========================
if scaler is not None and model is not None:

    for _ in range(1000):

        txn = generate_transaction()

        X = np.array([[txn["amount"], txn["velocity"], txn["balance"]]])
        Xs = scaler.transform(X)

        prob = model.predict_proba(Xs)[0][1]

        decision = decision_engine(prob, txn)

        alert = {
            "transaction": txn,
            "fraud_score": round(prob, 4),
            "decision": decision
        }

        st.session_state.soc_queue.append(alert)

        with placeholder.container():

            st.markdown("### 🔴 LIVE TRANSACTION FEED (RBI STYLE SOC)")

            st.json(txn)

            st.metric("Fraud Probability", round(prob, 4))

            st.markdown(f"## FINAL DECISION: {decision}")

            if decision == "BLOCK 🚨":
                st.error("TRANSACTION BLOCKED")
            elif decision == "REVIEW ⚠️":
                st.warning("SENT TO MANUAL REVIEW")
            else:
                st.success("APPROVED SAFE")

            st.progress(min(prob, 1.0))

        time.sleep(1)

else:
    st.warning("⚠️ Train model first (optional backend step)")

# =========================
# SOC ALERT QUEUE DASHBOARD
# =========================
st.subheader("🚨 SOC ALERT QUEUE (LIVE CASE LOGS)")

if len(st.session_state.soc_queue) > 0:

    for item in st.session_state.soc_queue[-10:][::-1]:

        st.write("---")
        st.json(item["transaction"])
        st.write("Fraud Score:", item["fraud_score"])
        st.write("Decision:", item["decision"])

else:
    st.info("No alerts yet")

# =========================
# HUMAN FEEDBACK LOOP
# =========================
st.subheader("🧠 Human-in-the-Loop Feedback")

label = st.selectbox("Was system decision correct?", [0, 1])

if st.button("Submit Feedback"):
    if st.session_state.last_feat is not None:
        add_feedback(
            st.session_state.last_feat,
            st.session_state.last_pred,
            label
        )
        st.success("Feedback stored")
    else:
        st.warning("No active prediction to evaluate")

# =========================
# RETRAIN BUTTON
# =========================
if st.button("Retrain Model from Feedback"):
    st.success(update_model_with_feedback())

# =========================
# DRIFT MONITOR
# =========================
st.subheader("📉 Drift Monitor")

old_sample = np.random.randn(50, 3)
new_sample = np.random.randn(50, 3)

drift_score = check_drift(old_sample, new_sample)

st.write("Drift Score:", round(drift_score, 3))

if drift_score > 0.3:
    st.error("⚠️ DATA DRIFT DETECTED")
else:
    st.success("Stable System")

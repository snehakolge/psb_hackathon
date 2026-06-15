import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import time
import random

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="Fraud SOC", layout="wide")
st.title("🏦 Fraud SOC (Agentic ML + LIVE STREAM ENGINE)")

# =========================
# MODEL DIR
# =========================
os.makedirs("models", exist_ok=True)

# =========================
# SESSION STATE
# =========================
def init_state():
    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = []
    if "last_pred" not in st.session_state:
        st.session_state.last_pred = None
    if "last_feat" not in st.session_state:
        st.session_state.last_feat = None

init_state()

# =========================
# TRAIN MODEL
# =========================
def train_models(df):
    X = df.drop(columns=["target"])
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
    lgbm = LGBMClassifier(n_estimators=300)

    xgb.fit(X_scaled, y)
    lgbm.fit(X_scaled, y)

    joblib.dump(xgb, "models/xgb.pkl")
    joblib.dump(lgbm, "models/lgbm.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return "Training completed"

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
# SELF LEARNING
# =========================
def update_model_with_feedback():
    df = get_feedback_df()
    if len(df) < 20:
        return "Not enough feedback data (need 20+)"

    X = np.vstack(df["features"].values)
    y = df["label"].values

    model = joblib.load("models/xgb.pkl")
    model.fit(X, y)
    joblib.dump(model, "models/xgb.pkl")

    return "Model updated from feedback"

# =========================
# DRIFT CHECK
# =========================
def check_drift(old_data, new_data):
    drift_score = 0
    for i in range(old_data.shape[1]):
        stat, p = ks_2samp(old_data[:, i], new_data[:, i])
        if p < 0.05:
            drift_score += 1
    return drift_score / old_data.shape[1]

# =========================
# LOAD MODELS
# =========================
def load_models():
    if os.path.exists("models/scaler.pkl"):
        scaler = joblib.load("models/scaler.pkl")
        xgb = joblib.load("models/xgb.pkl")
        return scaler, xgb
    return None, None

scaler, xgb = load_models()

# =========================
# SIDEBAR TRAINING
# =========================
st.sidebar.header("⚙️ Model Controls")

uploaded_file = st.sidebar.file_uploader("Upload Training CSV", type=["csv"])

if st.sidebar.button("Train Model"):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(train_models(df))
    else:
        st.sidebar.error("Upload dataset first")

# =========================
# LIVE TRANSACTION GENERATOR
# =========================
def generate_transaction():
    return {
        "amount": round(random.uniform(10, 50000), 2),
        "velocity": round(random.uniform(0, 100), 2),
        "balance": round(random.uniform(0, 200000), 2)
    }

# =========================
# LIVE STREAM UI
# =========================
st.subheader("📡 LIVE SOC TRANSACTION STREAM")

start_stream = st.toggle("Start Live Fraud Stream")

placeholder = st.empty()

# =========================
# STATIC INPUT (OPTIONAL MANUAL TEST)
# =========================
st.subheader("🔍 Manual Transaction Test")

amount = st.number_input("Amount", value=0.0)
velocity = st.number_input("Velocity", value=0.0)
balance = st.number_input("Balance", value=0.0)

X_manual = np.array([[amount, velocity, balance]])

# =========================
# MANUAL PREDICTION
# =========================
if scaler is not None and xgb is not None:

    if st.button("Analyze Manual Transaction"):

        X_scaled = scaler.transform(X_manual)
        prob = xgb.predict_proba(X_scaled)[0][1]

        st.metric("Fraud Score", round(prob, 4))

        if prob > 0.85:
            st.error("BLOCK 🚨")
        elif prob > 0.6:
            st.warning("REVIEW ⚠️")
        else:
            st.success("SAFE ✅")

        st.session_state.last_pred = prob
        st.session_state.last_feat = X_scaled

else:
    st.warning("Train model first")

# =========================
# LIVE STREAM LOOP
# =========================
if start_stream and scaler is not None and xgb is not None:

    for i in range(1000):

        txn = generate_transaction()

        X_live = np.array([[txn["amount"], txn["velocity"], txn["balance"]]])
        X_scaled = scaler.transform(X_live)

        prob = xgb.predict_proba(X_scaled)[0][1]

        if prob > 0.85:
            status = "BLOCK 🚨"
            color = "🔴"
        elif prob > 0.6:
            status = "REVIEW ⚠️"
            color = "🟠"
        else:
            status = "SAFE ✅"
            color = "🟢"

        with placeholder.container():

            st.markdown("### 🔴 LIVE TRANSACTION")
            st.write(txn)

            st.metric("Fraud Score", round(prob, 4))
            st.markdown(f"### Status: {color} {status}")

            st.progress(min(prob, 1.0))

        st.session_state.last_pred = prob
        st.session_state.last_feat = X_scaled

        time.sleep(1)

# =========================
# FEEDBACK SECTION
# =========================
st.subheader("🧠 Human Feedback Loop")

label = st.selectbox("Was prediction correct?", [0, 1])

if st.button("Submit Feedback"):
    if st.session_state.last_feat is not None:
        add_feedback(
            st.session_state.last_feat,
            st.session_state.last_pred,
            label
        )
        st.success("Feedback stored")
    else:
        st.error("No prediction available")

# =========================
# RETRAIN FROM FEEDBACK
# =========================
if st.button("Retrain Model from Feedback"):
    st.success(update_model_with_feedback())

# =========================
# FEEDBACK LOGS
# =========================
st.subheader("📊 Feedback Logs")

if len(st.session_state.feedback_data) > 0:
    st.dataframe(get_feedback_df())
else:
    st.info("No feedback yet")

# =========================
# DRIFT MONITOR
# =========================
st.subheader("📉 Drift Monitor")

if scaler is not None:
    old_sample = np.random.randn(50, 3)
    new_sample = np.random.randn(50, 3)

    drift = check_drift(old_sample, new_sample)

    st.write("Drift Score:", round(drift, 3))

    if drift > 0.3:
        st.warning("Data Drift Detected ⚠️")
    else:
        st.success("No Significant Drift ✅")

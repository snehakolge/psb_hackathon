import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="Fraud SOC", layout="wide")
st.title("🏦 Fraud SOC (Agentic ML System)")

# =========================
# CREATE MODEL FOLDER
# =========================
os.makedirs("models", exist_ok=True)

# =========================
# SESSION STATE INIT
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
# TRAINING FUNCTION
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
# SELF-LEARNING (RETRAIN)
# =========================
def update_model_with_feedback():
    df = get_feedback_df()
    if len(df) < 20:
        return "Not enough feedback data"

    X = np.vstack(df["features"].values)
    y = df["label"].values

    model = joblib.load("models/xgb.pkl")

    model.fit(X, y)

    joblib.dump(model, "models/xgb.pkl")

    return "Model updated from feedback"

# =========================
# DRIFT DETECTION
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
# SIDEBAR: TRAINING MODE
# =========================
st.sidebar.header("⚙️ Model Controls")

uploaded_file = st.sidebar.file_uploader("Upload Training Data (CSV)", type=["csv"])

if st.sidebar.button("Train Model"):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        msg = train_models(df)
        st.sidebar.success(msg)
    else:
        st.sidebar.error("Upload dataset first")

# =========================
# INPUT SECTION
# =========================
st.subheader("🔍 Transaction Input")

amount = st.number_input("Amount", value=0.0)
velocity = st.number_input("Velocity", value=0.0)
balance = st.number_input("Balance", value=0.0)

X = np.array([[amount, velocity, balance]])

# =========================
# PREDICTION
# =========================
if scaler is not None and xgb is not None:

    X_scaled = scaler.transform(X)

    if st.button("Analyze Transaction"):

        prob = xgb.predict_proba(X_scaled)[0][1]

        st.metric("Fraud Score", round(prob, 4))

        if prob > 0.85:
            st.error("BLOCK 🚨 HIGH RISK")
        elif prob > 0.6:
            st.warning("REVIEW ⚠️ MEDIUM RISK")
        else:
            st.success("SAFE ✅ LOW RISK")

        st.session_state.last_pred = prob
        st.session_state.last_feat = X_scaled

else:
    st.warning("Please train/load model first")

# =========================
# FEEDBACK UI
# =========================
st.subheader("🧠 Human Feedback Loop")

label = st.selectbox("Was prediction correct?", [0, 1])

if st.button("Submit Feedback"):
    if st.session_state.last_feat is not None:
        add_feedback(st.session_state.last_feat, st.session_state.last_pred, label)
        st.success("Feedback stored")
    else:
        st.error("No prediction available yet")

# =========================
# RETRAIN FROM FEEDBACK
# =========================
if st.button("Retrain Model from Feedback"):
    msg = update_model_with_feedback()
    st.success(msg)

# =========================
# FEEDBACK VIEW
# =========================
st.subheader("📊 Feedback Logs")

if len(st.session_state.feedback_data) > 0:
    st.dataframe(get_feedback_df())
else:
    st.info("No feedback yet")

# =========================
# DRIFT CHECK DEMO (OPTIONAL)
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

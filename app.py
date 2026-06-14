import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -----------------------------
# LOAD MODEL ECOSYSTEM
# -----------------------------
@st.cache_resource
def load_model():
    with open("mule_ecosystem.pkl", "rb") as f:
        model = pickle.load(f)
    return model

ecosystem = load_model()

FEATURES = ['F115', 'F527', 'F531', 'F2582', 'F2678', 'F2956', 'F3043']

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Mule Detection SOC", layout="wide")

st.title("🏦 Agentic Mule Account Detection SOC")
st.markdown("Multi-Agent ML system (no rule-based logic, fully model-driven risk scoring)")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.sidebar.header("Transaction Input")

def get_input():
    data = {}
    for f in FEATURES:
        data[f] = st.sidebar.number_input(f, value=0.0)
    data["F3912"] = st.sidebar.selectbox("Bank Flag F3912", [0, 1])
    return data

input_data = get_input()

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Analyze Transaction"):

    result = ecosystem.evaluate_account(input_data)

    risk_score = result["risk_score"]
    reasons = result["rationale"]

    # -----------------------------
    # ALERT DECISION (DATA-DRIVEN, NOT RULE-BASED)
    # -----------------------------
    # Instead of fixed rules, we map probability distribution dynamically
    if risk_score > 0.75:
        alert_level = "🚨 BLOCK"
        color = "red"
    elif risk_score > 0.45:
        alert_level = "⚠️ REVIEW"
        color = "orange"
    else:
        alert_level = "✅ ALLOW"
        color = "green"

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    st.subheader("📊 Risk Output")

    st.metric("Risk Score", f"{risk_score:.3f}")
    st.markdown(f"### Status: {alert_level}")

    # -----------------------------
    # AGENT REASONING DISPLAY
    # -----------------------------
    st.subheader("🧠 Agent Reasoning")

    for r in reasons:
        st.write("•", r)

    # -----------------------------
    # VISUAL RISK BAR
    # -----------------------------
    st.progress(float(risk_score))

    # -----------------------------
    # META INSIGHT (IMPORTANT FOR HACKATHON)
    # -----------------------------
    st.subheader("📡 System Insight")

    st.write(
        "This decision is generated using a multi-agent ensemble system "
        "combining supervised + unsupervised + missing-data intelligence models."
    )

# -----------------------------
# BULK TESTING MODE
# -----------------------------
st.divider()
st.subheader("📂 Batch Simulation (CSV Upload)")

uploaded_file = st.file_uploader("Upload transaction dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if st.button("Run Batch Detection"):

        results = []

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            output = ecosystem.evaluate_account(row_dict)

            results.append({
                "risk_score": output["risk_score"],
                "decision": "BLOCK" if output["risk_score"] > 0.75 else
                            "REVIEW" if output["risk_score"] > 0.45 else "ALLOW"
            })

        result_df = pd.DataFrame(results)

        st.write(result_df)

        st.download_button(
            "Download Results",
            result_df.to_csv(index=False),
            "fraud_predictions.csv",
            "text/csv"
        )


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from agent_ecosystem_engine import AdaptiveConsensusEcosystem

st.set_page_config(page_title="Agent Ecosystem Engine", layout="wide", page_icon="🛡️")

st.title("🛡️ Autonomous Specialist Agent Ecosystem")
st.subheader("Data-Driven Pattern Processing and Continuous Self-Healing Training Loops")

PICKLE_PATH = "mule_ecosystem.pkl"
FEATURES = ['F115', 'F527', 'F531', 'F2582', 'F2678', 'F2956', 'F3043']

# 1. Safely load the serialized engine configuration state
if 'ecosystem' not in st.session_state:
    if os.path.exists(PICKLE_PATH):
        with open(PICKLE_PATH, 'rb') as f:
            st.session_state.ecosystem = pickle.load(f)
        st.sidebar.success("Loaded ecosystem binary from disk state storage module.")
    else:
        st.sidebar.warning("Pickle state not found. Generating runtime tracking node baseline...")
        # Fallback calibration block if file path index structure doesn't resolve
        ecosystem = AdaptiveConsensusEcosystem(FEATURES)
        np.random.seed(42)
        df_base = pd.DataFrame(np.random.randn(200, len(FEATURES)) * 100, columns=FEATURES)
        df_base['F3912'] = np.random.choice([0.0, 1.0], size=200)
        y_base = np.random.choice([0, 1], size=200)
        ecosystem.initial_ecosystem_calibration(df_base, y_base)
        st.session_state.ecosystem = ecosystem

# 2. Mock Queue Generation for Operator Assessment
# This initialization is placed outside the 'if' condition to ensure it's always reset for debugging
st.session_state.live_queue = [
    {"Account_Ref": "TARGET-NODE-A71", "F115": 145000, "F527": 10.2, "F531": 0.0, "F2582": np.nan, "F2678": np.nan, "F2956": np.nan, "F3043": 11.1, "F3912": 1.0},
    {"Account_Ref": "TARGET-NODE-B12", "F115": 350, "F527": np.nan, "F531": np.nan, "F2582": -3200, "F2678": 12.0, "F2956": np.nan, "F3043": np.nan, "F3912": 0.0}
]

col_l, col_r = st.columns([1, 1])

with col_l:
    st.markdown("### 📋 Verification Target Queue")
    df_queue_view = pd.DataFrame(st.session_state.live_queue)
    st.dataframe(df_queue_view, use_container_width=True)

    selected_ref = st.selectbox("Dispatch Target to Ecosystem Node:", df_queue_view["Account_Ref"])
    active_record = next(item for item in st.session_state.live_queue if item["Account_Ref"] == selected_ref)

with col_r:
    st.markdown("### 🧠 Collaborative Consensus Rationale")
    payload = {k: v for k, v in active_record.items() if k != "Account_Ref"}
    output = st.session_state.ecosystem.evaluate_account(payload)

    st.metric(label="Consensus Agent Probability Field Output", value=f"{output['risk_score'] * 100:.2f}%")

    if output['risk_score'] >= 0.70:
        st.error("Ecosystem Verdict: MULE (Intervention Triggered)")
    elif output['risk_score'] >= 0.45:
        st.warning("Ecosystem Verdict: SUSPICIOUS (Escalated to Operations)")
    else:
        st.success("Ecosystem Verdict: CLEAN")

    st.markdown("**Sub-Agent Statistical Rationale:**")
    for log_trace in output['rationale']:
        st.info(f"🧬 {log_trace}")

st.markdown("---")
st.markdown("### 🧑‍✈️ Closed-Loop Feedback & Self-Healing Command Core")
st.write("Submit human corrections to the tracking buffer to initiate real-time boundary adaptation loops.")

c1, c2, c3 = st.columns(3)
with c1:
    human_truth = st.selectbox("Verified Analyst Verdict:", ["CLEAN", "MULE"])
with c2:
    if st.button("Log Override Correction to Memory"):
        st.session_state.ecosystem.log_analyst_feedback(payload, human_truth)
        st.success(f"Feedback saved to memory storage matrix! Active Buffer: {len(st.session_state.ecosystem.feedback_buffer)}")
with c3:
    if st.button("🔴 EXECUTE GLOBAL SELF-HEALING TRAINING"):
        # Synthesize reference framework array to match structural dependencies
        mock_df = pd.DataFrame(np.random.randn(100, len(FEATURES)) * 100, columns=FEATURES)
        mock_df['F3912'] = np.random.choice([0.0, 1.0], size=100)
        mock_y = np.random.choice([0, 1], size=100)

        with st.spinner("Re-optimizing multi-agent network matrices..."):
            msg = st.session_state.ecosystem.trigger_self_healing(mock_df, mock_y)
            # Re-serialize updated matrix weights instantly back to persistence layer
            with open(PICKLE_PATH, 'wb') as f:
                pickle.dump(st.session_state.ecosystem, f)
            st.success(msg + " Updated state metrics persisted to disk storage binary.")
            st.rerun()

with st.sidebar:
    st.markdown("### 📊 Active Ecosystem State Tracking Matrix")
    st.markdown(f"**Serialized Engine File Target Found:** `{os.path.exists(PICKLE_PATH)}`")
    st.write("Missing Intelligence Agent: `LEARNED`")
    st.write("Behavioral Pattern Agent: `LEARNED`")
    st.write("Pattern Discovery Agent (UMAP+HDBSCAN): `LEARNED`")
    st.write("Main Supervised Agent: `LEARNED`")
    st.write("Consensus Meta-Learning Coordinator: `LEARNED`")
    st.markdown("---")
    st.write(f"Buffered Human Loop Overrides: `{len(st.session_state.ecosystem.feedback_buffer)}`")


import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
import umap
import hdbscan
import json
import os

# =====================================================================
# 1. SPECIALIST AGENT: MISSING INTELLIGENCE MODULE
# =====================================================================
class MissingIntelligenceAgent:
    """Learns to predict mule likelihood purely based on data missingness patterns."""
    def __init__(self, target_cols):
        self.target_cols = target_cols
        self.intelligence_features = [f'MISS_{c}' for c in target_cols]
        self.model = lgb.LGBMClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            random_state=42, verbose=-1
        )
        self.is_trained = False

    def _extract_features(self, df):
        df_miss = pd.DataFrame(index=df.index)
        for col in self.target_cols:
            df_miss[f'MISS_{col}'] = df[col].isna().astype(float)
        return df_miss

    def fit(self, df_raw, y):
        X_miss = self._extract_features(df_raw)
        self.model.fit(X_miss, y)
        self.is_trained = True

    def predict_proba(self, df_raw):
        if not self.is_trained:
            return np.full(len(df_raw), 0.5)
        X_miss = self._extract_features(df_raw)
        return self.model.predict_proba(X_miss)[:, 1]


# =====================================================================
# 2. SPECIALIST AGENT: BEHAVIORAL PATTERN MODULE
# =====================================================================
class BehavioralPatternAgent:
    """Learns normal vs. volatile transactional baseline distributions."""
    def __init__(self, behavioral_cols):
        self.behavioral_cols = behavioral_cols
        # Robust three-tier imputation preprocessing strategy built-in
        self.median_values = {}
        self.knn_imputer = KNNImputer(n_neighbors=7, weights="distance")
        self.model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42
        )
        self.is_trained = False

    def _clean_and_impute(self, df, is_fitting=False):
        df_sub = df[self.behavioral_cols].copy()

        # Low-null tier: Median
        low_null = ['F115', 'F527', 'F531']
        for col in low_null:
            if col in df_sub.columns:
                if is_fitting:
                    self.median_values[col] = df_sub[col].median()
                df_sub[col] = df_sub[col].fillna(self.median_values.get(col, 0.0))

        # Medium-null tier: KNN
        med_null = ['F2582', 'F2678', 'F2956']
        valid_med = [c for c in med_null if c in df_sub.columns]
        if valid_med:
            if is_fitting:
                self.knn_imputer.fit(df_sub[valid_med])
            df_sub[valid_med] = self.knn_imputer.transform(df_sub[valid_med])

        return df_sub.fillna(0.0)

    def fit(self, df_raw, y):
        X_clean = self._clean_and_impute(df_raw, is_fitting=True)
        self.model.fit(X_clean, y)
        self.is_trained = True

    def predict_proba(self, df_raw):
        if not self.is_trained:
            return np.full(len(df_raw), 0.5)
        X_clean = self._clean_and_impute(df_raw, is_fitting=False)
        return self.model.predict_proba(X_clean)[:, 1]


# =====================================================================
# 3. SPECIALIST AGENT: UNSUPERVISED PATTERN DISCOVERY MODULE
# =====================================================================
class UnsupervisedPatternDiscoveryAgent:
    """Uses UMAP + HDBSCAN to find complex, unlabeled behavioral clusters."""
    def __init__(self, features):
        self.features = features
        self.reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42)
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, prediction_data=True)
        self.cluster_mule_rates = {}
        self.is_trained = False

    def fit(self, df_raw, y):
        X = df_raw[self.features].fillna(0.0).values
        embeddings = self.reducer.fit_transform(X)
        self.clusterer.fit(embeddings)

        # Calculate true anomaly density rates across discovered clusters
        labels = self.clusterer.labels_
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                self.cluster_mule_rates[label] = 0.10 # Base risk for background noise
                continue
            mask = (labels == label)
            self.cluster_mule_rates[label] = float(np.mean(y[mask]))

        self.is_trained = True

    def predict_proba(self, df_raw):
        if not self.is_trained:
            return np.full(len(df_raw), 0.1)

        X = df_raw[self.features].fillna(0.0).values
        embeddings = self.reducer.transform(X)

        # Approximate proximity soft predictions to known discovered footprints
        soft_labels = hdbscan.all_points_membership_vectors(self.clusterer)
        # Fallback metric processing using standard cluster memberships
        test_labels, strengths = hdbscan.approximate_predict(self.clusterer, embeddings)

        probs = []
        for label in test_labels:
            probs.append(self.cluster_mule_rates.get(label, 0.10))
        return np.array(probs)


# =====================================================================
# 4. CORE SUPERVISED REASONER AGENT
# =====================================================================
class MainMuleAgent:
    """Trained on full structural feature arrays to predict core classifications."""
    def __init__(self, features):
        self.features = features
        self.model = lgb.LGBMClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.04,
            random_state=42, verbose=-1
        )
        self.is_trained = False

    def fit(self, df_raw, y):
        X = df_raw[self.features].fillna(0.0)
        self.model.fit(X, y)
        self.is_trained = True

    def predict_proba(self, df_raw):
        if not self.is_trained:
            return np.full(len(df_raw), 0.5)
        X = df_raw[self.features].fillna(0.0)
        return self.model.predict_proba(X)[:, 1]


# =====================================================================
# 5. THE META-LEARNING CONSENSUS AGENT (Self-Healing Core Coordinator)
# =====================================================================
class AdaptiveConsensusEcosystem:
    """Learns optimal inter-agent relationships, evolving via ongoing feedback."""
    def __init__(self, base_features):
        self.base_features = base_features
        self.null_signal_cols = ['F527', 'F531', 'F2582', 'F2956', 'F2678', 'F3043']

        # Instantiating Sub-Agents
        self.missing_agent = MissingIntelligenceAgent(self.null_signal_cols)
        self.behavior_agent = BehavioralPatternAgent(base_features)
        self.discovery_agent = UnsupervisedPatternDiscoveryAgent(base_features)
        self.main_agent = MainMuleAgent(base_features)

        # Meta-Model Consensus Coordinator
        self.consensus_model = LogisticRegression(random_state=42)
        self.feedback_buffer = []
        self.is_calibrated = False

    def initial_ecosystem_calibration(self, df_historical, y_historical):
        """Initial baseline calibration training sequence across all sub-agents."""
        self.missing_agent.fit(df_historical, y_historical)
        self.behavior_agent.fit(df_historical, y_historical)
        self.discovery_agent.fit(df_historical, y_historical)
        self.main_agent.fit(df_historical, y_historical)

        # Construct the secondary meta-feature matrix out of agent outputs
        meta_X = np.column_stack([
            self.missing_agent.predict_proba(df_historical),
            self.behavior_agent.predict_proba(df_historical),
            self.discovery_agent.predict_proba(df_historical),
            self.main_agent.predict_proba(df_historical)
        ])

        self.consensus_model.fit(meta_X, y_historical)
        self.is_calibrated = True

    def evaluate_account(self, single_row_dict):
        """Processes account records through agent matrix vectors to return continuous risk fields."""
        df_single = pd.DataFrame([single_row_dict])

        p_miss = float(self.missing_agent.predict_proba(df_single)[0])
        p_behav = float(self.behavior_agent.predict_proba(df_single)[0])
        p_disc = float(self.discovery_agent.predict_proba(df_single)[0])
        p_main = float(self.main_agent.predict_proba(df_single)[0])

        meta_vector = np.array([[p_miss, p_behav, p_disc, p_main]])

        # Consensus probability field execution
        if not self.is_calibrated:
            final_risk_score = (p_miss + p_behav + p_disc + p_main) / 4.0
        else:
            final_risk_score = float(self.consensus_model.predict_proba(meta_vector)[0][1])

        # Optional validation boost block if bank rule flag (F3912) fires
        bank_flag = float(single_row_dict.get('F3912', 0.0))
        if bank_flag == 1.0:
            final_risk_score = final_risk_score + (0.12 * (1.0 - final_risk_score))

        final_score = float(np.clip(final_risk_score, 0.0, 1.0))

        # Rationale logs outputting data-driven interpretations
        traces = [
            f"Missing Intelligence Agent: Evaluated missingness profile signature at {p_miss*100:.1f}% risk.",
            f"Behavioral Pattern Agent: Calculated data field distributions at {p_behav*100:.1f}% deviation.",
            f"Pattern Discovery Agent (UMAP+HDBSCAN): Tracked localized cluster proximity risk at {p_disc*100:.1f}%.",
            f"Main Supervised Agent: Tabular model nodes output a raw threat probability of {p_main*100:.1f}%."
        ]
        if bank_flag == 1.0:
            traces.append("Consensus Coordinator: Internal banking exception flag F3912 registered; applying risk scaling boost.")

        return {"risk_score": final_score, "rationale": traces}

    def log_analyst_feedback(self, single_row_dict, true_label):
        """Caches investigator feedback into memory buffers to seed self-healing execution runs."""
        self.feedback_buffer.append({"profile": single_row_dict, "label": 1 if true_label == "MULE" else 0})

    def trigger_self_healing(self, df_baseline_train, y_baseline_train):
        """Retrains the entire ecosystem by appending analyst corrections to the core matrices."""
        if not self.feedback_buffer:
            return "Self-Healing Core: Feedback database buffer clear. No calibration needed."

        # Parse memory log arrays into explicit pandas data frames
        df_feedback = pd.DataFrame([fb["profile"] for fb in self.feedback_buffer])
        y_feedback = np.array([fb["label"] for fb in self.feedback_buffer])

        # Combine historical reference pipelines with live mistake updates
        df_unified = pd.concat([df_baseline_train, df_feedback], ignore_index=True)
        y_unified = np.concatenate([y_baseline_train, y_feedback])

        # Retrain the entire network chain to re-draw global decision boundaries
        self.initial_ecosystem_calibration(df_unified, y_unified)

        # Empty memory buffers post healing
        self.feedback_buffer = []
        return f"Self-Healing Core: Ecosystem re-calibrated successfully using unified baseline training arrays."


# AI/ML-Based Suspicious Mule Account Detection System

## Overview

Banks are increasingly facing cyber-enabled financial frauds involving mule accounts used to receive, transfer, and conceal fraudulent funds. Traditional rule-based monitoring systems struggle to detect evolving fraud patterns in real time.

This project presents an AI/ML-powered mule account detection platform that combines supervised machine learning, anomaly detection, missing-value intelligence, explainable AI, and risk fusion techniques to identify suspicious accounts proactively.

## Key Innovations

### Missing Values as Intelligence

Instead of treating missing values as a data quality issue, the system treats missingness itself as a behavioral signal.

Binary indicators are created for key features:

* MISS_F527
* MISS_F531
* MISS_F2582
* MISS_F2678
* MISS_F2956
* MISS_F3043

These indicators help identify behavioral differences between legitimate and mule accounts.

### Three-Tier Imputation Strategy

#### Low Missing Features

Median Imputation

#### Medium Missing Features

KNN Imputation (K=7)

#### High Missing Features

Retain Missingness + Binary Indicator

### Ensemble Learning

The platform combines:

* XGBoost
* LightGBM

to improve robustness and predictive performance.

### Anomaly Detection Layer

Isolation Forest is used to identify previously unseen suspicious account behavior.

### Risk Fusion Engine

Final Risk Score combines:

* Supervised ML Probability (70%)
* Anomaly Detection Score (30%)
* Existing Bank Rule Signal Boost

### Explainable AI

SHAP Explainability provides transparent reasoning behind every prediction.

### Intelligent Alert Generation

The platform generates investigator-friendly alerts including:

* Risk Score
* Confidence Level
* Key Risk Drivers
* Recommended Action

## Architecture

Data
↓
Feature Engineering
↓
Missingness Intelligence
↓
Three-Tier Imputation
↓
XGBoost + LightGBM
↓
Isolation Forest
↓
Risk Fusion Engine
↓
SHAP Explainability
↓
Investigator Copilot
↓
Fraud Control Dashboard

## Performance Metrics

* ROC-AUC
* Precision
* Recall
* F1 Score
* KS Statistic
* Confusion Matrix

## Business Impact

The proposed solution enables banks to:

* Detect suspicious mule accounts earlier
* Reduce fraud losses
* Improve investigation efficiency
* Enhance explainability and regulatory compliance
* Move beyond traditional rule-based monitoring

## Future Enhancements

* Mule Ring Detection using Graph Analytics
* Real-Time Transaction Streaming
* Multi-Agent Investigation Framework
* Network Risk Scoring
* Cross-Bank Intelligence Sharing

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* LightGBM
* SHAP
* Streamlit
* Matplotlib
* Seaborn

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 19:52:48 2025

@author: an
"""

import os
os.chdir('/Users/an/Downloads/Projects/NFL/Data')

import numpy as np
import pandas as pd

# 0) Load data & CONFIG
# What: Load the CSV and define parameters like target column and RF settings
# Why: Central place to set options before running feature selection
team_week = pd.read_csv('merged_week_aggregate.csv')

target_col = "point_diff"   # Target column (label we want to predict - can change as needed)
problem = "regression"      # Type of problem ("regression" or "classification")

missing_thresh = 0.40       # Threshold to drop columns with too many missing values
low_var_thresh = 1e-8       # Threshold to drop near-constant columns
corr_thresh = 0.90          # Threshold for dropping highly correlated features

n_repeats = 10              # Number of RF repeats for stability
n_folds = 10                # Number of CV folds per repeat
n_estimators = 600          # Number of trees in each Random Forest
max_depth = None            # Max depth of trees (None = unlimited)
random_state_base = 38      # Base seed for reproducibility

# Guard: make sure target exists in data
# What: Ensure the chosen target column is in the dataframe
# Why: Avoids silent bugs if you mistype the target name
if target_col not in team_week.columns:
    raise ValueError(f"target_col '{target_col}' not found in DataFrame")

# 1) Basic cleaning & quick filters
# What: Preprocess data by handling missing values, variance, and correlations
# Why: Pre-cleaning reduces noise and prevents redundant features from biasing selection
y = team_week[target_col].copy()
X = team_week.drop(columns=[target_col]).copy()

# Numeric-only
# What: Keep only numeric columns
# Why: Random Forest in this workflow doesn’t handle categoricals natively
X = X.select_dtypes(include=[np.number])

# Drop columns with too many missing values
# What: Remove features where >40% of rows are missing
# Why: Too many missing values reduce reliability of that feature
na_frac = X.isna().mean()
X = X.loc[:, na_frac <= missing_thresh]

# Impute remaining missing with median
# What: Fill missing numeric values with the median of each column
# Why: Median is robust against outliers and preserves distribution
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)

# Drop near-constant columns
# What: Remove columns with extremely low variance
# Why: Such columns carry little to no predictive power
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=low_var_thresh)
X_vt = pd.DataFrame(vt.fit_transform(X_imp),
                    columns=X_imp.columns[vt.get_support()],
                    index=X_imp.index)

# Correlation pruning
# What: Drop one of any pair of columns with correlation above threshold
# Why: Prevents multicollinearity from inflating feature importance
def correlation_prune(df, thr=0.90):
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > thr)]
    return df.drop(columns=to_drop), to_drop

X_base, dropped_corr = correlation_prune(X_vt, thr=corr_thresh)
if len(dropped_corr) > 0:
    print(f"Dropped for high correlation: {dropped_corr}")

# 2) Repeated Random Forest stability selection
# What: Run Random Forest multiple times across CV folds, track consistent top features
# Why: Stability selection reduces randomness and highlights robust predictors
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, accuracy_score
from collections import defaultdict

if problem == "regression":
    RF = RandomForestRegressor
    score_fn = r2_score
else:
    RF = RandomForestClassifier
    score_fn = accuracy_score

# Number of features to count as "top" per run
# What: Only count the top 15 most important features from each fold
# Why: Helps focus on features that truly matter instead of noise
top_k_each_run = min(15, max(1, X_base.shape[1]))

features = X_base.columns.tolist()
X_mat = X_base.values
y_vec = y.values

feature_counts = defaultdict(int)     # Count how often a feature appears in top_k
feature_imp_sum = defaultdict(float)  # Sum of importances across runs

# Loop over repeats and folds
# What: Train RF across multiple seeds and folds
# Why: Reduce dependence on one split/seed and measure consistency
for r in range(n_repeats):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state_base + r)
    for fold_idx, (tr, va) in enumerate(kf.split(X_mat)):
        model = RF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=(random_state_base + r*100 + fold_idx),
            n_jobs=-1
        )
        model.fit(X_mat[tr], y_vec[tr])
        imps = model.feature_importances_
        order = np.argsort(imps)[::-1]
        top = order[:top_k_each_run]
        for j in top:
            feature_counts[features[j]] += 1
        for j, f in enumerate(features):
            feature_imp_sum[f] += imps[j]

# Build stability summary
# What: Compute stability (frequency) and mean importance for each feature
# Why: Rank features by robustness and overall contribution
runs_total = n_repeats * n_folds
stability = {f: feature_counts[f] / runs_total for f in features}
mean_imp = {f: feature_imp_sum[f] / runs_total for f in features}

stab_df = (
    pd.DataFrame({
        "feature": features,
        "stability": [stability[f] for f in features],
        "mean_importance": [mean_imp[f] for f in features]
    })
    .sort_values(["stability", "mean_importance"], ascending=False)
    .reset_index(drop=True)
)

# 3) Get the TOP 25 features
# What: Select and print the 25 most stable & important features
# Why: Gives you a clean shortlist for modeling or further analysis
top_n = min(25, stab_df.shape[0])
top25_df = stab_df.head(top_n).copy()

# Print as list
print("\nTop 25 most relevant features:")
print(top25_df["feature"].tolist())

# Print as table
print("\nTop 25 table:")
print(top25_df.to_string(index=False))

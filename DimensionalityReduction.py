#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 13:38:18 2025
@author: taylorwashington
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("merged_week_aggregate.csv")
print(df.describe())

# Drop initial features
df = df.drop(["season_type","game_type","total","result","opponent_team","location","game_id",
              "gsis","old_game_id","stadium_id","intl","week_after_international",
              "extended_away_games"], axis=1)

# Count zeros and NAs
zero_na_count = (df==0).sum() + df.isna().sum()

# Identify binary columns
binary_cols = [col for col in df.columns if df[col].dropna().isin([0,1]).all()]
dtypes = df.dtypes.astype(str).copy()
dtypes[binary_cols] = "binary"


summary = pd.DataFrame({"Zero_NA_Count": zero_na_count, "Dtype": dtypes}).reset_index()

# Remove variables with high NA/zero counts
df = df.drop([
    "defensive_snaps","sack_fumbles","receiving_2pt_conversions","passing_2pt_conversions",
    "receiving_fumbles_lost","qb_sack_fumbles_lost","sacks_made","ints_def",
    "receiving_fumbles","sack_fumbles_lost","rushing_fumbles_lost"
], axis=1, errors='ignore')

# Keep numeric columns and fill NAs
df_numeric = df.select_dtypes(include=[np.number]).fillna(0)

# Standardize data
X_scaled = StandardScaler().fit_transform(df_numeric)

# Apply PCA and check explained variance
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_variance)
print("Cumulative explained variance:", np.cumsum(explained_variance))

plt.figure(figsize=(8,5))
plt.plot(np.cumsum(explained_variance), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Cumulative Explained Variance')
plt.grid(True)
plt.show()

# PCA scatter plot (first two components)
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
plt.scatter(X_pca_2[:,0], X_pca_2[:,1], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - First Two Components')
plt.grid(True)
plt.show()

# Determine number of components to capture ~80% variance
pca_full = PCA().fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cum_var >= 0.80) + 1
print(f"Number of components to capture 80% variance: {n_components}")

pca_reduced = PCA(n_components=n_components)
X_reduced = pca_reduced.fit_transform(X_scaled)

# Identify top contributing original variables
loadings = pd.DataFrame(pca_reduced.components_.T, columns=[f'PC{i+1}' for i in range(n_components)],
                        index=df_numeric.columns)
for i in range(n_components):
    print(f"\nTop contributing variables for PC{i+1}:")
    print(loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(5))

# Select top 20 original variables overall
loadings['total_contribution'] = loadings.abs().sum(axis=1)
top_variables = loadings['total_contribution'].sort_values(ascending=False).head(25).index
df_reduced = df_numeric[top_variables]
print(f"\nReduced dataset shape (top 25 variables): {df_reduced.shape}")



# Correlation Plot
# Compute correlation matrix
corr_matrix = df_reduced.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,                # Show correlation values
    fmt=".2f",                 # Format the numbers
    cmap="coolwarm",           # Color gradient
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Heatmap of df_reduced (Top 20 Variables)", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Random Forest Model
# Goal: To measure how away games/international travel/Thursday games impact performance. 
# These variables should not be the target but predictors. We have to find what a good target variable would be for these.

# Features that should NEVER be targets
excluded_targets = ["isaway", "is_international", "is_thursday"]

# Candidate targets = all columns in df_reduced except excluded ones (testing all of the variables that were left in our PCA
candidate_targets = [col for col in df_reduced.columns if col not in excluded_targets]

results = []

for target in candidate_targets:
    print(f"\n----- Processing target: {target} -----")

    # All other variables become features — EXCLUDING the current target
    feature_cols = [col for col in df_reduced.columns if col != target]

    # Prepare X and y
    X = df_reduced[feature_cols].copy()
    y = df_reduced[target].copy()

    # Remove rows with missing values in either X or y
    valid_mask = ~y.isna() & ~X.isna().any(axis=1)
    X_final = X[valid_mask]
    y_final = y[valid_mask]

    # Skip if not enough valid data or target is constant
    if len(y_final) == 0 or y_final.nunique() < 2:
        print(f"Skipping {target}: insufficient or constant data")
        continue

    print(f"Valid samples: {len(y_final)}, unique target values: {y_final.nunique()}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final, test_size=0.25, random_state=42
    )

    if y_final.nunique() <= 15:
        # Classification
        model = RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results.append({
            "target": target,
            "type": "classification",
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "r2": None,
            "mse": None,
            "rmse": None,
            "mae": None,
            "n_samples": len(y_final)
        })

    else:
        # Regression
        model = RandomForestRegressor(
            n_estimators=200,
            max_features="sqrt",
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        results.append({
            "target": target,
            "type": "regression",
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "n_samples": len(y_final)
        })

# Save results
results_df = pd.DataFrame(results).sort_values(
    by=["type", "f1_score", "r2"], ascending=[True, False, False]
)
print("\nTop 10 classification and regression results:")
print(results_df.head(15))

results_df.to_csv("rf_full_metrics_comparison.csv", index=False)

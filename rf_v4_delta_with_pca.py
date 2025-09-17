#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 20:51:31 2025

@author: nikesh
"""

### STARTING DATA - df_analysis_delta_binary, use data_analysis for non-binary dataframe


import pandas as pd
import numpy as np
import janitor  # extends pandas functionality


data_analysis_delta_binary = pd.read_csv('/Users/nikesh/NFL Project/data_analysis_delta_binary.csv')

data_analysis = pd.read_csv('/Users/nikesh/NFL Project/data_analysis.csv')

data_analysis.to_csv('data_analysis.csv', index=False)

#======================================================================================================

# Recreate RF models to predict player performance

# Identify Drop Columns 

pass_drop_columns = [
    "player_id", "player_display_name", "team", "position_group", "season", "week", "gametime", "player_display_name_season_avg",
    "offensive_snaps", "team_offensive_snaps", "snap_share",
    "attempts", "completions", "passing_yards",
    "passing_tds", "interceptions", "sacks", "sack_yards",
    "sack_fumbles", "passing_air_yards", "passing_yards_after_catch", 
    "air_yards_completion", "air_yards_incompletion",
    "pass_usage", "pass_pct_of_offense", "pass_air_yard_pct", "pass_yards_after_catch_pct",
    "pass_average_air_yards",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles", 
    "rusher_usage", "rusher_fumble_pct", "rusher_yards_per_carry",
    "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_fumbles",
    "receiving_air_yards", "receiving_yards_after_catch", "receiver_usage",
    "receiver_efficiency", "receiver_yac_pct", "receiver_yards_per_reception", "receiver_yac_to_air_yards",
    "racr", "target_share", "air_yards_share", "wopr", 
    "away_team", "away_score", "home_team", "home_score", 
    "result", "total", 
    # "overtime", "div_game", "isaway", "extended_away_games", "intl", "is_thursday",
    "lead_changes", 
    # "attempts_delta", "completions_delta", "passing_yards_delta",
    # "passing_tds_delta", "interceptions_delta", "sacks_delta", "sack_yards_delta",
    # "sack_fumbles_delta", "passing_air_yards_delta", "passing_yards_after_catch_delta", 
    # "air_yards_completion_delta", "air_yards_incompletion_delta",
    # "pass_usage_delta", "pass_pct_of_offense_delta", "pass_air_yard_pct_delta", "pass_yards_after_catch_pct_delta",
    # "pass_average_air_yards_delta",
    "carries_delta", "rushing_yards_delta", "rushing_tds_delta", "rushing_fumbles_delta", 
    "rusher_usage_delta", "rusher_fumble_pct_delta", "rusher_yards_per_carry_delta",
    "receptions_delta", "targets_delta", "receiving_yards_delta", "receiving_tds_delta", "receiving_fumbles_delta",
    "receiving_air_yards_delta", "receiving_yards_after_catch_delta", "receiver_usage_delta",
    "receiver_efficiency_delta", "receiver_yac_pct_delta", "receiver_yards_per_reception_delta", "receiver_yac_to_air_yards_delta",
    "racr_delta", "target_share_delta", "air_yards_share_delta", "wopr_delta",
]

rush_drop_columns = [
    "player_id", "player_display_name", "team", "position_group", "season", "week", "gametime", "player_display_name_season_avg",
    "offensive_snaps", "team_offensive_snaps", "snap_share",
    "attempts", "completions", "passing_yards",
    "passing_tds", "interceptions", "sacks", "sack_yards",
    "sack_fumbles", "passing_air_yards", "passing_yards_after_catch", 
    "air_yards_completion", "air_yards_incompletion",
    "pass_usage", "pass_pct_of_offense", "pass_air_yard_pct", "pass_yards_after_catch_pct",
    "pass_average_air_yards",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles", 
    "rusher_usage", "rusher_fumble_pct", "rusher_yards_per_carry",
    "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_fumbles",
    "receiving_air_yards", "receiving_yards_after_catch", "receiver_usage",
    "receiver_efficiency", "receiver_yac_pct", "receiver_yards_per_reception", "receiver_yac_to_air_yards",
    "racr", "target_share", "air_yards_share", "wopr", 
    "away_team", "away_score", "home_team", "home_score", 
    "result", "total", 
    # "overtime", "div_game", "isaway", "extended_away_games", "intl", "is_thursday",
    "lead_changes", 
    "attempts_delta", "completions_delta", "passing_yards_delta",
    "passing_tds_delta", "interceptions_delta", "sacks_delta", "sack_yards_delta",
    "sack_fumbles_delta", "passing_air_yards_delta", "passing_yards_after_catch_delta", 
    "air_yards_completion_delta", "air_yards_incompletion_delta",
    "pass_usage_delta", "pass_pct_of_offense_delta", "pass_air_yard_pct_delta", "pass_yards_after_catch_pct_delta",
    "pass_average_air_yards_delta",
    # "carries_delta", "rushing_yards_delta", "rushing_tds_delta", "rushing_fumbles_delta", 
    # "rusher_usage_delta", "rusher_fumble_pct_delta", "rusher_yards_per_carry_delta",
    "receptions_delta", "targets_delta", "receiving_yards_delta", "receiving_tds_delta", "receiving_fumbles_delta",
    "receiving_air_yards_delta", "receiving_yards_after_catch_delta", "receiver_usage_delta",
    "receiver_efficiency_delta", "receiver_yac_pct_delta", "receiver_yards_per_reception_delta", "receiver_yac_to_air_yards_delta",
    "racr_delta", "target_share_delta", "air_yards_share_delta", "wopr_delta",
]

rec_drop_columns = [
    "player_id", "player_display_name", "team", "position_group", "season", "week", "gametime", "player_display_name_season_avg",
    "offensive_snaps", "team_offensive_snaps", "snap_share",
    "attempts", "completions", "passing_yards",
    "passing_tds", "interceptions", "sacks", "sack_yards",
    "sack_fumbles", "passing_air_yards", "passing_yards_after_catch", 
    "air_yards_completion", "air_yards_incompletion",
    "pass_usage", "pass_pct_of_offense", "pass_air_yard_pct", "pass_yards_after_catch_pct",
    "pass_average_air_yards",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles", 
    "rusher_usage", "rusher_fumble_pct", "rusher_yards_per_carry",
    "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_fumbles",
    "receiving_air_yards", "receiving_yards_after_catch", "receiver_usage",
    "receiver_efficiency", "receiver_yac_pct", "receiver_yards_per_reception", "receiver_yac_to_air_yards",
    "racr", "target_share", "air_yards_share", "wopr", 
    "away_team", "away_score", "home_team", "home_score", 
    "result", "total", 
    # "overtime", "div_game", "isaway", "extended_away_games", "intl", "is_thursday",
    "lead_changes", 
    "attempts_delta", "completions_delta", "passing_yards_delta",
    "passing_tds_delta", "interceptions_delta", "sacks_delta", "sack_yards_delta",
    "sack_fumbles_delta", "passing_air_yards_delta", "passing_yards_after_catch_delta", 
    "air_yards_completion_delta", "air_yards_incompletion_delta",
    "pass_usage_delta", "pass_pct_of_offense_delta", "pass_air_yard_pct_delta", "pass_yards_after_catch_pct_delta",
    "pass_average_air_yards_delta",
    "carries_delta", "rushing_yards_delta", "rushing_tds_delta", "rushing_fumbles_delta", 
    "rusher_usage_delta", "rusher_fumble_pct_delta", "rusher_yards_per_carry_delta",
    # "receptions_delta", "targets_delta", "receiving_yards_delta", "receiving_tds_delta", "receiving_fumbles_delta",
    # "receiving_air_yards_delta", "receiving_yards_after_catch_delta", "receiver_usage_delta",
    # "receiver_efficiency_delta", "receiver_yac_pct_delta", "receiver_yards_per_reception_delta", "receiver_yac_to_air_yards_delta",
    # "racr_delta", "target_share_delta", "air_yards_share_delta", "wopr_delta",
]

# Build dataframes

rf_model_data_pass = data_analysis_delta_binary[data_analysis_delta_binary['position_group'] == 'QB'].copy()

rf_model_data_pass = rf_model_data_pass.drop(columns=pass_drop_columns)

rf_model_data_rush = data_analysis_delta_binary[data_analysis_delta_binary['position_group'] == 'RB'].copy()

rf_model_data_rush = rf_model_data_rush.drop(columns=rush_drop_columns)

rf_model_data_rec = data_analysis_delta_binary[data_analysis_delta_binary['position_group'].isin(['WR', 'RB', 'TE'])].copy()

rf_model_data_rec = rf_model_data_rec.drop(columns=rec_drop_columns)

# Check for NAs
print("rec has NaN:", rf_model_data_rec.isna().any().any())
print("rush has NaN:", rf_model_data_rush.isna().any().any())
print("pass has NaN:", rf_model_data_pass.isna().any().any())

# Impute with 0 for now (only travel_distance)
rf_model_data_rec.fillna(0, inplace=True)
rf_model_data_rush.fillna(0, inplace=True)
rf_model_data_pass.fillna(0, inplace=True)

print("NaNs imputed with a '0' value")


### PCA


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create a loop for PCA on all three position groups
dfs = {
    "rec":  rf_model_data_rec,
    "rush": rf_model_data_rush,
    "pass": rf_model_data_pass,
}

def pca_workflow(df: pd.DataFrame, name: str, exclude_cols=("result", "total"), var_target=0.80):
    print(f"\n===== PCA workflow for: {name} =====")
    
    # 1) Numeric-only feature matrix, drop target/other exclusions if present
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    df_numeric = df_numeric.drop(columns=[c for c in exclude_cols if c in df_numeric.columns], errors="ignore")
    
    # Safety: fill NaNs
    df_numeric = df_numeric.fillna(0)

    # If there are no numeric columns left, bail early
    if df_numeric.shape[1] == 0:
        print("No numeric feature columns available after exclusions.")
        return None

    # 2) Standardize
    X_scaled = StandardScaler().fit_transform(df_numeric)

    # 3) PCA with 10 comps (diagnostics)
    pca = PCA(n_components=min(10, df_numeric.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    print("Explained variance ratio:", explained_variance)
    print("Cumulative explained variance:", np.cumsum(explained_variance))

    plt.figure(figsize=(8,5))
    plt.plot(np.cumsum(explained_variance), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'PCA - Cumulative Explained Variance ({name})')
    plt.grid(True)
    plt.show()

    # 4) PCA scatter (first two components) — only if >=2 features
    if df_numeric.shape[1] >= 2:
        pca_2 = PCA(n_components=2)
        X_pca_2 = pca_2.fit_transform(X_scaled)
        plt.figure(figsize=(8,6))
        plt.scatter(X_pca_2[:,0], X_pca_2[:,1], alpha=0.7)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'PCA - First Two Components ({name})')
        plt.grid(True)
        plt.show()

    # 5) Choose components to reach ~80% variance (or var_target)
    pca_full = PCA().fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cum_var >= var_target) + 1)
    print(f"Number of components to capture {int(var_target*100)}% variance: {n_components}")

    pca_reduced = PCA(n_components=n_components)
    X_reduced = pca_reduced.fit_transform(X_scaled)

    # 6) Loadings & top contributors per PC
    loadings = pd.DataFrame(
        pca_reduced.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=df_numeric.columns
    )

    for i in range(n_components):
        print(f"\nTop contributing variables for PC{i+1} ({name}):")
        print(loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(5))

    # 7) Overall top variables by total absolute loading
    loadings['total_contribution'] = loadings.abs().sum(axis=1)
    k = min(25, loadings.shape[0])
    top_variables = loadings['total_contribution'].sort_values(ascending=False).head(k).index
    df_reduced = df_numeric[top_variables]
    print(f"\nReduced dataset shape (top {k} variables): {df_reduced.shape}")

    # 8) Correlation heatmap for the reduced set
    corr_matrix = df_reduced.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=(k <= 20),         # annotate only if not too crowded
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(f"Correlation Heatmap ({name}) - Top {k} Variables", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return {
        "df_numeric": df_numeric,
        "X_scaled": X_scaled,
        "pca_10": pca,
        "pca_reduced": pca_reduced,
        "n_components": n_components,
        "cum_var_at_n": cum_var[n_components-1],
        "loadings": loadings,
        "top_variables": list(top_variables),
        "df_reduced": df_reduced
    }

# Run for all three
results = {}
for name, dframe in dfs.items():
    results[name] = pca_workflow(dframe, name=name, exclude_cols=("result", "total"), var_target=0.80)

from IPython.display import display

# Top variables for PASS
display(results["pass"]["df_reduced"])
# Top variables for RUSH
display(results["rush"]["df_reduced"])
# Top variables for REC
display(results["rec"]["df_reduced"])


# ======================================================================================================================

# %% RF Models
# ### Random Forest Classification: Predicting `result` 
# This section trains a Random Forest to predict affect end result based on performance stats and game flags

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix)

### A List of variables to test on

# [
#     "snap_share_delta", "pass_usage_delta", "pass_pct_of_offense_delta", "pass_air_yard_pct_delta",
#     "pass_yards_after_catch_pct_delta", "pass_average_air_yards_delta",

#     "rusher_usage_delta", "rusher_fumble_pct_delta", "rusher_yards_per_carry_delta",

#     "receiver_usage_delta", "receiver_efficiency_delta", "receiver_yac_pct_delta",
#     "receiver_yards_per_reception_delta", "receiver_yac_to_air_yards_delta"
# ]

# CREATE RF MODELS THAT LOOP THROUGH ALL POSSIBLE TARGETS

# Use the reduced feature sets from PCA step
reduced_sets = {
    "pass": results["pass"]["df_reduced"],
    "rush": results["rush"]["df_reduced"],
    "rec":  results["rec"]["df_reduced"],
}

# Features that should NEVER be targets
excluded_targets = ["isaway", "is_international", "is_thursday"]

all_results = []

for segment, df_reduced in reduced_sets.items():
    print(f"\n================= SEGMENT: {segment.upper()} =================")
    # Candidate targets = all columns in df_reduced except excluded ones
    candidate_targets = [col for col in df_reduced.columns if col not in excluded_targets]

    seg_results = []

    for target in candidate_targets:
        print(f"\n----- Processing target: {target} -----")

        # All other variables become features — EXCLUDING the current target
        feature_cols = [col for col in df_reduced.columns if col != target]

        # Prepare X and y
        X = df_reduced[feature_cols].copy()
        y = df_reduced[target].copy()

        # Remove rows with missing values in either X or y
        valid_mask = ~y.isna() & ~X.isna().any(axis=1)
        X_final = X.loc[valid_mask]
        y_final = y.loc[valid_mask]

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

            row = {
                "segment": segment,
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
            }

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

            row = {
                "segment": segment,
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
            }

        seg_results.append(row)
        all_results.append(row)

    # Per-segment summary + CSV
    if seg_results:
        seg_df = pd.DataFrame(seg_results).sort_values(
            by=["type", "f1_score", "r2"], ascending=[True, False, False]
        )
        print(f"\nTop 15 for segment {segment.upper()}:")
        print(seg_df.head(15))
        seg_df.to_csv(f"rf_full_metrics_comparison_{segment}.csv", index=False)

# Combined results
results_df = pd.DataFrame(all_results).sort_values(
    by=["segment", "type", "f1_score", "r2"], ascending=[True, True, False, False]
)
print("\nCombined top 20 overall:")
print(results_df.head(20))
results_df.to_csv("rf_full_metrics_comparison_ALL.csv", index=False)

# # =====================================================================================================================
# ### Kesh's original feature importance list

# ### Print Feature importance list
# def top_feature_importances(rf, feature_names, top_n=25):
#     """
#     rf: a fitted RandomForestClassifier/Regressor
#     feature_names: list/Index of column names used to fit rf
#     """
#     # Per-tree importances -> mean/std across trees
#     tree_imps = np.vstack([est.feature_importances_ for est in rf.estimators_])
#     mean_imp = tree_imps.mean(axis=0)


#     fi = (
#         pd.DataFrame({
#             "feature": feature_names,
#             "mean_importance": mean_imp,
#         })
#         .sort_values("mean_importance", ascending=False)
#         .head(top_n)
#         .reset_index(drop=True)
#     )

#     # Pretty print
#     print(fi.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
#     return fi
# # =====================================================================================================================

# # %% pass_air_yard_pct_delta - QB

# #Define input and target
# X = rf_model_data_pass.drop(columns=["pass_air_yard_pct_delta", "passing_air_yards_delta"])
# y = rf_model_data_pass["pass_air_yard_pct_delta"]

# ### Values are binary or numeric
# # Convert categorical variables to dummy/indicator variables
# # X = pd.get_dummies(X, drop_first=True)

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# # Fit Random Forest
# print("Fitting Model")
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# # Predict and evaluate
# print("Predicting and Evaluating Model")
# y_pred = rf.predict(X_test)

# print("Classification Report for predicting `pass_air_yard_pct` from rf_model_data_pass:\n")
# print(classification_report(y_test, y_pred))
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# importances = rf.feature_importances_
# feature_names = X.columns

# sorted_idx = importances.argsort()[::-1][:25]
# plt.figure(figsize=(10, 15))
# plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
# plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
# plt.xlabel("Feature Importance")
# plt.title("Top Features for Predicting `pass_air_yard_pct` for PASS stats")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()

# top25_pass = top_feature_importances(rf, X.columns, top_n=25)

# print("top25_pass")
# print(top25_pass)

# # %% rusher_yards_per_carry_delta - RUSH

# #Define input and target
# X = rf_model_data_rush.drop(columns=["rusher_yards_per_carry_delta", "rushing_yards_delta"])
# y = rf_model_data_rush["rusher_yards_per_carry_delta"]

# ### Values are binary or numeric
# # Convert categorical variables to dummy/indicator variables
# # X = pd.get_dummies(X, drop_first=True)

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# # Fit Random Forest
# print("Fitting Model")
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# # Predict and evaluate
# print("Predicting and Evaluating Model")
# y_pred = rf.predict(X_test)

# print("Classification Report for predicting `rusher_yards_per_carry` from rf_model_data_rush:\n")
# print(classification_report(y_test, y_pred))
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# importances = rf.feature_importances_
# feature_names = X.columns

# sorted_idx = importances.argsort()[::-1][:25]
# plt.figure(figsize=(10, 15))
# plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
# plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
# plt.xlabel("Feature Importance")
# plt.title("Top Features for Predicting `rusher_yards_per_carry` for RUSH stats")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()

# top25_rush = top_feature_importances(rf, X.columns, top_n=25)

# print("top25_rush")
# print(top25_rush)

# # %% receiver_yac_pct_delta - REC

# #Define input and target
# X = rf_model_data_rec.drop(columns=["receiver_yac_pct_delta", "receiving_yards_after_catch_delta", "receiver_yac_to_air_yards_delta"]) 
# y = rf_model_data_rec["receiver_yac_pct_delta"]

# ### Values are binary or numeric
# # Convert categorical variables to dummy/indicator variables
# # X = pd.get_dummies(X, drop_first=True)

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# # Fit Random Forest
# print("Fitting Model")
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# # Predict and evaluate
# print("Predicting and Evaluating Model")
# y_pred = rf.predict(X_test)

# print("Classification Report for predicting `receiver_yac_pct` from rf_model_data_rec:\n")
# print(classification_report(y_test, y_pred))
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# importances = rf.feature_importances_
# feature_names = X.columns

# sorted_idx = importances.argsort()[::-1][:25]
# plt.figure(figsize=(10, 15))
# plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
# plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
# plt.xlabel("Feature Importance")
# plt.title("Top Features for Predicting `receiver_yac_pct` for REC stats")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()

# top25_rec = top_feature_importances(rf, X.columns, top_n=25)

# print("top25_rec")
# print(top25_rec)

print('FINITO')












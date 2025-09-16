#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 20:51:31 2025

@author: nikesh
"""

### STARTING DATA - df_merged csv from Kesh's branch on git


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import janitor  # extends pandas functionality
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

df_merged = pd.read_csv('/Users/nikesh/NFL Project/df_merged.csv')
print(df_merged.head())

print("\n".join(df_merged.columns))

# Save column names with datatypes to a txt file
# with open("df_merged_columns.txt", "w") as f:
#    for col, dtype in zip(df_merged.columns, df_merged.dtypes):
#       f.write(f"{col} : {dtype}\n")

#############################
#### Create Season Data #####
#############################
###
# Create a new dataframe for analysis 

df_analysis = (
    df_merged
      .loc[df_merged["position_group"].isin(["QB", "RB", "FB", "WR", "TE"])]
      .assign(
          snap_share = lambda d: d["offensive_snaps"].div(d["team_offensive_snaps"]),                              # On-Field %
          
          pass_usage = lambda d: d["attempts"].div(d["offensive_snaps"]),                                          # Passing Play Usage 
          pass_pct_of_offense = lambda d: d["attempts"].div(d["team_offensive_snaps"]),                            # Passing % of Offense
          pass_air_yard_pct = lambda d: d["passing_air_yards"].div(d["passing_yards"]),                            # Pass effectiveness
          pass_yards_after_catch_pct = lambda d: d["passing_yards_after_catch"].div(d["passing_yards"]),           # Receiver Reliance
          pass_average_air_yards = lambda d: d[["air_yards_completion", "air_yards_incompletion"]].mean(axis=1),   # Ratio of Air Yards to Total Pass Yards
          
          rusher_usage = lambda d: d["carries"].div(d["offensive_snaps"]),                                  # Rusher Usage
          rusher_fumble_pct = lambda d: d["rushing_fumbles"].div(d["carries"]),                             # Rusher Fumble %
          rusher_yards_per_carry = lambda d: d["rushing_yards"].div(d["carries"]),                          # Yards per Carry
          
          receiver_usage = lambda d: d["targets"].div(d["offensive_snaps"]),                                  # Receiver Usage
          receiver_efficiency = lambda d: d["receptions"].div(d["targets"]),                                  # Receiver Efficiency
          receiver_yac_pct = lambda d: d["receiving_yards_after_catch"].div(d["receiving_yards"]),            # Receiving after-catch %
          receiver_yards_per_reception = lambda d: d["receiving_yards"].div(d["receptions"]),                     # Yards per Reception
          receiver_yac_to_air_yards = lambda d: d["receiving_yards_after_catch"].div(d["receiving_air_yards"]),   # YAC to Air Yards Ratio
        )
      .replace([np.inf, -np.inf], np.nan)
      .copy()
)
     

# Columns to drop
df_cols_to_drop = [
    "season_type",
    "opponent_team", "depth_chart_position", "jersey_number", "football_name", "recent_team",
    "status", "status_description_abbr", "game_type", "player_name", "position",
    "game_id", "gameday", "weekday", "location", "stadium", "stadium_id", 
    "old_game_id", "gsis", "away_rest", "home_rest", "roof", "surface", "temp", "wind",
    "is_international", "week_after_intl", "defensive_snaps", "team_defensive_snaps",
    "special_team_snaps", "team_special_team_snaps",
    "sack_fumbles_lost", "passing_first_downs", "passing_2pt_conversions", 
    "rushing_first_downs", "rushing_fumbles_lost", 
    "receiving_fumbles_lost", "receiving_first_downs", "receiving_2pt_conversions", "yards_after_catch",
    "player_name_flat"
    ]


# Drop unnecessary columns
data_analysis = df_analysis.drop(columns=df_cols_to_drop)

print("COLUMNS IN ORDER")

print("\n".join(data_analysis.columns))

columns_in_order = [
    "player_id", "player_display_name", "team", "position_group", "season", "week", "gametime",
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
    "away_team", "away_score", "home_team", "home_score", "result", "total", 
    "overtime", "div_game", "isaway", "extended_away_games", "intl", "is_thursday",
    "lead_changes", "travel_distance_away"
]

data_analysis = data_analysis[columns_in_order]

# data_analysis.to_csv('data_analysis.csv', index=False)


### Finding Player Season Average to compare

average_stats = [
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
    ]


# Group by player_id and season, then average the stats
player_season_averages = (
    data_analysis.groupby(["player_id", "season"])
      .agg(
          player_display_name=("player_display_name", "first"),
          **{col: (col, lambda x: round(x.mean(), 2)) for col in average_stats}
      )
      .reset_index()
      .sort_values(by=["player_id", "season"])
)

print(player_season_averages.head())

# player_season_averages.to_csv('player_season_averages.csv', index=False)

### Create comparison of weekly stats to season averages
#
# Merge season averages back into the main dataframe
data_analysis = data_analysis.merge(
    player_season_averages,
    on=["player_id", "season"],
    suffixes=("", "_season_avg")
    )

# Create difference columns
for col in average_stats:
    data_analysis[f"{col}_delta"] = (
        data_analysis[col] - data_analysis[f"{col}_season_avg"]
    )

# Drop the season_avg columns to keep it light
season_avg_cols = [f"{col}_season_avg" for col in average_stats]
data_analysis = data_analysis.drop(columns=season_avg_cols)

data_analysis_delta_binary = data_analysis.copy()

# Find all columns with the "_delta" suffix
delta_cols = [col for col in data_analysis_delta_binary.columns if col.endswith("_delta")]

# Replace values: < 0 → 0, >= 0 → 1
data_analysis_delta_binary[delta_cols] = (data_analysis_delta_binary[delta_cols] >= 0).astype(int)

print(data_analysis_delta_binary.head())

data_analysis.to_csv('data_analysis.csv', index=False)

data_analysis_delta_binary.to_csv('data_analysis_delta_binary.csv', index=False)

#============================================================================

### Get averages by game flags

# Stat columns you want to average
delta_cols = [col + "_delta" for col in average_stats]

stat_cols = average_stats + delta_cols

# Binary flags to split on
binary_flags = ["isaway", "is_thursday", "extended_away_games", "intl"]

# Store results
all_results = {}

for flag in binary_flags:
    tmp = (
        data_analysis.groupby(["position_group", "season", flag])[stat_cols]
          .mean()
          .reset_index()
    )
    tmp[stat_cols] = tmp[stat_cols].round(2)
    tmp["flag"] = flag
    tmp = tmp.rename(columns={flag: "flag_value"})
    
    # Save each individual dataframe to CSV
    tmp.to_csv(f"{flag}_season_position_avgs.csv", index=False)
    
    all_results[flag] = tmp

# Combine into one long dataframe
flag_season_avgs = pd.concat(all_results.values(), ignore_index=True)

# Ensure it's sorted/grouped by position_group, then season, then flag
flag_season_avgs = flag_season_avgs.sort_values(
    by=["position_group", "season", "flag", "flag_value"]
)

# Move `flag` and 'flag_value' to be the first column
cols = ["flag", "flag_value"] + [col for col in flag_season_avgs.columns if col not in ["flag", "flag_value"]]

flag_season_avgs = flag_season_avgs[cols]

# Save season averages based on different game flags to csv
flag_season_avgs.to_csv("all_flag_season_position_avgs.csv", index=False)

# Roll up the averages across all four seasons (T4)
flag_aggregate_avgs = (flag_season_avgs.groupby(["position_group", "flag", "flag_value"])[stat_cols]
      .mean()
      .reset_index()
)

### Save 4-season average for position groups to csv
# send the T4 averages to csv
flag_aggregate_avgs.to_csv("all_flag_aggregate_position_avgs.csv", index=False)

#======================================================================================================

# Recreate RF models to predict player performance

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
    "lead_changes", "travel_distance_away",
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
    "lead_changes", "travel_distance_away",
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
    "lead_changes", "travel_distance_away",
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

rf_model_data_pass = data_analysis_delta_binary[data_analysis_delta_binary['position_group'] == 'QB'].copy()

rf_model_data_pass = rf_model_data_pass.drop(columns=pass_drop_columns)

rf_model_data_rush = data_analysis_delta_binary[data_analysis_delta_binary['position_group'] == 'RB'].copy()

rf_model_data_rush = rf_model_data_rush.drop(columns=rush_drop_columns)

rf_model_data_rec = data_analysis_delta_binary[data_analysis_delta_binary['position_group'].isin(['WR', 'RB', 'TE'])].copy()

rf_model_data_rec = rf_model_data_rec.drop(columns=rec_drop_columns)

print("rec has NaN:", rf_model_data_rec.isna().any().any())
print("rush has NaN:", rf_model_data_rush.isna().any().any())
print("pass has NaN:", rf_model_data_pass.isna().any().any())

rf_model_data_rec.fillna(0, inplace=True)
rf_model_data_rush.fillna(0, inplace=True)
rf_model_data_pass.fillna(0, inplace=True)

print("NaNs imputed with a '0' value")

# ======================================================================================================================

# %% RF Models
# ### Random Forest Classification: Predicting `result` 
# This section trains a Random Forest to predict affect end result based on performance stats and game flags


### Variables to test on

# [
#     "snap_share_delta", "pass_usage_delta", "pass_pct_of_offense_delta", "pass_air_yard_pct_delta",
#     "pass_yards_after_catch_pct_delta", "pass_average_air_yards_delta",

#     "rusher_usage_delta", "rusher_fumble_pct_delta", "rusher_yards_per_carry_delta",

#     "receiver_usage_delta", "receiver_efficiency_delta", "receiver_yac_pct_delta",
#     "receiver_yards_per_reception_delta", "receiver_yac_to_air_yards_delta"
# ]


### Print Feature importance list
def top_feature_importances(rf, feature_names, top_n=25):
    """
    rf: a fitted RandomForestClassifier/Regressor
    feature_names: list/Index of column names used to fit rf
    """
    # Per-tree importances -> mean/std across trees
    tree_imps = np.vstack([est.feature_importances_ for est in rf.estimators_])
    mean_imp = tree_imps.mean(axis=0)


    fi = (
        pd.DataFrame({
            "feature": feature_names,
            "mean_importance": mean_imp,
        })
        .sort_values("mean_importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Pretty print
    print(fi.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return fi

# %% pass_air_yard_pct_delta - QB

#Define input and target
X = rf_model_data_pass.drop(columns=["pass_air_yard_pct_delta", "passing_air_yards_delta"])
y = rf_model_data_pass["pass_air_yard_pct_delta"]

### Values are binary or numeric
# Convert categorical variables to dummy/indicator variables
# X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Fit Random Forest
print("Fitting Model")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
print("Predicting and Evaluating Model")
y_pred = rf.predict(X_test)

print("Classification Report for predicting `pass_air_yard_pct` from rf_model_data_pass:\n")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

importances = rf.feature_importances_
feature_names = X.columns

sorted_idx = importances.argsort()[::-1][:25]
plt.figure(figsize=(10, 15))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top Features for Predicting `pass_air_yard_pct` for PASS stats")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

top25_pass = top_feature_importances(rf, X.columns, top_n=25)

print("top25_pass")
print(top25_pass)

# %% rusher_yards_per_carry_delta - RUSH

#Define input and target
X = rf_model_data_rush.drop(columns=["rusher_yards_per_carry_delta", "rushing_yards_delta"])
y = rf_model_data_rush["rusher_yards_per_carry_delta"]

### Values are binary or numeric
# Convert categorical variables to dummy/indicator variables
# X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Fit Random Forest
print("Fitting Model")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
print("Predicting and Evaluating Model")
y_pred = rf.predict(X_test)

print("Classification Report for predicting `rusher_yards_per_carry` from rf_model_data_rush:\n")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

importances = rf.feature_importances_
feature_names = X.columns

sorted_idx = importances.argsort()[::-1][:25]
plt.figure(figsize=(10, 15))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top Features for Predicting `rusher_yards_per_carry` for RUSH stats")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

top25_rush = top_feature_importances(rf, X.columns, top_n=25)

print("top25_rush")
print(top25_rush)

# %% receiver_yac_pct_delta - REC

#Define input and target
X = rf_model_data_rec.drop(columns=["receiver_yac_pct_delta", "receiving_yards_after_catch_delta", "receiver_yac_to_air_yards_delta"]) 
y = rf_model_data_rec["receiver_yac_pct_delta"]

### Values are binary or numeric
# Convert categorical variables to dummy/indicator variables
# X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Fit Random Forest
print("Fitting Model")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
print("Predicting and Evaluating Model")
y_pred = rf.predict(X_test)

print("Classification Report for predicting `receiver_yac_pct` from rf_model_data_rec:\n")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

importances = rf.feature_importances_
feature_names = X.columns

sorted_idx = importances.argsort()[::-1][:25]
plt.figure(figsize=(10, 15))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top Features for Predicting `receiver_yac_pct` for REC stats")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

top25_rec = top_feature_importances(rf, X.columns, top_n=25)

print("top25_rec")
print(top25_rec)

## Map Team abbr to numeric values

# # Fixed NFL team mapping starting at 1
# team_mapping = {
#     'ARI': 1, 'ATL': 2, 'BAL': 3, 'BUF': 4, 'CAR': 5, 'CHI': 6, 'CIN': 7,
#     'CLE': 8, 'DAL': 9, 'DEN': 10, 'DET': 11, 'GB': 12, 'HOU': 13, 'IND': 14,
#     'JAX': 15, 'KC': 16, 'LV': 17, 'LAC': 18, 'LA': 19, 'MIA': 20, 'MIN': 21,
#     'NE': 22, 'NO': 23, 'NYG': 24, 'NYJ': 25, 'PHI': 26, 'PIT': 27,
#     'SEA': 28, 'SF': 29, 'TB': 30, 'TEN': 31, 'WAS': 32
# }

# # Map and convert to categorical
# data_analysis["team"] = data_analysis["team"].map(team_mapping).astype("category")
# data_analysis["home_team"] = data_analysis["home_team"].map(team_mapping).astype("category")
# data_analysis["away_team"] = data_analysis["away_team"].map(team_mapping).astype("category")

# print("Mapping Done")



print('FINITO')












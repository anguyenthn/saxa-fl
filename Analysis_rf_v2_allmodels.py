#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 01:09:12 2025

@author: nikesh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import janitor  # extends pandas functionality
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  # or RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report

weekly_data_import = pd.read_csv('/Users/nikesh/NFL Project/weekly_data.csv')
print(weekly_data_import.head())


# %%
### Begin Data Cleaning 
###

# Remove all rows from POSTSEASON Games
df = weekly_data_import[weekly_data_import['season_type'] != 'POST'].copy()

# Create column isaway to determine if player is playing away from home (0 = Home, 1 = Away)
df['isaway'] = (df['recent_team'] != df['home_team']).astype(int)

# Extract all unique values from stadium_id column
unique_stadium_ids = df['stadium_id'].dropna().unique().tolist()
print(unique_stadium_ids)

# Define set of international stadium_ids
intl_stadiums = {'LON00', 'LON02', 'MEX00', 'FRA00', 'GER00', 'SAO00'}

# Create new column is_international: 1 if stadium is outside US, else 0
df['is_international'] = df['stadium_id'].isin(intl_stadiums).astype(int)

# Extract all unique values from status_description_abbr column
status_abbr = df['status_description_abbr'].dropna().unique().tolist()
print(status_abbr)

# Extract all unique values from position group column
position_group = df['position_group'].dropna().unique().tolist()
print(position_group)

# Sort the dataframe by season, then by player_id, then by week in ascending order
df = df.sort_values(by=["season", "player_id", "week"], ascending=[True, True, True]).reset_index(drop=True)

### Add additional classification column for games after multi-away stints
# Compute prior weeks and isaway flags within each player_id group
g = df.groupby(["season", "player_id"], group_keys=False)
lag1_week = g["week"].shift(1)
lag2_week = g["week"].shift(2)
lag1_isaway = g["isaway"].shift(1)
lag2_isaway = g["isaway"].shift(2)

# Require strictly sequential weeks and both prior isaway == 1
seq_ok = (df["week"].eq(lag1_week + 1)) & (lag1_week.eq(lag2_week + 1))
away_ok = (lag1_isaway.eq(1)) & (lag2_isaway.eq(1))

df["extended_away_games"] = (seq_ok & away_ok).astype(int)

# Compute prior weeks and isintl flags within each player_id group
g = df.groupby(["season", "player_id"], group_keys=False)
lag1_week = g["week"].shift(1)
lag1_isintl = g["is_international"].shift(1)

# Marks week directly after intl game 
seq_ok = df["week"].notna() & lag1_week.notna() & (df["week"] == lag1_week + 1)
intl_ok = (lag1_isintl == 1)

df["week_after_intl"] = (seq_ok & intl_ok).astype(int)

# Create binary Thursday Game Identifier
df["is_thursday"] = df["weekday"].str.lower().eq("thursday").astype(int)

# Convert gametime (HH:MM) to datetime.time format
df["gametime"] = pd.to_datetime(df["gametime"], format="%H:%M", errors="coerce").dt.time

# Creates Binary for intl or week following
# Create a binary target column: 1 if either is_international or week_after_intl is 1
df["intl"] = ((df["is_international"] == 1) | (df["week_after_intl"] == 1)).astype(int)

# %% Impute Wind/Temp values
# Extract all unique values from stadium_id column
unique_stadium_ids = df['stadium_id'].dropna().unique().tolist()
print(unique_stadium_ids)

# Compute average temperature per stadium_id
avg_temp_by_stadium = (
    df.groupby("stadium_id", dropna=True)["temp"]
      .mean()
      .reset_index()
      .rename(columns={"temp": "avg_temp"})
      .sort_values("avg_temp", ascending=False)
)

print(avg_temp_by_stadium)

# Build mapping from stadium_id to avg_temp (drop NaNs first)
avg_temp_map = (
    avg_temp_by_stadium.dropna(subset=["avg_temp"])
    .set_index("stadium_id")["avg_temp"]
    .round(1)
    .to_dict()
)

# Impute temp using map; fallback to 60 degrees due to indoor/dome stadium
df["temp"] = df.apply(
    lambda row: avg_temp_map.get(row["stadium_id"], 60) if pd.isna(row["temp"]) else row["temp"],
    axis=1
)

# Compute average wind per stadium_id
avg_wind_by_stadium = (
    df.groupby("stadium_id", dropna=True)["wind"]
      .mean()
      .reset_index()
      .rename(columns={"wind": "avg_wind"})
      .sort_values("avg_wind", ascending=False)
)

# Build mapping from stadium_id to avg_wind (drop NaNs first)
avg_wind_map = (
    avg_wind_by_stadium.dropna(subset=["avg_wind"])
    .set_index("stadium_id")["avg_wind"]
    .round(1)
    .to_dict()
)

# Impute wind using map; fallback to 0 wind speed due to indoor/dome stadium
df["wind"] = df.apply(
    lambda row: avg_temp_map.get(row["stadium_id"], 0) if pd.isna(row["wind"]) else row["wind"],
    axis=1
)

###
# Add in missing surface info, for intl and SF stadium

# Set surface to 'grass' for specific stadiums
grass_stadiums = ['SFO01', 'GER00', 'MEX00', 'LON00', 'FRA00', 'SAO00']

df.loc[df["stadium_id"].isin(grass_stadiums), "surface"] = "grass"

# Drop rows where game_id is NaN
df = df[df["game_id"].notna()].copy()

# Drop the nfl_detail_id column due to NaNs
df = df.drop(columns=["nfl_detail_id"])

# Get the most common (mode) surface for each stadium_id, ignoring NaNs
most_common_surface = (
    df.dropna(subset=["surface"])
      .groupby("stadium_id")["surface"]
      .agg(lambda x: x.mode().iloc[0])
      .reset_index()
      .rename(columns={"surface": "most_common_surface"})
)

# Create mapping from stadium_id to most common surface
surface_map = most_common_surface.set_index("stadium_id")["most_common_surface"].to_dict()

# Fill NaNs in surface column using the map
df["surface"] = df.apply(
    lambda row: surface_map.get(row["stadium_id"], row["surface"]) if pd.isna(row["surface"]) else row["surface"],
    axis=1
)

# Check which columns still have NaN values
nan_summary = df.isna().sum()
nan_summary = nan_summary[nan_summary > 0].sort_values(ascending=False)

print("Columns with NaN values:")
print(nan_summary)

### Check csv file
df.to_csv('df_check.csv', index=False)

print("Done")

# %% RF Model
############################
#### Create RF dataset #####
############################

# Columns to drop
cols_to_drop = [
    "player_id", "player_display_name", "recent_team", "season_type",
    "opponent_team", "depth_chart_position", "jersey_number", "football_name",
    "status", "status_description_abbr", "game_type", "player_name", "position",
    "game_id", "gameday", "gametime", "weekday", "location", "stadium", "stadium_id", 
    "old_game_id", "gsis", "away_rest", "home_rest", "roof", "surface", "temp", "wind",
    "is_international", "week_after_intl"
]

# Create new DataFrame with specified columns dropped
rf_data_full = df.drop(columns=cols_to_drop)

## Map Team abbr to numeric values

# Fixed NFL team mapping starting at 1
team_mapping = {
    'ARI': 1, 'ATL': 2, 'BAL': 3, 'BUF': 4, 'CAR': 5, 'CHI': 6, 'CIN': 7,
    'CLE': 8, 'DAL': 9, 'DEN': 10, 'DET': 11, 'GB': 12, 'HOU': 13, 'IND': 14,
    'JAX': 15, 'KC': 16, 'LV': 17, 'LAC': 18, 'LA': 19, 'MIA': 20, 'MIN': 21,
    'NE': 22, 'NO': 23, 'NYG': 24, 'NYJ': 25, 'PHI': 26, 'PIT': 27,
    'SEA': 28, 'SF': 29, 'TB': 30, 'TEN': 31, 'WAS': 32
}

# Map and convert to categorical
rf_data_full["team"] = rf_data_full["team"].map(team_mapping).astype("category")
rf_data_full["home_team"] = rf_data_full["home_team"].map(team_mapping).astype("category")
rf_data_full["away_team"] = rf_data_full["away_team"].map(team_mapping).astype("category")

print("Mapping Done")

### R-Style Variable Summary
def r_style_summary(rf_data_full):
    return pd.DataFrame({
        "dtype": rf_data_full.dtypes,
        "non_nulls": rf_data_full.notnull().sum(),
        "nulls": rf_data_full.isnull().sum(),
        "unique_vals": rf_data_full.nunique(),
        "example_val": rf_data_full.apply(
            lambda x: x.dropna().unique()[0] if x.notna().any() else None
        )
    })

# Print the R-style summary
print("R-style Metadata Summary:")
print(r_style_summary(rf_data_full))

# df for summary txt file
summary_stats = r_style_summary(rf_data_full)

# Write to a txt file 
with open('rf_data_full.txt', 'w') as f:
    f.write("R-style Metadata Summary:\n\n")
    f.write(summary_stats.to_string())

# Weekly Data dictionary
rf_data_dict = pd.DataFrame({
    'column': rf_data_full.columns,
    'dtype': rf_data_full.dtypes.values,
    'non_nulls': rf_data_full.notnull().sum().values,
    'nulls': rf_data_full.isnull().sum().values,
    'unique_vals': rf_data_full.nunique().values,
    'example_val': rf_data_full.apply(lambda x: x.dropna().unique()[0] if x.notna().any() else None).values,
    'description': ['' for _ in rf_data_full.columns]  # Placeholder for manual descriptions
})

# Preview it
print(rf_data_dict.head())

rf_data_dict.to_csv('rf_data_dictionary.csv', index=False)

### Check csv file
rf_data_full.to_csv('rf_data_check.csv', index=False)


### Break out positions for RF target analysis (like-stat models)

pass_stats = ["attempts", "completions", "passing_yards", "passing_tds", "interceptions",
              "sacks", "sack_yards", "sack_fumbles", "sack_fumbles_lost",
              "passing_air_yards", "passing_yards_after_catch",
              "passing_first_downs", "passing_2pt_conversions"
              ]

rush_stats = ["carries", "rushing_yards", "rushing_tds",
              "rushing_fumbles", "rushing_fumbles_lost", "rushing_first_downs"
              ]


rec_stats = ['receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles',
             'receiving_fumbles_lost', 'receiving_air_yards', 'receiving_yards_after_catch', 'receiving_first_downs', 
             'receiving_2pt_conversions', 'racr', 'target_share', 'air_yards_share', 'wopr'
             ]


#Encode the data 
# Convert categorical columns to integer codes for Random Forest modeling
categorical_cols = ["team", "home_team", "away_team"]

for col in categorical_cols:
    rf_data_full[col] = rf_data_full[col].cat.codes
    
print("Data Encoded for RF Model")

##
## QB Data
##

rf_data_qb = rf_data_full[rf_data_full['position_group'] == 'QB'].copy()

# Drop Position Group and non-pass-stat Columns
rf_data_qb = rf_data_qb.drop(columns=["position_group"] + rush_stats + rec_stats)

### Check csv file
rf_data_qb.to_csv('rf_qb_data.csv', index=False)

print("QB Data Done")

##
## RUSH Data
##

rf_data_rush = rf_data_full[rf_data_full['position_group'] == 'RB'].copy()

rf_data_rush = rf_data_rush.drop(columns=["position_group"] + pass_stats + rec_stats)

rf_data_rush.to_csv('rf_rush_data.csv', index=False)

print("Rush Data Done")

##
## REC Data
##

rf_data_rec = rf_data_full[rf_data_full['position_group'].isin(['WR', 'RB', 'TE'])].copy()

rf_data_rec = rf_data_rec.drop(columns=["position_group"] + pass_stats + rush_stats)

rf_data_rec.to_csv('rf_rec_data.csv', index=False)

print("REC Data Done")

#########################
### Build RF models #####
#########################

# Prepare datasets and targets
datasets = {
    "rf_data_qb": rf_data_qb,
    "rf_data_rush": rf_data_rush,
    "rf_data_rec": rf_data_rec
}

targets = ["isaway", "intl", "is_thursday"]

# To store full classification report details
full_report_rows = []

# Loop through datasets and targets
for name, data in datasets.items():
    for target_col in targets:
        print(f"\n\n==============================")
        print(f"Dataset: {name} | Target: {target_col}")
        print(f"==============================")

        # Drop target and convert categoricals
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Handle categoricals
        X = pd.get_dummies(X, drop_first=True)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Fit model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Predict and collect full classification report
        y_pred = rf.predict(X_test)
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        # Flatten into row-per-label format
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                full_report_rows.append({
                    "dataset": name,
                    "target": target_col,
                    "label": label,
                    "precision": round(metrics.get("precision", 0), 4),
                    "recall": round(metrics.get("recall", 0), 4),
                    "f1-score": round(metrics.get("f1-score", 0), 4),
                    "support": int(metrics.get("support", 0))
                })

        # Feature importances
        importances = rf.feature_importances_
        feature_names = X.columns
        sorted_idx = importances.argsort()[::-1][:15]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title(f"{name} — Top Features for `{target_col}`")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

# Build final classification report DataFrame
classification_report_df = pd.DataFrame(full_report_rows)

# Display it
print("\n Full Classification Report for All Models:")
print(classification_report_df)

### Check csv file
classification_report_df.to_csv('rf_classification_report.csv', index=False)

print("Done")



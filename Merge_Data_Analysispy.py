#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 14:27:11 2025

@author: nikesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import janitor  # extends pandas functionality
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  # or RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report

# Import NFL Datasets
receiving_data_import = pd.read_csv('/Users/nikesh/NFL Project/NFL Datasets/receiving_data_NFL.csv')
official_schedule_data_import = pd.read_csv('/Users/nikesh/NFL Project/NFL Datasets/schedule_2002_2024_NFL.csv')
snap_counts_data_import = pd.read_csv('/Users/nikesh/NFL Project/NFL Datasets/snap_counts_2021_2024_NFL.csv')
air_yards_data_import = pd.read_csv('/Users/nikesh/NFL Project/NFL Datasets/air_yards_data_NFL.csv')
lead_changes_data_import = pd.read_csv('/Users/nikesh/NFL Project/NFL Datasets/lead_changes_NFL.csv')

#
# Import Public Data
weekly_data_import = pd.read_csv('/Users/nikesh/NFL Project/weekly_data.csv')
# print(weekly_data_import.head())

### R-Style Variable Summary
def r_style_summary(weekly_data_import):
    return pd.DataFrame({
        "dtype": weekly_data_import.dtypes,
        "non_nulls": weekly_data_import.notnull().sum(),
        "nulls": weekly_data_import.isnull().sum(),
        "unique_vals": weekly_data_import.nunique(),
        "example_val": weekly_data_import.apply(
            lambda x: x.dropna().unique()[0] if x.notna().any() else None
        )
    })

# Print the R-style summary

# print("R-style Metadata Summary:")
# print(r_style_summary(weekly_data_import))

# df for summary txt file
summary_stats = r_style_summary(weekly_data_import)

# Write to a txt file 
with open('weekly_data_summary.txt', 'w') as f:
    f.write("R-style Metadata Summary:\n\n")
    f.write(summary_stats.to_string())

# Weekly Data dictionary
weekly_data_dict = pd.DataFrame({
    'column': weekly_data_import.columns,
    'dtype': weekly_data_import.dtypes.values,
    'non_nulls': weekly_data_import.notnull().sum().values,
    'nulls': weekly_data_import.isnull().sum().values,
    'unique_vals': weekly_data_import.nunique().values,
    'example_val': weekly_data_import.apply(lambda x: x.dropna().unique()[0] if x.notna().any() else None).values,
    'description': ['' for _ in weekly_data_import.columns]  # Placeholder for manual descriptions
})

# Preview it
# print(weekly_data_dict.head())

weekly_data_dict.to_csv('weekly_data_dictionary.csv', index=False)

###
### Begin Data Cleaning 
###

# CLEANING MERGE DATA

#===================================================================================================
### Receiving Dataset - player level data

# Keep only rows where season >= 2021
df_receiving = receiving_data_import[receiving_data_import['season'].between(2021, 2024)].copy()

# Show all unique team names
# unique_teams = df_receiving['club_name'].unique()
# print("Unique club_name values:")
# print(unique_teams)

# Map full club names to abbreviations
club_name_map = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS", "Washington Football Team": "WAS"
}

df_receiving['team'] = df_receiving['club_name'].map(club_name_map)

# Create a new player_name column (first_name + last_name)
df_receiving['player_name_flat'] = (
    df_receiving['first_name'].astype(str) + df_receiving['last_name'].astype(str)
)

# Remove special characters and spaces
df_receiving['player_name_flat'] = df_receiving['player_name_flat'].apply(
    lambda x: re.sub(r'[^A-Za-z0-9]', '', x)  # keep only letters and numbers
)

### Check csv file
df_receiving.to_csv('df_receiving_check.csv', index=False)
print("Receiving_Data Cleaned")

#===================================================================================================
# Air Yards Dataset - player-level data

# There is no player column for these, this will get amrked to the qb from each game

# copy import dataset
df_air_yards = air_yards_data_import.copy()

# Extract season year from game_id (first 4 digits)
df_air_yards['season'] = df_air_yards['game_id'].astype(str).str[:4].astype(int)

# Keep only 2021–2024
df_air_yards = df_air_yards[df_air_yards['season'].between(2021, 2024)]

df_air_yards['team'] = df_air_yards['club_name'].map(club_name_map)

# ### Check number of rows per game_id
# #
# row_counts = df_air_yards.groupby('game_id').size().reset_index(name='row_count')
# # Show any game_ids that don't have exactly 2 rows
# bad_games = row_counts[row_counts['row_count'] != 2]
# #
# print(f"Total games: {row_counts.shape[0]}")
# print(f"Games with != 2 rows: {bad_games.shape[0]}")
# print(bad_games.head())
# #

### Check csv file
df_air_yards.to_csv('df_air_yards_check.csv', index=False)
print("Air_Yards_Data Cleaned")

#===================================================================================================
# Lead Changes Dataset - Game level data

# copy import dataset
df_lead_changes = lead_changes_data_import.copy()

# Extract season year from game_id (first 4 digits)
df_lead_changes['season'] = df_lead_changes['game_id'].astype(str).str[:4].astype(int)

# Keep only 2021–2024
df_lead_changes = df_lead_changes[df_lead_changes['season'].between(2021, 2024)]


### Check csv file
df_lead_changes.to_csv('df_lead_changes_check.csv', index=False)
print("Lead_Changes_Data Cleaned")

#===================================================================================================
# Snap Counts Dataset - player level data

# copy import dataset
df_snap_counts = snap_counts_data_import.copy()

df_snap_counts['team'] = df_snap_counts['club_name'].map(club_name_map)

# Create a new player_name column (first_name + last_name)
df_snap_counts['player_name_flat'] = (
    df_snap_counts['first_name'].astype(str) + df_snap_counts['last_name'].astype(str)
)

# Remove special characters and spaces
df_snap_counts['player_name_flat'] = df_snap_counts['player_name_flat'].apply(
    lambda x: re.sub(r'[^A-Za-z0-9]', '', x)  # keep only letters and numbers
)

### Check csv file
df_snap_counts.to_csv('df_snap_count_check.csv', index=False)
print("Snap_Count_Data Cleaned")

#===================================================================================================

### WEEKLY DATA CLEANING + ADD MERGE COLUMNS

#
# Remove all rows for Postseason Games
df = weekly_data_import[weekly_data_import['season_type'] != 'POST'].copy()

#
# Create column IsAway to determine if player is playing away from home (0 = Home, 1 = Away)
df['isaway'] = (df['recent_team'] != df['home_team']).astype(int)

#
# Extract all unique values from stadium_id column
unique_stadium_ids = df['stadium_id'].dropna().unique().tolist()
#print(unique_stadium_ids)

#
# Define set of international stadium_ids
intl_stadiums = {'LON00', 'LON02', 'MEX00', 'FRA00', 'GER00', 'SAO00'}

#
# Create new column is_international: 1 if stadium is outside US, else 0
df['is_international'] = df['stadium_id'].isin(intl_stadiums).astype(int)

#
# Extract all unique values from status_description_abbr column
status_abbr = df['status_description_abbr'].dropna().unique().tolist()
#print(status_abbr)

#
# Extract all unique values from position group column
position_group = df['position_group'].dropna().unique().tolist()
#print(position_group)

#
# Sort the dataframe by season, then by player_id, then by week in ascending order
df = df.sort_values(by=["season", "player_id", "week"], ascending=[True, True, True]).reset_index(drop=True)
#print(df.head())

# Remove special characters and spaces
df['player_name_flat'] = df['player_display_name'].apply(
    lambda x: re.sub(r'[^A-Za-z0-9]', '', x)  # keep only letters and numbers
)

#
### Add additional classification column for games after multi-away stints
# Compute prior weeks and isaway flags within each player_id group
g = df.groupby(["season", "player_id"], group_keys=False)
lag1_week = g["week"].shift(1)
lag2_week = g["week"].shift(2)
lag1_isaway = g["isaway"].shift(1)
lag2_isaway = g["isaway"].shift(2)

#
# Require strictly sequential weeks and both prior isaway == 1
seq_ok = (df["week"].eq(lag1_week + 1)) & (lag1_week.eq(lag2_week + 1))
away_ok = (lag1_isaway.eq(1)) & (lag2_isaway.eq(1))

# Add extended away travel column to df
df["extended_away_games"] = (seq_ok & away_ok).astype(int)

#
# Compute prior weeks and isintl flags within each player_id group
g = df.groupby(["season", "player_id"], group_keys=False)
lag1_week = g["week"].shift(1)
lag1_isintl = g["is_international"].shift(1)

#
# Marks week directly after intl game 
seq_ok = df["week"].notna() & lag1_week.notna() & (df["week"] == lag1_week + 1)
intl_ok = (lag1_isintl == 1)

df["week_after_intl"] = (seq_ok & intl_ok).astype(int)

#
# Creates Binary for intl or week following
# Create a binary target column: 1 if either is_international or week_after_intl is 1
df["intl"] = ((df["is_international"] == 1) | (df["week_after_intl"] == 1)).astype(int)

#
# Create binary Thursday Game Identifier
df["is_thursday"] = df["weekday"].str.lower().eq("thursday").astype(int)

#
# Convert gametime (HH:MM) to datetime.time format
df["gametime"] = pd.to_datetime(df["gametime"], format="%H:%M", errors="coerce").dt.time

#
### Impute Wind/Temp values
#

#
# Extract all unique values from stadium_id column
unique_stadium_ids = df['stadium_id'].dropna().unique().tolist()
#print(unique_stadium_ids)

#
# Compute average temperature per stadium_id
avg_temp_by_stadium = (
    df.groupby("stadium_id", dropna=True)["temp"]
      .mean()
      .reset_index()
      .rename(columns={"temp": "avg_temp"})
      .sort_values("avg_temp", ascending=False)
)
#print(avg_temp_by_stadium)

#
# Build mapping from stadium_id to avg_temp (drop NaNs first)
avg_temp_map = (
    avg_temp_by_stadium.dropna(subset=["avg_temp"])
    .set_index("stadium_id")["avg_temp"]
    .round(1)
    .to_dict()
)

#
# Impute temp using map; fallback to 60 degrees due to indoor/dome stadium
df["temp"] = df.apply(
    lambda row: avg_temp_map.get(row["stadium_id"], 60) if pd.isna(row["temp"]) else row["temp"],
    axis=1
)

#
# Compute average wind per stadium_id
avg_wind_by_stadium = (
    df.groupby("stadium_id", dropna=True)["wind"]
      .mean()
      .reset_index()
      .rename(columns={"wind": "avg_wind"})
      .sort_values("avg_wind", ascending=False)
)

#
# Build mapping from stadium_id to avg_wind (drop NaNs first)
avg_wind_map = (
    avg_wind_by_stadium.dropna(subset=["avg_wind"])
    .set_index("stadium_id")["avg_wind"]
    .round(1)
    .to_dict()
)

#
# Impute wind using map; fallback to 0 wind speed due to indoor/dome stadium
df["wind"] = df.apply(
    lambda row: avg_temp_map.get(row["stadium_id"], 0) if pd.isna(row["wind"]) else row["wind"],
    axis=1
)

#
# Add in missing surface info, for intl and SF stadium

#
# Set surface to 'grass' for specific stadiums
grass_stadiums = ['SFO01', 'GER00', 'MEX00', 'LON00', 'FRA00', 'SAO00']

df.loc[df["stadium_id"].isin(grass_stadiums), "surface"] = "grass"

# Get the most common (mode) surface for each stadium_id, ignoring NaNs
most_common_surface = (
    df.dropna(subset=["surface"])
      .groupby("stadium_id")["surface"]
      .agg(lambda x: x.mode().iloc[0])
      .reset_index()
      .rename(columns={"surface": "most_common_surface"})
)

# print(most_common_surface)

# Create mapping from stadium_id to most common surface
surface_map = most_common_surface.set_index("stadium_id")["most_common_surface"].to_dict()

# Fill NaNs in surface column using the map
df["surface"] = df.apply(
    lambda row: surface_map.get(row["stadium_id"], row["surface"]) if pd.isna(row["surface"]) else row["surface"],
    axis=1
)

#
### Drop rows where game_id is NaN
df = df[df["game_id"].notna()].copy()

#
# Drop the nfl_detail_id column due to NaN values
df = df.drop(columns=["nfl_detail_id", "game_id"])

# Check which columns still have NaN values
nan_summary = df.isna().sum()
nan_summary = nan_summary[nan_summary > 0].sort_values(ascending=False)

#print("Columns with NaN values:")
#print(nan_summary)

### Check csv file
df.to_csv('df_check.csv', index=False)

print("Done cleaning and prepping df")

############################
#### Merge Datasets #######
############################

# Select columns from snap count data
snap_cols = [
    "game_id", "team", "player_name_flat",
    "offensive_snaps", "team_offensive_snaps",
    "defensive_snaps", "team_defensive_snaps",
    "special_team_snaps", "team_special_team_snaps"
]

df_snap_counts_trimmed = df_snap_counts[snap_cols].copy()

# Drop all rows where defensive_snaps > 0
df_snap_counts_trimmed = df_snap_counts_trimmed[df_snap_counts_trimmed['defensive_snaps'] == 0].copy()

# # --- Check for duplicates on merge keys ---
# dup_keys = df_snap_counts_trimmed.duplicated(subset=["game_id", "team", "player_name_flat"], keep=False)
# dups = df_snap_counts_trimmed[dup_keys]

# if not dups.empty:
#     print(" Found duplicates in snap count data on keys [game_id, team, player_name_flat]:")
#     print(dups.sort_values(["game_id", "team", "player_name_flat"]).head(20))
# else:
#     print("No duplicates found on merge keys.")

# dups.to_csv('dups_check.csv', index=False)

# --- Merge into df ---
df = df.merge(
    df_snap_counts_trimmed,
    left_on=["old_game_id", "team", "player_name_flat"],
    right_on=["game_id", "team", "player_name_flat"],
    how="left"
)

print("Snap counts merged")

df.to_csv('merge_check.csv', index=False)


############################
#### Create RF dataset #####
############################

# Columns to drop
df_cols_to_drop = [
    "player_id", "player_display_name", "recent_team", "season_type",
    "opponent_team", "depth_chart_position", "jersey_number", "football_name",
    "status", "status_description_abbr", "game_type", "player_name", "position",
    "game_id", "gameday", "gametime", "weekday", "location", "stadium", "stadium_id", 
    "old_game_id", "gsis", "away_rest", "home_rest", "roof", "surface", "temp", "wind",
    "is_international", "week_after_intl"
]

# Create new DataFrame with specified columns dropped
rf_data_full = df.drop(columns=df_cols_to_drop)

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


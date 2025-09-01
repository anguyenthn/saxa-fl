#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 16:24:09 2025

@author: an
"""

import os
os.chdir('/Users/an/Downloads/Projects/NFL/Data')

import pandas as pd

# Load the weekly player-level dataset
# This file has one row per player per game, so we need to roll it up to the team level
df = pd.read_csv("weekly_data_public.csv")

# Make sure there is a "team" column for grouping
# If not present, use "recent_team" which indicates the player's current team
if "team" not in df.columns and "recent_team" in df.columns:
    df["team"] = df["recent_team"]

# Create a flag for away games
# 1 if this team is not the home team, 0 if they are at home
if "home_team" in df.columns and "team" in df.columns:
    df["isaway"] = (df["team"] != df["home_team"]).astype(int)
else:
    df["isaway"] = 0

# Create a flag for Thursday games
# Useful to study short-week effects on team performance
if "weekday" in df.columns:
    df["is_thursday"] = df["weekday"].astype(str).str.lower().eq("thursday").astype(int)
else:
    df["is_thursday"] = 0

# Create a flag for international games (London, Mexico, Germany, etc.)
# These may involve extra travel and rest differences
intl_stadiums = {"LON00", "LON02", "MEX00", "FRA00", "GER00", "SAO00"}
if "stadium_id" in df.columns:
    df["is_international"] = df["stadium_id"].isin(intl_stadiums).astype(int)
else:
    df["is_international"] = 0

# Identify if this week comes right after an international game for the same team
# This helps measure if the fatigue/travel effects spill into the following week
tw_flags = (
    df.groupby(["season", "week", "team"], as_index=False)["is_international"]
      .max()
      .rename(columns={"is_international": "is_international_tw"})
    if {"season","week","team"}.issubset(df.columns) else
    pd.DataFrame(columns=["season","week","team","is_international_tw"])
)

if not tw_flags.empty:
    tw_flags = tw_flags.sort_values(["team", "season", "week"])
    tw_flags["week_after_international"] = (
        tw_flags.groupby("team")["is_international_tw"].shift(1).fillna(0).astype(int)
    )
    df = df.merge(tw_flags, on=["season","week","team"], how="left")
    df["week_after_international"] = df["week_after_international"].fillna(0).astype(int)
else:
    df["is_international_tw"] = 0
    df["week_after_international"] = 0

# Create a combined international flag
# 1 if the game itself was international OR the week after one
df["intl"] = ((df["is_international"] == 1) | (df["week_after_international"] == 1)).astype(int)

# Split certain stats into offense vs defense to avoid mixing them
# Example: interceptions can be thrown by QBs (offense) or made by DB/LB (defense)

if "interceptions" in df.columns and "position_group" in df.columns:
    df["ints_thrown"] = df.apply(
        lambda r: r["interceptions"] if r["position_group"] == "QB" else 0, axis=1
    )
    df["ints_def"] = df.apply(
        lambda r: r["interceptions"] if r["position_group"] in {"DB","LB","DL"} else 0, axis=1
    )
else:
    df["ints_thrown"] = 0
    df["ints_def"] = 0

# Do the same for sacks: taken by QBs vs made by defenders
if "sacks" in df.columns and "position_group" in df.columns:
    df["sacks_taken"] = df.apply(
        lambda r: r["sacks"] if r["position_group"] == "QB" else 0, axis=1
    )
    df["sacks_made"] = df.apply(
        lambda r: r["sacks"] if r["position_group"] in {"DL","LB"} else 0, axis=1
    )
else:
    df["sacks_taken"] = 0
    df["sacks_made"] = 0

# Sack yards are only meaningful for QBs (yards lost when sacked)
if "sack_yards" in df.columns and "position_group" in df.columns:
    df["sack_yards_taken"] = df.apply(
        lambda r: r["sack_yards"] if r["position_group"] == "QB" else 0, axis=1
    )
else:
    df["sack_yards_taken"] = 0

# Fumbles lost on sacks: only for QBs
if "sack_fumbles_lost" in df.columns and "position_group" in df.columns:
    df["qb_sack_fumbles_lost"] = df.apply(
        lambda r: r["sack_fumbles_lost"] if r["position_group"] == "QB" else 0, axis=1
    )
else:
    df["qb_sack_fumbles_lost"] = 0

# Now we build rules for aggregation.
# The idea is: sum up player stats for each team-week, 
# average efficiency metrics, and carry over one copy of constant game info.

group_keys = ["season", "week", "team"]

sum_cols = [
    "attempts","completions","passing_yards","passing_tds",
    "carries","rushing_yards","rushing_tds","rushing_first_downs",
    "rushing_fumbles","rushing_fumbles_lost",
    "receptions","targets","receiving_yards","receiving_tds","receiving_first_downs",
    "receiving_fumbles","receiving_fumbles_lost",
    "passing_air_yards","passing_yards_after_catch",
    "receiving_air_yards","receiving_yards_after_catch",
    "sacks","sack_yards","sack_fumbles","sack_fumbles_lost",
    "interceptions",
    "ints_thrown","ints_def","sacks_taken","sacks_made","sack_yards_taken","qb_sack_fumbles_lost",
]

mean_cols = ["racr","wopr","target_share","air_yards_share"]

first_cols = [
    "home_team","away_team","home_score","away_score","result",
    "stadium_id","surface","roof","temp","wind","away_rest","home_rest","div_game","weekday"
]

binary_max_cols = ["isaway","is_thursday","is_international","week_after_international","intl"]

# Only keep the columns that actually exist in the dataset
sum_cols = [c for c in sum_cols if c in df.columns]
mean_cols = [c for c in mean_cols if c in df.columns]
first_cols = [c for c in first_cols if c in df.columns]
binary_max_cols = [c for c in binary_max_cols if c in df.columns]

# Build the aggregation dictionary
agg_dict = {**{c: "sum" for c in sum_cols},
            **{c: "mean" for c in mean_cols},
            **{c: "first" for c in first_cols},
            **{c: "max" for c in binary_max_cols}}

# Create one row per team per week
team_week = df.groupby(group_keys, as_index=False).agg(agg_dict)

# After aggregation, compute performance outcomes at the team level
# Points scored: home_score if team was home, else away_score
def _points_scored(row):
    if "home_team" in team_week.columns and "home_score" in team_week.columns:
        return row["home_score"] if row["team"] == row["home_team"] else row.get("away_score", pd.NA)
    return pd.NA

# Points allowed: opponent’s score
def _points_allowed(row):
    if {"home_team","home_score","away_score"}.issubset(team_week.columns):
        return row["away_score"] if row["team"] == row["home_team"] else row["home_score"]
    return pd.NA

team_week["points_scored"] = team_week.apply(_points_scored, axis=1)
team_week["points_allowed"] = team_week.apply(_points_allowed, axis=1)

# Point differential (performance measure) and win flag
team_week["point_diff"] = team_week["points_scored"] - team_week["points_allowed"]
team_week["win"] = (team_week["point_diff"] > 0).astype(int)

# Rest days for this team in this game
def _team_rest(row):
    if "home_rest" in team_week.columns and "away_rest" in team_week.columns:
        return row["home_rest"] if row["team"] == row["home_team"] else row["away_rest"]
    return pd.NA

team_week["team_rest_days"] = team_week.apply(_team_rest, axis=1)

# Save the final aggregated dataset
# team_week.to_csv("team_week_aggregate.csv", index=False)

print(team_week.head())

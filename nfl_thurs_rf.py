#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 22:44:26 2025

@author: an
"""

import os
os.chdir('/Users/an/Downloads/Projects/NFL')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1) Load
df = pd.read_csv("NFLpbp_Schedule_Merged.csv")

# keep only rows where the offense is defined (team with ball)
df = df[df["posteam"].notna()].copy()


# 2) Team–game aggregation
group_cols = ["game_id","posteam"]
agg_dict = {
    "epa": "mean",
    "wpa": "mean",
    "yards_gained": "sum",
    "success": "mean",
    "pass_attempt": "sum",
    "rush_attempt": "sum",
    "sack": "sum",
    "touchdown": "sum",
    "third_down_converted": "sum",
    "third_down_failed": "sum",
    "air_yards": "mean",
    "yards_after_catch": "mean",
    "score_differential": "mean",
}
agg_dict = {k:v for k,v in agg_dict.items() if k in df.columns}

game_team = df.groupby(group_cols, as_index=False).agg(agg_dict)

# carry one-per-team-game context (including is_thursday) from original frame
context = [
    "is_thursday","is_away_game","is_international","week_y",
    "total_line_y","temp_y","wind_y","away_rest","home_rest","spread_line_y"
]
have_ctx = [c for c in context if c in df.columns]
ctx = df[group_cols + have_ctx].drop_duplicates(subset=group_cols, keep="first")
game_team = game_team.merge(ctx, on=group_cols, how="left")

# derive posteam_rest & posteam_expected_margin if inputs are present
if {"away_rest","home_rest","is_away_game"}.issubset(game_team.columns):
    game_team["posteam_rest"] = np.where(
        game_team["is_away_game"]==1, game_team["away_rest"], game_team["home_rest"]
    )
if {"spread_line_y","is_away_game"}.issubset(game_team.columns):
    # spread_line_y is home spread; flip sign if posteam is away
    game_team["posteam_expected_margin"] = np.where(
        game_team["is_away_game"]==1, -game_team["spread_line_y"], game_team["spread_line_y"]
    )

# rates/mix
plays = 0
if "pass_attempt" in game_team.columns: plays += game_team["pass_attempt"]
if "rush_attempt" in game_team.columns: plays += game_team["rush_attempt"]
game_team["plays"] = plays

if {"third_down_converted","third_down_failed"}.issubset(game_team.columns):
    denom = (game_team["third_down_converted"]+game_team["third_down_failed"]).replace(0,np.nan)
    game_team["third_down_rate"] = game_team["third_down_converted"]/denom

if {"pass_attempt","plays"}.issubset(game_team.columns):
    game_team["pass_rate"] = np.where(game_team["plays"]>0,
                                      game_team["pass_attempt"]/game_team["plays"], np.nan)

# 3) Target & features
target_col = "epa"  # or we can use wpa tbh
candidate_features = [
    "is_thursday","is_away_game","is_international","week_y",
    "posteam_expected_margin","posteam_rest","total_line_y","temp_y","wind_y",
    "yards_gained","success","pass_rate","third_down_rate","sack","touchdown",
    "air_yards","yards_after_catch"
]
features = [c for c in candidate_features if c in game_team.columns]

model_df = game_team[["game_id","posteam",target_col]+features].dropna()
X = model_df[features]
y = model_df[target_col]

# 4) Train + tune n_estimators (and a couple safe knobs)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=138)

rf = RandomForestRegressor(random_state=138)
param_grid = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}
grid = GridSearchCV(
    rf, param_grid=param_grid, cv=5, n_jobs=-1,
    scoring="neg_mean_absolute_error", verbose=0
)
grid.fit(X_tr, y_tr)
best_rf = grid.best_estimator_
print("Best params:", grid.best_params_)

# 5) Evaluate
pred = best_rf.predict(X_te)
print("R^2:", r2_score(y_te, pred))
print("MAE:", mean_absolute_error(y_te, pred))

# 6) Feature importances + quick Thursday effect delta
fi = (pd.DataFrame({"feature":features,"importance":best_rf.feature_importances_})
      .sort_values("importance", ascending=False))
print("\nTop importances:\n", fi.head(15))

if "is_thursday" in X_te.columns:
    X1, X0 = X_te.copy(), X_te.copy()
    X1["is_thursday"] = 1
    X0["is_thursday"] = 0
    delta = best_rf.predict(X1).mean() - best_rf.predict(X0).mean()
    print(f"\nAvg predicted {target_col} difference (Thu=1 vs 0): {delta:.5f} per play")

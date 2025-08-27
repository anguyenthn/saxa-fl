#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 20:46:09 2025

@author: an
"""

# import libs

import os
os.chdir('/Users/an/Downloads/Projects/NFL/Data')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# load data
df = pd.read_csv('weekly_data_2002-2025.csv')

# create game-level table (one row per game)
# we aggregate totals across both teams to represent the overall game environment/performance
agg_spec = {
    # environment / schedule (constant within game)
    "roof": "first",
    "surface": "first",
    "temp": "first",
    "wind": "first",
    "weekday": "first",
    "div_game": "first",
    "away_rest": "first",
    "home_rest": "first",
    # scoring (needed for target)
    "home_score": "first",
    "away_score": "first",
    # offense
    "completions": "sum",
    "passing_yards": "sum",
    "passing_tds": "sum",
    "interceptions": "sum",
    "carries": "sum",
    "rushing_yards": "sum",
    "rushing_tds": "sum",
    "receptions": "sum",
    "targets": "sum",
    "receiving_yards": "sum",
    "receiving_tds": "sum",
    # defense
    "sacks": "sum",
    "sack_yards": "sum",
    "sack_fumbles": "sum",
    # advanced receiving (use mean at game level)
    "target_share": "mean",
    "wopr": "mean",
}

# keep only columns that actually exist to avoid key errors
agg_spec = {k: v for k, v in agg_spec.items() if k in df.columns}

df_game = df.groupby("game_id").agg(agg_spec).reset_index()

# target: absolute point differential (lower = closer game)
df_game["point_diff"] = (df_game["home_score"] - df_game["away_score"]).abs()

# optional helper flags for analysis (not used as features by default)
# thursday flag
if "weekday" in df_game.columns:
    df_game["is_thursday"] = (df_game["weekday"].astype(str).str.lower() == "thursday").astype(int)
# home win / away win
df_game["home_win"] = (df_game["home_score"] > df_game["away_score"]).astype(int)

# choose features
cat_cols = [c for c in ["roof", "surface", "weekday"] if c in df_game.columns]
num_cols = [
    "temp","wind","div_game","away_rest","home_rest",
    "completions","passing_yards","passing_tds","interceptions",
    "carries","rushing_yards","rushing_tds",
    "receptions","targets","receiving_yards","receiving_tds",
    "sacks","sack_yards","sack_fumbles",
    "target_share","wopr",
]
num_cols = [c for c in num_cols if c in df_game.columns]

# one-hot encode categorical variables
df_model = pd.get_dummies(df_game, columns=cat_cols, drop_first=True)

# define x and y
drop_cols = ["game_id","home_score","away_score","point_diff","home_win","is_thursday"]
drop_cols = [c for c in drop_cols if c in df_model.columns]
feature_cols = [c for c in df_model.columns if c not in drop_cols]

X = df_model[feature_cols]
y = df_model["point_diff"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# xgboost regressor
model = xgb.XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

# train
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
print("mse:", mean_squared_error(y_test, y_pred))
print("r2:", r2_score(y_test, y_pred))

# simple thursday vs other-days comparison on actuals and predictions
if "is_thursday" in df_game.columns:
    eval_df = pd.DataFrame({
        "game_id": df_game["game_id"],
        "is_thursday": df_game["is_thursday"],
        "point_diff": y
    })
    eval_df["yhat"] = model.predict(X)

    print("\nactual mean point_diff by thursday flag (0=other days, 1=thursday):")
    print(eval_df.groupby("is_thursday")["point_diff"].mean())

    print("\npredicted mean point_diff by thursday flag (0=other days, 1=thursday):")
    print(eval_df.groupby("is_thursday")["yhat"].mean())

# simple home vs away perspective
# since this is one-row-per-game, we compare outcomes when home wins vs away wins to see if home field correlates with blowouts
print("\nactual mean point_diff by home_win (1=home won, 0=away won):")
print(df_game.groupby("home_win")["point_diff"].mean())

# show top features by gain importance
booster = model.get_booster()
score = booster.get_score(importance_type="gain")
imp = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:20]
print("\ntop 20 features by gain importance:")
for k, v in imp:
    print(f"{k}: {v:.4f}")

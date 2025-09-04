#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 20:10:35 2025

@author: an
"""

import os
os.chdir('/Users/an/Downloads/Projects/NFL/Data')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
team_week = pd.read_csv('team_week_aggregate.csv')

# Choose target and features
target = "point_diff" # we can change this 
features = [
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds",
    "sacks_made", "interceptions", "ints_thrown",
    "isaway", "is_thursday", "intl"
]

X = team_week[features].fillna(0)
y = team_week[target].fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=38
)

# Random Forest model
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    random_state=38,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluate
print("Random Forest Results:")
print(f"R² score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")

# Feature importance
importances = pd.DataFrame({
    "feature": features,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importances:")
print(importances)





"""
Random Forest Results:
    R² score: 0.438
    RMSE: 10.88

Feature Importances:
    rushing_yards      – 0.295
    rushing_tds        – 0.122
    receptions         – 0.106
    passing_yards      – 0.091
    receiving_yards    – 0.090
    passing_tds        – 0.083
    receiving_tds      – 0.080
    ints_thrown        – 0.057
    interceptions      – 0.039
    isaway             – 0.024
    is_thursday        – 0.009
    intl               – 0.004
    sacks_made         – 0.000
"""







"""
What this means:

Model performance

The Random Forest explains about 44% of the
 variance in point differential (R² = 0.438).

An RMSE of ~10.9 means the model’s predictions
 of score differential are typically off by about
 11 points. That’s decent given NFL game volatility,
 but there’s room to improve with more features.

Feature importance

Rushing production dominates: rushing yards (0.30)
 is by far the strongest predictor of point differential,
 followed by rushing TDs (0.12). Teams that control the ground
 game tend to create bigger scoring margins.

Passing metrics matter, but less: passing yards (0.09)
 and passing TDs (0.08) are important but secondary.
 Interestingly, receptions and receiving yards (~0.10 each) 
 rival or exceed passing TD importance, highlighting the role of
 sustained drives over explosive plays.

Turnovers are relevant: interceptions and interceptions thrown
 contribute (~0.04–0.06), aligning with football intuition
 — turnovers swing games, but in aggregate, raw yardage dominates.

Situational flags (isaway, is_thursday, intl) have small but 
non-zero importance, with away status mattering the most (0.02).
 This is consistent with your t-test findings of home-field advantage.

Pass rush (sacks_made) registered near zero, meaning sack counts
 in aggregate don’t predict point differential as strongly as
 rushing or turnovers.
"""

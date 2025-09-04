#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 20:07:04 2025

@author: an
"""

import os
os.chdir('/Users/an/Downloads/Projects/NFL/Data')

import pandas as pd
from scipy.stats import ttest_ind

# Load the cleaned team-weekly data
team_week = pd.read_csv('team_week_aggregate.csv')


# Function to run a t-test comparing ANY metric under two conditions
def compare_metric(flag_col, label, metric):
    group1 = team_week.loc[team_week[flag_col] == 1, metric].dropna()
    group0 = team_week.loc[team_week[flag_col] == 0, metric].dropna()
    
    t_stat, p_val = ttest_ind(group1, group0, equal_var=False)  # Welch’s t-test
    print(f"{label} – {metric}")
    print(f"Mean {metric} (flag=1): {group1.mean():.2f}")
    print(f"Mean {metric} (flag=0): {group0.mean():.2f}")
    print(f"T-statistic: {t_stat:.3f}, P-value: {p_val:.4f}\n")


# Rerun with different metrics


# Away vs Home
compare_metric("isaway", "Away vs Home", "passing_yards")
compare_metric("isaway", "Away vs Home", "rushing_yards")

# Thursday vs Non-Thursday
compare_metric("is_thursday", "Thursday vs Non-Thursday", "passing_tds")
compare_metric("is_thursday", "Thursday vs Non-Thursday", "sacks")

# International/Week-After vs Normal
compare_metric("intl", "International/Week-After vs Normal", "receiving_yards")
compare_metric("intl", "International/Week-After vs Normal", "interceptions")



"""
Away vs Home – Passing Yards
Teams throw for fewer yards when playing on the road (233.5 vs 242.2).
 The difference is modest (~9 yards) but statistically significant (p < 0.01).
 This suggests defenses or environment may limit passing output away from home.

Away vs Home – Rushing Yards
Similarly, rushing production is slightly lower in away games (115.1 vs 119.5).
 The effect size (~4.5 yards) is small but statistically significant (p < 0.05).
 This reinforces a general home-field performance edge across both passing and rushing.

Thursday vs Non-Thursday – Passing TDs
Teams average slightly more passing TDs on Thursday (1.60 vs 1.45),
 but this difference is not statistically significant (p = 0.14).
 Short rest may not meaningfully affect passing scoring.

Thursday vs Non-Thursday – Sacks
Teams record fewer sacks on Thursday (2.24 vs 2.44),
 but again the difference is not statistically significant (p = 0.18).
 Pass rush performance appears unaffected by short-week games.

International/Week-After vs Normal – Receiving Yards
Receiving yards are virtually identical (238.2 vs 237.7). 
With an extremely high p-value (0.96), international games or their aftermath show no impact on receiving yardage.

International/Week-After vs Normal – Interceptions
Interception rates are almost the same (0.79 vs 0.77). 
The difference is negligible and not statistically significant (p = 0.83).
 Playing abroad or after travel does not increase turnovers through picks.
"""
 
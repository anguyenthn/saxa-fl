#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 16:30:36 2025

@author: an
"""

import os
os.chdir('/Users/an/Downloads/Projects/NFL/Data')

import pandas as pd
from scipy.stats import ttest_ind

# Load the cleaned team-weekly data
# This is public, pre-merged data
team_week = pd.read_csv('team_week_aggregate.csv')


# Function to run a t-test comparing point differential under two conditions
def compare_point_diff(flag_col, label):
    group1 = team_week.loc[team_week[flag_col] == 1, "point_diff"].dropna()
    group0 = team_week.loc[team_week[flag_col] == 0, "point_diff"].dropna()
    
    t_stat, p_val = ttest_ind(group1, group0, equal_var=False)  # Welch’s t-test
    print(f"{label}")
    print(f"Mean point_diff (flag=1): {group1.mean():.2f}")
    print(f"Mean point_diff (flag=0): {group0.mean():.2f}")
    print(f"T-statistic: {t_stat:.3f}, P-value: {p_val:.4f}\n")

# Compare away vs home
compare_point_diff("isaway", "Away vs Home")

# Compare Thursday games vs other days
compare_point_diff("is_thursday", "Thursday vs Non-Thursday")

# Compare international (or week-after-international) vs other games
compare_point_diff("intl", "International/Week-After vs Normal")

"""
The t-test here is a quick way to ask 
“is the average point differential different when teams are 
away/Thursday/international compared to when they are not?”
"""

""" 
Results:
    Away vs Home
    Mean point_diff (flag=1): -2.12
    Mean point_diff (flag=0): 2.18
    T-statistic: -7.237, P-value: 0.0000

    Thursday vs Non-Thursday
    Mean point_diff (flag=1): 0.00
    Mean point_diff (flag=0): 0.00
    T-statistic: 0.000, P-value: 1.0000

    International/Week-After vs Normal
    Mean point_diff (flag=1): -0.38
    Mean point_diff (flag=0): 0.01
    T-statistic: -0.262, P-value: 0.7943
"""

"""
What this means:
    What this means:
    
    Away vs Home:
      Teams perform significantly worse when playing away games, with an
      average point differential of -2.1 compared to +2.2 at home. This ~4-point
      swing is a clear and statistically significant home-field advantage.

    Thursday vs Non-Thursday:
      There is no measurable difference in point differential between Thursday
      games and games on other days. Short rest (Thursday games) does not
      appear to systematically impact team performance.

    International/Week-After vs Normal:
      Teams playing in international games or the week immediately after one
      perform about the same as in normal weeks. The difference (-0.4 vs 0.0)
      is very small and not statistically significant.
"""
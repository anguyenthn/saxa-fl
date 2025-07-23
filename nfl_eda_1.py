#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 16:54:59 2025

@author: an
"""

import os
os.chdir('/Users/an/Downloads')

import pandas as pd
import nfl_data_py as nfl
from datetime import datetime
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

# Load play-by-play and schedule data
pbp = nfl.import_pbp_data([2023])
schedule = nfl.import_schedules([2023])

# Merge to bring in game dates
pbp = pbp.merge(schedule[['game_id', 'gameday']], on='game_id', how='left')
pbp['gameday'] = pd.to_datetime(pbp['gameday'])
pbp['is_thu_game'] = (pbp['gameday'].dt.dayofweek == 3).astype(int)  # Thursday = 3

# Dictionary with variable types
performance_cols = {
    'yards_gained': 'continuous',
    'passing_yards': 'continuous',
    'rushing_yards': 'continuous',
    'receiving_yards': 'continuous',
    'solo_tackle': 'binary',
    'tackled_for_loss': 'binary'
}


# Loop through each metric with custom visualization
# This is because the first visualization I created didn't work well with this data

for col, col_type in performance_cols.items():
    print(f"\n===== {col.upper()} =====")
    
    temp_df = pbp[[col, 'is_thu_game']].dropna()
    if temp_df.empty:
        print("No data available.")
        continue

    # Grouped averages
    grouped = temp_df.groupby('is_thu_game')[col].mean()
    print("Average values:")
    print(grouped)

    # T-test
    thu = temp_df[temp_df['is_thu_game'] == 1][col]
    non_thu = temp_df[temp_df['is_thu_game'] == 0][col]
    t_stat, p_val = ttest_ind(thu, non_thu, equal_var=False)
    print(f"T-test: t = {t_stat:.2f}, p = {p_val:.4f}")

    # Custom visualizations
    if col_type == 'binary':
        # Bar plot for binary variable proportion
        proportion = temp_df.groupby('is_thu_game')[col].mean().reset_index()
        sns.barplot(x='is_thu_game', y=col, data=proportion)
        plt.title(f"{col}: Proportion by Game Type")
        plt.xticks([0, 1], ['Non-Thursday', 'Thursday'])
        plt.ylabel('Proportion')
        plt.xlabel('')
        plt.tight_layout()
        plt.show()

    elif col_type == 'continuous':
        # Violin plot for continuous variable
        sns.violinplot(x='is_thu_game', y=col, data=temp_df, inner='quartile', cut=0)
        plt.title(f"{col}: Thursday vs Non-Thursday")
        plt.xticks([0, 1], ['Non-Thursday', 'Thursday'])
        plt.ylabel(col)
        plt.xlabel('')
        plt.tight_layout()
        plt.show()

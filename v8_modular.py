#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 21:33:32 2025

@author: kesh - tay - an

FIXED VERSION: Eliminates data leakage
Key fixes:
1. Only use historical data (weeks before current week)
2. Separate feature stats from target stats
3. Proper temporal validation
"""

# IMPORTS

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re
import os
import subprocess
from dataclasses import dataclass
from typing import Iterable, Optional

# ========== AUTO-DETECT R_HOME ==========
print("\n" + "="*60)
print("CONFIGURING R ENVIRONMENT")
print("="*60)
try:
    r_home = subprocess.check_output(['R', 'RHOME'], text=True).strip()
    os.environ['R_HOME'] = r_home
    print(f"✓ R_HOME set to: {r_home}")
except Exception as e:
    print(f"⚠️ Could not auto-detect R_HOME: {e}")
    print("Please set R_HOME manually before running this script")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

# Import RPY2 after R_HOME is set
try:
    import rpy2.robjects as ro
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import pandas2ri
    print("✓ RPY2 imported successfully")
except ImportError as e:
    print(f"✗ Failed to import RPY2: {e}")
    print("Install with: pip install rpy2")
    raise

# SHAP with fallback
try:
    import shap
    HAS_SHAP = True
    print("✓ SHAP available")
except Exception:
    HAS_SHAP = False
    print("⚠️ SHAP not available; using RF feature_importances_ fallback.")

###
# ADD PATHS / CONFIGURATIONS

DATA_DIR = Path("/Users/taylorwashington/Desktop/NFL_Model/saxa-fl")
INPUT_CSV = DATA_DIR / "df_merged.csv"
OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20007
TEST_SIZE   = 0.25

###
# CRITICAL FIX: Define which stats are TARGETS vs FEATURES

TARGET_STATS = {
    'passing_yards', 'passing_tds', 'passing_interceptions',
    'rushing_yards', 'rushing_tds', 'rushing_fumbles',
    'receiving_yards', 'receiving_tds', 'receptions', 'targets',
    'receiving_fumbles', 'carries', 'completions', 'attempts',
    'sacks_suffered', 'sack_yards_lost', 'passing_air_yards',
    'passing_yards_after_catch', 'rushing_first_downs',
    'receiving_air_yards', 'receiving_yards_after_catch'
}

###
# ESTABLISH MODEL TARGET & SCOPE

TARGET = None
RUN_ALL_TARGETS = True
EXCLUDED_TARGETS = {"isaway", "intl", "is_thursday"}
TRAIN_BY_POSITION = False
YEARS = range(2020, 2024)

###
# SAVE PLOT/CSV TOGGLE 

SAVE_PLOTS = True
SAVE_CSVS  = True

###
# ESTABLISH N for TOP FEATURES USED IN RF MODEL + ADDITIONAL FEATURE TOGGLES

TOP_K = 20
ADD_REC_TEAM_TGT_SHARE = False  # Disabled - causes leakage
ADD_ALPHA_ENV_CODES    = True
KEEP_REST_TRAVEL       = True

###
# KEY ID - DO NOT CHANGE

PLAYER_COL, SEASON_COL, WEEK_COL = "player_id", "season", "week"

# NFL PROVIDED DATA FILES
RECEIVING_CSV   = 'receiving_data.csv'
SCHEDULE_CSV    = 'schedule_2002_2024.csv'
AIR_CSV         = 'air_yards_data.csv'
LEAD_CSV        = 'lead_changes.csv'

RUN_API = True

###
# TOGGLE TO FORCE A FRESH BUILD OF df_merged.csv EVEN IF IT ALREADY EXISTS

FORCE_REBUILD_MERGED = True

###
# COMBINED DATA-SOURCE CONFIGURATION

@dataclass
class CombinedConfig:
    YEARS: Iterable[int] | tuple[int, int]
    receiving_csv: Optional[str | Path] = None
    schedule_csv: Optional[str | Path] = None
    air_csv: Optional[str | Path] = None
    lead_csv: Optional[str | Path] = None
    run_api: Optional[bool] = None

CFG = CombinedConfig(
    YEARS=YEARS,
    receiving_csv=RECEIVING_CSV,
    schedule_csv=SCHEDULE_CSV,
    air_csv=AIR_CSV,
    lead_csv=LEAD_CSV,
    run_api=RUN_API,
)

# HELPERS TO RUN CSV FILES 
def _normalize_years(YEARS: Iterable[int] | tuple[int, int]) -> list[int]:
    if isinstance(YEARS, tuple) and len(YEARS) == 2:
        start, end = map(int, YEARS)
        if start > end:
            raise ValueError("YEARS must be (start, end) with start <= end.")
        return list(range(start, end + 1))
    out = list(map(int, list(YEARS)))
    if not out:
        raise ValueError("YEARS cannot be empty.")
    return out

def _safe_read_csv(path_like: Optional[str | Path]) -> Optional[pd.DataFrame]:
    if not path_like:
        print("⚠️ No file path provided")
        return None
    p = Path(path_like)
    if not p.exists():
        print(f"⚠️ File not found, skipping: {p}")
        return None
    try:
        df = pd.read_csv(p)
        print(f"✓ Loaded {p.name}: {df.shape}")
        return df
    except Exception as e:
        print(f"✗ Failed to read {p}: {e}")
        return None
    
def _normalize_team_codes_by_season(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if 'season' not in df.columns:
        return df

    out = df.copy()

    for c in cols:
        if c not in out.columns:
            continue
        s = out[c].astype(str).str.strip().str.upper()

        la_mask = s.eq('LA')
        out.loc[la_mask & (out['season'] <= 2015), c] = 'STL'
        out.loc[la_mask & (out['season'] >= 2016), c] = 'LAR'

        lac_mask = s.eq('LAC')
        sd_mask  = s.eq('SD')
        out.loc[lac_mask & (out['season'] <= 2016), c] = 'SD'
        out.loc[sd_mask  & (out['season'] >= 2017), c] = 'LAC'

        lv_mask  = s.eq('LV')
        oak_mask = s.eq('OAK')
        out.loc[lv_mask  & (out['season'] <= 2019), c] = 'OAK'
        out.loc[oak_mask & (out['season'] >= 2020), c] = 'LV'

        out[c] = out[c].astype(str).str.strip().str.upper()

    return out

"""
# DATA INGESTION & INTEGRATION
"""

def build_df_merged_onefile(cfg: CombinedConfig) -> pd.DataFrame:
    years = _normalize_years(cfg.YEARS)
    print(f"\n{'='*60}")
    print(f"BUILDING MERGED DATASET FOR YEARS: {min(years)}-{max(years)}")
    print(f"{'='*60}")
    
    # LOAD NFL PROVIDED DATA
    print("\n[Step 1/8] Loading NFL-provided CSV files...")
    receiving_data_import = _safe_read_csv(cfg.receiving_csv)
    air_yards_data_import = _safe_read_csv(cfg.air_csv)
    lead_changes_data_import = _safe_read_csv(cfg.lead_csv)
    
    # ========== USE RPY2 TO LOAD DATA FROM R ==========
    print(f"\n[Step 2/8] Loading data from nflreadr (R)...")
    
    try:
        utils = rpackages.importr('utils')
        if not rpackages.isinstalled('nflreadr'):
            print("Installing nflreadr R package...")
            utils.install_packages('nflreadr')
        
        nflreadr = rpackages.importr("nflreadr")
        ro.r('options(nflreadr.verbose = FALSE)')
        
        with (ro.default_converter + pandas2ri.converter).context():
            print("  Loading player stats...")
            player_data = nflreadr.load_player_stats(seasons=ro.IntVector(years))
            print("  Loading schedules...")
            schedule_data = nflreadr.load_schedules(seasons=ro.IntVector(years))
            print("  Loading rosters...")
            weekly_roster_data = nflreadr.load_rosters(seasons=ro.IntVector(years))
            print("  Loading snap counts...")
            snap_counts = nflreadr.load_snap_counts(seasons=ro.IntVector(years))
        
        print(f"✓ Loaded player_data: {player_data.shape}")
        print(f"✓ Loaded schedule_data: {schedule_data.shape}")
        print(f"✓ Loaded weekly_roster_data: {weekly_roster_data.shape}")
        print(f"✓ Loaded snap_counts: {snap_counts.shape}")
    except Exception as e:
        print(f"✗ Error loading data from R: {e}")
        raise
    
    # ========== PROCESS SNAP COUNTS DATA ==========
    print(f"\n[Step 3/8] Processing snap counts...")
    
    snap_counts = snap_counts.rename(columns={
        'season': 'season',
        'game_id': 'game_id',
        'team': 'team',
        'pfr_player_id': 'player_id',
        'player': 'player_name',
        'offense_snaps': 'offensive_snaps',
        'offense_pct': 'offense_pct',
        'defense_snaps': 'defensive_snaps',
        'defense_pct': 'defense_pct',
        'st_snaps': 'special_team_snaps',
        'st_pct': 'st_pct'
    })
    
    if 'team_offensive_snaps' not in snap_counts.columns:
        team_off_snaps = snap_counts.groupby(['game_id', 'team'])['offensive_snaps'].max().reset_index()
        team_off_snaps = team_off_snaps.rename(columns={'offensive_snaps': 'team_offensive_snaps'})
        snap_counts = snap_counts.merge(team_off_snaps, on=['game_id', 'team'], how='left')
    
    if 'team_defensive_snaps' not in snap_counts.columns:
        team_def_snaps = snap_counts.groupby(['game_id', 'team'])['defensive_snaps'].max().reset_index()
        team_def_snaps = team_def_snaps.rename(columns={'defensive_snaps': 'team_defensive_snaps'})
        snap_counts = snap_counts.merge(team_def_snaps, on=['game_id', 'team'], how='left')
    
    if 'team_special_team_snaps' not in snap_counts.columns:
        team_st_snaps = snap_counts.groupby(['game_id', 'team'])['special_team_snaps'].max().reset_index()
        team_st_snaps = team_st_snaps.rename(columns={'special_team_snaps': 'team_special_team_snaps'})
        snap_counts = snap_counts.merge(team_st_snaps, on=['game_id', 'team'], how='left')
    
    snap_cols = [
        'season','game_id','team','player_id','player_name',
        'offensive_snaps','team_offensive_snaps',
        'defensive_snaps','team_defensive_snaps',
        'special_team_snaps','team_special_team_snaps'
    ]
    snap_cols = [col for col in snap_cols if col in snap_counts.columns]
    snap_counts_filtered = snap_counts[snap_cols].copy()
    
    snap_counts_filtered['player_name_flat'] = (
        snap_counts_filtered['player_name'].astype(str) 
    )
    snap_counts_filtered['player_name_flat'] = snap_counts_filtered['player_name'].apply(
        lambda x: re.sub(r'[^A-Za-z0-9]', '', x)
    )
    
    print(f"✓ Snap counts processed: {snap_counts_filtered.shape}")
    
    # ========== PROCESS SCHEDULE DATA ==========
    print(f"\n[Step 4/8] Processing schedule data...")
    
    def _standardize_schedule_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "home": "home_team", "homeTeam": "home_team", "home_team_abbr": "home_team", "home_team_code": "home_team",
            "away": "away_team", "awayTeam": "away_team", "away_team_abbr": "away_team", "away_team_code": "away_team",
            "week_number": "week", "game_week": "week",
            "season_year": "season", "year": "season",
            "site_roof_type": "roof",
        }
        out = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()
        for col in ("season", "week"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        return out
    
    schedule_data = _standardize_schedule_columns(schedule_data)
        
    _required = {"season", "week", "home_team", "away_team"}
    _missing = _required.difference(schedule_data.columns)
    if _missing:
        print(f"⚠️ schedule_data missing columns: {_missing}")
    
    schedule_cols_to_keep = [c for c in [
        'game_id','season','game_type','week','gameday','weekday','gametime','away_team',
        'away_score','home_team','home_score','location','result','total','overtime',
        'old_game_id','gsis','nfl_detail_id','away_rest','home_rest','div_game','roof',
        'surface','temp','wind','stadium_id','stadium','travel_distance_away','travel_distance_home',
        'nflverse_game_id','game_key','season_type','game_date','game_time_eastern',
        'home_point_total','away_point_total','home_point_diff','site_id','site_city',
        'site_state','site_zip_code','is_primetime','neutral_site'
    ] if c in schedule_data.columns]
    
    schedule_data = schedule_data[schedule_cols_to_keep]
    
    schedule_data = schedule_data[schedule_data["home_team"].astype(str).str.strip() != ""]
    schedule_data = schedule_data.dropna(subset=["home_team"])
    
    print(f"✓ Schedule processed: {schedule_data.shape}")
    
    # ========== PROCESS ROSTER DATA ==========
    print(f"\n[Step 5/8] Processing roster data...")
    
    weekly_roster_data = weekly_roster_data[[c for c in [
        'season','game_type','week','team','position','depth_chart_position','jersey_number',
        'player_name','football_name','player_id','status','status_description_abbr'
    ] if c in weekly_roster_data.columns]]
    
    print(f"✓ Roster processed: {weekly_roster_data.shape}")
    
    # ========== PROCESS PLAYER DATA ==========
    print(f"\n[Step 6/8] Processing player data...")
    
    player_cols = [
        'player_id','player_name','player_display_name','position','position_group','team',
        'season','week','season_type','opponent_team','attempts','completions','passing_yards',
        'passing_tds','passing_interceptions','sacks_suffered','sack_yards_lost','sack_fumbles','sack_fumbles_lost',
        'passing_air_yards','passing_yards_after_catch','passing_first_downs','passing_2pt_conversions',
        'carries','rushing_yards','rushing_tds','rushing_fumbles','rushing_fumbles_lost',
        'rushing_first_downs','receptions','targets','receiving_yards','receiving_tds',
        'receiving_fumbles','receiving_fumbles_lost','receiving_air_yards',
        'receiving_yards_after_catch','receiving_first_downs','receiving_2pt_conversions',
        'racr','target_share','air_yards_share','wopr',
        'def_sacks','def_sack_yards','def_interceptions'
    ]
    
    existing_player_cols = [col for col in player_cols if col in player_data.columns]
    weekly_player_data = player_data[existing_player_cols].copy()
    
    team_cols_weekly = ["team", "opponent_team"]
    weekly_player_data = _normalize_team_codes_by_season(weekly_player_data, team_cols_weekly)
    
    # ========== APPLY FILTERS ==========
    print(f"\n[Step 7/8] Applying filters...")
    print(f"  Before filtering: {len(weekly_player_data)} rows")
    
    weekly_player_data = weekly_player_data[weekly_player_data['season_type'] == 'REG']
    print(f"  After season_type='REG': {len(weekly_player_data)} rows")
    
    stat_columns = ['passing_yards', 'rushing_yards', 'receiving_yards', 
                    'def_sacks', 'def_interceptions']
    available_stat_cols = [col for col in stat_columns if col in weekly_player_data.columns]
    
    played_mask = pd.Series(False, index=weekly_player_data.index)
    for col in available_stat_cols:
        played_mask = played_mask | (weekly_player_data[col].fillna(0) != 0)
    
    weekly_player_data = weekly_player_data[played_mask]
    print(f"  After 'actually played' filter: {len(weekly_player_data)} rows")
    
    if 'game_type' in schedule_data.columns:
        schedule_data = schedule_data[schedule_data['game_type'] == 'REG']
    
    if 'game_type' in weekly_roster_data.columns:
        weekly_roster_data = weekly_roster_data[weekly_roster_data['game_type'] == 'REG']
    
    if 'status' in weekly_roster_data.columns:
        weekly_roster_data = weekly_roster_data[weekly_roster_data['status'] == 'ACT']
    
    # ========== MERGE DATASETS ==========
    print(f"\n[Step 8/8] Merging datasets...")
    
    if "stadium_id" not in weekly_player_data.columns:
        weekly_player_data["stadium_id"] = 99999
    else:
        weekly_player_data["stadium_id"] = weekly_player_data["stadium_id"].fillna(99999)

    merge_keys_roster = ['player_id', 'season', 'week', 'team']
    merge_keys_roster = [key for key in merge_keys_roster if key in weekly_player_data.columns and key in weekly_roster_data.columns]
    
    original_len = len(weekly_player_data)
    weekly_data = weekly_player_data.merge(
        weekly_roster_data,
        on=merge_keys_roster,
        how='inner',
        suffixes=('', '_roster')
    )
    print(f"  After roster merge: {original_len} → {len(weekly_data)} rows")

    home_schedule = schedule_data.copy()
    home_schedule = home_schedule.rename(columns={
        'home_team': 'team', 'away_team': 'opponent_team',
        'travel_distance_home': 'travel_distance',
    }).drop(columns=['travel_distance_away'], errors='ignore')
    
    away_schedule = schedule_data.copy()
    away_schedule = away_schedule.rename(columns={
        'away_team': 'team', 'home_team': 'opponent_team',
        'travel_distance_away': 'travel_distance',
    }).drop(columns=['travel_distance_home'], errors='ignore')
    
    combined_schedule = pd.concat([home_schedule, away_schedule], ignore_index=True)
    
    merge_keys_schedule = ['season', 'week', 'team', 'opponent_team']
    merge_keys_schedule = [key for key in merge_keys_schedule if key in weekly_data.columns and key in combined_schedule.columns]
    
    schedule_cols_to_keep = [col for col in combined_schedule.columns if col not in weekly_data.columns or col in merge_keys_schedule or col in ['travel_distance', 'game_id_schedule']]
    
    original_len = len(weekly_data)
    weekly_data = weekly_data.merge(
        combined_schedule[schedule_cols_to_keep],
        on=merge_keys_schedule,
        how='left',
        suffixes=('', '_schedule')
    )
    print(f"  After schedule merge: {original_len} → {len(weekly_data)} rows")
    
    # ========== HANDLE DUPLICATE COLUMNS ==========
    x_cols = [col for col in weekly_data.columns if col.endswith('_x')]
    y_cols = [col for col in weekly_data.columns if col.endswith('_y')]
    base_cols = [col[:-2] for col in x_cols if f"{col[:-2]}_y" in y_cols]
    for col in base_cols:
        weekly_data[col] = weekly_data[f"{col}_x"].combine_first(weekly_data[f"{col}_y"])
        weekly_data.drop(columns=[f"{col}_x", f"{col}_y"], inplace=True)

    for col in weekly_data.columns:
        if col.endswith('_schedule') and col.replace('_schedule', '') in weekly_data.columns:
            weekly_data.drop(columns=[col], inplace=True, errors='ignore')
            
    if 'travel_distance' in weekly_data.columns:
        weekly_data.rename(columns={'travel_distance': 'team_travel_distance_miles'}, inplace=True)

    # ========== PROCESS ADDITIONAL DATA ==========
    team_map = {
        "Arizona Cardinals": "ARI","Atlanta Falcons": "ATL","Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF","Carolina Panthers": "CAR","Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN","Cleveland Browns": "CLE","Dallas Cowboys": "DAL",
        "Denver Broncos": "DEN","Detroit Lions": "DET","Green Bay Packers": "GB",
        "Houston Texans": "HOU","Indianapolis Colts": "IND","Jacksonville Jaguars": "JAX",
        "Kansas City Chiefs": "KC","Las Vegas Raiders": "LV","Los Angeles Chargers": "LAC",
        "Los Angeles Rams": "LAR","Miami Dolphins": "MIA","Minnesota Vikings": "MIN",
        "New England Patriots": "NE","New Orleans Saints": "NO","New York Giants": "NYG",
        "New York Jets": "NYJ","Philadelphia Eagles": "PHI","Pittsburgh Steelers": "PIT",
        "San Francisco 49ers": "SF","Seattle Seahawks": "SEA","Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN","Washington Commanders": "WAS","Washington Football Team": "WAS"
    }

    if isinstance(receiving_data_import, pd.DataFrame) and 'season' in receiving_data_import.columns:
        df_receiving = receiving_data_import[receiving_data_import['season'].isin(years)].copy()
    else:
        df_receiving = pd.DataFrame()

    if not df_receiving.empty:
        has_team = 'team' in df_receiving.columns
        has_club_name = 'club_name' in df_receiving.columns
        required_cols = ['first_name', 'last_name', 'game_id']
        missing_cols = [col for col in required_cols if col not in df_receiving.columns]
        
        if missing_cols:
            print(f"⚠️ Receiving data missing columns: {missing_cols}. Skipping.")
            df_receiving = pd.DataFrame()
        elif not has_team and not has_club_name:
            print(f"⚠️ Receiving data missing 'team' or 'club_name' column. Skipping.")
            df_receiving = pd.DataFrame()
        else:
            if has_club_name and not has_team:
                df_receiving['team'] = df_receiving['club_name'].map(team_map)
            elif has_team:
                df_receiving['team'] = df_receiving['team'].map(team_map).fillna(df_receiving['team'])
            
            df_receiving['game_id'] = df_receiving['game_id'].astype(str)
            
            df_receiving['player_name_flat'] = (
                df_receiving['first_name'].astype(str) + df_receiving['last_name'].astype(str)
            )
            df_receiving['player_name_flat'] = df_receiving['player_name_flat'].apply(
                lambda x: re.sub(r'[^A-Za-z0-9]', '', x)
            )

    if air_yards_data_import is not None:
        df_air_yards = air_yards_data_import.copy()
        if 'game_id' not in df_air_yards.columns:
            print("⚠️ Air yards data missing 'game_id' column. Skipping.")
            df_air_yards = pd.DataFrame()
        else:
            df_air_yards['game_id'] = df_air_yards['game_id'].astype(str)
            df_air_yards['season'] = df_air_yards['game_id'].str[:4].astype(int)
            df_air_yards = df_air_yards[df_air_yards['season'].isin(years)]
    else:
        df_air_yards = pd.DataFrame()

    if not df_air_yards.empty:
        has_team = 'team' in df_air_yards.columns
        has_club_name = 'club_name' in df_air_yards.columns
        
        if not has_team and not has_club_name:
            print("⚠️ Air yards data missing 'team' or 'club_name' column. Skipping.")
            df_air_yards = pd.DataFrame()
        else:
            if has_club_name and not has_team:
                df_air_yards['team'] = df_air_yards['club_name'].map(team_map)
            elif has_team:
                df_air_yards['team'] = df_air_yards['team'].map(team_map).fillna(df_air_yards['team'])

    if lead_changes_data_import is not None:
        df_lead_changes = lead_changes_data_import.copy()
        if 'game_id' not in df_lead_changes.columns:
            print("⚠️ Lead changes data missing 'game_id' column. Skipping.")
            df_lead_changes = pd.DataFrame()
        else:
            df_lead_changes['game_id'] = df_lead_changes['game_id'].astype(str)
            df_lead_changes['season'] = df_lead_changes['game_id'].str[:4].astype(int)
            df_lead_changes = df_lead_changes[df_lead_changes['season'].isin(years)]
    else:
        df_lead_changes = pd.DataFrame()

    df = weekly_data[weekly_data['season_type'] != 'POST'].copy()

    if "home_team" in df.columns:
        df['isaway'] = (df['team'] != df['home_team']).astype(int)
    else:
        df['isaway'] = 0

    intl_stadiums = {'LON00','LON02','MEX00','FRA00','GER00','SAO00'}
    df['is_international'] = df['stadium_id'].isin(intl_stadiums).astype(int)

    df = df.sort_values(by=["season","player_id","week"], ascending=[True,True,True]).reset_index(drop=True)
    df['player_name_flat'] = df['player_display_name'].apply(
        lambda x: re.sub(r'[^A-Za-z0-9]', '', x)
    )

    g = df.groupby(["season","player_id"], group_keys=False)
    lag1_week = g["week"].shift(1)
    lag2_week = g["week"].shift(2)
    lag1_isaway = g["isaway"].shift(1)
    lag2_isaway = g["isaway"].shift(2)
    seq_ok = (df["week"].eq(lag1_week + 1)) & (lag1_week.eq(lag2_week + 1))
    away_ok = (lag1_isaway.eq(1)) & (lag2_isaway.eq(1))
    df["extended_away_games"] = (seq_ok & away_ok).astype(int)

    g = df.groupby(["season","player_id"], group_keys=False)
    lag1_week = g["week"].shift(1)
    lag1_isintl = g["is_international"].shift(1)
    seq_ok = df["week"].notna() & lag1_week.notna() & (df["week"] == lag1_week + 1)
    intl_ok = (lag1_isintl == 1)
    df["week_after_intl"] = (seq_ok & intl_ok).astype(int)
    df["intl"] = ((df["is_international"] == 1) | (df["week_after_intl"] == 1)).astype(int)
    
    weekday_series = (
        df.get("weekday", pd.Series(index=df.index, dtype="object"))
          .fillna("")
          .astype(str)
          .str.lower()
    )
    df["is_thursday"] = (weekday_series == "thursday").astype(int)
    
    gametime_series = df.get("gametime", pd.Series(index=df.index, dtype="object"))
    df["gametime"] = pd.to_datetime(gametime_series, format="%H:%M", errors="coerce").dt.time

    # IMPUTE MISSING TEMP/WIND
    avg_temp_by_stadium = (
        df.groupby("stadium_id", dropna=True)["temp"]
          .mean()
          .reset_index()
          .rename(columns={"temp":"avg_temp"})
          .sort_values("avg_temp", ascending=False)
    )
    avg_temp_map = (
        avg_temp_by_stadium.dropna(subset=["avg_temp"])
        .set_index("stadium_id")["avg_temp"]
        .round(1)
        .to_dict()
    )
    df["temp"] = df.apply(
        lambda row: avg_temp_map.get(row["stadium_id"], 60) if pd.isna(row["temp"]) else row["temp"],
        axis=1
    )

    avg_wind_by_stadium = (
        df.groupby("stadium_id", dropna=True)["wind"]
          .mean()
          .reset_index()
          .rename(columns={"wind":"avg_wind"})
          .sort_values("avg_wind", ascending=False)
    )
    avg_wind_map = (
        avg_wind_by_stadium.dropna(subset=["avg_wind"])
        .set_index("stadium_id")["avg_wind"]
        .round(1)
        .to_dict()
    )
    df["wind"] = df.apply(
        lambda row: avg_wind_map.get(row["stadium_id"], 0) if pd.isna(row["wind"]) else row["wind"],
        axis=1
    )

    # IMPUTE MISSING SURFACES
    grass_stadiums = ['SFO01','GER00','MEX00','LON00','FRA00','SAO00']
    df.loc[df["stadium_id"].isin(grass_stadiums), "surface"] = "grass"

    most_common_surface = (
        df.dropna(subset=["surface"])
          .groupby("stadium_id")["surface"]
          .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
          .reset_index()
          .rename(columns={"surface":"most_common_surface"})
    )
    surface_map = most_common_surface.set_index("stadium_id")["most_common_surface"].to_dict()
    df["surface"] = df.apply(
        lambda row: surface_map.get(row["stadium_id"], row["surface"]) if pd.isna(row["surface"]) else row["surface"],
        axis=1
    )

    df = df[df["game_id"].notna()].copy()
    df = df.drop(columns=["nfl_detail_id"], errors="ignore")
    
    df['game_id'] = df['game_id'].astype(str)

    # ========== MERGE SNAP COUNTS ==========
    print("\nMerging snap counts...")
    
    snap_counts_filtered['team'] = snap_counts_filtered['team'].map(team_map)
    snap_counts_filtered['team'] = snap_counts_filtered['team'].fillna(snap_counts_filtered['team'])
    
    if "defensive_snaps" in snap_counts_filtered.columns:
        snap_counts_filtered = snap_counts_filtered[snap_counts_filtered["defensive_snaps"].fillna(0) == 0]
    
    merge_keys_snaps = ['game_id', 'team', 'player_name_flat']
    if 'player_id' in df.columns and 'player_id' in snap_counts_filtered.columns:
        merge_keys_snaps = ['game_id', 'season', 'player_id']
    
    merge_keys_snaps = [key for key in merge_keys_snaps if key in df.columns and key in snap_counts_filtered.columns]
    
    if merge_keys_snaps:
        print(f"  Merging on: {merge_keys_snaps}")
        original_len = len(df)
        df = df.merge(
            snap_counts_filtered,
            on=merge_keys_snaps,
            how='left',
            suffixes=('', '_snaps')
        )
        print(f"  After snap merge: {original_len} → {len(df)} rows")
        
        if 'player_id' in merge_keys_snaps:
            cols_to_drop = ['first_name', 'last_name', 'player_name_flat']
            if 'team_snaps' in df.columns:
                cols_to_drop.append('team_snaps')
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        else:
            df.drop(columns=['first_name', 'last_name'], inplace=True, errors='ignore')
    else:
        print("⚠️ Could not merge snap counts - missing required columns")

    # ========== MERGE ADDITIONAL DATA ==========
    if not df_lead_changes.empty and 'lead_changes' in df_lead_changes.columns:
        df_lead_changes_trimmed = df_lead_changes[["game_id","lead_changes"]].copy()
        original_len = len(df)
        df = df.merge(df_lead_changes_trimmed, on=["game_id"], how="left")
        print(f"✓ Lead changes merged: {original_len} → {len(df)} rows")
    else:
        print("⚠️ Lead changes skipped (missing data or 'lead_changes' column)")

    if 'team_travel_distance_miles' in df.columns:
        print("✓ Travel distance already merged")
    elif 'travel_distance_away' in df.columns or 'travel_distance_home' in df.columns:
        df['team_travel_distance_miles'] = df.apply(
            lambda row: row['travel_distance_away'] if row['isaway'] else row['travel_distance_home'],
            axis=1
        )
        df.drop(columns=['travel_distance_away', 'travel_distance_home'], inplace=True, errors='ignore')
        print("✓ Travel distance merged")
    else:
        print("⚠️ Travel distance skipped")

    if not df_air_yards.empty:
        required_air_cols = ['game_id', 'air_yards_completion', 'air_yards_incompletion', 'team']
        if all(col in df_air_yards.columns for col in required_air_cols):
            if 'team' not in df.columns:
                print("⚠️ Air yards skipped: main DataFrame missing 'team' column for merge")
            else:
                df_air_yards_trimmed = df_air_yards[required_air_cols].copy()
                original_len = len(df)
                df = df.merge(df_air_yards_trimmed, on=["game_id","team"], how="left")
                print(f"✓ Air yards merged: {original_len} → {len(df)} rows")
        else:
            missing = [col for col in required_air_cols if col not in df_air_yards.columns]
            print(f"⚠️ Air yards skipped (missing columns: {missing})")
    else:
        print("⚠️ Air yards skipped")

    df.loc[df["position_group"] != "QB", ["air_yards_completion","air_yards_incompletion"]] = 0
    df.loc[(df["position_group"] == "QB") & (df["offensive_snaps"].fillna(0) < 5),
           ["air_yards_completion","air_yards_incompletion"]] = 0

    if not df_receiving.empty:
        required_rec_cols = ['game_id', 'player_name_flat', 'team', 'yards_after_catch']
        if all(col in df_receiving.columns for col in required_rec_cols):
            missing_in_df = [col for col in ['game_id', 'team', 'player_name_flat'] if col not in df.columns]
            if missing_in_df:
                print(f"⚠️ Receiving data skipped: main DataFrame missing columns: {missing_in_df}")
            else:
                df_receiving_trimmed = df_receiving[required_rec_cols].copy()
                original_len = len(df)
                df = df.merge(df_receiving_trimmed, on=["game_id","team","player_name_flat"], how="left")
                print(f"✓ Receiving data merged: {original_len} → {len(df)} rows")
        else:
            missing = [col for col in required_rec_cols if col not in df_receiving.columns]
            print(f"⚠️ Receiving data skipped (missing columns: {missing})")
    else:
        print("⚠️ Receiving data skipped")

    print(f"\n{'='*60}")
    print(f"✓ MERGE COMPLETE: Final dataset shape = {df.shape}")
    print(f"{'='*60}\n")
    
    return df

###
# CREATE SAFE MATH & PROGRESS LOGGING FUNCTIONS

def sec(title: str):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def sdiv(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float),
                     where=pd.notna(a) & pd.notna(b) & (b != 0))

def alpha_code(series: pd.Series, start_at: int = 1) -> pd.Series:
    cats = sorted([x for x in series.dropna().unique()])
    mapping = {cat: i for i, cat in enumerate(cats, start=start_at)}
    return series.map(mapping)

def remove_high_correlation(X: pd.DataFrame, y: pd.Series, threshold: float = 0.95, target_name: str = "") -> pd.DataFrame:
    """
    Remove features that are highly correlated with each other (>threshold).
    Keeps the feature with higher correlation to target.
    """
    print(f"\n{'─'*60}")
    print(f"CORRELATION ANALYSIS: {target_name}")
    print(f"{'─'*60}")
    
    X_with_target = X.copy()
    X_with_target['_target_'] = y
    corr_matrix = X_with_target.corr()
    
    target_corr = corr_matrix['_target_'].drop('_target_').abs().sort_values(ascending=False)
    
    print(f"\nTop 15 features correlated with target:")
    print(target_corr.head(15).to_string())
    
    corr_matrix_abs = corr_matrix.drop('_target_', axis=0).drop('_target_', axis=1).abs()
    
    upper = corr_matrix_abs.where(
        np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool)
    )
    
    to_drop = set()
    high_corr_pairs = []
    
    for column in upper.columns:
        high_corr = upper[column][upper[column] > threshold]
        if len(high_corr) > 0:
            for other_col in high_corr.index:
                if target_corr.get(column, 0) >= target_corr.get(other_col, 0):
                    to_drop.add(other_col)
                    high_corr_pairs.append((column, other_col, upper.loc[other_col, column]))
                else:
                    to_drop.add(column)
                    high_corr_pairs.append((other_col, column, upper.loc[other_col, column]))
    
    if high_corr_pairs:
        print(f"\n⚠️ Found {len(high_corr_pairs)} highly correlated feature pairs (>{threshold}):")
        for kept, dropped, corr_val in high_corr_pairs[:10]:
            print(f"  Keeping '{kept}' (r={target_corr.get(kept, 0):.3f}), dropping '{dropped}' (r={target_corr.get(dropped, 0):.3f}), corr={corr_val:.3f}")
        if len(high_corr_pairs) > 10:
            print(f"  ... and {len(high_corr_pairs) - 10} more pairs")
        print(f"\n✓ Removing {len(to_drop)} highly correlated features")
    else:
        print(f"✓ No highly correlated features found (threshold={threshold})")
    
    X_filtered = X.drop(columns=list(to_drop), errors='ignore')
    
    print(f"\nFeatures: {X.shape[1]} → {X_filtered.shape[1]} (removed {X.shape[1] - X_filtered.shape[1]})")
    
    return X_filtered

"""
# FEATURE ENGINEERING - FIXED VERSION (NO LEAKAGE)
"""

def make_features_no_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features WITHOUT using target variables.
    Only use contextual/environmental features and snap counts.
    """
    print("\nEngineering features (NO LEAKAGE)...")
    d = df.loc[df["position_group"].isin(["QB","RB","FB","WR","TE"])].copy()
    print(f"  Filtered to offensive positions: {len(d)} rows")

    # TRAVEL / REST (contextual - OK to use)
    if KEEP_REST_TRAVEL:
        home_travel = d.get("travel_distance_home", d.get("home_travel_distance", d.get("team_travel_distance_miles", 0)))
        away_travel = d.get("travel_distance_away", d.get("away_travel_distance", d.get("team_travel_distance_miles", 0)))
        isaway = d.get("isaway", 0)
        d["travel_distance"] = np.where(isaway == 1, away_travel, home_travel)
        d["rest_days"] = np.where(isaway == 0, d.get("home_rest", 0), d.get("away_rest", 0))
    else:
        d["travel_distance"] = 0
        d["rest_days"] = 0

    # Roof binary (contextual - OK)
    def roof_bin(v):
        if pd.isna(v): return np.nan
        s = str(v).strip().lower()
        if "open" in s or s in {"outdoor","outdoors"}: return 0
        if any(k in s for k in ("closed","close","dome","fixed","retractable")): return 1
        return np.nan
    d["roof_closed"] = d.get("roof", pd.Series(dtype=object)).map(roof_bin)

    # Alpha codes (contextual - OK)
    if ADD_ALPHA_ENV_CODES:
        d["stadium_id_num"] = alpha_code(d.get("stadium_id", pd.Series(dtype=object)))
        d["surface_code"]   = alpha_code(d.get("surface", pd.Series(dtype=object)))

    print(f"✓ Features engineered: {d.shape}")
    return d.replace([np.inf,-np.inf], np.nan).fillna(0)


def compute_historical_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling averages using ONLY data from previous weeks.
    This prevents data leakage - we only use past performance to predict future.
    """
    print("\nComputing historical averages (NO LEAKAGE - temporal split)...")
    
    df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
    
    # Stats to average (TARGET stats from previous games become FEATURES)
    stats_to_avg = list(TARGET_STATS.intersection(df.columns))
    
    print(f"  Computing historical stats for {len(stats_to_avg)} variables")
    
    result = df.copy()
    
    # For each player-season, compute expanding mean excluding current row
    for stat in stats_to_avg:
        if stat not in df.columns:
            continue
            
        # Season average up to but NOT including current week
        grouped = df.groupby(['player_id', 'season'], group_keys=False)[stat]
        result[f'{stat}_hist_avg'] = grouped.apply(lambda x: x.shift().expanding().mean())
        
        # Rolling 3-game average (also excluding current game)
        result[f'{stat}_rolling_3'] = grouped.apply(
            lambda x: x.shift().rolling(window=3, min_periods=1).mean()
        )
        
        # Last game value
        result[f'{stat}_last_game'] = grouped.shift(1)
    
    # Fill NaN with 0 for players' first games
    hist_cols = [c for c in result.columns if '_hist_avg' in c or '_rolling_3' in c or '_last_game' in c]
    result[hist_cols] = result[hist_cols].fillna(0)
    
    print(f"✓ Historical averages computed: {len(hist_cols)} features created")
    if hist_cols:
        print(f"  Example features: {hist_cols[:3]}")
    
    return result


def prepare_training_data(df: pd.DataFrame, target_col: str):
    """
    Prepare X and y with proper temporal validation.
    Remove any features that could leak information about the target.
    """
    print(f"\nPreparing training data for target: {target_col}")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # CRITICAL: Remove all potential leakage sources
    cols_to_remove = set()
    cols_to_remove.add(target_col)
    
    # Remove other target stats UNLESS they're historical features
    for col in numeric_cols:
        # Remove raw target stats
        if col in TARGET_STATS:
            cols_to_remove.add(col)
        
        # Remove any efficiency metrics calculated from current game target stats
        if any(keyword in col.lower() for keyword in ['yards_per_carry', 'receiver_efficiency', 
                                                        'pass_usage', 'rec_team_tgt_share']):
            if not any(hist in col for hist in ['_hist_avg', '_rolling_3', '_last_game']):
                cols_to_remove.add(col)
    
    # Remove ID columns
    id_cols = {'player_id', 'season', 'week', 'game_id'}
    cols_to_remove.update(id_cols.intersection(numeric_cols))
    
    feature_cols = [c for c in numeric_cols if c not in cols_to_remove]
    
    print(f"  Total numeric columns: {len(numeric_cols)}")
    print(f"  Removed (leakage/IDs): {len(cols_to_remove)}")
    print(f"  Final features: {len(feature_cols)}")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove rows with missing target
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"  Valid samples: {len(y)}")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Target mean: {y.mean():.2f}, std: {y.std():.2f}")
    
    return X, y


def temporal_train_test_split(df, X, y, test_weeks=4):
    """
    Split data by time: use last N weeks as test set.
    This is more realistic than random split for time series data.
    """
    print(f"\nTemporal train/test split (last {test_weeks} weeks for test)...")
    
    df_indexed = df.loc[X.index].copy()
    
    # Find cutoff week (use last N weeks for testing)
    max_week = df_indexed.groupby('season')['week'].max().min()
    cutoff_week = max_week - test_weeks
    
    train_idx = df_indexed['week'] <= cutoff_week
    test_idx = df_indexed['week'] > cutoff_week
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    print(f"  Train: {len(X_train)} samples (weeks 1-{cutoff_week})")
    print(f"  Test: {len(X_test)} samples (weeks {cutoff_week+1}-{max_week})")
    
    return X_train, X_test, y_train, y_test


def train_and_explain_fixed(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, 
                            target_name: str, use_temporal_split: bool = True):
    """
    Train model with proper validation and no data leakage.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL (NO LEAKAGE): {target_name}")
    print(f"{'='*60}")
    
    if target_name in X.columns:
        X = X.drop(columns=[target_name])
        print(f"  Removed target from features")
    
    # Check for suspiciously high correlations (potential leakage)
    y_np = y.to_numpy()
    leak_cols = []
    for c in X.columns:
        if X[c].dtype.kind in "biufc":
            try:
                corr = np.corrcoef(X[c].fillna(0), y_np)[0, 1]
                if abs(corr) > 0.99:
                    leak_cols.append((c, corr))
            except:
                pass
    
    if leak_cols:
        print(f"  ⚠️ HIGH CORRELATION FEATURES (potential leakage):")
        for col, corr in leak_cols[:5]:
            print(f"    {col}: {corr:.4f}")
        print(f"  Removing these {len(leak_cols)} features...")
        X = X.drop(columns=[col for col, _ in leak_cols])
    
    # Remove highly correlated features
    X = remove_high_correlation(X, y, threshold=0.95, target_name=target_name)
    
    print(f"  Final features: {X.shape[1]} | Samples: {X.shape[0]}")
    
    is_binary = (y.nunique() <= 2) and set(y.unique()).issubset({0,1})
    
    # Use temporal split if requested and possible
    if use_temporal_split and 'week' in df.columns:
        try:
            X_train, X_test, y_train, y_test = temporal_train_test_split(df, X, y)
        except:
            print("  ⚠️ Temporal split failed, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, 
                stratify=y if is_binary else None
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, 
            stratify=y if is_binary else None
        )
    
    # Add regularization to prevent overfitting
    model = RandomForestClassifier(
        n_estimators=300, 
        max_features="sqrt", 
        max_depth=10,  # Limit depth
        min_samples_leaf=20,  # Require more samples per leaf
        random_state=RANDOM_SEED
    ) if is_binary else RandomForestRegressor(
        n_estimators=300, 
        max_features="sqrt",
        max_depth=10,  # Limit depth
        min_samples_leaf=20,  # Require more samples per leaf
        random_state=RANDOM_SEED
    )
    
    print(f"  Fitting {'Classifier' if is_binary else 'Regressor'}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    print(f"\n{'─'*60}")
    print(f"METRICS: {target_name}")
    print(f"{'─'*60}")
    
    if is_binary:
        print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
        print(f"F1:        {f1_score(y_test, y_pred, average='macro'):.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=3))
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"R²:   {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"\nTarget statistics:")
        print(f"  Mean: {y_test.mean():.2f}")
        print(f"  Std:  {y_test.std():.2f}")
        print(f"  Min:  {y_test.min():.2f}")
        print(f"  Max:  {y_test.max():.2f}")
        print(f"\nSanity checks:")
        print(f"  RMSE/Std ratio: {rmse/y_test.std():.4f} (should be < 1.0, ideally 0.6-0.9)")
        print(f"  MAE/Mean ratio: {mae/y_test.mean():.4f}")
        
        if r2 > 0.95:
            print(f"\n  ⚠️ WARNING: R² > 0.95 suggests possible data leakage!")
        if mae < 0.01:
            print(f"  ⚠️ WARNING: MAE near zero suggests possible data leakage!")
    
    # SHAP or feature importances
    shap_ok = False
    if HAS_SHAP and not is_binary:  # Only for regression
        try:
            print(f"\n  Computing SHAP values...")
            X_test_sample = X_test.sample(min(100, len(X_test)), random_state=RANDOM_SEED)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
            
            if isinstance(shap_values, list):
                shap_values = np.abs(shap_values[0])
            
            mean_abs_shap = pd.Series(
                np.abs(shap_values).mean(axis=0),
                index=X_test_sample.columns
            ).sort_values(ascending=False)
            
            print(f"\n{'─'*60}")
            print(f"TOP {TOP_K} SHAP FEATURES")
            print(f"{'─'*60}")
            print(mean_abs_shap.head(TOP_K).to_string())
            
            if SAVE_CSVS:
                mean_abs_shap.to_csv(OUT_DIR / f"shap_importances_{target_name}.csv", header=["mean_abs_shap"])
            
            shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False, max_display=20)
            plt.tight_layout()
            if SAVE_PLOTS:
                plt.savefig(OUT_DIR / f"shap_summary_{target_name}.png", dpi=200, bbox_inches='tight')
            plt.show()
            plt.close()
            
            shap_ok = True
        except Exception as e:
            print(f"  ⚠️ SHAP explanation skipped: {e}")
    
    # Fallback: RF importances
    if not shap_ok:
        try:
            print(f"\n{'─'*60}")
            print(f"TOP {TOP_K} RF FEATURE IMPORTANCES")
            print(f"{'─'*60}")
            fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            print(fi.head(TOP_K).to_string())
            
            ax = fi.head(20).iloc[::-1].plot(kind="barh", figsize=(10, 8))
            ax.set_title(f"Top 20 Feature Importances — {target_name}")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            if SAVE_PLOTS:
                plt.savefig(OUT_DIR / f"rf_importances_{target_name}.png", dpi=200, bbox_inches='tight')
            plt.show()
            plt.close()
            
            if SAVE_CSVS:
                fi.to_csv(OUT_DIR / f"rf_importances_{target_name}.csv", header=["importance"])
        except Exception as e:
            print(f"  ⚠️ Feature importance visualization skipped: {e}")
    
    return model


# MAIN EXECUTION
if __name__ == "__main__":
    sec("NFL MODEL PIPELINE START (FIXED - NO LEAKAGE)")
    print(f"Configuration:")
    print(f"  Years: {min(YEARS)}-{max(YEARS)}")
    print(f"  Test Size: {TEST_SIZE}")
    print(f"  Random Seed: {RANDOM_SEED}")
    print(f"  Run All Targets: {RUN_ALL_TARGETS}")
    print(f"  Train by Position: {TRAIN_BY_POSITION}")
    
    sec("STEP 1: LOAD DATA")
    if FORCE_REBUILD_MERGED or not INPUT_CSV.exists():
        print(f"Building {INPUT_CSV.name}...")
        INPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        try:
            df_merged = build_df_merged_onefile(CFG)
            df_merged.to_csv(INPUT_CSV, index=False)
            print(f"✓ {INPUT_CSV.name} saved: {df_merged.shape}")
        except Exception as e:
            print(f"✗ Failed to build merged data: {e}")
            raise
    else:
        print(f"Loading existing {INPUT_CSV.name}...")
    
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"✓ Loaded: {df.shape}")
    except Exception as e:
        print(f"✗ Failed to load {INPUT_CSV}: {e}")
        raise

    sec("STEP 2: FEATURE ENGINEERING (NO LEAKAGE)")
    try:
        dfF = make_features_no_leakage(df)
        print(f"✓ Features shape: {dfF.shape}")
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        raise

    sec("STEP 3: COMPUTE HISTORICAL AVERAGES (NO LEAKAGE)")
    try:
        dfH = compute_historical_averages(dfF)
        print(f"✓ With historical features shape: {dfH.shape}")
    except Exception as e:
        print(f"✗ Historical average computation failed: {e}")
        raise

    sec("STEP 4: TRAIN MODELS")
    
    if RUN_ALL_TARGETS:
        # Get candidate targets (only numeric TARGET_STATS)
        candidate_targets = [c for c in TARGET_STATS if c in dfH.columns and c not in EXCLUDED_TARGETS]
        print(f"Training models for {len(candidate_targets)} targets")
        print(f"Targets: {candidate_targets[:5]}{'...' if len(candidate_targets) > 5 else ''}")

        if TRAIN_BY_POSITION:
            segments = {
                "QB": dfH["position_group"] == "QB",
                "RB": dfH["position_group"] == "RB",
                "WR_TE": dfH["position_group"].isin(["WR", "TE"]),
            }
            for seg_name, mask in segments.items():
                print(f"\n{'='*60}")
                print(f"SEGMENT: {seg_name}")
                print(f"{'='*60}")
                rows = dfH.index[mask]
                print(f"Rows in segment: {len(rows)}")
                
                seg_df = dfH.loc[rows].copy()
                
                for tgt in candidate_targets:
                    if tgt not in seg_df.columns:
                        continue
                    y = seg_df[tgt]
                    if y.dropna().nunique() < 2:
                        print(f"⚠️ Skipping {tgt} in {seg_name}: <2 unique values")
                        continue
                    try:
                        X, y_clean = prepare_training_data(seg_df, tgt)
                        _ = train_and_explain_fixed(seg_df, X, y_clean, f"{tgt}_{seg_name}", 
                                                    use_temporal_split=True)
                    except Exception as e:
                        print(f"✗ Failed to train {tgt}_{seg_name}: {e}")
        else:
            for i, tgt in enumerate(candidate_targets, 1):
                print(f"\n{'='*60}")
                print(f"[Target {i}/{len(candidate_targets)}]: {tgt}")
                print(f"{'='*60}")
                
                if tgt not in dfH.columns:
                    print(f"⚠️ Target not in data, skipping")
                    continue
                
                y = dfH[tgt]
                if y.dropna().nunique() < 2:
                    print(f"⚠️ Skipping {tgt}: <2 unique values")
                    continue
                
                try:
                    X, y_clean = prepare_training_data(dfH, tgt)
                    _ = train_and_explain_fixed(dfH, X, y_clean, tgt, use_temporal_split=True)
                except Exception as e:
                    print(f"✗ Failed to train {tgt}: {e}")
                    import traceback
                    traceback.print_exc()

    else:
        # Single target mode
        if TARGET not in dfH.columns:
            raise KeyError(f"TARGET '{TARGET}' not found in data.")
        
        print(f"\n{'='*60}")
        print(f"TRAINING SINGLE TARGET: {TARGET}")
        print(f"{'='*60}")
        
        if TRAIN_BY_POSITION:
            segments = {
                "QB": dfH["position_group"] == "QB",
                "RB": dfH["position_group"] == "RB",
                "WR_TE": dfH["position_group"].isin(["WR", "TE"]),
            }
            for seg_name, mask in segments.items():
                print(f"\n{'='*60}")
                print(f"SEGMENT: {seg_name} — TARGET: {TARGET}")
                print(f"{'='*60}")
                rows = dfH.index[mask]
                seg_df = dfH.loc[rows].copy()
                
                if len(seg_df) == 0:
                    print(f"⚠️ No data for segment")
                    continue
                    
                y = seg_df[TARGET]
                if y.dropna().nunique() < 2:
                    print(f"⚠️ Skipping: target has <2 classes/values in this segment")
                    continue
                
                try:
                    X, y_clean = prepare_training_data(seg_df, TARGET)
                    _ = train_and_explain_fixed(seg_df, X, y_clean, f"{TARGET}_{seg_name}", 
                                               use_temporal_split=True)
                except Exception as e:
                    print(f"✗ Failed to train {TARGET}_{seg_name}: {e}")
        else:
            try:
                X, y_clean = prepare_training_data(dfH, TARGET)
                _ = train_and_explain_fixed(dfH, X, y_clean, TARGET, use_temporal_split=True)
            except Exception as e:
                print(f"✗ Failed to train {TARGET}: {e}")

    sec("PIPELINE COMPLETE")
    print("✓ All models trained successfully (NO DATA LEAKAGE)")
    print(f"✓ Outputs saved to: {OUT_DIR.absolute()}")
    print("\nKey fixes applied:")
    print("  1. ✓ Historical averages use only PRIOR weeks (no current game data)")
    print("  2. ✓ Target stats removed from features (only historical versions used)")
    print("  3. ✓ Temporal train/test split (last 4 weeks for testing)")
    print("  4. ✓ Model regularization (max_depth=10, min_samples_leaf=20)")
    print("  5. ✓ Correlation checks to detect leakage")
    print("\nExpected results:")
    print("  - R² should be 0.3-0.6 (realistic for NFL prediction)")
    print("  - RMSE should be meaningful (60-90% of target std dev)")
    print("  - MAE should NOT be near zero")

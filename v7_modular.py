#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 21:33:32 2025

@author: kesh - tay - an
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
    print("SHAP not available; using RF feature_importances_ fallback.")

DATA_DIR = Path("/Users/nikesh/NFL Project")
INPUT_CSV = DATA_DIR / "df_merged.csv"
OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20007
TEST_SIZE   = 0.25

TARGET = "passing_tds_binary"
TRAIN_BY_POSITION = False
YEARS = (2021, 2024)

SEASON_AVG_KIND = "loo"

SAVE_PLOTS = True
SAVE_CSVS  = True
TOP_K = 20

ADD_REC_TEAM_TGT_SHARE = True
ADD_ALPHA_ENV_CODES    = True
KEEP_REST_TRAVEL       = True

PLAYER_COL, SEASON_COL, WEEK_COL = "player_id", "season", "week"

WEEKLY_DATA_CSV = '/Users/nikesh/NFL Project/weekly_data.csv'
RECEIVING_CSV   = '/Users/nikesh/NFL Project/NFL Datasets/receiving_data_NFL.csv'
SCHEDULE_CSV    = '/Users/nikesh/NFL Project/NFL Datasets/schedule_2002_2024_NFL.csv'
SNAPS_CSV       = '/Users/nikesh/NFL Project/NFL Datasets/snap_counts_2021_2024_NFL.csv'
AIR_CSV         = '/Users/nikesh/NFL Project/NFL Datasets/air_yards_data_NFL.csv'
LEAD_CSV        = '/Users/nikesh/NFL Project/NFL Datasets/lead_changes_NFL.csv'

RUN_API = None

# Toggle to force a fresh build of df_merged.csv even if it already exists
FORCE_REBUILD_MERGED = False

from dataclasses import dataclass
from typing import Iterable, Optional
import nfl_data_py as nfl

@dataclass
class CombinedConfig:
    YEARS: Iterable[int] | tuple[int, int]
    weekly_data_csv: Optional[str | Path] = None
    receiving_csv: Optional[str | Path] = None
    schedule_csv: Optional[str | Path] = None
    snaps_csv: Optional[str | Path] = None
    air_csv: Optional[str | Path] = None
    lead_csv: Optional[str | Path] = None
    run_api: Optional[bool] = None

CFG = CombinedConfig(
    YEARS=YEARS,
    weekly_data_csv=WEEKLY_DATA_CSV,
    receiving_csv=RECEIVING_CSV,
    schedule_csv=SCHEDULE_CSV,
    snaps_csv=SNAPS_CSV,
    air_csv=AIR_CSV,
    lead_csv=LEAD_CSV,
    run_api=RUN_API,
)

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
        return None
    p = Path(path_like)
    if not p.exists():
        print(f"File not found, skipping: {p}")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"Failed to read {p}: {e}")
        return None

def build_df_merged_onefile(cfg: CombinedConfig) -> pd.DataFrame:
    years = _normalize_years(cfg.YEARS)

    receiving_data_import = _safe_read_csv(cfg.receiving_csv)
    official_schedule_data_import = _safe_read_csv(cfg.schedule_csv)
    snap_counts_data_import = _safe_read_csv(cfg.snaps_csv)
    air_yards_data_import = _safe_read_csv(cfg.air_csv)
    lead_changes_data_import = _safe_read_csv(cfg.lead_csv)

    schedule_data = _safe_read_csv(cfg.schedule_csv) or nfl.import_schedules(years)
    schedule_data = schedule_data[[c for c in [
        'game_id','season','game_type','week','gameday','weekday','gametime','away_team',
        'away_score','home_team','home_score','location','result','total','overtime',
        'old_game_id','gsis','nfl_detail_id','away_rest','home_rest','div_game','roof',
        'surface','temp','wind','stadium_id','stadium'
    ] if c in schedule_data.columns]]

    from nfl_data_py import import_weekly_rosters
    weekly_roster_data = import_weekly_rosters(years=years, columns=None)
    weekly_roster_data = weekly_roster_data[[c for c in [
        'season','game_type','week','team','position','depth_chart_position','jersey_number',
        'player_name','football_name','player_id','status','status_description_abbr'
    ] if c in weekly_roster_data.columns]]

    weekly_data_cols = [
        'player_id','player_name','player_display_name','position','position_group','recent_team',
        'season','week','season_type','opponent_team','attempts','completions','passing_yards',
        'passing_tds','interceptions','sacks','sack_yards','sack_fumbles','sack_fumbles_lost',
        'passing_air_yards','passing_yards_after_catch','passing_first_downs','passing_2pt_conversions',
        'carries','rushing_yards','rushing_tds','rushing_fumbles','rushing_fumbles_lost',
        'rushing_first_downs','receptions','targets','receiving_yards','receiving_tds',
        'receiving_fumbles','receiving_fumbles_lost','receiving_air_yards',
        'receiving_yards_after_catch','receiving_first_downs','receiving_2pt_conversions',
        'racr','target_share','air_yards_share','wopr'
    ]

    if (RUN_API is True) or (cfg.weekly_data_csv is None):
        weekly_player_data = nfl.import_weekly_data(years, weekly_data_cols, downcast=False)
    else:
        weekly_player_data = _safe_read_csv(cfg.weekly_data_csv)
        if weekly_player_data is None or weekly_player_data.empty:
            raise ValueError(f"weekly_data_csv not found/empty: {cfg.weekly_data_csv}")
        keep = [c for c in weekly_data_cols if c in weekly_player_data.columns]
        weekly_player_data = weekly_player_data[keep].copy()

    weekly_data = pd.merge(
        weekly_player_data,
        weekly_roster_data,
        on=['player_id','season','week'],
        how='left'
    )

    merged_home = pd.merge(
        weekly_data,
        schedule_data,
        left_on=['season','week','team','opponent_team'],
        right_on=['season','week','home_team','away_team'],
        how='left'
    )

    weekly_data = pd.merge(
        merged_home,
        schedule_data,
        left_on=['season','week','opponent_team','team'],
        right_on=['season','week','home_team','away_team'],
        how='left'
    )

    x_cols = [col for col in weekly_data.columns if col.endswith('_x')]
    y_cols = [col for col in weekly_data.columns if col.endswith('_y')]
    base_cols = [col[:-2] for col in x_cols if f"{col[:-2]}_y" in y_cols]
    for col in base_cols:
        weekly_data[col] = weekly_data[f"{col}_x"].combine_first(weekly_data[f"{col}_y"])
        weekly_data.drop(columns=[f"{col}_x", f"{col}_y"], inplace=True)

    club_name_map = {
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
        df_receiving['team'] = df_receiving['club_name'].map(club_name_map)
        df_receiving['player_name_flat'] = (
            df_receiving['first_name'].astype(str) + df_receiving['last_name'].astype(str)
        )
        df_receiving['player_name_flat'] = df_receiving['player_name_flat'].apply(
            lambda x: re.sub(r'[^A-Za-z0-9]', '', x)
        )

    if air_yards_data_import is not None:
        df_air_yards = air_yards_data_import.copy()
        df_air_yards['season'] = df_air_yards['game_id'].astype(str).str[:4].astype(int)
        df_air_yards = df_air_yards[df_air_yards['season'].isin(years)]
    else:
        df_air_yards = pd.DataFrame()

    if not df_air_yards.empty:
        df_air_yards['team'] = df_air_yards['club_name'].map(club_name_map)
        row_counts = df_air_yards.groupby('game_id').size().reset_index(name='row_count')
        bad_games = row_counts[row_counts['row_count'] != 2]
        print(f"Total games: {row_counts.shape[0]}")
        print(f"Games with != 2 rows: {bad_games.shape[0]}")
        print(bad_games.head())

    if lead_changes_data_import is not None:
        df_lead_changes = lead_changes_data_import.copy()
        df_lead_changes['season'] = df_lead_changes['game_id'].astype(str).str[:4].astype(int)
        df_lead_changes = df_lead_changes[df_lead_changes['season'].isin(years)]
    else:
        df_lead_changes = pd.DataFrame()

    if not df_lead_changes.empty:
        df_lead_changes.to_csv('df_lead_changes_check.csv', index=False)
        print("Lead_Changes_Data Cleaned")

    if snap_counts_data_import is not None:
        df_snap_counts = snap_counts_data_import.copy()
        df_snap_counts['team'] = df_snap_counts['club_name'].map(club_name_map)
        df_snap_counts['player_name_flat'] = (
            df_snap_counts['first_name'].astype(str) + df_snap_counts['last_name'].astype(str)
        )
        df_snap_counts['player_name_flat'] = df_snap_counts['player_name_flat'].apply(
            lambda x: re.sub(r'[^A-Za-z0-9]', '', x)
        )
    else:
        df_snap_counts = pd.DataFrame()

    df = weekly_data[weekly_data['season_type'] != 'POST'].copy()

    if "home_team" in df.columns:
        df['isaway'] = (df['recent_team'] != df['home_team']).astype(int)
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

    df["is_thursday"] = (df.get("weekday","").astype(str).str.lower() == "thursday").astype(int)
    df["gametime"] = pd.to_datetime(df["gametime"], format="%H:%M", errors="coerce").dt.time

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

    grass_stadiums = ['SFO01','GER00','MEX00','LON00','FRA00','SAO00']
    df.loc[df["stadium_id"].isin(grass_stadiums), "surface"] = "grass"

    most_common_surface = (
        df.dropna(subset=["surface"])
          .groupby("stadium_id")["surface"]
          .agg(lambda x: x.mode().iloc[0])
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

    if not df_snap_counts.empty:
        snap_cols = [
            "game_id","team","player_name_flat",
            "offensive_snaps","team_offensive_snaps",
            "defensive_snaps","team_defensive_snaps",
            "special_team_snaps","team_special_team_snaps"
        ]
        use_cols = [c for c in snap_cols if c in df_snap_counts.columns]
        df_snap_counts_trimmed = df_snap_counts[use_cols].copy()
    
        if "defensive_snaps" in df_snap_counts_trimmed.columns:
            df_snap_counts_trimmed = df_snap_counts_trimmed[df_snap_counts_trimmed["defensive_snaps"].fillna(0) == 0]
    
        df = df.merge(
            df_snap_counts_trimmed,
            left_on=["old_game_id","team","player_name_flat"],
            right_on=["game_id","team","player_name_flat"],
            how="left",
            suffixes=("", "_snaps")
        )
        df.drop(columns=["game_id_snaps"], inplace=True, errors="ignore")
        print("Snap counts merged")
    else:
        print("Snap counts skipped"
        )

    if not df_lead_changes.empty:
        df_lead_changes_trimmed = df_lead_changes[["game_id","lead_changes"]].copy()
        df = df.merge(df_lead_changes_trimmed, on=["game_id"], how="left")
        print("Lead Changes merged")
    else:
        print("Lead Changes skipped")

    if isinstance(official_schedule_data_import, pd.DataFrame) and not official_schedule_data_import.empty:
        if all(c in official_schedule_data_import.columns for c in ["game_id","travel_distance_away","travel_distance_home"]):
            df_travel_distance_trimmed = official_schedule_data_import[["game_id","travel_distance_away","travel_distance_home"]].copy()
            df = df.merge(df_travel_distance_trimmed, on=["game_id"], how="left")
            print("Travel Distance merged")
        else:
            print("Travel Distance skipped")
    else:
        print("Travel Distance skipped")

    if not df_air_yards.empty:
        df_air_yards_trimmed = df_air_yards[["game_id","air_yards_completion","air_yards_incompletion","team"]].copy()
        df = df.merge(df_air_yards_trimmed, on=["game_id","team"], how="left")
        print("Air Yards merged")
    else:
        print("Air Yards skipped")

    df.loc[df["position_group"] != "QB", ["air_yards_completion","air_yards_incompletion"]] = 0
    df.loc[(df["position_group"] == "QB") & (df["offensive_snaps"] < 5),
           ["air_yards_completion","air_yards_incompletion"]] = 0

    if not df_receiving.empty:
        df_receiving_trimmed = df_receiving[["game_id","player_name_flat","team","yards_after_catch"]].copy()
        df = df.merge(df_receiving_trimmed, on=["game_id","team","player_name_flat"], how="left")
        print("Receiving merged")
    else:
        print("Receiving skipped")

    return df

# Small helpers
# What: Section header printer and safe division.
# Why:  Clear logs; avoid div-by-zero NaNs.
def sec(title: str):
    print(f"\n{title}\n" + "-" * len(title))

def sdiv(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float),
                     where=pd.notna(a) & pd.notna(b) & (b != 0))

# What: Deterministic alpha-encoding for categoricals.
# Why:  Simple numeric codes that work well with trees.
def alpha_code(series: pd.Series, start_at: int = 1) -> pd.Series:
    cats = sorted([x for x in series.dropna().unique()])
    mapping = {cat: i for i, cat in enumerate(cats, start=start_at)}
    return series.map(mapping)


# Feature engineering
# What: Engineer core features for offense and basic environment encoding + optional extras.
# Why:  Rates and simple context help trees; extras add signal with low complexity.
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.loc[df["position_group"].isin(["QB","RB","FB","WR","TE"])].copy()

    # Travel / rest (robust fallbacks)
    if KEEP_REST_TRAVEL:
        home_travel = d.get("travel_distance_home", d.get("home_travel_distance", d.get("travel_distance", 0)))
        away_travel = d.get("travel_distance_away", d.get("away_travel_distance", d.get("travel_distance", 0)))
        isaway = d.get("isaway", 0)
        d["travel_distance"] = np.where(isaway == 1, away_travel, home_travel)
        d["rest_days"] = np.where(isaway == 0, d.get("home_rest", 0), d.get("away_rest", 0))
    else:
        d["travel_distance"] = 0
        d["rest_days"] = 0

    # Core efficiency/usage ratios (must keep)
    d["snap_share"] = sdiv(d.get("offensive_snaps",0).values, d.get("team_offensive_snaps",1).values)
    d["pass_usage"] = sdiv(d.get("attempts",0).values, d.get("offensive_snaps",1).values)
    d["rusher_yards_per_carry"] = sdiv(d.get("rushing_yards",0).values, d.get("carries",1).values)
    d["receiver_efficiency"] = sdiv(d.get("receptions",0).values, d.get("targets",1).values)

    # Simple environment signal (must keep): roof_closed
    def roof_bin(v):
        if pd.isna(v): return np.nan
        s = str(v).strip().lower()
        if "open" in s or s in {"outdoor","outdoors"}: return 0
        if any(k in s for k in ("closed","close","dome","fixed","retractable")): return 1
        return np.nan
    d["roof_closed"] = d.get("roof", pd.Series(dtype=object)).map(roof_bin)

    # Optional: alpha codes for stadium/surface
    if ADD_ALPHA_ENV_CODES:
        d["stadium_id_num"] = alpha_code(d.get("stadium_id", pd.Series(dtype=object)))
        d["surface_code"]   = alpha_code(d.get("surface", pd.Series(dtype=object)))

    # Optional: receiver team target share (WR/TE only)
    if ADD_REC_TEAM_TGT_SHARE:
        try:
            team_attempts = (
                d.loc[d["position_group"]=="QB", ["game_id","team","attempts"]]
                 .groupby(["game_id","team"], as_index=False)["attempts"].sum()
                 .rename(columns={"attempts":"team_attempts"})
            )
            d = d.merge(team_attempts, on=["game_id","team"], how="left")
            d["rec_team_tgt_share"] = np.where(
                d["position_group"].isin(["WR","TE"]),
                sdiv(d.get("targets",0).values, d.get("team_attempts",0).values),
                0.0
            )
            d["team_attempts"] = d["team_attempts"].fillna(0)
        except Exception:
            d["rec_team_tgt_share"] = 0.0

    return d.replace([np.inf,-np.inf], np.nan).fillna(0)


# Season baselines 
# What: Compute per-player-season baselines via Leave-One-Out / Prior / Season mean.
# Why:  Gives expected value per week to create fair deltas.
def season_avg(df: pd.DataFrame, kind: str = SEASON_AVG_KIND) -> pd.DataFrame:
    k = str(kind).lower()
    if k == "loo":
        out = df.copy()
        excl = {PLAYER_COL, SEASON_COL, WEEK_COL}
        num = [c for c in out.select_dtypes(include=[np.number]).columns if c not in excl]
        if not num: return out
        g = out.groupby([PLAYER_COL, SEASON_COL], dropna=False)
        gsum = g[num].transform("sum")
        gcnt = g[num].transform("count")
        cur = out[num].copy()
        cur_exists = cur.notna().astype(int)
        loo = (gsum - cur.fillna(0)).div((gcnt - cur_exists).replace(0, np.nan)).fillna(0.0)
        out[num] = loo
        return out
    if k == "prior":
        out = df.copy().sort_values([PLAYER_COL, SEASON_COL, WEEK_COL], kind="mergesort")
        excl = {PLAYER_COL, SEASON_COL, WEEK_COL}
        num = [c for c in out.select_dtypes(include=[np.number]).columns if c not in excl]
        if not num: return out
        g = out.groupby([PLAYER_COL, SEASON_COL], dropna=False)
        csum = g[num].transform(lambda s: s.fillna(0).cumsum()).shift(1)
        ccnt = g[num].transform(lambda s: s.notna().cumsum().astype("int64")).shift(1)
        out[num] = csum.div(ccnt.replace(0, np.nan))
        return out.fillna(0)
    if k == "season":
        out = df.copy()
        excl = {PLAYER_COL, SEASON_COL}
        num = [c for c in out.select_dtypes(include=[np.number]).columns if c not in excl]
        if not num: return out
        out[num] = out.groupby([PLAYER_COL, SEASON_COL], dropna=False)[num].transform("mean")
        return out.fillna(0)
    raise ValueError("SEASON_AVG_KIND must be one of {'loo','prior','season'}")


# Deltas → binaries 
# What: Vectorized (current - baseline >= 0) for all numerics in one shot.
# Why:  Fast and avoids DataFrame fragmentation warnings.
def add_delta_binaries(df_now: pd.DataFrame, df_base: pd.DataFrame) -> pd.DataFrame:
    key = [PLAYER_COL, SEASON_COL, WEEK_COL]
    base = df_base[key + [c for c in df_base.columns if c not in key]].copy()
    base = base.add_suffix("_season_avg")
    for k in key: base.rename(columns={f"{k}_season_avg": k}, inplace=True)
    m = df_now.merge(base, on=key, how="left")

    num_now = m.select_dtypes(include=[np.number]).columns.tolist()
    pairs = [(c, f"{c}_season_avg") for c in num_now if f"{c}_season_avg" in m.columns]
    if not pairs: return m.fillna(0)

    cur_cols, base_cols = zip(*pairs)
    cur_vals  = m.loc[:, list(cur_cols)].to_numpy()
    base_vals = m.loc[:, list(base_cols)].to_numpy()
    bin_mat = (cur_vals - base_vals >= 0).astype(np.uint8)

    bin_cols = [f"{c}_binary" for c in cur_cols]
    bin_df = pd.DataFrame(bin_mat, index=m.index, columns=bin_cols)
    out = pd.concat([m.drop(columns=list(base_cols), errors="ignore"), bin_df], axis=1)
    return out.replace([np.inf,-np.inf], np.nan).fillna(0)


# Train + explain
# What: Fit RF (cls/reg), print metrics, run SHAP robustly, else RF importances.
# Why:  Always produce performance + interpretable drivers.
def train_and_explain(X: pd.DataFrame, y: pd.Series, target_name: str):
    # Leakage guard v1: drop target if it slipped into features
    if target_name in X.columns:
        X = X.drop(columns=[target_name], errors="ignore")

    # Leakage guard v2: drop exact duplicates of y
    y_np = y.to_numpy()
    leak_cols = []
    for c in list(X.columns):
        s = X[c]
        if s.dtype.kind not in "biufc": continue
        a = s.to_numpy()
        if a.shape == y_np.shape and np.array_equal(a, y_np): leak_cols.append(c)
    if leak_cols:
        print(f"Leakage removed from features: {leak_cols}")
        X = X.drop(columns=leak_cols, errors="ignore")

    is_binary = (y.nunique() <= 2) and set(y.unique()).issubset({0,1})
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y if is_binary else None
    )

    model = RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=RANDOM_SEED) \
            if is_binary else \
            RandomForestRegressor(n_estimators=300, max_features="sqrt", random_state=RANDOM_SEED)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)

    sec(f"Metrics for target: {target_name}")
    if is_binary:
        print("Accuracy:", round(accuracy_score(yte, yhat), 4))
        print("Precision (macro):", round(precision_score(yte, yhat, average="macro", zero_division=0), 4))
        print("Recall (macro):", round(recall_score(yte, yhat, average="macro", zero_division=0), 4))
        print("F1 (macro):", round(f1_score(yte, yhat, average="macro"), 4))
        print("\nClassification report:\n", classification_report(yte, yhat, digits=3))
        print("Confusion matrix:\n", confusion_matrix(yte, yhat))
    else:
        mse = mean_squared_error(yte, yhat); rmse = float(np.sqrt(mse))
        r2 = r2_score(yte, yhat); mae = mean_absolute_error(yte, yhat)
        print("R2:", round(r2, 4)); print("RMSE:", round(rmse, 4)); print("MAE:", round(mae, 4))

    # SHAP (robust)
    shap_ok = False
    if HAS_SHAP:
        try:
            Xte_np = np.asarray(Xte, dtype=float)
            feat_names = [str(c) for c in Xte.columns]
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(Xte_np)
            if isinstance(sv, list):
                sv_use = np.asarray(sv[1], dtype=float) if (is_binary and len(sv) == 2) else \
                         np.mean([np.abs(np.asarray(a, float)) for a in sv], axis=0)
            else:
                sv_use = np.asarray(sv, dtype=float)
            if sv_use.ndim == 3:
                sv_use = sv_use[...,0] if sv_use.shape[-1] == 1 else np.mean(np.abs(sv_use), axis=-1)
            if sv_use.ndim == 1: sv_use = sv_use.reshape(-1, 1)

            sec("Top SHAP features (mean |SHAP|)")
            mean_abs = pd.Series(np.abs(sv_use).mean(axis=0), index=feat_names).sort_values(ascending=False)
            print(mean_abs.head(TOP_K).to_string())
            if SAVE_CSVS: mean_abs.to_csv(OUT_DIR / f"shap_importances_{target_name}.csv", header=["mean_abs_shap"])

            try:
                shap.summary_plot(sv_use, pd.DataFrame(Xte_np, columns=feat_names), plot_type="bar", show=False)
                plt.tight_layout(); 
                if SAVE_PLOTS: plt.savefig(OUT_DIR / f"shap_summary_bar_{target_name}.png", dpi=200)
                plt.show(); plt.close()
                shap.summary_plot(sv_use, pd.DataFrame(Xte_np, columns=feat_names), show=False)
                plt.tight_layout();
                if SAVE_PLOTS: plt.savefig(OUT_DIR / f"shap_beeswarm_{target_name}.png", dpi=200)
                plt.show(); plt.close()
                shap_ok = True
            except Exception as e_plot:
                print(f"SHAP plot fallback used: {e_plot}")
                ax = mean_abs.head(20).iloc[::-1].plot(kind="barh")
                ax.set_title(f"Top SHAP (mean |SHAP|) — {target_name}")
                plt.tight_layout()
                if SAVE_PLOTS: plt.savefig(OUT_DIR / f"shap_bar_fallback_{target_name}.png", dpi=200)
                plt.show(); plt.close()
                shap_ok = True
        except Exception as e:
            print(f"SHAP explanation skipped: {e}")

    # Fallback: RF importances
    if not shap_ok:
        try:
            fi = pd.Series(model.feature_importances_, index=[str(c) for c in X.columns]).sort_values(ascending=False)
            sec("Top RF feature_importances_")
            print(fi.head(TOP_K).to_string())
            ax = fi.head(20).iloc[::-1].plot(kind="barh")
            ax.set_title(f"Top RF Importances — {target_name}")
            plt.tight_layout()
            if SAVE_PLOTS: plt.savefig(OUT_DIR / f"rf_importances_bar_{target_name}.png", dpi=200)
            plt.show(); plt.close()
            if SAVE_CSVS: fi.to_csv(OUT_DIR / f"rf_importances_{target_name}.csv", header=["rf_importance"])
        except Exception as e:
            print(f"RF importance fallback skipped: {e}")

    return model


# (DISABLED) Old drop choreography (kept for reference)
"""
# What: Legacy explicit drop/reorder lists from earlier script.
# Why:  Brittle over time; prefer selecting numerics + dropping IDs and the TARGET.
DROP_COLS = [
    "season_type","opponent_team","depth_chart_position","jersey_number","football_name","recent_team",
    "status","status_description_abbr","game_type","player_name","position","game_id","gameday","weekday",
    "location","stadium","old_game_id","gsis","away_rest","home_rest","is_international","week_after_intl",
    "defensive_snaps","team_defensive_snaps","special_team_snaps","team_special_team_snaps","sack_fumbles_lost",
    "passing_first_downs","passing_2pt_conversions","rushing_first_downs","rushing_fumbles_lost",
    "receiving_fumbles_lost","receiving_first_downs","receiving_2pt_conversions","yards_after_catch",
    "player_name_flat","travel_distance_home","travel_distance_away"
]
COLUMNS_IN_ORDER = [
    "player_id","player_display_name","team","position_group","season","week","gametime",
    "stadium_id","stadium_id_num","roof","roof_closed","surface","surface_code","temp","wind",
    "away_team","away_score","home_team","home_score","result","total","overtime","div_game",
    "isaway","extended_away_games","intl","is_thursday","lead_changes","rest_days","travel_distance",
    "offensive_snaps","team_offensive_snaps","snap_share","attempts","completions","passing_yards",
    "passing_tds","interceptions","sacks","sack_yards","sack_fumbles","passing_air_yards",
    "passing_yards_after_catch","air_yards_completion","air_yards_incompletion","pass_usage",
    "pass_pct_of_offense","pass_air_yard_pct","pass_yards_after_catch_pct","pass_average_air_yards",
    "carries","rushing_yards","rushing_tds","rushing_fumbles","rusher_usage","rusher_fumble_pct",
    "rusher_yards_per_carry","receptions","targets","receiving_yards","receiving_tds","receiving_fumbles",
    "receiving_air_yards","receiving_yards_after_catch","receiver_usage","receiver_efficiency",
    "receiver_yac_pct","receiver_yards_per_reception","receiver_yac_to_air_yards","racr","target_share",
    "air_yards_share","wopr","rec_team_tgt_share"
]
"""


# Main
# What: Run the full pipeline with optional per-segment training.
# Why:  Single clear entry point with concise logs and robust defaults.
if __name__ == "__main__":
    sec("Load data")
    if FORCE_REBUILD_MERGED or not INPUT_CSV.exists():
        print("Building df_merged.csv ...")
        INPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_merged = build_df_merged_onefile(CFG)
        df_merged.to_csv(INPUT_CSV, index=False)
        print("df_merged.csv built.")
    df = pd.read_csv(INPUT_CSV); print("Loaded:", df.shape)

    sec("Feature engineering")
    dfF = make_features(df); print("Features shape:", dfF.shape)

    sec(f"Season baselines ({SEASON_AVG_KIND.upper()})")
    dfBase = season_avg(dfF, kind=SEASON_AVG_KIND); print("Baseline shape:", dfBase.shape)

    sec("Deltas → binaries")
    dfB = add_delta_binaries(dfF, dfBase); print("With binaries shape:", dfB.shape)

    # Build raw numeric feature matrix and label
    id_cols = {"player_id","season","week"}
    X_full = dfB.select_dtypes(include=[np.number]).drop(columns=[c for c in id_cols if c in dfB.columns], errors="ignore")
    if TARGET not in dfB.columns:
        raise KeyError(f"TARGET '{TARGET}' not found in data.")
    y_full = dfB[TARGET].astype(int if dfB[TARGET].dropna().isin([0,1]).all() else float)

    if TRAIN_BY_POSITION:
        # Per-segment subsets: pass (QB), rush (RB), rec (WR/TE)
        segments = {
            "pass_QB": dfB["position_group"] == "QB",
            "rush_RB": dfB["position_group"] == "RB",
            "rec_WRTE": dfB["position_group"].isin(["WR","TE"]),
        }
        for seg_name, mask in segments.items():
            sec(f"Train + explain for segment: {seg_name} — target: {TARGET}")
            rows = dfB.index[mask]
            X = X_full.loc[rows].drop(columns=[TARGET], errors="ignore")
            y = y_full.loc[rows]
            if len(y.unique()) < 2:
                print("Skipping: target has <2 classes/values in this segment.")
                continue
            _ = train_and_explain(X, y, f"{TARGET}_{seg_name}")
    else:
        sec(f"Train + explain for target: {TARGET}")
        X = X_full.drop(columns=[TARGET], errors="ignore")
        y = y_full
        _ = train_and_explain(X, y, TARGET)

    sec("Done")
    print("Pipeline complete. Metrics printed, plots shown/saved.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: nik - an - tay
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# =============================
# Config
# =============================
DATA_DIR = Path("/Users/nikesh/NFL Project/Modular")
INPUT_CSV = DATA_DIR / "df_merged.csv"

OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_INTERMEDIATE_CSVS = False
SAVE_RF_SUMMARIES = True

DATA_ANALYSIS_CSV = OUT_DIR / "data_analysis.csv"
COMPARE_SEASON_AVG_BINARY_CSV = OUT_DIR / "compare_szn_avg_binary.csv"
LOO_SZN_AVG_CSV = OUT_DIR / "df_LOO_szn_avg.csv"
PRIOR_GAMES_AVG_CSV = OUT_DIR / "df_prior_games_avg.csv"
WHOLE_SEASON_AVG_CSV = OUT_DIR / "whole_season_avg.csv"
FLAG_SZN_POS_AVG_CSV = OUT_DIR / "all_flag_szn_pos_avgs.csv"
FLAG_AGG_SZN_POS_AVG_CSV = OUT_DIR / "all_flag_agg_pos_avg.csv"

RF_SEG_METRICS_CSV = OUT_DIR / "rf_full_metrics_comparison_{segment}.csv"
RF_ALL_METRICS_CSV = OUT_DIR / "rf_full_metrics_comparison_ALL.csv"

PLAYER_COL = "player_id"
SEASON_COL = "season"
WEEK_COL   = "week"
BINARY_FLAGS = ["isaway", "is_thursday", "extended_away_games", "intl"]
SEASON_AVG_TYPE = "loo"   # 'loo' | 'prior' | 'season'

RANDOM_SEED = 99

# =============================
# Column sets
# =============================
# CHANGED: keep stadium_id, roof, surface, temp, wind (removed from DROP_COLS)
DROP_COLS = [
    "season_type",
    "opponent_team", "depth_chart_position", "jersey_number", "football_name", "recent_team",
    "status", "status_description_abbr", "game_type", "player_name", "position",
    "game_id", "gameday", "weekday", "location", "stadium",  # keep stadium_id (drop 'stadium' name only)
    "old_game_id", "gsis", "away_rest", "home_rest",  # keep roof/surface/temp/wind
    "is_international", "week_after_intl", "defensive_snaps", "team_defensive_snaps",
    "special_team_snaps", "team_special_team_snaps",
    "sack_fumbles_lost", "passing_first_downs", "passing_2pt_conversions",
    "rushing_first_downs", "rushing_fumbles_lost",
    "receiving_fumbles_lost", "receiving_first_downs", "receiving_2pt_conversions", "yards_after_catch",
    "player_name_flat",
    "travel_distance_home", "travel_distance_away"
]

COLUMNS_IN_ORDER = [
    "player_id", "player_display_name", "team", "position_group", "season", "week", "gametime",
    # environment
    "stadium_id", "stadium_id_num", "roof", "roof_closed", "surface", "surface_code", "temp", "wind",
    # game result / flags
    "away_team", "away_score", "home_team", "home_score", "result", "total",
    "overtime", "div_game", "isaway", "extended_away_games", "intl", "is_thursday",
    "lead_changes", "rest_days", "travel_distance",
    # snaps / passing
    "offensive_snaps", "team_offensive_snaps", "snap_share",
    "attempts", "completions", "passing_yards",
    "passing_tds", "interceptions", "sacks", "sack_yards",
    "sack_fumbles", "passing_air_yards", "passing_yards_after_catch",
    "air_yards_completion", "air_yards_incompletion",
    "pass_usage", "pass_pct_of_offense", "pass_air_yard_pct", "pass_yards_after_catch_pct",
    "pass_average_air_yards",
    # rushing
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
    "rusher_usage", "rusher_fumble_pct", "rusher_yards_per_carry",
    # receiving
    "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_fumbles",
    "receiving_air_yards", "receiving_yards_after_catch", "receiver_usage",
    "receiver_efficiency", "receiver_yac_pct", "receiver_yards_per_reception", "receiver_yac_to_air_yards",
    "racr", "target_share", "air_yards_share", "wopr",
    "rec_team_tgt_share"
]

# stats being compared to the players seasonal average
AVERAGE_STATS = [
    "offensive_snaps", "team_offensive_snaps", "snap_share",
    "attempts", "completions", "passing_yards",
    "passing_tds", "interceptions", "sacks", "sack_yards",
    "sack_fumbles", "passing_air_yards", "passing_yards_after_catch",
    "air_yards_completion", "air_yards_incompletion",
    "pass_usage", "pass_pct_of_offense", "pass_air_yard_pct", "pass_yards_after_catch_pct",
    "pass_average_air_yards",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
    "rusher_usage", "rusher_fumble_pct", "rusher_yards_per_carry",
    "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_fumbles",
    "receiving_air_yards", "receiving_yards_after_catch", "receiver_usage",
    "receiver_efficiency", "receiver_yac_pct", "receiver_yards_per_reception", "receiver_yac_to_air_yards",
    "racr", "target_share", "air_yards_share", "wopr",
    "rec_team_tgt_share",
]

# ==========================================================
# Functions
# ==========================================================
def _roof_binary(val):
    """
    Returns 1 if game is indoors (closed/dome/fixed/retractable), 
    0 if game is outdoors (open/outdoors). Otherwise NaN
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in {"open", "outdoor", "outdoors"} or "open" in s:
        return 0
    if s in {"closed", "dome", "fixed", "retractable"} or \
       "close" in s or "dome" in s or "fixed" in s or "retractable" in s:
        return 1
    return np.nan


def _make_alpha_code(series: pd.Series, start_at: int = 1) -> pd.Series:
    """
    Deterministic alphabetical coding for categorical strings.
    Missing values remain NaN.
    """
    cats = sorted([x for x in series.dropna().unique()])
    mapping = {cat: i for i, cat in enumerate(cats, start=start_at)}
    return series.map(mapping), mapping


def make_analysis_frame(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Filter offensive players and add engineered features.
    Includes:
      - rec_team_tgt_share = receiver targets / total team QB attempts
      - stadium_id_num (alphabetical codes)
      - surface_code (alphabetical codes)
      - roof_closed (1=closed/dome/fixed, 0=open/outdoors)
    """
    offense_groups = ["QB", "RB", "FB", "WR", "TE"]

    # Team QB attempts per game/team
    team_qb_attempts = (
        df_merged.loc[df_merged["position_group"] == "QB", ["game_id", "team", "attempts"]]
                 .groupby(["game_id", "team"], as_index=False)["attempts"]
                 .sum()
                 .rename(columns={"attempts": "team_attempts"})
    )

    # Base filtered frame
    d = (
        df_merged
        .loc[df_merged["position_group"].isin(offense_groups)]
        .merge(team_qb_attempts, on=["game_id", "team"], how="left")
        .assign(
            travel_distance=lambda d: np.where(
                (d["isaway"] == 0) & (d["intl"] == 1),
                d["travel_distance_home"],
                np.where(d["isaway"] == 1, d["travel_distance_away"], 0)
            ),
            rest_days=lambda d: np.where(d["isaway"] == 0, d["away_rest"], d["home_rest"]),
            snap_share=lambda d: d["offensive_snaps"].div(d["team_offensive_snaps"]),
            # Passing
            pass_usage=lambda d: d["attempts"].div(d["offensive_snaps"]),
            pass_pct_of_offense=lambda d: d["attempts"].div(d["team_offensive_snaps"]),
            pass_air_yard_pct=lambda d: d["passing_air_yards"].div(d["passing_yards"]),
            pass_yards_after_catch_pct=lambda d: d["passing_yards_after_catch"].div(d["passing_yards"]),
            pass_average_air_yards=lambda d: d[["air_yards_completion", "air_yards_incompletion"]].mean(axis=1),
            # Rushing
            rusher_usage=lambda d: d["carries"].div(d["offensive_snaps"]),
            rusher_fumble_pct=lambda d: d["rushing_fumbles"].div(d["carries"]),
            rusher_yards_per_carry=lambda d: d["rushing_yards"].div(d["carries"]),
            # Receiving
            receiver_usage=lambda d: d["targets"].div(d["offensive_snaps"]),
            receiver_efficiency=lambda d: d["receptions"].div(d["targets"]),
            receiver_yac_pct=lambda d: d["receiving_yards_after_catch"].div(d["receiving_yards"]),
            receiver_yards_per_reception=lambda d: d["receiving_yards"].div(d["receptions"]),
            receiver_yac_to_air_yards=lambda d: d["receiving_yards_after_catch"].div(d["receiving_air_yards"]),
            # NEW: team target share for receivers (WR/TE)
            rec_team_tgt_share=lambda d: np.where(
                d["position_group"].isin(["WR", "TE"]),
                d["targets"] / d["team_attempts"],
                0.0
            ),
        )
        .copy()
    )

    # ===== Environment encodings (numeric) =====
    # stadium_id -> alphabetical codes (1..K)
    d["stadium_id_num"], stadium_map = _make_alpha_code(d.get("stadium_id", pd.Series(dtype=object)))
    # surface -> alphabetical codes (1..S)
    d["surface_code"], surface_map = _make_alpha_code(d.get("surface", pd.Series(dtype=object)))
    # roof -> binary closed/open
    d["roof_closed"] = d.get("roof", pd.Series(dtype=object)).map(_roof_binary)

    # Replace inf with NaN; keep NaNs if any ambiguous roof types occur
    d = d.replace([np.inf, -np.inf], np.nan).fillna({"team_attempts": 0, "rec_team_tgt_share": 0})

    # Console audit of environment categories
    try:
        unique_roof = sorted(pd.Series(d["roof"].unique(), dtype=object).dropna().astype(str))
        unique_surface = sorted(pd.Series(d["surface"].unique(), dtype=object).dropna().astype(str))
        print(f"Unique roof values: {unique_roof}")
        print(f"Unique surface values: {unique_surface}")
        print(f"Stadium ID map (alphabetical -> code): {stadium_map}")
        print(f"Surface map (alphabetical -> code): {surface_map}")
    except Exception:
        pass

    return d


def make_loo_avg(df: pd.DataFrame,
                 player_col: str = PLAYER_COL,
                 season_col: str = SEASON_COL,
                 week_col: str = WEEK_COL,
                 fillna_value: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    exclude = {player_col, season_col, week_col}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if not numeric_cols:
        return df
    g = df.groupby([player_col, season_col], dropna=False)
    gsum = g[numeric_cols].transform("sum")
    gcount = g[numeric_cols].transform("count")
    cur_vals = df[numeric_cols].copy()
    cur_exists = cur_vals.notna().astype(int)
    loo_sum = gsum - cur_vals.fillna(0)
    loo_cnt = gcount - cur_exists
    loo_avg = loo_sum.div(loo_cnt.replace(0, np.nan)).fillna(fillna_value)
    out = df.copy()
    out[numeric_cols] = loo_avg
    return out


def make_prior_games_avg(df: pd.DataFrame,
                         player_col: str = PLAYER_COL,
                         season_col: str = SEASON_COL,
                         week_col: str = WEEK_COL,
                         fillna_with: float | None = None) -> pd.DataFrame:
    out = df.copy().sort_values([player_col, season_col, week_col], kind="mergesort")
    exclude = {player_col, season_col, week_col}
    numeric_cols = [c for c in out.select_dtypes(include=[np.number]).columns if c not in exclude]
    if not numeric_cols:
        return out
    g = out.groupby([player_col, season_col], dropna=False)
    csum = g[numeric_cols].transform(lambda s: s.fillna(0).cumsum())
    ccnt = g[numeric_cols].transform(lambda s: s.notna().cumsum().astype("int64"))
    prior_sum = csum.shift(1)
    prior_cnt = ccnt.shift(1)
    prior_avg = prior_sum.div(prior_cnt.replace(0, np.nan))
    out[numeric_cols] = prior_avg
    if fillna_with is not None:
        out[numeric_cols] = out[numeric_cols].fillna(fillna_with)
    out = out.loc[df.index]
    return out


def make_season_avg(df: pd.DataFrame,
                    player_col: str = PLAYER_COL,
                    season_col: str = SEASON_COL,
                    fillna_with: float | None = None) -> pd.DataFrame:
    out = df.copy()
    exclude = {player_col, season_col}
    numeric_cols = [c for c in out.select_dtypes(include="number").columns if c not in exclude]
    if not numeric_cols:
        return out
    g = out.groupby([player_col, season_col], dropna=False)
    season_means = g[numeric_cols].transform("mean")
    out[numeric_cols] = season_means
    if fillna_with is not None:
        out[numeric_cols] = out[numeric_cols].fillna(fillna_with)
    return out

def season_avg(df: pd.DataFrame, kind: str = SEASON_AVG_TYPE) -> pd.DataFrame:
    k = str(kind).strip().lower()
    if k == "loo":
        return make_loo_avg(
            df,
            player_col=PLAYER_COL,
            season_col=SEASON_COL,
            week_col=WEEK_COL,
            fillna_value=0.0,
        )
    if k == "prior":
        return make_prior_games_avg(
            df,
            player_col=PLAYER_COL,
            season_col=SEASON_COL,
            week_col=WEEK_COL,
            fillna_with=None,
        )
    if k == "season":
        return make_season_avg(
            df,
            player_col=PLAYER_COL,
            season_col=SEASON_COL,
            fillna_with=None,
        )
    raise ValueError(f"Unknown kind='{kind}'. Choose 'loo' | 'prior' | 'season'.")

def add_delta_columns(data_analysis: pd.DataFrame, get_season_avg_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [PLAYER_COL, SEASON_COL, WEEK_COL]
    keep_cols = key_cols + [c for c in AVERAGE_STATS if c in get_season_avg_df.columns]
    loo_min = get_season_avg_df[keep_cols].rename(columns={
        c: f"{c}_season_avg" for c in AVERAGE_STATS if c in get_season_avg_df.columns
    })
    merged = data_analysis.merge(loo_min, on=key_cols, how="left")
    for col in AVERAGE_STATS:
        if col in merged.columns and f"{col}_season_avg" in merged.columns:
            merged[f"{col}_binary"] = merged[col] - merged[f"{col}_season_avg"]
    season_avg_cols = [f"{col}_season_avg" for col in AVERAGE_STATS if f"{col}_season_avg" in merged.columns]
    merged = merged.drop(columns=season_avg_cols, errors="ignore")
    return merged


def make_binary_from_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    binary_cols = [c for c in out.columns if c.endswith("_binary")]
    if binary_cols:
        out[binary_cols] = (out[binary_cols] >= 0).astype(int)
    return out


def season_flag_averages(data_analysis: pd.DataFrame) -> pd.DataFrame:
    binary_cols = [f"{c}_binary" for c in AVERAGE_STATS if f"{c}_binary" in data_analysis.columns]
    stat_cols = [c for c in AVERAGE_STATS if c in data_analysis.columns] + binary_cols
    frames = []
    for flag in [f for f in BINARY_FLAGS if f in data_analysis.columns]:
        tmp = (
            data_analysis.groupby(["position_group", "season", flag])[stat_cols]
            .mean()
            .reset_index()
        )
        tmp[stat_cols] = tmp[stat_cols].round(2)
        tmp["flag"] = flag
        tmp = tmp.rename(columns={flag: "flag_value"})
        cols = ["flag", "flag_value"] + [c for c in tmp.columns if c not in ("flag", "flag_value")]
        frames.append(tmp[cols])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(by=["position_group", "season", "flag", "flag_value"])
    return out


def aggregate_T4(flag_season_avg: pd.DataFrame) -> pd.DataFrame:
    if flag_season_avg.empty:
        return flag_season_avg
    cols = [c for c in flag_season_avg.columns if c not in ["position_group", "flag", "flag_value", "season"]]
    agg = (
        flag_season_avg
        .groupby(["position_group", "flag", "flag_value"], as_index=False)[cols]
        .mean()
    )
    return agg


# =======================================================================================
# PCA 
# =======================================================================================
def pca_workflow(df: pd.DataFrame, name: str, exclude_cols=("result", "total"), var_target=0.80):
    print(f"\n===== PCA workflow for: {name} =====")
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    df_numeric = df_numeric.drop(columns=[c for c in exclude_cols if c in df_numeric.columns], errors="ignore")
    df_numeric = df_numeric.fillna(0)
    if df_numeric.shape[1] == 0:
        print("No numeric feature columns available after exclusions.")
        return None
    X_scaled = StandardScaler().fit_transform(df_numeric)
    pca = PCA(n_components=min(10, df_numeric.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    print("Explained variance ratio:", explained_variance)
    print("Cumulative explained variance:", np.cumsum(explained_variance))
    try:
        plt.figure(figsize=(8,5))
        plt.plot(np.cumsum(explained_variance), marker='o')
        plt.xlabel('Number of Components'); plt.ylabel('Cumulative Explained Variance')
        plt.title(f'PCA - Cumulative Explained Variance ({name})')
        plt.grid(True); plt.tight_layout(); plt.show()
    except Exception:
        pass
    pca_full = PCA().fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cum_var >= var_target) + 1)
    print(f"Number of components to capture {int(var_target*100)}% variance: {n_components}")
    pca_reduced = PCA(n_components=n_components)
    X_reduced = pca_reduced.fit_transform(X_scaled)
    loadings = pd.DataFrame(
        pca_reduced.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=df_numeric.columns
    )
    for i in range(n_components):
        print(f"\nTop contributing variables for PC{i+1} ({name}):")
        print(loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(5))
    loadings['total_contribution'] = loadings.abs().sum(axis=1)
    k = min(25, loadings.shape[0])
    top_variables = loadings['total_contribution'].sort_values(ascending=False).head(k).index
    df_reduced = df_numeric[top_variables]
    print(f"\nReduced dataset shape (top {k} variables): {df_reduced.shape}")
    try:
        corr_matrix = df_reduced.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=(k <= 20), fmt=".2f", cmap="coolwarm",
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f"Correlation Heatmap ({name}) - Top {k} Variables", fontsize=14)
        plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.show()
    except Exception:
        pass
    return {
        "df_numeric": df_numeric,
        "X_scaled": X_scaled,
        "pca_10": pca,
        "pca_reduced": pca_reduced,
        "n_components": n_components,
        "cum_var_at_n": cum_var[n_components-1],
        "loadings": loadings,
        "top_variables": list(top_variables),
        "df_reduced": df_reduced
    }


def rf_exhaustive_targets(reduced_sets: dict[str, pd.DataFrame],
                          excluded_targets: list[str] | tuple[str, ...] = ("isaway", "intl", "is_thursday"),
                          test_size: float = 0.25,
                          random_state: int = RANDOM_SEED) -> pd.DataFrame:
    all_results = []
    for segment, df_reduced in reduced_sets.items():
        print(f"\n================= SEGMENT: {segment.upper()} =================")
        candidate_targets = [col for col in df_reduced.columns if col not in excluded_targets]
        seg_results = []

        for target in candidate_targets:
            print(f"\n----- Processing target: {target} -----")
            feature_cols = [col for col in df_reduced.columns if col != target]
            X = df_reduced[feature_cols].copy()
            y = df_reduced[target].copy()

            # basic validity
            valid_mask = ~y.isna() & ~X.isna().any(axis=1)
            X_final = X.loc[valid_mask]
            y_final = y.loc[valid_mask]

            if len(y_final) == 0 or y_final.nunique() < 2:
                print(f"Skipping {target}: insufficient or constant data")
                continue

            # --- INLINE target-type detection (no external helper) ---
            y_clean = y_final.dropna()
            is_cls = (
                str(target).endswith("_binary") or
                pd.api.types.is_bool_dtype(y_clean) or
                y_clean.isin([0, 1]).all() or
                (pd.api.types.is_integer_dtype(y_clean) and y_clean.nunique() <= 15)
            )
            print(f"Valid samples: {len(y_final)}, unique target values: {y_final.nunique()} "
                  f"-> {'classification' if is_cls else 'regression'}")

            X_train, X_test, y_train, y_test = train_test_split(
                X_final, y_final, test_size=test_size, random_state=random_state
            )

            if is_cls:
                # Coerce 0/1 floats to ints for cleanliness
                if y_train.dropna().isin([0, 1]).all():
                    y_train = y_train.astype(int)
                    y_test  = y_test.astype(int)

                model = RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                row = {
                    "segment": segment, "target": target, "type": "classification",
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
                    "recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
                    "f1_score": f1_score(y_test, y_pred, average='macro', zero_division=0),
                    "r2": None, "mse": None, "rmse": None, "mae": None,
                    "n_samples": len(y_final)
                }
            else:
                model = RandomForestRegressor(n_estimators=200, max_features="sqrt", random_state=random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                row = {
                    "segment": segment, "target": target, "type": "regression",
                    "accuracy": None, "precision": None, "recall": None, "f1_score": None,
                    "r2": r2_score(y_test, y_pred),
                    "mse": mse, "rmse": np.sqrt(mse), "mae": mean_absolute_error(y_test, y_pred),
                    "n_samples": len(y_final)
                }

            seg_results.append(row)
            all_results.append(row)

        if seg_results and SAVE_RF_SUMMARIES:
            seg_df = pd.DataFrame(seg_results).sort_values(
                by=["type", "f1_score", "r2"], ascending=[True, False, False]
            )
            print(f"\nTop 15 for segment {segment.upper()}:")
            print(seg_df.head(15).to_string(index=False))
            seg_df.to_csv(str(RF_SEG_METRICS_CSV).format(segment=segment), index=False)

    results_df = pd.DataFrame(all_results).sort_values(
        by=["segment", "type", "f1_score", "r2"], ascending=[True, True, False, False]
    )
    if SAVE_RF_SUMMARIES:
        results_df.to_csv(RF_ALL_METRICS_CSV, index=False)
    print("\nCombined top 20 overall:")
    print(results_df.head(20).to_string(index=False))
    return results_df



# =============================
# Main
# =============================
if __name__ == "__main__":
    # 1) Read base
    df_merged = pd.read_csv(INPUT_CSV)
    print(df_merged.head())
    print("Cols:", ", ".join(df_merged.columns))

    # 2) Engineer features (includes environment encodings)
    df_analysis = make_analysis_frame(df_merged)

    # 3) Drop unneeded cols and reorder
    to_drop = [c for c in DROP_COLS if c in df_analysis.columns]
    data_analysis = df_analysis.drop(columns=to_drop, errors="ignore").copy()
    final_order = [c for c in COLUMNS_IN_ORDER if c in data_analysis.columns]
    if final_order:
        data_analysis = data_analysis[final_order]

    # 4) LOO / prior / season (same shape)
    df_loo_szn_avg     = make_loo_avg(df_analysis, fillna_value=0.0)
    df_prior_games_avg = make_prior_games_avg(df_analysis, fillna_with=None)
    df_season_avg      = make_season_avg(df_analysis, fillna_with=None)

    if SAVE_INTERMEDIATE_CSVS:
        df_loo_szn_avg.to_csv(LOO_SZN_AVG_CSV, index=False)
        df_prior_games_avg.to_csv(PRIOR_GAMES_AVG_CSV, index=False)
        df_season_avg.to_csv(WHOLE_SEASON_AVG_CSV, index=False)

    # 5) Merge selected seasonal average & compute deltas
    df_szn_choice = season_avg(data_analysis, kind=SEASON_AVG_TYPE)
    data_analysis = add_delta_columns(data_analysis, df_szn_choice)

    # 6) Binary deltas
    compare_szn_avg_binary = make_binary_from_deltas(data_analysis)

    # 7) Write core datasets (optional)
    if SAVE_INTERMEDIATE_CSVS:
        data_analysis.to_csv(DATA_ANALYSIS_CSV, index=False)
        compare_szn_avg_binary.to_csv(COMPARE_SEASON_AVG_BINARY_CSV, index=False)

    # 8) Season averages by flags + T4 (optional)
    flag_season_avg = season_flag_averages(data_analysis)
    if not flag_season_avg.empty and SAVE_INTERMEDIATE_CSVS:
        for flag in [f for f in BINARY_FLAGS if f in data_analysis.columns]:
            frag = flag_season_avg.loc[flag_season_avg["flag"] == flag].copy()
            frag.to_csv(OUT_DIR / f"{flag}_season_position_avg.csv", index=False)
        flag_season_avg.to_csv(FLAG_SZN_POS_AVG_CSV, index=False)
        flag_aggregate_avg = aggregate_T4(flag_season_avg)
        flag_aggregate_avg.to_csv(FLAG_AGG_SZN_POS_AVG_CSV, index=False)

    # =============================
    # Build RF inputs (in-memory)
    # =============================
    rf_model_data_pass = compare_szn_avg_binary.loc[compare_szn_avg_binary['position_group'] == 'QB'].copy()
    rf_model_data_rush = compare_szn_avg_binary.loc[compare_szn_avg_binary['position_group'] == 'RB'].copy()
    rf_model_data_rec  = compare_szn_avg_binary.loc[compare_szn_avg_binary['position_group'].isin(['WR', 'RB', 'TE'])].copy()

    # Safe drop non-features
    pass_drop_columns = [
        "player_id", "player_display_name", "team", "position_group", "season", "week", "gametime", "player_display_name_season_avg",
        "away_team", "away_score", "home_team", "home_score", "result", "total", "lead_changes",
    ]
    rush_drop_columns = pass_drop_columns.copy()
    rec_drop_columns  = pass_drop_columns.copy()

    rf_model_data_pass = rf_model_data_pass.drop(columns=pass_drop_columns, errors="ignore")
    rf_model_data_rush = rf_model_data_rush.drop(columns=rush_drop_columns, errors="ignore")
    rf_model_data_rec  = rf_model_data_rec.drop(columns=rec_drop_columns,  errors="ignore")

    # Quick NA check + impute
    print("rec has NaN:", rf_model_data_rec.isna().any().any())
    print("rush has NaN:", rf_model_data_rush.isna().any().any())
    print("pass has NaN:", rf_model_data_pass.isna().any().any())
    rf_model_data_rec  = rf_model_data_rec.fillna(0)
    rf_model_data_rush = rf_model_data_rush.fillna(0)
    rf_model_data_pass = rf_model_data_pass.fillna(0)
    print("NaNs imputed with 0 (for model inputs)")

    # =============================
    # PCA per segment
    # =============================
    dfs_for_pca = {"rec": rf_model_data_rec, "rush": rf_model_data_rush, "pass": rf_model_data_pass}
    pca_results = {name: pca_workflow(df, name=name, exclude_cols=("result", "total"), var_target=0.80)
                   for name, df in dfs_for_pca.items()}

    reduced_sets = {seg: res["df_reduced"] for seg, res in pca_results.items() if res is not None}

    # =============================
    # RF loop over targets
    # =============================
    _ = rf_exhaustive_targets(reduced_sets, excluded_targets=["isaway", "intl", "is_thursday"], test_size=0.25)

    print("FINITO")

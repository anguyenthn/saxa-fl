#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 21:33:32 2025

@author: kesh - tay - an
"""

from __future__ import annotations

# What: Minimal NFL modeling pipeline with RF + SHAP, LOO baselines, and optional extras.
# Why:  Keep it concise and robust, but let you flip on richer features when useful.

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

# What: Optional SHAP for explanations with safe fallbacks.
# Why:  Explain feature impact; if SHAP fails/unavailable, fall back to RF importances.
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
    print("SHAP not available; using RF feature_importances_ fallback.")


# Config
# What: Central config for paths, switches, and output.
# Why:  Tweak once; everything else reads from here.
DATA_DIR = Path("/Users/an/Downloads/Projects/NFL/Data")
INPUT_CSV = DATA_DIR / "df_merged.csv"
OUT_DIR = Path("."); OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20007
TEST_SIZE   = 0.25

# Core target and training scope
TARGET = "passing_tds_binary"            # choose target column
TRAIN_BY_POSITION = False                # set True to train per segment (QB/RB/WR+TE)

# Seasonal baseline kind
SEASON_AVG_KIND = "loo"                  # 'loo' | 'prior' | 'season'

# Outputs
SAVE_PLOTS = True
SAVE_CSVS  = True
TOP_K = 20

# Optional “nice to keep” feature toggles
ADD_REC_TEAM_TGT_SHARE = True           # WR/TE team target share
ADD_ALPHA_ENV_CODES    = True           # stadium_id_num / surface_code
KEEP_REST_TRAVEL       = True           # rest_days / travel_distance

# IDs
PLAYER_COL, SEASON_COL, WEEK_COL = "player_id", "season", "week"


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

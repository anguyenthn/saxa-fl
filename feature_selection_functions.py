#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:46:51 2025

@author: an
"""

# 1. Imports
# Why: Keep it minimal and fast to read/run.
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# 2. Config
# Centralize paths/params to change once.
DATA_PATH = '/Users/nikesh/NFL Project/data_analysis_delta_binary.csv'
SELECTION_TARGET = 'point_diff'      # set to a real numeric column in the segment
SELECTION_PROBLEM = 'regression'     # 'regression' or 'classification'
DO_PCA_PLOT = False                  # set True if we want PCA diagnostics

# 3. Simple segment builder
# Reuse for pass/rush/rec (or defense later).
def build_segment(df: pd.DataFrame,
                  position_filter: List[str],
                  drop_cols: List[str]) -> pd.DataFrame:
    seg = df[df['position_group'].isin(position_filter)].copy()
    seg = seg.drop(columns=[c for c in drop_cols if c in seg.columns], errors='ignore')
    seg = seg.fillna(0)
    return seg

# 4. PCA diagnostic (no plotting)
# Quick sense of dimensionality; optional.
def pca_top_vars(df_num: pd.DataFrame, var_target: float = 0.80, top_k: int = 25) -> List[str]:
    if df_num.shape[1] == 0:
        return []
    Xs = StandardScaler().fit_transform(df_num.fillna(0))
    p = PCA().fit(Xs)
    n = int(np.argmax(np.cumsum(p.explained_variance_ratio_) >= var_target) + 1)
    n = max(1, n)
    pr = PCA(n_components=n).fit(Xs)
    loadings = pd.DataFrame(pr.components_.T, index=df_num.columns)
    total = loadings.abs().sum(axis=1)
    return total.sort_values(ascending=False).head(min(top_k, len(total))).index.tolist()

# 5. Correlation prune
# Drop one of highly correlated pairs to reduce redundancy.
def correlation_prune(df: pd.DataFrame, thr: float = 0.90) -> pd.DataFrame:
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > thr)]
    return df.drop(columns=to_drop)

# 6. Stability selection (very compact)
# Get a robust top feature list with repeated RF.
def stability_select(X: pd.DataFrame,
                     y: pd.Series,
                     problem: str = "regression",
                     n_repeats: int = 5,
                     n_folds: int = 5,
                     n_estimators: int = 400,
                     max_depth: Optional[int] = None,
                     random_seed: int = 38,
                     top_k_each_run: Optional[int] = None,
                     missing_thresh: float = 0.40,
                     low_var_thresh: float = 1e-8,
                     corr_thresh: float = 0.90) -> pd.DataFrame:
    X = X.select_dtypes(include=[np.number])
    X = X.loc[:, X.isna().mean() <= missing_thresh]

    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)

    vt = VarianceThreshold(threshold=low_var_thresh)
    X_vt = pd.DataFrame(vt.fit_transform(X_imp),
                        columns=X_imp.columns[vt.get_support()],
                        index=X_imp.index)

    X_base = correlation_prune(X_vt, thr=corr_thresh)

    if X_base.shape[1] == 0:
        return pd.DataFrame(columns=["feature","stability","mean_importance"])

    if problem == "regression":
        RF = RandomForestRegressor
    else:
        RF = RandomForestClassifier

    feats = X_base.columns.tolist()
    if top_k_each_run is None:
        top_k_each_run = min(15, max(1, len(feats)))

    counts = defaultdict(int)
    sums = defaultdict(float)

    X_mat = X_base.values
    y_vec = y.values

    for r in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed + r)
        for fold_idx, (tr, va) in enumerate(kf.split(X_mat)):
            model = RF(n_estimators=n_estimators,
                       max_depth=max_depth,
                       random_state=random_seed + r*100 + fold_idx,
                       n_jobs=-1)
            model.fit(X_mat[tr], y_vec[tr])
            imps = model.feature_importances_
            order = np.argsort(imps)[::-1][:top_k_each_run]
            for j in order:
                counts[feats[j]] += 1
            for j, f in enumerate(feats):
                sums[f] += imps[j]

    runs = n_repeats * n_folds
    stab = (pd.DataFrame({
                "feature": feats,
                "stability": [counts[f]/runs for f in feats],
                "mean_importance": [sums[f]/runs for f in feats]
            })
            .sort_values(["stability","mean_importance"], ascending=False)
            .reset_index(drop=True))
    return stab

# 7. Minimal runner
# Do one pass (e.g., PASS segment), write out top features for modeling.
def main():
    # 7.1 Load data
    df = pd.read_csv(DATA_PATH)

    # 7.2 Define drops (paste your lists)
    pass_drop_columns = [...]  # <- your list here

    # 7.3 Build segment
    seg = build_segment(df, position_filter=['QB'], drop_cols=pass_drop_columns)

    # 7.4 PCA diagnostic shortlist (optional)
    if DO_PCA_PLOT:
        df_num = seg.select_dtypes(include=[np.number]).drop(columns=[c for c in ("result","total") if c in seg], errors="ignore")
        pca_vars = pca_top_vars(df_num, var_target=0.80, top_k=25)
        pd.Series(pca_vars, name="pca_top").to_csv("pca_top_vars_pass.csv", index=False)

    # 7.5 Stability selection shortlist
    if SELECTION_TARGET not in seg.columns:
        raise ValueError(f"{SELECTION_TARGET} not found in segment columns.")
    y = seg[SELECTION_TARGET].copy()
    X = seg.drop(columns=[SELECTION_TARGET])

    stab = stability_select(X, y, problem=SELECTION_PROBLEM,
                            n_repeats=5, n_folds=5, n_estimators=400)

    top25 = stab.head(min(25, stab.shape[0]))["feature"].tolist()

    # 7.6 Save outputs for the modeling file
    pd.Series(top25, name="top_features").to_csv("top25_pass.csv", index=False)
    stab.to_csv("stability_pass_full.csv", index=False)

    # 7.7 (Optional) Save cleaned matrix restricted to top features
    X_top = X[top25].copy()
    X_top.to_csv("X_pass_top25.csv", index=False)
    print("Saved: top25_pass.csv, stability_pass_full.csv, X_pass_top25.csv")

if __name__ == "__main__":
    main()

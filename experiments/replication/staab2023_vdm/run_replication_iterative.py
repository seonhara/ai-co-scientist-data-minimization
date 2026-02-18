# experiments/replication/staab2023_vdm/run_replication_iterative.py
# -*- coding: utf-8 -*-
"""
Replication (simplified) of Staab et al. 2023 vDM "Iterative" baseline
on UCI Adult.

Reference:
- "From Principle to Practice: Vertical Data Minimization for Machine Learning"
  (Staab et al., arXiv:2311.10500 / IEEE S&P 2024)

Key idea:
- Replace feature removal with vertical generalization (bucketing).
- Discrete: sort categories by LR weight; group into k buckets via DP to minimize within-bucket variance.
- Continuous: split into k-quantiles.
- Iteratively reduce buckets per attribute while keeping classifier error under threshold T.

Notes:
- The original paper also uses adversary error (privacy risk) in the attribute ordering.
  Here we provide a utility-driven approximation for Adult (no adversary evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Iterative minimizer hyperparams (paper: reduce buckets while error <= T) :contentReference[oaicite:3]{index=3}
BASE_K = 6          # initial buckets per attribute (k hyperparameter)
MIN_K = 2           # do not go below this
ERROR_THRESHOLD = 0.18  # max classification error allowed (1-accuracy) while compressing
MAX_PASSES = 1      # keep small (Adult is small experiment)

@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1": self.f1,
        }


def load_adult_raw() -> Tuple[pd.DataFrame, pd.Series]:
    adult = fetch_ucirepo(id=2)
    x = adult.data.features.copy()
    y = adult.data.targets.iloc[:, 0].copy()
    # target to 0/1
    y_bin = y.apply(lambda v: 1 if ">50K" in str(v) else 0)
    # clean missing "?"
    x = x.replace(r"^\s*\?\s*$", pd.NA, regex=True).dropna()
    y_bin = y_bin.loc[x.index]
    return x, y_bin


def build_lr_pipeline(cat_cols: List[str], num_cols: List[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(with_mean=False), num_cols),
        ]
    )
    clf = LogisticRegression(
        max_iter=4000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def dp_group_sorted_scores(scores: np.ndarray, k: int) -> List[Tuple[int, int]]:
    """
    Given scores sorted by category order, partition into k contiguous groups
    minimizing mean within-group variance (DP).
    Returns list of (start_idx, end_idx) inclusive.
    """
    n = len(scores)
    if k >= n:
        return [(i, i) for i in range(n)]

    # precompute variance cost for interval [i, j]
    cost = np.full((n, n), 0.0)
    for i in range(n):
        for j in range(i, n):
            seg = scores[i:j + 1]
            cost[i, j] = float(np.var(seg)) if len(seg) > 1 else 0.0

    dp = np.full((n, k), np.inf)
    prev = np.full((n, k), -1, dtype=int)

    # base: 1 group
    for j in range(n):
        dp[j, 0] = cost[0, j]

    # k groups
    for g in range(1, k):
        for j in range(g, n):
            best_val = np.inf
            best_i = -1
            for i in range(g - 1, j):
                val = dp[i, g - 1] + cost[i + 1, j]
                if val < best_val:
                    best_val = val
                    best_i = i
            dp[j, g] = best_val
            prev[j, g] = best_i

    # reconstruct
    groups: List[Tuple[int, int]] = []
    j = n - 1
    g = k - 1
    while g >= 0:
        i = prev[j, g]
        start = 0 if i == -1 else i + 1
        groups.append((start, j))
        j = i
        g -= 1
    groups.reverse()
    return groups


def make_discrete_bucket_map(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    col: str,
    k: int,
) -> Dict[object, int]:
    """
    Paper idea: fit LR; sort discrete values by LR weight; group into k buckets via DP. :contentReference[oaicite:4]{index=4}
    """
    # fit LR using only this single feature (one-hot)
    cat_cols, num_cols = [col], []
    pipe = build_lr_pipeline(cat_cols, num_cols)
    pipe.fit(x_train[[col]], y_train)

    enc: OneHotEncoder = pipe.named_steps["pre"].named_transformers_["cat"]
    clf: LogisticRegression = pipe.named_steps["clf"]

    cats = enc.categories_[0].tolist()
    # coef_ shape: (1, n_features)
    weights = clf.coef_.ravel()
    # one-hot columns correspond to cats in order
    cat_weights = np.array(weights[: len(cats)], dtype=float)

    # sort categories by weight
    order = np.argsort(cat_weights)
    cats_sorted = [cats[i] for i in order]
    scores_sorted = cat_weights[order]

    groups = dp_group_sorted_scores(scores_sorted, k=k)

    bucket_map: Dict[object, int] = {}
    for bucket_id, (s, e) in enumerate(groups):
        for idx in range(s, e + 1):
            bucket_map[cats_sorted[idx]] = bucket_id
    return bucket_map


def apply_generalization(
    x: pd.DataFrame,
    discrete_maps: Dict[str, Dict[object, int]],
    continuous_bins: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Return generalized dataframe where:
    - discrete cols replaced by bucket id (int)
    - continuous cols replaced by bin id (int)
    """
    xg = x.copy()
    for col, mp in discrete_maps.items():
        xg[col] = xg[col].map(mp).astype("Int64")

    for col, edges in continuous_bins.items():
        # bin labels: 0..k-1
        xg[col] = pd.cut(xg[col].astype(float), bins=edges, labels=False, include_lowest=True)
        xg[col] = xg[col].astype("Int64")

    # drop rows with NA bins (rare edge cases)
    xg = xg.dropna()
    return xg


def get_column_types(x: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in x.columns if x[c].dtype == "object"]
    num_cols = [c for c in x.columns if c not in cat_cols]
    return cat_cols, num_cols


def utility_eval_lr(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Metrics:
    cat_cols, num_cols = get_column_types(x_train)
    pipe = build_lr_pipeline(cat_cols, num_cols)
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    return compute_metrics(y_test.to_numpy(), pred)


def build_quantile_edges(series: pd.Series, k: int) -> np.ndarray:
    # edges length: k+1
    qs = np.linspace(0, 1, k + 1)
    edges = series.astype(float).quantile(qs).to_numpy()
    # ensure strictly increasing (fallback)
    edges = np.unique(edges)
    if len(edges) < 3:
        mn, mx = float(series.min()), float(series.max())
        edges = np.linspace(mn, mx, k + 1)
    return edges


def main() -> None:
    x, y = load_adult_raw()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # baseline (no generalization)
    base_metrics = utility_eval_lr(x_train, x_test, y_train, y_test)
    print("[Baseline] ", base_metrics.as_dict(), "Error=", round(1 - base_metrics.accuracy, 4))

    cat_cols, num_cols = get_column_types(x_train)

    # ordering approximation:
    # paper uses (Δclf - Δadv) to order attributes. :contentReference[oaicite:5]{index=5}
    # here: order by Δclf only (utility-driven), i.e., "remove attribute i" impact.
    deltas: List[Tuple[str, float]] = []
    for col in x_train.columns:
        xt_tr = x_train.drop(columns=[col])
        xt_te = x_test.drop(columns=[col])
        m = utility_eval_lr(xt_tr, xt_te, y_train, y_test)
        deltas.append((col, (1 - m.accuracy) - (1 - base_metrics.accuracy)))  # Δerror
    deltas.sort(key=lambda t: t[1])  # smaller Δerror first (safer to minimize)
    ordered_cols = [c for c, _ in deltas]
    print("[Order approx] first 5:", ordered_cols[:5])

    discrete_maps: Dict[str, Dict[object, int]] = {}
    continuous_bins: Dict[str, np.ndarray] = {}
    current_k: Dict[str, int] = {c: BASE_K for c in x_train.columns}

    rows = []
    xg_train = x_train.copy()
    xg_test = x_test.copy()

    for _ in range(MAX_PASSES):
        for col in ordered_cols:
            # initialize mapping/bins for this col at current k
            k = current_k[col]

            def rebuild_and_eval(k_try: int) -> Metrics:
                d_maps = dict(discrete_maps)
                c_bins = dict(continuous_bins)

                if col in cat_cols:
                    d_maps[col] = make_discrete_bucket_map(x_train, y_train, col, k_try)
                else:
                    c_bins[col] = build_quantile_edges(x_train[col], k_try)

                tr = apply_generalization(x_train, d_maps, c_bins)
                te = apply_generalization(x_test, d_maps, c_bins)

                # align indices after dropna
                common_tr = tr.index.intersection(y_train.index)
                common_te = te.index.intersection(y_test.index)

                m = utility_eval_lr(tr.loc[common_tr], te.loc[common_te], y_train.loc[common_tr], y_test.loc[common_te])
                return m

            # create initial generalization at current k if missing
            if col not in discrete_maps and col in cat_cols:
                discrete_maps[col] = make_discrete_bucket_map(x_train, y_train, col, k)
            if col not in continuous_bins and col in num_cols:
                continuous_bins[col] = build_quantile_edges(x_train[col], k)

            # now try reducing k while error <= threshold T :contentReference[oaicite:6]{index=6}
            while current_k[col] > MIN_K:
                k_try = current_k[col] - 1
                m_try = rebuild_and_eval(k_try)
                err_try = 1 - m_try.accuracy
                if err_try <= ERROR_THRESHOLD:
                    # accept reduction
                    current_k[col] = k_try
                    if col in cat_cols:
                        discrete_maps[col] = make_discrete_bucket_map(x_train, y_train, col, k_try)
                    else:
                        continuous_bins[col] = build_quantile_edges(x_train[col], k_try)
                else:
                    break

            # evaluate current state for logging
            xg_train = apply_generalization(x_train, discrete_maps, continuous_bins)
            xg_test = apply_generalization(x_test, discrete_maps, continuous_bins)

            common_tr = xg_train.index.intersection(y_train.index)
            common_te = xg_test.index.intersection(y_test.index)

            m_now = utility_eval_lr(
                xg_train.loc[common_tr], xg_test.loc[common_te],
                y_train.loc[common_tr], y_test.loc[common_te]
            )
            rows.append({
                "Step": len(rows),
                "Column": col,
                "k_after": current_k[col],
                **m_now.as_dict(),
                "Error": round(1 - m_now.accuracy, 4),
            })
            print(f"[Step {len(rows)-1}] {col} k={current_k[col]} metrics={m_now.as_dict()}")

    out = pd.DataFrame(rows)
    out.to_csv("experiments/replication/staab2023_vdm/results_staab2023_iterative_adult.csv", index=False)
    print("[Saved] experiments/replication/staab2023_vdm/results_staab2023_iterative_adult.csv")


if __name__ == "__main__":
    main()

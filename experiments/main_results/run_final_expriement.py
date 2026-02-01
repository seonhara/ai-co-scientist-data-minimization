# -*- coding: utf-8 -*-
"""
Main entry point for reproducing all figures and tables reported in the paper.

Usage:
    python run_final_experiment.py
"""

"""
Adult Dataset - Data Minimization vs Performance (Portfolio-ready version)

Models:
- Logistic Regression
- Random Forest
- Gradient Boosting

Metrics:
- Accuracy / Precision / Recall / F1 (binary classification)

Strategies:
- H: Heuristic baseline
- P/C/R: LLM-derived strategies (prefix-based feature removal)

This script is designed to be:
- reproducible (fixed random seeds)
- collaboration-friendly (clear configuration, logging, CLI)
- maintainable (typed functions, validation, modular structure)
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    random_state: int = 42
    test_size: float = 0.2
    use_rf_class_weight_balanced: bool = False
    strategies: Tuple[str, ...] = ("H", "P", "C", "R")
    levels: Tuple[int, ...] = (0, 1, 2, 3)  # 0..3
    output_dir: Path = Path("results")
    dataset_ucimlrepo_id: int = 2


# 전략 정의는 “연구 설계의 핵심”이므로 config처럼 상수로 분리해둠.
STRATEGY_GROUPS: Dict[str, List[List[str]]] = {
    "H": [
        ["age"],
        ["education_", "education-num"],
        ["occupation_"],
    ],
    "P": [  # 대표: GPT
        ["fnlwgt", "native-country_", "race_", "sex_"],
        ["marital-status_", "relationship_"],
        ["occupation_", "workclass_"],
    ],
    "C": [  # 대표: Copilot
        ["fnlwgt", "native-country_", "race_", "sex_", "marital-status_"],
        ["occupation_", "relationship_", "workclass_", "education_"],
        ["capital-gain", "capital-loss", "age", "education-num"],
    ],
    "R": [  # 대표: GPT
        ["fnlwgt", "native-country_"],
        ["race_", "sex_"],
        ["marital-status_", "relationship_"],
    ],
}


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def validate_strategy_groups(strategy_groups: Dict[str, List[List[str]]]) -> None:
    """Fail fast if strategy definition is malformed."""
    if not strategy_groups:
        raise ValueError("STRATEGY_GROUPS is empty.")

    for strategy, groups in strategy_groups.items():
        if not isinstance(strategy, str) or not strategy:
            raise ValueError("Strategy key must be a non-empty string.")
        if not isinstance(groups, list) or len(groups) != 3:
            raise ValueError(
                f"Strategy '{strategy}' must have exactly 3 groups (levels 1~3)."
            )
        for idx, group in enumerate(groups, start=1):
            if not isinstance(group, list) or not group:
                raise ValueError(
                    f"Strategy '{strategy}' group#{idx} must be a non-empty list."
                )
            if not all(isinstance(p, str) and p for p in group):
                raise ValueError(
                    f"Strategy '{strategy}' group#{idx} contains invalid prefix."
                )


def load_adult_dataset_ucimlrepo(dataset_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    adult = fetch_ucirepo(id=dataset_id)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()
    LOGGER.info("Loaded Adult dataset (ucimlrepo id=%s). Raw features=%d", dataset_id, X.shape[1])
    return X, y


def preprocess_adult(
    X: pd.DataFrame,
    y: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    - Convert target to binary: >50K -> 1 else 0
    - Drop rows with missing values ('?')
    - One-hot encode categorical variables
    """
    y_bin = y.iloc[:, 0].apply(lambda v: 1 if ">50K" in str(v) else 0).astype(int)

    X_clean = X.replace(r"^\s*\?\s*$", pd.NA, regex=True).dropna()
    y_clean = y_bin.loc[X_clean.index]

    X_encoded = pd.get_dummies(X_clean, drop_first=True)
    LOGGER.info("Preprocessed: encoded_shape=%s, pos_rate=%.4f",
                X_encoded.shape, float(y_clean.mean()))
    return X_encoded, y_clean


def prefixes_to_remove(strategy: str, level: int) -> List[str]:
    """
    level:
      0: remove nothing
      1: remove group 1
      2: remove group 1 + group 2
      3: remove group 1 + group 2 + group 3
    """
    if strategy not in STRATEGY_GROUPS:
        raise ValueError(f"Unknown strategy '{strategy}'. Allowed={list(STRATEGY_GROUPS.keys())}")
    if level not in (0, 1, 2, 3):
        raise ValueError("level must be one of {0,1,2,3}")

    remove: List[str] = []
    for i in range(level):
        remove.extend(STRATEGY_GROUPS[strategy][i])
    return remove


def columns_matching_prefixes(columns: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    """
    Return a stable (deterministic) list of columns to remove:
    - exact match if prefix equals a column name
    - else prefix-match for one-hot encoded columns
    """
    col_set = set(columns)
    to_remove: List[str] = []

    for prefix in prefixes:
        if prefix in col_set:
            to_remove.append(prefix)
        else:
            to_remove.extend([c for c in columns if c.startswith(prefix)])

    # unique while preserving order
    seen = set()
    deduped = []
    for c in to_remove:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def apply_minimization(df: pd.DataFrame, strategy: str, level: int) -> pd.DataFrame:
    prefixes = prefixes_to_remove(strategy=strategy, level=level)
    if not prefixes:
        return df

    remove_cols = columns_matching_prefixes(df.columns.tolist(), prefixes)
    reduced = df.drop(columns=remove_cols, errors="ignore")

    if reduced.shape[1] == 0:
        raise ValueError(
            f"[EMPTY FEATURES] strategy={strategy}, level={level}. "
            "Your prefixes removed all columns. Check STRATEGY_GROUPS."
        )
    return reduced


def build_models(cfg: ExperimentConfig):
    logistic = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(
                max_iter=3000,
                random_state=cfg.random_state,
                solver="liblinear",
            )),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=cfg.random_state,
        n_jobs=-1,
        class_weight="balanced" if cfg.use_rf_class_weight_balanced else None,
    )

    gb = GradientBoostingClassifier(random_state=cfg.random_state)

    return {
        "LogisticRegression": logistic,
        "RandomForest": rf,
        "GradientBoosting": gb,
    }


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_experiments(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    models = build_models(cfg)
    rows: List[Dict[str, object]] = []

    for strategy in cfg.strategies:
        for level in cfg.levels:
            X_train_lvl = apply_minimization(X_train, strategy=strategy, level=level)
            X_test_lvl = apply_minimization(X_test, strategy=strategy, level=level)

            LOGGER.info("strategy=%s level=%d num_features=%d", strategy, level, X_train_lvl.shape[1])

            for model_name, model in models.items():
                model.fit(X_train_lvl, y_train)
                y_pred = model.predict(X_test_lvl)

                row: Dict[str, object] = {
                    "Strategy": strategy,
                    "Level": level,
                    "Num_Features": int(X_train_lvl.shape[1]),
                    "Model": model_name,
                }
                row.update(compute_metrics(y_test, y_pred))
                rows.append(row)

    results = pd.DataFrame(rows)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
    results[metric_cols] = results[metric_cols].round(4)
    return results


def make_delta_table(results_df: pd.DataFrame, metric: str = "F1") -> pd.DataFrame:
    base = (
        results_df[results_df["Level"] == 0][["Strategy", "Model", metric]]
        .rename(columns={metric: f"{metric}_L0"})
    )
    merged = results_df.merge(base, on=["Strategy", "Model"], how="left")
    merged[f"Delta_{metric}"] = (merged[metric] - merged[f"{metric}_L0"]).round(4)
    return merged.sort_values(["Strategy", "Model", "Level"])


def save_tables(results_df: pd.DataFrame, delta_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "results_level_model_metrics.csv"
    delta_path = out_dir / "results_delta_f1_from_level0.csv"

    results_df.to_csv(metrics_path, index=False)
    delta_df.to_csv(delta_path, index=False)

    LOGGER.info("Saved: %s", metrics_path)
    LOGGER.info("Saved: %s", delta_path)


def plot_tradeoff(results_df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / f"tradeoff_{metric.lower()}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for strategy in sorted(results_df["Strategy"].unique()):
        df_s = results_df[results_df["Strategy"] == strategy]
        out_path = fig_dir / f"tradeoff_{metric.lower()}_strategy_{strategy}.png"

        plt.figure()
        for model_name in sorted(df_s["Model"].unique()):
            df_m = df_s[df_s["Model"] == model_name].sort_values("Level")
            plt.plot(df_m["Level"], df_m[metric], marker="o", label=model_name)

        plt.xlabel("Data Minimization Level")
        plt.ylabel(metric)
        plt.title(f"Strategy {strategy}: {metric} vs Data Minimization Level")
        plt.xticks([0, 1, 2, 3])
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        LOGGER.info("Saved: %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adult Dataset minimization experiments.")
    parser.add_argument("--output-dir", default="results", help="Directory to save outputs.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, ...).")
    parser.add_argument("--rf-balanced", action="store_true", help="Use class_weight='balanced' in RF.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    validate_strategy_groups(STRATEGY_GROUPS)

    cfg = ExperimentConfig(
        random_state=args.seed,
        test_size=args.test_size,
        use_rf_class_weight_balanced=bool(args.rf_balanced),
        output_dir=Path(args.output_dir),
    )

    X_raw, y_raw = load_adult_dataset_ucimlrepo(cfg.dataset_ucimlrepo_id)
    X_encoded, y_clean = preprocess_adult(X_raw, y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_clean,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y_clean,
    )

    results_df = run_experiments(X_train, X_test, y_train, y_test, cfg)
    delta_df = make_delta_table(results_df, metric="F1")

    save_tables(results_df, delta_df, cfg.output_dir)
    plot_tradeoff(results_df, metric="Accuracy", out_dir=cfg.output_dir)
    plot_tradeoff(results_df, metric="F1", out_dir=cfg.output_dir)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()

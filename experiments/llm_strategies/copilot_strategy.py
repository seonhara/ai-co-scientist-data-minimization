# -*- coding: utf-8 -*-
"""
Adult Dataset - Data Minimization vs Performance (Extended Research Version)

Models: Logistic Regression / Random Forest / Gradient Boosting
Metrics: Accuracy / Precision / Recall / F1
Strategies:
  H: Heuristic baseline
  P/C/R: LLM-derived strategies (prefix-based feature removal)
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt


RANDOM_STATE = 42
TEST_SIZE = 0.2

# (C) Optional: handle class imbalance in RF (set True/False)
USE_RF_CLASS_WEIGHT_BALANCED = False

# (A) Move strategy definitions out of function (global constant)
STRATEGY_GROUPS = {
    # P = Privacy-first (대표: GPT)
    "P": [
        ["fnlwgt", "native-country_", "race_", "sex_"],
        ["marital-status_", "relationship_"],
        ["occupation_", "workclass_"],
    ],

    # C = Conservative (대표: Copilot)
    "C": [
        ["fnlwgt", "native-country_", "race_", "sex_", "marital-status_"],
        ["occupation_", "relationship_", "workclass_", "education_"],
        ["capital-gain", "capital-loss", "hours-per-week", "age", "education-num"],
    ],

    # R = Robust (대표: GPT)
    "R": [
        ["fnlwgt", "native-country_"],
        ["race_", "sex_"],
        ["marital-status_", "relationship_"],
    ],
}


def load_adult_ucimlrepo():
    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()
    print("Raw columns:")
    print(X.columns.tolist())
    return X, y


def preprocess(X: pd.DataFrame, y: pd.DataFrame):
    # target to 0/1
    y_bin = y.iloc[:, 0].apply(lambda x: 1 if ">50K" in str(x) else 0)

    # handle missing values: '?' or ' ?'
    X_clean = X.replace(r"^\s*\?\s*$", pd.NA, regex=True).dropna()
    y_clean = y_bin.loc[X_clean.index]

    # one-hot encoding (categoricals only)
    X_encoded = pd.get_dummies(X_clean, drop_first=True)
    return X_encoded, y_clean


def remove_features_by_strategy_level(df: pd.DataFrame, strategy: str, level: int) -> pd.DataFrame:
    """
    level:
      0: remove nothing
      1: remove group 1
      2: remove group 1 + group 2
      3: remove group 1 + group 2 + group 3
    """
    if strategy not in STRATEGY_GROUPS:
        raise ValueError(f"Unknown strategy: {strategy}. Use one of {list(STRATEGY_GROUPS.keys())}")

    if level not in (0, 1, 2, 3):
        raise ValueError("level must be one of {0, 1, 2, 3}")

    cols = df.columns

    remove_prefixes = []
    for i in range(1, level + 1):
        remove_prefixes.extend(STRATEGY_GROUPS[strategy][i - 1])

    remove_cols = []
    for prefix in remove_prefixes:
        if prefix in cols:
            # exact match (fnlwgt, education-num, hours-per-week, capital-gain/loss, age, ...)
            remove_cols.append(prefix)
        else:
            # prefix match for one-hot columns
            remove_cols.extend([c for c in cols if c.startswith(prefix)])

    remove_cols = list(set(remove_cols))
    return df.drop(columns=remove_cols, errors="ignore")


def get_models():
    # (B) Specify solver for reproducibility/stability
    logistic = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )),
    ])

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced" if USE_RF_CLASS_WEIGHT_BALANCED else None,
    )

    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)

    return {
        "LogisticRegression": logistic,
        "RandomForest": rf,
        "GradientBoosting": gb,
    }


def evaluate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_experiments(X_train, X_test, y_train, y_test, strategies=("H", "P", "C", "R"), levels=4):
    models = get_models()
    results = []

    for strategy in strategies:
        for level in range(levels):
            X_train_lvl = remove_features_by_strategy_level(X_train, strategy, level)
            X_test_lvl = remove_features_by_strategy_level(X_test, strategy, level)

            # print once per (strategy, level), not per model
            print(strategy, level, X_train_lvl.shape[1])

            for model_name, model in models.items():
                model.fit(X_train_lvl, y_train)
                y_pred = model.predict(X_test_lvl)

                row = {
                    "Strategy": strategy,
                    "Level": level,
                    "Num_Features": X_train_lvl.shape[1],
                    "Model": model_name,
                }
                row.update(evaluate_metrics(y_test, y_pred))
                results.append(row)

    results_df = pd.DataFrame(results)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
    results_df[metric_cols] = results_df[metric_cols].round(4)
    return results_df


def save_results(results_df: pd.DataFrame, csv_path="results_level_model_metrics.csv"):
    results_df.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")


def plot_tradeoff_by_strategy(results_df: pd.DataFrame, metric="F1"):
    for strategy in results_df["Strategy"].unique():
        out_path = f"tradeoff_{metric.lower()}_strategy_{strategy}.png"
        plt.figure()

        df_s = results_df[results_df["Strategy"] == strategy]
        for model_name in df_s["Model"].unique():
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
        print(f"[Saved] {out_path}")


def main():
    X, y = load_adult_ucimlrepo()
    X_encoded, y_clean = preprocess(X, y)

    print("Encoded shape:", X_encoded.shape)
    print("Target distribution:", y_clean.value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_clean,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_clean,
    )

    results_df = run_experiments(X_train, X_test, y_train, y_test, levels=4)
    print(results_df)

    save_results(results_df)
    plot_tradeoff_by_strategy(results_df, metric="Accuracy")
    plot_tradeoff_by_strategy(results_df, metric="F1")


if __name__ == "__main__":
    main()

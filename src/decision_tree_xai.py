
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


FEATURES = [
    "icu_beds",
    "general_beds",
    "doctors",
    "nurses",
]

LABEL = "acceptable_policy"
DEFAULT_MEAN_WAIT_TARGET = 20.0
DEFAULT_HIGH_RISK_WAIT_TARGET = 10.0


def load_acceptability_targets(path: Path) -> tuple[float, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"GA result is required for current XAI targets: {path}"
        )

    result = json.loads(path.read_text(encoding="utf-8"))
    objectives = result.get("applied_objectives", {})

    mean_wait = float(
        objectives.get(
            "acceptable_mean_wait_min",
            DEFAULT_MEAN_WAIT_TARGET,
        )
    )
    high_risk_wait = float(
        objectives.get(
            "acceptable_high_risk_wait_min",
            DEFAULT_HIGH_RISK_WAIT_TARGET,
        )
    )

    if mean_wait <= 0 or high_risk_wait <= 0:
        raise ValueError("GA acceptability targets must be greater than zero.")

    return mean_wait, high_risk_wait


def load_ga_history(
    path: Path,
    mean_wait_target: float,
    high_risk_wait_target: float,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = FEATURES + [
        "mean_waiting_time_min",
        "high_risk_mean_waiting_time_min",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"GA history is missing required columns: {missing}")

    df = df.dropna(subset=required).copy()

    # Explainability target:
    # acceptable if both waiting times meet the objectives from this GA run.
    df[LABEL] = (
        (df["mean_waiting_time_min"] <= mean_wait_target)
        & (df["high_risk_mean_waiting_time_min"] <= high_risk_wait_target)
    ).astype(int)

    return df


def train_tree(
    df: pd.DataFrame,
    max_depth: int = 4,
    min_samples_leaf: int = 4,
    seed: int = 42,
):
    X = df[FEATURES].copy()
    y = df[LABEL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y if y.value_counts().min() >= 2 else None,
    )

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
    )
    tree.fit(X_train, y_train)

    pred = tree.predict(X_test)

    metrics = {
        "rows_used": int(len(df)),
        "acceptable_policies": int(y.sum()),
        "unacceptable_policies": int((1 - y).sum()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "tree_depth": int(tree.get_depth()),
        "leaf_count": int(tree.get_n_leaves()),
    }

    return tree, metrics, X_test, y_test, pred


def best_policy_row(df: pd.DataFrame) -> pd.Series:
    # Prefer highest-fitness acceptable policy, otherwise highest fitness overall.
    acceptable = df[df[LABEL] == 1]

    if not acceptable.empty and "fitness" in df.columns:
        return acceptable.loc[acceptable["fitness"].idxmax()]

    if "fitness" in df.columns:
        return df.loc[df["fitness"].idxmax()]

    return df.iloc[0]


def optimized_policy_row(df: pd.DataFrame, ga_result: dict) -> pd.Series:
    """Return the GA recommendation itself, not a different acceptable row."""
    policy = ga_result.get("best_policy", {})
    if not all(feature in policy for feature in FEATURES):
        return best_policy_row(df)

    matches = pd.Series(True, index=df.index)
    for feature in FEATURES:
        matches &= pd.to_numeric(df[feature], errors="coerce").eq(
            float(policy[feature])
        )

    candidates = df[matches]
    if candidates.empty:
        raise ValueError(
            "The optimized policy in best_policy.json is absent from GA history."
        )
    if "fitness" in candidates.columns:
        return candidates.loc[candidates["fitness"].idxmax()]
    return candidates.iloc[0]


def decision_path_rules(
    tree: DecisionTreeClassifier,
    policy: pd.Series,
) -> list[str]:
    X_one = pd.DataFrame(
        [{f: float(policy[f]) for f in FEATURES}],
        columns=FEATURES,
    )

    node_indicator = tree.decision_path(X_one)
    leaf_id = tree.apply(X_one)[0]

    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    rules = []

    node_ids = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]

    for node_id in node_ids:
        if node_id == leaf_id:
            continue

        feature_idx = feature[node_id]
        if feature_idx < 0:
            continue

        name = FEATURES[feature_idx]
        actual = float(policy[name])
        threshold_value = float(threshold[node_id])

        operator = "<=" if actual <= threshold_value else ">"
        rules.append(
            f"{name} {operator} {threshold_value:.2f} "
            f"(actual: {actual:.2f})"
        )

    return rules


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decision Tree classifier for explainable GA policy quality."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/genetic_algorithm/ga_history.csv"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/decision_tree_xai"),
    )
    parser.add_argument(
        "--ga-result",
        type=Path,
        default=Path("results/genetic_algorithm/best_policy.json"),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/decision_tree_xai.joblib"),
    )
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=4)

    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    mean_wait_target, high_risk_wait_target = load_acceptability_targets(
        args.ga_result
    )
    ga_result = json.loads(args.ga_result.read_text(encoding="utf-8"))

    print(f"Loading GA history: {args.input}")
    print(
        "Using current GA acceptability targets: "
        f"mean wait <= {mean_wait_target:.2f} min, "
        f"high-risk wait <= {high_risk_wait_target:.2f} min"
    )
    df = load_ga_history(
        args.input,
        mean_wait_target,
        high_risk_wait_target,
    )

    print("Training interpretable Decision Tree classifier...")
    tree, metrics, X_test, y_test, pred = train_tree(
        df,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )

    best = optimized_policy_row(df, ga_result)
    path_rules = decision_path_rules(tree, best)

    best_input = pd.DataFrame(
        [{f: float(best[f]) for f in FEATURES}],
        columns=FEATURES,
    )
    predicted_class = int(tree.predict(best_input)[0])
    probabilities = tree.predict_proba(best_input)[0]
    class_probabilities = dict(zip(tree.classes_, probabilities))
    predicted_probability = float(class_probabilities.get(1, 0.0))

    summary = {
        "optimized_policy": {
            "icu_beds": int(best["icu_beds"]),
            "general_beds": int(best["general_beds"]),
            "doctors": int(best["doctors"]),
            "nurses": int(best["nurses"]),
        },
        "actual_policy_class": (
            "acceptable" if int(best[LABEL]) == 1 else "unacceptable"
        ),
        "decision_tree_class": (
            "acceptable" if predicted_class == 1 else "unacceptable"
        ),
        "decision_tree_probability_acceptable": predicted_probability,
        "decision_path_rules": path_rules,
        "acceptability_definition": {
            "mean_waiting_time_min_max": mean_wait_target,
            "high_risk_mean_waiting_time_min_max": high_risk_wait_target,
            "source": "latest_ga_applied_objectives",
        },
    }

    rules_text = export_text(
        tree,
        feature_names=FEATURES,
        decimals=2,
    )

    joblib.dump(tree, args.model_out)

    with open(
        args.results_dir / "decision_tree_rules.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(rules_text)

    with open(
        args.results_dir / "tree_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    with open(
        args.results_dir / "best_policy_explanation.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    best.to_frame().T.to_csv(
        args.results_dir / "best_policy_row.csv",
        index=False,
    )

    pd.DataFrame(
        classification_report(
            y_test,
            pred,
            labels=[0, 1],
            target_names=["Unacceptable", "Acceptable"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(
        args.results_dir / "classification_report.csv"
    )

    pd.DataFrame(
        confusion_matrix(y_test, pred, labels=[0, 1]),
        index=["actual_unacceptable", "actual_acceptable"],
        columns=["pred_unacceptable", "pred_acceptable"],
    ).to_csv(args.results_dir / "confusion_matrix.csv")

    print("\nDECISION TREE XAI COMPLETE")
    print("\nTree quality:")
    print(json.dumps(metrics, indent=2))

    print("\nBest policy:")
    print(json.dumps(summary["optimized_policy"], indent=2))

    print("\nDecision path:")
    for rule in path_rules:
        print(f"- {rule}")

    print(
        "\nTree classification:",
        summary["decision_tree_class"],
        f"(P acceptable={predicted_probability:.3f})",
    )

    print("\nFull extracted rules:")
    print(rules_text)

    print(f"\nModel saved to: {args.model_out}")
    print(f"Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()

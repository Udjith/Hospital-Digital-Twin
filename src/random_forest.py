
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 42
TARGET = "high_resource_need"

# Keep features that are available before or around admission / early ICU care.
# Avoid direct outcome variables such as actual LOS, mortality, discharge status,
# and fields used to construct the target.
CANDIDATE_FEATURES = [
    # demographics / admission
    "age",
    "gender",
    "ethnicity",
    "admissionheight",
    "admissionweight",
    "unittype",
    "unitadmitsource",
    "unitstaytype",
    "hospitaladmitsource",
    "primary_admission_dx",
    "primary_diagnosis",

    # early severity / physiology
    "aps_heartrate",
    "aps_meanbp",
    "aps_respiratoryrate",
    "aps_temperature",
    "aps_sodium",
    "aps_creatinine",
    "aps_glucose",
    "apache_apachescore",
    "apache_acutephysiologyscore",
    "apache_predictedicumortality",
    "apache_predictedhospitalmortality",

    # summarized vitals / labs
    "vital_heartrate_mean",
    "vital_sao2_mean",
    "vital_respiration_mean",
    "lab_glucose_mean",
    "lab_creatinine_mean",
    "lab_sodium_mean",
    "lab_potassium_mean",

    # operational context, when present
    "arrival_hour",
    "arrival_dayofweek",
    "department",
    "available_icu_beds",
    "available_general_beds",
    "available_doctors",
    "available_nurses",
]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing from {path}")

    # Keep only rows with a valid binary target.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    if df.empty:
        raise ValueError("No valid target rows remain after filtering.")

    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    available = [c for c in CANDIDATE_FEATURES if c in df.columns]

    if not available:
        raise ValueError("None of the configured Random Forest features exist in the dataset.")

    X = df[available].copy()
    y = df[TARGET].copy()

    # Remove columns that are effectively empty.
    usable = [c for c in X.columns if X[c].notna().sum() > 10]
    X = X[usable]

    return X, y, usable


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = [
        c for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c])
    ]
    categorical_features = [
        c for c in X.columns
        if c not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(len(model.feature_importances_))])

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importance.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest for high resource need prediction.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/synthetic/augmented_dataset.csv"),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/random_forest_pipeline.joblib"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/random_forest"),
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.input}")
    df = load_dataset(args.input)

    X, y, features = select_features(df)

    print(f"Rows used: {len(df):,}")
    print(f"Features used: {len(features)}")
    print(f"Positive class rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    pipeline = build_pipeline(X)

    print("Training Random Forest...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features_used": features,
    }

    print("\nMetrics:")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{key}: {metrics[key]:.4f}")

    # Save model.
    joblib.dump(pipeline, args.model_out)

    # Save predictions.
    pred_df = X_test.copy()
    pred_df["actual_high_resource_need"] = y_test.to_numpy()
    pred_df["predicted_high_resource_need"] = y_pred
    pred_df["predicted_probability"] = y_prob
    pred_df.to_csv(args.results_dir / "test_predictions.csv", index=False)

    # Save metrics and reports.
    with open(args.results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Low resource", "High resource"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        args.results_dir / "classification_report.csv"
    )

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(
        cm,
        index=["actual_low", "actual_high"],
        columns=["pred_low", "pred_high"],
    ).to_csv(args.results_dir / "confusion_matrix.csv")

    importance = get_feature_importance(pipeline)
    importance.to_csv(args.results_dir / "feature_importance.csv", index=False)

    print(f"\nModel saved: {args.model_out}")
    print(f"Results saved: {args.results_dir}")
    print("\nTop 10 model features:")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

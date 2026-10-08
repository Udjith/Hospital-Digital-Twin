
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _sample_numeric(
    series: pd.Series,
    n: int,
    rng: np.random.Generator,
    jitter_fraction: float = 0.03,
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
) -> np.ndarray:
    """
    Bootstrap a numeric column from observed values and add small jitter.
    This preserves the empirical distribution while avoiding exact copies.
    """
    s = _safe_numeric(series).dropna()
    if s.empty:
        return np.full(n, np.nan)

    base = rng.choice(s.to_numpy(dtype=float), size=n, replace=True)

    q_low, q_high = s.quantile(list(clip_quantiles)).to_numpy(dtype=float)
    std = float(s.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        std = max(abs(float(s.mean())) * 0.01, 1e-6)

    noise = rng.normal(0, std * jitter_fraction, size=n)
    out = base + noise
    out = np.clip(out, q_low, q_high)
    return out


def _sample_categorical(series: pd.Series, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample categorical values using the observed category frequencies.
    """
    s = series.astype("string").replace({"<NA>": pd.NA}).dropna()
    if s.empty:
        return np.array([pd.NA] * n, dtype=object)

    probs = s.value_counts(normalize=True)
    return rng.choice(probs.index.to_numpy(dtype=object), size=n, p=probs.to_numpy())


def _generate_arrival_times(n: int, rng: np.random.Generator) -> pd.Series:
    """
    Generate one year of synthetic arrivals with more arrivals during daytime.
    """
    start = pd.Timestamp("2026-01-01 00:00:00")
    days = rng.integers(0, 365, size=n)

    # Mildly realistic hourly arrival pattern: daytime/evening > overnight.
    hours = np.arange(24)
    weights = np.array([
        0.55, 0.45, 0.40, 0.38, 0.42, 0.55,
        0.75, 0.95, 1.15, 1.25, 1.30, 1.30,
        1.25, 1.25, 1.30, 1.35, 1.40, 1.45,
        1.40, 1.30, 1.15, 0.95, 0.80, 0.65
    ], dtype=float)
    weights = weights / weights.sum()

    sampled_hours = rng.choice(hours, size=n, p=weights)
    minutes = rng.integers(0, 60, size=n)

    timestamps = (
        start
        + pd.to_timedelta(days, unit="D")
        + pd.to_timedelta(sampled_hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
    )
    return pd.Series(timestamps)


def generate_synthetic_dataset(
    real_df: pd.DataFrame,
    n_rows: int = 7500,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a single-hospital synthetic dataset based on empirical patterns
    from the processed eICU master table.

    The generated dataset is intended for project simulation / ML experiments,
    not clinical use.
    """
    rng = np.random.default_rng(seed)
    synth = pd.DataFrame(index=np.arange(n_rows))

    # ------------------------------------------------------------------
    # Identifiers / single-hospital context
    # ------------------------------------------------------------------
    synth["synthetic_patient_id"] = np.arange(1, n_rows + 1)
    synth["hospital_id"] = 1
    synth["hospital_name"] = "Project Synthetic Hospital"

    # ------------------------------------------------------------------
    # Patient / clinical features
    # ------------------------------------------------------------------
    categorical_cols = [
        "gender",
        "ethnicity",
        "unittype",
        "unitadmitsource",
        "unitstaytype",
        "hospitaladmitsource",
        "primary_admission_dx",
        "primary_diagnosis",
        "numbedscategory",
        "teachingstatus",
        "region",
    ]

    numeric_cols = [
        "age",
        "admissionheight",
        "admissionweight",
        "icu_los_hours",
        "hospital_los_hours",
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
        "vital_heartrate_mean",
        "vital_sao2_mean",
        "vital_respiration_mean",
        "lab_glucose_mean",
        "lab_creatinine_mean",
        "lab_sodium_mean",
        "lab_potassium_mean",
        "treatment_count",
        "provider_record_count",
    ]

    for col in categorical_cols:
        if col in real_df.columns:
            synth[col] = _sample_categorical(real_df[col], n_rows, rng)

    for col in numeric_cols:
        if col in real_df.columns:
            synth[col] = _sample_numeric(real_df[col], n_rows, rng)

    # ------------------------------------------------------------------
    # Resource-need target
    # ------------------------------------------------------------------
    # IMPORTANT: Generate the target FROM early clinical features.
    # Do not generate the target first and then modify predictor features,
    # because that would create artificial target leakage.
    risk = np.zeros(n_rows, dtype=float)

    def add_zscore(col: str, weight: float, invert: bool = False) -> None:
        nonlocal risk
        if col not in synth.columns:
            return
        x = pd.to_numeric(synth[col], errors="coerce")
        median = float(x.median()) if x.notna().any() else 0.0
        x = x.fillna(median)
        std = float(x.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        z = (x.to_numpy(dtype=float) - float(x.mean())) / std
        risk += (-weight if invert else weight) * z

    add_zscore("apache_apachescore", 0.90)
    add_zscore("apache_acutephysiologyscore", 0.65)
    add_zscore("apache_predictedicumortality", 0.75)
    add_zscore("vital_heartrate_mean", 0.25)
    add_zscore("vital_respiration_mean", 0.25)
    add_zscore("lab_creatinine_mean", 0.25)
    add_zscore("age", 0.20)
    add_zscore("vital_sao2_mean", 0.35, invert=True)

    # Add noise so the model has a realistic, imperfect prediction problem.
    risk += rng.normal(0, 1.25, size=n_rows)

    # Calibrate prevalence to roughly match the real eICU target distribution.
    if "high_resource_need" in real_df.columns:
        real_target = pd.to_numeric(real_df["high_resource_need"], errors="coerce").dropna()
        p_high = float(real_target.mean()) if not real_target.empty else 0.40
    else:
        p_high = 0.40

    threshold = float(np.quantile(risk, 1 - p_high))
    synth["high_resource_need"] = (risk >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Operational / patient-flow variables
    # ------------------------------------------------------------------
    synth["arrival_time"] = _generate_arrival_times(n_rows, rng)
    synth["arrival_hour"] = synth["arrival_time"].dt.hour
    synth["arrival_dayofweek"] = synth["arrival_time"].dt.dayofweek

    # Priority: high-resource patients tend to receive higher urgency.
    base_priority = rng.choice([1, 2, 3, 4, 5], size=n_rows, p=[0.10, 0.20, 0.35, 0.25, 0.10])
    high_mask = synth["high_resource_need"].eq(1).to_numpy()
    base_priority[high_mask] = np.maximum(
        1,
        base_priority[high_mask] - rng.choice([0, 1, 2], size=high_mask.sum(), p=[0.25, 0.50, 0.25])
    )
    synth["priority_level"] = base_priority

    # Department assignment.
    unit_series = synth.get("unittype", pd.Series(["ICU"] * n_rows))
    unit_text = unit_series.astype(str).str.lower()
    synth["department"] = np.where(
        unit_text.str.contains("card"),
        "Cardiac ICU",
        np.where(
            unit_text.str.contains("neuro"),
            "Neuro ICU",
            np.where(
                unit_text.str.contains("surg"),
                "Surgical ICU",
                "Medical ICU"
            ),
        ),
    )

    # Resource need.
    synth["bed_requirement"] = np.where(
        synth["high_resource_need"].eq(1),
        "ICU",
        rng.choice(["ICU", "General"], size=n_rows, p=[0.25, 0.75])
    )

    # One synthetic hospital's resource state at patient arrival.
    synth["available_icu_beds"] = rng.integers(0, 11, size=n_rows)
    synth["available_general_beds"] = rng.integers(5, 41, size=n_rows)
    synth["available_doctors"] = rng.integers(2, 13, size=n_rows)
    synth["available_nurses"] = rng.integers(6, 31, size=n_rows)

    # Treatment time (minutes) influenced by resource need and priority.
    synth["treatment_time_min"] = (
        rng.gamma(shape=2.2, scale=55, size=n_rows)
        + synth["high_resource_need"] * rng.normal(90, 25, size=n_rows)
        + (6 - synth["priority_level"]) * rng.normal(12, 4, size=n_rows)
    ).clip(15, 720).round(1)

    # Waiting time (minutes): higher when fewer beds/staff are available.
    staff_pressure = (
        1 / synth["available_doctors"].clip(lower=1)
        + 1 / synth["available_nurses"].clip(lower=1)
    )
    bed_pressure = np.where(
        synth["bed_requirement"].eq("ICU"),
        1 / (synth["available_icu_beds"] + 1),
        1 / (synth["available_general_beds"] + 1),
    )

    wait = (
        rng.gamma(shape=2.0, scale=18, size=n_rows)
        + 150 * bed_pressure
        + 80 * staff_pressure
        - (6 - synth["priority_level"]) * 4
    )
    synth["waiting_time_min"] = np.clip(wait, 0, 360).round(1)

    # LOS in hours. Use empirical LOS when available, otherwise a synthetic fallback.
    if "icu_los_hours" not in synth.columns:
        synth["icu_los_hours"] = rng.lognormal(mean=4.0, sigma=0.7, size=n_rows)

    los_adjustment = synth["high_resource_need"] * rng.normal(24, 8, size=n_rows)
    synth["length_of_stay_hours"] = (
        pd.to_numeric(synth["icu_los_hours"], errors="coerce")
        .fillna(pd.Series(rng.lognormal(mean=4.0, sigma=0.7, size=n_rows)))
        + los_adjustment
    ).clip(4, 720).round(1)

    # Simulated discharge status.
    synth["discharge_status"] = np.where(
        synth["high_resource_need"].eq(1)
        & (rng.random(n_rows) < 0.08),
        "Expired",
        "Alive",
    )

    return synth.reset_index(drop=True)


def validate_synthetic_data(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simple quality report comparing numeric distributions.
    This is a project-level validation, not a clinical validation.
    """
    rows = []

    common_numeric = [
        c for c in synth_df.columns
        if c in real_df.columns
        and pd.api.types.is_numeric_dtype(synth_df[c])
    ]

    for col in common_numeric:
        real = pd.to_numeric(real_df[col], errors="coerce").dropna()
        synth = pd.to_numeric(synth_df[col], errors="coerce").dropna()

        if real.empty or synth.empty:
            continue

        real_mean = float(real.mean())
        synth_mean = float(synth.mean())
        denom = max(abs(real_mean), 1e-6)
        mean_diff_pct = abs(synth_mean - real_mean) / denom * 100

        rows.append(
            {
                "feature": col,
                "real_mean": round(real_mean, 4),
                "synthetic_mean": round(synth_mean, 4),
                "mean_difference_percent": round(mean_diff_pct, 2),
                "real_std": round(float(real.std(ddof=0)), 4),
                "synthetic_std": round(float(synth.std(ddof=0)), 4),
                "real_missing_percent": round(float(real_df[col].isna().mean() * 100), 2),
                "synthetic_missing_percent": round(float(synth_df[col].isna().mean() * 100), 2),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_difference_percent")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate project synthetic single-hospital data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/eicu_master.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic"),
    )
    parser.add_argument("--rows", type=int, default=7500)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading processed data: {args.input}")
    real_df = pd.read_csv(args.input)

    print(f"Generating {args.rows:,} synthetic encounters...")
    synth_df = generate_synthetic_dataset(real_df, n_rows=args.rows, seed=args.seed)

    synthetic_path = args.output_dir / "synthetic_hospital.csv"
    synth_df.to_csv(synthetic_path, index=False)

    print("Validating synthetic data...")
    validation = validate_synthetic_data(real_df, synth_df)
    validation_path = args.output_dir / "synthetic_validation.csv"
    validation.to_csv(validation_path, index=False)

    # Augmented dataset: align columns and combine real + synthetic records.
    real_aug = real_df.copy()
    real_aug["data_source"] = "real"

    synth_aug = synth_df.copy()
    synth_aug["data_source"] = "synthetic"

    augmented = pd.concat([real_aug, synth_aug], ignore_index=True, sort=False)
    augmented_path = args.output_dir / "augmented_dataset.csv"
    augmented.to_csv(augmented_path, index=False)

    print("\nDONE")
    print(f"Synthetic rows: {len(synth_df):,}")
    print(f"Synthetic columns: {synth_df.shape[1]:,}")
    print(f"Saved: {synthetic_path}")
    print(f"Validation report: {validation_path}")
    print(f"Augmented dataset: {augmented_path}")
    print(f"Augmented rows: {len(augmented):,}")

    print("\nHigh-resource distribution:")
    print(synth_df["high_resource_need"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()

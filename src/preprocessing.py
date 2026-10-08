
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def find_file(raw_dir: Path, stem: str) -> Path:
    """
    Finds files such as:
      patient.csv
      patient(1).csv
      patient(2).csv
      patient.csv.gz
    """
    patterns = [
        str(raw_dir / f"{stem}.csv"),
        str(raw_dir / f"{stem}*.csv"),
        str(raw_dir / f"{stem}.csv.gz"),
        str(raw_dir / f"{stem}*.csv.gz"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))

    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(f"Could not find a file beginning with '{stem}' in {raw_dir}")

    return Path(matches[0])


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert eICU numeric fields to numeric and replace common invalid values."""
    out = pd.to_numeric(series, errors="coerce")
    # eICU frequently uses negative sentinel values in clinical variables.
    out = out.mask(out < 0)
    return out


def clean_age(series: pd.Series) -> pd.Series:
    """Convert age to numeric. eICU '> 89' is represented as 90."""
    s = series.astype("string").str.strip()
    s = s.replace({"> 89": "90", ">89": "90"})
    return pd.to_numeric(s, errors="coerce")


def flatten_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Flatten MultiIndex columns produced by groupby.agg()."""
    cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            left, right = col
            cols.append(f"{prefix}{left}_{right}")
        else:
            cols.append(f"{prefix}{col}")
    df.columns = cols
    return df


# ---------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------

def build_patient_table(raw_dir: Path) -> pd.DataFrame:
    patient = pd.read_csv(find_file(raw_dir, "patient"))

    keep = [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "hospitalid",
        "wardid",
        "gender",
        "age",
        "ethnicity",
        "apacheadmissiondx",
        "admissionheight",
        "admissionweight",
        "hospitaladmitoffset",
        "hospitaladmitsource",
        "unittype",
        "unitadmitsource",
        "unitvisitnumber",
        "unitstaytype",
        "unitdischargeoffset",
        "unitdischargestatus",
        "hospitaldischargeoffset",
        "hospitaldischargestatus",
    ]
    keep = [c for c in keep if c in patient.columns]
    patient = patient[keep].copy()

    patient["age"] = clean_age(patient["age"])

    for col in [
        "admissionheight",
        "admissionweight",
        "hospitaladmitoffset",
        "unitdischargeoffset",
        "hospitaldischargeoffset",
    ]:
        if col in patient.columns:
            patient[col] = clean_numeric(patient[col])

    # Useful derived duration variables (hours).
    if "unitdischargeoffset" in patient.columns:
        patient["icu_los_hours"] = patient["unitdischargeoffset"] / 60.0

    if "hospitaldischargeoffset" in patient.columns and "hospitaladmitoffset" in patient.columns:
        patient["hospital_los_hours"] = (
            patient["hospitaldischargeoffset"] - patient["hospitaladmitoffset"]
        ) / 60.0

    return patient


def build_hospital_table(raw_dir: Path) -> pd.DataFrame:
    hospital = pd.read_csv(find_file(raw_dir, "hospital"))
    keep = [c for c in ["hospitalid", "numbedscategory", "teachingstatus", "region"] if c in hospital.columns]
    return hospital[keep].drop_duplicates("hospitalid")


def build_apache_aps(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(find_file(raw_dir, "apacheApsVar"))

    clinical_cols = [
        "intubated", "vent", "dialysis", "eyes", "motor", "verbal",
        "urine", "wbc", "temperature", "respiratoryrate", "sodium",
        "heartrate", "meanbp", "ph", "hematocrit", "creatinine",
        "albumin", "pao2", "pco2", "bun", "glucose", "bilirubin", "fio2",
    ]

    keep = ["patientunitstayid"] + [c for c in clinical_cols if c in df.columns]
    df = df[keep].copy()

    for col in keep:
        if col != "patientunitstayid":
            df[col] = clean_numeric(df[col])

    return df.rename(columns={c: f"aps_{c}" for c in df.columns if c != "patientunitstayid"})


def build_apache_prediction(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(find_file(raw_dir, "apachePredVar"))

    keep = [
        "patientunitstayid",
        "saps3day1",
        "ventday1",
        "diabetes",
        "electivesurgery",
        "readmit",
        "activetx",
        "age",
        "creatinine",
        "pao2",
        "fio2",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    for col in keep:
        if col != "patientunitstayid":
            df[col] = clean_numeric(df[col])

    return df.rename(columns={c: f"pred_{c}" for c in df.columns if c != "patientunitstayid"})


def build_apache_result(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(find_file(raw_dir, "apachePatientResult"))

    # One row exists for APACHE IV and one for IVa. Prefer IVa.
    if "apacheversion" in df.columns:
        iva = df[df["apacheversion"].astype(str).str.upper() == "IVA"].copy()
        if not iva.empty:
            df = iva

    keep = [
        "patientunitstayid",
        "acutephysiologyscore",
        "apachescore",
        "predictedicumortality",
        "actualicumortality",
        "predictediculos",
        "actualiculos",
        "predictedhospitalmortality",
        "actualhospitalmortality",
        "predictedhospitallos",
        "actualhospitallos",
        "actualventdays",
        "predventdays",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    numeric_cols = [
        "acutephysiologyscore",
        "apachescore",
        "predictedicumortality",
        "predictediculos",
        "predictedhospitalmortality",
        "predictedhospitallos",
        "actualiculos",
        "actualhospitallos",
        "actualventdays",
        "predventdays",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    return df.rename(columns={c: f"apache_{c}" for c in df.columns if c != "patientunitstayid"})


def build_admission_dx(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(find_file(raw_dir, "admissionDx"))

    if df.empty:
        return pd.DataFrame(columns=["patientunitstayid"])

    df = df.sort_values(["patientunitstayid", "admitdxenteredoffset"])
    first_dx = (
        df.groupby("patientunitstayid", as_index=False)
        .first()[["patientunitstayid", "admitdxname"]]
        .rename(columns={"admitdxname": "primary_admission_dx"})
    )

    counts = (
        df.groupby("patientunitstayid")
        .size()
        .rename("admission_dx_count")
        .reset_index()
    )

    return first_dx.merge(counts, on="patientunitstayid", how="outer")


def build_diagnosis_summary(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(find_file(raw_dir, "diagnosis"))

    if df.empty:
        return pd.DataFrame(columns=["patientunitstayid"])

    counts = (
        df.groupby("patientunitstayid")
        .size()
        .rename("diagnosis_count")
        .reset_index()
    )

    first_icd = (
        df.sort_values(["patientunitstayid", "diagnosisoffset"])
        .groupby("patientunitstayid", as_index=False)
        .first()[["patientunitstayid", "icd9code", "diagnosisstring"]]
        .rename(
            columns={
                "icd9code": "primary_icd9",
                "diagnosisstring": "primary_diagnosis",
            }
        )
    )

    return first_icd.merge(counts, on="patientunitstayid", how="outer")


def build_vital_periodic_summary(raw_dir: Path, chunksize: int = 200_000) -> pd.DataFrame:
    path = find_file(raw_dir, "vitalPeriodic")
    wanted = [
        "patientunitstayid",
        "temperature",
        "sao2",
        "heartrate",
        "respiration",
        "systemicsystolic",
        "systemicdiastolic",
        "systemicmean",
    ]

    pieces = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in wanted, chunksize=chunksize):
        numeric = [c for c in chunk.columns if c != "patientunitstayid"]
        for c in numeric:
            chunk[c] = clean_numeric(chunk[c])

        agg = chunk.groupby("patientunitstayid")[numeric].agg(["mean", "min", "max"])
        pieces.append(agg)

    if not pieces:
        return pd.DataFrame(columns=["patientunitstayid"])

    combined = pd.concat(pieces)
    combined = combined.groupby(level=0).mean()
    combined = flatten_columns(combined, "vital_").reset_index()
    return combined


def build_vital_aperiodic_summary(raw_dir: Path, chunksize: int = 200_000) -> pd.DataFrame:
    path = find_file(raw_dir, "vitalAperiodic")
    wanted = [
        "patientunitstayid",
        "noninvasivesystolic",
        "noninvasivediastolic",
        "noninvasivemean",
        "cardiacoutput",
    ]

    pieces = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in wanted, chunksize=chunksize):
        numeric = [c for c in chunk.columns if c != "patientunitstayid"]
        for c in numeric:
            chunk[c] = clean_numeric(chunk[c])

        agg = chunk.groupby("patientunitstayid")[numeric].agg(["mean", "min", "max"])
        pieces.append(agg)

    if not pieces:
        return pd.DataFrame(columns=["patientunitstayid"])

    combined = pd.concat(pieces)
    combined = combined.groupby(level=0).mean()
    combined = flatten_columns(combined, "aperiodic_").reset_index()
    return combined


def build_lab_summary(raw_dir: Path, chunksize: int = 200_000) -> pd.DataFrame:
    path = find_file(raw_dir, "lab")

    selected_labs = {
        "glucose": "lab_glucose",
        "creatinine": "lab_creatinine",
        "BUN": "lab_bun",
        "sodium": "lab_sodium",
        "potassium": "lab_potassium",
        "WBC x 1000": "lab_wbc",
        "Hgb": "lab_hgb",
        "Hct": "lab_hct",
        "platelets x 1000": "lab_platelets",
    }

    all_rows = []
    for chunk in pd.read_csv(
        path,
        usecols=["patientunitstayid", "labname", "labresult"],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["labname"].isin(selected_labs)].copy()
        chunk["labresult"] = clean_numeric(chunk["labresult"])
        all_rows.append(chunk)

    if not all_rows:
        return pd.DataFrame(columns=["patientunitstayid"])

    df = pd.concat(all_rows, ignore_index=True)

    agg = (
        df.groupby(["patientunitstayid", "labname"])["labresult"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    wide_parts = []
    for stat in ["mean", "min", "max"]:
        wide = agg.pivot(index="patientunitstayid", columns="labname", values=stat)
        wide = wide.rename(
            columns={lab: f"{name}_{stat}" for lab, name in selected_labs.items()}
        )
        wide_parts.append(wide)

    out = pd.concat(wide_parts, axis=1).reset_index()
    return out


def build_treatment_summary(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(find_file(raw_dir, "treatment"), usecols=["patientunitstayid", "treatmentstring"])
    out = (
        df.groupby("patientunitstayid")
        .agg(
            treatment_count=("treatmentstring", "size"),
            unique_treatments=("treatmentstring", "nunique"),
        )
        .reset_index()
    )
    return out


def build_provider_summary(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        find_file(raw_dir, "carePlanCareProvider"),
        usecols=["patientunitstayid", "providertype", "specialty"],
    )

    out = (
        df.groupby("patientunitstayid")
        .agg(
            provider_record_count=("providertype", "size"),
            provider_type_count=("providertype", "nunique"),
            provider_specialty_count=("specialty", "nunique"),
        )
        .reset_index()
    )
    return out


def build_respiratory_summary(raw_dir: Path) -> pd.DataFrame:
    path = find_file(raw_dir, "respiratoryCare")
    df = pd.read_csv(
        path,
        usecols=lambda c: c in {
            "patientunitstayid",
            "airwaytype",
            "ventstartoffset",
            "ventendoffset",
        },
    )

    if df.empty:
        return pd.DataFrame(columns=["patientunitstayid"])

    for col in ["ventstartoffset", "ventendoffset"]:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    df["has_respiratory_care"] = 1
    df["has_airway"] = df.get("airwaytype", pd.Series(index=df.index, dtype="object")).notna().astype(int)

    agg_map = {
        "has_respiratory_care": "max",
        "has_airway": "max",
    }
    if "ventstartoffset" in df.columns:
        agg_map["ventstartoffset"] = "min"
    if "ventendoffset" in df.columns:
        agg_map["ventendoffset"] = "max"

    out = df.groupby("patientunitstayid").agg(agg_map).reset_index()
    return out


def build_intake_output_summary(raw_dir: Path) -> pd.DataFrame:
    path = find_file(raw_dir, "intakeOutput")
    wanted = ["patientunitstayid", "intaketotal", "outputtotal", "dialysistotal", "nettotal"]

    df = pd.read_csv(path, usecols=lambda c: c in wanted)
    numeric = [c for c in df.columns if c != "patientunitstayid"]

    for c in numeric:
        df[c] = clean_numeric(df[c])

    out = (
        df.groupby("patientunitstayid")[numeric]
        .agg(["mean", "max"])
    )
    out = flatten_columns(out, "io_").reset_index()
    return out


# ---------------------------------------------------------------------
# Master dataset
# ---------------------------------------------------------------------

def build_master_dataset(raw_dir: Path) -> pd.DataFrame:
    print("Loading patient table...")
    master = build_patient_table(raw_dir)

    print("Adding hospital information...")
    master = master.merge(build_hospital_table(raw_dir), on="hospitalid", how="left")

    builders = [
        ("APACHE APS", build_apache_aps),
        ("APACHE prediction variables", build_apache_prediction),
        ("APACHE outcomes", build_apache_result),
        ("Admission diagnosis", build_admission_dx),
        ("Diagnosis summary", build_diagnosis_summary),
        ("Periodic vitals", build_vital_periodic_summary),
        ("Aperiodic vitals", build_vital_aperiodic_summary),
        ("Laboratory summary", build_lab_summary),
        ("Treatment summary", build_treatment_summary),
        ("Provider summary", build_provider_summary),
        ("Respiratory summary", build_respiratory_summary),
        ("Intake/output summary", build_intake_output_summary),
    ]

    for label, builder in builders:
        print(f"Adding {label}...")
        try:
            feature_df = builder(raw_dir)
            if not feature_df.empty:
                master = master.merge(feature_df, on="patientunitstayid", how="left")
        except FileNotFoundError:
            print(f"  Skipped: source file not found for {label}")

    # Derived outcome used later for the Random Forest.
    # This represents high resource demand, not a diagnosis.
    actual_los = pd.to_numeric(
        master.get("apache_actualiculos", pd.Series(index=master.index, dtype=float)),
        errors="coerce",
    )
    vent_days = pd.to_numeric(
        master.get("apache_actualventdays", pd.Series(index=master.index, dtype=float)),
        errors="coerce",
    )

    master["high_resource_need"] = np.where(
        (actual_los >= 3.0) | (vent_days > 0),
        1,
        0,
    )

    # Preserve rows even when the target-related APACHE result is unavailable.
    missing_outcome = actual_los.isna() & vent_days.isna()
    master.loc[missing_outcome, "high_resource_need"] = np.nan

    # Remove exact duplicate stays, if any.
    master = master.drop_duplicates("patientunitstayid").reset_index(drop=True)

    return master


def save_quality_report(df: pd.DataFrame, output_dir: Path) -> None:
    report = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_count": [int(df[c].isna().sum()) for c in df.columns],
            "missing_percent": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
            "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )
    report.to_csv(output_dir / "data_quality_report.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one-row-per-ICU-stay eICU master dataset.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    master = build_master_dataset(args.raw_dir)

    output_file = args.output_dir / "eicu_master.csv"
    master.to_csv(output_file, index=False)
    save_quality_report(master, args.output_dir)

    print("\nDONE")
    print(f"Rows: {len(master):,}")
    print(f"Columns: {master.shape[1]:,}")
    print(f"Unique ICU stays: {master['patientunitstayid'].nunique():,}")
    print(f"Saved: {output_file}")
    print(f"Quality report: {args.output_dir / 'data_quality_report.csv'}")

    if "high_resource_need" in master.columns:
        print("\nHigh-resource target distribution:")
        print(master["high_resource_need"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

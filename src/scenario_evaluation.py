"""V2 fixed-window simulation. The legacy GA simulator remains in digital_twin."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd
import simpy

from digital_twin import (
    HospitalPolicy, HealthcareDigitalTwin, add_random_forest_predictions, file_sha256,
)

VERSION = "2.0-time-window"

# Measured demonstration policy; demand, targets and simulation rules are unchanged.
DEFAULT_V2_RESOURCES = dict(icu_beds=210, general_beds=280, doctors=85, nurses=100)


@dataclass(frozen=True)
class ScenarioConfig:
    simulation_duration_hours: float = 24.0
    best_case_arrival_rate: float = 12.0
    average_case_arrival_rate: float = 22.0
    worst_case_arrival_rate: float = 35.0
    replications_per_scenario: int = 10
    mean_wait_target_minutes: float = 20.0
    high_risk_wait_target_minutes: float = 10.0
    max_utilization_target: float = 0.90
    robustness_threshold: float = 90.0
    random_seed: int = 42

    def validate(self):
        if not all(np.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Configuration values must be finite.")
        if not 1 <= self.simulation_duration_hours <= 72:
            raise ValueError("Simulation duration must be between 1 and 72 hours.")
        rates = (self.best_case_arrival_rate, self.average_case_arrival_rate,
                 self.worst_case_arrival_rate)
        if not 0 <= rates[0] <= rates[1] <= rates[2] <= 1000:
            raise ValueError("Arrival rates must satisfy 0 <= Best <= Average <= Worst <= 1000 patients/hour.")
        if int(self.replications_per_scenario) != self.replications_per_scenario or not 3 <= self.replications_per_scenario <= 30:
            raise ValueError("Replications per scenario must be an integer from 3 to 30.")
        if min(self.mean_wait_target_minutes, self.high_risk_wait_target_minutes) < 0:
            raise ValueError("Wait targets must be nonnegative minutes.")
        if not 0 <= self.max_utilization_target <= 1:
            raise ValueError("Maximum utilization must be a fraction between 0 and 1.")
        if not 0 <= self.robustness_threshold <= 100:
            raise ValueError("Robustness threshold must be between 0 and 100 percent.")
        if int(self.random_seed) != self.random_seed or self.random_seed < 0:
            raise ValueError("Random seed must be a nonnegative integer.")


def validate_policy(policy):
    for key in ("icu_beds", "general_beds", "doctors", "nurses"):
        value = getattr(policy, key)
        if not np.isfinite(value) or int(value) != value or not 1 <= value <= 10000:
            raise ValueError(f"{key} must be an integer between 1 and 10000.")
    if policy.icu_risk_threshold != 0.50:
        raise ValueError("V2 requires the fixed RF threshold 0.50.")


@lru_cache(maxsize=4)
def _load_profiles(data_path, data_stamp, model_path, model_stamp):
    # Stamp arguments invalidate cached data/predictions when either file changes.
    return add_random_forest_predictions(pd.read_csv(data_path, low_memory=False), Path(model_path))


def load_patient_profiles(data_path: Path, model_path: Path) -> pd.DataFrame:
    """Cache RF inference once per source version; callers receive an isolated copy."""
    data_path, model_path = Path(data_path).resolve(), Path(model_path).resolve()
    def stamp(path):
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    if not model_path.exists():
        raise FileNotFoundError(f"RF model is required: {model_path}")
    return _load_profiles(str(data_path), stamp(data_path), str(model_path), stamp(model_path)).copy()


def generate_arrivals(arrival_rate, simulation_duration_hours, rng):
    """First arrival follows an exponential interval; window is [0, end)."""
    if not np.isfinite(arrival_rate) or arrival_rate < 0:
        raise ValueError("Arrival rate must be finite and nonnegative.")
    if not np.isfinite(simulation_duration_hours) or simulation_duration_hours <= 0:
        raise ValueError("Simulation duration must be finite and positive.")
    arrivals = []
    if arrival_rate == 0:
        return arrivals
    time = 0.0
    while True:
        time += float(rng.exponential(60.0 / arrival_rate))
        if time >= simulation_duration_hours * 60.0:
            return arrivals
        arrivals.append(time)


def prepare_profiles(patient_df):
    if patient_df.empty:
        raise ValueError("Patient profiles cannot be empty.")
    if "predicted_probability" not in patient_df:
        raise ValueError("Patient profiles require cached RF predicted_probability.")
    records = patient_df.to_dict("records")
    def number(row, key, fallback):
        value = pd.to_numeric(row.get(key), errors="coerce")
        return float(value) if pd.notna(value) and np.isfinite(value) else fallback
    profiles = []
    for row in records:
        treatment = float(np.clip(number(row, "treatment_time_min", 120), 15, 720))
        los = float(np.clip(number(row, "length_of_stay_hours", max(treatment / 60, 6)) * 60,
                            treatment, 14 * 24 * 60))
        risk = number(row, "predicted_probability", 0)
        if not 0 <= risk <= 1:
            raise ValueError("RF probabilities must be between 0 and 1.")
        profiles.append((risk >= .50, treatment, los, HealthcareDigitalTwin._priority_from_row(row)))
    return profiles


def policy_verdict(metrics, config):
    conditions = {
        "mean_wait_pass": metrics["mean_wait"] <= config.mean_wait_target_minutes,
        "high_risk_wait_pass": metrics["high_risk_mean_wait"] <= config.high_risk_wait_target_minutes,
    }
    for name in ("icu", "general", "doctor", "nurse"):
        conditions[f"{name}_util_pass"] = metrics[f"{name}_utilization"] <= config.max_utilization_target
    return ("ACCEPTABLE" if all(conditions.values()) else "UNACCEPTABLE"), conditions


def _simulate(profiles, policy, arrival_rate, config, seed):
    rng = np.random.default_rng(seed)
    arrivals = generate_arrivals(arrival_rate, config.simulation_duration_hours, rng)
    # Sampling with replacement permits arbitrarily many arrivals without source mutation.
    choices = rng.integers(0, len(profiles), size=len(arrivals))
    env = simpy.Environment()
    window = config.simulation_duration_hours * 60.0
    capacities = {"icu": policy.icu_beds, "general": policy.general_beds,
                  "doctor": policy.doctors, "nurse": policy.nurses}
    available = capacities.copy()
    busy = dict.fromkeys(capacities, 0.0)
    waiting, patients = [], []

    def dispatch():
        # Allocate bed + doctor + nurse atomically; no idle partial staff holdings.
        waiting.sort(key=lambda p: (p["priority"], p["index"]))
        for p in waiting[:]:
            bed = p["bed"]
            if all(available[k] > 0 for k in (bed, "doctor", "nurse")):
                waiting.remove(p)
                p["start"] = env.now
                for key, duration in ((bed, p["los"]), ("doctor", p["treatment"]), ("nurse", p["treatment"])):
                    available[key] -= 1
                    busy[key] += min(duration, window - env.now)
                env.process(treat(p))

    def treat(p):
        yield env.timeout(p["treatment"])
        available["doctor"] += 1
        available["nurse"] += 1
        dispatch()
        yield env.timeout(p["los"] - p["treatment"])
        available[p["bed"]] += 1
        p["completed"] = True
        dispatch()

    def arrive():
        for index, (time, choice) in enumerate(zip(arrivals, choices)):
            yield env.timeout(time - env.now)
            high_risk, treatment, los, priority = profiles[choice]
            p = dict(index=index, arrival=time, high_risk=high_risk,
                     bed="icu" if high_risk else "general", treatment=treatment,
                     los=los, priority=priority, start=None, completed=False)
            patients.append(p)
            waiting.append(p)
            dispatch()

    env.process(arrive())
    env.run(until=window)
    waits = [(p["start"] if p["start"] is not None else window) - p["arrival"] for p in patients]
    high_waits = [w for p, w in zip(patients, waits) if p["high_risk"]]
    completed = sum(p["completed"] for p in patients)
    metrics = dict(patients_arrived=len(patients), patients_completed=completed,
                   patients_waiting_at_end=len(waiting),
                   patients_in_treatment_at_end=len(patients) - completed - len(waiting),
                   unfinished_patients=len(patients) - completed,
                   mean_wait=float(np.mean(waits)) if waits else 0.0,
                   median_wait=float(np.median(waits)) if waits else 0.0,
                   p95_wait=float(np.percentile(waits, 95)) if waits else 0.0,
                   high_risk_mean_wait=float(np.mean(high_waits)) if high_waits else 0.0,
                   throughput=completed / config.simulation_duration_hours,
                   simulation_duration_hours=config.simulation_duration_hours,
                   arrival_rate=arrival_rate, seed=int(seed), high_risk_patients=sum(p["high_risk"] for p in patients),
                   wait_censored_count=len(waiting))
    for key in capacities:
        metrics[f"{key}_utilization"] = float(np.clip(busy[key] / (capacities[key] * window), 0, 1))
    metrics["policy_verdict"], metrics["condition_breakdown"] = policy_verdict(metrics, config)
    return metrics


def simulate_time_window(patient_df, policy, arrival_rate, config=None, seed=42):
    config = config or ScenarioConfig()
    config.validate()
    validate_policy(policy)
    return _simulate(prepare_profiles(patient_df), policy, arrival_rate, config, seed)


def evaluate_current_policy(patient_df, policy, config=None):
    config = config or ScenarioConfig()
    config.validate()
    validate_policy(policy)
    profiles = prepare_profiles(patient_df)
    scenarios = {}
    for scenario_index, (name, rate) in enumerate(zip(
        ("best", "average", "worst"),
        (config.best_case_arrival_rate, config.average_case_arrival_rate, config.worst_case_arrival_rate),
    )):
        runs = [_simulate(profiles, policy, rate, config,
                          int(config.random_seed) * 1000 + scenario_index * 100 + run)
                for run in range(int(config.replications_per_scenario))]
        accepted = sum(r["policy_verdict"] == "ACCEPTABLE" for r in runs)
        robustness = accepted / len(runs) * 100
        numeric_keys = [key for key, value in runs[0].items() if isinstance(value, (float, int)) and key != "seed"]
        aggregates = {key: {"mean": float(np.mean([r[key] for r in runs])),
                            "min": float(min(r[key] for r in runs)),
                            "max": float(max(r[key] for r in runs))} for key in numeric_keys}
        scenarios[name] = dict(arrival_rate=rate, total_runs=len(runs), acceptable_run_count=accepted,
                               acceptable_run_percentage=robustness, scenario_robustness=robustness,
                               scenario_verdict="ACCEPTABLE" if robustness >= config.robustness_threshold else "UNACCEPTABLE",
                               metrics=aggregates, runs=runs)
    status = ("ROBUST" if scenarios["worst"]["scenario_robustness"] >= config.robustness_threshold
              else "CONDITIONALLY ACCEPTABLE" if scenarios["average"]["scenario_robustness"] >= config.robustness_threshold
              else "UNACCEPTABLE")
    return dict(schema_version=2, simulator_version=VERSION, configuration=asdict(config),
                resource_configuration=asdict(policy), scenarios=scenarios,
                overall_robustness=sum(s["acceptable_run_count"] for s in scenarios.values()) / (3 * config.replications_per_scenario) * 100,
                overall_policy_status=status,
                metric_definitions={
                    "seeds": "base_seed * 1000 + scenario_index * 100 + run_index; scenario_index Best=0, Average=1, Worst=2.",
                    "wait": "Observed wait for every arrival: service_start - arrival, or window_end - arrival if still waiting (right censored). Empty groups use zero.",
                    "in_treatment": "Holding a bed, including post-treatment stay; waiting patients hold no resources.",
                    "throughput": "Completed hospital flows per hour of observation.",
                    "utilization": "Occupied resource minutes inside [0, window_end) / (capacity * window_minutes).",
                    "allocation": "Priority then arrival order among feasible patients; bed, doctor and nurse acquired together. Staff held during treatment, bed during full LOS.",
                    "percentiles": "Scenario wait metrics are means of per-run statistics, not pooled patient percentiles.",
                })


def save_current_policy_results(result, results_dir, data_path=None, model_path=None):
    """Only writes the dedicated current-policy artifact, never legacy/model artifacts."""
    artifact = dict(result)
    artifact["provenance"] = {"simulator_sha256": file_sha256(Path(__file__)),
                              "legacy_helpers_sha256": file_sha256(Path(__file__).with_name("digital_twin.py"))}
    for label, path in (("data", data_path), ("model", model_path)):
        if path is not None:
            artifact["provenance"][f"{label}_path"] = str(Path(path).resolve())
            artifact["provenance"][f"{label}_sha256"] = file_sha256(Path(path))
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "current_policy_evaluation.json"
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="V2 current hospital policy scenario evaluation")
    parser.add_argument("--input", type=Path, default=Path("data/synthetic/synthetic_hospital.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/random_forest_pipeline.joblib"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/digital_twin"))
    for key, value in asdict(ScenarioConfig()).items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    for key, value in DEFAULT_V2_RESOURCES.items():
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=value)
    args = vars(parser.parse_args())
    config = ScenarioConfig(**{key: args[key] for key in asdict(ScenarioConfig())})
    policy = HospitalPolicy(**{key: args[key] for key in ("icu_beds", "general_beds", "doctors", "nurses")})
    config.validate()
    validate_policy(policy)
    result = evaluate_current_policy(load_patient_profiles(args["input"], args["model"]), policy, config)
    path = save_current_policy_results(result, args["results_dir"], args["input"], args["model"])
    print(json.dumps({"overall_policy_status": result["overall_policy_status"],
                      "overall_robustness": result["overall_robustness"],
                      "scenarios": {name: {key: s[key] for key in ("scenario_robustness", "scenario_verdict")} for name, s in result["scenarios"].items()},
                      "artifact": str(path)}, indent=2))


if __name__ == "__main__":
    main()

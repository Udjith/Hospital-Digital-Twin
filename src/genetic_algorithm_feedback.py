from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from digital_twin import (
    HospitalPolicy,
    add_random_forest_predictions,
    derive_ga_bounds,
    simulate_hospital,
    validate_baseline_artifact,
)

RANDOM_SEED = 42
FIXED_ICU_RISK_THRESHOLD = 0.50

DEFAULT_OBJECTIVES = {
    "acceptable_mean_wait_min": 20.0,
    "acceptable_high_risk_wait_min": 10.0,
    "mean_wait_weight": 1.0,
    "high_risk_wait_weight": 2.0,
    "utilization_weight_scale": 1.0,
    "resource_cost_weight_scale": 1.0,
    "throughput_reward_weight": 0.10,
}


def load_feedback(
    path: Path | None,
    baseline_bounds: dict,
) -> tuple[dict, dict, str]:
    bounds = {k: list(v) for k, v in baseline_bounds.items()}
    objectives = dict(DEFAULT_OBJECTIVES)
    summary = "No Human-in-the-Loop feedback applied."

    if path is None or not path.exists():
        return bounds, objectives, summary

    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("apply_feedback", False):
        return bounds, objectives, data.get("summary") or summary

    for resource, limits in (data.get("constraints") or {}).items():
        if resource not in bounds or not isinstance(limits, dict):
            continue
        if limits.get("min") is not None:
            bounds[resource][0] = max(
                bounds[resource][0],
                int(limits["min"]),
            )
        if limits.get("max") is not None:
            bounds[resource][1] = min(
                bounds[resource][1],
                int(limits["max"]),
            )
        if bounds[resource][0] > bounds[resource][1]:
            raise ValueError(
                f"Invalid feedback constraint for {resource}: "
                f"min {bounds[resource][0]} > max {bounds[resource][1]}"
            )

    for key, value in (data.get("objectives") or {}).items():
        if key in objectives and value is not None:
            objectives[key] = float(value)

    summary = data.get("summary") or "Human feedback applied."
    return bounds, objectives, summary


def clamp(value, low, high):
    return max(low, min(high, value))


def random_policy(rng: random.Random, bounds: dict) -> HospitalPolicy:
    return HospitalPolicy(
        icu_beds=rng.randint(*bounds["icu_beds"]),
        general_beds=rng.randint(*bounds["general_beds"]),
        doctors=rng.randint(*bounds["doctors"]),
        nurses=rng.randint(*bounds["nurses"]),
        icu_risk_threshold=FIXED_ICU_RISK_THRESHOLD,
    )


def policy_key(policy: HospitalPolicy) -> tuple:
    return (
        policy.icu_beds,
        policy.general_beds,
        policy.doctors,
        policy.nurses,
        FIXED_ICU_RISK_THRESHOLD,
    )


def calculate_fitness(metrics: dict, policy: HospitalPolicy, objectives: dict) -> float:
    mean_wait = float(metrics.get("mean_waiting_time_min") or 0.0)
    high_wait = float(metrics.get("high_risk_mean_waiting_time_min") or mean_wait)
    throughput = float(metrics.get("throughput_patients_per_day") or 0.0)

    icu_util = float(metrics.get("icu_bed_utilization") or 0.0)
    gen_util = float(metrics.get("general_bed_utilization") or 0.0)
    doc_util = float(metrics.get("doctor_utilization") or 0.0)
    nurse_util = float(metrics.get("nurse_utilization") or 0.0)

    wait_penalty = max(0.0, mean_wait - objectives["acceptable_mean_wait_min"])
    high_wait_penalty = max(
        0.0,
        high_wait - objectives["acceptable_high_risk_wait_min"],
    )

    def band_penalty(value: float, low: float, high: float) -> float:
        if value < low:
            return low - value
        if value > high:
            return value - high
        return 0.0

    util_scale = objectives["utilization_weight_scale"]
    utilization_penalty = util_scale * (
        12.0 * band_penalty(icu_util, 0.55, 0.85)
        + 12.0 * band_penalty(gen_util, 0.55, 0.85)
        + 8.0 * band_penalty(doc_util, 0.35, 0.80)
        + 8.0 * band_penalty(nurse_util, 0.35, 0.80)
    )

    resource_cost = (
        0.35 * policy.icu_beds
        + 0.15 * policy.general_beds
        + 1.20 * policy.doctors
        + 0.55 * policy.nurses
    ) * objectives["resource_cost_weight_scale"]

    penalty = (
        objectives["mean_wait_weight"] * wait_penalty
        + objectives["high_risk_wait_weight"] * high_wait_penalty
        + utilization_penalty
        + resource_cost
        - objectives["throughput_reward_weight"] * throughput
    )
    return -float(penalty)


def tournament_selection(scored, rng, tournament_size=3):
    contenders = rng.sample(scored, k=min(tournament_size, len(scored)))
    return max(contenders, key=lambda x: x[1])[0]


def crossover(a: HospitalPolicy, b: HospitalPolicy, rng: random.Random) -> HospitalPolicy:
    return HospitalPolicy(
        icu_beds=rng.choice([a.icu_beds, b.icu_beds]),
        general_beds=rng.choice([a.general_beds, b.general_beds]),
        doctors=rng.choice([a.doctors, b.doctors]),
        nurses=rng.choice([a.nurses, b.nurses]),
        icu_risk_threshold=FIXED_ICU_RISK_THRESHOLD,
    )


def mutate(policy: HospitalPolicy, rng: random.Random, bounds: dict, mutation_rate=0.25):
    p = HospitalPolicy(**asdict(policy))
    mutation_steps = {
        "icu_beds": [-2, -1, 1, 2],
        "general_beds": [-5, -3, 3, 5],
        "doctors": [-2, -1, 1, 2],
        "nurses": [-4, -2, 2, 4],
    }
    for attr, steps in mutation_steps.items():
        if rng.random() < mutation_rate:
            lo, hi = bounds[attr]
            setattr(p, attr, int(clamp(getattr(p, attr) + rng.choice(steps), lo, hi)))
    p.icu_risk_threshold = FIXED_ICU_RISK_THRESHOLD
    return p


def prepare_patients(input_path: Path, model_path: Path, max_patients: int) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    if max_patients > 0 and len(df) > max_patients:
        if "arrival_time" in df.columns:
            df["_arrival_sort"] = pd.to_datetime(df["arrival_time"], errors="coerce")
            df = df.sort_values("_arrival_sort").head(max_patients).drop(columns="_arrival_sort").copy()
        else:
            df = df.head(max_patients).copy()
    return add_random_forest_predictions(df, model_path)


def run_ga(patient_df, bounds, objectives, population_size=12, generations=10,
           mutation_rate=0.25, elite_count=2, seed=RANDOM_SEED):
    rng = random.Random(seed)
    population = [random_policy(rng, bounds) for _ in range(population_size)]
    cache = {}
    history_rows = []
    best_policy = None
    best_metrics = None
    best_fitness = float("-inf")

    for generation in range(1, generations + 1):
        scored = []
        for policy in population:
            key = policy_key(policy)
            if key in cache:
                fitness, metrics = cache[key]
            else:
                metrics, _ = simulate_hospital(patient_df, policy)
                fitness = calculate_fitness(metrics, policy, objectives)
                cache[key] = (fitness, metrics)

            scored.append((policy, fitness, metrics))
            history_rows.append({
                "generation": generation,
                "fitness": fitness,
                **asdict(policy),
                **metrics,
            })

            if fitness > best_fitness:
                best_fitness = fitness
                best_policy = HospitalPolicy(**asdict(policy))
                best_metrics = dict(metrics)

        scored.sort(key=lambda x: x[1], reverse=True)
        gen_best = scored[0]
        print(
            f"Generation {generation:02d}/{generations} | "
            f"best fitness={gen_best[1]:.4f} | "
            f"wait={gen_best[2]['mean_waiting_time_min']:.2f} min | "
            f"ICU={gen_best[0].icu_beds}, General={gen_best[0].general_beds}, "
            f"Doctors={gen_best[0].doctors}, Nurses={gen_best[0].nurses}"
        )

        next_population = [HospitalPolicy(**asdict(x[0])) for x in scored[:elite_count]]
        while len(next_population) < population_size:
            a = tournament_selection(scored, rng)
            b = tournament_selection(scored, rng)
            next_population.append(mutate(crossover(a, b, rng), rng, bounds, mutation_rate))
        population = next_population

    return best_policy, best_metrics, pd.DataFrame(history_rows), best_fitness


def main():
    parser = argparse.ArgumentParser(description="Feedback-aware GA optimization.")
    parser.add_argument("--input", type=Path, default=Path("data/synthetic/synthetic_hospital.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/random_forest_pipeline.joblib"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/genetic_algorithm"))
    parser.add_argument("--feedback", type=Path, default=Path("results/feedback/interpreted_feedback.json"))
    parser.add_argument("--baseline", type=Path, default=Path("results/digital_twin/baseline_metrics.json"))
    parser.add_argument("--ignore-feedback", action="store_true")
    parser.add_argument("--max-patients", type=int, default=500)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--mutation-rate", type=float, default=0.25)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_errors = validate_baseline_artifact(baseline, args.input, args.model)
    if baseline_errors:
        raise ValueError(
            "Canonical baseline validation failed: " + "; ".join(baseline_errors)
        )
    baseline_bounds = derive_ga_bounds(baseline["policy"])
    feedback_path = None if args.ignore_feedback else args.feedback
    bounds, objectives, feedback_summary = load_feedback(
        feedback_path,
        baseline_bounds,
    )

    print("Applied GA bounds:")
    print(json.dumps(bounds, indent=2))
    print("Applied objectives:")
    print(json.dumps(objectives, indent=2))
    print(f"Feedback interpretation: {feedback_summary}\n")

    patients = prepare_patients(args.input, args.model, args.max_patients)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    best_policy, best_metrics, history, best_fitness = run_ga(
        patients,
        bounds,
        objectives,
        args.population,
        args.generations,
        args.mutation_rate,
        args.elite_count,
        args.seed,
    )

    output = {
        "best_policy": asdict(best_policy),
        "best_fitness": best_fitness,
        "best_metrics": best_metrics,
        "applied_bounds": bounds,
        "applied_objectives": objectives,
        "feedback_summary": feedback_summary,
        "feedback_applied": bool(feedback_path and feedback_path.exists()),
        "baseline_provenance": {
            "simulator_version": baseline["baseline_metadata"]["simulator_version"],
            "hospital_scale_multiplier": baseline["baseline_metadata"][
                "hospital_scale_multiplier"
            ],
            "baseline_generated_at_utc": baseline["baseline_metadata"][
                "generated_at_utc"
            ],
            "baseline_policy": {
                key: int(baseline["policy"][key])
                for key in ["icu_beds", "general_beds", "doctors", "nurses"]
            },
            "ga_search_bounds": {
                key: list(value)
                for key, value in baseline_bounds.items()
            },
        },
    }
    (args.results_dir / "best_policy.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    history.to_csv(args.results_dir / "ga_history.csv", index=False)
    history.sort_values(["generation", "fitness"], ascending=[True, False]).groupby(
        "generation", as_index=False
    ).first().to_csv(args.results_dir / "best_by_generation.csv", index=False)

    print("\nGENETIC ALGORITHM COMPLETE")
    print("\nBest policy:")
    print(json.dumps(asdict(best_policy), indent=2))
    print(f"\nBest fitness:\n{best_fitness:.4f}")
    print("\nBest Digital Twin metrics:")
    print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()

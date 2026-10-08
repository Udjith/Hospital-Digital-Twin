
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from digital_twin import (
    HOSPITAL_SCALE_MULTIPLIER,
    HospitalPolicy,
    add_random_forest_predictions,
    derive_ga_bounds,
    simulate_hospital,
    validate_baseline_artifact,
)

RANDOM_SEED = 42

DEFAULT_OBJECTIVES = {
    "acceptable_mean_wait_min": 20.0,
    "acceptable_high_risk_wait_min": 10.0,
    "mean_wait_weight": 1.0,
    "high_risk_wait_weight": 2.0,
    "utilization_weight_scale": 1.0,
    "resource_cost_weight_scale": 1.0,
    "throughput_reward_weight": 0.10,
}

FIXED_ICU_RISK_THRESHOLD = 0.50


def safe_print(value=""):
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe = text.encode(encoding, errors="replace").decode(
        encoding,
        errors="replace",
    )
    print(safe)


def load_feedback(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def apply_feedback(feedback: dict, baseline_bounds: dict):
    bounds = {k: list(v) for k, v in baseline_bounds.items()}
    objectives = dict(DEFAULT_OBJECTIVES)

    if not feedback or not feedback.get("apply_feedback", False):
        return {k: tuple(v) for k, v in bounds.items()}, objectives

    constraints = feedback.get("constraints", {})

    for resource in bounds:
        constraint = constraints.get(resource, {}) or {}

        if constraint.get("min") is not None:
            bounds[resource][0] = max(
                bounds[resource][0],
                int(constraint["min"]),
            )

        if constraint.get("max") is not None:
            bounds[resource][1] = min(
                bounds[resource][1],
                int(constraint["max"]),
            )

        if bounds[resource][0] > bounds[resource][1]:
            raise ValueError(
                f"Invalid feedback constraint for {resource}: "
                f"min {bounds[resource][0]} > max {bounds[resource][1]}"
            )

    incoming_objectives = feedback.get("objectives", {}) or {}

    for key in objectives:
        if incoming_objectives.get(key) is not None:
            objectives[key] = float(incoming_objectives[key])

    return {k: tuple(v) for k, v in bounds.items()}, objectives


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


def policy_key(policy: HospitalPolicy):
    return (
        policy.icu_beds,
        policy.general_beds,
        policy.doctors,
        policy.nurses,
        FIXED_ICU_RISK_THRESHOLD,
    )


def calculate_fitness(metrics: dict, policy: HospitalPolicy, objectives: dict) -> float:
    mean_wait = float(metrics.get("mean_waiting_time_min") or 0.0)
    high_risk_wait = float(
        metrics.get("high_risk_mean_waiting_time_min") or mean_wait
    )
    throughput = float(metrics.get("throughput_patients_per_day") or 0.0)

    icu_util = float(metrics.get("icu_bed_utilization") or 0.0)
    gen_util = float(metrics.get("general_bed_utilization") or 0.0)
    doc_util = float(metrics.get("doctor_utilization") or 0.0)
    nurse_util = float(metrics.get("nurse_utilization") or 0.0)

    wait_penalty = max(
        0.0,
        mean_wait - objectives["acceptable_mean_wait_min"],
    )
    high_risk_wait_penalty = max(
        0.0,
        high_risk_wait - objectives["acceptable_high_risk_wait_min"],
    )

    def band_penalty(value, low, high):
        if value < low:
            return low - value
        if value > high:
            return value - high
        return 0.0

    utilization_penalty = (
        12.0 * band_penalty(icu_util, 0.55, 0.85)
        + 12.0 * band_penalty(gen_util, 0.55, 0.85)
        + 8.0 * band_penalty(doc_util, 0.35, 0.80)
        + 8.0 * band_penalty(nurse_util, 0.35, 0.80)
    ) * objectives["utilization_weight_scale"]

    resource_cost = (
        0.35 * policy.icu_beds
        + 0.15 * policy.general_beds
        + 1.20 * policy.doctors
        + 0.55 * policy.nurses
    ) * objectives["resource_cost_weight_scale"]

    penalty = (
        objectives["mean_wait_weight"] * wait_penalty
        + objectives["high_risk_wait_weight"] * high_risk_wait_penalty
        + utilization_penalty
        + resource_cost
        - objectives["throughput_reward_weight"] * throughput
    )

    return -float(penalty)


def evaluate_policy(patient_df, policy, objectives):
    metrics, _ = simulate_hospital(patient_df, policy)
    return calculate_fitness(metrics, policy, objectives), metrics


def tournament_selection(scored, rng, tournament_size=3):
    contenders = rng.sample(
        scored,
        k=min(tournament_size, len(scored)),
    )
    return max(contenders, key=lambda x: x[1])[0]


def crossover(a, b, rng):
    return HospitalPolicy(
        icu_beds=rng.choice([a.icu_beds, b.icu_beds]),
        general_beds=rng.choice([a.general_beds, b.general_beds]),
        doctors=rng.choice([a.doctors, b.doctors]),
        nurses=rng.choice([a.nurses, b.nurses]),
        icu_risk_threshold=FIXED_ICU_RISK_THRESHOLD,
    )


def mutate(policy, rng, bounds, mutation_rate=0.25):
    p = HospitalPolicy(**asdict(policy))

    if rng.random() < mutation_rate:
        p.icu_beds = int(
            clamp(
                p.icu_beds + rng.choice([-2, -1, 1, 2]),
                *bounds["icu_beds"],
            )
        )

    if rng.random() < mutation_rate:
        p.general_beds = int(
            clamp(
                p.general_beds + rng.choice([-5, -3, 3, 5]),
                *bounds["general_beds"],
            )
        )

    if rng.random() < mutation_rate:
        p.doctors = int(
            clamp(
                p.doctors + rng.choice([-2, -1, 1, 2]),
                *bounds["doctors"],
            )
        )

    if rng.random() < mutation_rate:
        p.nurses = int(
            clamp(
                p.nurses + rng.choice([-4, -2, 2, 4]),
                *bounds["nurses"],
            )
        )

    p.icu_risk_threshold = FIXED_ICU_RISK_THRESHOLD
    return p


def prepare_patients(input_path, model_path, max_patients):
    df = pd.read_csv(input_path, low_memory=False)

    if max_patients > 0 and len(df) > max_patients:
        if "arrival_time" in df.columns:
            df["_arrival_sort"] = pd.to_datetime(
                df["arrival_time"],
                errors="coerce",
            )
            df = (
                df.sort_values("_arrival_sort")
                .head(max_patients)
                .drop(columns="_arrival_sort")
                .copy()
            )
        else:
            df = df.head(max_patients).copy()

    return add_random_forest_predictions(df, model_path)


def run_ga(
    patient_df,
    bounds,
    objectives,
    population_size=12,
    generations=10,
    mutation_rate=0.25,
    elite_count=2,
    seed=RANDOM_SEED,
):
    rng = random.Random(seed)
    population = [
        random_policy(rng, bounds)
        for _ in range(population_size)
    ]

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
                fitness, metrics = evaluate_policy(
                    patient_df,
                    policy,
                    objectives,
                )
                cache[key] = (fitness, metrics)

            scored.append((policy, fitness, metrics))

            history_rows.append(
                {
                    "generation": generation,
                    "fitness": fitness,
                    **asdict(policy),
                    **metrics,
                }
            )

            if fitness > best_fitness:
                best_fitness = fitness
                best_policy = HospitalPolicy(**asdict(policy))
                best_metrics = dict(metrics)

        scored.sort(key=lambda x: x[1], reverse=True)
        gen_best = scored[0]

        safe_print(
            f"Generation {generation:02d}/{generations} | "
            f"best fitness={gen_best[1]:.4f} | "
            f"wait={gen_best[2]['mean_waiting_time_min']:.2f} min | "
            f"ICU={gen_best[0].icu_beds}, "
            f"General={gen_best[0].general_beds}, "
            f"Doctors={gen_best[0].doctors}, "
            f"Nurses={gen_best[0].nurses}, "
            f"Threshold={gen_best[0].icu_risk_threshold:.3f}"
        )

        next_population = [
            HospitalPolicy(**asdict(x[0]))
            for x in scored[:elite_count]
        ]

        while len(next_population) < population_size:
            parent_a = tournament_selection(scored, rng)
            parent_b = tournament_selection(scored, rng)
            child = crossover(parent_a, parent_b, rng)
            child = mutate(
                child,
                rng,
                bounds,
                mutation_rate,
            )
            next_population.append(child)

        population = next_population

    return (
        best_policy,
        best_metrics,
        pd.DataFrame(history_rows),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/synthetic/synthetic_hospital.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/random_forest_pipeline.joblib"),
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        default=Path("results/feedback/interpreted_feedback.json"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/digital_twin/baseline_metrics.json"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/genetic_algorithm"),
    )
    parser.add_argument("--max-patients", type=int, default=500)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--mutation-rate", type=float, default=0.25)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_feedback(args.baseline)
    baseline_errors = validate_baseline_artifact(
        baseline,
        args.input,
        args.model,
    )
    if baseline_errors:
        raise ValueError(
            "Canonical baseline validation failed: "
            + "; ".join(baseline_errors)
        )

    baseline_policy = baseline["policy"]
    baseline_bounds = derive_ga_bounds(baseline_policy)
    feedback = load_feedback(args.feedback)
    bounds, objectives = apply_feedback(feedback, baseline_bounds)

    safe_print("\nApplied GA bounds:")
    safe_print(json.dumps({k: list(v) for k, v in bounds.items()}, indent=2))
    safe_print("\nApplied objectives:")
    safe_print(json.dumps(objectives, indent=2))

    if feedback.get("summary"):
        safe_print("\nFeedback interpretation:")
        safe_print(feedback["summary"])

    patients = prepare_patients(
        args.input,
        args.model,
        args.max_patients,
    )

    safe_print(
        f"\nRunning GA with population={args.population}, "
        f"generations={args.generations}, patients={len(patients)}"
    )

    best_policy, best_metrics, history = run_ga(
        patients,
        bounds=bounds,
        objectives=objectives,
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        elite_count=args.elite_count,
        seed=args.seed,
    )

    best_fitness = calculate_fitness(
        best_metrics,
        best_policy,
        objectives,
    )

    output = {
        "best_policy": asdict(best_policy),
        "best_fitness": best_fitness,
        "best_metrics": best_metrics,
        "applied_bounds": {
            k: list(v)
            for k, v in bounds.items()
        },
        "applied_objectives": objectives,
        "feedback_summary": feedback.get("summary"),
        "baseline_provenance": {
            "simulator_version": baseline["baseline_metadata"]["simulator_version"],
            "hospital_scale_multiplier": baseline["baseline_metadata"][
                "hospital_scale_multiplier"
            ],
            "baseline_generated_at_utc": baseline["baseline_metadata"][
                "generated_at_utc"
            ],
            "baseline_policy": {
                key: int(baseline_policy[key])
                for key in ["icu_beds", "general_beds", "doctors", "nurses"]
            },
            "ga_search_bounds": {
                key: list(value)
                for key, value in baseline_bounds.items()
            },
            "workload_scale_matches_canonical": bool(
                float(baseline["baseline_metadata"]["hospital_scale_multiplier"])
                == HOSPITAL_SCALE_MULTIPLIER
            ),
        },
    }

    (args.results_dir / "best_policy.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    history.to_csv(
        args.results_dir / "ga_history.csv",
        index=False,
    )

    best_by_generation = (
        history.sort_values(
            ["generation", "fitness"],
            ascending=[True, False],
        )
        .groupby("generation", as_index=False)
        .first()
    )
    best_by_generation.to_csv(
        args.results_dir / "best_by_generation.csv",
        index=False,
    )

    safe_print("\nGENETIC ALGORITHM COMPLETE")
    safe_print("\nBest policy:")
    safe_print(json.dumps(asdict(best_policy), indent=2))
    safe_print("\nBest fitness:")
    safe_print(round(best_fitness, 4))
    safe_print("\nBest Digital Twin metrics:")
    safe_print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()

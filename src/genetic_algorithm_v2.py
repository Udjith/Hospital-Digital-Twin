"""Scenario-based GA. No simulation implementation lives here; V1 remains intact."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd

from digital_twin import HospitalPolicy, file_sha256
from genetic_algorithm import random_policy, crossover, policy_key, tournament_selection
from scenario_evaluation import (
    DEFAULT_V2_RESOURCES, ScenarioConfig, evaluate_current_policy,
    load_patient_profiles, validate_policy,
)

VERSION = "2.0-scenario-ga"
GENES = ("icu_beds", "general_beds", "doctors", "nurses")
SCENARIOS = ("best", "average", "worst")
DEFAULT_BOUNDS = dict(icu_beds=(100, 400), general_beds=(150, 600),
                      doctors=(40, 180), nurses=(60, 220))
SEED_STRATEGY = "base_seed * 1000 + scenario_index * 100 + run_index; fixed across all candidates and generations"
RESOURCE_COSTS = dict(icu_beds=.35, general_beds=.15, doctors=1.20, nurses=.55)


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 12
    generations: int = 10
    ga_replications_per_scenario: int = 3
    seed: int = 42
    mutation_rate: float = .25
    elite_count: int = 2
    patience: int = 5
    improvement_tolerance: float = .01
    prioritize_high_risk: bool = True
    prioritize_efficiency: bool = True
    scenario_weights: tuple[float, float, float] = (1., 2., 3.)

    def validate(self):
        for name, low, high in (("population_size", 2, 100), ("generations", 1, 200),
                                ("ga_replications_per_scenario", 1, 10),
                                ("elite_count", 1, self.population_size), ("patience", 2, 200),
                                ("seed", 0, 2**32 - 1)):
            value = getattr(self, name)
            if not np.isfinite(value) or int(value) != value or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}.")
        if not np.isfinite(self.mutation_rate) or not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1.")
        if not np.isfinite(self.improvement_tolerance) or self.improvement_tolerance < 0:
            raise ValueError("Improvement tolerance must be finite and nonnegative.")
        if len(self.scenario_weights) != 3 or any(not np.isfinite(w) or w <= 0 for w in self.scenario_weights):
            raise ValueError("Best/Average/Worst weights must be finite and positive.")


class SearchScenarioConfig(ScenarioConfig):
    """Only search replication validation differs; Part 1 evaluation is reused."""
    def validate(self):
        values = asdict(self)
        values["replications_per_scenario"] = 3
        ScenarioConfig(**values).validate()
        n = self.replications_per_scenario
        if int(n) != n or not 1 <= n <= 10:
            raise ValueError("GA replications per scenario must be an integer from 1 to 10.")


def validate_bounds(bounds, current_policy, *, require_current_feasible=True):
    validate_policy(current_policy)
    if set(bounds) != set(GENES):
        raise ValueError("Bounds must specify ICU beds, General beds, doctors and nurses.")
    for name in GENES:
        limits = bounds[name]
        if len(limits) != 2:
            raise ValueError(f"{name} requires minimum and maximum bounds.")
        low, high = limits
        if any(not np.isfinite(v) or int(v) != v for v in limits) or not 1 <= low <= high <= 10000:
            raise ValueError(f"Invalid {name} bounds: require integer 1 <= minimum <= maximum <= 10000.")
        current = getattr(current_policy, name)
        if require_current_feasible and not low <= current <= high:
            raise ValueError(f"Current {name}={current} is outside optimization bounds [{low}, {high}]. Adjust the bounds or the current policy explicitly.")


def initial_population(current_policy, bounds, size, rng, *, allow_infeasible_current=False):
    validate_bounds(bounds, current_policy, require_current_feasible=not allow_infeasible_current)
    values = asdict(current_policy)
    if allow_infeasible_current:
        for name in GENES:
            values[name] = max(bounds[name][0], min(bounds[name][1], values[name]))
    return [HospitalPolicy(**values)] + [random_policy(rng, bounds) for _ in range(size - 1)]


def mutate_v2(policy, rng, bounds, mutation_rate=.25):
    values = asdict(policy)
    for name in GENES:
        low, high = bounds[name]
        if rng.random() < mutation_rate:
            # Integer local step scales to the explicitly allowed search range.
            radius = max(1, int(np.ceil((high - low) * .10)))
            values[name] = max(low, min(high, values[name] + rng.randint(-radius, radius)))
    return HospitalPolicy(**values)


def calculate_fitness_v2(evaluation, policy, bounds, options):
    """Exact coefficients and all contributions are exported in the result."""
    config = evaluation["configuration"]
    h = 2. if options.prioritize_high_risk else 1.
    cost_scale = 100. if options.prioritize_efficiency else 50.
    components = dict.fromkeys(("robust_policy_bonus", "robustness_reward", "condition_penalty",
                               "wait_excess_penalty", "utilization_excess_penalty", "delay_penalty",
                               "unfinished_penalty", "scenario_deficit_penalty", "resource_cost_penalty"), 0.)
    if all(s["scenario_verdict"] == "ACCEPTABLE" for s in evaluation["scenarios"].values()):
        components["robust_policy_bonus"] = 100000.
    weights = np.array(options.scenario_weights, dtype=float)
    weights /= weights.sum()
    for name, weight in zip(SCENARIOS, weights):
        scenario = evaluation["scenarios"][name]
        runs = scenario["runs"]
        for run in runs:
            fraction = float(weight / len(runs))
            components["robustness_reward"] += fraction * 10000 * (run["policy_verdict"] == "ACCEPTABLE")
            components["condition_penalty"] -= fraction * 2000 * sum(not v for v in run["condition_breakdown"].values())
            mean_excess = max(0., run["mean_wait"] - config["mean_wait_target_minutes"]) / max(1., config["mean_wait_target_minutes"])
            high_excess = max(0., run["high_risk_mean_wait"] - config["high_risk_wait_target_minutes"]) / max(1., config["high_risk_wait_target_minutes"])
            components["wait_excess_penalty"] -= fraction * 4000 * (mean_excess + h * high_excess)
            util_excess = sum(max(0., run[f"{key}_utilization"] - config["max_utilization_target"])
                              for key in ("icu", "general", "doctor", "nurse")) / max(.01, config["max_utilization_target"])
            components["utilization_excess_penalty"] -= fraction * 4000 * util_excess
            components["delay_penalty"] -= fraction * (2 * run["mean_wait"] + 4 * h * run["high_risk_mean_wait"] + .1 * run["p95_wait"])
            components["unfinished_penalty"] -= fraction * 5 * run["unfinished_patients"] / max(1, run["patients_arrived"])
        deficit = max(0., config["robustness_threshold"] - scenario["scenario_robustness"]) / 100
        components["scenario_deficit_penalty"] -= float(weight) * 2000 * deficit
    cost = sum(RESOURCE_COSTS[k] * getattr(policy, k) for k in GENES)
    max_cost = sum(RESOURCE_COSTS[k] * bounds[k][1] for k in GENES)
    components["resource_cost_penalty"] = -cost_scale * cost / max_cost
    return float(sum(components.values())), components


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def automatic_bounds(current_policy):
    """Exact floor(70%)/ceil(150%) integer bounds, refreshed from current policy."""
    return {gene: (max(1, 7 * getattr(current_policy, gene) // 10),
                   (3 * getattr(current_policy, gene) + 1) // 2) for gene in GENES}


def configuration_provenance(current_policy, config, bounds, options, dependencies=None, allow_infeasible_current=False):
    # JSON normalization also removes tuple/list differences in saved artifacts.
    return json.loads(json.dumps(dict(version=VERSION, current_policy=asdict(current_policy),
                     scenario_configuration=asdict(config), bounds=bounds, ga_configuration=asdict(options),
                     automatic_bounds=automatic_bounds(current_policy), effective_bounds=bounds,
                     allow_infeasible_current=allow_infeasible_current,
                     current_policy_feasible=all(bounds[k][0] <= getattr(current_policy, k) <= bounds[k][1] for k in GENES),
                     initialization_strategy="nearest feasible current policy then random feasible candidates" if allow_infeasible_current else "current policy then random feasible candidates",
                     seed_strategy=SEED_STRATEGY, dependencies=dependencies or {}), sort_keys=True))


def result_is_current(result, current_policy, config, bounds, options, dependencies=None, allow_infeasible_current=False):
    return bool(result and result.get("provenance") == configuration_provenance(
        current_policy, config, bounds, options, dependencies, allow_infeasible_current))


def dependency_fingerprints(data_path, model_path):
    paths = {"patient_data": Path(data_path), "rf_model": Path(model_path)}
    for name in ("digital_twin.py", "scenario_evaluation.py", "genetic_algorithm.py",
                 "genetic_algorithm_v2.py", "decision_tree_xai_v2.py", "feedback_interpreter.py", "llm_explanation.py"):
        paths[name] = Path(__file__).with_name(name)
    return {name: {"path": str(path.resolve()), "sha256": file_sha256(path)} for name, path in paths.items()}


def flatten_scenarios(evaluation, policy, generation, candidate_index, fitness, cache_hit):
    rows = []
    config = evaluation["configuration"]
    for name, scenario in evaluation["scenarios"].items():
        rows.append({"generation": generation, "candidate_index": candidate_index,
                         "policy_id": canonical_hash(asdict(policy)), **asdict(policy), **config,
                         "scenario": name, "arrival_rate": scenario["arrival_rate"],
                         "scenario_robustness": scenario["scenario_robustness"],
                         "scenario_verdict": scenario["scenario_verdict"],
                         "overall_robustness": evaluation["overall_robustness"],
                         "overall_policy_status": evaluation["overall_policy_status"],
                         **{f"{n}_robustness": s["scenario_robustness"] for n, s in evaluation["scenarios"].items()},
                         **{k: v["mean"] for k, v in scenario["metrics"].items()},
                         "fitness": fitness, "cache_hit": cache_hit})
    return rows


def explain_changes(current, recommended):
    """Descriptive comparisons, never invented causal attribution."""
    notes = []
    before, after = current["resource_configuration"], recommended["resource_configuration"]
    for gene, util in zip(GENES, ("icu", "general", "doctor", "nurse")):
        if before[gene] == after[gene]:
            continue
        comparisons = []
        for name in SCENARIOS:
            old, new = current["scenarios"][name], recommended["scenarios"][name]
            comparisons.append(f"{name.title()}: {util} utilization {old['metrics'][util + '_utilization']['mean']:.1%} -> {new['metrics'][util + '_utilization']['mean']:.1%}, mean wait {old['metrics']['mean_wait']['mean']:.2f} -> {new['metrics']['mean_wait']['mean']:.2f} min")
        notes.append(f"{gene.replace('_', ' ').title()} changed {before[gene]} -> {after[gene]}. " + "; ".join(comparisons) + ". These outcomes reflect the combined policy change; they do not isolate this resource's causal effect.")
    for name in SCENARIOS:
        old, new = current["scenarios"][name], recommended["scenarios"][name]
        notes.append(f"{name.title()} robustness changed {old['scenario_robustness']:.1f}% -> {new['scenario_robustness']:.1f}%; high-risk mean wait {old['metrics']['high_risk_mean_wait']['mean']:.2f} -> {new['metrics']['high_risk_mean_wait']['mean']:.2f} min.")
    return notes


def run_ga_v2(patient_df, current_policy, bounds, config=None, options=None,
              progress_callback=None, dependencies=None, allow_infeasible_current=False):
    config, options = config or ScenarioConfig(), options or GAConfig()
    config.validate()
    options.validate()
    validate_bounds(bounds, current_policy, require_current_feasible=not allow_infeasible_current)
    bounds = {k: tuple(map(int, bounds[k])) for k in GENES}
    provenance = configuration_provenance(current_policy, config, bounds, options, dependencies, allow_infeasible_current)
    # File fingerprints are used for UI compatibility. The in-memory hash audits
    # callers supplying a DataFrame directly, without altering its contents.
    profile_hash = hashlib.sha256(pd.util.hash_pandas_object(patient_df, index=True).values.tobytes()).hexdigest()
    search_config = SearchScenarioConfig(**{**asdict(config), "replications_per_scenario": options.ga_replications_per_scenario})
    rng = random.Random(options.seed)
    population = initial_population(current_policy, bounds, options.population_size, rng,
                                    allow_infeasible_current=allow_infeasible_current)
    cache, history, generations = {}, [], []
    best_policy, best_fitness, best_evaluation = None, float("-inf"), None
    stagnant, stopping_reference, stopped_early = 0, float("-inf"), False
    started = time.perf_counter()
    for generation in range(options.generations):
        scored = []
        for index, policy in enumerate(population):
            key = policy_key(policy)
            hit = key in cache
            if not hit:
                evaluation = evaluate_current_policy(patient_df, policy, search_config)
                fitness, components = calculate_fitness_v2(evaluation, policy, bounds, options)
                cache[key] = (fitness, evaluation, components)
            fitness, evaluation, components = cache[key]
            scored.append((policy, fitness, evaluation))
            history.extend(flatten_scenarios(evaluation, policy, generation, index, fitness, hit))
            if fitness > best_fitness:
                best_policy, best_fitness, best_evaluation = HospitalPolicy(**asdict(policy)), fitness, evaluation
        scored.sort(key=lambda item: item[1], reverse=True)
        if best_fitness > stopping_reference + options.improvement_tolerance:
            stagnant, stopping_reference = 0, best_fitness
        else:
            stagnant += 1
        record = dict(generation=generation, best_fitness=best_fitness,
                      overall_robustness=best_evaluation["overall_robustness"],
                      best_policy=asdict(best_policy), unique_evaluations=len(cache))
        generations.append(record)
        if progress_callback:
            progress_callback(record)
        if stagnant >= options.patience and generation + 1 < options.generations:
            stopped_early = True
            break
        if generation + 1 == options.generations:
            break
        population = [HospitalPolicy(**asdict(item[0])) for item in scored[:options.elite_count]]
        while len(population) < options.population_size:
            a, b = tournament_selection(scored, rng), tournament_selection(scored, rng)
            population.append(mutate_v2(crossover(a, b, rng), rng, bounds, options.mutation_rate))
    # One full verification of the chosen recommendation; never re-rank on it.
    verified = evaluate_current_policy(patient_df, best_policy, config)
    current_verified = verified if policy_key(best_policy) == policy_key(current_policy) else evaluate_current_policy(patient_df, current_policy, config)
    evaluations = [dict(policy=asdict(HospitalPolicy(*key[:4])), fitness=value[0],
                        fitness_components=value[2], evaluation=value[1]) for key, value in cache.items()]
    result = dict(schema_version=2, version=VERSION, provenance=provenance,
                  patient_profiles_sha256=profile_hash, best_policy=asdict(best_policy),
                  best_fitness=best_fitness, search_evaluation=best_evaluation,
                  verified_evaluation=verified, current_evaluation=current_verified,
                  evaluations=evaluations, history=history, generations=generations,
                  generations_completed=len(generations), early_stopping_occurred=stopped_early,
                  unique_candidate_evaluations=len(cache), cache_hits=options.population_size * len(generations) - len(cache),
                  final_verification_replications=config.replications_per_scenario)
    result["change_summary"] = explain_changes(current_verified, verified)
    result["run_id"] = canonical_hash(result)
    result["runtime_seconds"] = time.perf_counter() - started
    return result


def save_ga_v2(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["history"]).to_csv(directory / "ga_history.csv", index=False)
    pd.DataFrame(result["generations"]).to_csv(directory / "best_by_generation.csv", index=False)
    (directory / "best_policy.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="V2 GA; optional JSON experiment uses scenario_configuration, current_policy, bounds, ga_configuration.")
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--input", type=Path, default=Path("data/synthetic/synthetic_hospital.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/random_forest_pipeline.joblib"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/genetic_algorithm_v2"))
    args = parser.parse_args()
    values = json.loads(args.configuration.read_text()) if args.configuration else {}
    config = ScenarioConfig(**values.get("scenario_configuration", {}))
    options = GAConfig(**values.get("ga_configuration", {}))
    policy = HospitalPolicy(**values.get("current_policy", DEFAULT_V2_RESOURCES))
    bounds = values.get("bounds", automatic_bounds(policy))
    allow_infeasible = values.get("allow_infeasible_current", False)
    config.validate()
    options.validate()
    validate_bounds(bounds, policy, require_current_feasible=not allow_infeasible)
    result = run_ga_v2(load_patient_profiles(args.input, args.model), policy, bounds, config, options,
                       progress_callback=lambda row: print(json.dumps(row), flush=True),
                       dependencies=dependency_fingerprints(args.input, args.model),
                       allow_infeasible_current=allow_infeasible)
    save_ga_v2(result, args.results_dir)
    print(json.dumps({k: result[k] for k in ("best_policy", "best_fitness", "runtime_seconds", "generations_completed")}))


if __name__ == "__main__":
    main()

"""Optional live Groq availability parsing, followed by real constrained GA."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from feedback_interpreter import interpret_feedback_v2, apply_feedback_v2
from genetic_algorithm_v2 import GAConfig, automatic_bounds, run_ga_v2, result_is_current
from scenario_evaluation import DEFAULT_V2_RESOURCES, HospitalPolicy, ScenarioConfig, load_patient_profiles


def main():
    policy = HospitalPolicy(**DEFAULT_V2_RESOURCES)
    automatic = automatic_bounds(policy)
    config = ScenarioConfig(simulation_duration_hours=4, replications_per_scenario=3)
    options = GAConfig(population_size=4, generations=2, ga_replications_per_scenario=1)
    started = time.perf_counter()
    observations = []
    nurse_bounds = None
    for instruction, kind in (("only 10 nurses are available", "nurses"),
            ("do not use more than 60 doctors", "doctors"),
            ("try to use fewer nurses", "preference"),
            ("at least 120 doctors but no more than 100", "contradiction")):
        record = interpret_feedback_v2(instruction, automatic, config)
        if not record["available"]:
            print("Live feedback unavailable; mocked schema and constraint tests cover this path.")
            return
        row = {"instruction": instruction, "interpretation": record["interpretation"]}
        try:
            bounds, _, adjusted_options = apply_feedback_v2(record["interpretation"], automatic, policy, config, options)
        except ValueError as exc:
            assert kind == "contradiction", str(exc)
            row["validation_error"] = str(exc)
        else:
            assert kind != "contradiction", "Contradiction was not rejected"
            row["effective_bounds"] = bounds
            if kind == "nurses":
                assert bounds["nurses"] == (1, 10)
                nurse_bounds = bounds
            elif kind == "doctors":
                assert bounds["doctors"] == (59, 60)
            else:
                assert bounds == automatic
                assert adjusted_options.prioritize_efficiency
                assert all(value is None for limits in record["interpretation"]["resource_constraints"].values() for value in limits.values())
        observations.append(row)
        print(kind, "PASS", flush=True)
    profiles = load_patient_profiles(ROOT / "data/synthetic/synthetic_hospital.csv", ROOT / "models/random_forest_pipeline.joblib")
    dependencies = {"feedback": observations[0]["interpretation"]}
    result = run_ga_v2(profiles, policy, nurse_bounds, config, options,
                      dependencies=dependencies, allow_infeasible_current=True)
    assert result_is_current(result, policy, config, nurse_bounds, options, dependencies, allow_infeasible_current=True)
    assert not result_is_current(result, policy, config, automatic, options, dependencies, allow_infeasible_current=True)
    assert all(row["nurses"] <= 10 for row in result["history"])
    output = {"live_feedback": observations, "automatic_bounds": automatic,
        "effective_bounds": nurse_bounds, "previous_current_policy": asdict(policy),
        "recommended_policy": result["best_policy"], "generation_zero_first_policy": {
            gene: result["history"][0][gene] for gene in automatic},
        "current_policy_feasible": result["provenance"]["current_policy_feasible"],
        "verified_status": result["verified_evaluation"]["overall_policy_status"],
        "ga_runtime_seconds": result["runtime_seconds"], "total_runtime_seconds": time.perf_counter() - started,
        "all_candidates_feasible": True, "stale_detection": "PASS"}
    directory = ROOT / "results/part3_validation"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "availability_constraints.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "live_feedback"}, indent=2))


if __name__ == "__main__":
    main()

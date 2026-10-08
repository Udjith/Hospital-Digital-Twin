"""Small real-data V2 integration run. Preserves every legacy artifact."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from genetic_algorithm_v2 import (
    automatic_bounds, GAConfig, run_ga_v2, save_ga_v2, dependency_fingerprints,
    result_is_current,
)
from decision_tree_xai_v2 import train_xai_v2, save_xai_v2, xai_is_current
from scenario_evaluation import DEFAULT_V2_RESOURCES, HospitalPolicy, ScenarioConfig, load_patient_profiles


def main():
    root = Path(__file__).resolve().parents[1]
    data, model = root / "data/synthetic/synthetic_hospital.csv", root / "models/random_forest_pipeline.joblib"
    started = time.perf_counter()
    profiles = load_patient_profiles(data, model)
    load_seconds = time.perf_counter() - started
    config, options = ScenarioConfig(), GAConfig(population_size=12, generations=6)
    bounds = automatic_bounds(HospitalPolicy(**DEFAULT_V2_RESOURCES))
    dependencies = {**dependency_fingerprints(data, model), "automatic_bounds": bounds,
        "bound_strategy": "floor(0.70*current), ceil(1.50*current), minimum 1",
        "feedback": {"enabled": False, "instruction": "", "interpretation": {}, "available": None}}
    if "--reuse" in sys.argv:
        result = json.loads((root / "results/part3_validation/ga/best_policy.json").read_text())
        assert result_is_current(result, HospitalPolicy(**DEFAULT_V2_RESOURCES), config, bounds, options, dependencies)
    else:
        result = run_ga_v2(profiles, HospitalPolicy(**DEFAULT_V2_RESOURCES), bounds,
                           config, options, dependencies=dependencies,
                           progress_callback=lambda r: print(json.dumps(r), flush=True))
    assert result["current_evaluation"]["overall_robustness"] == 20 / 30 * 100
    assert result["verified_evaluation"]["overall_robustness"] > result["current_evaluation"]["overall_robustness"]
    assert all(len(s["runs"]) == 10 for s in result["verified_evaluation"]["scenarios"].values())
    report, tree, training = train_xai_v2(result)
    assert xai_is_current(report, result)
    output = root / "results/part3_validation/ga"
    xai_output = root / "results/part3_validation/xai"
    save_ga_v2(result, output)
    save_xai_v2(report, tree, training, xai_output)
    summary = dict(best_policy=result["best_policy"], best_fitness=result["best_fitness"],
                   current_robustness=result["current_evaluation"]["overall_robustness"],
                   recommended_robustness=result["verified_evaluation"]["overall_robustness"],
                   current_status=result["current_evaluation"]["overall_policy_status"],
                   recommended_status=result["verified_evaluation"]["overall_policy_status"],
                   scenarios={name: {"current": result["current_evaluation"]["scenarios"][name]["scenario_robustness"],
                                     "recommended": s["scenario_robustness"],
                                     "mean_wait": s["metrics"]["mean_wait"]["mean"],
                                     "high_risk_mean_wait": s["metrics"]["high_risk_mean_wait"]["mean"]}
                              for name, s in result["verified_evaluation"]["scenarios"].items()},
                   rf_load_seconds=load_seconds, runtime_seconds=result["runtime_seconds"],
                   generations_completed=result["generations_completed"],
                   unique_candidates=result["unique_candidate_evaluations"], cache_hits=result["cache_hits"],
                   xai_status=report["status"], xai_metrics=report["metrics"])
    print(json.dumps(summary, indent=2), flush=True)

    from streamlit.testing.v1 import AppTest
    # Match the saved real experiment. Do not rerun it just to validate rendering.
    app = AppTest.from_file(str(root / "src/dashboard.py"), default_timeout=60)
    app.session_state["v2_ga_result"] = result
    app.session_state["v2_xai_report"] = report
    app.session_state["ga_population"] = options.population_size
    app.session_state["ga_generations"] = options.generations
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["optimization_run"]
    assert any(m.label == "GA Search Fitness" for m in app.metric)
    # Demand, bounds, targets, priority and current-resource changes invalidate
    # the displayed recommendation and its explanation.
    next(w for w in app.number_input if w.label == "Worst Arrival Rate (patients/hour)").set_value(36.).run()
    assert not app.session_state["optimization_run"]
    assert any("stale" in w.value for w in app.warning)
    assert not any(m.label == "GA Search Fitness" for m in app.metric)
    assert not app.exception
    assert not any(w.label.startswith("Minimum ICU") for w in app.number_input)
    app.number_input(key="icu_beds_number").set_value(300).run()
    assert not app.session_state["optimization_run"]
    assert not app.exception

    # Exercise the actual Run Optimization action cheaply, reusing RF profiles
    # and preserving the earlier real-data evidence files.
    with patch("genetic_algorithm_v2.save_ga_v2"), patch("decision_tree_xai_v2.save_xai_v2"), patch.dict(os.environ, {"GROQ_API_KEY": ""}), patch("llm_explanation.Path.write_text"):
        action = AppTest.from_file(str(root / "src/dashboard.py"), default_timeout=60)
        action.session_state["ga_population"] = 4
        action.session_state["ga_generations"] = 1
        action.run()
        next(w for w in action.number_input if w.label == "Simulation Duration (hours)").set_value(4.)
        next(w for w in action.number_input if w.label == "Replications Per Scenario").set_value(3)
        next(w for w in action.number_input if w.label == "GA Replications Per Scenario").set_value(1)
        action.run()
        next(b for b in action.button if b.label == "Run Optimization").click().run(timeout=60)
        assert not action.exception, [e.value for e in action.exception]
        assert not action.error, [e.value for e in action.error]
        assert action.session_state["optimization_run"]
        assert action.session_state["v2_ga_result"]["final_verification_replications"] == 3
        next(b for b in action.button if b.label == "Use Recommended Values").click().run()
        assert not action.exception
    print("DASHBOARD: real metrics render; demand changes hide stale GA/XAI; automatic bounds replace manual widgets; Run Optimization and Use Recommended Values pass", flush=True)

    # CLI entry points stay separate from V1 and generate a coherent XAI bundle.
    with tempfile.TemporaryDirectory(prefix="ga_v2_cli_", dir=str(output)) as directory:
        assert Path(directory).resolve().is_relative_to(root)
        experiment = {"current_policy": asdict(HospitalPolicy(**DEFAULT_V2_RESOURCES)),
                      "bounds": bounds,
                      "scenario_configuration": {**asdict(config), "simulation_duration_hours": 4., "replications_per_scenario": 3},
                      "ga_configuration": {**asdict(options), "population_size": 4, "generations": 1, "ga_replications_per_scenario": 1}}
        configuration_file = Path(directory) / "configuration.json"
        configuration_file.write_text(json.dumps(experiment))
        subprocess.run([sys.executable, "src/genetic_algorithm_v2.py", "--configuration", str(configuration_file),
                        "--results-dir", directory], check=True, capture_output=True, text=True)
        subprocess.run([sys.executable, "src/decision_tree_xai_v2.py", "--ga-result", str(Path(directory) / "best_policy.json"),
                        "--results-dir", str(Path(directory) / "xai")], check=True, capture_output=True, text=True)
        assert (Path(directory) / "xai/explanation.json").exists()
    summary["dashboard_validation"] = True
    summary["cli_validation"] = True
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("V2 CLI checks passed. Legacy GA/XAI and model artifacts preserved.")


if __name__ == "__main__":
    main()

"""Small real-data integration validation; run from the repository root."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scenario_evaluation import (
    DEFAULT_V2_RESOURCES, ScenarioConfig, HospitalPolicy, load_patient_profiles, evaluate_current_policy,
    save_current_policy_results,
)


def main():
    start = time.perf_counter()
    data = Path("data/synthetic/synthetic_hospital.csv")
    model = Path("models/random_forest_pipeline.joblib")
    df = load_patient_profiles(data, model)
    outputs = {"profile_count": len(df), "rf_load_seconds": round(time.perf_counter() - start, 3)}
    c = ScenarioConfig(best_case_arrival_rate=5., average_case_arrival_rate=10.,
                       worst_case_arrival_rate=20., replications_per_scenario=3)
    for label, policy, config in [
        ("low_24h", HospitalPolicy(1, 1, 1, 1), c),
        ("high_24h", HospitalPolicy(500, 500, 100, 100), c),
        ("manual_4h", HospitalPolicy(30, 40, 10, 20), replace(c, simulation_duration_hours=4.)),
        ("default_24h", HospitalPolicy(**DEFAULT_V2_RESOURCES), ScenarioConfig()),
    ]:
        start = time.perf_counter()
        result = evaluate_current_policy(df, policy, config)
        outputs[label] = {
            "seconds": round(time.perf_counter() - start, 3),
            "overall_robustness": result["overall_robustness"],
            "status": result["overall_policy_status"],
            "scenarios": {name: {
                **{key: round(s["metrics"][key]["mean"], 2)
                   for key in ["patients_arrived", "patients_completed", "mean_wait", "high_risk_mean_wait"]},
                "robustness": s["scenario_robustness"],
            } for name, s in result["scenarios"].items()},
        }
        if label == "default_24h":
            save_current_policy_results(result, Path("results/digital_twin"), data, model)
        if label == "manual_4h":
            assert result == evaluate_current_policy(df, policy, config)
            changed = evaluate_current_policy(df, policy, replace(config, random_seed=43))
            assert result["scenarios"]["best"]["metrics"]["patients_arrived"] != changed["scenarios"]["best"]["metrics"]["patients_arrived"]
    assert outputs["low_24h"]["status"] == "UNACCEPTABLE"
    assert outputs["high_24h"]["overall_robustness"] > outputs["low_24h"]["overall_robustness"]
    try:
        replace(c, best_case_arrival_rate=11.).validate()
    except ValueError as exc:
        outputs["invalid_order"] = str(exc)
    else:
        raise AssertionError("Invalid order was accepted")
    print(json.dumps(outputs, indent=2))
    Path("results/digital_twin/v2_validation.json").write_text(json.dumps(outputs, indent=2))
    with tempfile.TemporaryDirectory(prefix="v2_cli_", dir="results/digital_twin") as directory:
        completed = subprocess.run([
            sys.executable, "src/digital_twin.py", "--simulation-duration-hours", "4",
            "--best-case-arrival-rate", "5", "--average-case-arrival-rate", "10",
            "--worst-case-arrival-rate", "20", "--replications-per-scenario", "3",
            "--results-dir", directory,
        ], check=True, capture_output=True, text=True)
        assert "overall_policy_status" in completed.stdout
        artifact = json.loads((Path(directory) / "current_policy_evaluation.json").read_text())
        assert artifact["configuration"]["simulation_duration_hours"] == 4
        assert "model_sha256" in artifact["provenance"]
        print("V2_CLI: valid artifact and provenance")


if __name__ == "__main__":
    main()

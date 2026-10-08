"""Measured default-demand experiment and optional live Groq integration check."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scenario_evaluation import DEFAULT_V2_RESOURCES, HospitalPolicy, ScenarioConfig, load_patient_profiles
from genetic_algorithm_v2 import GAConfig, automatic_bounds, run_ga_v2
from decision_tree_xai_v2 import train_xai_v2
from feedback_interpreter import interpret_feedback_v2, apply_feedback_v2
from llm_explanation import generate_explanation_v2


def main():
    started = time.perf_counter()
    profiles = load_patient_profiles(ROOT / "data/synthetic/synthetic_hospital.csv", ROOT / "models/random_forest_pipeline.joblib")
    policy, config = HospitalPolicy(**DEFAULT_V2_RESOURCES), ScenarioConfig()
    options = GAConfig(population_size=6, generations=3)
    bounds = automatic_bounds(policy)
    result = run_ga_v2(profiles, policy, bounds, config, options)
    report, _, _ = train_xai_v2(result)
    print("Default-demand search and full verification complete", flush=True)
    # Never print the key or raw SDK error details. The two optional requests are
    # the feature's normal schema-enforced calls, not separate diagnostic APIs.
    feedback = interpret_feedback_v2("Do not use more than 110 doctors. Prioritize high-risk patients. Try to make the worst-case scenario robust.", bounds, config)
    if feedback["available"]:
        effective, _, adjusted_options = apply_feedback_v2(feedback["interpretation"], bounds, policy, config, options)
        assert effective["doctors"][1] == 110
        assert adjusted_options.prioritize_high_risk
        assert adjusted_options.scenario_weights[2] > options.scenario_weights[2]
    print("Feedback integration checked", flush=True)
    explanation = generate_explanation_v2(result, report)
    assert explanation["authoritative_status"] == result["verified_evaluation"]["overall_policy_status"]
    output = dict(configuration=asdict(config), automatic_bounds=bounds, ga_configuration=asdict(options),
        current_policy=asdict(policy), recommended_policy=result["best_policy"],
        current_robustness=result["current_evaluation"]["overall_robustness"],
        recommended_robustness=result["verified_evaluation"]["overall_robustness"],
        status=result["verified_evaluation"]["overall_policy_status"],
        scenarios={name: {"current": result["current_evaluation"]["scenarios"][name]["scenario_robustness"],
            "recommended": s["scenario_robustness"], "verdict": s["scenario_verdict"]}
            for name, s in result["verified_evaluation"]["scenarios"].items()},
        ga_runtime_seconds=result["runtime_seconds"], total_runtime_seconds=time.perf_counter() - started,
        xai_status=report["status"], feedback=feedback, explanation=explanation,
        groq_key_configured=bool(os.getenv("GROQ_API_KEY")))
    directory = ROOT / "results/part3_validation"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "experiment.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key not in ("feedback", "explanation")}, indent=2))
    print("Groq feedback available:", feedback["available"], "Groq explanation available:", explanation["available"])


if __name__ == "__main__":
    main()

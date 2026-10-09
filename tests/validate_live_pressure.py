"""Compact live acceptance and adaptive improvement evidence; no network calls."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from test_live_pressure import prediction, transient, pressure_search
from test_live_policy_ga import hospital, waiting
from test_live_diagnosis import search
from live_hospital_state import LiveCapacity, LiveHospitalState, OperatingPolicy
from live_policy_control import DEMO_CAPACITY, NORMAL_POLICY_VALUES
from live_pressure import LiveTargets
from live_policy_ga import optimize_live_policy, LiveGAOptions
from live_policy_explanation import explain_live_policy, explain_live_tree
from scenario_evaluation import load_patient_profiles


def concise(result):
    return dict(verdict=result["verdict"], robustness=result["robustness"], metrics=result["aggregate"],
        run_verdicts=[run["verdict"] for run in result["runs"]])


def ga_example(result):
    explanation = explain_live_policy(result)
    return dict(outcome=result["outcome"], current_policy=result["current_policy"], optimized_policy=result["optimized_policy"],
        current=concise(result["current_verified"]), optimized=concise(result["optimized_verified"]),
        diagnosis=result["diagnosis"], xai=explain_live_tree(result), explanation=explanation["explanation"],
        human_options=["Apply Optimized Operating Policy", "Reject Recommendation", "Modify / Retry"],
        ga_runtime_seconds=result["runtime_seconds"], verified_candidate_count=result["verified_candidate_count"])


def main():
    started = time.perf_counter()
    brief = prediction(transient())
    full_brief = prediction(transient(True))
    overloaded = hospital(LiveCapacity(10, 10, 1, 10))
    waiting(overloaded, stay=90.).treatment_minutes = 50.
    overloaded._dispatch()
    sustained = prediction(overloaded)
    assert brief["verdict"] == full_brief["verdict"] == "HANDLED"
    assert sustained["verdict"] == "CANNOT HANDLE"

    state = hospital(LiveCapacity(10, 10, 1, 10))
    waiting(state, stay=120.).treatment_minutes = .25
    waiting(state, high=True, stay=120.).treatment_minutes = .25
    checkpoint = state.state_hash()
    success = optimize_live_policy(state, None, 120., 5, replace(LiveTargets(), high_risk_wait_target_minutes=.1),
        LiveGAOptions(population=4, generations=2, search_replications=1))
    assert state.state_hash() == checkpoint
    assert success["current_verified"]["robustness"] == 0 and success["optimized_verified"]["robustness"] == 100
    _, _, _, partial = search()
    assert partial["outcome"] == "PARTIAL IMPROVEMENT"
    _, _, pressure_result = pressure_search()

    # Real RF profiles are loaded once, without duplicating model/data artifacts.
    profiles = load_patient_profiles(ROOT / "data/synthetic/synthetic_hospital.csv", ROOT / "models/random_forest_pipeline.joblib")
    real = LiveHospitalState(profiles, LiveCapacity(**DEMO_CAPACITY), 4., 42, OperatingPolicy(**NORMAL_POLICY_VALUES))
    real.start(); real.advance_large_skip(720.)
    event = real.propose_event("PATIENT_SURGE", 30., dict(count=30))
    before = real.state_hash()
    runtime = time.perf_counter()
    forecast = prediction(real, event=event)
    forecast_runtime = time.perf_counter() - runtime
    assert real.state_hash() == before

    output = dict(transient_91_percent=concise(brief), transient_full_saturation=concise(full_brief),
        sustained_doctor_overload=concise(sustained), ga_success=ga_example(success), ga_partial=ga_example(partial),
        ga_sustained_pressure_improvement=ga_example(pressure_result),
        real_profile_surge=concise(forecast), real_profile_fixed_capacity=asdict(real.capacity),
        real_profile_source_unchanged=True, real_profile_forecast_runtime_seconds=forecast_runtime,
        runtime_seconds=time.perf_counter() - started, network_calls=0)
    destination = ROOT / "results/live_twin/pressure_validation.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(transient=brief["verdict"], sustained=sustained["verdict"],
        success_robustness=[success["current_verified"]["robustness"], success["optimized_verified"]["robustness"]],
        partial=partial["outcome"], real_profile_verdict=forecast["verdict"],
        success_ga_seconds=success["runtime_seconds"], partial_ga_seconds=partial["runtime_seconds"],
        real_profile_forecast_seconds=forecast_runtime, runtime_seconds=output["runtime_seconds"]), indent=2))


if __name__ == "__main__":
    main()

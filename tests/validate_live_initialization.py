"""Compact real-profile warm-start A-F validation; no provider calls or checkpoints on disk."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from scenario_evaluation import load_patient_profiles
from live_hospital_state import LiveHospitalState, LiveCapacity, OperatingPolicy
from live_initialization import WarmStartSettings
from live_policy_control import DEMO_CAPACITY, NORMAL_POLICY_VALUES
from live_lookahead import simulate_lookahead_from_current_state
from live_policy_ga import optimize_live_policy, LiveGAOptions
from live_event_context import ScenarioEvents, apply_event_context
from test_live_initialization import ownership
from test_live_scenarios import mocked_interpret, payload, event
from live_scenario_interpreter import scenario_context


def result_summary(prediction):
    a = prediction["aggregate"]
    return dict(verdict=prediction["verdict"], robustness=prediction["robustness"],
        mean_wait=a["mean_wait"], high_risk_mean_wait=a["high_risk_mean_wait"],
        queue_peak=a["queue_peak"], queue_end=a["queue_at_horizon_end"],
        recovery_minutes=a["recovery_time_minutes"])


def main():
    started = time.perf_counter()
    profiles = load_patient_profiles(ROOT / "data/synthetic/synthetic_hospital.csv", ROOT / "models/random_forest_pipeline.joblib")
    init_started = time.perf_counter()
    state = LiveHospitalState(profiles, LiveCapacity(**DEMO_CAPACITY), 4., 42,
        OperatingPolicy(**NORMAL_POLICY_VALUES), warm_start=WarmStartSettings())
    init_seconds = time.perf_counter() - init_started
    original_hash = state.state_hash()
    ownership(state)
    discharge_times = [p.expected_discharge_time for p in state.active_patients.values() if p.expected_discharge_time is not None]
    report = dict(A=dict(initial=state.initial_state, initial_live_arrivals=state._arrived,
        earliest_discharge_minutes=min(discharge_times), latest_discharge_minutes=max(discharge_times),
        distinct_discharge_times=len(set(discharge_times)), initialization_seconds=init_seconds))
    state.start(); state.advance_large_skip(1440.)
    ownership(state)
    report["B"] = dict(metrics=state.metrics(), resources=state.resource_summary())
    before = state.state_hash()
    mild = state.propose_event("PATIENT_SURGE", 20., dict(count=12))
    report["C"] = result_summary(simulate_lookahead_from_current_state(state, mild, 120., 5))
    assert state.state_hash() == before
    apply_event_context(state, mild); state.advance_large_skip(120.)
    ownership(state)
    assert state.event_metrics(mild.event_id)["surge_patients_introduced"] == 12
    surge = state.propose_event("PATIENT_SURGE", 25., dict(count=60))
    shortage = state.propose_event("DOCTOR_SHORTAGE", 120., dict(count=70))
    shortage.event_id = f"EVT-{state._stress_sequence + 2:06d}"
    stress = ScenarioEvents((surge, shortage))
    before = state.state_hash()
    optimized = optimize_live_policy(state, stress, 240., 3,
        options=LiveGAOptions(population=6, generations=2, search_replications=1))
    assert state.state_hash() == before
    assert optimized["current_verified"]["verdict"] != "HANDLED"
    report["D"] = dict(current=result_summary(optimized["current_verified"]),
        optimized=result_summary(optimized["optimized_verified"]), outcome=optimized["outcome"],
        physical_capacity=asdict(state.capacity), current_policy=optimized["current_policy"],
        optimized_policy=optimized["optimized_policy"], ga_runtime_seconds=optimized["runtime_seconds"],
        checkpoint_unchanged=True)
    parsed = mocked_interpret("A bus accident sends 40 patients over 20 minutes.", state, payload(event(count=40)))
    report["E"] = dict(result=result_summary(simulate_lookahead_from_current_state(state, scenario_context(parsed), 120., 5)),
        provider="mocked raw Groq-shaped JSON; no network", current_active_patients=len(state.active_patients))
    assert state.state_hash() == before
    state.reset()
    assert state.state_hash() == original_hash
    ownership(state)
    report["F"] = dict(same_initial_state_hash=True, initial_active=len(state.active_patients), live_arrivals=state._arrived)
    report["runtime_seconds"] = time.perf_counter() - started
    destination = ROOT / "results/live_twin/warm_start_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

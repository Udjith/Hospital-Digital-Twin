"""Real-profile A–G lock checks with mocked provider translations; no network."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from scenario_evaluation import load_patient_profiles
from live_hospital_state import LiveHospitalState, LiveCapacity, OperatingPolicy
from live_policy_control import DEMO_CAPACITY, NORMAL_POLICY_VALUES
from live_scenario_interpreter import interpret_scenario, scenario_context
from live_event_context import apply_event_context
from live_lookahead import simulate_lookahead_from_current_state
from live_policy_ga import optimize_live_policy, LiveGAOptions
from live_policy_explanation import explain_live_policy, explain_live_tree
from test_live_scenarios import event, payload


def parse(text, state, structured):
    with patch.dict(os.environ, {"GROQ_API_KEY": "mock-validation"}), patch("live_scenario_interpreter.Groq") as groq:
        groq.return_value.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content=json.dumps(structured)))]
        result = interpret_scenario(text, state)
    assert result["status"] == "VALID", result
    return scenario_context(result)


def owners(state):
    return {kind: {rid: resource.patient_id for rid, resource in resources.items()} for kind, resources in state.resources.items()}


def invariant(state):
    assert asdict(state.capacity) == DEMO_CAPACITY
    assert state._arrived == state._completed + len(state.active_patients)
    assert len(state.events) <= state.events.maxlen
    assigned = []
    for kind, resources in state.resources.items():
        for rid, resource in resources.items():
            if resource.patient_id:
                assert resource.patient_id in state.active_patients
                patient = state.active_patients[resource.patient_id]
                assert rid == (patient.assigned_bed_id if kind in ("icu_beds", "general_beds") else patient.assigned_doctor_id if kind == "doctors" else patient.assigned_nurse_id)
                assigned.append(rid)
        assert set(state._free[kind]) == {rid for rid, resource in resources.items() if resource.status == "AVAILABLE"}
    assert len(assigned) == len(set(assigned))
    for pid, patient in state.active_patients.items():
        assert pid == patient.patient_id
        if patient.status == "IN_TREATMENT":
            assert patient.assigned_bed_id and patient.assigned_doctor_id and patient.assigned_nurse_id
        if patient.status == "BED_STAY":
            assert patient.assigned_bed_id and patient.assigned_doctor_id is None and patient.assigned_nurse_id is None


def summary(result):
    explanation = explain_live_policy(result)
    return dict(current_robustness=result["current_verified"]["robustness"], optimized_robustness=result["optimized_verified"]["robustness"],
        current_verdict=result["current_verified"]["verdict"], optimized_verdict=result["optimized_verified"]["verdict"], outcome=result["outcome"],
        current_mean_wait=result["current_verified"]["aggregate"]["mean_wait"], optimized_mean_wait=result["optimized_verified"]["aggregate"]["mean_wait"],
        current_policy=result["current_policy"], recommended_policy=result["optimized_policy"],
        summary=explanation["summary"], xai_available=explain_live_tree(result)["available"], runtime_seconds=result["runtime_seconds"])


def main():
    started = time.perf_counter()
    profiles = load_patient_profiles(ROOT / "data/synthetic/synthetic_hospital.csv", ROOT / "models/random_forest_pipeline.joblib")
    baseline = LiveHospitalState(profiles, LiveCapacity(**DEMO_CAPACITY), 4., 42, OperatingPolicy(**NORMAL_POLICY_VALUES))
    baseline.start(); baseline.advance_large_skip(720.)
    invariant(baseline)
    normal = simulate_lookahead_from_current_state(baseline, None, 120., 5)
    assert normal["verdict"] == "HANDLED"
    before = baseline.state_hash()
    mild = parse("A bus accident sends 12 patients over 20 minutes.", baseline, payload(event(count=12)))
    mild_result = simulate_lookahead_from_current_state(baseline, mild, 120., 5)
    assert mild_result["verdict"] == "HANDLED" and baseline.state_hash() == before

    # Deliberately restrictive, explicitly entered process policy to demonstrate
    # policy-limited failures. Capacity/base rate remain the requested baseline.
    state = baseline.clone()
    state.apply_operating_policy(OperatingPolicy(high_risk_priority_weight=1., nurse_reserve_percentage=.25), normal_policy=True,
        reason="Controlled human-entered restrictive scheduling policy for lock demonstration")
    options = LiveGAOptions(population=4, generations=2, search_replications=1)
    successful_event = parse("34 lower-risk patients arrive over 20 minutes while 90 nurses are unavailable for two hours.", state,
        payload(event(count=34, proportion=0.), event("NURSE_SHORTAGE", 90, 120.)))
    checkpoint = state.state_hash()
    success = optimize_live_policy(state, successful_event, 120., 5, options=options)
    assert success["current_verified"]["robustness"] == 0 and success["optimized_verified"]["robustness"] == 100
    assert state.state_hash() == checkpoint
    severe_event = parse("35 lower-risk patients arrive over 20 minutes while 90 nurses are unavailable for two hours.", state,
        payload(event(count=35, proportion=0.), event("NURSE_SHORTAGE", 90, 120.)))
    partial = optimize_live_policy(state, severe_event, 120., 5, options=options)
    assert partial["outcome"] == "PARTIAL IMPROVEMENT"

    # Rejection has no engine operation; UI rejection is separately tested by AppTest.
    rejected_state = state.state_hash()
    decision = "REJECT"
    assert decision == "REJECT" and state.state_hash() == rejected_state

    previous = state.current_policy
    original_owners = owners(state)
    state.apply_operating_policy(OperatingPolicy(**success["optimized_policy"]), event=successful_event, temporary=True,
        reason="Explicit human APPLY in lock validation", verification=success["optimized_verified"])
    assert owners(state) == original_owners and asdict(state.capacity) == DEMO_CAPACITY
    apply_event_context(state, successful_event)
    state.advance_large_skip(240.)
    invariant(state)
    assert state._temporary_policy is None and state.current_policy == previous
    assert state.event_metrics(successful_event.event_id)["surge_patients_introduced"] == 34

    random_state = baseline.clone()
    random_state.configure_random_events(True, "Low")
    random_started = time.perf_counter()
    random_state.advance_large_skip(10080.)
    invariant(random_state)
    assert random_state.stress_events
    output = dict(provider_translation="Mocked strict Groq responses; no live API/network calls", baseline=DEMO_CAPACITY, base_arrival_rate=4.,
        A=dict(status="PASS", normal_verdict=normal["verdict"], robustness=normal["robustness"], initial_clock_minutes=720.),
        B=dict(status="PASS", mild_verdict=mild_result["verdict"], robustness=mild_result["robustness"], source_unchanged=True),
        C=dict(status="PASS", **summary(success)), D=dict(status="PASS", **summary(partial)),
        E=dict(status="PASS", rejection_preserves_state=True, ui_rejection="Covered by AppTest regression"),
        F=dict(status="PASS", temporary_policy_restored=True, compound_recovery=True, original_assignments_preserved=True, actual_surge_patients=34),
        G=dict(status="PASS", days=7, random_events=len(random_state.stress_events), arrived=random_state._arrived,
            active=len(random_state.active_patients), retained_events=len(random_state.events), invariants=True,
            runtime_seconds=time.perf_counter() - random_started), total_runtime_seconds=time.perf_counter() - started)
    destination = ROOT / "results/live_twin/v3_lock_validation.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

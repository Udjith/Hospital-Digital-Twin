"""Compact reload/UI and real-profile failure evidence; no Groq requests."""
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streamlit.testing.v1 import AppTest
import live_hospital_state as live
from live_policy_control import DEMO_CAPACITY, NORMAL_POLICY_VALUES
from live_policy_ga import LiveGAOptions, optimize_live_policy, optimization_is_current
from live_policy_explanation import explain_live_policy, explain_live_tree, explanation_is_current
from scenario_evaluation import ScenarioConfig, load_patient_profiles
from test_live_diagnosis import search


def button(app, label):
    return next(w for w in app.button if w.label == label)


def dashboard_check():
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}), patch("live_policy_explanation.Groq") as groq:
        app = AppTest.from_file(str(ROOT / "src/dashboard.py"), default_timeout=60).run()
        button(app, "Start Live Simulation").click().run()
        state = app.session_state["v3_live_hospital"]
        app.number_input(key="v3_manual_delta").set_value(60.).run()
        button(app, "Advance Simulation").click().run()
        app.number_input(key="v3_ga_population").set_value(4).run()
        app.number_input(key="v3_ga_generations").set_value(1).run()
        app.number_input(key="v3_ga_search_reps").set_value(1).run()
        before = state.state_hash()
        retained_policy = state.current_policy
        importlib.reload(live)
        assert type(retained_policy) is not live.OperatingPolicy
        app.run()
        assert not app.exception and state.state_hash() == before
        button(app, "Optimize Current Operating Policy").click().run()
        assert not app.exception, [e.value for e in app.exception]
        assert state.state_hash() == before
        assert any("Why the Optimized Policy Works" in h.value for h in app.subheader)
        explanation = app.session_state["v3_live_ai"]
        result = app.session_state["v3_live_ga_result"]
        assert explanation_is_current(explanation, result)
        assert any("Possible Operational Responses for Human Review" in m.value for m in app.markdown)
        importlib.reload(live)
        app.run()
        assert not app.exception and state.state_hash() == before
        assert not any("STALE" in w.value for w in app.warning)
        button(app, "Advance Simulation").click().run()
        assert any("STALE" in w.value for w in app.warning)
        groq.assert_not_called()
    # Exercise prominent failure diagnosis using a real retained live state.
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
        app = AppTest.from_file(str(ROOT / "src/dashboard.py"), default_timeout=60).run()
        for kind in DEMO_CAPACITY:
            app.number_input(key="v3_capacity_" + kind).set_value(1)
        app.number_input(key="v3_arrival_rate").set_value(60.)
        app.run()
        button(app, "Start Live Simulation").click().run()
        button(app, "Advance Simulation").click().run()
        app.number_input(key="v3_ga_population").set_value(4).run()
        app.number_input(key="v3_ga_generations").set_value(1).run()
        app.number_input(key="v3_ga_search_reps").set_value(1).run()
        button(app, "Optimize Current Operating Policy").click().run()
        assert not app.exception, [e.value for e in app.exception]
        assert any("Why This Policy Still Fails" in h.value for h in app.subheader)
        assert any("Primary Limiting Factor" in m.value for m in app.markdown)
        assert any(w.label == "Failed Verified Replications" for w in app.metric)
        assert app.session_state["v3_live_ai"]["diagnosis"]["blocking_conditions"]
    return "PASS"


def main():
    started = time.perf_counter()
    dashboard = dashboard_check()
    profiles = load_patient_profiles(ROOT / "data/synthetic/synthetic_hospital.csv",
                                     ROOT / "models/random_forest_pipeline.joblib")
    state = live.LiveHospitalState(profiles, live.LiveCapacity(**DEMO_CAPACITY), 4., 42,
                                   live.OperatingPolicy(**NORMAL_POLICY_VALUES))
    state.start(); state.advance_large_skip(720.)
    state.apply_event(state.propose_event("NURSE_SHORTAGE", 120., dict(count=130)))
    event = state.propose_event("PATIENT_SURGE", 30., dict(count=30, high_risk_proportion=.3))
    checkpoint = state.state_hash()
    options = LiveGAOptions(population=4, generations=2, search_replications=1)
    result = optimize_live_policy(state, event, 120., 5, options=options)
    assert state.state_hash() == checkpoint
    assert optimization_is_current(result, state, event, 120., 5, ScenarioConfig(), options)
    explanation = explain_live_policy(result)
    assert explanation["diagnosis"]["likely_constraint_type"] == "STAFF_BOTTLENECK"
    nurse = next(f for f in explanation["diagnosis"]["blocking_conditions"] if f["metric"] == "nurse_utilization")
    assert nurse["resource_pressure"]["peak_utilization"] == 1. and nurse["target"] == .9 and nurse["failed_replications"] == 5
    _, _, _, success = search(True)
    success_explanation = explain_live_policy(success)
    _, _, _, partial = search()
    partial_explanation = explain_live_policy(partial)
    hash_started = time.perf_counter()
    for _ in range(50):
        assert state.state_hash() == checkpoint
    hash_ms = (time.perf_counter() - hash_started) * 1000 / 50
    output = dict(dashboard=dashboard, hot_reload="PASS", fixed_capacity=asdict(state.capacity),
        source_clock_minutes=state.sim_time_minutes, real_profile_source_unchanged=True,
        real_profile_failure=dict(current_metrics=result["current_verified"]["aggregate"],
            optimized_metrics=result["optimized_verified"]["aggregate"], diagnosis=result["diagnosis"],
            explanation=explanation["explanation"], human_review=explanation["operational_responses"],
            ga_runtime_seconds=result["runtime_seconds"]),
        controlled_success=dict(current_robustness=success["current_verified"]["robustness"],
            optimized_robustness=success["optimized_verified"]["robustness"], explanation=success_explanation["explanation"],
            xai=success_explanation["structured_payload"]["decision_tree_xai"]),
        controlled_partial=dict(current_robustness=partial["current_verified"]["robustness"],
            optimized_robustness=partial["optimized_verified"]["robustness"],
            current_high_risk_wait=partial["current_verified"]["aggregate"]["high_risk_mean_wait"],
            optimized_high_risk_wait=partial["optimized_verified"]["aggregate"]["high_risk_mean_wait"],
            blockers=partial["diagnosis"]["blocking_conditions"], explanation=partial_explanation["explanation"]),
        canonical_hash_milliseconds=hash_ms,
        network_calls=0, runtime_seconds=time.perf_counter() - started)
    destination = ROOT / "results/live_twin/diagnosis_validation.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

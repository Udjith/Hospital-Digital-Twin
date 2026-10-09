"""Compact V3 Part 3 dashboard and real-profile evidence, without API calls."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streamlit.testing.v1 import AppTest
from live_hospital_state import LiveHospitalState, LiveCapacity, OperatingPolicy
from live_policy_control import DEMO_CAPACITY, NORMAL_POLICY_VALUES, SIMULATION_SPEEDS
from live_policy_ga import optimize_live_policy, LiveGAOptions
from live_lookahead import simulate_lookahead_from_current_state
from scenario_evaluation import load_patient_profiles


def button(app, label):
    return next(w for w in app.button if w.label == label)


def owners(state):
    return {kind: {rid: r.patient_id for rid, r in resources.items() if r.patient_id}
            for kind, resources in state.resources.items()}


def dashboard_check():
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}), patch("live_policy_explanation.Groq") as groq:
        app = AppTest.from_file(str(ROOT / "src/dashboard.py"), default_timeout=60).run()
        assert not app.exception
        for kind, value in DEMO_CAPACITY.items():
            assert app.number_input(key="v3_capacity_" + kind).value == value
        assert app.number_input(key="v3_arrival_rate").value == 4.
        assert app.selectbox(key="v3_speed").options == [f"{value}x" for value in SIMULATION_SPEEDS]
        assert app.number_input(key="v3_manual_delta").max == 43200.
        button(app, "Start Live Simulation").click().run()
        state = app.session_state["v3_live_hospital"]
        assert asdict(state.current_policy) == NORMAL_POLICY_VALUES
        app.number_input(key="v3_manual_delta").set_value(60.).run()
        button(app, "Advance Simulation").click().run()
        assert state.active_patients
        app.number_input(key="v3_ga_population").set_value(4).run()
        app.number_input(key="v3_ga_generations").set_value(1).run()
        app.number_input(key="v3_ga_search_reps").set_value(1).run()
        checkpoint = state.state_hash()
        button(app, "Optimize Current Operating Policy").click().run()
        assert not app.exception, [e.value for e in app.exception]
        result = app.session_state["v3_live_ga_result"]
        assert state.state_hash() == checkpoint
        assert len(result["optimized_verified"]["runs"]) == 5
        assert not button(app, "Apply Optimized Operating Policy").disabled
        previous_owners = owners(state)
        button(app, "Apply Optimized Operating Policy").click().run()
        assert owners(state) == previous_owners and asdict(state.capacity) == DEMO_CAPACITY
        assert state.policy_history[-1]["applied"]
        assert app.session_state["v3_policy_recommendations"][-1]["applied"]
        assert any("STALE" in w.value for w in app.warning)
        # A seven-day request creates debt; each callback advances at most an hour.
        app.number_input(key="v3_manual_delta").set_value(10080.).run()
        before = state.sim_time_minutes
        button(app, "Advance Simulation").click().run()
        skip = app.session_state["v3_skip_driver"]
        assert state.sim_time_minutes - before == 60. and skip.remaining_minutes == 10020.
        button(app, "Pause").click().run()
        frozen = state.sim_time_minutes
        app.run()
        assert state.sim_time_minutes == frozen and skip.paused
        button(app, "Cancel Remaining Manual Skip").click().run()
        assert skip.remaining_minutes == 0.
        button(app, "Reset Live Twin").click().run()
        assert not state.policy_history and "v3_live_ga_result" not in app.session_state
        assert "v3_policy_recommendations" not in app.session_state
        assert not app.exception
        groq.assert_not_called()
    return "PASS"


def prediction_summary(result):
    return dict(verdict=result["verdict"], robustness=result["robustness"],
                metrics=result["aggregate"], seeds=result["provenance"]["replication_seeds"])


def main():
    started = time.perf_counter()
    destination = ROOT / "results/live_twin/part3_validation.json"
    previous = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else None
    dashboard = dashboard_check()
    profiles = load_patient_profiles(ROOT / "data/synthetic/synthetic_hospital.csv",
                                     ROOT / "models/random_forest_pipeline.joblib")
    state = LiveHospitalState(profiles, LiveCapacity(**DEMO_CAPACITY), 4., 42,
                              OperatingPolicy(**NORMAL_POLICY_VALUES))
    state.start()
    state.advance_large_skip(720.)
    initial = dict(sim_time_minutes=state.sim_time_minutes, metrics=state.metrics(),
                   free_resources={k: v["available"] for k, v in state.resource_summary().items()})
    standalone = state.propose_event("PATIENT_SURGE", 30., dict(count=30, high_risk_proportion=.3))
    plain_prediction = simulate_lookahead_from_current_state(state, standalone, 120., 5)
    # A deliberate severe manual shortage, not a default random-event magnitude.
    shortage = state.propose_event("NURSE_SHORTAGE", 120., dict(count=90))
    state.apply_event(shortage)
    surge = state.propose_event("PATIENT_SURGE", 30., dict(count=30, high_risk_proportion=.3))
    checkpoint = state.state_hash()
    settings = LiveGAOptions()
    result = optimize_live_policy(state, surge, 120., 5, options=settings)
    assert state.state_hash() == checkpoint
    previous_policy, assigned = state.current_policy, owners(state)
    state.apply_operating_policy(OperatingPolicy(**result["optimized_policy"]),
        reason="Validated major synthetic stress demonstration", event=surge, temporary=True,
        fitness=result["search_fitness"], verification=result["optimized_verified"])
    assert owners(state) == assigned and asdict(state.capacity) == DEMO_CAPACITY
    state.apply_event(surge)
    state.advance_large_skip(180.)
    assert state.event_metrics(surge.event_id)["surge_patients_introduced"] == 30
    assert state.current_policy == previous_policy and state._temporary_policy is None
    assert state.policy_history[0]["restored"]
    lifecycle = [{k: row[k] for k in ("sim_time_minutes", "event_type", "event_id", "description")}
        for row in state.events if row["event_type"] in ("POLICY_CHANGED", "POLICY_RESTORED", "EVENT_ENDED")]
    # Keep extended baseline evidence small; compare identical seven-day futures.
    normal = LiveHospitalState(profiles, LiveCapacity(**DEMO_CAPACITY), 4., 42,
                               OperatingPolicy(**NORMAL_POLICY_VALUES))
    normal.start()
    normal.configure_random_events(True, "Low", ("PATIENT_SURGE", "ARRIVAL_RATE_SPIKE", "DOCTOR_SHORTAGE",
                                                 "NURSE_SHORTAGE", "ICU_BED_OUTAGE", "GENERAL_BED_OUTAGE"))
    incremental = normal.clone()
    skip_started = time.perf_counter()
    normal.advance_large_skip(10080.)
    skip_seconds = time.perf_counter() - skip_started
    for _ in range(168):
        incremental.advance_simulation(60.)
    assert normal.state_hash() == incremental.state_hash()
    output = dict(dashboard=dashboard, fixed_capacity=DEMO_CAPACITY, baseline_arrival_rate=4.,
        initial_state=initial, standalone_surge=prediction_summary(plain_prediction),
        combined_event=dict(surge=surge.specification(), shortage=shortage.specification()),
        ga_settings=asdict(settings), current_policy=result["current_policy"], optimized_policy=result["optimized_policy"],
        current_prediction=prediction_summary(result["current_verified"]),
        optimized_prediction=prediction_summary(result["optimized_verified"]),
        ga_runtime_seconds=result["runtime_seconds"], generations_completed=result["generations_completed"],
        distinct_candidates=result["unique_evaluations"], adaptive_status=result["adaptive_status"],
        original_assignments_preserved=True, actual_surge_patients=30, temporary_policy_restored=True,
        lifecycle=lifecycle, seven_day_low_random=dict(metrics=normal.metrics(), bounded_skip_seconds=skip_seconds,
            equals_incremental=True, retained_events=len(normal.events)),
        total_runtime_seconds=time.perf_counter() - started)
    if previous and previous.get("acceptance_schema") == result["provenance"]["version"] and previous["ga_settings"] == output["ga_settings"] and previous["current_prediction"]["seeds"] == output["current_prediction"]["seeds"]:
        for key in ("current_policy", "optimized_policy", "current_prediction", "optimized_prediction"):
            assert output[key] == previous[key], f"Fresh-process reproducibility failed for {key}."
        output["fresh_process_reproducibility"] = True
    output["acceptance_schema"] = result["provenance"]["version"]
    output["seed_strategy"] = result["provenance"]["seed_strategy"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

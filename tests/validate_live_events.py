"""Compact real-profile + Streamlit validation for V3 Part 2; no API calls."""
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
from live_hospital_state import LiveCapacity, LiveHospitalState
from live_lookahead import simulate_lookahead_from_current_state, preview_is_current
from scenario_evaluation import load_patient_profiles


def button(app, label):
    return next(w for w in app.button if w.label == label)


@patch("live_dashboard.wall_seconds", new=lambda: 0.)
def main():
    started = time.perf_counter()
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}), patch("feedback_interpreter.Groq") as groq:
        app = AppTest.from_file(str(ROOT / "src/dashboard.py"), default_timeout=60).run()
        assert not app.exception
        button(app, "Start Live Simulation").click().run()
        state = app.session_state["v3_live_hospital"]
        app.number_input(key="v3_manual_delta").set_value(60.).run()
        button(app, "Fast Forward").click().run()
        assert state.active_patients
        app.selectbox(key="v3_event_preset").set_value("Traffic Accident").run()
        button(app, "Load Preset").click().run()
        assert app.selectbox(key="v3_event_type").value == "PATIENT_SURGE"
        assert app.number_input(key="v3_event_count").value == 12
        button(app, "Prepare Proposed Event").click().run()
        before = state.state_hash()
        button(app, "Run Event Look-Ahead").click().run()
        assert not app.exception, [e.value for e in app.exception]
        assert before == state.state_hash()
        assert not button(app, "Apply Event").disabled
        app.number_input(key="mean_wait_number").set_value(21.).run()
        assert button(app, "Apply Event").disabled
        assert any("STALE" in w.value for w in app.warning)
        button(app, "Run Event Look-Ahead").click().run()
        assert not button(app, "Apply Event").disabled
        button(app, "Fast Forward").click().run()
        assert button(app, "Apply Event").disabled
        button(app, "Prepare Proposed Event").click().run()
        button(app, "Run Event Look-Ahead").click().run()
        before_arrivals = state._arrived
        event_id = app.session_state["v3_proposed_event"].event_id
        button(app, "Apply Event").click().run()
        assert not app.exception, [e.value for e in app.exception]
        assert event_id in state.stress_events
        assert state._arrived == before_arrivals  # spread event, not immediate
        app.number_input(key="v3_manual_delta").set_value(20.).run()
        button(app, "Fast Forward").click().run()
        assert state.event_metrics(event_id)["surge_patients_introduced"] == 12
        app.checkbox(key="v3_show_resources").set_value(True).run()
        app.selectbox(key="v3_filter_icu_beds").set_value("OUT_OF_SERVICE").run()
        assert not app.exception
        button(app, "Reset Live Twin").click().run()
        assert not state.stress_events and not state.active_patients
        assert "v3_lookahead_result" not in app.session_state
        groq.assert_not_called()

    data = load_patient_profiles(ROOT / "data/synthetic/synthetic_hospital.csv", ROOT / "models/random_forest_pipeline.joblib")
    demo = LiveHospitalState(data, LiveCapacity(8, 12, 6, 6), arrival_rate=12., seed=42)
    demo.start()
    demo.advance_simulation(60.)
    assert demo.active_patients
    free = {kind: summary["available"] for kind, summary in demo.resource_summary().items()}
    baseline = demo.state_hash()
    event = demo.propose_event("PATIENT_SURGE", 20., dict(count=12), "Traffic accident: +12 patients over 20 min")
    prediction_started = time.perf_counter()
    result = simulate_lookahead_from_current_state(demo, event, 120, 5)
    prediction_seconds = time.perf_counter() - prediction_started
    assert demo.state_hash() == baseline and preview_is_current(result, demo)
    assert result == simulate_lookahead_from_current_state(demo, event, 120, 5)
    event.lookahead_run = True
    demo.apply_event(event)
    demo.advance_simulation(20.)
    assert demo.event_metrics(event.event_id)["surge_patients_introduced"] == 12
    event_arrivals = [row for row in demo.events if row["event_id"] == event.event_id and row["event_type"] == "PATIENT_ARRIVED"]
    assert len(event_arrivals) == 12
    shortage = demo.propose_event("DOCTOR_SHORTAGE", 30., dict(count=3), "3 doctors unavailable for 30 min")
    ownership = {rid: r.patient_id for rid, r in demo.resources["doctors"].items()}
    demo.apply_event(shortage)
    for rid, pid in ownership.items():
        assert demo.resources["doctors"][rid].patient_id == pid
    during_shortage = demo.resource_summary()["doctors"]
    demo.advance_simulation(30.)
    assert demo.stress_events[shortage.event_id].status == "ENDED"
    assert not any(r.outage_event_id == shortage.event_id for r in demo.resources["doctors"].values())
    # Continue until a real event patient's normal treatment releases staff.
    event_patients = [p for p in demo.active_patients.values() if p.source_event_id == event.event_id]
    treatment_end = min((p.expected_treatment_completion for p in event_patients if p.expected_treatment_completion and p.expected_treatment_completion > demo.sim_time_minutes), default=demo.sim_time_minutes)
    while demo.sim_time_minutes < min(treatment_end, 540.):
        demo.advance_simulation(min(60., min(treatment_end, 540.) - demo.sim_time_minutes))
    lifecycle = [row for row in demo.events if row["event_id"] in (event.event_id, shortage.event_id)
        and row["event_type"] in ("EVENT_CREATED", "EVENT_APPLIED", "EVENT_ENDED", "RESOURCES_ASSIGNED", "RESOURCE_RELEASED")][:18]
    output = dict(dashboard="PASS", runtime_seconds=time.perf_counter() - started,
        lookahead_runtime_seconds=prediction_seconds, initial_sim_time=60., capacity=asdict(demo.capacity),
        current_free_resources=free, current_queue=result["provenance"]["current_queue"],
        predicted=result["aggregate"], robustness=result["robustness"], verdict=result["verdict"],
        replication_seeds=result["provenance"]["replication_seeds"],
        real_state_unchanged_by_preview=True, reproducible_prediction=True,
        event_patients_injected=len(event_arrivals), event_metrics=demo.event_metrics(event.event_id),
        during_shortage=during_shortage, shortage_restored=True, lifecycle=lifecycle)
    output_path = ROOT / "results/live_twin/part2_validation.json"
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

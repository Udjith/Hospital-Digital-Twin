"""V3 Streamlit persistence/locks and compact real-profile lifecycle evidence."""
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
import scenario_evaluation as scenarios
from live_hospital_state import LiveCapacity, LiveHospitalState


def button(app, label):
    return next(widget for widget in app.button if widget.label == label)


def main():
    started = time.perf_counter()
    scenarios._load_profiles.cache_clear()
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}), patch("feedback_interpreter.Groq") as groq, \
            patch("scenario_evaluation.add_random_forest_predictions", wraps=scenarios.add_random_forest_predictions) as inference:
        app = AppTest.from_file(str(ROOT / "src/dashboard.py"), default_timeout=60).run()
        assert not app.exception, [e.value for e in app.exception]
        assert [tab.label for tab in app.tabs] == ["Live Twin", "Dashboard", "Optimize", "Review"]
        assert inference.call_count == 0
        for kind in ("icu_beds", "general_beds", "doctors", "nurses"):
            app.number_input(key=f"v3_capacity_{kind}").set_value(1)
        app.number_input(key="v3_arrival_rate").set_value(60.)
        app.run()
        button(app, "Start Live Simulation").click().run()
        assert not app.exception, [e.value for e in app.exception]
        hospital = app.session_state["v3_live_hospital"]
        assert hospital.status == "RUNNING"
        assert inference.call_count == 1
        assert all(app.number_input(key=f"v3_capacity_{kind}").disabled for kind in asdict(hospital.capacity))
        button(app, "Advance Simulation").click().run()
        assert hospital.sim_time_minutes == 5.
        assert hospital.metrics()["currently_waiting"] > 0
        before = hospital.snapshot()
        for _ in range(3):
            app.run()
            assert app.session_state["v3_live_hospital"] is hospital
            assert hospital.snapshot() == before
        app.selectbox(key="v3_speed").set_value(60).run()
        app.number_input(key="icu_beds_number").set_value(211).run()
        assert hospital.capacity.icu_beds == 1  # V2 controls cannot alter live physical resources.
        assert hospital.snapshot() == before
        button(app, "Pause").click().run()
        assert hospital.status == "PAUSED"
        clock = hospital.sim_time_minutes
        app.run()
        assert hospital.sim_time_minutes == clock
        button(app, "Resume").click().run()
        assert hospital.status == "RUNNING"
        app.checkbox(key="v3_show_resources").set_value(True).run()
        assert not app.exception
        button(app, "Prepare Compact Live Snapshot").click().run()
        assert json.loads(app.session_state["v3_snapshot_download"])["sim_time_minutes"] == clock
        button(app, "Stop Live Simulation").click().run()
        assert hospital.status == "STOPPED"
        assert app.number_input(key="v3_capacity_nurses").disabled
        button(app, "Reset Live Twin").click().run()
        assert hospital.status == "READY"
        assert hospital.sim_time_minutes == 0
        assert not hospital.events and not hospital.active_patients
        assert not app.number_input(key="v3_capacity_nurses").disabled
        assert all(r.status == "AVAILABLE" for resources in hospital.resources.values() for r in resources.values())
        app.number_input(key="v3_capacity_icu_beds").set_value(2).run()
        button(app, "Start Live Simulation").click().run()
        assert not app.exception and not app.error, [e.value for e in app.exception]
        assert app.session_state["v3_live_hospital"].capacity.icu_beds == 2
        assert inference.call_count == 1  # Restart reuses RF cache, not repeated inference.
        groq.assert_not_called()

    # Real augmented profile, shortest existing stay: no invented clinical timing.
    data = scenarios.load_patient_profiles(ROOT / "data/synthetic/synthetic_hospital.csv", ROOT / "models/random_forest_pipeline.joblib")
    demo_profile = data.sort_values(["length_of_stay_hours", "treatment_time_min"]).head(1)
    demo = LiveHospitalState(demo_profile, LiveCapacity(1, 1, 1, 1), arrival_rate=12., seed=42)
    demo.start()
    demo.advance_simulation(20.)
    first = demo.active_patients["P000001"]
    end = first.expected_discharge_time + 1.
    while demo.sim_time_minutes < end:
        demo.advance_simulation(min(60., end - demo.sim_time_minutes))
    relevant = [event for event in demo.events if
        (event["event_type"] == "PATIENT_ARRIVED" and event["patient_id"] in ("P000001", "P000002", "P000003"))
        or (event["patient_id"] in ("P000001", "P000002") and event["event_type"] in
            ("PATIENT_QUEUED", "RESOURCES_ASSIGNED", "RESOURCE_RELEASED", "TREATMENT_COMPLETED", "PATIENT_DISCHARGED"))]
    assert any(e["event_type"] == "PATIENT_QUEUED" for e in relevant)
    assert first.status == "DISCHARGED"
    assert demo.active_patients["P000002"].status == "IN_TREATMENT"
    assert demo.active_patients["P000002"].treatment_start_time == first.discharge_time
    benchmark_started = time.perf_counter()
    benchmark = LiveHospitalState(data)
    benchmark.start()
    for _ in range(24):
        benchmark.advance_simulation(60.)
    benchmark_seconds = time.perf_counter() - benchmark_started
    output = dict(dashboard="PASS", rf_inference_calls_across_two_starts=1, network_calls=0,
        runtime_seconds=time.perf_counter() - started, demo_profile={"risk_probability": first.risk_probability,
            "treatment_minutes": first.treatment_minutes, "stay_minutes": first.stay_minutes},
        demo_sim_time_minutes=demo.sim_time_minutes, demo_metrics=demo.metrics(), events=relevant,
        benchmark_24h={"runtime_seconds": benchmark_seconds,
            "arrivals": benchmark.metrics()["total_patients_arrived"], "completed": benchmark.metrics()["completed_patients"],
            "active": benchmark.metrics()["currently_active"], "event_history_length": len(benchmark.events)})
    directory = ROOT / "results/live_twin"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "validation_summary.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

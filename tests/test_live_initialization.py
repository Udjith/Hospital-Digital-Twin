"""Warm incumbents share the existing lifecycle/checkpoint, not a second simulator."""
from dataclasses import asdict, replace
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from live_hospital_state import LiveHospitalState, LiveCapacity
from live_initialization import WarmStartSettings
from live_lookahead import simulate_lookahead_from_current_state, preview_is_current
from live_policy_ga import optimize_live_policy, LiveGAOptions
from live_scenario_interpreter import interpret_scenario, scenario_context
from test_live_scenarios import event, payload


def make(seed=42, settings=None, rate=4.):
    profiles = pd.DataFrame(dict(predicted_probability=[.1, .49, .5, .9],
        treatment_time_min=[30., 45., 60., 90.], length_of_stay_hours=[12., 24., 48., 72.]))
    return LiveHospitalState(profiles, LiveCapacity(20, 40, 10, 20), rate, seed,
        warm_start=settings or WarmStartSettings())


def ownership(state):
    for patient in state.active_patients.values():
        held = {kind: [r for r in resources.values() if r.patient_id == patient.patient_id]
                for kind, resources in state.resources.items()}
        if patient.status == "WAITING_FOR_RESOURCES":
            assert not any(held.values())
        else:
            assert len(held[patient.required_bed_type]) == 1
            assert len(held["icu_beds"]) + len(held["general_beds"]) == 1
            assert len(held["doctors"]) == len(held["nurses"]) == int(patient.status == "IN_TREATMENT")
            assert patient.high_risk == (patient.risk_probability >= .50)
        assert not patient.source_event_id if patient.initialized_patient else True
    for kind, resources in state.resources.items():
        assert len(state._free[kind]) == len(set(state._free[kind]))
        assert set(state._free[kind]) == {rid for rid, r in resources.items() if r.status == "AVAILABLE"}
        for r in resources.values():
            assert r.patient_id is None or r.patient_id in state.active_patients
    assert state._arrived + state.initial_state["active_patients"] == state._completed + len(state.active_patients)


class InitializationTests(unittest.TestCase):
    def test_nonempty_targets_fixed_capacity_and_ownership(self):
        state = make()
        self.assertEqual(asdict(state.capacity), dict(icu_beds=20, general_beds=40, doctors=10, nurses=20))
        self.assertEqual(state.initial_state["active_patients"], 21)
        self.assertEqual(state.initial_state["resources"]["icu_beds"]["occupied_or_busy"], 11)
        self.assertEqual(state.initial_state["resources"]["general_beds"]["occupied_or_busy"], 10)
        self.assertEqual(state.initial_state["in_treatment"], 2)
        self.assertTrue(state.initial_state["warnings"])
        ownership(state)

    def test_same_seed_exact_reproduction_different_seed_changes_timing(self):
        self.assertEqual(make().state_hash(), make().state_hash())
        self.assertNotEqual(make().state_hash(), make(seed=43).state_hash())

    def test_residual_events_staggered_and_consistent_negative_age(self):
        state = make()
        deadlines = []
        for p in state.active_patients.values():
            self.assertLess(p.arrival_time, 0)
            self.assertAlmostEqual(p.pre_start_age_minutes, -p.arrival_time)
            self.assertGreater(p.expected_discharge_time, 0)
            self.assertLess(p.expected_discharge_time, p.stay_minutes)
            self.assertEqual(p.observed_wait(0), 0.)
            if p.status == "IN_TREATMENT":
                self.assertTrue(0 < p.expected_treatment_completion < p.treatment_minutes)
            else:
                self.assertLessEqual(p.expected_treatment_completion, 0)
                self.assertIsNone(p.assigned_doctor_id)
                self.assertIsNone(p.assigned_nurse_id)
            deadlines.append(p.expected_discharge_time)
        self.assertEqual(len(deadlines), len(set(deadlines)))
        self.assertTrue(all(row[0] >= 0 for row in state._agenda))

    def test_real_lifecycle_releases_staff_before_bed_and_reuses_resources(self):
        state = make(rate=0.)
        patient = next(p for p in state.active_patients.values() if p.status == "IN_TREATMENT")
        bed, doctor, nurse = patient.assigned_bed_id, patient.assigned_doctor_id, patient.assigned_nurse_id
        state.start(); state.advance_large_skip(patient.expected_treatment_completion)
        self.assertEqual(patient.status, "BED_STAY")
        self.assertIsNone(state.resources["doctors"][doctor].patient_id)
        self.assertIsNone(state.resources["nurses"][nurse].patient_id)
        self.assertEqual(state.resources[patient.required_bed_type][bed].patient_id, patient.patient_id)
        state.advance_large_skip(patient.expected_discharge_time - state.sim_time_minutes)
        self.assertIsNone(state.resources[patient.required_bed_type][bed].patient_id)
        self.assertEqual(patient.status, "DISCHARGED")
        ownership(state)

    def test_queue_unassigned_age_consistent_and_live_wait_not_contaminated(self):
        state = make(settings=WarmStartSettings(initial_queue=3), rate=0.)
        self.assertEqual(len(state._waiting), 3)
        for p in state._waiting.values():
            self.assertTrue(0 <= p.observed_wait(0) <= 15)
        ownership(state)
        self.assertEqual(state.metrics()["total_patients_arrived"], 0)
        state.start(); state.advance_simulation(1.)
        self.assertEqual(state.metrics()["mean_wait_so_far"], 0.)
        ownership(state)

    def test_future_arrival_and_profile_rng_not_consumed_by_initialization(self):
        warm, empty = make(), make(settings=WarmStartSettings(enabled=False))
        self.assertEqual(warm._arrival_rng.bit_generator.state, empty._arrival_rng.bit_generator.state)
        self.assertEqual(warm._profile_rng.bit_generator.state, empty._profile_rng.bit_generator.state)
        self.assertEqual(warm._random_rng.bit_generator.state, empty._random_rng.bit_generator.state)
        warm.start(); empty.start()
        self.assertEqual(next(e[0] for e in warm._agenda if e[3] == "ARRIVAL"), next(e[0] for e in empty._agenda if e[3] == "ARRIVAL"))
        warm.advance_large_skip(1440.); empty.advance_large_skip(1440.)
        self.assertEqual(warm._arrived, empty._arrived)
        self.assertGreater(warm._arrived, 0)
        ownership(warm)

    def test_clones_preserve_state_rng_events_and_independent_forecast(self):
        state = make(settings=WarmStartSettings(initial_queue=2))
        state.start()
        before = state.state_hash()
        clone = state.checkpoint()
        self.assertEqual(clone.state_hash(), before)
        event = state.propose_event("PATIENT_SURGE", 20., dict(count=5))
        forecast = simulate_lookahead_from_current_state(state, event, 120., 2)
        self.assertEqual(forecast["provenance"]["warm_start"], state.initial_state)
        self.assertEqual(state.state_hash(), before)
        self.assertEqual(forecast["aggregate"]["event_metrics"]["surge_patients_introduced"], 5.)
        clone.advance_simulation(1.)
        self.assertNotEqual(clone.state_hash(), before)
        self.assertEqual(state.state_hash(), before)

    def test_manual_events_and_safe_busy_outages_restore(self):
        for event_type, kind in (("DOCTOR_SHORTAGE", "doctors"), ("NURSE_SHORTAGE", "nurses"),
                ("ICU_BED_OUTAGE", "icu_beds"), ("GENERAL_BED_OUTAGE", "general_beds")):
            state = make(rate=0.); state.start()
            owners = {rid: r.patient_id for rid, r in state.resources[kind].items()}
            state.apply_event(state.propose_event(event_type, 5., dict(count=getattr(state.capacity, kind))))
            for rid, pid in owners.items():
                if pid:
                    self.assertEqual(state.resources[kind][rid].patient_id, pid)
                    self.assertEqual(state.resources[kind][rid].status, "OUT_OF_SERVICE_PENDING")
            state.advance_simulation(5.)
            self.assertTrue(all(r.outage_event_id is None for r in state.resources[kind].values()))
            ownership(state)
        state = make(); state.start()
        state.apply_event(state.propose_event("ARRIVAL_RATE_SPIKE", 10., dict(arrival_rate=8.)))
        state.advance_simulation(10.)
        self.assertEqual(state.effective_arrival_rate, 4.)

    def test_automatic_events_preserve_warm_ownership(self):
        state = make(); state.start()
        state.configure_random_events(True, "Low", ["PATIENT_SURGE"])
        state.advance_large_skip(4320.)
        self.assertTrue(state.stress_events)
        ownership(state)
        self.assertLessEqual(len(state.events), 1000)

    def test_ga_and_natural_language_use_same_warm_checkpoint(self):
        state = make(); state.start()
        before = state.state_hash()
        with patch.dict(os.environ, {"GROQ_API_KEY": "test"}), patch("live_scenario_interpreter.Groq") as groq:
            groq.return_value.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content=json.dumps(payload(event(count=40)))))]
            parsed = interpret_scenario("A bus accident sends 40 patients over 20 minutes.", state)
        context = scenario_context(parsed)
        result = optimize_live_policy(state, context, 120., 2,
            options=LiveGAOptions(population=4, generations=1, search_replications=1))
        self.assertEqual(state.state_hash(), before)
        self.assertEqual(result["provenance"]["live_state_hash"], before)
        self.assertEqual(result["provenance"]["physical_capacity"], asdict(state.capacity))
        self.assertEqual(result["provenance"]["warm_start"], state.initial_state)

    def test_reset_reproduces_warm_empty_and_restored_warm(self):
        state = make(); before = state.state_hash()
        state.start(); state.advance_large_skip(1440.)
        state.reset()
        self.assertEqual(state.state_hash(), before)
        state.reset(warm_start=False)
        self.assertFalse(state.active_patients)
        self.assertFalse(state._agenda)
        self.assertTrue(all(v["occupied_or_busy"] == 0 for v in state.resource_summary().values()))
        state.reset(warm_start=True)
        self.assertEqual(state.state_hash(), before)

    def test_settings_hash_stale_and_snapshot_numeric_serializable(self):
        state = make(); state.start()
        forecast = simulate_lookahead_from_current_state(state, None, 60., 1)
        self.assertTrue(preview_is_current(forecast, state))
        state.warm_start_settings = replace(state.warm_start_settings, icu_occupancy_target=.60)
        self.assertFalse(preview_is_current(forecast, state))
        state.reset()
        snapshot = state.snapshot()
        self.assertTrue(snapshot["initial_state"]["enabled"])
        self.assertEqual(snapshot["live_state_hash"], state.state_hash())
        self.assertIsInstance(snapshot["active_patients"][0]["arrival_time"], float)
        json.dumps(snapshot, allow_nan=False)

    def test_invalid_settings_and_absent_profile_subset_fail_clearly(self):
        for settings in (WarmStartSettings(icu_occupancy_target=1.1), WarmStartSettings(initial_queue=6),
                WarmStartSettings(initial_queue=True), WarmStartSettings(doctor_busy_target=float("nan"))):
            with self.assertRaises(ValueError):
                make(settings=settings)
        profiles = pd.DataFrame(dict(predicted_probability=[.1], treatment_time_min=[30.], length_of_stay_hours=[3.]))
        with self.assertRaisesRegex(ValueError, "RF profiles routed to icu_beds"):
            LiveHospitalState(profiles, warm_start=WarmStartSettings())

    @patch("live_dashboard.wall_seconds", new=lambda: 0.)
    def test_dashboard_default_warm_reset_empty_and_restore(self):
        from streamlit.testing.v1 import AppTest
        def button(app, label):
            return next(b for b in app.button if b.label == label)
        app = AppTest.from_file(str(ROOT / "src/dashboard.py"), default_timeout=60).run()
        self.assertTrue(app.checkbox(key="v3_warm_enabled").value)
        button(app, "Start Live Simulation").click().run()
        state = app.session_state["v3_live_hospital"]
        self.assertEqual(state.initial_state["active_patients"], 259)
        self.assertEqual(state._arrived, 0)
        self.assertTrue(app.number_input(key="v3_warm_icu_occupancy_target").disabled)
        app.selectbox(key="v3_reset_initialization").set_value("Reset Empty").run()
        button(app, "Reset Live Twin").click().run()
        self.assertFalse(state.active_patients)
        app.selectbox(key="v3_reset_initialization").set_value("Reset with Warm Start").run()
        button(app, "Reset Live Twin").click().run()
        self.assertEqual(state.initial_state["active_patients"], 259)
        app.number_input(key="v3_warm_icu_occupancy_target").set_value(60).run()
        button(app, "Reset Live Twin").click().run()
        self.assertEqual(state.initial_state["resources"]["icu_beds"]["occupied_or_busy"], 132)
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()

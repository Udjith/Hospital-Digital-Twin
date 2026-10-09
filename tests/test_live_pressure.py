"""Operational pressure acceptance, verification, evidence and human review."""
from dataclasses import asdict, replace
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_live_policy_ga import hospital, waiting
from test_live_diagnosis import search
from live_hospital_state import LiveCapacity, OperatingPolicy
from live_pressure import PressureIntegrator, LiveTargets, live_run_verdict
from live_lookahead import simulate_lookahead_from_current_state, preview_is_current
from live_policy_ga import optimize_live_policy, LiveGAOptions, live_fitness, optimization_is_current
from live_policy_explanation import explain_live_tree, explain_live_policy


def prediction(state, horizon=120., targets=None, event=None):
    return simulate_lookahead_from_current_state(state, event, horizon, 3, targets or LiveTargets())


def transient(full=False):
    state = hospital(LiveCapacity(50, 50, 1 if full else 34, 50))
    for _ in range(1 if full else 31):
        patient = waiting(state, stay=240.)
        patient.treatment_minutes = 4.
    state._dispatch()
    return state


def pressure_search():
    state = hospital(LiveCapacity(10, 100, 11, 100))
    event = state.propose_event("DOCTOR_SHORTAGE", 60., dict(count=1))
    state.apply_event(event)
    for _ in range(10):
        waiting(state, stay=120.).treatment_minutes = 4.
    state._dispatch()
    for _ in range(10):
        waiting(state, stay=120.).treatment_minutes = 20.
    waiting(state, high=True, stay=120.).treatment_minutes = 20.
    return state, event, optimize_live_policy(state, event, 120., 5, LiveTargets(),
        LiveGAOptions(population=4, generations=2, search_replications=1))


class LivePressureTests(unittest.TestCase):
    def test_ga_reduces_continuous_doctor_saturation_from_current_busy_checkpoint(self):
        state, _, result = pressure_search()
        before, after = result["current_verified"], result["optimized_verified"]
        self.assertEqual(before["robustness"], 0.)
        self.assertEqual(after["robustness"], 100.)
        self.assertEqual(before["aggregate"]["resource_pressure"]["doctor"]["longest_full_saturation_streak"], 24.)
        self.assertEqual(after["aggregate"]["resource_pressure"]["doctor"]["longest_full_saturation_streak"], 4.)
        self.assertEqual(before["aggregate"]["high_risk_mean_wait"], 24.)
        self.assertEqual(after["aggregate"]["high_risk_mean_wait"], 4.)
        self.assertEqual(state.resource_summary()["doctors"]["occupied_or_busy"], 10)
        self.assertEqual(asdict(state.capacity), result["provenance"]["physical_capacity"])

    def test_ga_solves_fifteen_second_high_risk_delay_despite_transient_full_peak(self):
        state = hospital(LiveCapacity(10, 10, 1, 10))
        waiting(state, stay=120.).treatment_minutes = .25
        waiting(state, high=True, stay=120.).treatment_minutes = .25
        targets = replace(LiveTargets(), high_risk_wait_target_minutes=.1)
        result = optimize_live_policy(state, None, 120., 5, targets,
            LiveGAOptions(population=4, generations=2, search_replications=1))
        self.assertEqual(result["current_verified"]["aggregate"]["high_risk_mean_wait"], .25)
        self.assertEqual(result["optimized_verified"]["aggregate"]["high_risk_mean_wait"], 0.)
        self.assertEqual(result["optimized_verified"]["robustness"], 100.)
        self.assertEqual(result["outcome"], "VERIFIED SUCCESS")
        self.assertEqual(result["optimized_verified"]["aggregate"]["resource_pressure"]["doctor"]["longest_full_saturation_streak"], .5)

    def test_dashboard_apply_reject_modify_and_grace_stale_paths(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parents[1]
        def button(app, label):
            return next(widget for widget in app.button if widget.label == label)
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}), patch("live_policy_explanation.Groq") as groq:
            app = AppTest.from_file(str(root / "src/dashboard.py"), default_timeout=60).run()
            button(app, "Start Live Simulation").click().run()
            app.number_input(key="v3_ga_population").set_value(4)
            app.number_input(key="v3_ga_generations").set_value(1)
            app.number_input(key="v3_ga_search_reps").set_value(1)
            app.run()
            button(app, "Optimize Current Operating Policy").click().run()
            self.assertFalse(app.exception)
            state = app.session_state["v3_live_hospital"]
            before = state.state_hash()
            button(app, "Reject Recommendation").click().run()
            self.assertEqual(before, state.state_hash())
            self.assertTrue(button(app, "Apply Optimized Operating Policy").disabled)
            button(app, "Modify / Retry").click().run()
            self.assertTrue(any("Modify / Retry:" in message.value for message in app.info))
            app.number_input(key="v3_overload_grace").set_value(45.).run()
            self.assertTrue(any("STALE live policy" in message.value for message in app.warning))
            button(app, "Optimize Current Operating Policy").click().run()
            capacity = asdict(state.capacity)
            owners = copy.deepcopy(state.resources)
            button(app, "Apply Optimized Operating Policy").click().run()
            self.assertEqual(asdict(state.capacity), capacity)
            self.assertEqual(state.resources, owners)
            self.assertTrue(state.policy_history)
            self.assertFalse(app.exception)
            groq.assert_not_called()

    def test_known_intervals_exact_time_weighted_area_and_streaks(self):
        pressure = PressureIntegrator(.9)
        for utilization, minutes in ((.5, 10), (1., 4), (.95, 3), (.2, 3), (1., 2)):
            pressure.integrate(utilization, minutes)
        values = pressure.result()
        self.assertAlmostEqual(values["time_weighted_utilization"], 14.45 / 22)
        self.assertEqual(values["minutes_above_target"], 9.)
        self.assertEqual(values["longest_above_target_streak"], 7.)
        self.assertEqual(values["minutes_at_full_saturation"], 6.)
        self.assertEqual(values["longest_full_saturation_streak"], 4.)

    def test_transient_91_percent_is_warning_not_failure(self):
        result = prediction(transient())
        run = result["runs"][0]
        self.assertAlmostEqual(run["doctor_utilization"], 31 / 34)
        self.assertEqual(run["resource_pressure"]["doctor"]["longest_above_target_streak"], 4.)
        self.assertIn("doctor", run["resource_warnings"])
        self.assertEqual(run["verdict"], "HANDLED_RUN")
        self.assertEqual(result["robustness"], 100.)

    def test_brief_full_saturation_and_valid_bed_stay_are_handled(self):
        result = prediction(transient(True))
        self.assertEqual(result["verdict"], "HANDLED")
        self.assertEqual(result["aggregate"]["unfinished_patients"], 1.)
        self.assertEqual(result["aggregate"]["queue_at_horizon_end"], 0.)
        self.assertAlmostEqual(result["aggregate"]["resource_pressure"]["doctor"]["time_weighted_utilization"], 4 / 120)

    def test_sustained_overload_fails_even_with_low_average(self):
        # Configure fixture treatment before acquisition.
        state = hospital(LiveCapacity(10, 10, 1, 10))
        waiting(state, stay=90.).treatment_minutes = 50.
        state._dispatch()
        result = prediction(state)
        run = result["runs"][0]
        self.assertEqual(run["verdict"], "FAILED_RUN")
        self.assertLess(run["resource_pressure"]["doctor"]["time_weighted_utilization"], .9)
        self.assertEqual(run["resource_pressure"]["doctor"]["longest_above_target_streak"], 50.)

    def test_critical_saturation_independently_fails(self):
        state = hospital(LiveCapacity(10, 10, 1, 10))
        waiting(state).treatment_minutes = 20.
        state._dispatch()
        result = prediction(state)
        run = result["runs"][0]
        self.assertTrue(run["resource_pressure"]["doctor"]["longest_above_target_streak"] <= 30)
        self.assertFalse(run["condition_breakdown"]["doctor_util_pass"])
        self.assertEqual(run["resource_pressure"]["doctor"]["longest_full_saturation_streak"], 20.)

    def test_sustained_91_percent_without_full_saturation_fails(self):
        state = hospital(LiveCapacity(50, 50, 34, 50))
        for _ in range(31):
            waiting(state, stay=120.).treatment_minutes = 40.
        state._dispatch()
        run = prediction(state)["runs"][0]
        pressure = run["resource_pressure"]["doctor"]
        self.assertLess(pressure["time_weighted_utilization"], .9)
        self.assertEqual(pressure["longest_above_target_streak"], 40.)
        self.assertEqual(pressure["minutes_at_full_saturation"], 0.)
        self.assertFalse(run["condition_breakdown"]["doctor_util_pass"])

    def test_time_weighted_average_can_fail_without_long_streak(self):
        state = transient(True)
        run = copy.deepcopy(prediction(state)["runs"][0])
        run["resource_pressure"]["doctor"]["time_weighted_utilization"] = .94
        verdict, conditions = live_run_verdict(run, LiveTargets())
        self.assertEqual(verdict, "FAILED_RUN")
        self.assertFalse(conditions["doctor_util_pass"])

    def test_exact_grace_boundary_passes_and_config_changes_stale(self):
        state = hospital(LiveCapacity(10, 10, 1, 10))
        waiting(state).treatment_minutes = 15.
        state._dispatch()
        result = prediction(state)
        self.assertEqual(result["verdict"], "HANDLED")
        changed = replace(LiveTargets(), critical_saturation_grace_minutes=14.)
        self.assertFalse(preview_is_current(result, state, targets=changed))
        self.assertEqual(prediction(state, targets=changed)["verdict"], "CANNOT HANDLE")

    def test_outage_boundaries_use_usable_capacity_without_interrupting_owners(self):
        state = hospital(LiveCapacity(10, 10, 2, 10))
        patient = waiting(state); patient.treatment_minutes = 20.
        state._dispatch()
        owner = patient.assigned_doctor_id
        event = state.propose_event("DOCTOR_SHORTAGE", 10., dict(count=1))
        before = state.state_hash()
        result = prediction(state, 60., event=event)
        pressure = result["aggregate"]["resource_pressure"]["doctor"]
        self.assertAlmostEqual(pressure["time_weighted_utilization"], .25)
        self.assertEqual(pressure["longest_full_saturation_streak"], 10.)
        self.assertEqual(result["verdict"], "HANDLED")
        self.assertEqual(state.state_hash(), before)
        self.assertEqual(patient.assigned_doctor_id, owner)

    def test_genuine_queue_and_event_waiting_fail_recovery(self):
        state = hospital(LiveCapacity(10, 10, 1, 10))
        event = state.propose_event("PATIENT_SURGE", 0., dict(count=8))
        result = prediction(state, 5., targets=replace(LiveTargets(), mean_wait_target_minutes=100., high_risk_wait_target_minutes=100.), event=event)
        self.assertFalse(result["runs"][0]["condition_breakdown"]["recovery_pass"])
        self.assertEqual(result["verdict"], "CANNOT HANDLE")
        self.assertGreater(result["runs"][0]["unresolved_event_waiting"], 0)

    def test_reproducibility_and_original_state_rng_unchanged(self):
        state = hospital(rate=4.)
        before = state.state_hash()
        self.assertEqual(prediction(state), prediction(state))
        self.assertEqual(state.state_hash(), before)

    def test_best_verified_partial_policy_is_returned_and_all_candidates_verified(self):
        state, targets, options, result = search()
        self.assertEqual(result["outcome"], "PARTIAL IMPROVEMENT")
        self.assertLess(result["optimized_verified"]["aggregate"]["high_risk_mean_wait"], result["current_verified"]["aggregate"]["high_risk_mean_wait"])
        self.assertEqual(result["verified_candidate_count"], result["unique_evaluations"])
        self.assertTrue(optimization_is_current(result, state, None, 60., 5, targets, options))
        for observation in result["evaluations"]:
            policy = OperatingPolicy(**observation["policy"])
            experiment = simulate_lookahead_from_current_state(state, None, 60., 5, targets, result["experiment_seed"], policy)
            score, _ = live_fitness(experiment, policy, targets, 60., result["current_verified"])
            self.assertGreaterEqual(result["verified_fitness"] + 1e-9, score)

    def test_fitness_penalizes_sustained_pressure_not_peak_alone(self):
        state = transient(True)
        evaluation = prediction(state)
        score, components = live_fitness(evaluation, state.current_policy, LiveTargets(), 120.)
        self.assertEqual(components["violation_penalty"], 0.)
        self.assertEqual(components["failed_condition_penalty"], 0.)
        self.assertLess(components["pressure_duration_penalty"], 2.)
        state = hospital(LiveCapacity(10, 10, 1, 10)); waiting(state, stay=90.).treatment_minutes = 50.
        state._dispatch()
        worse, _ = live_fitness(prediction(state), state.current_policy, LiveTargets(), 120.)
        self.assertGreater(score, worse + 5000)

    def test_xai_labels_and_groq_payload_use_sustained_evidence(self):
        _, _, _, result = search(True)
        self.assertEqual(result["outcome"], "VERIFIED SUCCESS")
        xai = explain_live_tree(result)
        self.assertTrue(xai["available"])
        for observation in result["evaluations"]:
            for run in observation["evaluation"]["runs"]:
                self.assertEqual(run["verdict"], live_run_verdict(run, LiveTargets(**result["provenance"]["targets"]))[0])
        with patch.dict(os.environ, {"GROQ_API_KEY": "test"}), patch("live_policy_explanation.Groq") as groq:
            request = groq.return_value.chat.completions.create
            request.return_value.choices = [MagicMock(message=MagicMock(content='{"themes":["diagnosis"]}'))]
            explanation = explain_live_policy(result, True)
            payload = json.loads(request.call_args.kwargs["messages"][1]["content"])
            self.assertIn("resource_pressure", payload["optimized_policy"]["verified_metrics"])
            self.assertEqual(payload["optimized_policy"]["verdict"], "HANDLED")
            request.return_value.choices = [MagicMock(message=MagicMock(content='{"verdict":"CANNOT HANDLE"}'))]
            self.assertEqual(explain_live_policy(result, True)["provider"], "deterministic fallback")

    def test_apply_partial_policy_preserves_capacity_and_existing_assignments(self):
        state, _, _, result = search()
        state._dispatch()
        ownership = {kind: {rid: asdict(resource) for rid, resource in resources.items()} for kind, resources in state.resources.items()}
        capacity = asdict(state.capacity)
        state.apply_operating_policy(OperatingPolicy(**result["optimized_policy"]), verification=result["optimized_verified"])
        self.assertEqual(asdict(state.capacity), capacity)
        self.assertEqual(ownership, {kind: {rid: asdict(resource) for rid, resource in resources.items()} for kind, resources in state.resources.items()})

    def test_random_and_manual_events_still_enter_prediction_without_mutation(self):
        state = hospital(rate=4.)
        state.configure_random_events(True, "High", ["PATIENT_SURGE"])
        state.advance_large_skip(180.)
        self.assertTrue(state.stress_events)
        event = next(iter(state.stress_events.values()))
        checkpoint = state.state_hash()
        prediction(state, event=event)
        self.assertEqual(checkpoint, state.state_hash())
        proposal = state.propose_event("PATIENT_SURGE", 5., dict(count=3))
        forecast = prediction(state, event=proposal)
        self.assertEqual(forecast["aggregate"]["event_metrics"]["surge_patients_introduced"], 3.)


if __name__ == "__main__":
    unittest.main()

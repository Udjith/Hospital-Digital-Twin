"""Fixed-capacity policy search, reserves, restoration and bounded clock tests."""
from dataclasses import asdict, replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from live_hospital_state import LiveCapacity, LiveHospitalState, LiveClockDriver, OperatingPolicy, Patient
from live_policy_control import GENES, POLICY_BOUNDS, SIMULATION_SPEEDS
from live_policy_ga import LiveGAOptions, optimize_live_policy, optimization_is_current, crossover_mutate, random_policy
from live_policy_explanation import explain_live_tree, explain_live_policy
from live_skip_control import LiveSkipDriver
from live_pressure import LiveTargets
from scenario_evaluation import ScenarioConfig


def hospital(capacity=None, rate=0., policy=None):
    profiles = pd.DataFrame(dict(predicted_probability=[.1, .9], treatment_time_min=[15., 15.], length_of_stay_hours=[.5, .5]))
    state = LiveHospitalState(profiles, capacity or LiveCapacity(3, 10, 4, 4), rate, operating_policy=policy)
    state.start()
    return state


def waiting(state, high=False, event=None, arrival=None, stay=30.):
    state._patient_sequence += 1
    arrival = state.sim_time_minutes if arrival is None else arrival
    patient = Patient(f"P{state._patient_sequence:06d}", arrival, .9 if high else .1, high,
        "icu_beds" if high else "general_beds", 15., stay, 3, queue_entry_time=arrival, source_event_id=event)
    state.active_patients[patient.patient_id] = state._waiting[patient.patient_id] = patient
    state._arrived += 1
    state._high_arrived += int(high)
    if event:
        stats = state.stress_events[event].statistics
        stats["arrived"] += 1
        stats["high_arrived"] = stats.get("high_arrived", 0) + int(high)
    return patient


def active_surge(state):
    event = state.propose_event("PATIENT_SURGE", 90, dict(count=1))
    event.status, event.applied = "ACTIVE", True
    state.stress_events[event.event_id] = event
    return event


class LivePolicyTests(unittest.TestCase):
    def test_priority_high_risk_aging_and_incident_flags(self):
        state = hospital(policy=OperatingPolicy(2., .1, surge_priority_strength=3.))
        event = active_surge(state)
        low = waiting(state, arrival=0.)
        state.sim_time_minutes = 40.
        high = waiting(state, high=True)
        incident = waiting(state, event=event.event_id)
        self.assertLess(state.queue_priority(low), state.queue_priority(high))
        self.assertLess(state.queue_priority(incident), state.queue_priority(high))
        self.assertEqual(state.queue_priority(high)[0], 1.)

    def test_icu_reserve_protects_incident_high_risk_and_routing(self):
        state = hospital(policy=OperatingPolicy(icu_reserve_percentage=.3))
        event = active_surge(state)
        routine = [waiting(state, high=True, stay=90.) for _ in range(3)]
        state._dispatch()
        self.assertEqual(routine[2].status, "WAITING_FOR_RESOURCES")
        self.assertEqual(len(state._free["icu_beds"]), 1)
        incident = waiting(state, high=True, event=event.event_id)
        state._dispatch()
        self.assertEqual(incident.status, "IN_TREATMENT")
        self.assertTrue(incident.assigned_bed_id.startswith("ICU-"))
        self.assertTrue(all(r.patient_id is None for r in state.resources["general_beds"].values()))

    def test_aging_wakeup_prevents_reserve_starvation_without_arrivals(self):
        state = hospital(policy=OperatingPolicy(waiting_time_aging_weight=.1, icu_reserve_percentage=.3))
        active_surge(state)
        patients = [waiting(state, high=True, stay=90.) for _ in range(3)]
        state._dispatch()
        self.assertIsNone(patients[2].treatment_start_time)
        state.advance_simulation(60.)
        self.assertEqual(patients[2].treatment_start_time, 60.)

    def staff_reserve(self, gene):
        state = hospital(policy=OperatingPolicy(**{gene: .25}))
        event = active_surge(state)
        low = [waiting(state) for _ in range(4)]
        state._dispatch()
        self.assertEqual(sum(p.status == "IN_TREATMENT" for p in low), 3)
        ownership = {kind: {rid: r.patient_id for rid, r in resources.items() if r.patient_id} for kind, resources in state.resources.items()}
        high = waiting(state, high=True)
        state._dispatch()
        self.assertEqual(high.status, "IN_TREATMENT")
        for kind, resources in ownership.items():
            for rid, pid in resources.items():
                self.assertEqual(state.resources[kind][rid].patient_id, pid)
        self.assertEqual(asdict(state.capacity), asdict(LiveCapacity(3, 10, 4, 4)))

    def test_doctor_reserve_preserves_assignments(self):
        self.staff_reserve("doctor_reserve_percentage")

    def test_nurse_reserve_preserves_assignments(self):
        self.staff_reserve("nurse_reserve_percentage")

    def test_reserves_not_active_in_normal_operation(self):
        state = hospital(policy=OperatingPolicy(doctor_reserve_percentage=.25, nurse_reserve_percentage=.25))
        patients = [waiting(state) for _ in range(4)]
        state._dispatch()
        self.assertTrue(all(p.status == "IN_TREATMENT" for p in patients))

    def test_genetic_operators_use_only_bounded_discrete_policy_genes(self):
        rng = np.random.default_rng(42)
        first, second = random_policy(rng), random_policy(rng)
        for _ in range(100):
            policy = crossover_mutate(first, second, rng)
            self.assertEqual(set(asdict(policy)), set(GENES))
            for gene, (_, maximum, step) in POLICY_BOUNDS.items():
                value = getattr(policy, gene)
                self.assertTrue(0 <= value <= maximum)
                self.assertAlmostEqual(value / step, round(value / step))
            first, second = second, policy

    def search(self, state=None, targets=None):
        state = state or hospital(LiveCapacity(1, 1, 1, 1))
        if not state._waiting:
            waiting(state)
            waiting(state, high=True)
        options = LiveGAOptions(population=4, generations=2, search_replications=1)
        targets = targets or replace(LiveTargets(), mean_wait_target_minutes=100., high_risk_wait_target_minutes=10., max_utilization_target=1., critical_saturation_grace_minutes=30.)
        return state, options, targets, optimize_live_policy(state, None, 60, 3, targets, options)

    def test_search_improves_stress_without_changing_fixed_capacity(self):
        state, options, targets, result = self.search()
        self.assertEqual(result["generation_zero"][0], asdict(state.current_policy))
        self.assertEqual(result["current_verified"]["robustness"], 0.)
        self.assertEqual(result["optimized_verified"]["robustness"], 100.)
        self.assertLess(result["optimized_verified"]["aggregate"]["high_risk_mean_wait"], result["current_verified"]["aggregate"]["high_risk_mean_wait"])
        for observation in result["evaluations"]:
            self.assertEqual(observation["evaluation"]["provenance"]["physical_capacity"], asdict(state.capacity))
        self.assertEqual(len(result["optimized_verified"]["runs"]), 3)
        self.assertTrue(optimization_is_current(result, state, None, 60, 3, targets, options))

    def test_ga_common_random_numbers_reproducibility_and_clone_isolation(self):
        state, options, targets, result = self.search()
        before = state.state_hash()
        repeated = optimize_live_policy(state, None, 60, 3, targets, options)
        result.pop("runtime_seconds"); repeated.pop("runtime_seconds")
        self.assertEqual(result, repeated)
        self.assertEqual(state.state_hash(), before)
        seeds = [o["evaluation"]["provenance"]["replication_seeds"] for o in result["evaluations"]]
        self.assertTrue(all(s == seeds[0] for s in seeds))
        self.assertEqual(result["optimized_verified"]["provenance"]["replication_seeds"][:1], seeds[0])

    def test_no_feasible_policy_is_explicit_and_no_capacity_recommendation(self):
        _, _, _, result = self.search(targets=ScenarioConfig())
        self.assertIn(result["outcome"], ("PARTIAL IMPROVEMENT", "NO MEANINGFUL IMPROVEMENT"))
        self.assertIn("below the configured robustness", result["feasibility_message"])
        self.assertNotIn("doctors", result["optimized_policy"])

    def test_apply_changes_only_policy_preserves_owners_and_is_stale(self):
        state, options, targets, result = self.search()
        state._dispatch()
        owners = {kind: {rid: asdict(r) for rid, r in resources.items()} for kind, resources in state.resources.items()}
        capacity = state.capacity
        state.apply_operating_policy(OperatingPolicy(**result["optimized_policy"]), verification=result["optimized_verified"])
        self.assertIs(state.capacity, capacity)
        self.assertEqual(owners, {kind: {rid: asdict(r) for rid, r in resources.items()} for kind, resources in state.resources.items()})
        self.assertFalse(optimization_is_current(result, state, None, 60, 3, targets, options))
        self.assertEqual(state.events[-1]["event_type"], "POLICY_CHANGED")

    def test_temporary_policy_restores_after_event_recovery_with_history(self):
        state = hospital()
        event = state.propose_event("PATIENT_SURGE", 10, dict(count=1))
        previous = state.current_policy
        state.apply_operating_policy(OperatingPolicy(2., .1), event=event, temporary=True)
        state.advance_simulation(5.)
        self.assertNotEqual(state.current_policy, previous)  # proposed event not yet applied
        event = state.propose_event("PATIENT_SURGE", 10, dict(count=1))
        state.apply_event(event)
        state.advance_simulation(10.)
        self.assertEqual(state.current_policy, previous)
        self.assertTrue(state.policy_history[0]["restored"])
        self.assertTrue(any(e["event_type"] == "POLICY_RESTORED" for e in state.events))

    def test_persistent_policy_never_auto_restores(self):
        state = hospital()
        event = state.propose_event("PATIENT_SURGE", 10, dict(count=1))
        policy = OperatingPolicy(2., .1)
        state.apply_operating_policy(policy, event=event, temporary=False)
        state.apply_event(event)
        state.advance_simulation(60.)
        self.assertEqual(state.current_policy, policy)
        self.assertFalse(state.policy_history[0]["restored"])

    def test_temporary_policy_waits_for_combined_stress_recovery(self):
        state = hospital()
        shortage = state.propose_event("NURSE_SHORTAGE", 30, dict(count=1))
        state.apply_event(shortage)
        surge = state.propose_event("PATIENT_SURGE", 10, dict(count=1))
        previous = state.current_policy
        state.apply_operating_policy(OperatingPolicy(2., .1), event=surge, temporary=True)
        state.apply_event(surge)
        state.advance_simulation(10.)
        self.assertNotEqual(state.current_policy, previous)
        for index in range(205):
            archived = replace(surge, event_id=f"ARCHIVED-{index}", status="ENDED")
            state.stress_events[archived.event_id] = archived
        state._trim_stress_history()
        self.assertIn(surge.event_id, state.stress_events)
        state.advance_simulation(20.)
        self.assertEqual(state.current_policy, previous)

    def test_equivalent_checkpoints_have_stable_future_experiment(self):
        state, options, targets, result = self.search()
        independent = hospital(LiveCapacity(1, 1, 1, 1))
        waiting(independent); waiting(independent, high=True)
        repeated = optimize_live_policy(independent, None, 60, 3, targets, options)
        self.assertEqual(result["experiment_seed"], repeated["experiment_seed"])
        for key in ("optimized_policy", "current_verified", "optimized_verified", "generation_zero"):
            if "verified" in key:
                self.assertEqual(result[key]["aggregate"], repeated[key]["aggregate"])
                self.assertEqual(result[key]["provenance"]["replication_seeds"], repeated[key]["provenance"]["replication_seeds"])
            else:
                self.assertEqual(result[key], repeated[key])

    def test_extended_speeds_bounded_catchup_pause_without_dropped_events(self):
        state = hospital(rate=4.)
        incremental = state.clone()
        clock = LiveClockDriver(state)
        clock.tick(0., 3600, True)
        clock.tick(10., 3600, True)
        self.assertEqual(state.sim_time_minutes, 60.)
        self.assertEqual(clock.pending_minutes, 540.)
        state.pause(); clock.tick(100., 3600, True)
        self.assertEqual(state.sim_time_minutes, 60.)
        state.resume(); clock.reanchor(100.)
        for _ in range(9):
            clock.tick(100., 3600, True)
        incremental.advance_large_skip(600.)
        self.assertEqual(state.metrics(), incremental.metrics())
        self.assertEqual(state._agenda, incremental._agenda)
        self.assertTrue({120, 300, 600, 1200, 3600}.issubset(SIMULATION_SPEEDS))

    def test_seven_day_skip_matches_incremental_and_respects_events(self):
        state = hospital(rate=1.)
        state.configure_random_events(True, "Low", ("PATIENT_SURGE", "DOCTOR_SHORTAGE"))
        other = state.clone()
        progress = []
        state.advance_large_skip(10080., progress=progress.append)
        for _ in range(168):
            other.advance_simulation(60.)
        self.assertEqual(state.snapshot(), other.snapshot())
        self.assertEqual(state._agenda, other._agenda)
        self.assertEqual(len(progress), 168)
        self.assertEqual(progress[-1], 1.)

    def test_thirty_day_skip_ui_is_bounded_pausable_cancellable(self):
        state = hospital()
        skip = LiveSkipDriver(state)
        skip.request(43200.)
        for _ in range(3):
            self.assertEqual(skip.tick(), 60.)
        self.assertEqual(state.sim_time_minutes, 180.)
        skip.paused = True
        self.assertEqual(skip.tick(), 0.)
        skip.paused = False
        skip.tick()
        skip.cancel()
        self.assertEqual(skip.remaining_minutes, 0.)

    def test_full_thirty_day_skip_completes_in_bounded_chunks(self):
        state = hospital(rate=1.)
        skip = LiveSkipDriver(state)
        skip.request(43200.)
        while skip.remaining_minutes:
            self.assertLessEqual(skip.tick(), 60.)
        self.assertEqual(skip.chunks_completed, 720)
        self.assertEqual(state.sim_time_minutes, 43200.)
        self.assertLessEqual(len(state.events), 1000)

    def test_policy_history_bounded_and_reset_restores_initial_policy(self):
        state = hospital()
        for index in range(210):
            state.apply_operating_policy(OperatingPolicy(high_risk_priority_weight=(index % 3)))
        self.assertEqual(len(state.policy_history), 200)
        state.reset()
        self.assertEqual(state.current_policy, OperatingPolicy())
        self.assertFalse(state.policy_history)

    def test_live_tree_uses_real_labels_and_single_class_is_graceful(self):
        _, _, _, result = self.search()
        xai = explain_live_tree(result)
        self.assertEqual(sum(xai["class_distribution"].values()), sum(len(o["evaluation"]["runs"]) for o in result["evaluations"]))
        _, _, _, single = self.search(targets=ScenarioConfig())
        self.assertFalse(explain_live_tree(single)["available"])
        result["evaluations"][0]["evaluation"]["runs"][0]["verdict"] = "invented"
        with self.assertRaises(ValueError):
            explain_live_tree(result)

    def test_no_key_invalid_groq_fallback_cannot_override_status(self):
        _, _, _, result = self.search()
        import os
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}), patch("live_policy_explanation.Groq") as groq:
            explanation = explain_live_policy(result, True)
            groq.assert_not_called()
        self.assertIn(result["optimized_verified"]["verdict"], explanation["summary"])
        with patch.dict(os.environ, {"GROQ_API_KEY": "test"}), patch("live_policy_explanation.Groq") as groq:
            groq.return_value.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content='{"themes":["invented"]}'))]
            invalid = explain_live_policy(result, True)
            self.assertEqual(invalid["provider"], "deterministic fallback")

    def test_valid_groq_selects_only_verified_facts_and_carries_provenance(self):
        _, _, _, result = self.search()
        import os
        with patch.dict(os.environ, {"GROQ_API_KEY": "test"}), patch("live_policy_explanation.Groq") as groq:
            request = groq.return_value.chat.completions.create
            request.return_value.choices = [MagicMock(message=MagicMock(content='{"themes":["high_risk_wait","recovery"]}'))]
            explanation = explain_live_policy(result, True)
            self.assertEqual(request.call_args.kwargs["model"], "openai/gpt-oss-120b")
            self.assertTrue(request.call_args.kwargs["response_format"]["json_schema"]["strict"])
            self.assertIn("unchanged", explanation["explanation"])
            self.assertIn(result["optimized_verified"]["verdict"], explanation["summary"])
            self.assertEqual(explanation["provenance"], result["provenance"])

    def test_live_recommendation_stales_for_targets_context_and_ga_settings(self):
        state, options, targets, result = self.search()
        self.assertFalse(optimization_is_current(result, state, None, 60, 3,
            replace(targets, robustness_threshold=80.), options))
        self.assertFalse(optimization_is_current(result, state, None, 60, 3,
            targets, replace(options, generations=3)))
        event = state.propose_event("PATIENT_SURGE", 10, dict(count=2))
        self.assertFalse(optimization_is_current(result, state, event, 60, 3, targets, options))
        self.assertFalse(optimization_is_current(result, state, None, 60, 5, targets, options))


if __name__ == "__main__":
    unittest.main()

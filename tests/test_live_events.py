"""V3 Part 2: clone isolation, explicit application and event lifecycle."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from live_hospital_state import LiveCapacity, LiveHospitalState
from live_lookahead import simulate_lookahead_from_current_state, preview_is_current, handling_verdict
from scenario_evaluation import ScenarioConfig


def hospital(rate=0., seed=42, capacity=None):
    profiles = pd.DataFrame(dict(predicted_probability=[.1, .9],
        treatment_time_min=[15., 15.], length_of_stay_hours=[.5, .5]))
    state = LiveHospitalState(profiles, capacity or LiveCapacity(2, 2, 2, 2), rate, seed)
    state.start()
    return state


class EventTests(unittest.TestCase):
    def surge(self, state, count=4, duration=10., **params):
        return state.propose_event("PATIENT_SURGE", duration, dict(count=count, **params))

    def test_checkpoint_independent_complete_and_preserves_aliases(self):
        state = hospital(60.)
        state.advance_simulation(10.)
        before = state.state_hash()
        clone = state.checkpoint()
        self.assertIs(clone._profiles, state._profiles)
        for pid in clone._waiting:
            self.assertIs(clone._waiting[pid], clone.active_patients[pid])
        clone.advance_simulation(50.)
        self.assertEqual(before, state.state_hash())
        self.assertNotEqual(clone.sim_time_minutes, state.sim_time_minutes)
        self.assertIsNot(clone._arrival_rng, state._arrival_rng)

    def test_preview_does_not_mutate_live_state_rng_agenda_or_proposal(self):
        state = hospital(20.)
        state.advance_simulation(10.)
        event = self.surge(state)
        before, rng = state.state_hash(), state._arrival_rng.bit_generator.state
        result = simulate_lookahead_from_current_state(state, event, 60, 3)
        self.assertEqual(before, state.state_hash())
        self.assertEqual(rng, state._arrival_rng.bit_generator.state)
        self.assertFalse(event.applied)
        self.assertTrue(preview_is_current(result, state, event))
        self.assertEqual(len(set(result["provenance"]["replication_seeds"])), 3)
        json.dumps(result, allow_nan=False)

    def test_same_checkpoint_event_seed_reproducible_other_seed_varies(self):
        state = hospital(20.)
        event = self.surge(state, 6)
        a = simulate_lookahead_from_current_state(state, event, 60, 3, future_seed=1)
        self.assertEqual(a, simulate_lookahead_from_current_state(state, event, 60, 3, future_seed=1))
        b = simulate_lookahead_from_current_state(state, event, 60, 3, future_seed=2)
        self.assertNotEqual(a["runs"], b["runs"])

    def test_exact_surge_count_risk_mix_and_normal_resources(self):
        state = hospital(capacity=LiveCapacity(10, 10, 20, 20))
        event = self.surge(state, 12, 20, high_risk_proportion=.75)
        state.apply_event(event)
        state.advance_simulation(20.)
        patients = list(state.active_patients.values())
        self.assertEqual(len(patients), 12)
        self.assertEqual(sum(p.high_risk for p in patients), 9)
        for p in patients:
            self.assertEqual(p.required_bed_type, "icu_beds" if p.risk_probability >= .5 else "general_beds")
            held = [r for resources in state.resources.values() for r in resources.values() if r.patient_id == p.patient_id]
            self.assertEqual(sum(r.kind.endswith("beds") for r in held), 1)
            if p.status == "IN_TREATMENT":
                self.assertEqual(sum(r.kind == "doctors" for r in held), 1)
                self.assertEqual(sum(r.kind == "nurses" for r in held), 1)
        self.assertEqual(state.event_metrics(event.event_id)["surge_patients_introduced"], 12)

    def test_event_metrics_exclude_background(self):
        state = hospital(60.)
        event = self.surge(state, 3, 0.)
        state.apply_event(event)
        state.advance_simulation(60.)
        self.assertGreater(state._arrived, 3)
        stats = state.event_metrics(event.event_id)
        self.assertEqual(stats["surge_patients_introduced"], 3)
        self.assertEqual(stats["surge_patients_completed"] + stats["event_unfinished_patients"], 3)

    def test_spike_restores_base_and_highest_overlap_wins(self):
        state = hospital(22.)
        first = state.propose_event("ARRIVAL_RATE_SPIKE", 30, dict(arrival_rate=40.))
        state.apply_event(first)
        second = state.propose_event("ARRIVAL_RATE_SPIKE", 10, dict(arrival_rate=60.))
        state.apply_event(second)
        self.assertEqual(state.arrival_rate, 22.)
        self.assertEqual(state.effective_arrival_rate, 60.)
        state.advance_simulation(10.)
        self.assertEqual(state.effective_arrival_rate, 40.)
        state.advance_simulation(20.)
        self.assertEqual(state.effective_arrival_rate, 22.)
        self.assertEqual(sum(e[3] == "ARRIVAL" for e in state._agenda), 1)

    def outage_busy(self, event_type, kind):
        state = hospital(capacity=LiveCapacity(1, 1, 1, 1))
        surge = self.surge(state, 1, 0., high_risk_proportion=1.)
        state.apply_event(surge)
        patient = next(iter(state.active_patients.values()))
        resource = next(iter(state.resources[kind].values()))
        event = state.propose_event(event_type, 40., dict(count=1))
        state.apply_event(event)
        self.assertEqual(resource.status, "OUT_OF_SERVICE_PENDING")
        self.assertEqual(resource.patient_id, patient.patient_id)
        release_at = 30. if kind.endswith("beds") else 15.
        state.advance_simulation(release_at)
        self.assertEqual(resource.status, "OUT_OF_SERVICE")
        self.assertIsNone(resource.patient_id)
        self.assertEqual(state.resource_summary()[kind]["occupied_or_busy"], 0)
        state.advance_simulation(40. - release_at)
        self.assertEqual(resource.status, "AVAILABLE")
        self.assertEqual(state.resource_summary()[kind]["usable"], 1)

    def test_doctor_shortage_does_not_interrupt_and_restores(self):
        self.outage_busy("DOCTOR_SHORTAGE", "doctors")

    def test_nurse_shortage_does_not_interrupt_and_restores(self):
        self.outage_busy("NURSE_SHORTAGE", "nurses")

    def test_icu_outage_never_evicts_and_restores(self):
        self.outage_busy("ICU_BED_OUTAGE", "icu_beds")

    def test_general_outage_never_evicts(self):
        state = hospital(capacity=LiveCapacity(1, 1, 1, 1))
        state.apply_event(self.surge(state, 1, 0., high_risk_proportion=0.))
        state.apply_event(state.propose_event("GENERAL_BED_OUTAGE", 20., dict(count=1)))
        r = state.resources["general_beds"]["GEN-001"]
        self.assertEqual(r.status, "OUT_OF_SERVICE_PENDING")
        state.advance_simulation(20.)
        self.assertEqual(r.status, "OCCUPIED")
        self.assertIsNotNone(r.patient_id)

    def test_free_first_overlaps_and_queue_recovers(self):
        state = hospital()
        state.apply_event(self.surge(state, 1, 0., high_risk_proportion=1.))
        state.apply_event(state.propose_event("DOCTOR_SHORTAGE", 10., dict(count=1)))
        self.assertEqual(state.resources["doctors"]["DOC-001"].status, "BUSY")
        self.assertEqual(state.resources["doctors"]["DOC-002"].status, "OUT_OF_SERVICE")
        state.apply_event(state.propose_event("NURSE_SHORTAGE", 20., dict(count=1)))
        state.apply_event(self.surge(state, 1, 0., high_risk_proportion=0.))
        self.assertEqual(len(state._waiting), 1)
        state.advance_simulation(20.)
        self.assertEqual(len(state._waiting), 0)
        for kind, resources in state.resources.items():
            free = state._free[kind]
            self.assertEqual(len(free), len(set(free)))
            self.assertEqual(set(free), {r.resource_id for r in resources.values() if r.status == "AVAILABLE"})

    def test_impossible_and_overlapping_outages_rejected(self):
        state = hospital()
        for kind, duration, params in (("BAD", 1, {}), ("NURSE_SHORTAGE", 10, dict(count=-1)),
            ("PATIENT_SURGE", -1, dict(count=2)), ("DOCTOR_SHORTAGE", 1, dict(count=3)),
            ("ARRIVAL_RATE_SPIKE", 0, dict(arrival_rate=30.)), ("PATIENT_SURGE", 1, dict(count=2, bad=True))):
            with self.assertRaises(ValueError):
                state.propose_event(kind, duration, params)
        event = state.propose_event("DOCTOR_SHORTAGE", 30, dict(count=2), start_delay=10)
        state.apply_event(event)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            state.apply_event(state.propose_event("DOCTOR_SHORTAGE", 30, dict(count=1)))

    def test_back_to_back_outages_restore_before_next_start(self):
        state = hospital()
        state.apply_event(state.propose_event("DOCTOR_SHORTAGE", 10, dict(count=2)))
        state.apply_event(state.propose_event("DOCTOR_SHORTAGE", 10, dict(count=2), start_delay=10))
        state.advance_simulation(10.)
        self.assertEqual(state.resource_summary()["doctors"]["temporarily_unavailable"], 2)
        self.assertEqual(len(state.stress_events["EVT-000002"].affected_resources), 2)
        state.advance_simulation(10.)
        self.assertEqual(len(state._free["doctors"]), 2)

    def test_zero_usable_capacity_is_conservatively_saturated(self):
        state = hospital()
        event = state.propose_event("DOCTOR_SHORTAGE", 30, dict(count=2))
        result = simulate_lookahead_from_current_state(state, event, 60, 1)
        self.assertEqual(result["aggregate"]["doctor_utilization"], 1.)
        self.assertFalse(result["runs"][0]["condition_breakdown"]["doctor_util_pass"])

    def test_clone_retains_pending_surge_outage_spike_and_random_schedule(self):
        state = hospital(20.)
        state.apply_event(self.surge(state, 10, 50.))
        state.apply_event(state.propose_event("NURSE_SHORTAGE", 40, dict(count=1)))
        state.apply_event(state.propose_event("ARRIVAL_RATE_SPIKE", 30, dict(arrival_rate=40.)))
        state.configure_random_events(True, "High", ("PATIENT_SURGE",))
        clone = state.clone()
        state.advance_simulation(60.)
        clone.advance_simulation(60.)
        self.assertEqual(state.snapshot(), clone.snapshot())
        self.assertEqual(state._agenda, clone._agenda)

    def test_preview_stale_on_clock_targets_settings_and_application(self):
        state = hospital()
        event = self.surge(state, 1, 0.)
        result = simulate_lookahead_from_current_state(state, event, 60, 1)
        self.assertFalse(preview_is_current(result, state, targets=replace(ScenarioConfig(), robustness_threshold=80.)))
        self.assertFalse(preview_is_current(result, state, horizon=120))
        self.assertEqual(state._arrived, 0)
        state.apply_event(event)
        self.assertEqual(state._arrived, 1)
        self.assertFalse(preview_is_current(result, state))
        with self.assertRaises(ValueError):
            state.apply_event(event)
        state.advance_simulation(1.)
        self.assertFalse(preview_is_current(result, state))

    def test_small_recovery_and_not_recovered(self):
        state = hospital()
        result = simulate_lookahead_from_current_state(state, self.surge(state, 1, 10.), 60, 1)
        self.assertEqual(result["aggregate"]["recovery_time_minutes"], 10.)
        blocked = hospital(capacity=LiveCapacity(1, 1, 1, 1))
        result = simulate_lookahead_from_current_state(blocked, self.surge(blocked, 20, 0., high_risk_proportion=1.), 10, 1)
        self.assertIsNone(result["aggregate"]["recovery_time_minutes"])

    def test_handling_verdict_exact_rules(self):
        targets = ScenarioConfig()
        self.assertEqual(handling_verdict(90, 0, 0, targets), "HANDLED")
        self.assertEqual(handling_verdict(80, 10, 5, targets), "AT RISK")
        self.assertEqual(handling_verdict(0, 0, 0, targets), "CANNOT HANDLE")
        self.assertEqual(handling_verdict(40, 41, 0, targets), "CANNOT HANDLE")

    def test_random_events_off_default_simulated_time_reproducible(self):
        a, b = hospital(), hospital()
        self.assertFalse(any(row[3] == "RANDOM_STRESS" for row in a._agenda))
        for state in (a, b):
            state.configure_random_events(True, "High", ("PATIENT_SURGE",))
            agenda = list(state._agenda)
            state.configure_random_events(True, "High", ("PATIENT_SURGE",))
            self.assertEqual(agenda, state._agenda)
            for _ in range(10):
                state.advance_simulation(60.)
            self.assertTrue(state.stress_events)
            self.assertEqual(sum(row[3] == "RANDOM_STRESS" for row in state._agenda), 1)
        self.assertEqual(a.snapshot(), b.snapshot())
        a.configure_random_events(False)
        self.assertFalse(any(row[3] == "RANDOM_STRESS" for row in a._agenda))

    def test_event_log_has_assignment_release_and_is_bounded(self):
        state = hospital()
        state._event_limit = 20
        from collections import deque
        state.events = deque(maxlen=20)
        event = self.surge(state, 2, 0.)
        state.apply_event(event)
        self.assertTrue(any(row["event_id"] == event.event_id and row["event_type"] == "RESOURCES_ASSIGNED" for row in state.events))
        state.advance_simulation(30.)
        self.assertTrue(any(row["event_type"] == "RESOURCE_RELEASED" for row in state.events))
        self.assertLessEqual(len(state.events), 20)

    def test_random_mixed_events_preserve_resource_conservation_and_end(self):
        state = hospital(12., capacity=LiveCapacity(10, 10, 10, 10))
        state.configure_random_events(True, "High")
        for _ in range(48):
            state.advance_simulation(60.)
            for kind, resources in state.resources.items():
                self.assertEqual(len(state._free[kind]), len(set(state._free[kind])))
                self.assertEqual(set(state._free[kind]), {r.resource_id for r in resources.values() if r.status == "AVAILABLE"})
                self.assertTrue(all(r.patient_id is None or r.patient_id in state.active_patients for r in resources.values()))
        state.configure_random_events(False)
        for _ in range(4):
            state.advance_simulation(60.)
        self.assertFalse(state.event_rows())
        self.assertEqual(state.effective_arrival_rate, state.arrival_rate)
        self.assertTrue(all(r.outage_event_id is None for resources in state.resources.values() for r in resources.values()))


if __name__ == "__main__":
    unittest.main()

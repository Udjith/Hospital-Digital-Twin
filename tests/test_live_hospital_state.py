"""Deterministic V3 lifecycle, clock, ownership, history and metric checks."""
from dataclasses import asdict, FrozenInstanceError
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from live_hospital_state import KINDS, LiveCapacity, LiveClockDriver, LiveHospitalState, OperatingPolicy


def profiles(probabilities=(.9,)):
    return pd.DataFrame({"predicted_probability": list(probabilities),
        "treatment_time_min": [15.] * len(probabilities), "length_of_stay_hours": [.5] * len(probabilities),
        "real_patient_id": ["DO-NOT-EXPOSE"] * len(probabilities)})


class LiveStateTests(unittest.TestCase):
    def hospital(self, **kwargs):
        hospital = LiveHospitalState(profiles(), capacity=LiveCapacity(1, 1, 1, 1), arrival_rate=60., **kwargs)
        hospital.start()
        return hospital

    def first_patient(self):
        hospital = self.hospital()
        hospital.advance_simulation(3.)
        return hospital, hospital.active_patients["P000001"]

    def test_resource_ids_unique_stable_and_capacity_fixed(self):
        hospital = self.hospital()
        ids = [r for resources in hospital.resources.values() for r in resources]
        self.assertEqual(ids, ["ICU-001", "GEN-001", "DOC-001", "NUR-001"])
        self.assertEqual(len(ids), len(set(ids)))
        hospital.advance_simulation(60.)
        self.assertEqual(ids, [r for resources in hospital.resources.values() for r in resources])
        with self.assertRaises(FrozenInstanceError):
            hospital.capacity.nurses = 5
        with self.assertRaises(AttributeError):
            hospital.capacity = LiveCapacity(2, 2, 2, 2)

    def test_exactly_one_correct_bed_doctor_and_nurse_per_patient(self):
        hospital = LiveHospitalState(profiles((.49, .5, .9)), LiveCapacity(2, 2, 4, 4), 60.)
        hospital.start()
        hospital.advance_simulation(15.)
        seen = set()
        for patient in hospital.active_patients.values():
            expected_kind = "icu_beds" if patient.risk_probability >= .50 else "general_beds"
            self.assertEqual(patient.required_bed_type, expected_kind)
            seen.add(expected_kind)
            held = {kind: [r for r in resources.values() if r.patient_id == patient.patient_id]
                    for kind, resources in hospital.resources.items()}
            if patient.status == "IN_TREATMENT":
                self.assertEqual(len(held["icu_beds"]) + len(held["general_beds"]), 1)
                self.assertEqual(len(held[expected_kind]), 1)
                self.assertEqual(len(held["doctors"]), 1)
                self.assertEqual(len(held["nurses"]), 1)
            else:
                self.assertTrue(all(not records for records in held.values()))
        self.assertEqual(seen, {"icu_beds", "general_beds"})

    def test_staff_release_after_treatment_bed_held_until_discharge(self):
        hospital, patient = self.first_patient()
        self.assertEqual(patient.status, "IN_TREATMENT")
        self.assertEqual(patient.assigned_bed_id, "ICU-001")
        self.assertEqual(patient.assigned_doctor_id, "DOC-001")
        self.assertEqual(patient.assigned_nurse_id, "NUR-001")
        hospital.advance_simulation(patient.expected_treatment_completion - hospital.sim_time_minutes + 1e-8)
        self.assertEqual(patient.status, "BED_STAY")
        self.assertIsNone(patient.assigned_doctor_id)
        self.assertIsNone(patient.assigned_nurse_id)
        self.assertEqual(hospital.resources["doctors"]["DOC-001"].status, "AVAILABLE")
        self.assertIsNone(hospital.resources["nurses"]["NUR-001"].patient_id)
        self.assertEqual(hospital.resources["icu_beds"]["ICU-001"].patient_id, patient.patient_id)

    def test_discharge_releases_bed_and_queued_patient_gets_all_freed_resources(self):
        hospital, first = self.first_patient()
        hospital.advance_simulation(6.)
        queued = hospital.active_patients["P000002"]
        self.assertEqual(queued.status, "WAITING_FOR_RESOURCES")
        self.assertIsNone(queued.assigned_bed_id)
        self.assertIsNone(queued.assigned_doctor_id)
        self.assertIsNone(queued.assigned_nurse_id)
        hospital.advance_simulation(first.expected_discharge_time - hospital.sim_time_minutes + 1e-8)
        self.assertEqual(first.status, "DISCHARGED")
        self.assertIsNone(first.assigned_bed_id)
        self.assertNotIn(first.patient_id, hospital.active_patients)
        self.assertEqual(queued.status, "IN_TREATMENT")
        self.assertEqual(queued.assigned_bed_id, "ICU-001")
        self.assertEqual(queued.assigned_doctor_id, "DOC-001")
        self.assertEqual(queued.assigned_nurse_id, "NUR-001")
        self.assertAlmostEqual(queued.treatment_start_time, first.discharge_time)

    def test_bed_really_returns_to_available_when_no_waiter(self):
        # A low rate has no second arrival before the first discharge for this seed.
        hospital = LiveHospitalState(profiles(), LiveCapacity(1, 1, 1, 1), arrival_rate=1.)
        hospital.start()
        for _ in range(3):
            hospital.advance_simulation(60.)
        self.assertEqual(hospital.metrics()["completed_patients"], 1)
        resource = hospital.resources["icu_beds"]["ICU-001"]
        self.assertEqual(resource.status, "AVAILABLE")
        self.assertIsNone(resource.patient_id)

    def test_waiting_when_either_staff_resource_unavailable_and_no_partial_hold(self):
        hospital = LiveHospitalState(profiles((.1, .9)), LiveCapacity(20, 20, 1, 1), 60.)
        hospital.start()
        hospital.advance_simulation(10.)
        self.assertGreater(hospital.metrics()["currently_waiting"], 0)
        self.assertEqual(hospital.metrics()["in_treatment"], 1)
        self.assertEqual(sum(r["occupied_or_busy"] for k, r in hospital.resource_summary().items() if k.endswith("beds")), 1)

    def test_seed_reproducibility_different_seed_and_exact_exponential_feed(self):
        a, b, c = self.hospital(seed=42), self.hospital(seed=42), self.hospital(seed=43)
        for hospital in (a, b, c):
            hospital.advance_simulation(15.)
        arrivals = lambda h: [e["sim_time_minutes"] for e in h.events if e["event_type"] == "PATIENT_ARRIVED"]
        self.assertEqual(a.snapshot(), b.snapshot())
        self.assertNotEqual(arrivals(a), arrivals(c))
        rng, generated, time = np.random.default_rng(42), [], 0.
        while True:
            time += rng.exponential(60. / 60.)
            if time > 15.:
                break
            generated.append(time)
        self.assertEqual(arrivals(a), generated)

    def test_incremental_partition_matches_single_step(self):
        a, b = self.hospital(), self.hospital()
        a.advance_simulation(60.)
        for _ in range(12):
            b.advance_simulation(5.)
        self.assertEqual(list(a.events), list(b.events))
        self.assertEqual(a.patient_rows(), b.patient_rows())
        self.assertAlmostEqual(a.metrics()["mean_wait_so_far"], b.metrics()["mean_wait_so_far"])
        for kind in KINDS:
            self.assertAlmostEqual(a.metrics()["cumulative_utilization"][kind], b.metrics()["cumulative_utilization"][kind])

    def test_pause_resume_stop_and_reset(self):
        hospital = self.hospital()
        hospital.advance_simulation(20.)
        hospital.pause()
        self.assertEqual(hospital.status, "PAUSED")
        hospital.resume()
        self.assertEqual(hospital.status, "RUNNING")
        hospital.stop()
        with self.assertRaises(ValueError):
            hospital.advance_simulation(1.)
        hospital.reset()
        self.assertEqual(hospital.status, "READY")
        self.assertEqual(hospital.sim_time_minutes, 0)
        self.assertFalse(hospital.events)
        self.assertFalse(hospital.active_patients)
        self.assertFalse(hospital.completed_patients)
        self.assertEqual(hospital.metrics()["currently_waiting"], 0)
        self.assertTrue(all(r.status == "AVAILABLE" and r.patient_id is None for resources in hospital.resources.values() for r in resources.values()))
        hospital.start()
        hospital.advance_simulation(3.)
        self.assertAlmostEqual(hospital.active_patients["P000001"].arrival_time, np.random.default_rng(42).exponential(1.))

    def test_event_log_assignments_releases_and_bounds(self):
        hospital = self.hospital(event_limit=40, completed_limit=2)
        hospital.advance_simulation(40.)
        events = list(hospital.events)
        # High arrival volume evicts old events; inspect full lifecycle separately.
        self.assertLessEqual(len(events), 40)
        full = self.hospital()
        full.advance_simulation(40.)
        first = [event for event in full.events if event["patient_id"] == "P000001"]
        assigned = next(e for e in first if e["event_type"] == "RESOURCES_ASSIGNED")
        self.assertEqual(assigned["resource_ids"], ["ICU-001", "DOC-001", "NUR-001"])
        releases = [e for e in first if e["event_type"] == "RESOURCE_RELEASED"]
        self.assertEqual([e["resource_ids"][0] for e in releases], ["DOC-001", "NUR-001", "ICU-001"])
        for _ in range(3):
            hospital.advance_simulation(60.)
        self.assertEqual(len(hospital.completed_patients), 2)
        self.assertEqual(len(hospital.events), 40)

    def test_snapshot_serializable_compact_and_no_real_ids(self):
        hospital = self.hospital()
        hospital.advance_simulation(60.)
        snapshot = hospital.snapshot(patient_limit=2, event_limit=3)
        serialized = json.dumps(snapshot, allow_nan=False)
        self.assertNotIn("DO-NOT-EXPOSE", serialized)
        self.assertNotIn("real_patient_id", serialized)
        self.assertTrue(snapshot["patient_rows_truncated"])
        self.assertEqual(len(snapshot["active_patients"]), 2)
        self.assertEqual(len(snapshot["recent_events"]), 3)
        self.assertEqual(snapshot["configuration"]["rf_threshold"], .50)

    def test_cumulative_wait_counts_only_elapsed_wait_and_utilization_is_integrated(self):
        hospital, first = self.first_patient()
        hospital.advance_simulation(7.)
        expected = sum(p.observed_wait(hospital.sim_time_minutes) for p in hospital.active_patients.values()) / hospital.metrics()["total_patients_arrived"]
        self.assertAlmostEqual(hospital.metrics()["mean_wait_so_far"], expected)
        expected_util = (10. - first.arrival_time) / 10.
        self.assertAlmostEqual(hospital.metrics()["cumulative_utilization"]["doctors"], expected_util)
        self.assertAlmostEqual(hospital.resource_summary()["doctors"]["current_utilization"], 1.)

    def test_rolling_values_and_histories_prune_after_sixty_minutes(self):
        hospital = self.hospital()
        for _ in range(3):
            hospital.advance_simulation(60.)
        rolling = hospital.metrics()["rolling_60_minutes"]
        self.assertEqual(rolling["observed_minutes"], 60.)
        self.assertTrue(all(p.arrival_time > 120 for p in hospital._recent_arrivals))
        self.assertTrue(all(segment[1] > 120 for segment in hospital._segments))
        self.assertLessEqual(rolling["arrivals"], hospital.metrics()["total_patients_arrived"])
        self.assertTrue(all(0 <= value <= 1 for value in rolling["utilization"].values()))
        self.assertGreater(rolling["mean_queue_length"], 0.)

    def test_operating_priority_hooks_and_neutral_placeholders(self):
        hospital, first = self.first_patient()
        self.assertEqual(asdict(hospital.current_policy), dict(high_risk_priority_weight=0., waiting_time_aging_weight=0.,
            icu_reserve_percentage=0., doctor_reserve_percentage=0., nurse_reserve_percentage=0., surge_priority_strength=0.))
        self.assertEqual(hospital.queue_priority(first)[0], first.base_priority)
        with self.assertRaisesRegex(ValueError, "fractions"):
            LiveHospitalState(profiles(), operating_policy=OperatingPolicy(icu_reserve_percentage=10))

    def test_wall_clock_adapter_counts_elapsed_once_speed_pause_and_catchup(self):
        hospital = self.hospital()
        clock = LiveClockDriver(hospital)
        clock.tick(100., speed=60, enabled=True)
        clock.tick(101., speed=60, enabled=True)
        self.assertAlmostEqual(hospital.sim_time_minutes, 1.)
        clock.tick(101., speed=60, enabled=True)
        self.assertAlmostEqual(hospital.sim_time_minutes, 1.)
        hospital.pause()
        clock.tick(500., speed=60, enabled=True)
        hospital.resume()
        clock.reanchor(500.)
        clock.tick(501., speed=60, enabled=True)
        self.assertAlmostEqual(hospital.sim_time_minutes, 2.)
        clock.tick(701., speed=60, enabled=True)
        self.assertAlmostEqual(hospital.sim_time_minutes, 62.)
        self.assertAlmostEqual(clock.pending_minutes, 140.)
        clock.tick(701., speed=60, enabled=True)
        self.assertAlmostEqual(hospital.sim_time_minutes, 122.)

    def test_zero_arrival_rate_and_bounded_steps(self):
        hospital = LiveHospitalState(profiles(), arrival_rate=0.)
        hospital.start()
        hospital.advance_simulation(60.)
        self.assertEqual(hospital.metrics()["total_patients_arrived"], 0)
        self.assertEqual(hospital.metrics()["mean_wait_so_far"], 0)
        for invalid in (-1, 61, float("nan")):
            with self.assertRaises(ValueError):
                hospital.advance_simulation(invalid)

    def test_profiles_not_mutated_or_retained_as_source_dataframe(self):
        data = profiles()
        original = data.copy(deep=True)
        hospital = LiveHospitalState(data)
        hospital.start()
        hospital.advance_simulation(60.)
        pd.testing.assert_frame_equal(data, original)
        self.assertFalse(any(isinstance(value, pd.DataFrame) for value in hospital.__dict__.values()))


if __name__ == "__main__":
    unittest.main()

"""Operational scheduling controls; never modifies physical resource capacity."""
from live_event_context import event_members
from collections import deque
from dataclasses import asdict
import math

POLICY_BOUNDS = dict(high_risk_priority_weight=(0., 5., .1),
    waiting_time_aging_weight=(0., 2., .1), icu_reserve_percentage=(0., .30, .01),
    doctor_reserve_percentage=(0., .25, .01), nurse_reserve_percentage=(0., .25, .01),
    surge_priority_strength=(0., 5., .1))
GENES = tuple(POLICY_BOUNDS)
NORMAL_POLICY_VALUES = dict(high_risk_priority_weight=1., waiting_time_aging_weight=.1,
    icu_reserve_percentage=0., doctor_reserve_percentage=0., nurse_reserve_percentage=0.,
    surge_priority_strength=0.)
DEMO_CAPACITY = dict(icu_beds=220, general_beds=550, doctors=80, nurses=140)
DEMO_ARRIVAL_RATE = 4.
SIMULATION_SPEEDS = (1, 5, 10, 30, 60, 120, 300, 600, 1200, 3600)
MAX_MANUAL_MINUTES = 43200.
RESERVE_AGING_GRACE_MINUTES = 60.


class LivePolicyControl:
    def _reset_policy_control(self):
        self._policy = self._initial_policy
        self._normal_policy = self._initial_policy
        self._temporary_policy = None
        self._policy_dirty = False
        self._aging_due = None
        self.policy_history = deque(maxlen=200)

    @property
    def normal_policy(self):
        return self._normal_policy

    def event_patient_flag(self, patient):
        # Incident backlog remains in operational recovery after arrival spread ends.
        return bool(patient.source_event_id and patient.source_event_id in self.stress_events and
            (self.stress_events[patient.source_event_id].status == "ACTIVE" or patient.patient_id in self._waiting))

    def stress_active(self):
        return any(e.status == "ACTIVE" for e in self.stress_events.values()) or any(
            self.event_patient_flag(p) for p in self._waiting.values())

    def reserve_counts(self):
        if not self.stress_active():
            return dict(icu_beds=0, doctors=0, nurses=0)
        summary = self.resource_summary()
        return {kind: math.ceil(summary[kind]["usable"] * getattr(self.current_policy, field) - 1e-12)
            for kind, field in (("icu_beds", "icu_reserve_percentage"),
                ("doctors", "doctor_reserve_percentage"), ("nurses", "nurse_reserve_percentage"))}

    def allocation_allowed(self, patient, reserves):
        aged = (self.current_policy.waiting_time_aging_weight > 0 and
                self.sim_time_minutes - patient.arrival_time >= RESERVE_AGING_GRACE_MINUTES)
        incident = self.event_patient_flag(patient)
        staff_priority = patient.high_risk or (incident and self.current_policy.surge_priority_strength > 0) or aged
        if not staff_priority and any(len(self._free[k]) <= reserves[k] for k in ("doctors", "nurses")):
            return False
        # ICU remains exclusively RF-high-risk. During an incident, reserve protects
        # incoming incident ICU demand; aged routine ICU patients may borrow it.
        if reserves["icu_beds"] and patient.required_bed_type == "icu_beds" and not (incident or aged):
            surge_context = any(e.event_type == "PATIENT_SURGE" and e.status == "ACTIVE"
                                for e in self.stress_events.values()) or any(
                self.event_patient_flag(p) and p.high_risk for p in self._waiting.values())
            if surge_context:
                return len(self._free["icu_beds"]) > reserves["icu_beds"]
        return True

    def _schedule_reserve_aging(self, patient):
        deadline = patient.arrival_time + RESERVE_AGING_GRACE_MINUTES
        if self.current_policy.waiting_time_aging_weight > 0 and deadline > self.sim_time_minutes and (
                self._aging_due is None or deadline < self._aging_due):
            self._aging_due = deadline
            self._schedule(deadline, 5, "POLICY_AGING")

    def event_recovered(self, event_id, original_queue):
        event = self.stress_events.get(event_id)
        return bool(event and event.status == "ENDED" and len(self._waiting) <= original_queue + 2
            and not any(p.source_event_id == event_id for p in self._waiting.values()))

    def apply_operating_policy(self, policy, reason="Human-reviewed operating policy", event=None,
                               temporary=False, fitness=None, verification=None, normal_policy=False):
        if self.status not in ("RUNNING", "PAUSED"):
            raise ValueError("Start the live twin before changing operating policy.")
        policy.validate()
        if temporary and event is None:
            raise ValueError("Temporary policy requires a triggering event.")
        previous = self.current_policy
        restore_target = self._temporary_policy["previous"] if self._temporary_policy else previous
        if self._temporary_policy:
            self._temporary_policy["record"]["superseded"] = True
        record = dict(sim_time_minutes=self.sim_time_minutes, previous_policy=asdict(previous),
            new_policy=asdict(policy), reason=reason, triggering_event_id=event.event_id if event else None,
            ga_fitness=fitness, verified_verdict=verification.get("verdict") if verification else None,
            verified_robustness=verification.get("robustness") if verification else None,
            applied=True, restored=False, temporary=bool(temporary))
        self.policy_history.append(record)
        self._policy = policy
        self._policy_dirty = True
        self._temporary_policy = dict(previous=restore_target, event_id=event.event_id,
            original_queue=len(self._waiting), record=record,
            recovery_event_ids=tuple(sorted({member.event_id for member in event_members(event)} | {e.event_id for e in self.stress_events.values()
                if e.status == "ACTIVE"}))) if temporary else None
        if normal_policy and not temporary:
            self._normal_policy = policy
        self._event("POLICY_CHANGED", reason + "; physical capacities unchanged.", event_id=record["triggering_event_id"])

    def restore_normal_policy(self, automatic=False):
        temporary = self._temporary_policy
        previous = self.current_policy
        restored = temporary["previous"] if temporary else self._normal_policy
        self._policy = restored
        self._temporary_policy = None
        self._policy_dirty = True
        if temporary:
            temporary["record"]["restored"] = True
            temporary["record"]["restored_sim_time_minutes"] = self.sim_time_minutes
        self.policy_history.append(dict(sim_time_minutes=self.sim_time_minutes,
            previous_policy=asdict(previous), new_policy=asdict(restored), reason="Event recovery" if automatic else "Manual restore",
            triggering_event_id=temporary["event_id"] if temporary else None,
            applied=True, restored=True, temporary=False))
        self._event("POLICY_RESTORED", "Previous operating policy restored; physical capacities unchanged.",
            event_id=temporary["event_id"] if temporary else None)

    def _check_policy_recovery(self):
        pending = self._temporary_policy
        if pending and all(self.event_recovered(event_id, pending["original_queue"])
                for event_id in pending["recovery_event_ids"]) and not any(
                e.status == "ACTIVE" for e in self.stress_events.values()):
            self.restore_normal_policy(automatic=True)
            self._policy_dirty = False
            self._dispatch()

    def advance_large_skip(self, minutes, progress=None):
        """CLI/test helper; UI uses persistent skip debt one chunk per refresh."""
        if not math.isfinite(minutes) or not 0 <= minutes <= MAX_MANUAL_MINUTES:
            raise ValueError("Manual skip must be between zero and 43200 simulated minutes.")
        remaining = float(minutes)
        while remaining:
            step = min(self.MAX_STEP_MINUTES, remaining)
            self.advance_simulation(step)
            remaining -= step
            if progress:
                progress((minutes - remaining) / minutes)

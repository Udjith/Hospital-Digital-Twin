"""V3 simulated live feed. Independent state engine; V2 remains unchanged."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math

import numpy as np

from live_events import LiveEventSupport
from live_policy_control import LivePolicyControl, POLICY_BOUNDS, SIMULATION_SPEEDS, MAX_MANUAL_MINUTES
from live_initialization import WarmStartSettings, initialize_hospital

from scenario_evaluation import HospitalPolicy, prepare_profiles, validate_policy

VERSION = "3.3-adaptive-policy"
KINDS = ("icu_beds", "general_beds", "doctors", "nurses")
PREFIXES = dict(icu_beds="ICU", general_beds="GEN", doctors="DOC", nurses="NUR")


@dataclass(frozen=True)
class LiveCapacity:
    icu_beds: int = 210
    general_beds: int = 280
    doctors: int = 85
    nurses: int = 100

    def validate(self):
        validate_policy(HospitalPolicy(**asdict(self)))


@dataclass(frozen=True)
class OperatingPolicy:
    # Zero extra weights preserves V2 clinical priority, then arrival order.
    high_risk_priority_weight: float = 0.0
    waiting_time_aging_weight: float = 0.0
    icu_reserve_percentage: float = 0.0
    doctor_reserve_percentage: float = 0.0
    nurse_reserve_percentage: float = 0.0
    surge_priority_strength: float = 0.0

    def validate(self):
        values = asdict(self)
        if any(not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError("Operating-policy values must be finite and nonnegative.")
        for name, (_, maximum, _) in POLICY_BOUNDS.items():
            if values[name] > maximum:
                raise ValueError(f"{name} must be between zero and {maximum}; reserve values are fractions.")



@dataclass
class Resource:
    resource_id: str
    kind: str
    status: str = "AVAILABLE"
    patient_id: str | None = None
    outage_event_id: str | None = None


@dataclass
class Patient:
    patient_id: str
    arrival_time: float
    risk_probability: float
    high_risk: bool
    required_bed_type: str
    treatment_minutes: float
    stay_minutes: float
    base_priority: int
    status: str = "WAITING_FOR_RESOURCES"
    assigned_bed_id: str | None = None
    assigned_doctor_id: str | None = None
    assigned_nurse_id: str | None = None
    queue_entry_time: float | None = None
    treatment_start_time: float | None = None
    expected_treatment_completion: float | None = None
    expected_discharge_time: float | None = None
    discharge_time: float | None = None
    source_event_id: str | None = None
    initialized_patient: bool = False
    pre_start_age_minutes: float = 0.0
    initial_elapsed_treatment: float = 0.0
    initial_elapsed_los: float = 0.0

    def observed_wait(self, now):
        return (now if self.treatment_start_time is None else self.treatment_start_time) - self.arrival_time


class LiveHospitalState(LivePolicyControl, LiveEventSupport):
    """Heap-based discrete events, explicitly advanced in at most 60-minute steps.

    Completed rows and recent events are capped. Only active patients persist in
    the main mapping; cumulative counters do not require retaining old patients.
    """
    MAX_STEP_MINUTES = 60.0

    def __init__(self, patient_profiles, capacity=None, arrival_rate=22., seed=42,
                 operating_policy=None, event_limit=1000, completed_limit=200, warm_start=None):
        self._capacity = capacity or LiveCapacity()
        self._capacity.validate()
        if not math.isfinite(arrival_rate) or not 0 <= arrival_rate <= 1000:
            raise ValueError("Live arrival rate must be between 0 and 1000 patients/hour.")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Live seed must be a nonnegative 32-bit integer.")
        if type(event_limit) is not int or not 1 <= event_limit <= 2000:
            raise ValueError("Event history limit must be an integer from 1 to 2000.")
        if type(completed_limit) is not int or not 1 <= completed_limit <= 1000:
            raise ValueError("Completed history limit must be an integer from 1 to 1000.")
        self._policy = operating_policy or OperatingPolicy()
        self._policy.validate()
        self._initial_policy = self._policy
        # Reuse V2 normalization, routing threshold, treatment/LOS and priority.
        prepared = prepare_profiles(patient_profiles)
        self._profiles = tuple((float(probability), *profile) for probability, profile in
            zip(patient_profiles["predicted_probability"], prepared))
        if any(not math.isfinite(profile[0]) or not 0 <= profile[0] <= 1 for profile in self._profiles):
            raise ValueError("Live profiles require finite RF probabilities between 0 and 1.")
        self.profile_sha256 = hashlib.sha256(json.dumps(self._profiles).encode()).hexdigest()
        self._arrival_rate, self._seed = float(arrival_rate), seed
        self._event_limit, self._completed_limit = event_limit, completed_limit
        # Legacy API callers remain empty; normal dashboard explicitly enables defaults.
        self.warm_start_settings = warm_start if warm_start is not None else WarmStartSettings(enabled=False)
        self.warm_start_settings.validate()
        self.reset()

    @property
    def capacity(self):
        return self._capacity

    @property
    def current_policy(self):
        return self._policy

    @property
    def arrival_rate(self):
        return self._arrival_rate

    @property
    def seed(self):
        return self._seed

    def reset(self, warm_start=None):
        if warm_start is not None:
            from dataclasses import replace
            self.warm_start_settings = replace(self.warm_start_settings, enabled=warm_start)
        self.warm_start_settings.validate()
        self.sim_time_minutes = 0.0
        self.status = "READY"
        self.resources = {kind: {f"{PREFIXES[kind]}-{index:03d}": Resource(f"{PREFIXES[kind]}-{index:03d}", kind)
            for index in range(1, getattr(self.capacity, kind) + 1)} for kind in KINDS}
        self._free = {kind: sorted(resources) for kind, resources in self.resources.items()}
        for ids in self._free.values():
            heapq.heapify(ids)
        self.active_patients = {}
        self.completed_patients = deque(maxlen=self._completed_limit)
        self.events = deque(maxlen=self._event_limit)
        self._waiting = {}
        self._agenda = []
        self._sequence = self._patient_sequence = 0
        self._arrived = self._completed = self._high_arrived = 0
        self._served_wait = self._high_served_wait = 0.0
        self._initial_served_wait = self._initial_high_served_wait = 0.0
        self._initial_completed = 0
        self._busy_minutes = dict.fromkeys(KINDS, 0.0)
        self._segments = deque()
        self._recent_arrivals, self._recent_completions = deque(), deque()
        # Separate profile stream: stepping/allocation never changes demand draws.
        self._arrival_rng = np.random.default_rng(self.seed)
        self._profile_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1]))
        self._reset_stress_events()
        self._reset_policy_control()
        initialize_hospital(self)

    def _event(self, event_type, description, patient_id=None, resource_ids=None, event_id=None):
        self.events.append(dict(sim_time_minutes=self.sim_time_minutes, event_type=event_type,
            patient_id=patient_id, resource_ids=resource_ids or [], description=description, event_id=event_id))

    def _schedule(self, time, priority, event_type, patient_id=None):
        self._sequence += 1
        heapq.heappush(self._agenda, (time, priority, self._sequence, event_type, patient_id))

    def _next_arrival(self):
        if self.effective_arrival_rate:
            interval = float(self._arrival_rng.exponential(60. / self.effective_arrival_rate))
            self._schedule(self.sim_time_minutes + interval, 2, "ARRIVAL")

    def start(self):
        if self.status != "READY":
            raise ValueError("Reset Live Twin before starting a new session.")
        self.status = "RUNNING"
        self._event("SESSION_STARTED", "Simulated live feed started; physical capacities are fixed.")
        self._next_arrival()

    def pause(self):
        if self.status == "RUNNING":
            self.status = "PAUSED"
            self._event("SESSION_PAUSED", "Live clock paused.")

    def resume(self):
        if self.status == "PAUSED":
            self.status = "RUNNING"
            self._event("SESSION_RESUMED", "Live clock resumed.")

    def stop(self):
        if self.status in ("RUNNING", "PAUSED"):
            self.status = "STOPPED"
            self._event("SESSION_STOPPED", "Live clock stopped; state retained until reset.")

    def queue_priority(self, patient):
        waiting = self.sim_time_minutes - patient.arrival_time
        return (patient.base_priority - self.current_policy.high_risk_priority_weight * int(patient.high_risk)
                - self.current_policy.surge_priority_strength * int(self.event_patient_flag(patient))
                - self.current_policy.waiting_time_aging_weight * waiting, patient.arrival_time, patient.patient_id)

    def _acquire(self, kind, patient):
        resource_id = heapq.heappop(self._free[kind])
        resource = self.resources[kind][resource_id]
        resource.patient_id = patient.patient_id
        resource.status = "OCCUPIED" if kind.endswith("beds") else "BUSY"
        return resource_id

    def _release(self, kind, resource_id, patient):
        resource = self.resources[kind][resource_id]
        if resource.patient_id != patient.patient_id:
            raise RuntimeError("Resource ownership mismatch.")
        resource.patient_id = None
        resource.status = "OUT_OF_SERVICE" if resource.outage_event_id else "AVAILABLE"
        if not resource.outage_event_id:
            heapq.heappush(self._free[kind], resource_id)
        self._event("RESOURCE_RELEASED", f"{resource_id} released from {patient.patient_id}.", patient.patient_id, [resource_id], event_id=resource.outage_event_id or patient.source_event_id)

    def _dispatch(self):
        reserves = self.reserve_counts()
        for patient in sorted(self._waiting.values(), key=self.queue_priority):
            bed_kind = patient.required_bed_type
            # Atomic acquisition: a waiting patient never holds partial resources.
            if not all(self._free[k] for k in (bed_kind, "doctors", "nurses")):
                continue
            if not self.allocation_allowed(patient, reserves):
                self._schedule_reserve_aging(patient)
                continue
            del self._waiting[patient.patient_id]
            patient.assigned_bed_id = self._acquire(bed_kind, patient)
            patient.assigned_doctor_id = self._acquire("doctors", patient)
            patient.assigned_nurse_id = self._acquire("nurses", patient)
            patient.status = "IN_TREATMENT"
            patient.treatment_start_time = self.sim_time_minutes
            patient.expected_treatment_completion = self.sim_time_minutes + patient.treatment_minutes
            patient.expected_discharge_time = self.sim_time_minutes + patient.stay_minutes
            wait = patient.observed_wait(self.sim_time_minutes)
            if patient.initialized_patient:
                self._initial_served_wait += wait
                self._initial_high_served_wait += wait if patient.high_risk else 0.
            else:
                self._served_wait += wait
                if patient.high_risk:
                    self._high_served_wait += wait
            if patient.source_event_id:
                stats = self.stress_events[patient.source_event_id].statistics
                stats["served_wait"] = stats.get("served_wait", 0.) + wait
                stats["high_served_wait"] = stats.get("high_served_wait", 0.) + (wait if patient.high_risk else 0.)
            assigned = [patient.assigned_bed_id, patient.assigned_doctor_id, patient.assigned_nurse_id]
            self._event("RESOURCES_ASSIGNED", f"{patient.patient_id} assigned " + ", ".join(assigned) + ".", patient.patient_id, assigned, event_id=patient.source_event_id)
            self._schedule(patient.expected_treatment_completion, 0, "TREATMENT_COMPLETE", patient.patient_id)
            self._schedule(patient.expected_discharge_time, 1, "DISCHARGE", patient.patient_id)

    def _arrive(self, source_event_id=None, risk_override=None):
        pool = self._profiles if risk_override is None else tuple(p for p in self._profiles if bool(p[1]) == risk_override)
        probability, high_risk, treatment, stay, priority = pool[int(self._profile_rng.integers(len(pool)))]
        self._patient_sequence += 1
        patient = Patient(f"P{self._patient_sequence:06d}", self.sim_time_minutes, probability, high_risk,
            "icu_beds" if high_risk else "general_beds", treatment, stay, priority, queue_entry_time=self.sim_time_minutes, source_event_id=source_event_id)
        self.active_patients[patient.patient_id] = patient
        self._waiting[patient.patient_id] = patient
        self._arrived += 1
        self._high_arrived += int(high_risk)
        self._recent_arrivals.append(patient)
        if source_event_id:
            stats = self.stress_events[source_event_id].statistics
            stats["arrived"] += 1
            stats["high_arrived"] = stats.get("high_arrived", 0) + int(high_risk)
        self._event("PATIENT_ARRIVED", f"{patient.patient_id} arrived.", patient.patient_id, event_id=source_event_id)
        self._event("RISK_CLASSIFIED", f"{patient.patient_id}: {'high' if high_risk else 'lower'} RF risk; requires {'ICU' if high_risk else 'General'} bed.", patient.patient_id, event_id=source_event_id)
        self._dispatch()
        if patient.status == "WAITING_FOR_RESOURCES":
            self._event("PATIENT_QUEUED", f"{patient.patient_id} waiting for bed, doctor and nurse availability.", patient.patient_id, event_id=source_event_id)
        if source_event_id is None:
            self._next_arrival()

    def _integrate(self, end):
        start = self.sim_time_minutes
        if end > start:
            busy = {k: sum(r.patient_id is not None for r in self.resources[k].values()) for k in KINDS}
            self._segments.append((start, end, busy, len(self._waiting)))
            for kind in KINDS:
                self._busy_minutes[kind] += busy[kind] * (end - start)
            self.sim_time_minutes = end

    def _prune_rolling(self):
        cutoff = self.sim_time_minutes - 60
        while self._segments and self._segments[0][1] <= cutoff:
            self._segments.popleft()
        while self._recent_arrivals and self._recent_arrivals[0].arrival_time <= cutoff:
            self._recent_arrivals.popleft()
        while self._recent_completions and self._recent_completions[0] <= cutoff:
            self._recent_completions.popleft()

    def advance_simulation(self, delta_minutes):
        if self.status not in ("RUNNING", "PAUSED"):
            raise ValueError("Start or resume the live twin before advancing; stopped sessions require reset.")
        if not math.isfinite(delta_minutes) or not 0 <= delta_minutes <= self.MAX_STEP_MINUTES:
            raise ValueError("Each live advance must be between 0 and 60 simulated minutes.")
        self._advance_to(self.sim_time_minutes + delta_minutes)

    def _advance_to(self, target):
        """Shared event-to-event scheduler; callers validate their observation window."""
        if self._policy_dirty:
            self._policy_dirty = False
            self._dispatch()
        processed_events = 0
        while self._agenda and self._agenda[0][0] <= target:
            time, _, _, event_type, patient_id = heapq.heappop(self._agenda)
            processed_events += 1
            self._integrate(time)
            if event_type == "ARRIVAL":
                self._arrive()
            elif event_type == "SURGE_ARRIVAL":
                self._arrive(*patient_id)
            elif event_type == "STRESS_START":
                self._start_stress_event(self.stress_events[patient_id])
            elif event_type == "STRESS_END":
                self._end_stress_event(self.stress_events[patient_id])
            elif event_type == "RANDOM_STRESS":
                self._random_event()
            elif event_type == "POLICY_AGING":
                if self._aging_due == time:
                    self._aging_due = None
                    self._dispatch()
            else:
                patient = self.active_patients[patient_id]
                if event_type == "TREATMENT_COMPLETE":
                    self._release("doctors", patient.assigned_doctor_id, patient)
                    self._release("nurses", patient.assigned_nurse_id, patient)
                    patient.assigned_doctor_id = patient.assigned_nurse_id = None
                    patient.status = "BED_STAY"
                    self._event("TREATMENT_COMPLETED", f"{patient_id} completed treatment; bed retained until discharge.", patient_id, [patient.assigned_bed_id])
                else:
                    self._release(patient.required_bed_type, patient.assigned_bed_id, patient)
                    patient.assigned_bed_id = None
                    patient.status, patient.discharge_time = "DISCHARGED", self.sim_time_minutes
                    self._completed += 1
                    self._initial_completed += int(patient.initialized_patient)
                    if patient.source_event_id:
                        self.stress_events[patient.source_event_id].statistics["completed"] += 1
                    self._recent_completions.append(self.sim_time_minutes)
                    self.completed_patients.append(patient)
                    del self.active_patients[patient_id]
                    self._event("PATIENT_DISCHARGED", f"{patient_id} discharged.", patient_id, event_id=patient.source_event_id)
                self._dispatch()
            self._check_policy_recovery()
            self._prune_rolling()
        self._integrate(target)
        self._check_policy_recovery()
        self._prune_rolling()
        return processed_events

    def fast_forward(self, minutes, progress=None, should_continue=None):
        """FAST_FORWARD: same events and arithmetic, without playback/UI throttling.

        Retain ordinary 60-minute observation boundaries for exact cumulative
        floating-point/rolling-history equivalence. Within each boundary the
        scheduler jumps directly to heap timestamps, never minute-by-minute.
        No persistent execution-mode field enters provenance/state hashing.
        """
        if self.status not in ("RUNNING", "PAUSED"):
            raise ValueError("Start the live twin before Fast Forward.")
        if not math.isfinite(minutes) or not 0 <= minutes <= MAX_MANUAL_MINUTES:
            raise ValueError("Fast Forward must be between zero and 43200 minutes (30 days).")
        remaining, processed, boundaries = float(minutes), 0, 0
        while remaining and (should_continue is None or should_continue()):
            delta = min(remaining, self.MAX_STEP_MINUTES)
            processed += self._advance_to(self.sim_time_minutes + delta)
            remaining -= delta
            boundaries += 1
            if progress:
                progress(minutes - remaining, minutes, processed)
        return dict(advanced_minutes=minutes - remaining, processed_events=processed,
                    integration_boundaries=boundaries)

    def resource_summary(self):
        summary = {}
        for kind, resources in self.resources.items():
            total = getattr(self.capacity, kind)
            busy = sum(r.patient_id is not None for r in resources.values())
            unavailable = sum(r.status == "OUT_OF_SERVICE" for r in resources.values())
            pending = sum(r.status == "OUT_OF_SERVICE_PENDING" for r in resources.values())
            usable = total - unavailable
            summary[kind] = dict(total=total, occupied_or_busy=busy,
                available=len(self._free[kind]), temporarily_unavailable=unavailable,
                pending_unavailable=pending, usable=usable, planned_usable=usable - pending,
                current_utilization=busy / usable if usable else 1.)
        return summary

    def metrics(self):
        now = self.sim_time_minutes
        waiting = list(self._waiting.values())
        live_waiting = [p for p in waiting if not p.initialized_patient]
        wait_sum = self._served_wait + sum(p.observed_wait(now) for p in live_waiting)
        high_sum = self._high_served_wait + sum(p.observed_wait(now) for p in live_waiting if p.high_risk)
        duration = min(60., now)
        cutoff = now - duration
        rolling_busy = dict.fromkeys(KINDS, 0.)
        queue_area = 0.
        for start, end, busy, queue in self._segments:
            overlap = max(0., end - max(start, cutoff))
            queue_area += overlap * queue
            for kind in KINDS:
                rolling_busy[kind] += overlap * busy[kind]
        recent = list(self._recent_arrivals)
        return dict(total_patients_arrived=self._arrived, completed_patients=self._completed,
            initial_patients_completed=self._initial_completed,
            live_arrivals_completed=self._completed - self._initial_completed,
            initial_active_patients=self.initial_state["active_patients"],
            initial_patients_remaining=sum(p.initialized_patient for p in self.active_patients.values()),
            currently_waiting=len(waiting), currently_active=len(self.active_patients),
            in_treatment=sum(p.status == "IN_TREATMENT" for p in self.active_patients.values()),
            bed_stay=sum(p.status == "BED_STAY" for p in self.active_patients.values()),
            mean_wait_so_far=wait_sum / self._arrived if self._arrived else 0.,
            high_risk_mean_wait_so_far=high_sum / self._high_arrived if self._high_arrived else 0.,
            throughput_per_hour=self._completed * 60 / now if now else 0.,
            cumulative_utilization={k: min(1., max(0., self._busy_minutes[k] / (getattr(self.capacity, k) * now))) if now else 0. for k in KINDS},
            rolling_60_minutes={"observed_minutes": duration, "arrivals": len(recent),
                "completions": len(self._recent_completions), "queue_length_now": len(waiting),
                "mean_queue_length": queue_area / duration if duration else 0.,
                "mean_wait": sum(p.observed_wait(now) for p in recent) / len(recent) if recent else 0.,
                "utilization": {k: min(1., max(0., rolling_busy[k] / (getattr(self.capacity, k) * duration))) if duration else 0. for k in KINDS}})

    def patient_rows(self, limit=50, include_completed=False):
        if not 1 <= limit <= 200:
            raise ValueError("Patient display limit must be between 1 and 200.")
        patients = list(self.active_patients.values())[:limit]
        if include_completed:
            patients += list(reversed(self.completed_patients))[:max(0, limit - len(patients))]
        return [{**asdict(p), "risk_classification": "HIGH" if p.high_risk else "LOWER",
                 "waiting_time_minutes": p.observed_wait(self.sim_time_minutes)} for p in patients]

    def resource_rows(self, kind, status="All", limit=100):
        if kind not in KINDS or status not in ("All", "AVAILABLE", "OCCUPIED", "BUSY", "OUT_OF_SERVICE_PENDING", "OUT_OF_SERVICE") or not 1 <= limit <= 200:
            raise ValueError("Invalid resource table filter or row limit.")
        rows = []
        for resource in self.resources[kind].values():
            if status == "All" or resource.status == status:
                rows.append(asdict(resource))
                if len(rows) == limit:
                    break
        return rows

    def snapshot(self, patient_limit=50, event_limit=100):
        if not 0 <= event_limit <= 2000:
            raise ValueError("Snapshot event limit must be between 0 and 2000.")
        return {"schema_version": 3, "engine_version": VERSION, "status": self.status,
            "sim_time_minutes": self.sim_time_minutes, "elapsed_hours": self.sim_time_minutes / 60,
            "configuration": {"capacity": asdict(self.capacity), "arrival_rate": self.arrival_rate, "seed": self.seed,
                "rf_threshold": .50, "profile_sha256": self.profile_sha256},
            "resources": self.resource_summary(), "queues": {"waiting": len(self._waiting)},
            "active_patients": self.patient_rows(patient_limit),
            "active_patient_count": len(self.active_patients), "patient_rows_truncated": len(self.active_patients) > patient_limit,
            "metrics": self.metrics(), "current_policy": asdict(self.current_policy),
            "initial_state": self.initial_state, "live_state_hash": self.state_hash(),
            "normal_policy": asdict(self.normal_policy), "policy_history": list(self.policy_history)[-20:],
            "recent_events": list(self.events)[-event_limit:] if event_limit else [],
            "event_history_limit": self._event_limit,
            "effective_arrival_rate": self.effective_arrival_rate, "stress_events": self.event_rows(),
            "limitations": "Simulated live feed, not EHR integration. Operational priority is not clinical triage. Human-reviewed adaptive operating policies; physical capacities fixed."}


class LiveClockDriver:
    """Wall-clock adapter only; event correctness lives in advance_simulation.

    Accumulate elapsed monotonic time once, retaining bounded-step catch-up debt.
    Paused/stopped wall time is never simulated. Manual advances are independent.
    """
    def __init__(self, hospital):
        self.hospital = hospital
        self.last_wall_seconds = None
        self.speed = 1
        self.enabled = False
        self.pending_minutes = 0.

    def reanchor(self, now):
        self.last_wall_seconds = now

    def tick(self, now, speed=1, enabled=False):
        if speed not in SIMULATION_SPEEDS or not math.isfinite(now):
            raise ValueError("Invalid live clock settings.")
        if self.last_wall_seconds is not None:
            elapsed = max(0., now - self.last_wall_seconds)
            if self.enabled and self.hospital.status == "RUNNING":
                self.pending_minutes += elapsed * self.speed / 60.
        if self.hospital.status == "RUNNING" and (enabled or self.enabled) and self.pending_minutes:
            delta = min(self.pending_minutes, self.hospital.MAX_STEP_MINUTES)
            self.hospital.advance_simulation(delta)
            self.pending_minutes -= delta
        self.last_wall_seconds, self.speed, self.enabled = now, speed, enabled

"""Synthetic operational stress events and additive live-engine support."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import heapq
import math
from live_canonical import canonical_hash, live_hash_input

import numpy as np

EVENT_TYPES = ("PATIENT_SURGE", "ARRIVAL_RATE_SPIKE", "DOCTOR_SHORTAGE",
               "NURSE_SHORTAGE", "ICU_BED_OUTAGE", "GENERAL_BED_OUTAGE")
OUTAGE_KINDS = dict(DOCTOR_SHORTAGE="doctors", NURSE_SHORTAGE="nurses",
                   ICU_BED_OUTAGE="icu_beds", GENERAL_BED_OUTAGE="general_beds")


@dataclass
class LiveEvent:
    event_id: str
    event_type: str
    created_sim_time: float
    start_sim_time: float
    duration_minutes: float
    parameters: dict
    description: str = ""
    source: str = "manual simulated event"
    status: str = "PROPOSED"
    applied: bool = False
    lookahead_run: bool = False
    affected_resources: list = field(default_factory=list)
    statistics: dict = field(default_factory=lambda: dict(arrived=0, completed=0))

    @property
    def end_sim_time(self):
        return self.start_sim_time + self.duration_minutes

    def specification(self):
        return {k: getattr(self, k) for k in ("event_id", "event_type", "created_sim_time",
            "start_sim_time", "duration_minutes", "parameters", "description", "source")}

    def validate(self, hospital):
        if self.event_type not in EVENT_TYPES or not self.event_id:
            raise ValueError("Unsupported event type or missing event ID.")
        if any(not math.isfinite(t) or t < 0 for t in
               (self.created_sim_time, self.start_sim_time, self.duration_minutes)):
            raise ValueError("Event times must be finite and nonnegative.")
        if self.start_sim_time < hospital.sim_time_minutes or self.duration_minutes > 1440:
            raise ValueError("Event start cannot be in the past; duration cannot exceed 1440 minutes.")
        allowed = {"count", "high_risk_proportion"} if self.event_type == "PATIENT_SURGE" else (
            {"arrival_rate"} if self.event_type == "ARRIVAL_RATE_SPIKE" else {"count"})
        if set(self.parameters) - allowed:
            raise ValueError("Unsupported event parameter.")
        if self.event_type == "ARRIVAL_RATE_SPIKE":
            rate = self.parameters.get("arrival_rate")
            if not isinstance(rate, (float, int)) or not math.isfinite(rate) or not hospital.arrival_rate <= rate <= 1000:
                raise ValueError("Spike rate must be between the base rate and 1000 patients/hour.")
        else:
            count = self.parameters.get("count")
            maximum = 1000 if self.event_type == "PATIENT_SURGE" else getattr(hospital.capacity, OUTAGE_KINDS[self.event_type])
            if type(count) is not int or not 1 <= count <= maximum:
                raise ValueError(f"Event count must be an integer from 1 to {maximum}.")
        if self.event_type != "PATIENT_SURGE" and self.duration_minutes <= 0:
            raise ValueError("Temporary events require a positive duration.")
        proportion = self.parameters.get("high_risk_proportion")
        if proportion is not None:
            if not isinstance(proportion, (float, int)) or not math.isfinite(proportion) or not 0 <= proportion <= 1:
                raise ValueError("High-risk proportion must be between 0 and 1.")
            high_count = math.floor(self.parameters["count"] * proportion + .5)
            for risk, needed in ((True, high_count), (False, self.parameters["count"] - high_count)):
                if needed and not any(bool(p[1]) == risk for p in hospital._profiles):
                    raise ValueError("Requested risk mix is unavailable in the cached RF patient profiles.")


class LiveEventSupport:
    """No event machinery changes the normal feed until explicitly enabled/applied."""
    def _reset_stress_events(self):
        self.stress_events = {}
        self._stress_sequence = 0
        self._stress_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 2]))
        self._random_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 3]))
        self._random_settings = (False, "Low", ())

    @property
    def effective_arrival_rate(self):
        return max([self.arrival_rate] + [e.parameters["arrival_rate"] for e in self.stress_events.values()
            if e.event_type == "ARRIVAL_RATE_SPIKE" and e.status == "ACTIVE"])

    def propose_event(self, event_type, duration_minutes, parameters, description="", start_delay=0., source="manual simulated event"):
        event = LiveEvent(f"EVT-{self._stress_sequence + 1:06d}", event_type,
            self.sim_time_minutes, self.sim_time_minutes + start_delay, float(duration_minutes),
            copy.deepcopy(parameters), description or event_type.replace("_", " ").title(), source)
        event.validate(self)
        return event

    def clone(self):
        # Share only immutable clinical tuples. deepcopy preserves internal patient
        # aliases (queue/active/rolling), heaps, generators and resource ownership.
        return copy.deepcopy(self, {id(self._profiles): self._profiles})

    def checkpoint(self):
        return self.clone()

    def state_hash(self):
        return canonical_hash(live_hash_input(self))

    def reseed_future(self, seed):
        self._arrival_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        self._profile_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self._stress_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        self._random_rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
        # Existing pending arrivals, event effects and patients remain unchanged.

    def apply_event(self, proposed):
        if self.status not in ("RUNNING", "PAUSED"):
            raise ValueError("Start the live twin before applying events.")
        proposed.validate(self)
        if proposed.event_id in self.stress_events:
            raise ValueError("This event has already been applied.")
        pending = [e for e in self.stress_events.values() if e.status in ("SCHEDULED", "ACTIVE")]
        if len(pending) >= 100:
            raise ValueError("At most 100 simultaneous/scheduled stress events are supported.")
        self._validate_outage_overlap(proposed, pending)
        event = copy.deepcopy(proposed)
        event.applied, event.status = True, "SCHEDULED"
        self._stress_sequence += 1
        self.stress_events[event.event_id] = event
        self._event("EVENT_CREATED", event.description, event_id=event.event_id)
        self._schedule(event.start_sim_time, -2, "STRESS_START", event.event_id)
        self._schedule(event.end_sim_time, -3 if event.duration_minutes else 3, "STRESS_END", event.event_id)
        # Apply immediate effects without advancing clock or altering playback.
        self.advance_simulation(0.)
        self._trim_stress_history()
        return event

    def _validate_outage_overlap(self, proposed, pending):
        # Disjoint reservations: reject overlapping outages whose combined demand
        # exceeds physical capacity, including events scheduled for the future.
        kind = OUTAGE_KINDS.get(proposed.event_type)
        if kind:
            same = [e for e in pending if OUTAGE_KINDS.get(e.event_type) == kind]
            times = [proposed.start_sim_time] + [e.start_sim_time for e in same
                if proposed.start_sim_time <= e.start_sim_time < proposed.end_sim_time]
            if any(proposed.parameters["count"] + sum(e.parameters["count"] for e in same
                if e.start_sim_time <= t < e.end_sim_time) > getattr(self.capacity, kind) for t in times):
                raise ValueError("Overlapping outages exceed the physical resource total.")

    def _trim_stress_history(self):
        ended = [k for k, e in self.stress_events.items() if e.status == "ENDED"]
        pending_policy = getattr(self, "_temporary_policy", None)
        protected = pending_policy["recovery_event_ids"] if pending_policy else ()
        for key in ended[:-200]:
            if key not in protected and not any(p.source_event_id == key for p in self.active_patients.values()):
                del self.stress_events[key]

    def _refresh_arrival_feed(self, previous):
        if previous != self.effective_arrival_rate:
            self._agenda = [row for row in self._agenda if row[3] != "ARRIVAL"]
            heapq.heapify(self._agenda)
            self._next_arrival()

    def _start_stress_event(self, event):
        previous = self.effective_arrival_rate
        event.status = "ACTIVE"
        if event.event_type == "PATIENT_SURGE":
            count = event.parameters["count"]
            offsets = sorted(self._stress_rng.uniform(0, event.duration_minutes, count)) if event.duration_minutes else [0.] * count
            proportion = event.parameters.get("high_risk_proportion")
            risks = [None] * count
            if proportion is not None:
                high = math.floor(count * proportion + .5)
                risks = [True] * high + [False] * (count - high)
                self._stress_rng.shuffle(risks)
            for offset, risk in zip(offsets, risks):
                self._schedule(self.sim_time_minutes + float(offset), 2, "SURGE_ARRIVAL", (event.event_id, risk))
        elif event.event_type in OUTAGE_KINDS:
            kind = OUTAGE_KINDS[event.event_type]
            candidates = sorted((r for r in self.resources[kind].values() if r.outage_event_id is None),
                key=lambda r: (r.patient_id is not None, r.resource_id))
            selected = candidates[:event.parameters["count"]]
            for resource in selected:
                resource.outage_event_id = event.event_id
                resource.status = "OUT_OF_SERVICE_PENDING" if resource.patient_id else "OUT_OF_SERVICE"
                event.affected_resources.append(resource.resource_id)
            selected_ids = set(event.affected_resources)
            self._free[kind] = [r for r in self._free[kind] if r not in selected_ids]
            heapq.heapify(self._free[kind])
        self._refresh_arrival_feed(previous)
        self._event("EVENT_APPLIED", event.description, resource_ids=event.affected_resources, event_id=event.event_id)

    def _end_stress_event(self, event):
        previous = self.effective_arrival_rate
        event.status = "ENDED"
        kind = OUTAGE_KINDS.get(event.event_type)
        if kind:
            for rid in event.affected_resources:
                resource = self.resources[kind][rid]
                resource.outage_event_id = None
                if resource.patient_id:
                    resource.status = "OCCUPIED" if kind.endswith("beds") else "BUSY"
                else:
                    resource.status = "AVAILABLE"
                    heapq.heappush(self._free[kind], rid)
        self._refresh_arrival_feed(previous)
        self._event("EVENT_ENDED", event.description + "; temporary effects restored.",
                    resource_ids=event.affected_resources, event_id=event.event_id)
        self._dispatch()
        self._trim_stress_history()

    def event_metrics(self, event_id):
        event = self.stress_events[event_id]
        patients = [p for p in self.active_patients.values() if p.source_event_id == event_id]
        waiting = [p for p in patients if p.treatment_start_time is None]
        stats = event.statistics
        wait = stats.get("served_wait", 0.) + sum(p.observed_wait(self.sim_time_minutes) for p in waiting)
        high_wait = stats.get("high_served_wait", 0.) + sum(p.observed_wait(self.sim_time_minutes) for p in waiting if p.high_risk)
        return dict(surge_patients_introduced=stats["arrived"], surge_patients_completed=stats["completed"],
            event_mean_wait=wait / stats["arrived"] if stats["arrived"] else 0.,
            event_high_risk_mean_wait=high_wait / stats.get("high_arrived", 1) if stats.get("high_arrived") else 0.,
            event_unfinished_patients=len(patients), event_waiting_patients=len(waiting))

    def event_rows(self, active_only=True):
        return [{**asdict(e), "end_sim_time": e.end_sim_time,
            "remaining_minutes": max(0., e.end_sim_time - self.sim_time_minutes)}
            for e in self.stress_events.values() if not active_only or e.status in ("SCHEDULED", "ACTIVE")]

    def configure_random_events(self, enabled=False, frequency="Low", allowed_types=EVENT_TYPES):
        allowed_types = tuple(sorted(set(allowed_types)))
        if frequency not in ("Low", "Medium", "High") or set(allowed_types) - set(EVENT_TYPES) or (enabled and not allowed_types):
            raise ValueError("Select a valid random-event frequency and at least one supported event type.")
        settings = (bool(enabled), frequency, allowed_types)
        if settings == self._random_settings:
            return
        self._random_settings = settings
        self._agenda = [row for row in self._agenda if row[3] != "RANDOM_STRESS"]
        heapq.heapify(self._agenda)
        if enabled:
            self._schedule_random_event()

    def _schedule_random_event(self):
        mean = {"Low": 360., "Medium": 180., "High": 60.}[self._random_settings[1]]
        self._schedule(self.sim_time_minutes + float(self._random_rng.exponential(mean)), 4, "RANDOM_STRESS")

    def _random_event(self):
        kind = str(self._random_rng.choice(self._random_settings[2]))
        duration = float(self._random_rng.uniform(30, 180))
        if kind == "PATIENT_SURGE":
            params = dict(count=int(self._random_rng.integers(5, 31)))
        elif kind == "ARRIVAL_RATE_SPIKE":
            params = dict(arrival_rate=min(1000., self.arrival_rate * float(self._random_rng.uniform(1.25, 2.))))
        else:
            capacity = getattr(self.capacity, OUTAGE_KINDS[kind])
            fraction = float(self._random_rng.uniform(.05, .15 if "OUTAGE" in kind else .20))
            params = dict(count=max(1, math.floor(capacity * fraction)))
        try:
            proposed = self.propose_event(kind, duration, params, source="synthetic random stress generator")
            self._validate_outage_overlap(proposed, [e for e in self.stress_events.values()
                if e.status in ("SCHEDULED", "ACTIVE")])
            # Schedule here; avoid recursively advancing inside the event loop.
            proposed.applied, proposed.status = True, "SCHEDULED"
            if kind in OUTAGE_KINDS and any(e.status == "ACTIVE" and e.event_type == kind for e in self.stress_events.values()):
                raise ValueError("Resource outage already active; random duplicate skipped.")
            if len(self.event_rows()) >= 100:
                raise ValueError("Active event limit reached.")
            self._stress_sequence += 1
            self.stress_events[proposed.event_id] = proposed
            self._event("EVENT_CREATED", proposed.description, event_id=proposed.event_id)
            self._schedule(proposed.start_sim_time, -2, "STRESS_START", proposed.event_id)
            self._schedule(proposed.end_sim_time, -3 if proposed.duration_minutes else 3, "STRESS_END", proposed.event_id)
        except ValueError as exc:
            self._event("RANDOM_EVENT_SKIPPED", str(exc))
        self._schedule_random_event()

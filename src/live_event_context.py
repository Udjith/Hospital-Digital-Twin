"""Compound proposals composed exclusively of existing LiveEvent entities."""
from dataclasses import dataclass
import copy


def event_members(context):
    return tuple(context.events) if hasattr(context, "events") else (() if context is None else (context,))


@dataclass(frozen=True)
class ScenarioEvents:
    events: tuple

    @property
    def event_id(self):
        return self.events[0].event_id

    @property
    def start_sim_time(self):
        return min(e.start_sim_time for e in self.events)

    @property
    def description(self):
        return " + ".join(e.description for e in self.events)

    def specification(self):
        return {**self.events[0].specification(), "related_events": [e.specification() for e in self.events[1:]]}

    def validate(self, hospital):
        if not 1 <= len(self.events) <= 6 or len({e.event_id for e in self.events}) != len(self.events):
            raise ValueError("A scenario requires 1–6 distinct event entities.")
        trial = hospital.clone()
        for event in ordered_events(self):
            trial.apply_event(copy.deepcopy(event))


def ordered_events(context):
    # Same-time availability changes precede arrivals; existing scheduler/order
    # within each individual event is unchanged.
    return sorted(event_members(context), key=lambda e: (e.start_sim_time,
        1 if e.event_type == "PATIENT_SURGE" else 0, e.event_id))


def apply_event_context(hospital, context):
    if not hasattr(context, "events"):
        return hospital.apply_event(context)
    context.validate(hospital)  # Complete preflight; invalid combinations apply nothing.
    checkpoint = hospital.checkpoint()
    try:
        return [hospital.apply_event(copy.deepcopy(event)) for event in ordered_events(context)]
    except Exception:
        hospital.__dict__.clear()
        hospital.__dict__.update(checkpoint.__dict__)
        raise


def context_event_metrics(hospital, context):
    members = event_members(context)
    if not members:
        return {}
    if len(members) == 1:
        return hospital.event_metrics(members[0].event_id)
    rows = [hospital.event_metrics(e.event_id) for e in members]
    counts = [row["surge_patients_introduced"] for row in rows]
    high_counts = [hospital.stress_events[e.event_id].statistics.get("high_arrived", 0) for e in members]
    result = {key: sum(row[key] for row in rows) for key in rows[0] if "mean_wait" not in key}
    result["event_mean_wait"] = sum(row["event_mean_wait"] * count for row, count in zip(rows, counts)) / sum(counts) if sum(counts) else 0.
    result["event_high_risk_mean_wait"] = sum(row["event_high_risk_mean_wait"] * count for row, count in zip(rows, high_counts)) / sum(high_counts) if sum(high_counts) else 0.
    return result

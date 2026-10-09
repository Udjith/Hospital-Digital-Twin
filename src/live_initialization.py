"""Seeded initial occupancy only; normal arrival/event streams are untouched."""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class WarmStartSettings:
    enabled: bool = True
    icu_occupancy_target: float = .55
    general_occupancy_target: float = .25
    doctor_busy_target: float = .20
    nurse_busy_target: float = .20
    initial_queue: int | None = None  # Auto: zero with free resources, at most 3 otherwise.

    def validate(self):
        if type(self.enabled) is not bool:
            raise ValueError("Warm-start enabled must be a boolean.")
        for key, value in asdict(self).items():
            if key.endswith("target") and (type(value) not in (int, float) or
                    not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{key} must be a fraction between zero and one.")
        if self.initial_queue is not None and (type(self.initial_queue) is not int or not 0 <= self.initial_queue <= 5):
            raise ValueError("Initial queue must be Auto or an integer from zero to five.")


def initialize_hospital(hospital):
    """Create incumbents with consistent negative admission offsets and residual events."""
    from live_hospital_state import Patient
    settings = hospital.warm_start_settings
    settings.validate()
    rng = np.random.default_rng(np.random.SeedSequence([hospital.seed, 4004]))
    target_count = lambda kind, fraction: math.floor(getattr(hospital.capacity, kind) * fraction + .5)
    beds = {"icu_beds": target_count("icu_beds", settings.icu_occupancy_target),
            "general_beds": target_count("general_beds", settings.general_occupancy_target)} if settings.enabled else dict(icu_beds=0, general_beds=0)
    requested = dict(doctors=target_count("doctors", settings.doctor_busy_target),
                     nurses=target_count("nurses", settings.nurse_busy_target))
    treatment_count = min(sum(beds.values()), requested["doctors"], requested["nurses"]) if settings.enabled else 0
    treatment_indices = set(rng.choice(sum(beds.values()), treatment_count, replace=False).tolist()) if treatment_count else set()
    warnings = []
    if settings.enabled and (requested["doctors"] != treatment_count or requested["nurses"] != treatment_count):
        warnings.append(f"Paired treatment assignments limited both staff busy counts to {treatment_count}; requested doctors {requested['doctors']}, nurses {requested['nurses']}.")
    # Preflight profile pools before taking ownership. No alternative routing model.
    pools = {}
    offset = 0
    for kind, count in beds.items():
        pools[kind] = tuple(p for p in hospital._profiles if bool(p[1]) == (kind == "icu_beds"))
        if count and not pools[kind]:
            raise ValueError(f"Warm start needs existing RF profiles routed to {kind}; adjust its occupancy target.")
        pools[(kind, "bed_stay")] = tuple(p for p in pools[kind] if p[3] > p[2])
        if any(index not in treatment_indices for index in range(offset, offset + count)) and not pools[(kind, "bed_stay")]:
            raise ValueError(f"Warm start needs {kind} profiles with LOS longer than treatment for BED_STAY.")
        offset += count
    ordinal = 0
    for kind, count in beds.items():
        for _ in range(count):
            treating = ordinal in treatment_indices
            ordinal += 1
            pool = pools[kind] if treating else pools[(kind, "bed_stay")]
            if not pool:
                raise ValueError(f"Warm start needs {kind} profiles with LOS longer than treatment for BED_STAY.")
            probability, high, treatment, stay, priority = pool[int(rng.integers(len(pool)))]
            elapsed = float(rng.uniform(.05, .95) * treatment if treating else
                treatment + rng.uniform(.05, .95) * (stay - treatment))
            hospital._patient_sequence += 1
            patient = Patient(f"P{hospital._patient_sequence:06d}", -elapsed, probability, high, kind,
                treatment, stay, priority, status="IN_TREATMENT" if treating else "BED_STAY",
                queue_entry_time=-elapsed, treatment_start_time=-elapsed,
                expected_treatment_completion=treatment - elapsed, expected_discharge_time=stay - elapsed,
                initialized_patient=True, pre_start_age_minutes=elapsed,
                initial_elapsed_treatment=min(elapsed, treatment), initial_elapsed_los=elapsed)
            hospital.active_patients[patient.patient_id] = patient
            patient.assigned_bed_id = hospital._acquire(kind, patient)
            if treating:
                patient.assigned_doctor_id = hospital._acquire("doctors", patient)
                patient.assigned_nurse_id = hospital._acquire("nurses", patient)
                hospital._schedule(patient.expected_treatment_completion, 0, "TREATMENT_COMPLETE", patient.patient_id)
            hospital._schedule(patient.expected_discharge_time, 1, "DISCHARGE", patient.patient_id)
    queue_count = 0
    if settings.enabled:
        queue_count = settings.initial_queue
        if queue_count is None:
            queue_count = 0 if all(hospital._free[k] for k in hospital.resources) else min(3, max(1, sum(beds.values()) // 100))
        for _ in range(queue_count):
            probability, high, treatment, stay, priority = hospital._profiles[int(rng.integers(len(hospital._profiles)))]
            age = float(rng.uniform(0., 15.))
            hospital._patient_sequence += 1
            patient = Patient(f"P{hospital._patient_sequence:06d}", -age, probability, high,
                "icu_beds" if high else "general_beds", treatment, stay, priority,
                queue_entry_time=-age, initialized_patient=True, pre_start_age_minutes=age)
            hospital.active_patients[patient.patient_id] = hospital._waiting[patient.patient_id] = patient
        hospital._policy_dirty = bool(queue_count)
    hospital.initial_state = dict(enabled=settings.enabled, seed=hospital.seed,
        seed_strategy="default_rng(SeedSequence([live_seed, 4004]))", settings=asdict(settings),
        profile_sha256=hospital.profile_sha256, active_patients=len(hospital.active_patients),
        waiting_patients=queue_count, in_treatment=treatment_count,
        bed_stay=sum(beds.values()) - treatment_count,
        resources=hospital.resource_summary(), warnings=warnings)
    if settings.enabled:
        hospital._event("WARM_START_INITIALIZED", f"Simulated initial operating state: {len(hospital.active_patients)} incumbents; {treatment_count} in treatment; {queue_count} waiting. Live arrivals start at zero.")

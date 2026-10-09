"""Live-only event-driven pressure integration and authoritative run criteria."""
from dataclasses import dataclass
import math

from scenario_evaluation import ScenarioConfig

RESOURCES = ("icu", "general", "doctor", "nurse")
PRESSURE_FIELDS = ("time_weighted_utilization", "peak_utilization", "minutes_above_target",
    "longest_above_target_streak", "minutes_at_full_saturation", "longest_full_saturation_streak")


@dataclass(frozen=True)
class LiveTargets(ScenarioConfig):
    sustained_overload_grace_minutes: float = 30.
    critical_saturation_grace_minutes: float = 15.

    def validate(self):
        super().validate()
        for name in ("sustained_overload_grace_minutes", "critical_saturation_grace_minutes"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1440:
                raise ValueError(f"{name} must be between 0 and 1440 simulated minutes.")


class PressureIntegrator:
    """Integrate constant-occupancy intervals; no polling or engine mutation."""
    def __init__(self, target):
        self.target = target
        self.duration = self.area = self.peak = 0.
        self.above = self.full = self.above_streak = self.full_streak = 0.
        self.longest_above = self.longest_full = 0.

    def observe(self, utilization):
        self.peak = max(self.peak, utilization)

    def integrate(self, utilization, minutes):
        if minutes < 0 or not 0 <= utilization <= 1:
            raise ValueError("Invalid utilization interval.")
        self.observe(utilization)
        if minutes == 0:
            return
        self.duration += minutes
        self.area += utilization * minutes
        if utilization > self.target:
            self.above += minutes
            self.above_streak += minutes
            self.longest_above = max(self.longest_above, self.above_streak)
        else:
            self.above_streak = 0.
        if utilization >= 1. - 1e-12:
            self.full += minutes
            self.full_streak += minutes
            self.longest_full = max(self.longest_full, self.full_streak)
        else:
            self.full_streak = 0.

    def result(self):
        return dict(time_weighted_utilization=self.area / self.duration if self.duration else 0.,
            peak_utilization=self.peak, minutes_above_target=self.above,
            longest_above_target_streak=self.longest_above, minutes_at_full_saturation=self.full,
            longest_full_saturation_streak=self.longest_full)


def resource_conditions(pressure, targets):
    return dict(average_utilization_pass=pressure["time_weighted_utilization"] <= targets.max_utilization_target,
        sustained_overload_pass=pressure["longest_above_target_streak"] <= getattr(targets, "sustained_overload_grace_minutes", 30.),
        critical_saturation_pass=pressure["longest_full_saturation_streak"] <= getattr(targets, "critical_saturation_grace_minutes", 15.))


def live_run_verdict(metrics, targets):
    conditions = dict(mean_wait_pass=metrics["mean_wait"] <= targets.mean_wait_target_minutes,
        high_risk_wait_pass=metrics["high_risk_mean_wait"] <= targets.high_risk_wait_target_minutes)
    for name in RESOURCES:
        conditions[name + "_util_pass"] = all(resource_conditions(metrics["resource_pressure"][name], targets).values())
    conditions["recovery_pass"] = metrics["queue_at_horizon_end"] <= metrics["checkpoint_queue"] + 2 and metrics["unresolved_event_waiting"] == 0
    return ("HANDLED_RUN" if all(conditions.values()) else "FAILED_RUN"), conditions


def aggregate_pressure(runs):
    return {name: {field: sum(row["resource_pressure"][name][field] for row in runs) / len(runs)
        for field in PRESSURE_FIELDS} for name in RESOURCES}

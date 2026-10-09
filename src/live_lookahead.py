"""Read-only, current-state predictive experiments; never invokes GA or LLM."""
from dataclasses import asdict
import copy
import math

import numpy as np

from live_hospital_state import KINDS, VERSION
from live_event_context import event_members, apply_event_context, context_event_metrics
from live_pressure import LiveTargets, PressureIntegrator, live_run_verdict, aggregate_pressure

UTIL_NAMES = dict(icu_beds="icu", general_beds="general", doctors="doctor", nurses="nurse")


def preview_is_current(result, hospital, event=None, targets=None, horizon=None, replications=None):
    provenance = result["provenance"]
    return (provenance["live_state_hash"] == hospital.state_hash()
        and (event is None or provenance["proposed_event"] == event.specification())
        and (targets is None or provenance["targets"] == target_values(targets))
        and (horizon is None or provenance["horizon_minutes"] == horizon)
        and (replications is None or provenance["replications"] == replications))


def target_values(targets):
    return {k: getattr(targets, k) for k in ("mean_wait_target_minutes",
        "high_risk_wait_target_minutes", "max_utilization_target", "robustness_threshold")} | {
        k: getattr(targets, k, default) for k, default in (("sustained_overload_grace_minutes", 30.),
            ("critical_saturation_grace_minutes", 15.))}


def handling_verdict(robustness, mean_wait, high_wait, targets):
    if robustness >= targets.robustness_threshold:
        return "HANDLED"
    severe = (mean_wait > 2 * targets.mean_wait_target_minutes or
              high_wait > 2 * targets.high_risk_wait_target_minutes)
    if robustness == 0 or (robustness <= 50 and severe):
        return "CANNOT HANDLE"
    return "AT RISK"


def simulate_lookahead_from_current_state(hospital, proposed_event, horizon_minutes=120,
                                        replications=5, targets=None, future_seed=None, policy_override=None):
    targets = targets or LiveTargets()
    LiveTargets(**target_values(targets)).validate()
    if hospital.status not in ("RUNNING", "PAUSED"):
        raise ValueError("Look-ahead requires a running or paused live session.")
    if not math.isfinite(horizon_minutes) or not 0 < horizon_minutes <= 480:
        raise ValueError("Look-ahead horizon must be greater than zero and at most 480 minutes.")
    if type(replications) is not int or not 1 <= replications <= 20:
        raise ValueError("Look-ahead replications must be an integer from 1 to 20.")
    future_seed = hospital.seed if future_seed is None else future_seed
    if type(future_seed) is not int or future_seed < 0:
        raise ValueError("Future seed must be a nonnegative integer.")
    members = event_members(proposed_event)
    known = [e.event_id in hospital.stress_events for e in members]
    if any(known) and not all(known):
        raise ValueError("Scenario is partially applied; prepare a new context.")
    existing_event = bool(members) and all(known)
    if proposed_event is not None and not existing_event:
        proposed_event.validate(hospital)
        if any(e.start_sim_time >= hospital.sim_time_minutes + horizon_minutes for e in members):
            raise ValueError("Every proposed event must start within the look-ahead horizon.")
    if existing_event and any(e.specification() != hospital.stress_events[e.event_id].specification() for e in members):
        raise ValueError("Event context does not match the checkpoint's applied events.")
    if policy_override is not None:
        policy_override.validate()
    checkpoint = hospital.checkpoint()
    state_hash = hospital.state_hash()
    start = checkpoint.sim_time_minutes
    end = start + horizon_minutes
    initial_queue = len(checkpoint._waiting)
    initial_arrivals, initial_completed = checkpoint._arrived, checkpoint._completed
    rows, seeds = [], []
    for index in range(replications):
        seed = int(np.random.SeedSequence([future_seed, 3002, index]).generate_state(1)[0])
        seeds.append(seed)
        clone = checkpoint.clone()
        clone.reseed_future(seed)
        if policy_override is not None:
            clone._policy = policy_override
            clone._temporary_policy = None
            clone._policy_dirty = True
        # Queue incumbents carry their elapsed wait; already-served incumbents
        # occupy capacity but do not dilute the future waiting cohort.
        cohort = dict(clone._waiting)
        seen = set(clone.active_patients)
        pressure = {k: PressureIntegrator(targets.max_utilization_target) for k in UTIL_NAMES}
        current_resources = clone.resource_summary()
        queue_peak = initial_queue
        recovery = None

        def observe():
            nonlocal queue_peak, recovery, current_resources
            for pid, patient in clone.active_patients.items():
                if pid not in seen:
                    cohort[pid] = patient
                    seen.add(pid)
            queue_peak = max(queue_peak, len(clone._waiting))
            current_resources = clone.resource_summary()
            for kind, resource in current_resources.items():
                pressure[kind].observe(resource["current_utilization"])
            if proposed_event is not None:
                if recovery is None and all(clone.event_recovered(e.event_id, initial_queue) for e in members):
                    recovery = max(0., clone.sim_time_minutes - min(e.start_sim_time for e in members))
            elif recovery is None and len(clone._waiting) <= initial_queue + 2:
                recovery = max(0., clone.sim_time_minutes - start)

        if proposed_event is not None and not existing_event:
            apply_event_context(clone, copy.deepcopy(proposed_event))
        else:
            clone.advance_simulation(0.)
        observe()
        while clone.sim_time_minutes < end:
            next_time = clone._agenda[0][0] if clone._agenda else end
            delta = min(60., end - clone.sim_time_minutes, max(0., next_time - clone.sim_time_minutes))
            for kind, resource in current_resources.items():
                pressure[kind].integrate(resource["current_utilization"], delta)
            clone.advance_simulation(delta)
            observe()
        waits = [p.observed_wait(end) for p in cohort.values()]
        high_waits = [p.observed_wait(end) for p in cohort.values() if p.high_risk]
        metrics = dict(additional_arrivals=clone._arrived - initial_arrivals,
            completed_patients=clone._completed - initial_completed, queue_peak=queue_peak,
            queue_at_horizon_end=len(clone._waiting), unfinished_patients=len(clone.active_patients),
            mean_wait=float(np.mean(waits)) if waits else 0.,
            high_risk_mean_wait=float(np.mean(high_waits)) if high_waits else 0.,
            p95_wait=float(np.percentile(waits, 95)) if waits else 0.,
            recovery_time_minutes=recovery, checkpoint_queue=initial_queue, unresolved_event_waiting=sum(p.source_event_id is not None for p in clone._waiting.values()), seed=seed,
            resource_pressure={UTIL_NAMES[k]: accumulator.result() for k, accumulator in pressure.items()},
            event_metrics=context_event_metrics(clone, proposed_event))
        for kind, name in UTIL_NAMES.items():
            metrics[f"{name}_utilization"] = float(pressure[kind].peak)
        verdict, conditions = live_run_verdict(metrics, targets)
        metrics.update(verdict=verdict, condition_breakdown=conditions,
            resource_warnings=[name for name, values in metrics["resource_pressure"].items()
                if values["peak_utilization"] > targets.max_utilization_target and conditions[name + "_util_pass"]])
        rows.append(metrics)
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in ("seed", "recovery_time_minutes")]
    aggregate = {k: float(np.mean([row[k] for row in rows])) for k in numeric}
    ranges = {k: dict(min=min(row[k] for row in rows), max=max(row[k] for row in rows)) for k in numeric}
    acceptable = sum(row["verdict"] == "HANDLED_RUN" for row in rows)
    robustness = 100. * acceptable / replications
    recoveries = [row["recovery_time_minutes"] for row in rows if row["recovery_time_minutes"] is not None]
    aggregate["recovery_time_minutes"] = float(np.mean(recoveries)) if recoveries else None
    aggregate["recovered_replications"] = len(recoveries)
    aggregate["resource_pressure"] = aggregate_pressure(rows)
    aggregate["event_metrics"] = {k: float(np.mean([row["event_metrics"][k] for row in rows])) for k in rows[0]["event_metrics"]}
    return dict(aggregate=aggregate, ranges=ranges, runs=rows, acceptable_runs=acceptable,
        robustness=robustness, verdict=handling_verdict(robustness, aggregate["mean_wait"],
            aggregate["high_risk_mean_wait"], targets),
        provenance=dict(schema_version="3.4-sustained-pressure", engine_version=VERSION, live_state_hash=state_hash,
            sim_time_minutes=start, physical_capacity=asdict(hospital.capacity),
            effective_capacity=hospital.resource_summary(), current_queue=initial_queue,
            current_policy=asdict(hospital.current_policy), base_arrival_rate=hospital.arrival_rate,
            effective_arrival_rate=hospital.effective_arrival_rate,
            profile_sha256=hospital.profile_sha256, proposed_event=proposed_event.specification() if proposed_event is not None else None,
            evaluated_policy=asdict(policy_override) if policy_override is not None else asdict(hospital.current_policy),
            horizon_minutes=horizon_minutes, replications=replications, future_base_seed=future_seed,
            seed_strategy="SeedSequence([future_base_seed, 3002, replication_index]); independent streams 0..3; known pending events retained",
            replication_seeds=seeds, targets=target_values(targets)),
        limitations="Simulated operational stress prediction, not an emergency alert or clinical triage system. Peak utilization is diagnostic only; acceptance uses window-average and continuous overload/saturation durations plus end-queue and event-wait recovery. Treatment/bed stay is not non-recovery.")

"""Verified operational evidence and deterministic explanation inputs; no GA logic."""
import math

import numpy as np

from live_canonical import plain, canonical_hash
from live_lookahead import handling_verdict, target_values
from live_pressure import LiveTargets, live_run_verdict, resource_conditions, aggregate_pressure, PRESSURE_FIELDS

DIAGNOSIS_VERSION = "live-condition-diagnosis-v2-sustained-pressure"
CONDITIONS = dict(mean_wait=("mean_wait_pass", "mean_wait_target_minutes"),
    high_risk_mean_wait=("high_risk_wait_pass", "high_risk_wait_target_minutes"),
    icu_utilization=("icu_util_pass", "max_utilization_target"),
    general_utilization=("general_util_pass", "max_utilization_target"),
    doctor_utilization=("doctor_util_pass", "max_utilization_target"),
    nurse_utilization=("nurse_util_pass", "max_utilization_target"))
RESOURCES = tuple(k for k in CONDITIONS if "utilization" in k)
RUN_NUMERIC = set(CONDITIONS) | {"p95_wait", "additional_arrivals", "completed_patients", "queue_peak",
    "queue_at_horizon_end", "unfinished_patients", "checkpoint_queue", "unresolved_event_waiting"}
EVENT_METRICS = {"surge_patients_introduced", "surge_patients_completed", "event_mean_wait", "event_high_risk_mean_wait",
    "event_unfinished_patients", "event_waiting_patients"}
PROVENANCE_FIELDS = {"version", "live_state_hash", "state_hash_schema", "live_state_context", "sim_time_minutes",
    "physical_capacity", "effective_capacity", "current_queue", "current_policy", "proposed_event", "profile_sha256",
    "horizon_minutes", "verification_replications", "targets", "ga_options", "gene_bounds", "seed_strategy", "live_seed", "policy_assumption"}
LABELS = dict(mean_wait="Mean wait", high_risk_mean_wait="High-risk wait", icu_utilization="ICU utilization",
    general_utilization="General-bed utilization", doctor_utilization="Doctor utilization", nurse_utilization="Nurse utilization",
    robustness="Robustness", queue_growth="Queue growth", non_recovery="Non-recovery", recovery="End-queue / event-wait recovery")


def _equal(first, second):
    return math.isclose(float(first), float(second), rel_tol=1e-9, abs_tol=1e-9)


def validate_prediction(prediction, targets, policy, capacity, replications):
    """Reject altered labels/aggregates before diagnosis or any provider request."""
    from live_hospital_state import OperatingPolicy
    OperatingPolicy(**policy).validate()
    prediction = plain(prediction)
    runs = prediction["runs"]
    if not runs or len(runs) != replications:
        raise ValueError("Verification replication count does not match provenance.")
    provenance = prediction["provenance"]
    if (provenance["targets"] != target_values(targets) or provenance["physical_capacity"] != capacity
            or provenance["evaluated_policy"] != policy):
        raise ValueError("Verification configuration/policy does not match the explanation context.")
    for run in runs:
        if set(run) != RUN_NUMERIC | {"recovery_time_minutes", "seed", "event_metrics", "verdict", "condition_breakdown", "resource_pressure", "resource_warnings"}:
            raise ValueError("Unsupported or missing verified run fields.")
        if any(type(run[k]) not in (int, float) or run[k] < 0 for k in RUN_NUMERIC) or any(run[k] > 1 for k in RESOURCES):
            raise ValueError("Verified metrics must be nonnegative; resource utilization cannot exceed one.")
        if set(run["event_metrics"]) - EVENT_METRICS or any(
                type(v) not in (int, float) or v < 0 for v in run["event_metrics"].values()):
            raise ValueError("Unsupported or invalid event-specific metrics.")
        if set(run["resource_pressure"]) != {"icu", "general", "doctor", "nurse"}:
            raise ValueError("Missing resource-pressure evidence.")
        for name, pressure in run["resource_pressure"].items():
            if set(pressure) != set(PRESSURE_FIELDS) or any(type(v) not in (int, float) or v < 0 for v in pressure.values()):
                raise ValueError("Invalid resource-pressure evidence.")
            if not 0 <= pressure["time_weighted_utilization"] <= pressure["peak_utilization"] <= 1:
                raise ValueError("Invalid average/peak utilization.")
            horizon = provenance["horizon_minutes"]
            if not (pressure["longest_full_saturation_streak"] <= pressure["minutes_at_full_saturation"] <= horizon
                    and pressure["longest_above_target_streak"] <= pressure["minutes_above_target"] <= horizon
                    and (pressure["minutes_at_full_saturation"] <= pressure["minutes_above_target"] or targets.max_utilization_target == 1)):
                raise ValueError("Invalid pressure duration ordering.")
            if pressure["longest_above_target_streak"] > pressure["minutes_above_target"] or pressure["minutes_at_full_saturation"] > horizon:
                raise ValueError("Invalid continuous pressure duration.")
            if not _equal(run[name + "_utilization"], pressure["peak_utilization"]):
                raise ValueError("Peak evidence mismatch.")
        verdict, conditions = live_run_verdict(run, targets)
        expected_warnings = [name for name, pressure in run["resource_pressure"].items()
            if pressure["peak_utilization"] > targets.max_utilization_target and conditions[name + "_util_pass"]]
        if sorted(run["resource_warnings"]) != sorted(expected_warnings):
            raise ValueError("Resource warnings disagree with verified evidence.")
        if run["verdict"] != verdict or run["condition_breakdown"] != conditions:
            raise ValueError("Verification labels disagree with deterministic Digital Twin conditions.")
    numeric = [k for k, v in runs[0].items() if type(v) in (int, float) and k not in ("seed", "recovery_time_minutes")]
    expected = {k: float(np.mean([run[k] for run in runs])) for k in numeric}
    recovered = [run["recovery_time_minutes"] for run in runs if run["recovery_time_minutes"] is not None]
    expected["recovered_replications"] = len(recovered)
    expected["recovery_time_minutes"] = float(np.mean(recovered)) if recovered else None
    if set(prediction["aggregate"]) != set(expected) | {"event_metrics", "resource_pressure"}:
        raise ValueError("Unsupported or missing verified aggregate fields.")
    for key, value in expected.items():
        actual = prediction["aggregate"][key]
        if (value is None and actual is not None) or (value is not None and (actual is None or not _equal(value, actual))):
            raise ValueError(f"Verified aggregate {key} does not match replication evidence.")
    if prediction["aggregate"]["resource_pressure"] != aggregate_pressure(runs):
        raise ValueError("Aggregate resource pressure disagrees with replications.")
    expected_event = {k: float(np.mean([run["event_metrics"][k] for run in runs])) for k in runs[0]["event_metrics"]}
    if set(prediction["aggregate"]["event_metrics"]) != set(expected_event) or any(
            not _equal(value, prediction["aggregate"]["event_metrics"][key]) for key, value in expected_event.items()):
        raise ValueError("Event-specific aggregates do not match verified replications.")
    acceptable = sum(run["verdict"] == "HANDLED_RUN" for run in runs)
    robustness = 100. * acceptable / len(runs)
    if (prediction["acceptable_runs"] != acceptable or not _equal(prediction["robustness"], robustness)
            or prediction["verdict"] != handling_verdict(robustness, expected["mean_wait"], expected["high_risk_mean_wait"], targets)):
        raise ValueError("Verified robustness/handling verdict disagrees with replication evidence.")
    return prediction


def metric_changes(before, after):
    old, new = before["aggregate"], after["aggregate"]
    values = {metric: (old[metric], new[metric], False) for metric in
        (*CONDITIONS, "p95_wait", "queue_peak", "queue_at_horizon_end", "unfinished_patients", "recovery_time_minutes")}
    values.update(robustness=(before["robustness"], after["robustness"], True),
        recovered_replications=(old["recovered_replications"], new["recovered_replications"], True),
        completed_patients=(old["completed_patients"], new["completed_patients"], True))
    for resource in ("icu", "general", "doctor", "nurse"):
        for field in PRESSURE_FIELDS:
            if field != "peak_utilization":
                values[resource + "_" + field] = (old["resource_pressure"][resource][field], new["resource_pressure"][resource][field], False)
    groups = dict(improved={}, unchanged={}, worsened={})
    for metric, (a, b, higher_better) in values.items():
        if a is None or b is None:
            direction = "unchanged" if a is b else ("improved" if a is None else "worsened")
        elif _equal(a, b) or abs(b - a) < (.001 if "utilization" in metric else 1 / 60 if "wait" in metric or "minutes" in metric or "streak" in metric else .1):
            direction = "unchanged"
        else:
            direction = "improved" if (b > a) == higher_better else "worsened"
        groups[direction][metric] = dict(current=a, optimized=b,
            change=b - a if a is not None and b is not None else None)
    return groups


def meaningful_improvement(changes):
    return any(metric not in RESOURCES for metric in changes["improved"])


def _condition(metric, values, target, count, total, acceptance=True):
    observed = max(values)
    critical = count == total or (observed >= .99 if "utilization" in metric else observed > 2 * target)
    return dict(metric=metric, label=LABELS[metric], observed=observed, observed_mean=float(np.mean(values)),
        observed_min=min(values), observed_statistic="maximum verified value (resource values are per-run peaks)",
        target=target, severity="critical" if critical else "major", failed_replications=count,
        total_replications=total, affects_acceptance=acceptance)


def _classification(failures, waits, queue_growth, nonrecovery, runs):
    total = len(runs)
    repeated = [f for f in failures if f["failed_replications"] / total >= .5]
    resources = {f["metric"] for f in failures}
    severe_multi = sum(sum(run["resource_pressure"][k.split("_")[0]]["time_weighted_utilization"] >= .95 for k in RESOURCES) >= 2 for run in runs) / total >= .5
    growing_overload = sum(run["queue_at_horizon_end"] > queue_growth["target"] and
        run["additional_arrivals"] > run["completed_patients"] for run in runs) / total >= .5
    if len(resources) >= 2:
        return "DEMAND_OVERLOAD" if len(repeated) >= 2 and severe_multi and growing_overload else "MULTIPLE_RESOURCE_BOTTLENECK"
    if any(f["metric"] in ("doctor_utilization", "nurse_utilization") for f in repeated):
        return "STAFF_BOTTLENECK"
    if repeated:
        return "BED_CAPACITY_BOTTLENECK"
    if failures:
        return "PHYSICAL_CAPACITY_BOTTLENECK"
    if nonrecovery / total >= .5:
        return "RECOVERY_FAILURE"
    if waits:
        return "WAITING_TIME_TARGET_FAILURE"
    if nonrecovery:
        return "RECOVERY_FAILURE"
    return "PROCESS_POLICY_LIMITATION"


def diagnose_prediction(prediction, targets, initial_queue=0, policy=None):
    runs, total = prediction["runs"], len(prediction["runs"])
    failed = []
    for metric, (flag, target) in CONDITIONS.items():
        count = sum(not run["condition_breakdown"][flag] for run in runs)
        if count:
            if metric in RESOURCES:
                name = metric.split("_")[0]
                pressure_rows = [run["resource_pressure"][name] for run in runs]
                condition = _condition(metric, [r["time_weighted_utilization"] for r in pressure_rows], targets.max_utilization_target, count, total)
                condition["observed_statistic"] = "maximum verified time-weighted utilization; durations can block even when average passes"
                condition["resource_pressure"] = {field: max(r[field] for r in pressure_rows) for field in PRESSURE_FIELDS}
                condition["thresholds"] = dict(utilization_target=targets.max_utilization_target,
                    sustained_overload_grace_minutes=getattr(targets, "sustained_overload_grace_minutes", 30.),
                    critical_saturation_grace_minutes=getattr(targets, "critical_saturation_grace_minutes", 15.))
                condition["failure_reasons"] = {reason: sum(not resource_conditions(r, targets)[reason] for r in pressure_rows)
                    for reason in resource_conditions(pressure_rows[0], targets)}
                failed.append(condition)
            else:
                failed.append(_condition(metric, [run[metric] for run in runs], getattr(targets, target), count, total))
    queue_count = sum(run["queue_at_horizon_end"] > initial_queue for run in runs)
    queue = _condition("queue_growth", [run["queue_at_horizon_end"] for run in runs], initial_queue, queue_count, total, False)
    historical_nonrecovery = sum(run["recovery_time_minutes"] is None for run in runs)
    nonrecovery = sum(not run["condition_breakdown"]["recovery_pass"] for run in runs)
    observations = []
    if queue_count:
        observations.append(queue)
    if historical_nonrecovery:
        observations.append(dict(metric="non_recovery", label=LABELS["non_recovery"], observed=historical_nonrecovery / total,
            target=0., severity="critical" if historical_nonrecovery == total else "major", failed_replications=historical_nonrecovery,
            total_replications=total, affects_acceptance=False, observed_statistic="fraction with no historical recovery time; event expiry may be beyond horizon, not itself an acceptance failure"))
    acceptable = prediction["robustness"] >= targets.robustness_threshold
    # Acceptance is scenario robustness, not a requirement that every run passes.
    failed.sort(key=lambda f: (-f["failed_replications"], -int(f["severity"] == "critical"),
        -(f["observed"] - f["target"]) / max(f["target"], .01), f["metric"]))
    resources = [f for f in failed if f["metric"] in RESOURCES]
    waits = [f for f in failed if f["metric"] not in RESOURCES]
    category = _classification(resources, waits, queue, nonrecovery, runs) if not acceptable else "NO_BLOCKING_CONDITION"
    if category == "WAITING_TIME_TARGET_FAILURE" and queue_count and policy and any(
            policy[k] > 0 for k in policy if "reserve" in k):
        category = "PROCESS_POLICY_LIMITATION"
    categories = [category]
    if nonrecovery and category != "RECOVERY_FAILURE":
        categories.append("RECOVERY_FAILURE")
    if waits and category not in ("PROCESS_POLICY_LIMITATION", "WAITING_TIME_TARGET_FAILURE"):
        categories.append("WAITING_TIME_TARGET_FAILURE")
    recovery_failures = sum(not run["condition_breakdown"]["recovery_pass"] for run in runs)
    if recovery_failures:
        failed.append(dict(metric="recovery", label=LABELS["recovery"], observed=max(run["queue_at_horizon_end"] for run in runs),
            target=initial_queue + 2, severity="critical" if recovery_failures == total else "major",
            failed_replications=recovery_failures, total_replications=total, affects_acceptance=True,
            unresolved_event_waiting=max(run["unresolved_event_waiting"] for run in runs)))
        if "RECOVERY_FAILURE" not in categories:
            categories.append("RECOVERY_FAILURE")
    if recovery_failures and not resources and not waits and not acceptable:
        category = "RECOVERY_FAILURE"
        categories[0] = category
    blockers = list(failed) if not acceptable else []
    if not acceptable:
        blockers.append(dict(metric="robustness", label=LABELS["robustness"], observed=prediction["robustness"],
            target=targets.robustness_threshold, severity="critical" if prediction["robustness"] == 0 else "major",
            failed_replications=total - prediction["acceptable_runs"], total_replications=total,
            affects_acceptance=True, observed_statistic="percentage of acceptable replications"))
    resource_category = category in ("STAFF_BOTTLENECK", "BED_CAPACITY_BOTTLENECK", "PHYSICAL_CAPACITY_BOTTLENECK",
                                     "MULTIPLE_RESOURCE_BOTTLENECK", "DEMAND_OVERLOAD")
    primary = (resources[0] if resources and resource_category else failed[0]) if failed and not acceptable else None
    return dict(overall_result="VERIFIED_ACCEPTABLE_POLICY" if acceptable else "NO_VERIFIED_FEASIBLE_POLICY",
        authoritative_verdict=prediction["verdict"], robustness=prediction["robustness"],
        blocking_conditions=blockers, residual_failed_conditions=failed if acceptable else [],
        operational_observations=observations, likely_constraint_type=category, classifications=categories,
        primary_limiting_factor=primary, secondary_limiting_factors=[f for f in failed if f is not primary] if not acceptable else [],
        limitation="Diagnostic evidence from finite-horizon verified simulations, not proof that every possible operating policy is infeasible.")


def diagnosis_inputs(result):
    return plain({k: result[k] for k in ("run_id", "provenance", "current_policy", "optimized_policy",
                                         "current_verified", "optimized_verified")})


def diagnose_live_result(result):
    inputs = diagnosis_inputs(result)
    provenance = inputs["provenance"]
    if set(provenance) - PROVENANCE_FIELDS:
        raise ValueError("Unsupported explanation provenance fields.")
    targets = LiveTargets(**provenance["targets"])
    targets.validate()
    before, after = [validate_prediction(inputs[k], targets, inputs[p], provenance["physical_capacity"],
        provenance["verification_replications"]) for k, p in
        (("current_verified", "current_policy"), ("optimized_verified", "optimized_policy"))]
    changes = metric_changes(before, after)
    diagnosis = diagnose_prediction(after, targets, provenance["current_queue"], inputs["optimized_policy"])
    diagnosis["current_policy_diagnosis"] = diagnose_prediction(before, targets, provenance["current_queue"], inputs["current_policy"])
    for item in (diagnosis, diagnosis["current_policy_diagnosis"]):
        for condition in item["blocking_conditions"] + item["residual_failed_conditions"]:
            if condition["metric"] in RESOURCES:
                kind = dict(icu_utilization="icu_beds", general_utilization="general_beds",
                    doctor_utilization="doctors", nurse_utilization="nurses")[condition["metric"]]
                initial = provenance["effective_capacity"][kind]["current_utilization"]
                condition["initial_state_utilization"] = initial
                condition["already_above_target_at_checkpoint"] = initial > condition["target"]
    diagnosis["metric_changes"] = changes
    diagnosis["secondary_improvements"] = {metric + "_improved": metric in changes["improved"] for metric in
        ("mean_wait", "high_risk_mean_wait", "p95_wait", "queue_peak", *RESOURCES, "recovery_time_minutes")}
    diagnosis["operational_result"] = ("VERIFIED ACCEPTABLE POLICY" if after["verdict"] == before["verdict"] == "HANDLED" else
        "VERIFIED FEASIBILITY RECOVERY" if after["verdict"] == "HANDLED" else
        "Operational improvement without feasibility recovery" if meaningful_improvement(changes) else "No verified operational improvement")
    diagnosis["recommendation_outcome"] = "VERIFIED SUCCESS" if after["verdict"] == "HANDLED" else "PARTIAL IMPROVEMENT" if meaningful_improvement(changes) else "NO MEANINGFUL IMPROVEMENT"
    diagnosis["policy_changes"] = {key: dict(current=inputs["current_policy"][key], optimized=value)
        for key, value in inputs["optimized_policy"].items() if value != inputs["current_policy"][key]}
    diagnosis["provenance"] = dict(version=DIAGNOSIS_VERSION, live_state_hash=provenance["live_state_hash"],
        input_sha256=canonical_hash(inputs))
    diagnosis["transient_resource_warnings"] = {name: dict(warned_replications=sum(name in run["resource_warnings"] for run in after["runs"]),
        total_replications=len(after["runs"]), evidence=after["aggregate"]["resource_pressure"][name])
        for name in ("icu", "general", "doctor", "nurse") if any(name in run["resource_warnings"] for run in after["runs"])}
    return plain(diagnosis)


def diagnosis_is_current(result):
    try:
        return result.get("diagnosis") == diagnose_live_result(result)
    except (ValueError, KeyError, TypeError):
        return False


def operational_responses(diagnosis):
    metrics = {f["metric"] for f in diagnosis["blocking_conditions"]}
    responses = []
    if "nurse_utilization" in metrics:
        responses += ["Review temporary nursing support or permitted staff reassignment.",
            "Review noncritical workload and external staffing support under hospital policy."]
    if "doctor_utilization" in metrics:
        responses += ["Review temporary doctor reinforcement or permitted shift reassignment.",
            "Review deferral of lower-priority workload under hospital policy."]
    if "icu_utilization" in metrics:
        responses += ["Review approved temporary ICU capacity, nonurgent admissions and eligible transfers under hospital policy."]
    if "general_utilization" in metrics:
        responses += ["Review safe discharge/transfer opportunities, elective admissions and approved overflow capacity."]
    if diagnosis["likely_constraint_type"] in ("DEMAND_OVERLOAD", "MULTIPLE_RESOURCE_BOTTLENECK"):
        responses += ["Review external support, permitted incoming-demand rerouting and coordinated transfers."]
    if "RECOVERY_FAILURE" in diagnosis["classifications"]:
        responses += ["Review a longer surge-policy/recovery monitoring window and permitted incoming-load reduction."]
    if diagnosis["likely_constraint_type"] in ("PROCESS_POLICY_LIMITATION", "WAITING_TIME_TARGET_FAILURE"):
        responses += ["Review queue/reserve policy tradeoffs and compare additional finite-horizon experiments."]
    if not responses:
        responses = ["Review verified results and residual failures before operational use; continue recovery monitoring."]
    return dict(title="Possible Operational Responses for Human Review", suggestions=responses,
        limitations="Options for clinician/administrator review only. No staffing, bed capacity, admissions, transfers or live policy is changed automatically.")


def build_explanation_payload(result, xai=None):
    # Regenerate learned evidence from the actual GA observations, never accept
    # arbitrary externally supplied rule strings as facts.
    from live_policy_explanation import explain_live_tree
    diagnosis = diagnose_live_result(result)
    if "diagnosis" in result and plain(result["diagnosis"]) != diagnosis:
        raise ValueError("Stored diagnosis is stale for the verified explanation inputs.")
    actual_xai = explain_live_tree(result)
    if xai is not None and plain(xai) != plain(actual_xai):
        raise ValueError("XAI input does not match the actual learned GA patterns.")
    inputs, provenance = diagnosis_inputs(result), plain(result["provenance"])
    event = provenance["proposed_event"]
    payload = dict(schema_version="live-explanation-input-v1", run_id=result["run_id"], source_provenance=provenance,
        event_context=dict(event=event, event_type=event["event_type"] if event else None,
            magnitude=event["parameters"] if event else None, duration_minutes=event["duration_minutes"] if event else None,
            horizon_minutes=provenance["horizon_minutes"], current_live_state={
                **provenance.get("live_state_context", dict(sim_time_minutes=provenance["sim_time_minutes"], queue=provenance["current_queue"])),
                "resources": provenance["effective_capacity"]}, live_state_hash=provenance["live_state_hash"]),
        targets=provenance["targets"], verification_replications=provenance["verification_replications"],
        physical_capacity=dict(resources=provenance["physical_capacity"], remained_unchanged=True),
        current_policy=dict(genes=inputs["current_policy"], verdict=inputs["current_verified"]["verdict"],
            robustness=inputs["current_verified"]["robustness"], verified_metrics=inputs["current_verified"]["aggregate"],
            diagnosis=diagnosis["current_policy_diagnosis"]),
        optimized_policy=dict(genes=inputs["optimized_policy"], verdict=inputs["optimized_verified"]["verdict"],
            robustness=inputs["optimized_verified"]["robustness"], verified_metrics=inputs["optimized_verified"]["aggregate"]),
        blocking_conditions=diagnosis["blocking_conditions"], deterministic_diagnosis=diagnosis,
        ga_improvements=dict(metrics=diagnosis["metric_changes"], policy_changes=diagnosis["policy_changes"]),
        decision_tree_xai={k: v for k, v in actual_xai.items() if k not in ("provenance", "run_id")},
        capacity_interpretation=dict(category=diagnosis["likely_constraint_type"], scope={
            "STAFF_BOTTLENECK": "staff-limited", "BED_CAPACITY_BOTTLENECK": "bed-limited",
            "DEMAND_OVERLOAD": "demand-limited", "MULTIPLE_RESOURCE_BOTTLENECK": "multiple-bottlenecks",
            "PHYSICAL_CAPACITY_BOTTLENECK": "resource-limited", "PROCESS_POLICY_LIMITATION": "process-limited",
            "WAITING_TIME_TARGET_FAILURE": "waiting-target-limited", "RECOVERY_FAILURE": "recovery-limited",
            "NO_BLOCKING_CONDITION": "no-verified-blocker"}[diagnosis["likely_constraint_type"]],
            finite_heuristic_search=True, mathematical_infeasibility_proven=False, physical_resources_fixed=True),
        human_review_responses=operational_responses(diagnosis),
        authority="Digital Twin evidence and deterministic diagnosis are authoritative; GA searches; Decision Tree explains correlations; LLM communicates; clinician/administrator decides.")
    return plain(payload)

"""Human-reviewed live process optimization, separate from the V2 resource GA."""
from dataclasses import asdict
from collections import deque

import pandas as pd
import streamlit as st

from live_hospital_state import OperatingPolicy
from live_event_context import ScenarioEvents
from live_events import LiveEvent
from live_policy_control import GENES, POLICY_BOUNDS
from live_policy_ga import LiveGAOptions, optimize_live_policy, optimization_is_current
from live_policy_explanation import explain_live_tree, explain_live_policy, explanation_is_current
from live_lookahead import simulate_lookahead_from_current_state, preview_is_current
from live_pressure import PRESSURE_FIELDS
from live_scenario_interpreter import audit_entry
from live_policy_diagnosis import diagnose_prediction
from live_time_display import format_duration, format_timestamp, format_time_text


def policy_table(current, recommended=None):
    return pd.DataFrame([dict(Parameter=gene.replace("_", " ").title(),
        Current=f"{current[gene]:.0%}" if "reserve" in gene else f"{current[gene]:.1f}",
        **({"Optimized": f"{recommended[gene]:.0%}" if "reserve" in gene else f"{recommended[gene]:.1f}"} if recommended else {})) for gene in GENES])


def render_adaptive_optimization(hospital, proposed_event, targets, horizon, reps, request=False):
    st.subheader("Adaptive Operating Policy")
    st.caption("Fixed physical capacity: " + " / ".join(f"{value} {kind.replace('_', ' ')}" for kind, value in asdict(hospital.capacity).items()) + ". V3 GA optimizes scheduling only. These are operational simulation controls, not validated clinical triage rules.")
    with st.expander("Current Operating Policy and Live GA Settings"):
        st.dataframe(policy_table(asdict(hospital.current_policy)), hide_index=True, width="stretch")
        col1, col2, col3 = st.columns(3)
        pop = col1.number_input("Live GA Population", 4, 40, 10, key="v3_ga_population",
            help="Candidate scheduling policies per generation. Physical resources cannot change.")
        generations = col2.number_input("Live GA Generations", 1, 30, 6, key="v3_ga_generations",
            help="Maximum evolutionary cycles; three generations without improvement may stop search early.")
        search_reps = col3.number_input("Live GA Search Replications", 1, 10, 3, key="v3_ga_search_reps",
            help="Common future seeds compare candidates fairly. Final verification uses the full look-ahead replication count.")
        st.caption(f"Final verification: {reps} runs over {format_duration(horizon)}. The comparison uses its own reproducible shared future experiment, so numbers may differ from the earlier event preview.")
    options = LiveGAOptions(population=int(pop), generations=int(generations), search_replications=int(search_reps), seed=hospital.seed)
    contexts = {"Current state / existing events": None}
    contexts.update({f"Applied {e.event_id}: {e.description}": e for e in hospital.stress_events.values() if e.status in ("SCHEDULED", "ACTIVE")})
    applied_specs = st.session_state.get("v3_applied_scenario_events", [])
    if applied_specs and all(row["event_id"] in hospital.stress_events for row in applied_specs) and any(hospital.stress_events[row["event_id"]].status in ("SCHEDULED", "ACTIVE") for row in applied_specs):
        bundle = ScenarioEvents(tuple(LiveEvent(**row) for row in applied_specs))
        contexts["Applied interpreted scenario: " + bundle.description] = bundle
    if proposed_event is not None and proposed_event.start_sim_time >= hospital.sim_time_minutes:
        contexts[f"Proposed {proposed_event.event_id}: {proposed_event.description}"] = proposed_event
    if request and proposed_event is not None:
        event = proposed_event
        st.session_state.v3_ga_context_name = next(k for k, e in contexts.items() if e is proposed_event)
    if st.session_state.get("v3_ga_context_name") not in contexts:
        st.session_state.v3_ga_context_name = next(iter(contexts))
    name = st.selectbox("Adaptive Optimization Context", list(contexts), format_func=format_time_text,
        key="v3_ga_context_name", help="Evaluate the current checkpoint, an applied event or a proposed event. Every candidate uses the same fixed physical resources.")
    event = contexts[name]
    skip = st.session_state.get("v3_skip_driver")
    busy_skip = bool(skip and skip.remaining_minutes)
    enabled = hospital.status in ("RUNNING", "PAUSED") and not busy_skip
    if st.button("Preview Current / Applied Event", disabled=not enabled,
        help="Applied and automatically generated events use the same current-state prediction and human-reviewed optimization workflow."):
        st.session_state.v3_applied_preview = simulate_lookahead_from_current_state(hospital, event, horizon, int(reps), targets)
    applied_preview = st.session_state.get("v3_applied_preview")
    if applied_preview and preview_is_current(applied_preview, hospital, targets=targets, horizon=horizon, replications=int(reps)) and applied_preview["provenance"]["proposed_event"] == (event.specification() if event else None):
        st.write(f"Current/applied event prediction: {applied_preview['verdict']} | Robustness {applied_preview['robustness']:.1f}%")
        preview_diagnosis = diagnose_prediction(applied_preview, targets, len(hospital._waiting), asdict(hospital.current_policy))
        st.caption("Deterministic diagnosis: " + preview_diagnosis["likely_constraint_type"])
        if applied_preview["verdict"] != "HANDLED":
            st.warning("This event is at risk. Run adaptive operating-policy optimization below; nothing is applied automatically.")
    run = st.button("Optimize Current Operating Policy", disabled=not enabled,
        help="Evaluate process-policy candidates from this current checkpoint using fixed physical resources. Nothing is applied automatically.")
    if (run or request) and enabled:
        try:
            progress = st.progress(0., text="Evaluating the current live checkpoint...")
            def update(completed, total, fitness):
                progress.progress(completed / total, text=f"Generation {completed}/{total} | Best fitness {fitness:.2f}")
            with st.spinner("Optimizing operating policy; live state is not advanced during search..."):
                result = optimize_live_policy(hospital, event, horizon, int(reps), targets, options, progress=update)
            audit_entry(st.session_state, hospital, "GA_RECOMMENDATION", run_id=result["run_id"], event_context=event.specification() if event else None,
                verdict=result["optimized_verified"]["verdict"], outcome=result["outcome"], policy=result["optimized_policy"])
            st.session_state.v3_live_ga_result = result
            st.session_state.pop("v3_recommendation_decision", None)
            st.session_state.pop("v3_modify_retry", None)
            recommendations = st.session_state.setdefault("v3_policy_recommendations", deque(maxlen=200))
            recommendations.append(dict(run_id=result["run_id"], sim_time_minutes=hospital.sim_time_minutes,
                previous_policy=result["current_policy"], new_policy=result["optimized_policy"],
                reason="Human-reviewed live GA recommendation", triggering_event_id=event.event_id if event else None,
                ga_fitness=result["search_fitness"], verified_verdict=result["optimized_verified"]["verdict"],
                verified_robustness=result["optimized_verified"]["robustness"], applied=False, restored=False))
            st.session_state.v3_live_xai = explain_live_tree(result)
            st.session_state.v3_live_ai = explain_live_policy(result, xai=st.session_state.v3_live_xai)
            progress.progress(1., text=f"Completed {result['generations_completed']} generations in {result['runtime_seconds']:.2f}s.")
        except ValueError as exc:
            st.error(f"Live policy optimization could not run: {exc}")
    result = st.session_state.get("v3_live_ga_result")
    current = optimization_is_current(result, hospital, event, horizon, int(reps), targets, options) if result else False
    if result:
        if not current:
            st.warning("STALE live policy recommendation/explanations: checkpoint, event context or settings changed. Run optimization again before applying. Pause playback to review a stable checkpoint.")
        else:
            before, after = result["current_verified"], result["optimized_verified"]
            with st.container(border=True):
                st.markdown("**" + ("OPTIMIZED POLICY CAN HANDLE EVENT" if after["verdict"] == "HANDLED" else "BEST AVAILABLE IMPROVEMENT — EVENT STILL " + after["verdict"] if result["outcome"] == "PARTIAL IMPROVEMENT" else "NO MEANINGFUL IMPROVEMENT") + "**")
                st.write(f"Robustness: {before['robustness']:.1f}% → {after['robustness']:.1f}%")
                source = result["diagnosis"]["current_policy_diagnosis"] if after["verdict"] == "HANDLED" else result["diagnosis"]
                primary_summary = source["primary_limiting_factor"]
                st.write(("Primary problem addressed: " if after["verdict"] == "HANDLED" else "Remaining problem: ") + (primary_summary["label"] if primary_summary else "Operational targets met."))
                changed = list(result["diagnosis"]["policy_changes"])
                st.caption("Key policy changes: " + (", ".join(g.replace("_", " ") for g in changed[:3]) if changed else "Current policy retained.") + ". Physical resources: unchanged.")
            st.markdown("**" + {"HANDLED": "CURRENT POLICY HANDLES EVENT", "AT RISK": "CURRENT POLICY AT RISK",
                "CANNOT HANDLE": "CURRENT POLICY CANNOT HANDLE EVENT"}[before["verdict"]] + "**")
            st.write(f"Current policy: {before['verdict']} | Optimized policy: {after['verdict']}")
            if after["verdict"] != "HANDLED":
                st.warning(result["feasibility_message"])
            with st.expander("Operating Policy Details"):
                st.dataframe(policy_table(result["current_policy"], result["optimized_policy"]), hide_index=True, width="stretch")
            rows = [dict(Metric="Predictive robustness", Current=f"{before['robustness']:.1f}%", Optimized=f"{after['robustness']:.1f}%")]
            for metric in ("mean_wait", "high_risk_mean_wait", "p95_wait", "queue_peak", "icu_utilization", "general_utilization", "doctor_utilization", "nurse_utilization", "recovery_time_minutes", "unfinished_patients"):
                def display(value):
                    if "wait" in metric or "time" in metric:
                        return format_duration(value) if value is not None else "Not recovered within horizon"
                    return f"{value:.1%}" if "utilization" in metric else f"{value:.1f}"
                rows.append(dict(Metric=metric.replace("_", " ").title(), Current=display(before['aggregate'][metric]), Optimized=display(after['aggregate'][metric])))
            with st.expander("Current vs Optimized Outcome Details"):
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            with st.expander("Current vs Optimized Resource Pressure"):
                comparison = []
                for resource in ("icu", "general", "doctor", "nurse"):
                    for field in PRESSURE_FIELDS:
                        a, b = [prediction["aggregate"]["resource_pressure"][resource][field] for prediction in (before, after)]
                        comparison.append(dict(Resource=resource.title(), Metric=field.replace("_", " "),
                            Current=f"{a:.1%}" if "utilization" in field else format_duration(a),
                            Optimized=f"{b:.1%}" if "utilization" in field else format_duration(b)))
                st.dataframe(pd.DataFrame(comparison), hide_index=True)
            st.caption("Verified outcome: " + result["outcome"])
            with st.expander("GA Search Details"):
                st.caption(f"Search fitness {result['search_fitness']:.2f} | {result['generations_completed']} generations | {result['unique_evaluations']} distinct candidates | Runtime {result['runtime_seconds']:.2f}s | Early stop {result['early_stopped']}. Physical capacity unchanged.")
            diagnosis = result["diagnosis"]
            st.subheader("Why the Optimized Policy Works" if after["verdict"] == "HANDLED" else "Why This Policy Still Fails")
            st.caption("AUTHORITATIVE EVIDENCE: verified Digital Twin metrics and deterministic condition diagnosis. GA searches; Decision Tree explains correlations; AI communicates; clinician/administrator decides.")
            st.markdown("**" + diagnosis["operational_result"] + "**")
            primary = diagnosis["primary_limiting_factor"]
            def condition_value(metric, value):
                if "utilization" in metric or metric == "non_recovery":
                    return f"{value:.1%}"
                if "wait" in metric or "minutes" in metric or "streak" in metric:
                    return format_duration(value)
                return f"{value:.1f}%" if metric == "robustness" else f"{value:.1f}"
            if primary:
                st.markdown(f"**Primary Limiting Factor: {primary['label']}**")
                cols = st.columns(3)
                reason = primary.get("failure_reasons", {})
                if reason and reason["average_utilization_pass"] == 0:
                    critical_reason = reason["critical_saturation_pass"] >= reason["sustained_overload_pass"]
                    field = "longest_full_saturation_streak" if critical_reason else "longest_above_target_streak"
                    threshold = "critical_saturation_grace_minutes" if critical_reason else "sustained_overload_grace_minutes"
                    cols[0].metric("Longest Full Saturation" if critical_reason else "Longest Continuous Overload", format_duration(primary["resource_pressure"][field]))
                    cols[1].metric("Configured Grace", format_duration(primary["thresholds"][threshold]))
                else:
                    cols[0].metric("Observed Time-Weighted Maximum", condition_value(primary["metric"], primary["observed"]))
                    cols[1].metric("Configured Target", condition_value(primary["metric"], primary["target"]))
                cols[2].metric("Failed Verified Replications", f"{primary['failed_replications']} / {primary['total_replications']}")
            st.write("Diagnosis: " + diagnosis["likely_constraint_type"])
            with st.expander("Verified Diagnosis Details"):
                all_conditions = diagnosis["blocking_conditions"] or diagnosis["residual_failed_conditions"]
                if all_conditions:
                    st.dataframe(pd.DataFrame([dict(Condition=f["label"], Observed=condition_value(f["metric"], f["observed"]),
                        Target=condition_value(f["metric"], f["target"]), Severity=f["severity"],
                        **{"Failed Replications": f"{f['failed_replications']} / {f['total_replications']}"}) for f in all_conditions]), hide_index=True)
                    st.caption("Observed resource utilization is the maximum time-weighted mean across verified runs. Duration failures can block acceptance even when that mean passes; pressure details show the cause.")
                else:
                    st.success("The verified policy meets the configured robustness threshold with no failed run conditions.")
                if diagnosis["operational_observations"]:
                    st.caption("End-queue and unresolved event waiting are authoritative recovery conditions; discharge completion and historical recovery-time observations remain diagnostic.")
                    st.dataframe(pd.DataFrame(diagnosis["operational_observations"]), hide_index=True)
                with st.expander("Verified Blocking Pressure Details and Transient Warnings"):
                    for condition in all_conditions:
                        if "resource_pressure" in condition:
                            st.write(condition["label"], condition["resource_pressure"], condition["thresholds"], condition["failure_reasons"])
                    for resource, warning in diagnosis["transient_resource_warnings"].items():
                        st.caption(f"{resource.title()}: transient warning in {warning['warned_replications']}/{warning['total_replications']} runs; peak alone did not fail the resource condition.")
                changes = diagnosis["metric_changes"]
                st.dataframe(pd.DataFrame([dict(Metric=metric.replace("_", " ").title(), Change=direction,
                    Current=condition_value(metric, row["current"]) if row["current"] is not None else "Not recovered",
                    Optimized=condition_value(metric, row["optimized"]) if row["optimized"] is not None else "Not recovered")
                    for direction, group in changes.items() for metric, row in group.items()]), hide_index=True)
            xai = explain_live_tree(result)
            st.session_state.v3_live_xai = xai
            if xai.get("available"):
                features = list(dict.fromkeys(item["feature"].replace("_", " ") for item in xai["rule_path"]))
                st.caption("XAI insight: learned " + xai["prediction"] + " patterns involve " + (", ".join(features) if features else "the tested policy cohort") + ". Correlation does not determine the verdict.")
            else:
                st.caption("XAI insight unavailable: all tested policies received the same deterministic class.")
            with st.expander("Decision Tree XAI Details"):
                st.caption(xai.get("message", "Explanation unavailable."))
                st.write(f"Training samples: {xai.get('training_sample_count', 0)} | Classes: {xai.get('class_distribution', {})}")
                if xai.get("available"):
                    st.write(" AND ".join(xai["rules"]) + " -> learned " + xai["prediction"])
                    st.caption(f"Learned class probabilities: {xai['class_probabilities']}; correlations do not determine the verdict.")
                    if after["verdict"] != "HANDLED":
                        pattern = xai.get("failure_pattern")
                        if pattern:
                            st.write("Tested policies following " + " AND ".join(pattern["rules"]) + " were associated with UNACCEPTABLE outcomes.")
                            st.caption(f"Observed unacceptable replications for this tested policy: {pattern['observed_unacceptable_replications']}/{pattern['tested_replications']}. " + pattern["limitations"])
                        else:
                            st.caption("No unacceptable learned path was available; no failure rule is invented.")
                    st.caption(f"Depth {xai['tree_depth']} | Leaves {xai['leaf_count']} | Training accuracy {xai['training_accuracy']:.1%} | Group-validation accuracy {xai['validation_accuracy'] if xai['validation_accuracy'] is not None else 'unavailable'}. Training accuracy is not proof of generalization.")
            explanation = st.session_state.get("v3_live_ai")
            if explanation and not explanation_is_current(explanation, result, xai):
                st.warning("STALE AI explanation inputs or generation state. Showing a fresh deterministic explanation from verified evidence.")
                explanation = None
            if explanation is None:
                explanation = explain_live_policy(result, xai=xai)
                st.session_state.v3_live_ai = explanation
            st.markdown("**AI Explanation**" if explanation["provider"].startswith("Groq") else "**Explanation (deterministic fallback)**")
            st.write(explanation["summary"])
            with st.expander("Detailed Explanation"):
                st.write(explanation["explanation"])
                st.caption(explanation["message"])
                if st.button("Generate Groq Live Explanation"):
                    st.session_state.v3_live_ai = explain_live_policy(result, use_groq=True, xai=xai)
                    st.rerun()
            responses = explanation["operational_responses"]
            st.markdown("**" + responses["title"] + "**")
            for suggestion in responses["suggestions"]:
                st.write("• " + suggestion)
            st.caption(responses["limitations"])
            mode = st.radio("Operating Policy Application", ("Apply Until Event Recovery", "Keep Policy Until Manually Changed"),
                index=0 if event is not None else 1, key="v3_policy_application_mode",
                help="Temporary policies await their triggering event and restore the previous policy after queue recovery. Proposed events still require separate explicit application.")
            if event is None and mode == "Apply Until Event Recovery":
                st.info("Choose an event context for automatic restoration; current-state policy application is persistent.")
            rejected = st.session_state.get("v3_recommendation_decision") == (result["run_id"], "REJECT")
            if rejected:
                st.info("Recommendation rejected. Live policy and capacity remain unchanged. Modify / Retry to request a new recommendation.")
            if st.button("Apply Optimized Operating Policy", disabled=not enabled or rejected,
                help="Change future queue/allocation rules only. Existing patients and resource ownership remain intact."):
                audit_entry(st.session_state, hospital, "HUMAN_APPLY_POLICY", run_id=result["run_id"], decision="APPLY", policy=result["optimized_policy"])
                hospital.apply_operating_policy(OperatingPolicy(**result["optimized_policy"]),
                    reason=f"Human-reviewed live GA {result['run_id']}", event=event,
                    temporary=event is not None and mode == "Apply Until Event Recovery",
                    fitness=result["search_fitness"], verification=after)
                audit_entry(st.session_state, hospital, "POLICY_APPLIED", run_id=result["run_id"], outcome="APPLIED", physical_capacity_unchanged=True)
                for record in reversed(st.session_state.get("v3_policy_recommendations", ())):
                    if record["run_id"] == result["run_id"]:
                        record["applied"] = True
                        break
                st.rerun()
            if st.button("Reject Recommendation"):
                audit_entry(st.session_state, hospital, "HUMAN_REJECT", run_id=result["run_id"], decision="REJECT")
                st.session_state.v3_recommendation_decision = (result["run_id"], "REJECT")
                for record in reversed(st.session_state.get("v3_policy_recommendations", ())):
                    if record["run_id"] == result["run_id"]:
                        record["human_decision"] = "REJECT"
                        break
                st.rerun()
            if st.button("Modify / Retry"):
                audit_entry(st.session_state, hospital, "HUMAN_MODIFY_RETRY", run_id=result["run_id"], decision="MODIFY_RETRY")
                st.session_state.v3_modify_retry = True
                st.rerun()
    if hospital.current_policy != hospital.normal_policy or hospital._temporary_policy:
        if hospital._temporary_policy:
            pending = hospital._temporary_policy
            st.caption(f"Temporary operating policy bound to {pending['event_id']}; awaiting event application/recovery if necessary. Physical resources unchanged.")
        if st.button("Restore Normal Operating Policy", disabled=not enabled):
            hospital.restore_normal_policy()
            st.rerun()
    with st.expander("Advanced Operating Policy"):
        active_policy = asdict(hospital.current_policy)
        if st.session_state.get("v3_manual_policy_reference") != active_policy:
            for gene, value in active_policy.items():
                st.session_state["v3_manual_" + gene] = float(value * (100 if "reserve" in gene else 1))
            st.session_state.v3_manual_policy_reference = active_policy
        values = {}
        for gene, (_, maximum, step) in POLICY_BOUNDS.items():
            scale = 100 if "reserve" in gene else 1
            st.session_state.setdefault("v3_manual_" + gene, float(active_policy[gene] * scale))
            entry = st.number_input(gene.replace("_", " ").title() + (" (%)" if scale == 100 else ""),
                0., maximum * scale, value=None, step=step * scale,
                key="v3_manual_" + gene,
                help="Operational simulation scheduling parameter. Reserves apply during stress/recovery; positive aging permits reserve borrowing after 60 minutes. Existing patients are never interrupted.")
            values[gene] = entry / scale if entry is not None else None
        complete = all(value is not None for value in values.values())
        if not complete:
            st.info("Enter all six operating-policy values before applying a manual policy.")
        if st.button("Apply Manual Operating Policy", disabled=not enabled or not complete):
            hospital.apply_operating_policy(OperatingPolicy(**values), reason="Manual operating-policy override", normal_policy=True)
            st.rerun()
    with st.expander("Provenance / Audit"):
        st.json(list(st.session_state.get("v3_scenario_audit", []))[-30:])
        if result:
            st.json(result["provenance"])
    with st.expander("Operating Policy History"):
        history = list(hospital.policy_history)[-30:]
        if history:
            rows = [{**row, "sim_time_minutes": format_timestamp(row["sim_time_minutes"])} for row in reversed(history)]
            for row in rows:
                if "restored_sim_time_minutes" in row:
                    row["restored_sim_time_minutes"] = format_timestamp(row["restored_sim_time_minutes"])
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        recommendations = list(st.session_state.get("v3_policy_recommendations", ()))[-30:]
        if recommendations:
            st.caption("Recommendation audit (including unapplied results). Restoration is recorded in the applied-policy history above.")
            st.dataframe(pd.DataFrame([{**row, "sim_time_minutes": format_timestamp(row["sim_time_minutes"])} for row in reversed(recommendations)]), hide_index=True)
        st.caption("In-memory policy history is capped at 200. Recommendations remain unapplied until explicitly reviewed; no per-generation disk exports.")

"""Human-confirmed natural-language event entry over the existing live pipeline."""
import streamlit as st

from live_scenario_interpreter import interpret_scenario, validate_scenario, scenario_context, interpretation_is_current, audit_entry
from live_event_context import apply_event_context
from live_lookahead import simulate_lookahead_from_current_state, preview_is_current
from live_policy_diagnosis import diagnose_prediction
from live_time_display import format_duration


def render_scenario_entry(hospital, targets, horizon, reps):
    st.subheader("Describe a Hospital Scenario")
    text = st.text_area("Hospital Scenario", key="v3_scenario_text", max_chars=4000, height=110,
        placeholder="Example: A bus accident brings 35 patients over 20 minutes, around 40% are high-risk, and 15 nurses are unavailable for the next hour.",
        help="Interpretation proposes supported simulated events only. It does not apply events, run GA or decide the authoritative verdict.")
    st.caption('Examples: "A bus accident sends 40 patients over 20 minutes." · "25 nurses are unavailable for two hours." · "Patient arrivals increase to 10 per hour for 90 minutes." · "20 ICU beds become unavailable for one hour." · "30 patients arrive while 10 doctors are unavailable."')
    enabled = hospital.status in ("RUNNING", "PAUSED")
    if st.button("Interpret Scenario", disabled=not enabled):
        with st.spinner("Interpreting supported simulated events..."):
            interpretation = interpret_scenario(text, hospital)
        st.session_state.v3_scenario_interpretation = interpretation
        st.session_state.pop("v3_scenario_preview", None)
        if interpretation["status"] == "VALID":
            for key in ("v3_proposed_event", "v3_proposal_form", "v3_lookahead_result"):
                st.session_state.pop(key, None)
        audit_entry(st.session_state, hospital, "INTERPRETATION", original_text=text,
            validation_outcome=interpretation["status"], parsed_events=interpretation.get("parsed_events", []),
            assumptions=interpretation.get("assumptions", []), message=interpretation.get("message"),
            technical_reason=interpretation.get("technical_reason"))
    interpretation = st.session_state.get("v3_scenario_interpretation")
    if not interpretation:
        return None, False
    if interpretation["status"] != "VALID":
        st.info(interpretation["message"])
        if interpretation.get("technical_reason"):
            with st.expander("Scenario Interpretation Details"):
                st.write(interpretation["technical_reason"])
        return None, False
    current = interpretation_is_current(interpretation, hospital, text)
    context = scenario_context(interpretation)
    with st.container(border=True):
        st.markdown("**SCENARIO INTERPRETATION**")
        for event in context.events:
            params = event.parameters
            description = (f"+{params['count']} patients over {format_duration(event.duration_minutes)}" if event.event_type == "PATIENT_SURGE" else
                f"{params['arrival_rate']:g} patients/hour for {format_duration(event.duration_minutes)}" if event.event_type == "ARRIVAL_RATE_SPIKE" else
                f"{params['count']} temporarily unavailable for {format_duration(event.duration_minutes)}")
            st.write(event.description + ": " + description)
            if "high_risk_proportion" in params:
                st.caption(f"High-risk profile share: {params['high_risk_proportion']:.0%}; existing RF predictions are preserved.")
            if event.start_sim_time > event.created_sim_time:
                st.caption("Starts after " + format_duration(event.start_sim_time - event.created_sim_time))
        for label in ("assumptions", "warnings"):
            if interpretation[label]:
                st.markdown("**" + label.title() + "**")
                for message in interpretation[label]:
                    st.write(message)
        if not current:
            st.warning("STALE scenario interpretation: live state or scenario text changed. Interpret again before previewing or applying.")
            unchanged_text_and_clock = text == interpretation["original_text"] and all(
                e.created_sim_time == hospital.sim_time_minutes and e.event_id == f"EVT-{hospital._stress_sequence + index + 1:06d}"
                for index, e in enumerate(context.events))
            if unchanged_text_and_clock and st.button("Revalidate Unchanged Scenario",
                help="Recheck the same event values after a policy change without another AI call. All old forecasts remain stale until look-ahead runs again."):
                data = dict(events=[dict(event_type=e.event_type, duration_minutes=e.duration_minutes,
                    start_delay_minutes=e.start_sim_time - e.created_sim_time,
                    parameters={key: e.parameters.get(key) for key in ("count", "arrival_rate", "high_risk_proportion")}) for e in context.events],
                    assumptions=interpretation["assumptions"], warnings=interpretation["warnings"], clarification=None)
                try:
                    st.session_state.v3_scenario_interpretation = validate_scenario(data, hospital, text)
                    st.session_state.pop("v3_scenario_preview", None)
                    audit_entry(st.session_state, hospital, "SCENARIO_REVALIDATED", outcome="VALID")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        if st.button("Run Look-Ahead", disabled=not enabled or not current):
            try:
                with st.spinner("Predicting the confirmed scenario from independent checkpoints..."):
                    result = simulate_lookahead_from_current_state(hospital, context, horizon, int(reps), targets)
                st.session_state.v3_scenario_preview = result
                audit_entry(st.session_state, hospital, "SCENARIO_LOOKAHEAD", interpretation_hash=interpretation["interpretation_hash"],
                    state_hash=result["provenance"]["live_state_hash"], verdict=result["verdict"], robustness=result["robustness"],
                    horizon_minutes=horizon, replications=int(reps))
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Edit Scenario"):
            audit_entry(st.session_state, hospital, "EDIT_SCENARIO", interpretation_hash=interpretation["interpretation_hash"])
            st.session_state.pop("v3_scenario_interpretation", None)
            st.session_state.pop("v3_scenario_preview", None)
            st.rerun()
        if st.button("Cancel"):
            audit_entry(st.session_state, hospital, "CANCEL_SCENARIO", interpretation_hash=interpretation["interpretation_hash"])
            st.session_state.pop("v3_scenario_interpretation", None)
            st.session_state.pop("v3_scenario_preview", None)
            st.rerun()
    result = st.session_state.get("v3_scenario_preview")
    preview_current = bool(result) and current and preview_is_current(result, hospital, context, targets, horizon, int(reps))
    request = False
    if result:
        if not preview_current:
            st.warning("STALE scenario look-ahead: rerun prediction before applying events or optimizing this scenario.")
        else:
            diagnosis = diagnose_prediction(result, targets, len(hospital._waiting))
            st.markdown(f"**Current scenario verdict: {result['verdict']}**")
            st.metric("Scenario Robustness", f"{result['robustness']:.1f}%")
            primary = diagnosis["primary_limiting_factor"]
            st.write("Main reason: " + (primary["label"] if primary else "Configured operational targets met."))
            with st.expander("Scenario Replication and Resource Pressure Details"):
                st.json(dict(metrics=result["aggregate"], runs=result["runs"], diagnosis=diagnosis))
            with st.expander("Scenario Provenance / Audit"):
                st.json(result["provenance"])
            request = st.button("Optimize Scenario Operating Policy", type="primary")
            if st.button("Apply Scenario Events", help="Explicitly applies the previewed events, not a GA policy. Fixed physical totals and current assignments are preserved."):
                try:
                    apply_event_context(hospital, context)
                    st.session_state.v3_applied_scenario_events = interpretation["parsed_events"]
                    st.session_state.v3_ga_context_name = "Applied interpreted scenario: " + context.description
                    st.session_state.pop("v3_scenario_interpretation", None)
                    st.session_state.pop("v3_scenario_preview", None)
                    audit_entry(st.session_state, hospital, "APPLY_SCENARIO_EVENTS", interpretation_hash=interpretation["interpretation_hash"],
                        event_ids=[e.event_id for e in context.events])
                    st.rerun()
                except Exception as exc:
                    audit_entry(st.session_state, hospital, "SCENARIO_APPLY_FAILED", error_type=type(exc).__name__)
                    st.error("Scenario events were not applied. Refresh and rerun look-ahead before retrying.")
    return (context if preview_current else None), request

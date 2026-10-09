"""Event proposal, read-only prediction, and explicit application UI."""
from collections import deque
import math

import pandas as pd
import streamlit as st

from live_events import EVENT_TYPES, OUTAGE_KINDS
from live_scenario_dashboard import render_scenario_entry
from live_adaptive_dashboard import render_adaptive_optimization
from live_time_display import format_duration, format_timestamp, format_time_text
from live_lookahead import simulate_lookahead_from_current_state, preview_is_current, target_values
from live_pressure import LiveTargets, RESOURCES, PRESSURE_FIELDS

PRESETS = ("Custom", "Traffic Accident", "ED Surge", "Doctor Shortage", "Nurse Shortage", "ICU Maintenance Outage")


def load_preset(hospital):
    preset = st.session_state.v3_event_preset
    values = dict(v3_event_count=12, v3_event_duration=20., v3_event_mix=False,
                  v3_event_proportion=.5, v3_event_immediate=False)
    if preset == "Traffic Accident":
        values.update(v3_event_type="PATIENT_SURGE", v3_event_mix=True)
        # Elevated relative to the cached clinical pool, without changing predictions.
        fraction = sum(bool(p[1]) for p in hospital._profiles) / len(hospital._profiles)
        values["v3_event_proportion"] = min(1., fraction + .2)
    elif preset == "ED Surge":
        values.update(v3_event_type="ARRIVAL_RATE_SPIKE", v3_event_duration=90.,
                      v3_event_rate=min(1000., hospital.arrival_rate * 1.5))
    elif preset != "Custom":
        event_type, kind, duration = {
            "Doctor Shortage": ("DOCTOR_SHORTAGE", "doctors", 120.),
            "Nurse Shortage": ("NURSE_SHORTAGE", "nurses", 120.),
            "ICU Maintenance Outage": ("ICU_BED_OUTAGE", "icu_beds", 180.)}[preset]
        values.update(v3_event_type=event_type, v3_event_count=max(1, math.ceil(getattr(hospital.capacity, kind) * .1)),
                      v3_event_duration=duration)
    st.session_state.update(values)


def render_event_controls(hospital, targets=None):
    targets = targets or LiveTargets()
    if st.session_state.get("v3_modify_retry"):
        st.info("Modify / Retry: adjust event magnitude/duration, look-ahead horizon, operational targets or overload tolerances below, then prepare/preview the event and run optimization again. Physical capacity is unchanged.")
    with st.expander("Advanced Live Twin Acceptance Settings"):
        sustained = st.number_input("Sustained Overload Grace (simulated minutes)", 0., 1440., 30., key="v3_overload_grace",
            help="A continuous interval above the utilization target fails only when longer than this grace period. Average utilization must also satisfy its target.")
        critical = st.number_input("Full-Saturation Grace (simulated minutes)", 0., 1440., 15., key="v3_saturation_grace",
            help="A continuous 100% interval fails only when longer than this grace period. Brief peaks remain diagnostic warnings.")
    targets = LiveTargets(**{**target_values(targets), "sustained_overload_grace_minutes": sustained,
        "critical_saturation_grace_minutes": critical})
    enabled = hospital.status in ("RUNNING", "PAUSED")
    st.subheader("Event Injection / Event Preview")
    st.caption("Synthetic operational stress events, not a production emergency alert system. Preview evaluates the current operating policy; it never runs GA or changes physical totals.")
    with st.expander("Random Synthetic Events (off by default)"):
        random_enabled = st.checkbox("Enable Random Events", key="v3_random_events", disabled=not enabled,
            help="Schedules reproducible stress events in simulated time. Refreshing the dashboard does not generate an event.")
        frequency = st.selectbox("Event Frequency", ("Low", "Medium", "High"), key="v3_random_frequency",
            help="Mean simulated intervals: Low 360 minutes, Medium 180, High 60. Arrivals of events are exponential.")
        allowed = st.multiselect("Allowed Random Event Types", EVENT_TYPES, default=list(EVENT_TYPES), key="v3_random_types")
        if enabled:
            try:
                hospital.configure_random_events(random_enabled, frequency, allowed)
            except ValueError as exc:
                st.error(str(exc))
    with st.expander("Look-Ahead Settings"):
        horizon = st.selectbox("Look-Ahead Horizon (minutes)", (30, 60, 120, 240, 480), index=2, key="v3_lookahead_horizon",
            format_func=lambda value: f"{format_duration(value)} ({value} minutes)")
        reps = st.number_input("Look-Ahead Replications", 1, 20, 5, key="v3_lookahead_reps",
            help="Each independent future begins from the same complete checkpoint. The real RNG and hospital state remain unchanged.")
        delay = st.number_input("Event Start Delay (minutes)", 0., 480., 0., key="v3_event_delay")
        st.caption(f"Live targets: mean wait <= {format_duration(targets.mean_wait_target_minutes)}; high-risk <= {format_duration(targets.high_risk_wait_target_minutes)}; time-weighted utilization <= {targets.max_utilization_target:.0%}; longest overload <= {format_duration(sustained)}; longest full saturation <= {format_duration(critical)}; robustness >= {targets.robustness_threshold:g}%. Peaks alone do not fail a run.")
    event_context, optimize_request = render_scenario_entry(hospital, targets, horizon, int(reps))
    with st.expander("Advanced / Manual Event Builder"):
        col, action = st.columns([3, 1])
        col.selectbox("Simulated Event Preset", PRESETS, key="v3_event_preset")
        action.button("Load Preset", on_click=load_preset, args=(hospital,), disabled=not enabled)
        event_type = st.selectbox("Event Type", EVENT_TYPES, key="v3_event_type")
        duration = st.number_input("Event Duration / Arrival Spread (minutes)", 0., 1440., value=st.session_state.get("v3_event_duration", 20.), key="v3_event_duration",
            help="Surge arrivals are spread uniformly within this interval. Temporary outages and spikes expire automatically after this duration.")
        params = {}
        immediate = False
        if event_type == "ARRIVAL_RATE_SPIKE":
            params["arrival_rate"] = st.number_input("Temporary Arrival Rate (patients/hour)", 0., 1000.,
                value=st.session_state.get("v3_event_rate", min(1000., hospital.arrival_rate * 1.5)), key="v3_event_rate",
                help="Base rate is preserved. If spikes overlap, the highest active rate wins.")
        else:
            count = st.number_input("Event Patient / Unavailable Resource Count", 1, 1000 if event_type == "PATIENT_SURGE" else 10000, value=st.session_state.get("v3_event_count", 12), key="v3_event_count",
                help="Surges add exactly this many patients. Outages cannot exceed physical capacity or interrupt assignments.")
            params["count"] = int(count)
        if event_type == "PATIENT_SURGE":
            immediate = st.checkbox("Immediate Surge Arrival", key="v3_event_immediate")
            if st.checkbox("Override Surge High-Risk Profile Mix", key="v3_event_mix",
                help="Samples existing RF-high/low clinical profile subsets; RF probabilities and threshold 0.50 are preserved."):
                params["high_risk_proportion"] = st.number_input("Surge High-Risk Proportion", 0., 1., value=st.session_state.get("v3_event_proportion", .5),
                    step=.05, key="v3_event_proportion")
        duration = 0. if immediate else duration
        spec = (event_type, duration, params, delay, st.session_state.v3_event_preset)
        if st.button("Prepare Proposed Event", disabled=not enabled):
            try:
                description = f"{st.session_state.v3_event_preset if st.session_state.v3_event_preset != 'Custom' else event_type.replace('_', ' ').title()}: {params}; {duration:g} min"
                st.session_state.pop("v3_scenario_interpretation", None)
                st.session_state.pop("v3_scenario_preview", None)
                event_context = None
                st.session_state.v3_proposed_event = hospital.propose_event(event_type, duration, params, description, delay)
                st.session_state.v3_proposal_form = spec
                st.session_state.pop("v3_lookahead_result", None)
            except ValueError as exc:
                st.error(str(exc))
        event = st.session_state.get("v3_proposed_event")
        if event is not None:
            st.markdown(f"**Proposed Event {event.event_id}** — {format_time_text(event.description)}")
            st.caption(f"Start {format_timestamp(event.start_sim_time)} | Duration / arrival spread {format_duration(event.duration_minutes)} | Look-ahead horizon {format_duration(horizon)}")
            form_current = st.session_state.get("v3_proposal_form") == spec
            if not form_current:
                st.warning("Event controls changed. Prepare the proposed event again before previewing or applying.")
            resources = hospital.resource_summary()
            st.caption("Current free resources: " + " | ".join(f"{kind.replace('_', ' ')} {r['available']}" for kind, r in resources.items()) + f" | Queue {len(hospital._waiting)}")
            if st.button("Run Event Look-Ahead", disabled=not enabled or not form_current):
                audit = st.session_state.setdefault("v3_preview_audit", deque(maxlen=50))
                audit.append(dict(sim_time_minutes=hospital.sim_time_minutes, event_type="LOOKAHEAD_STARTED", event_id=event.event_id))
                try:
                    with st.spinner("Predicting from independent current-state checkpoints..."):
                        result = simulate_lookahead_from_current_state(hospital, event, horizon, int(reps), targets)
                    event.lookahead_run = True
                    st.session_state.v3_lookahead_result = result
                    audit.append(dict(sim_time_minutes=hospital.sim_time_minutes, event_type="LOOKAHEAD_RESULT",
                        event_id=event.event_id, description=result["verdict"]))
                except ValueError as exc:
                    st.error(f"Look-ahead unavailable: {exc}. Prepare a new event at the current clock.")
            result = st.session_state.get("v3_lookahead_result")
            event_context = event if form_current and event.start_sim_time >= hospital.sim_time_minutes else None
            current = bool(result) and form_current and preview_is_current(result, hospital, event, targets, horizon, int(reps))
            if result:
                if not current:
                    st.warning("STALE event preview: live state or settings changed. Prepare/preview again. Pause automatic playback to retain a current preview.")
                st.markdown(f"**{'Historical prediction' if not current else 'Can the current policy handle it?'}: {result['verdict']}**")
                if current and result["verdict"] == "HANDLED":
                    st.success("Current operating policy can handle this event.")
                elif current:
                    st.warning("Current operating policy is at risk under this event. Adaptive optimization can test scheduling changes with fixed physical capacity.")
                    optimize_request = st.button("Optimize Operating Policy", type="primary")
                aggregate = result["aggregate"]
                cols = st.columns(4)
                for col, label, value in zip(cols, ("Predictive Robustness", "Predicted Mean Wait", "Predicted High-Risk Wait", "Predicted P95 Wait"),
                    (f"{result['robustness']:.1f}%", format_duration(aggregate['mean_wait']), format_duration(aggregate['high_risk_mean_wait']), format_duration(aggregate['p95_wait']))):
                    col.metric(label, value)
                st.caption(f"Additional arrivals {aggregate['additional_arrivals']:.1f} | Completions {aggregate['completed_patients']:.1f} | Queue peak {aggregate['queue_peak']:.1f} | End queue {aggregate['queue_at_horizon_end']:.1f} | Unfinished {aggregate['unfinished_patients']:.1f}")
                st.caption("Predicted resource peaks: " + " | ".join(f"{name.title()} {aggregate[name + '_utilization']:.1%}" for name in ("icu", "general", "doctor", "nurse")))
                recovery = aggregate["recovery_time_minutes"]
                st.caption("Not recovered within look-ahead horizon." if recovery is None else
                    f"Mean queue-recovery time among recovered runs: {format_duration(recovery)} after event start ({aggregate['recovered_replications']}/{reps} runs). Recovery does not mean all patients have discharged.")
                with st.expander("Replication and Resource Pressure Details"):
                    pressure_rows = [dict(Resource=name.title(), **{field: f"{values[field]:.1%}" if "utilization" in field else format_duration(values[field]) for field in PRESSURE_FIELDS})
                        for name, values in aggregate["resource_pressure"].items()]
                    st.dataframe(pd.DataFrame(pressure_rows), hide_index=True)
                    st.caption("Durations above are averages across runs; detailed per-run pressure and warnings follow.")
                    st.caption("HANDLED: robustness meets threshold. CANNOT HANDLE: zero acceptable runs, or robustness ≤50% and average mean/high-risk wait exceeds twice its target. AT RISK: other below-threshold outcomes. Waits include the current queue and new arrivals, censored at the horizon. Resource means and continuous pressure durations must meet their limits; end queue must be at most checkpoint queue + 2, with no unresolved event waiting patients. Treatment and bed stay are not recovery failures.")
                    display_runs = [{k: format_duration(v) if k in
                        ("mean_wait", "high_risk_mean_wait", "p95_wait", "recovery_time_minutes") else v
                        for k, v in row.items() if k not in ("event_metrics", "condition_breakdown")}
                        for row in result["runs"]]
                    st.dataframe(pd.DataFrame(display_runs).rename(columns={"recovery_time_minutes": "Recovery Time"}), hide_index=True)
            if st.button("Apply Event", disabled=not enabled or not current,
                help="Apply only this current previewed proposal to the actual hospital. Playback state and physical totals are retained."):
                try:
                    hospital.apply_event(event)
                    st.session_state.pop("v3_proposed_event", None)
                    st.session_state.pop("v3_lookahead_result", None)
                    st.rerun()
                except ValueError as exc:
                    st.error(f"Event was not applied: {exc}")
    render_adaptive_optimization(hospital, event_context, targets, horizon, int(reps), request=optimize_request)
    st.subheader("Active Events")
    active = hospital.event_rows()
    if active:
        st.dataframe(pd.DataFrame([dict(Event=e["event_id"], Type=e["event_type"], Start=format_timestamp(e["start_sim_time"]),
            End=format_timestamp(e["end_sim_time"]), Status=e["status"], Magnitude=str(e["parameters"]),
            Duration=format_duration(e["duration_minutes"]), Remaining=format_duration(e["remaining_minutes"])) for e in active]), hide_index=True, width="stretch")
    else:
        st.caption("No active or scheduled stress events.")
    st.caption(f"Base feed {hospital.arrival_rate:g}/hour | Effective feed {hospital.effective_arrival_rate:g}/hour. Temporary effects expire automatically.")
    with st.expander("Recent Stress Events and Preview Audit"):
        ended = hospital.event_rows(active_only=False)[-20:]
        if ended:
            st.dataframe(pd.DataFrame([dict(Event=e["event_id"], Type=e["event_type"], Status=e["status"],
                **{k: format_duration(v) if k in ("event_mean_wait", "event_high_risk_mean_wait") else v
                   for k, v in hospital.event_metrics(e["event_id"]).items()}) for e in ended]), hide_index=True)
        audit = list(st.session_state.get("v3_preview_audit", []))[-20:]
        if audit:
            st.dataframe(pd.DataFrame([{**row, "sim_time_minutes": format_timestamp(row["sim_time_minutes"])}
                for row in audit]).rename(columns={"sim_time_minutes": "Simulated Time"}), hide_index=True)
        st.caption("Preview audit stays in the UI session; previewing does not append to or mutate the actual hospital event log.")

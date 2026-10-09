"""V3 view kept separate from the validated V2 dashboard workflow."""
from dataclasses import asdict
import json
import time

import pandas as pd
import streamlit as st

from live_event_dashboard import render_event_controls
from live_time_display import format_duration, format_timestamp, format_time_text
from live_hospital_state import KINDS, LiveCapacity, LiveClockDriver, LiveHospitalState
from scenario_evaluation import load_patient_profiles
from live_policy_control import DEMO_CAPACITY, DEMO_ARRIVAL_RATE, NORMAL_POLICY_VALUES, SIMULATION_SPEEDS, MAX_MANUAL_MINUTES
from live_hospital_state import OperatingPolicy
from live_skip_control import LiveSkipDriver
from live_initialization import WarmStartSettings
from live_presentation import status_html, expanded_status_html, overview_html
from live_presentation import section_anchor, render_initial_details
from live_pressure import LiveTargets

LABELS = dict(icu_beds="ICU Beds", general_beds="General Beds", doctors="Doctors", nurses="Nurses")


def wall_seconds():
    """Monotonic playback clock; separate seam for deterministic UI tests."""
    return time.monotonic()


def format_time(minutes):
    return format_timestamp(minutes)


def get_live_hospital(state):
    """Read-only lookup: widget reruns must not construct or reset a session."""
    return state.get("v3_live_hospital")


@st.fragment(run_every="1s")
def render_live_twin(data_path, model_path, targets=None, target_controls=False):
    # Permanent slots precede controls; variable content stays within each slot.
    with st.container(key="v3_top_status", border=False):
        section_anchor("live")
        status_slot = st.empty()
        with st.expander("Expand Status", expanded=False, key="v3_expand_status", on_change="ignore"):
            expanded_slot = st.empty()
    hospital = get_live_hospital(st.session_state)
    if hospital is not None and not hasattr(hospital, "initial_state"):
        st.warning("This retained session predates warm-start initialization. Reset it explicitly to use the new initialization settings.")
        if st.button("Reset Legacy Live Session"):
            for key in ("v3_live_hospital", "v3_clock_driver", "v3_skip_driver", "v3_proposed_event",
                    "v3_lookahead_result", "v3_live_ga_result", "v3_live_xai", "v3_live_ai",
                    "v3_scenario_interpretation", "v3_scenario_preview", "v3_snapshot_download"):
                st.session_state.pop(key, None)
            st.rerun()
        return
    if "v3_pending_autoplay" in st.session_state:
        st.session_state.v3_autoplay = st.session_state.pop("v3_pending_autoplay")
    locked = hospital is not None and hospital.status != "READY"
    with st.expander("Hospital Setup · Fixed Physical Capacity", expanded=hospital is None):
        st.subheader("Fixed Live Hospital Capacity")
        capacity_values = {}
        for column, kind in zip(st.columns(4), KINDS):
            with column:
                capacity_values[kind] = st.number_input(f"Live {LABELS[kind]}", 1, 10000,
                    DEMO_CAPACITY[kind], key=f"v3_capacity_{kind}", disabled=locked,
                    help="Fixed physical capacity for this live session. Reset Live Twin before editing; adaptive optimization cannot change it.")
        feed_col, seed_col = st.columns(2)
        with feed_col:
            rate = st.number_input("Live Base Arrival Rate (patients/hour)", 0., 1000., DEMO_ARRIVAL_RATE,
                key="v3_arrival_rate", disabled=locked,
                help="Poisson live arrival feed with exponential inter-arrival minutes of mean 60/rate. Reset before changing the feed.")
        with seed_col:
            seed = st.number_input("Live Random Seed", 0, 2**32 - 1, 42, key="v3_seed", disabled=locked,
                help="Same live configuration and seed reproduce arrivals and clinical profile sampling. Synthetic patient IDs are used.")
        with st.expander("Advanced Live Twin Initialization Settings"):
            warm_enabled = st.checkbox("Enable Warm Start", value=True, key="v3_warm_enabled", disabled=locked,
                help="Create a seeded simulated initial operating state using existing RF profiles. Reset before changing settings; these are not real hospital measurements.")
            warm_values = {}
            for col, (name, label, default) in zip(st.columns(4), (
                    ("icu_occupancy_target", "ICU Initial Occupancy Target (%)", 55),
                    ("general_occupancy_target", "General Bed Initial Occupancy Target (%)", 25),
                    ("doctor_busy_target", "Doctor Initial Busy Target (%)", 20),
                    ("nurse_busy_target", "Nurse Initial Busy Target (%)", 20))):
                warm_values[name] = col.number_input(label, 0, 100, default, key="v3_warm_" + name,
                    disabled=locked, help="Target percentage of fixed physical capacity. Treatment uses one doctor AND one nurse, so actual staff busy counts may be below target.") / 100.
            queue_choice = st.selectbox("Initial Waiting Queue", ("Auto", 0, 1, 2, 3, 4, 5), key="v3_warm_queue", disabled=locked,
                help="Auto uses zero when all resource types have free capacity, otherwise at most three. Explicit queued patients use existing RF profiles and seeded ages of up to 15 minutes.")
            reset_mode = st.selectbox("Reset Initialization", ("Reset with Warm Start", "Reset Empty"), key="v3_reset_initialization",
                help="Reset Empty is for debugging. Reset with Warm Start reproduces initialization for unchanged seed, capacities, settings and profile pool.")
            warm_settings = WarmStartSettings(enabled=warm_enabled and reset_mode != "Reset Empty",
                initial_queue=None if queue_choice == "Auto" else queue_choice, **warm_values)
        if target_controls:
            with st.expander("Operational Targets", expanded=False):
                mean_wait = st.number_input("Mean Wait Target (minutes)", 1., 120., 20., key="mean_wait_number",
                    help="Maximum acceptable average observed wait in look-ahead replications.")
                high_wait = st.number_input("High-Risk Wait Target (minutes)", 1., 120., 10., key="high_risk_wait_number",
                    help="Maximum acceptable average wait for RF-classified high-risk patients; RF threshold stays 0.50.")
                utilization = st.number_input("Maximum Utilization (%)", 0., 100., 90., key="live_utilization_target",
                    help="Time-weighted utilization target. Brief peaks are warnings; sustained-pressure grace periods are configured in Advanced Live Twin settings.")
                robustness = st.number_input("Robustness Threshold (%)", 0., 100., 90., key="live_robustness_threshold",
                    help="Required percentage of handled look-ahead replications.")
            targets = LiveTargets(mean_wait_target_minutes=mean_wait, high_risk_wait_target_minutes=high_wait,
                max_utilization_target=utilization / 100., robustness_threshold=robustness)
    with st.container(key="v3_playback"):
        playback_speed, playback_auto = st.columns([1, 2])
        speed = playback_speed.selectbox("Simulation Speed", SIMULATION_SPEEDS, index=2,
            format_func=lambda x: f"{x}x", key="v3_speed",
            help="Playback speed. Simulated events retain their deterministic ordering at every speed.")
        auto = playback_auto.toggle("Automatic Live Playback", key="v3_autoplay",
            help="Advance using elapsed wall time, with bounded catch-up steps. Turn off to use manual stepping. No infinite loop or background simulation thread.")
        # Accrue elapsed time exactly once, before processing control transitions.
        driver = st.session_state.get("v3_clock_driver")
        if driver is not None:
            skip = st.session_state.get("v3_skip_driver")
            skipping = bool(skip and skip.remaining_minutes)
            if not st.session_state.get("v3_synchronous_action", False):
                driver.tick(wall_seconds(), speed=speed, enabled=auto and not skipping)
        controls = st.columns(5)
        with controls[0]:
            start = st.button("Start Live Simulation", type="primary", disabled=hospital is not None and hospital.status != "READY", width="stretch")
        with controls[1]:
            pause = st.button("Pause", disabled=hospital is None or hospital.status != "RUNNING", width="stretch")
        with controls[2]:
            resume = st.button("Resume", disabled=hospital is None or hospital.status != "PAUSED", width="stretch")
        with controls[3]:
            stop = st.button("Stop Live Simulation", disabled=hospital is None or hospital.status not in ("RUNNING", "PAUSED"), width="stretch")
        with controls[4]:
            reset = st.button("Reset Live Twin", disabled=hospital is None, width="stretch")
        start_status = st.empty()
        status_slot.markdown(status_html(hospital, speed), unsafe_allow_html=True)
        if hospital is not None:
            expanded_slot.markdown(expanded_status_html(hospital), unsafe_allow_html=True)
        else:
            expanded_slot.info("Patient flow and performance appear after Start.")
        if start:
            try:
                with start_status.container():
                    st.info("Loading cached RF profiles and preparing live state...")
                    # Cache is owned by the validated V2 profile loader; inference
                    # is performed once per source/model version, never per arrival.
                    profiles = load_patient_profiles(data_path, model_path)
                    hospital = LiveHospitalState(profiles, LiveCapacity(**capacity_values), float(rate), int(seed),
                        operating_policy=OperatingPolicy(**NORMAL_POLICY_VALUES), warm_start=warm_settings)
                hospital.start()
                driver = LiveClockDriver(hospital)
                start_playback = bool(warm_settings.enabled)
                driver.tick(wall_seconds(), speed=speed, enabled=start_playback)
                st.session_state.v3_pending_autoplay = start_playback
                st.session_state.v3_live_hospital = hospital
                st.session_state.v3_clock_driver = driver
                st.session_state.v3_skip_driver = LiveSkipDriver(hospital)
                st.session_state.pop("v3_snapshot_download", None)
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(f"Live twin could not start: {exc}")
        if hospital is not None:
            if pause:
                hospital.pause()
                if st.session_state.get("v3_skip_driver"):
                    st.session_state.v3_skip_driver.paused = True
            if resume:
                hospital.resume()
                if st.session_state.get("v3_skip_driver"):
                    st.session_state.v3_skip_driver.paused = False
                driver.reanchor(wall_seconds())
            if stop:
                hospital.stop()
                if st.session_state.get("v3_skip_driver"):
                    st.session_state.v3_skip_driver.cancel()
            if reset:
                hospital.warm_start_settings = warm_settings
                hospital.reset()
                for key in ("v3_proposed_event", "v3_lookahead_result", "v3_preview_audit", "v3_proposal_form", "v3_live_ga_result", "v3_live_ga_status", "v3_live_ga_error", "v3_live_xai", "v3_live_ai", "v3_policy_recommendations", "v3_manual_policy_reference", "v3_scenario_interpretation", "v3_scenario_preview", "v3_scenario_audit", "v3_applied_scenario_events", "v3_modify_retry", "v3_recommendation_decision"):
                    st.session_state.pop(key, None)
                st.session_state.v3_clock_driver = LiveClockDriver(hospital)
                st.session_state.v3_skip_driver = LiveSkipDriver(hospital)
                st.session_state.v3_pending_autoplay = False
                st.session_state.pop("v3_snapshot_download", None)
            if pause or resume or stop or reset:
                st.rerun()
        with st.expander("Fast Forward Controls", key="v3_fast_forward_controls"):
            step_col, action_col = st.columns([2, 1])
            with step_col:
                delta = st.number_input("Fast Forward (simulated minutes)", 0.1, MAX_MANUAL_MINUTES, 5., key="v3_manual_delta",
                    help="Process all scheduled events without playback throttling or intermediate dashboard tables. Maximum 30 days; physical capacity and stochastic streams are unchanged.")
            with action_col:
                if st.button("Fast Forward", disabled=hospital is None or hospital.status not in ("RUNNING", "PAUSED"), width="stretch"):
                    skip = st.session_state.setdefault("v3_skip_driver", LiveSkipDriver(hospital))
                    try:
                        skip.request(float(delta))
                        driver.enabled = False
                        driver.reanchor(wall_seconds())
                    except ValueError as exc:
                        st.error(str(exc))
            if hospital is not None:
                skip = st.session_state.setdefault("v3_skip_driver", LiveSkipDriver(hospital))
                st.caption("Presets · custom duration is entered above. Pause / Resume remain beside the clock.")
                presets = (("+5 min", 5), ("+30 min", 30), ("+1 hour", 60), ("+6 hours", 360),
                    ("+12 hours", 720), ("+1 day", 1440), ("+3 days", 4320), ("+7 days", 10080))
                for row in (presets[:4], presets[4:]):
                    for col, (label, minutes) in zip(st.columns(4), row):
                        if col.button(label, disabled=bool(skip.remaining_minutes) or hospital.status not in ("RUNNING", "PAUSED")):
                            skip.request(minutes)
                            driver.enabled = False
                            driver.reanchor(wall_seconds())
            if hospital is not None and skip.remaining_minutes:
                status_slot.markdown(status_html(hospital, speed, skipping=True), unsafe_allow_html=True)
                indicator = st.progress(1 - skip.remaining_minutes / skip.total_minutes,
                    text=f"Fast Forward {'paused' if skip.paused else 'in progress'}: {format_duration(skip.remaining_minutes)} remaining.")
                if st.button("Cancel Remaining Fast Forward"):
                    skip.cancel()
                elif not skip.paused:
                    # Only this indicator updates during execution, not the tables,
                    # event preview, GA/XAI/LLM or snapshots below this block.
                    last_update = [time.perf_counter()]
                    def progress(fraction):
                        now = time.perf_counter()
                        if fraction == 1 or now - last_update[0] >= .1:
                            indicator.progress(fraction, text=f"Fast Forward: {format_duration(skip.remaining_minutes)} remaining.")
                            last_update[0] = now
                    driver.enabled = False
                    try:
                        skip.fast_forward(progress)
                    finally:
                        driver.reanchor(wall_seconds())
                        driver.enabled = bool(auto and not skip.remaining_minutes)
                if not skip.remaining_minutes:
                    st.rerun()
    with st.container(key="v3_live_overview"):
        if hospital is None:
            st.info("Set physical resources and start the warm-start simulated live feed.")
        else:
            st.subheader("Live Hospital Status")
            st.caption(f"Elapsed {hospital.sim_time_minutes:.2f} simulated minutes ({format_duration(hospital.sim_time_minutes)} elapsed). Capacities stay fixed until reset. Automatic playback {'on' if auto else 'off'}.")
            if driver and driver.pending_minutes > .01:
                st.caption(f"Pending clock catch-up: {format_duration(driver.pending_minutes)}; no events are discarded.")
            initial = hospital.initial_state
            with st.container(border=True):
                st.markdown("**INITIAL HOSPITAL STATE — simulated operating state**")
                resources = initial["resources"]
                st.caption(f"Warm Start: {'Enabled' if initial['enabled'] else 'Disabled'} | {initial['active_patients']} initial active patients | ICU {resources['icu_beds']['occupied_or_busy']} / {resources['icu_beds']['total']} | General {resources['general_beds']['occupied_or_busy']} / {resources['general_beds']['total']} | Staff busy {resources['doctors']['occupied_or_busy']} doctors / {resources['nurses']['occupied_or_busy']} nurses")
                with st.expander("Initial Hospital State Details", expanded=False):
                    render_initial_details(initial)
            st.markdown(overview_html(hospital), unsafe_allow_html=True)
            metrics = hospital.metrics()
            with st.expander("Cumulative and Rolling Live Metrics"):
                st.dataframe(pd.DataFrame([{"Resource": LABELS[kind], **row} for kind, row in hospital.resource_summary().items()]), hide_index=True, width="stretch")
                st.write(f"Arrived: {metrics['total_patients_arrived']} | Active: {metrics['currently_active']} | Completed: {metrics['completed_patients']}")
                st.caption(f"Completed during observation: {metrics['initial_patients_completed']} initial incumbents + {metrics['live_arrivals_completed']} live arrivals. Cumulative wait measures live arrivals only; look-ahead also includes current queued incumbents.")
                st.write(f"Cumulative observed mean wait: {format_duration(metrics['mean_wait_so_far'])} | High-risk mean: {format_duration(metrics['high_risk_mean_wait_so_far'])} | Throughput: {metrics['throughput_per_hour']:.2f}/hour")
                st.caption("Waiting patients contribute elapsed wait only; future waits are not counted. Current cards show instantaneous occupancy, while cumulative/rolling utilization integrates occupied minutes.")
                rolling = metrics["rolling_60_minutes"]
                st.write(f"Last {format_duration(rolling['observed_minutes'])}: {rolling['arrivals']} arrivals, {rolling['completions']} completions, observed mean wait {format_duration(rolling['mean_wait'])}, mean queue {rolling['mean_queue_length']:.2f}.")
                st.dataframe(pd.DataFrame([{"Resource": LABELS[k], "Cumulative utilization": metrics["cumulative_utilization"][k],
                    "Rolling utilization": rolling["utilization"][k]} for k in KINDS]), hide_index=True, width="stretch")
    # Render parents in page order. Read-only technical callbacks are collected
    # while actions render, then populated below in permanent child slots.
    technical_queue = {name: [] for name in ("scenario", "policy", "events", "state")}
    with st.container(key="v3_event_workspace"):
        render_event_controls(hospital, targets, technical_slots=technical_queue)
    technical_region = st.container(key="v3_technical_details")
    with technical_region:
        section_anchor("technical-details")
        st.subheader("Technical Details")
        st.caption("Inspect the verified pressure, search, explanations, ownership and audit evidence.")
        technical_slots = {name: st.container(key="v3_technical_" + name) for name in ("scenario", "policy", "events", "state")}
        for name, callbacks in technical_queue.items():
            with technical_slots[name]:
                for callback in callbacks:
                    callback()
    if hospital is None:
        return
    with technical_slots["state"]:
        with st.expander("Active Patient Details"):
            st.subheader("Live Active Patients")
            limit = st.selectbox("Patient Row Limit", (25, 50, 100, 200), index=1, key="v3_patient_limit")
            include_completed = st.checkbox("Include Recent Completed Patients", key="v3_include_completed",
                help="Fill unused display rows from the capped recent-completion history; cumulative completion metrics retain the full count.")
            rows = hospital.patient_rows(limit, include_completed)
            if rows:
                columns = ["patient_id", "source_event_id", "risk_classification", "risk_probability", "required_bed_type", "assigned_bed_id",
                    "assigned_doctor_id", "assigned_nurse_id", "status", "arrival_time", "waiting_time_minutes",
                    "queue_entry_time", "treatment_start_time", "expected_treatment_completion", "expected_discharge_time", "discharge_time"]
                frame = pd.DataFrame(rows)[columns]
                frame["required_bed_type"] = frame["required_bed_type"].map({"icu_beds": "ICU", "general_beds": "General"})
                for name in ("arrival_time", "queue_entry_time", "treatment_start_time",
                             "expected_treatment_completion", "expected_discharge_time", "discharge_time"):
                    frame[name] = frame[name].map(lambda value: "Before start: " + format_duration(-value) + " earlier" if pd.notna(value) and value < 0 else format_timestamp(value))
                frame["waiting_time_minutes"] = frame["waiting_time_minutes"].map(format_duration)
                frame = frame.rename(columns=dict(patient_id="Patient ID", risk_classification="Risk", risk_probability="RF Probability",
                    required_bed_type="Bed Type", assigned_bed_id="Bed ID", assigned_doctor_id="Doctor ID", assigned_nurse_id="Nurse ID",
                    status="Status", arrival_time="Arrival", waiting_time_minutes="Wait", queue_entry_time="Queue Entry",
                    treatment_start_time="Treatment Start", expected_treatment_completion="Expected Staff Release",
                    expected_discharge_time="Expected Discharge", discharge_time="Discharged"))
                st.dataframe(frame, hide_index=True, width="stretch", column_config={"RF Probability": st.column_config.NumberColumn(format="%.2f")})
                st.caption(f"Showing at most {limit} rows; timestamps use Day N HH:MM:SS (session starts Day 1); Wait is a compact duration. Raw minutes remain in JSON snapshots.")
            else:
                st.info("No active patients at this simulated time.")
        with st.expander("Recent Live Events"):
            audit_rows = [{"patient_id": None, "resource_ids": [], "description": "Read-only prediction audit", **row}
                          for row in st.session_state.get("v3_preview_audit", [])]
            event_rows = sorted(list(hospital.events) + audit_rows,
                                key=lambda row: row["sim_time_minutes"], reverse=True)[:50]
            if event_rows:
                st.dataframe(pd.DataFrame([{**row, "description": format_time_text(row["description"]), "simulated_time": format_time(row["sim_time_minutes"])} for row in event_rows])[
                    ["simulated_time", "event_type", "event_id", "patient_id", "resource_ids", "description"]], hide_index=True, width="stretch")
            st.caption(f"Showing latest 50 live/audit events; live in-memory log is capped at {hospital.events.maxlen}. No per-refresh disk writes.")
        with st.expander("Inspect Live Resources"):
            if st.checkbox("Show Resource Table", key="v3_show_resources"):
                kind = st.selectbox("Resource Type", KINDS, format_func=lambda k: LABELS[k], key="v3_resource_kind")
                status = st.selectbox("Resource Status", ("All", "AVAILABLE", "OCCUPIED" if kind.endswith("beds") else "BUSY", "OUT_OF_SERVICE_PENDING", "OUT_OF_SERVICE"), key=f"v3_filter_{kind}")
                resource_frame = pd.DataFrame(hospital.resource_rows(kind, status, limit=100)).rename(columns={
                    "resource_id": "ID", "kind": "Type", "status": "Status", "patient_id": "Patient"})
                st.dataframe(resource_frame, hide_index=True, width="stretch")
                st.caption("At most 100 resource rows are rendered. Stable IDs and patient ownership are kept in the session engine.")
        with st.expander("Live Operating Policy and Snapshot"):
            st.json(asdict(hospital.current_policy))
            st.caption("Adaptive operational scheduling policy. Physical capacities remain fixed. Reserve and surge weights affect only future queue/allocation decisions; this is not validated clinical triage.")
            if st.button("Prepare Compact Live Snapshot"):
                st.session_state.v3_snapshot_download = json.dumps(hospital.snapshot(patient_limit=25, event_limit=50), indent=2, allow_nan=False)
            if st.session_state.get("v3_snapshot_download"):
                st.caption("Prepared snapshot is a historical export; prepare again to capture current state.")
                st.download_button("Download Live State Snapshot", st.session_state.v3_snapshot_download,
                    file_name="live_hospital_snapshot.json", mime="application/json")
        st.caption("Simulated operational decision-support prototype. Not clinical advice or production EHR integration.")

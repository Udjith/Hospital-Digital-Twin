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

LABELS = dict(icu_beds="ICU Beds", general_beds="General Beds", doctors="Doctors", nurses="Nurses")


def format_time(minutes):
    return format_timestamp(minutes)


def get_live_hospital(state):
    """Read-only lookup: widget reruns must not construct or reset a session."""
    return state.get("v3_live_hospital")


@st.fragment(run_every="1s")
def render_live_twin(data_path, model_path, targets=None):
    st.header("V3 Live Twin")
    st.caption("Simulated live hospital feed, not an EHR integration. Fixed physical capacities are separate from the V2 sidebar and GA recommendation. Operational queue priority is not a clinical triage protocol.")
    hospital = get_live_hospital(st.session_state)
    locked = hospital is not None and hospital.status != "READY"
    with st.container(border=True):
        st.subheader("Fixed Live Hospital Capacity")
        capacity_values = {}
        for column, kind in zip(st.columns(4), KINDS):
            with column:
                capacity_values[kind] = st.number_input(f"Live {LABELS[kind]}", 1, 10000,
                    DEMO_CAPACITY[kind], key=f"v3_capacity_{kind}", disabled=locked,
                    help="Fixed physical capacity for this live session. Reset Live Twin before editing; V2 optimization cannot change it.")
        feed_col, seed_col, speed_col = st.columns(3)
        with feed_col:
            rate = st.number_input("Live Base Arrival Rate (patients/hour)", 0., 1000., DEMO_ARRIVAL_RATE,
                key="v3_arrival_rate", disabled=locked,
                help="Poisson live arrival feed with exponential inter-arrival minutes of mean 60/rate. Reset before changing the feed.")
        with seed_col:
            seed = st.number_input("Live Random Seed", 0, 2**32 - 1, 42, key="v3_seed", disabled=locked,
                help="Same live configuration and seed reproduce arrivals and clinical profile sampling. Synthetic patient IDs are used.")
        with speed_col:
            speed = st.selectbox("Simulation Speed", SIMULATION_SPEEDS, index=2,
                format_func=lambda x: f"{x}x", key="v3_speed",
                help="Automatic playback converts elapsed monotonic wall seconds to simulated minutes at this multiplier. Manual advances use the chosen minute step.")
        auto = st.toggle("Automatic Live Playback", key="v3_autoplay",
            help="Advance using elapsed wall time, with bounded catch-up steps. Turn off to use manual stepping. No infinite loop or background simulation thread.")
        # Accrue elapsed time exactly once, before processing control transitions.
        driver = st.session_state.get("v3_clock_driver")
        if driver is not None:
            skip = st.session_state.get("v3_skip_driver")
            skipping = bool(skip and skip.remaining_minutes)
            driver.tick(time.monotonic(), speed=speed, enabled=auto and not skipping)
            if skipping:
                skip.tick()
                driver.reanchor(time.monotonic())
                if not skip.remaining_minutes:
                    driver.enabled = bool(auto)
        controls = st.columns(5)
        with controls[0]:
            start = st.button("Start Live Simulation", disabled=hospital is not None and hospital.status != "READY", width="stretch")
        with controls[1]:
            pause = st.button("Pause", disabled=hospital is None or hospital.status != "RUNNING", width="stretch")
        with controls[2]:
            resume = st.button("Resume", disabled=hospital is None or hospital.status != "PAUSED", width="stretch")
        with controls[3]:
            stop = st.button("Stop Live Simulation", disabled=hospital is None or hospital.status not in ("RUNNING", "PAUSED"), width="stretch")
        with controls[4]:
            reset = st.button("Reset Live Twin", disabled=hospital is None, width="stretch")
        if start:
            try:
                with st.spinner("Loading cached RF profiles and preparing live state..."):
                    # Cache is owned by the validated V2 profile loader; inference
                    # is performed once per source/model version, never per arrival.
                    profiles = load_patient_profiles(data_path, model_path)
                    hospital = LiveHospitalState(profiles, LiveCapacity(**capacity_values), float(rate), int(seed),
                        operating_policy=OperatingPolicy(**NORMAL_POLICY_VALUES))
                hospital.start()
                driver = LiveClockDriver(hospital)
                driver.tick(time.monotonic(), speed=speed, enabled=auto)
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
                driver.reanchor(time.monotonic())
            if stop:
                hospital.stop()
                if st.session_state.get("v3_skip_driver"):
                    st.session_state.v3_skip_driver.cancel()
            if reset:
                hospital.reset()
                for key in ("v3_proposed_event", "v3_lookahead_result", "v3_preview_audit", "v3_proposal_form", "v3_live_ga_result", "v3_live_xai", "v3_live_ai", "v3_policy_recommendations", "v3_manual_policy_reference", "v3_scenario_interpretation", "v3_scenario_preview", "v3_scenario_audit", "v3_applied_scenario_events", "v3_modify_retry", "v3_recommendation_decision"):
                    st.session_state.pop(key, None)
                st.session_state.v3_clock_driver = LiveClockDriver(hospital)
                st.session_state.v3_skip_driver = LiveSkipDriver(hospital)
                st.session_state.pop("v3_snapshot_download", None)
            if pause or resume or stop or reset:
                st.rerun()
        step_col, action_col = st.columns([2, 1])
        with step_col:
            delta = st.number_input("Manual Advance (simulated minutes)", 0.1, MAX_MANUAL_MINUTES, 5., key="v3_manual_delta",
                help="Explicit simulated minutes, up to 30 days. Large requests become cancellable skip debt, processed in 60-minute chunks without dropping events.")
        with action_col:
            if st.button("Advance Simulation", disabled=hospital is None or hospital.status not in ("RUNNING", "PAUSED"), width="stretch"):
                skip = st.session_state.setdefault("v3_skip_driver", LiveSkipDriver(hospital))
                try:
                    skip.request(float(delta))
                    skip.tick()
                    driver.enabled = bool(auto and not skip.remaining_minutes)
                    driver.reanchor(time.monotonic())
                except ValueError as exc:
                    st.error(str(exc))
        if hospital is not None:
            skip = st.session_state.setdefault("v3_skip_driver", LiveSkipDriver(hospital))
            with st.expander("Manual Skip Presets"):
                presets = (("+5 min", 5), ("+30 min", 30), ("+1 hour", 60), ("+6 hours", 360),
                    ("+12 hours", 720), ("+1 day", 1440), ("+3 days", 4320), ("+7 days", 10080))
                for col, (label, minutes) in zip(st.columns(8), presets):
                    if col.button(label, disabled=bool(skip.remaining_minutes) or hospital.status not in ("RUNNING", "PAUSED")):
                        skip.request(minutes)
                        skip.tick()
                        driver.enabled = bool(auto and not skip.remaining_minutes)
                        driver.reanchor(time.monotonic())
            if skip.remaining_minutes:
                st.progress(1 - skip.remaining_minutes / skip.total_minutes,
                    text=f"Manual skip {'paused' if skip.paused else 'in progress'}: {format_duration(skip.remaining_minutes)} remaining; at most 60 minutes per refresh.")
                if st.button("Cancel Remaining Manual Skip"):
                    skip.cancel()
                    driver.enabled = bool(auto)
                    driver.reanchor(time.monotonic())
    if hospital is None:
        st.info("Set physical resources and start the simulated live feed. V2 scenario analysis and optimization remain available in their existing tabs.")
        return
    st.subheader(f"{format_time(hospital.sim_time_minutes)} | {hospital.status}")
    st.caption(f"Elapsed {hospital.sim_time_minutes:.2f} simulated minutes ({format_duration(hospital.sim_time_minutes)} elapsed). Capacities stay fixed until reset. Automatic playback {'on' if auto else 'off'}.")
    if driver and driver.pending_minutes > .01:
        st.caption(f"Pending clock catch-up: {format_duration(driver.pending_minutes)}; no events are discarded.")
    summary = hospital.resource_summary()
    for col, kind in zip(st.columns(4), KINDS):
        resource = summary[kind]
        verb = "occupied" if kind.endswith("beds") else "busy"
        col.metric(f"Live {LABELS[kind]} {verb}", f"{resource['occupied_or_busy']} / {resource['total']}")
        col.caption(f"{resource['available']} free | {resource['current_utilization']:.1%} of usable capacity")
        col.caption(f"Total {resource['total']} | Usable {resource['usable']} | Out {resource['temporarily_unavailable']} | Pending {resource['pending_unavailable']}")
    metrics = hospital.metrics()
    for col, (label, key) in zip(st.columns(4), (("Waiting Patients", "currently_waiting"),
            ("In Treatment", "in_treatment"), ("Bed-Stay Patients", "bed_stay"), ("Completed During Session", "completed_patients"))):
        col.metric(label, metrics[key])
    with st.expander("Cumulative and Rolling Live Metrics"):
        st.write(f"Arrived: {metrics['total_patients_arrived']} | Active: {metrics['currently_active']} | Completed: {metrics['completed_patients']}")
        st.write(f"Cumulative observed mean wait: {format_duration(metrics['mean_wait_so_far'])} | High-risk mean: {format_duration(metrics['high_risk_mean_wait_so_far'])} | Throughput: {metrics['throughput_per_hour']:.2f}/hour")
        st.caption("Waiting patients contribute elapsed wait only; future waits are not counted. Current cards show instantaneous occupancy, while cumulative/rolling utilization integrates occupied minutes.")
        rolling = metrics["rolling_60_minutes"]
        st.write(f"Last {format_duration(rolling['observed_minutes'])}: {rolling['arrivals']} arrivals, {rolling['completions']} completions, observed mean wait {format_duration(rolling['mean_wait'])}, mean queue {rolling['mean_queue_length']:.2f}.")
        st.dataframe(pd.DataFrame([{"Resource": LABELS[k], "Cumulative utilization": metrics["cumulative_utilization"][k],
            "Rolling utilization": rolling["utilization"][k]} for k in KINDS]), hide_index=True, width="stretch")
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
            frame[name] = frame[name].map(format_timestamp)
        frame["waiting_time_minutes"] = frame["waiting_time_minutes"].map(format_duration)
        frame = frame.rename(columns=dict(patient_id="Patient ID", risk_classification="Risk", risk_probability="RF Probability",
            required_bed_type="Bed Type", assigned_bed_id="Bed ID", assigned_doctor_id="Doctor ID", assigned_nurse_id="Nurse ID",
            status="Status", arrival_time="Arrival", waiting_time_minutes="Wait", queue_entry_time="Queue Entry",
            treatment_start_time="Treatment Start", expected_treatment_completion="Expected Staff Release",
            expected_discharge_time="Expected Discharge", discharge_time="Discharged"))
        st.dataframe(frame, hide_index=True, width="stretch")
        st.caption(f"Showing at most {limit} rows; timestamps use Day N HH:MM:SS (session starts Day 1); Wait is a compact duration. Raw minutes remain in JSON snapshots.")
    else:
        st.info("No active patients at this simulated time.")
    render_event_controls(hospital, targets)
    st.subheader("Recent Live Events")
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

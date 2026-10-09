"""Read-only presentation helpers for the final live operations dashboard."""
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st

from live_time_display import format_duration, format_timestamp
from live_canonical import canonical_hash

SECTIONS = (("live", "Live"), ("stress-event", "Stress Event"), ("scenario", "Scenario"),
            ("policy", "Policy"), ("active-events", "Active Events"), ("technical-details", "Technical Details"))


def section_anchor(name):
    """Plain in-page targets: no callbacks, script reruns or simulation actions."""
    st.markdown(f'<div id="{e(name)}" class="section-anchor"></div>', unsafe_allow_html=True)


def sidebar_navigation_html():
    return '<div class="nav-title">LIVE DIGITAL TWIN</div><nav class="section-nav" aria-label="Dashboard sections">' + ''.join(
        f'<a href="#{name}" target="_self">{label}</a>' for name, label in SECTIONS) + '</nav>'


def render_technical_detail(queue, region, callback):
    """Defer read-only evidence until the lower, permanent section is rendered.

    Never backfill root containers around live widgets: native fragment updates
    must receive their parent blocks in the same order as the displayed page.
    """
    if queue is not None:
        queue.append(callback)
    else:
        with region:
            callback()


def initial_state_html(initial):
    settings = initial["settings"]
    configuration = [("Enabled", "Yes" if initial["enabled"] else "No"), ("Seed", initial["seed"])]
    for key, label in (("icu_occupancy_target", "ICU occupancy target"), ("general_occupancy_target", "General occupancy target"),
                       ("doctor_busy_target", "Doctor busy target"), ("nurse_busy_target", "Nurse busy target")):
        configuration.append((label, f'{settings[key]:.0%}'))
    configuration.append(("Initial queue", "Auto" if settings["initial_queue"] is None else settings["initial_queue"]))
    patient_state = [("Active patients", initial["active_patients"]), ("Waiting", initial["waiting_patients"]),
                     ("In treatment", initial["in_treatment"]), ("Bed stay", initial["bed_stay"])]
    resources = []
    for kind, label in (("icu_beds", "ICU Beds"), ("general_beds", "General Beds"), ("doctors", "Doctors"), ("nurses", "Nurses")):
        row = initial["resources"][kind]
        pairs = [("Total", row["total"]), ("Occupied" if kind.endswith("beds") else "Busy", row["occupied_or_busy"]),
                 ("Available", row["available"]), ("Usable", row["usable"]), ("Out of service", row["temporarily_unavailable"]),
                 ("Pending outage", row["pending_unavailable"]), ("Planned usable", row["planned_usable"]),
                 ("Actual utilization", f'{row["current_utilization"]:.1%}')]
        resources.append(card_group(label, pairs))
    return ('<div class="initial-config-grid">' + card_group("Warm Start Configuration", configuration) +
            card_group("Initial Patient State", patient_state) + '</div><div class="eyebrow initial-resource-heading">Initial Resource State</div><div class="initial-resource-grid">' + ''.join(resources) + '</div>')


def render_initial_details(initial):
    st.markdown(initial_state_html(initial), unsafe_allow_html=True)
    st.markdown("**Notes / Warnings**")
    st.caption("Simulated initialization, not measured hospital occupancy. Live arrivals are counted separately from initial incumbents. Each treatment uses one doctor and one nurse; paired availability can prevent both staff targets from being reached.")
    for warning in initial["warnings"]:
        st.info(warning)
    if not initial["warnings"]:
        st.caption("No initialization warnings.")
    with st.expander("Advanced Initialization Provenance", expanded=False):
        st.write("Seed strategy: " + initial["seed_strategy"])
        st.write("Profile checksum: " + initial["profile_sha256"])
        st.write("Canonical initialization hash: " + canonical_hash(initial))
        st.caption("Hash describes the retained initialization evidence, not the current evolving hospital state. Targets and actual initial occupancy are shown above.")
    with st.expander("Advanced Raw Initialization Data", expanded=False):
        st.json(initial)


@lru_cache(maxsize=1)
def operations_css():
    return (Path(__file__).parent / "assets/operations.css").read_text(encoding="utf-8")


def apply_operations_theme():
    st.markdown("<style>" + operations_css() + "</style>", unsafe_allow_html=True)


def e(value):
    return escape(str(value), quote=True)


def verdict_class(verdict):
    return {"HANDLED": "handled", "AT RISK": "at-risk", "CANNOT HANDLE": "cannot-handle"}.get(verdict, "neutral")


def card_group(title, pairs):
    return '<div class="overview-card"><h4>' + e(title) + '</h4>' + ''.join(
        '<div class="kv"><span>' + e(label) + '</span><strong>' + e(value) + '</strong></div>'
        for label, value in pairs) + '</div>'


def status_html(hospital, speed, skipping=False):
    if hospital is None:
        return '<div class="status-card"><div class="eyebrow">Hospital operations · simulated live feed</div><div class="status-title">Hospital Digital Twin</div><p>Start a warm-start hospital to monitor live operations and test stress scenarios.</p></div>'
    resources, metrics = hospital.resource_summary(), hospital.metrics()
    label, color = ("FAST FORWARDING", "working") if skipping else ("LIVE", "handled") if hospital.status == "RUNNING" else (hospital.status, "neutral")
    cells = []
    for kind, name in (("icu_beds", "ICU"), ("general_beds", "General"), ("doctors", "Doctors"), ("nurses", "Nurses")):
        r = resources[kind]
        load = r["current_utilization"]
        cells.append(f'<div class="resource-cell"><div class="resource-label">{name}</div><div class="resource-value">{r["occupied_or_busy"]} / {r["total"]}</div><div class="load-track"><span class="load-fill" style="width:{min(100, max(0, load * 100)):.2f}%"></span></div><div class="resource-sub">{load:.0%} of usable capacity · {r["available"]} available</div></div>')
    return (f'<div class="status-card"><div class="status-header"><div><div class="eyebrow">Live hospital operations</div><div class="status-title">Hospital Digital Twin</div></div><div><div class="clock">{e(format_timestamp(hospital.sim_time_minutes))}</div><div class="clock-meta"><span class="chip live {color}">● {e(label)}</span> &nbsp; {speed}x &nbsp; | {e(hospital.status)}</div></div></div><div class="status-resources">' + ''.join(cells) +
        f'</div><div class="status-footer"><span>Active <strong>{metrics["currently_active"]}</strong></span><span>Waiting <strong>{metrics["currently_waiting"]}</strong></span><span>Arrival rate <strong>{hospital.effective_arrival_rate:g}/h</strong></span></div></div>')


def expanded_status_html(hospital):
    m = hospital.metrics()
    return '<div class="detail-grid">' + ''.join((
        card_group("Patient Flow", [("In treatment", m["in_treatment"]), ("Bed stay", m["bed_stay"]), ("Live arrivals", m["total_patients_arrived"]), ("Initial incumbents remaining", m["initial_patients_remaining"])]),
        card_group("Performance", [("Completed", m["completed_patients"]), ("Mean wait", format_duration(m["mean_wait_so_far"])), ("High-risk wait", format_duration(m["high_risk_mean_wait_so_far"])), ("Throughput", f'{m["throughput_per_hour"]:.2f}/h')]),
        card_group("Operating State", [("Active stress events", sum(event.status == "ACTIVE" for event in hospital.stress_events.values())), ("Policy mode", "Temporary" if hospital._temporary_policy else "Normal / persistent"), ("Warm Start", "Enabled" if hospital.initial_state["enabled"] else "Disabled"), ("Waiting patients", m["currently_waiting"])]))) + '</div>'


def overview_html(hospital):
    m, r = hospital.metrics(), hospital.resource_summary()
    return '<div class="overview-grid">' + ''.join((
        card_group("Patient Flow", [("In treatment", m["in_treatment"]), ("Bed stay", m["bed_stay"])]),
        card_group("Resource Load", [("ICU / General", f'{r["icu_beds"]["current_utilization"]:.0%} / {r["general_beds"]["current_utilization"]:.0%}'), ("Doctors / Nurses", f'{r["doctors"]["current_utilization"]:.0%} / {r["nurses"]["current_utilization"]:.0%}')]),
        card_group("Operations", [("Completions", m["completed_patients"]), ("Throughput", f'{m["throughput_per_hour"]:.2f}/h')]),
        card_group("Current Situation", [("Active stress events", sum(event.status == "ACTIVE" for event in hospital.stress_events.values())), ("Policy mode", "Temporary event policy" if hospital._temporary_policy else "Normal / persistent")])) ) + '</div>'


def forecast_html(result, diagnosis, targets):
    a = result["aggregate"]
    primary = diagnosis["primary_limiting_factor"]
    reason = primary["label"] if primary else "Configured operational targets met."
    rows = [("Mean wait", format_duration(a["mean_wait"]), "≤ " + format_duration(targets.mean_wait_target_minutes), a["mean_wait"] <= targets.mean_wait_target_minutes),
            ("High-risk wait", format_duration(a["high_risk_mean_wait"]), "≤ " + format_duration(targets.high_risk_wait_target_minutes), a["high_risk_mean_wait"] <= targets.high_risk_wait_target_minutes)]
    pressure = a["resource_pressure"]["doctor"]["time_weighted_utilization"]
    rows.append(("Doctor pressure · mean", f"{pressure:.1%}", f"≤ {targets.max_utilization_target:.0%}", pressure <= targets.max_utilization_target))
    recovered = sum(run["condition_breakdown"]["recovery_pass"] for run in result["runs"])
    rows.append(("Queue recovery", f"{recovered} / {len(result['runs'])} runs", "End queue + event waiting", recovered == len(result["runs"])))
    table = '<div class="comparison-rows"><div class="comparison-row header"><span>Metric</span><span>Observed · average</span><span>Target / reference</span><span></span></div>'
    for label, observed, target, passed in rows:
        table += f'<div class="comparison-row"><span>{e(label)}</span><span>{e(observed)}</span><span>{e(target)}</span><span class="condition-{"pass" if passed else "fail"}">{"✓" if passed else "!"}</span></div>'
    return f'<div class="forecast-card {verdict_class(result["verdict"])}"><div class="eyebrow">Look-Ahead Forecast</div><div class="verdict-title">{e(result["verdict"])}</div><div class="robustness-value">Predictive Robustness: {result["robustness"]:.1f}%</div><div class="forecast-reason">Main limiting factor: <strong>{e(reason)}</strong></div>{table}</div></div>'


def policy_comparison_html(result):
    before, after = result["current_verified"], result["optimized_verified"]
    def pane(title, prediction):
        a = prediction["aggregate"]
        return f'<div class="policy-pane"><div class="eyebrow">{title}</div><span class="chip {verdict_class(prediction["verdict"])}">{e(prediction["verdict"])}</span><div class="robustness-value">{prediction["robustness"]:.1f}%</div><div class="kv"><span>Mean wait</span><strong>{e(format_duration(a["mean_wait"]))}</strong></div><div class="kv"><span>High-risk wait</span><strong>{e(format_duration(a["high_risk_mean_wait"]))}</strong></div></div>'
    changes = []
    for gene, value in result["current_policy"].items():
        updated = result["optimized_policy"][gene]
        if value != updated:
            old, new = (f"{value:.0%}", f"{updated:.0%}") if "reserve" in gene else (f"{value:.1f}", f"{updated:.1f}")
            changes.append(card_group(gene.replace("_", " ").title(), [("Recommended", f"{old} → {new}")]))
    return f'<div class="dashboard-card"><span class="chip {verdict_class(after["verdict"])}">{e(result["outcome"])}</span><div class="policy-comparison">{pane("CURRENT", before)}<div class="comparison-arrow">→</div>{pane("OPTIMIZED", after)}</div><div class="eyebrow">Recommended policy changes</div><div class="policy-changes">' + (''.join(changes) if changes else '<p>Current policy retained.</p>') + '</div><div class="resource-sub">Physical resources unchanged · all six genes available in Policy Details</div></div>'


def explanation_html(explanation):
    provider = "Groq explanation" if explanation["provider"].startswith("Groq") else "Explanation (deterministic fallback)"
    return '<div class="explanation-card"><div class="eyebrow">AI Explanation</div><p>' + e(explanation["summary"]) + '</p><div class="provider-meta">' + e(provider) + '</div></div>'

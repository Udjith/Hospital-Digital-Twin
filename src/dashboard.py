
from __future__ import annotations

import html
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from digital_twin import (
    BASELINE_MAX_PATIENTS,
    HOSPITAL_SCALE_MULTIPLIER,
    SIMULATOR_VERSION,
    derive_ga_bounds,
    validate_baseline_artifact,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_SYNTHETIC = PROJECT_ROOT / "data" / "synthetic" / "synthetic_hospital.csv"
RF_MODEL = PROJECT_ROOT / "models" / "random_forest_pipeline.joblib"

RF_DIR = PROJECT_ROOT / "results" / "random_forest"
DT_DIR = PROJECT_ROOT / "results" / "digital_twin"
GA_DIR = PROJECT_ROOT / "results" / "genetic_algorithm"
XAI_DIR = PROJECT_ROOT / "results" / "decision_tree_xai"
LLM_DIR = PROJECT_ROOT / "results" / "llm"
FEEDBACK_DIR = PROJECT_ROOT / "results" / "feedback"

FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def load_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def load_csv(path: Path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_text(path: Path, default=""):
    if not path.exists():
        return default
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return default


def run_python(script_name: str, args: list[str] | None = None):
    args = args or []
    script_path = SRC_DIR / script_name

    if not script_path.exists():
        return False, f"Missing script: {script_path}"

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, str(script_path), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()

    if result.returncode != 0:
        return False, f"{output}\n{error}".strip()

    return True, output


def fmt(value, decimals=2, suffix=""):
    try:
        number = float(value)
        if pd.isna(number):
            return "-"
        return f"{number:.{decimals}f}{suffix}"
    except Exception:
        return "-"


def fmt_int(value):
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return "-"


def fmt_pct(value, decimals=1):
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except Exception:
        return "-"


def fmt_minutes(value):
    return fmt(value, 2, " min")


def fmt_throughput(value):
    return fmt(value, 2, "/day")


def fmt_fitness(value):
    return fmt(value, 2)


def fmt_probability(value):
    return fmt(value, 2)


def delta_number(new_value, old_value, decimals=2, suffix=""):
    try:
        change = float(new_value) - float(old_value)
        return f"{change:+.{decimals}f}{suffix}"
    except Exception:
        return None


def ga_result_matches_baseline(result: dict, baseline: dict) -> bool:
    """Reject recommendations produced for a different canonical baseline."""
    if not result or not baseline:
        return False
    provenance = result.get("baseline_provenance", {})
    metadata = baseline.get("baseline_metadata", {})
    expected_policy = {
        key: int(baseline.get("policy", {}).get(key, -1))
        for key in ["icu_beds", "general_beds", "doctors", "nurses"]
    }
    try:
        scale_matches = float(provenance.get("hospital_scale_multiplier")) == float(
            metadata.get("hospital_scale_multiplier")
        )
    except (TypeError, ValueError):
        scale_matches = False
    return all(
        [
            provenance.get("simulator_version") == SIMULATOR_VERSION,
            provenance.get("baseline_generated_at_utc")
            == metadata.get("generated_at_utc"),
            provenance.get("baseline_policy") == expected_policy,
            scale_matches,
        ]
    )


def build_manual_interpretation(
    baseline_bounds: dict,
    max_icu: int,
    max_general: int,
    max_doctors: int,
    max_nurses: int,
    mean_wait_target: float,
    high_risk_wait_target: float,
    prioritize_high_risk: bool,
    prioritize_efficiency: bool,
):
    return {
        "apply_feedback": True,
        "summary": "Structured dashboard controls supplied by the clinician/administrator.",
        "constraints": {
            "icu_beds": {
                "min": baseline_bounds["icu_beds"][0],
                "max": int(max_icu),
            },
            "general_beds": {
                "min": baseline_bounds["general_beds"][0],
                "max": int(max_general),
            },
            "doctors": {
                "min": baseline_bounds["doctors"][0],
                "max": int(max_doctors),
            },
            "nurses": {
                "min": baseline_bounds["nurses"][0],
                "max": int(max_nurses),
            },
        },
        "objectives": {
            "acceptable_mean_wait_min": float(mean_wait_target),
            "acceptable_high_risk_wait_min": float(high_risk_wait_target),
            "mean_wait_weight": 1.0,
            "high_risk_wait_weight": 3.0 if prioritize_high_risk else 2.0,
            "utilization_weight_scale": 1.0,
            "resource_cost_weight_scale": 1.35 if prioritize_efficiency else 1.0,
            "throughput_reward_weight": 0.10,
        },
    }


def merge_llm_feedback(base: dict, llm_data: dict):
    """
    Sidebar values are hard boundaries.
    Natural-language feedback may tighten them, but never loosen them.
    """
    merged = json.loads(json.dumps(base))

    if not isinstance(llm_data, dict) or not llm_data.get("apply_feedback", False):
        return merged

    llm_constraints = llm_data.get("constraints", {}) or {}

    for resource, base_range in merged["constraints"].items():
        incoming = llm_constraints.get(resource, {}) or {}

        incoming_min = incoming.get("min")
        incoming_max = incoming.get("max")

        if incoming_min is not None:
            base_range["min"] = max(
                int(base_range["min"]),
                int(incoming_min),
            )

        if incoming_max is not None:
            base_range["max"] = min(
                int(base_range["max"]),
                int(incoming_max),
            )

        if base_range["min"] > base_range["max"]:
            raise ValueError(
                f"Feedback creates an invalid {resource} range: "
                f"{base_range['min']} to {base_range['max']}."
            )

    llm_obj = llm_data.get("objectives", {}) or {}
    for key, value in llm_obj.items():
        if value is not None and key in merged["objectives"]:
            merged["objectives"][key] = value

    summary = llm_data.get("summary")
    if summary:
        merged["summary"] = (
            base["summary"] + " Natural-language feedback: " + summary
        )

    return merged


def monochrome_bar_chart(df, category, value, title, dark=False):
    color = "#82c0ea"
    chart = (
        alt.Chart(df)
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
        )
        .encode(
            x=alt.X(f"{category}:N", title=None, sort=None),
            y=alt.Y(f"{value}:Q", title=None),
            tooltip=[
                alt.Tooltip(f"{category}:N"),
                alt.Tooltip(f"{value}:Q", format=".2f"),
            ],
            color=alt.value(color),
        )
        .properties(height=260, title=title)
    )
    return style_chart(chart, dark)


def monochrome_line_chart(df, x, y, title, dark=False):
    color = "#82c0ea"
    chart = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X(f"{x}:Q", title="Generation"),
            y=alt.Y(f"{y}:Q", title="Fitness"),
            tooltip=[
                alt.Tooltip(f"{x}:Q", format=".0f"),
                alt.Tooltip(f"{y}:Q", format=".2f"),
            ],
            color=alt.value(color),
        )
        .properties(height=280, title=title)
    )
    return style_chart(chart, dark)


def style_chart(chart, dark=False):
    """Apply readable monochrome text and grid colors inside Altair SVGs."""
    text_color = "#f1f2f4" if dark else "#202328"
    muted_color = "#b8bbc1" if dark else "#5f646b"
    grid_color = "#343941" if dark else "#d8dbe0"
    return (
        chart.configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor=muted_color,
            titleColor=text_color,
            gridColor=grid_color,
            domainColor=grid_color,
            tickColor=grid_color,
        )
        .configure_title(color=text_color, fontSize=15, anchor="start")
        .configure_legend(
            labelColor=muted_color,
            titleColor=text_color,
        )
    )


def build_change_notes(
    baseline_policy: dict,
    recommendation: dict,
    baseline_metrics: dict,
    optimized_metrics: dict,
    interpreted: dict,
):
    notes = []

    resource_names = {
        "icu_beds": "ICU beds",
        "general_beds": "general beds",
        "doctors": "modeled concurrent doctor capacity",
        "nurses": "modeled concurrent nurse capacity",
    }
    constraints = interpreted.get("constraints", {}) if interpreted else {}

    for key, label in resource_names.items():
        if key in recommendation and key in baseline_policy:
            old = int(baseline_policy[key])
            new = int(recommendation[key])
            if old != new:
                direction = "increased" if new > old else "reduced"
                limit = constraints.get(key, {}) or {}
                range_text = ""
                if limit.get("min") is not None and limit.get("max") is not None:
                    range_text = (
                        f" within the applied range {int(limit['min'])}-"
                        f"{int(limit['max'])}"
                    )
                notes.append(
                    f"Under the applied constraints, the optimizer selected {new} "
                    f"{label}, {direction} from the baseline value of {old}{range_text}."
                )

    if baseline_metrics and optimized_metrics:
        old_wait = baseline_metrics.get("mean_waiting_time_min")
        new_wait = optimized_metrics.get("mean_waiting_time_min")
        try:
            if old_wait is not None and new_wait is not None:
                change = float(new_wait) - float(old_wait)
                direction = "decreased" if change < 0 else "increased"
                notes.append(
                    f"Mean simulated waiting time {direction} from "
                    f"{float(old_wait):.2f} to {float(new_wait):.2f} minutes "
                    f"({change:+.2f} minutes)."
                )
        except Exception:
            pass

    summary = interpreted.get("summary") if interpreted else None
    if summary:
        notes.append(f"Applied feedback: {summary}")

    return notes[:7]


def build_ai_summary(
    recommendation: dict,
    optimized_metrics: dict,
    xai_result: dict,
    interpreted: dict,
    baseline_bounds: dict,
):
    """Build a concise current-run summary without another LLM request."""
    bullets = [
        (
            "Recommended policy: "
            f"{fmt_int(recommendation.get('icu_beds'))} ICU beds, "
            f"{fmt_int(recommendation.get('general_beds'))} general beds, "
            f"{fmt_int(recommendation.get('doctors'))} modeled concurrent doctors, "
            f"and {fmt_int(recommendation.get('nurses'))} modeled concurrent nurses."
        ),
        (
            "Expected performance: mean wait "
            f"{fmt_minutes(optimized_metrics.get('mean_waiting_time_min'))}; "
            "high-risk mean wait "
            f"{fmt_minutes(optimized_metrics.get('high_risk_mean_waiting_time_min'))}; "
            f"P95 wait {fmt_minutes(optimized_metrics.get('p95_waiting_time_min'))}; "
            "throughput "
            f"{fmt_throughput(optimized_metrics.get('throughput_patients_per_day'))}."
        ),
    ]

    classification = str(
        xai_result.get("decision_tree_class", "unknown")
    ).upper()
    rules = xai_result.get("decision_path_rules", []) or []
    if rules:
        decisive_rule = " ".join(str(rules[-1]).split())
        if len(decisive_rule) > 120:
            decisive_rule = decisive_rule[:117].rstrip() + "..."
        bullets.append(
            f"XAI check: {classification}; final active rule: {decisive_rule}."
        )
    else:
        bullets.append(
            "XAI check: no current decision-tree path is available for this run."
        )

    constraints = interpreted.get("constraints", {}) if interpreted else {}
    resource_labels = {
        "icu_beds": "ICU beds",
        "general_beds": "general beds",
        "doctors": "concurrent doctors",
        "nurses": "concurrent nurses",
    }
    applied_ranges = []
    for key, label in resource_labels.items():
        limits = constraints.get(key, {}) or {}
        low = limits.get("min")
        high = limits.get("max")
        if low is None or high is None or key not in baseline_bounds:
            continue
        base_low, base_high = baseline_bounds[key]
        if int(low) > int(base_low) or int(high) < int(base_high):
            applied_ranges.append(f"{label} {fmt_int(low)}-{fmt_int(high)}")

    objectives = interpreted.get("objectives", {}) if interpreted else {}
    high_risk_weight = objectives.get("high_risk_wait_weight")
    constraint_details = applied_ranges[:2]
    try:
        if high_risk_weight is not None and float(high_risk_weight) > 2.0:
            constraint_details.append(
                f"high-risk waiting weight {fmt(high_risk_weight, 2)}"
            )
    except (TypeError, ValueError):
        pass

    if constraint_details:
        bullets.append("Applied constraint: " + "; ".join(constraint_details) + ".")

    bullets.append(
        "Limitation: estimates depend on the modeled cohort, arrival pattern, "
        "resource durations, and simulation assumptions."
    )
    return bullets[:5]


# ============================================================
# Streamlit setup
# ============================================================

st.set_page_config(
    page_title="Healthcare Digital Twin",
    page_icon=":material/local_hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Baseline data + bounds
# ============================================================

baseline_metrics = load_json(DT_DIR / "baseline_metrics.json")
baseline_errors = (
    validate_baseline_artifact(
        baseline_metrics,
        DATA_SYNTHETIC,
        RF_MODEL,
    )
    if baseline_metrics
    else ["baseline result is missing or unreadable"]
)
baseline_valid = not baseline_errors
baseline_policy = (
    {
        key: int(baseline_metrics["policy"][key])
        for key in ["icu_beds", "general_beds", "doctors", "nurses"]
    }
    if baseline_valid
    else {
        "icu_beds": 1,
        "general_beds": 1,
        "doctors": 1,
        "nurses": 1,
    }
)
baseline_metadata = (
    baseline_metrics.get("baseline_metadata", {})
    if baseline_valid
    else {}
)
if not baseline_valid:
    baseline_metrics = {}
synthetic_df = load_csv(DATA_SYNTHETIC)
rf_metrics = load_json(RF_DIR / "metrics.json")
rf_importance = load_csv(RF_DIR / "feature_importance.csv")

# The GA and sidebar share the same baseline-relative search envelope.
BASE_BOUNDS = derive_ga_bounds(baseline_policy)


# ============================================================
# Session state + synchronized controls
# ============================================================

CONTROL_DEFAULTS = {
    "icu_beds": BASE_BOUNDS["icu_beds"][1],
    "general_beds": BASE_BOUNDS["general_beds"][1],
    "doctors": BASE_BOUNDS["doctors"][1],
    "nurses": BASE_BOUNDS["nurses"][1],
    "mean_wait": 20.0,
    "high_risk_wait": 10.0,
}

for control, default in CONTROL_DEFAULTS.items():
    st.session_state.setdefault(f"{control}_slider", default)
    st.session_state.setdefault(f"{control}_number", default)

for resource in ["icu_beds", "general_beds", "doctors", "nurses"]:
    low, high = BASE_BOUNDS[resource]
    current = st.session_state.get(
        f"{resource}_number",
        baseline_policy[resource],
    )
    bounded = max(low, min(high, int(current)))
    st.session_state[f"{resource}_slider"] = bounded
    st.session_state[f"{resource}_number"] = bounded

st.session_state.setdefault("prioritize_high_risk", True)
st.session_state.setdefault("prioritize_efficiency", True)
st.session_state.setdefault("natural_feedback", "")
st.session_state.setdefault("optimization_run", False)
st.session_state.setdefault("last_run_time", None)
st.session_state.setdefault("run_log", "")
st.session_state.setdefault("review_status", None)
st.session_state.setdefault("latest_warning", None)
st.session_state.setdefault("xai_refreshed", False)
st.session_state.setdefault("llm_generated", False)
st.session_state.setdefault("modification_prepared", False)

saved_ga_result = (
    load_json(GA_DIR / "best_policy.json")
    if st.session_state.optimization_run
    else {}
)
if st.session_state.optimization_run and not ga_result_matches_baseline(
    saved_ga_result,
    baseline_metrics,
):
    st.session_state.optimization_run = False
    st.session_state.last_run_time = None
    st.session_state.review_status = None
    st.session_state.xai_refreshed = False
    st.session_state.llm_generated = False
    st.session_state.latest_warning = (
        "A recommendation from an incompatible baseline was ignored. "
        "Run optimization for the current large-hospital baseline."
    )


def sync_from_slider(name):
    st.session_state[f"{name}_number"] = st.session_state[f"{name}_slider"]


def sync_from_number(name):
    st.session_state[f"{name}_slider"] = st.session_state[f"{name}_number"]


def set_control(name, value):
    st.session_state[f"{name}_slider"] = value
    st.session_state[f"{name}_number"] = value


def preset_capacity(resource, factor=1.0):
    low, high = BASE_BOUNDS[resource]
    value = int(round(baseline_policy[resource] * factor))
    return max(low, min(high, value))


PRESETS = {
    "Normal Operations": {
        "icu_beds": BASE_BOUNDS["icu_beds"][1],
        "general_beds": BASE_BOUNDS["general_beds"][1],
        "doctors": BASE_BOUNDS["doctors"][1],
        "nurses": BASE_BOUNDS["nurses"][1],
        "mean_wait": 20.0,
        "high_risk_wait": 10.0,
        "high_priority": True,
        "efficiency": True,
        "instruction": "",
    },
    "Staff Shortage": {
        "icu_beds": BASE_BOUNDS["icu_beds"][1],
        "general_beds": BASE_BOUNDS["general_beds"][1],
        "doctors": 12,
        "nurses": preset_capacity("nurses", 0.80),
        "mean_wait": 20.0,
        "high_risk_wait": 10.0,
        "high_priority": True,
        "efficiency": True,
        "instruction": (
            "We have a temporary staffing shortage. Do not use more than "
            f"12 doctors or {preset_capacity('nurses', 0.80)} nurses, "
            "and keep high-risk patient waiting as low as possible."
        ),
    },
    "ICU Pressure": {
        "icu_beds": preset_capacity("icu_beds", 0.85),
        "general_beds": preset_capacity("general_beds", 1.10),
        "doctors": preset_capacity("doctors", 1.05),
        "nurses": preset_capacity("nurses", 1.05),
        "mean_wait": 25.0,
        "high_risk_wait": 8.0,
        "high_priority": True,
        "efficiency": False,
        "instruction": (
            "ICU expansion is not possible. Keep ICU beds at or below "
            f"{preset_capacity('icu_beds', 0.85)} and "
            "compensate with other available resources when appropriate."
        ),
    },
    "Cost Saving": {
        "icu_beds": preset_capacity("icu_beds", 0.90),
        "general_beds": preset_capacity("general_beds", 0.90),
        "doctors": preset_capacity("doctors", 0.90),
        "nurses": preset_capacity("nurses", 0.90),
        "mean_wait": 30.0,
        "high_risk_wait": 18.0,
        "high_priority": False,
        "efficiency": True,
        "instruction": (
            "Reduce unnecessary staffing and bed capacity. A moderate increase "
            "in waiting time is acceptable if resource efficiency improves."
        ),
    },
    "High-Risk Priority": {
        "icu_beds": preset_capacity("icu_beds", 1.15),
        "general_beds": preset_capacity("general_beds", 1.05),
        "doctors": preset_capacity("doctors", 1.15),
        "nurses": preset_capacity("nurses", 1.15),
        "mean_wait": 30.0,
        "high_risk_wait": 5.0,
        "high_priority": True,
        "efficiency": False,
        "instruction": (
            "Prioritize high-risk patients and keep their average waiting time "
            "below 5 minutes, even if resource use increases."
        ),
    },
}


def apply_preset():
    name = st.session_state.get("scenario_preset")
    if name not in PRESETS:
        return

    preset = PRESETS[name]
    for key in [
        "icu_beds",
        "general_beds",
        "doctors",
        "nurses",
        "mean_wait",
        "high_risk_wait",
    ]:
        set_control(key, preset[key])

    st.session_state.prioritize_high_risk = preset["high_priority"]
    st.session_state.prioritize_efficiency = preset["efficiency"]
    st.session_state.natural_feedback = preset["instruction"]


def reset_to_baseline():
    for resource in ["icu_beds", "general_beds", "doctors", "nurses"]:
        low, high = BASE_BOUNDS[resource]
        set_control(
            resource,
            max(low, min(high, baseline_policy[resource])),
        )
    set_control("mean_wait", 20.0)
    set_control("high_risk_wait", 10.0)
    st.session_state.prioritize_high_risk = True
    st.session_state.prioritize_efficiency = True
    st.session_state.natural_feedback = ""
    st.session_state.scenario_preset = "Custom"


def use_recommended_values():
    result = load_json(GA_DIR / "best_policy.json")
    recommendation = result.get("best_policy", {})
    if not recommendation:
        return

    for key in ["icu_beds", "general_beds", "doctors", "nurses"]:
        if key in recommendation:
            set_control(key, int(recommendation[key]))

    st.session_state.scenario_preset = "Custom"
    st.session_state.modification_prepared = True


# ============================================================
# Dark theme + CSS
# ============================================================

dark_mode = True
COLORS = {
    "bg": "#090c10",
    "bg2": "#0e1319",
    "surface": "#151b22",
    "surface2": "#1a222b",
    "surface3": "#202a34",
    "text": "#f3f6f8",
    "muted": "#aab4be",
    "border": "#33404c",
    "accent": "#82c0ea",
    "accent_hover": "#96ccef",
    "accent_text": "#071018",
    "shadow": "rgba(0,0,0,0.58)",
    "shadow_soft": "rgba(0,0,0,0.32)",
    "slider": "#82c0ea",
    "slider_track": "#34414d",
    "input": "#111820",
    "success": "#82c0ea",
}

st.markdown(
    f"""
    <style>
    :root {{
        --app-bg: {COLORS["bg"]};
        --app-bg-2: {COLORS["bg2"]};
        --surface: {COLORS["surface"]};
        --surface-2: {COLORS["surface2"]};
        --surface-3: {COLORS["surface3"]};
        --text: {COLORS["text"]};
        --muted: {COLORS["muted"]};
        --border: {COLORS["border"]};
        --accent: {COLORS["accent"]};
        --accent-hover: {COLORS["accent_hover"]};
        --accent-text: {COLORS["accent_text"]};
        --shadow: {COLORS["shadow"]};
        --shadow-soft: {COLORS["shadow_soft"]};
        --slider: {COLORS["slider"]};
        --slider-track: {COLORS["slider_track"]};
        --input: {COLORS["input"]};
    }}

    html, body, .stApp, [data-testid="stAppViewContainer"] {{
        background:
            radial-gradient(circle at 10% 0%, var(--app-bg-2) 0%, transparent 34%),
            linear-gradient(180deg, var(--app-bg) 0%, var(--app-bg-2) 100%);
        color: var(--text) !important;
    }}

    [data-testid="stHeader"] {{ background: transparent; }}

    .block-container {{
        max-width: 1500px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--surface), var(--surface-2));
        border-right: 1px solid var(--border);
        box-shadow: 10px 0 30px var(--shadow-soft);
    }}

    [data-testid="stSidebar"] * {{
        color: var(--text);
    }}

    .hero {{
        padding: 1.6rem 1.8rem;
        border: 1px solid var(--border);
        border-radius: 22px;
        background: linear-gradient(145deg, var(--surface), var(--surface-2));
        box-shadow:
            0 16px 34px var(--shadow-soft),
            inset 0 1px 0 rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }}

    .hero h1 {{
        margin: 0.15rem 0 0;
        font-size: 2.08rem;
        letter-spacing: -0.035em;
        color: var(--text) !important;
    }}

    .hero p {{
        margin: 0.5rem 0 0;
        max-width: 920px;
        line-height: 1.55;
        color: var(--muted) !important;
    }}

    .section-kicker {{
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }}

    .status-strip {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0,1fr));
        gap: 0.75rem;
        margin: 0 0 1rem;
    }}

    .status-box {{
        border: 1px solid var(--border);
        background: linear-gradient(145deg, var(--surface), var(--surface-2));
        border-radius: 15px;
        padding: 0.85rem 1rem;
        box-shadow: 0 8px 18px var(--shadow-soft);
        min-width: 0;
        min-height: 88px;
    }}

    .status-label {{
        color: var(--muted);
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.075em;
        font-weight: 750;
    }}

    .status-value {{
        color: var(--text);
        font-size: 1rem;
        font-weight: 760;
        margin-top: 0.15rem;
        overflow-wrap: anywhere;
        line-height: 1.3;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: linear-gradient(145deg, var(--surface), var(--surface-2));
        border: 1px solid var(--border) !important;
        border-radius: 20px !important;
        box-shadow:
            0 15px 30px var(--shadow-soft),
            0 3px 8px var(--shadow-soft),
            inset 0 1px 0 rgba(255,255,255,0.07);
        padding: 0.45rem 0.55rem;
        margin-bottom: 0.7rem;
    }}

    [data-testid="stMetric"] {{
        background: linear-gradient(150deg, var(--surface), var(--surface-3));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.95rem 1rem;
        min-height: 106px;
        box-shadow:
            0 9px 20px var(--shadow-soft),
            inset 0 1px 0 rgba(255,255,255,0.07);
        transition: transform 140ms ease, box-shadow 140ms ease;
    }}

    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 14px 26px var(--shadow);
    }}

    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricLabel"] {{
        color: var(--muted) !important;
        font-weight: 650 !important;
        white-space: normal !important;
        overflow-wrap: anywhere;
        line-height: 1.25 !important;
    }}

    [data-testid="stMetricValue"] *,
    [data-testid="stMetricValue"] {{
        color: var(--text) !important;
        font-size: clamp(1.35rem, 1.8vw, 2rem) !important;
        line-height: 1.15 !important;
        white-space: normal !important;
        overflow-wrap: anywhere;
    }}

    [data-testid="stMetricDelta"] *,
    [data-testid="stMetricDelta"] {{
        color: var(--muted) !important;
        white-space: normal !important;
        overflow-wrap: anywhere;
    }}

    [data-baseweb="tab-list"] {{
        gap: 0.35rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.35rem;
        box-shadow: 0 8px 18px var(--shadow-soft);
    }}

    [data-baseweb="tab"] {{
        border-radius: 10px;
        padding: 0.55rem 0.9rem;
        color: var(--muted) !important;
    }}

    [aria-selected="true"][data-baseweb="tab"] {{
        background: var(--surface-3);
        color: var(--text) !important;
        font-weight: 750;
        box-shadow: inset 0 -2px 0 var(--accent);
    }}

    input, textarea,
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {{
        background: var(--input) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
        border-radius: 11px !important;
    }}

    input::placeholder, textarea::placeholder {{
        color: var(--muted) !important;
        opacity: 0.9;
    }}

    [data-testid="stSlider"] [data-baseweb="slider"] > div {{
        background: var(--slider-track) !important;
    }}

    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {{
        background: var(--accent) !important;
    }}

    [data-testid="stSlider"] [role="slider"] {{
        background: var(--slider) !important;
        border: 3px solid var(--surface) !important;
        box-shadow: 0 0 0 1px var(--accent), 0 3px 10px var(--shadow) !important;
    }}

    .stButton > button {{
        border-radius: 12px;
        border: 1px solid var(--border);
        background: var(--surface-3);
        color: var(--text) !important;
        min-height: 2.75rem;
        font-weight: 750;
        box-shadow: 0 8px 17px var(--shadow-soft);
        transition: transform 140ms ease, box-shadow 140ms ease;
    }}

    .stButton > button * {{
        color: inherit !important;
    }}

    .stButton > button[kind="primary"] {{
        background: var(--accent);
        color: var(--accent-text) !important;
        border-color: var(--accent);
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 12px 22px var(--shadow);
        border-color: var(--accent);
        color: var(--accent-hover) !important;
    }}

    .stButton > button[kind="primary"]:hover {{
        background: var(--accent-hover);
        color: var(--accent-text) !important;
    }}

    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stHeader"] button[kind="header"],
    [data-testid="stHeader"] button[kind="header"] * {{
        color: var(--muted) !important;
        fill: var(--muted) !important;
        stroke: var(--muted) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stTooltipIcon"] {{
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        position: static !important;
        flex: 0 0 1.1rem !important;
        width: 1.1rem !important;
        height: 1.1rem !important;
        min-width: 1.1rem !important;
        margin-left: 0.3rem !important;
        padding: 0 !important;
        border: 0 !important;
        border-radius: 999px !important;
        background: transparent !important;
        color: var(--muted) !important;
        opacity: 0.9;
        vertical-align: middle;
        transition: color 120ms ease, background 120ms ease, opacity 120ms ease;
    }}

    [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {{
        width: 0.82rem !important;
        height: 0.82rem !important;
        color: inherit !important;
    }}

    [data-testid="stSidebar"] [data-testid="stTooltipIcon"]:hover {{
        color: var(--accent) !important;
        background: rgba(130, 192, 234, 0.12) !important;
        opacity: 1;
    }}

    .sidebar-section {{
        margin: 1.25rem 0 0.65rem;
        padding-top: 0.85rem;
        border-top: 1px solid var(--border);
    }}

    .sidebar-section-title {{
        color: var(--accent) !important;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.075em;
        line-height: 1.25;
        text-transform: uppercase;
    }}

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
        display: flex !important;
        align-items: center !important;
        margin-bottom: 0.22rem;
    }}

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        font-size: 0.88rem;
        font-weight: 650;
        line-height: 1.3;
    }}

    /* Keep the sidebar CTA area independent from Streamlit's column sizing. */
    [data-testid="stSidebar"] .st-key-optimization_actions,
    [data-testid="stSidebar"] .st-key-optimization_actions [data-testid="stVerticalBlock"],
    [data-testid="stSidebar"] .st-key-optimization_actions .stButton {{
        width: 100% !important;
        min-width: 0 !important;
    }}

    [data-testid="stSidebar"] .st-key-optimization_actions .stButton {{
        margin-bottom: 0.5rem;
    }}

    [data-testid="stSidebar"] .st-key-optimization_actions .stButton > button {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        min-width: 100% !important;
        min-height: 2.8rem !important;
        height: auto !important;
        padding: 0.65rem 0.9rem !important;
        white-space: nowrap !important;
        word-break: normal !important;
        overflow-wrap: normal !important;
    }}

    [data-testid="stSidebar"] .st-key-optimization_actions .stButton > button * {{
        white-space: nowrap !important;
        word-break: normal !important;
        overflow-wrap: normal !important;
        text-align: center !important;
    }}

    [data-testid="stExpander"],
    [data-testid="stStatusWidget"],
    [data-testid="stAlert"] {{
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border);
        border-radius: 14px;
        box-shadow: 0 7px 16px var(--shadow-soft);
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 7px 16px var(--shadow-soft);
    }}

    h1, h2, h3, h4, h5, h6, p, label, span {{
        color: var(--text);
    }}

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {{
        color: var(--muted) !important;
    }}

    hr {{ border-color: var(--border) !important; }}

    .rule-flow {{
        display: flex;
        align-items: stretch;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin: 0.6rem 0 1rem;
    }}

    .rule-step {{
        flex: 1 1 220px;
        padding: 0.85rem 1rem;
        border-radius: 14px;
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        background: var(--surface-3);
        box-shadow: 0 7px 16px var(--shadow-soft);
        color: var(--text);
    }}

    .rule-num {{
        display: inline-flex;
        min-width: 1.55rem;
        padding: 0 0.3rem;
        height: 1.55rem;
        border-radius: 999px;
        align-items: center;
        justify-content: center;
        background: var(--accent);
        color: var(--accent-text) !important;
        font-size: 0.78rem;
        font-weight: 800;
        margin-right: 0.4rem;
    }}

    .change-box {{
        padding: 0.9rem 1rem;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: var(--surface-3);
        color: var(--text);
        margin-bottom: 0.45rem;
    }}

    code, pre {{
        color: var(--text) !important;
    }}

    [data-testid="stDecoration"] {{ display: none; }}

    @media (max-width: 900px) {{
        .status-strip {{
            grid-template-columns: repeat(2, minmax(0,1fr));
        }}

        [data-testid="stMetric"] {{
            min-height: 96px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

def sidebar_section(title):
    st.markdown(
        f'<div class="sidebar-section"><div class="sidebar-section-title">{title}</div></div>',
        unsafe_allow_html=True,
    )


def int_control(label, name, low, high, help_text):
    left, right = st.columns([2.2, 1])
    with left:
        st.slider(
            label,
            min_value=low,
            max_value=high,
            key=f"{name}_slider",
            on_change=sync_from_slider,
            args=(name,),
            help=help_text,
        )
    with right:
        st.number_input(
            "Direct entry",
            min_value=low,
            max_value=high,
            step=1,
            key=f"{name}_number",
            on_change=sync_from_number,
            args=(name,),
            help=f"Type the {label.lower()} value directly.",
        )
    return int(st.session_state[f"{name}_number"])


def float_control(label, name, low, high, step, help_text):
    left, right = st.columns([2.2, 1])
    with left:
        st.slider(
            label,
            min_value=float(low),
            max_value=float(high),
            step=float(step),
            key=f"{name}_slider",
            on_change=sync_from_slider,
            args=(name,),
            help=help_text,
        )
    with right:
        st.number_input(
            "Direct entry",
            min_value=float(low),
            max_value=float(high),
            step=float(step),
            format="%.2f",
            key=f"{name}_number",
            on_change=sync_from_number,
            args=(name,),
            help=f"Type the {label.lower()} directly.",
        )
    return float(st.session_state[f"{name}_number"])


with st.sidebar:
    st.markdown("### Optimization Controls")
    st.caption(
        "Use the slider or type a value directly. Hover over the info icons for details."
    )

    st.selectbox(
        "Scenario preset",
        ["Custom", *PRESETS.keys()],
        key="scenario_preset",
        on_change=apply_preset,
        help="Loads a complete example operating scenario. You can still edit every value afterward.",
    )

    sidebar_section("Resource Constraints")

    max_icu = int_control(
        "Maximum ICU beds",
        "icu_beds",
        *BASE_BOUNDS["icu_beds"],
        "Highest ICU-bed capacity the optimizer may use.",
    )
    max_general = int_control(
        "Maximum general beds",
        "general_beds",
        *BASE_BOUNDS["general_beds"],
        "Highest general-bed capacity the optimizer may use.",
    )
    max_doctors = int_control(
        "Maximum concurrent doctors",
        "doctors",
        *BASE_BOUNDS["doctors"],
        "Highest simultaneous doctor capacity the optimizer may allocate; not total employees.",
    )
    max_nurses = int_control(
        "Maximum concurrent nurses",
        "nurses",
        *BASE_BOUNDS["nurses"],
        "Highest simultaneous nurse capacity the optimizer may allocate; not total employees.",
    )

    sidebar_section("Operational Targets")

    mean_wait_target = float_control(
        "Mean wait target (min)",
        "mean_wait",
        1.0,
        120.0,
        1.0,
        "Average waiting-time target. Exceeding it is penalized by the GA.",
    )
    high_risk_wait_target = float_control(
        "High-risk wait target (min)",
        "high_risk_wait",
        1.0,
        120.0,
        1.0,
        "Waiting-time target for higher-risk patients.",
    )

    st.checkbox(
        "Prioritize high-risk patients",
        key="prioritize_high_risk",
        help="Places more weight on high-risk patient waiting time in the GA fitness function.",
    )
    st.checkbox(
        "Prioritize resource efficiency",
        key="prioritize_efficiency",
        help="Places more penalty on unnecessary beds and staffing.",
    )

    sidebar_section("Natural-Language Feedback")
    st.text_area(
        "Natural-language instruction",
        key="natural_feedback",
        height=100,
        placeholder="Example: We only have 10 nurses available this week.",
        help=(
            "Write a normal-language constraint or preference. Groq translates it "
            "into supported GA constraints/objectives."
        ),
    )

    use_llm_interpreter = st.checkbox(
        "Interpret instruction with AI",
        value=True,
        disabled=not bool(st.session_state.natural_feedback.strip()),
        help=(
            "The LLM only translates your instruction. The Genetic Algorithm still "
            "performs the actual optimization."
        ),
    )

    sidebar_section("Simulation Settings")
    ga_patients = st.select_slider(
        "Simulation Patients",
        options=[250, 500, 750, 1000],
        value=500,
        help="More patients improve scenario coverage but increase runtime.",
    )

    sidebar_section("GA Settings")
    ga_population = st.select_slider(
        "GA Population",
        options=[8, 10, 12, 16, 20],
        value=12,
        help="Candidate policies tested in each GA generation.",
    )
    ga_generations = st.select_slider(
        "GA Generations",
        options=[5, 8, 10, 15, 20],
        value=10,
        help="Evolutionary improvement cycles performed by the GA.",
    )

    sidebar_section("Optimization Actions")
    with st.container(key="optimization_actions"):
        st.button(
            "Reset to Baseline",
            type="secondary",
            width="stretch",
            on_click=reset_to_baseline,
            disabled=not baseline_valid,
        )
        st.button(
            "Use Recommended Values",
            type="secondary",
            width="stretch",
            on_click=use_recommended_values,
            disabled=not st.session_state.optimization_run,
        )
        run_clicked = st.button(
            "Run Optimization",
            type="primary",
            width="stretch",
        )


# ============================================================
# Header + top status
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="section-kicker">Clinical Operations Decision Support</div>
        <h1>Healthcare Digital Twin</h1>
        <p>
            Review current hospital capacity, test operational constraints,
            optimize resource allocation, and inspect explainable recommendations
            before accepting or modifying a policy.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if run_clicked:
    system_status = "Optimizing"
elif st.session_state.optimization_run:
    if st.session_state.review_status:
        system_status = st.session_state.review_status
    else:
        system_status = "Awaiting Review"
else:
    system_status = "Baseline" if baseline_valid else "Baseline Invalid"

status_run = st.session_state.last_run_time or "Not run"
status_scenario = st.session_state.get("scenario_preset", "Custom")
status_data = (
    f"{BASELINE_MAX_PATIENTS:,}-patient cohort / "
    f"{HOSPITAL_SCALE_MULTIPLIER:.1f}x workload"
    if not synthetic_df.empty
    else "Data unavailable"
)

st.markdown(
    f"""
    <div class="status-strip">
        <div class="status-box">
            <div class="status-label">System status</div>
            <div class="status-value">{html.escape(system_status)}</div>
        </div>
        <div class="status-box">
            <div class="status-label">Latest optimization</div>
            <div class="status-value">{html.escape(status_run)}</div>
        </div>
        <div class="status-box">
            <div class="status-label">Scenario</div>
            <div class="status-value">{html.escape(status_scenario)}</div>
        </div>
        <div class="status-box">
            <div class="status-label">Simulation dataset</div>
            <div class="status-value">{html.escape(status_data)}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not baseline_valid:
    st.error(
        "The saved Digital Twin baseline was not displayed because its "
        "provenance check failed. Regenerate it with `python src/digital_twin.py`."
    )
    with st.expander("Baseline validation details"):
        for baseline_error in baseline_errors:
            st.write(f"- {baseline_error}")


# ============================================================
# Pre-run validation
# ============================================================

constraint_warnings = []

if max_icu < BASE_BOUNDS["icu_beds"][0]:
    constraint_warnings.append("ICU-bed maximum is below the GA search minimum.")
if max_general < BASE_BOUNDS["general_beds"][0]:
    constraint_warnings.append("General-bed maximum is below the GA search minimum.")
if max_doctors < BASE_BOUNDS["doctors"][0]:
    constraint_warnings.append("Doctor maximum is below the GA search minimum.")
if max_nurses < BASE_BOUNDS["nurses"][0]:
    constraint_warnings.append("Nurse maximum is below the GA search minimum.")

if high_risk_wait_target > mean_wait_target and st.session_state.prioritize_high_risk:
    constraint_warnings.append(
        "High-risk priority is enabled, but the high-risk wait target is looser than the overall mean-wait target."
    )

if constraint_warnings:
    with st.container(border=True):
        st.warning("Please review these settings before optimization:")
        for warning in constraint_warnings:
            st.write(f"- {warning}")


# ============================================================
# Run optimization
# ============================================================

if run_clicked:
    st.session_state.latest_warning = None
    st.session_state.xai_refreshed = False
    st.session_state.llm_generated = False

    structured_feedback = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "decision": "Needs Modification",
        "rating": 4,
        "comments": st.session_state.natural_feedback.strip(),
        "constraints": {
            "max_icu_beds": int(max_icu),
            "max_general_beds": int(max_general),
            "max_doctors": int(max_doctors),
            "max_nurses": int(max_nurses),
        },
        "targets": {
            "mean_wait_min": float(mean_wait_target),
            "high_risk_wait_min": float(high_risk_wait_target),
        },
    }

    (FEEDBACK_DIR / "latest_feedback.json").write_text(
        json.dumps(structured_feedback, indent=2),
        encoding="utf-8",
    )

    manual_interpretation = build_manual_interpretation(
        BASE_BOUNDS,
        max_icu,
        max_general,
        max_doctors,
        max_nurses,
        mean_wait_target,
        high_risk_wait_target,
        st.session_state.prioritize_high_risk,
        st.session_state.prioritize_efficiency,
    )

    interpreted = manual_interpretation
    logs = []
    warnings = []
    fatal_error = None

    with st.status("Running optimization...", expanded=True) as status:
        st.write("Preparing constraints")

        if st.session_state.natural_feedback.strip() and use_llm_interpreter:
            st.write("Interpreting feedback")

            if not os.getenv("GROQ_API_KEY"):
                warnings.append(
                    "GROQ_API_KEY is unavailable to Streamlit. Explicit controls were used."
                )
            else:
                ok, output = run_python("feedback_interpreter.py")
                logs.append(output)

                if ok:
                    llm_interpretation = load_json(
                        FEEDBACK_DIR / "interpreted_feedback.json"
                    )
                    try:
                        interpreted = merge_llm_feedback(
                            manual_interpretation,
                            llm_interpretation,
                        )
                    except ValueError as exc:
                        fatal_error = str(exc)
                else:
                    warnings.append(
                        "Natural-language interpretation failed; explicit controls were used."
                    )

        if fatal_error is None:
            (FEEDBACK_DIR / "interpreted_feedback.json").write_text(
                json.dumps(interpreted, indent=2),
                encoding="utf-8",
            )

            st.write("Running Digital Twin + Genetic Algorithm")
            ok, output = run_python(
                "genetic_algorithm.py",
                [
                    "--max-patients", str(ga_patients),
                    "--population", str(ga_population),
                    "--generations", str(ga_generations),
                ],
            )
            logs.append(output)

            if not ok:
                fatal_error = "Genetic Algorithm / Digital Twin stage failed."

        if fatal_error is None:
            st.write("Refreshing explainability")
            xai_ok, xai_output = run_python("decision_tree_xai.py")
            logs.append(xai_output)

            if not xai_ok:
                warnings.append("Decision Tree XAI refresh failed.")
            else:
                st.session_state.xai_refreshed = True

            st.write("Generating explanation")
            if os.getenv("GROQ_API_KEY"):
                llm_ok, llm_output = run_python("llm_explanation.py")
                logs.append(llm_output)
                if not llm_ok:
                    warnings.append("LLM explanation generation failed.")
                else:
                    st.session_state.llm_generated = True
            else:
                warnings.append(
                    "LLM explanation skipped because GROQ_API_KEY is unavailable."
                )

        if fatal_error:
            status.update(label="Optimization failed", state="error")
            st.error(fatal_error)
        else:
            st.session_state.optimization_run = True
            st.session_state.last_run_time = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            st.session_state.review_status = None
            st.session_state.modification_prepared = False
            st.write("Complete")

            if warnings:
                status.update(
                    label="Optimization completed with warnings",
                    state="complete",
                )
                st.session_state.latest_warning = " ".join(warnings)
            else:
                status.update(
                    label="Optimization completed",
                    state="complete",
                )

    st.session_state.run_log = "\n\n".join(logs)

    if not fatal_error:
        st.success(
            "Optimization complete. The latest recommendation is available in Optimize and Dashboard."
        )


# ============================================================
# Load latest optimization after run
# ============================================================

ga_result = (
    load_json(GA_DIR / "best_policy.json")
    if st.session_state.optimization_run
    else {}
)
if ga_result and not ga_result_matches_baseline(ga_result, baseline_metrics):
    ga_result = {}
best_policy = ga_result.get("best_policy", {})
best_metrics = ga_result.get("best_metrics", {})
best_fitness = ga_result.get("best_fitness")
interpreted_feedback = load_json(FEEDBACK_DIR / "interpreted_feedback.json")


# ============================================================
# Main navigation
# ============================================================

dashboard_tab, optimize_tab, review_tab = st.tabs(
    ["Dashboard", "Optimize", "Review"]
)


# ============================================================
# DASHBOARD
# ============================================================

with dashboard_tab:
    # Reserve the first Dashboard position for the session-scoped comparison.
    # Its contents are rendered below without duplicating the existing metrics.
    comparison_panel = st.container(border=True)

    with st.container(border=True):
        st.header("Current Hospital State")
        st.caption(
            "This section is the baseline. Opening the dashboard does not run the optimizer."
        )

        b1, b2, b3, b4 = st.columns(4)
        b1.metric(
            "Current ICU Beds",
            fmt_int(baseline_policy["icu_beds"] if baseline_valid else None),
            help="ICU-bed capacity in the baseline simulated hospital.",
        )
        b2.metric(
            "Current General Beds",
            fmt_int(baseline_policy["general_beds"] if baseline_valid else None),
            help="General-bed capacity in the baseline simulated hospital.",
        )
        b3.metric(
            "Modeled Concurrent Doctors",
            fmt_int(baseline_policy["doctors"] if baseline_valid else None),
            help=(
                "Number of doctors simultaneously available in the simulation. "
                "This is not the hospital's total employed medical staff."
            ),
        )
        b4.metric(
            "Modeled Concurrent Nurses",
            fmt_int(baseline_policy["nurses"] if baseline_valid else None),
            help=(
                "Number of nurses simultaneously available in the simulation. "
                "This is not the hospital's total employed nursing staff."
            ),
        )

        if baseline_metrics:
            generated_at = baseline_metadata.get("generated_at_utc", "unknown")
            methodology = baseline_metadata.get("capacity_methodology", {})
            offered_load = methodology.get("offered_concurrent_load", {})
            st.caption(
                f"Validated baseline: earliest {BASELINE_MAX_PATIENTS:,} arrivals; "
                f"{HOSPITAL_SCALE_MULTIPLIER:.1f}x deterministic workload scale; "
                f"generated {generated_at}."
            )
            st.info(
                "Demand-derived baseline at a 75% utilization target: "
                f"ICU {fmt(offered_load.get('icu_beds'), 2)} -> "
                f"{baseline_policy['icu_beds']} beds, general "
                f"{fmt(offered_load.get('general_beds'), 2)} -> "
                f"{baseline_policy['general_beds']} beds, doctors "
                f"{fmt(offered_load.get('doctors'), 2)} -> "
                f"{baseline_policy['doctors']}, and nurses "
                f"{fmt(offered_load.get('nurses'), 2)} -> "
                f"{baseline_policy['nurses']}."
            )
            st.subheader("Baseline Performance")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric(
                "Mean Wait",
                fmt_minutes(baseline_metrics.get("mean_waiting_time_min")),
            )
            m2.metric(
                "Median Wait",
                fmt_minutes(baseline_metrics.get("median_waiting_time_min")),
            )
            m3.metric(
                "P95 Wait",
                fmt_minutes(baseline_metrics.get("p95_waiting_time_min")),
            )
            m4.metric(
                "High-Risk Mean Wait",
                fmt_minutes(
                    baseline_metrics.get("high_risk_mean_waiting_time_min")
                ),
            )
            m5.metric(
                "Throughput",
                fmt_throughput(baseline_metrics.get("throughput_patients_per_day")),
            )

            if float(baseline_metrics.get("mean_waiting_time_min", 0)) > 1440:
                load_profile = baseline_metadata.get("load_profile", {})
                st.warning(
                    "This baseline is a validated overload simulation, not a stale "
                    "historical result. The cohort creates an average offered load "
                    f"of {fmt(load_profile.get('average_icu_beds_required_during_arrivals'), 2)} "
                    f"ICU beds during arrivals, versus {baseline_policy['icu_beds']} "
                    "available. Beds remain occupied for each patient's full length "
                    "of stay, so the ICU queue accumulates."
                )

            util_df = pd.DataFrame(
                {
                    "Resource": ["ICU Beds", "General Beds", "Doctors", "Nurses"],
                    "Utilization (%)": [
                        baseline_metrics.get("icu_bed_utilization", 0) * 100,
                        baseline_metrics.get("general_bed_utilization", 0) * 100,
                        baseline_metrics.get("doctor_utilization", 0) * 100,
                        baseline_metrics.get("nurse_utilization", 0) * 100,
                    ],
                }
            ).round(2)

            st.altair_chart(
                monochrome_bar_chart(
                    util_df,
                    "Resource",
                    "Utilization (%)",
                    "Baseline Resource Utilization",
                    dark_mode,
                ),
                width="stretch",
            )
            st.caption(
                "Sizing utilization uses the cohort's arrival window. The chart "
                "uses the full simulation duration, including the discharge tail, "
                "so its percentages are lower."
            )

            with st.expander("Baseline capacity methodology"):
                st.json(methodology)
        else:
            st.info(
                "Baseline metrics are unavailable until a validated Digital Twin "
                "baseline is generated."
            )

    with comparison_panel:
        st.header("Baseline vs Latest Recommendation")
        if not st.session_state.optimization_run or not best_policy:
            st.info(
                "No optimized policy has been generated in this session. "
                "Configure constraints in the sidebar and select Run Optimization."
            )
        else:
            st.caption(
                "Values show validated baseline -> latest recommendation. "
                "Deltas are recommendation minus baseline."
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "ICU Beds",
                f"{fmt_int(baseline_policy['icu_beds'])} -> "
                f"{fmt_int(best_policy.get('icu_beds'))}",
                delta=delta_number(
                    best_policy.get("icu_beds"), baseline_policy["icu_beds"], 0
                ),
                delta_color="off",
            )
            c2.metric(
                "General Beds",
                f"{fmt_int(baseline_policy['general_beds'])} -> "
                f"{fmt_int(best_policy.get('general_beds'))}",
                delta=delta_number(
                    best_policy.get("general_beds"),
                    baseline_policy["general_beds"],
                    0,
                ),
                delta_color="off",
            )
            c3.metric(
                "Modeled Concurrent Doctors",
                f"{fmt_int(baseline_policy['doctors'])} -> "
                f"{fmt_int(best_policy.get('doctors'))}",
                delta=delta_number(
                    best_policy.get("doctors"), baseline_policy["doctors"], 0
                ),
                delta_color="off",
                help="Simultaneous modeled capacity, not total employed doctors.",
            )
            c4.metric(
                "Modeled Concurrent Nurses",
                f"{fmt_int(baseline_policy['nurses'])} -> "
                f"{fmt_int(best_policy.get('nurses'))}",
                delta=delta_number(
                    best_policy.get("nurses"), baseline_policy["nurses"], 0
                ),
                delta_color="off",
                help="Simultaneous modeled capacity, not total employed nurses.",
            )

            w1, w2, w3, w4 = st.columns(4)
            w1.metric(
                "Mean Wait",
                f"{fmt_minutes(baseline_metrics.get('mean_waiting_time_min'))} -> "
                f"{fmt_minutes(best_metrics.get('mean_waiting_time_min'))}",
                delta=delta_number(
                    best_metrics.get("mean_waiting_time_min"),
                    baseline_metrics.get("mean_waiting_time_min"),
                    2,
                    " min",
                ),
                delta_color="off",
            )
            w2.metric(
                "High-Risk Mean Wait",
                f"{fmt_minutes(baseline_metrics.get('high_risk_mean_waiting_time_min'))} -> "
                f"{fmt_minutes(best_metrics.get('high_risk_mean_waiting_time_min'))}",
                delta=delta_number(
                    best_metrics.get("high_risk_mean_waiting_time_min"),
                    baseline_metrics.get("high_risk_mean_waiting_time_min"),
                    2,
                    " min",
                ),
                delta_color="off",
            )
            w3.metric(
                "P95 Wait",
                f"{fmt_minutes(baseline_metrics.get('p95_waiting_time_min'))} -> "
                f"{fmt_minutes(best_metrics.get('p95_waiting_time_min'))}",
                delta=delta_number(
                    best_metrics.get("p95_waiting_time_min"),
                    baseline_metrics.get("p95_waiting_time_min"),
                    2,
                    " min",
                ),
                delta_color="off",
            )
            w4.metric(
                "Throughput",
                f"{fmt_throughput(baseline_metrics.get('throughput_patients_per_day'))} -> "
                f"{fmt_throughput(best_metrics.get('throughput_patients_per_day'))}",
                delta=delta_number(
                    best_metrics.get("throughput_patients_per_day"),
                    baseline_metrics.get("throughput_patients_per_day"),
                    2,
                    "/day",
                ),
                delta_color="off",
            )

    with st.container(border=True):
        st.header("Dataset & Prediction Model")

        if not synthetic_df.empty:
            d1, d2, d3 = st.columns(3)
            d1.metric("Synthetic Encounters", f"{len(synthetic_df):,}")
            d2.metric("Available Features", f"{len(synthetic_df.columns):,}")

            if "high_resource_need" in synthetic_df.columns:
                positive_rate = pd.to_numeric(
                    synthetic_df["high_resource_need"],
                    errors="coerce",
                ).mean()
                d3.metric(
                    "High-Resource Cases",
                    fmt_pct(positive_rate, 1),
                )

        if rf_metrics:
            st.subheader("Random Forest")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Accuracy", fmt_pct(rf_metrics.get("accuracy"), 1))
            r2.metric("Precision", fmt_pct(rf_metrics.get("precision"), 1))
            r3.metric("F1 Score", fmt_pct(rf_metrics.get("f1"), 1))
            r4.metric("ROC-AUC", fmt_pct(rf_metrics.get("roc_auc"), 1))

            if not rf_importance.empty:
                feature_col = next(
                    (
                        c
                        for c in rf_importance.columns
                        if c.lower() in {"feature", "feature_name", "features"}
                    ),
                    None,
                )
                importance_col = next(
                    (
                        c
                        for c in rf_importance.columns
                        if c.lower() in {"importance", "feature_importance"}
                    ),
                    None,
                )

                if feature_col and importance_col:
                    feature_df = (
                        rf_importance[[feature_col, importance_col]]
                        .head(10)
                        .rename(
                            columns={
                                feature_col: "Feature",
                                importance_col: "Importance",
                            }
                        )
                        .copy()
                    )
                    feature_df["Importance"] = feature_df["Importance"].round(4)
                    st.altair_chart(
                        monochrome_bar_chart(
                            feature_df,
                            "Feature",
                            "Importance",
                            "Top Predictive Features",
                            dark_mode,
                        ),
                        width="stretch",
                    )


# ============================================================
# OPTIMIZE
# ============================================================

with optimize_tab:
    with st.container(border=True):
        st.header("Optimization")

        if not st.session_state.optimization_run:
            st.info(
                "No optimized policy has been generated in this session. "
                "Configure constraints in the sidebar and select Run Optimization."
            )

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Max ICU Beds", fmt_int(max_icu))
            s2.metric("Max General Beds", fmt_int(max_general))
            s3.metric("Max Concurrent Doctors", fmt_int(max_doctors))
            s4.metric("Max Concurrent Nurses", fmt_int(max_nurses))

            t1, t2 = st.columns(2)
            t1.metric("Mean Wait Target", fmt_minutes(mean_wait_target))
            t2.metric(
                "High-Risk Wait Target",
                fmt_minutes(high_risk_wait_target),
            )

        else:
            st.success(
                f"Latest optimization completed at {st.session_state.last_run_time}."
            )

            st.subheader("Recommended Policy")
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("ICU Beds", fmt_int(best_policy.get("icu_beds")))
            p2.metric("General Beds", fmt_int(best_policy.get("general_beds")))
            p3.metric(
                "Modeled Concurrent Doctors",
                fmt_int(best_policy.get("doctors")),
                help="Simultaneous modeled capacity, not total employed doctors.",
            )
            p4.metric(
                "Modeled Concurrent Nurses",
                fmt_int(best_policy.get("nurses")),
                help="Simultaneous modeled capacity, not total employed nurses.",
            )
            p5.metric(
                "Fitness",
                fmt_fitness(best_fitness),
                help="Higher (less negative) is better only within the current GA objective.",
            )

            st.subheader("Expected Digital Twin Performance")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric(
                "Mean Wait",
                fmt_minutes(best_metrics.get("mean_waiting_time_min")),
            )
            d2.metric(
                "High-Risk Wait",
                fmt_minutes(best_metrics.get("high_risk_mean_waiting_time_min")),
            )
            d3.metric(
                "P95 Wait",
                fmt_minutes(best_metrics.get("p95_waiting_time_min")),
            )
            d4.metric(
                "Throughput",
                fmt_throughput(best_metrics.get("throughput_patients_per_day")),
            )

            resource_compare = pd.DataFrame(
                {
                    "Resource": ["ICU Beds", "General Beds", "Doctors", "Nurses"],
                    "Baseline": [
                        baseline_policy["icu_beds"],
                        baseline_policy["general_beds"],
                        baseline_policy["doctors"],
                        baseline_policy["nurses"],
                    ],
                    "Recommended": [
                        best_policy.get("icu_beds", 0),
                        best_policy.get("general_beds", 0),
                        best_policy.get("doctors", 0),
                        best_policy.get("nurses", 0),
                    ],
                }
            )

            compare_long = resource_compare.melt(
                id_vars="Resource",
                var_name="Policy",
                value_name="Count",
            )
            palette = ["#65727e", "#82c0ea"]

            comparison_chart = style_chart((
                alt.Chart(compare_long)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Resource:N", title=None),
                    xOffset="Policy:N",
                    y=alt.Y("Count:Q", title="Resource count"),
                    color=alt.Color(
                        "Policy:N",
                        scale=alt.Scale(domain=["Baseline", "Recommended"], range=palette),
                        legend=alt.Legend(title=None),
                    ),
                    tooltip=[
                        alt.Tooltip("Resource:N"),
                        alt.Tooltip("Policy:N"),
                        alt.Tooltip("Count:Q", format=".0f"),
                    ],
                )
                .properties(height=290, title="Baseline vs Recommended Resources")
            ), dark_mode)
            st.altair_chart(comparison_chart, width="stretch")

            ga_history = load_csv(GA_DIR / "best_by_generation.csv")
            if not ga_history.empty and {"generation", "fitness"}.issubset(
                ga_history.columns
            ):
                ga_history = ga_history[["generation", "fitness"]].copy()
                ga_history["generation"] = pd.to_numeric(
                    ga_history["generation"],
                    errors="coerce",
                )
                ga_history["fitness"] = pd.to_numeric(
                    ga_history["fitness"],
                    errors="coerce",
                ).round(2)

                st.altair_chart(
                    monochrome_line_chart(
                        ga_history,
                        "generation",
                        "fitness",
                        "GA Improvement Across Generations",
                        dark_mode,
                    ),
                    width="stretch",
                )

    if st.session_state.optimization_run:
        with st.container(border=True):
            st.header("Why This Changed")
            notes = build_change_notes(
                baseline_policy,
                best_policy,
                baseline_metrics,
                best_metrics,
                interpreted_feedback,
            )

            if notes:
                for note in notes:
                    st.markdown(
                        f'<div class="change-box">{html.escape(note)}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.write(
                    "The optimized recommendation did not produce a material change "
                    "from the baseline values."
                )

        with st.container(border=True):
            st.header("Applied Constraints")
            constraints = interpreted_feedback.get("constraints", {}) or {}
            objectives = interpreted_feedback.get("objectives", {}) or {}

            labels = {
                "icu_beds": "ICU Beds",
                "general_beds": "General Beds",
                "doctors": "Concurrent Doctors",
                "nurses": "Concurrent Nurses",
            }
            constraint_columns = st.columns(4)
            for column, key in zip(constraint_columns, labels):
                limits = constraints.get(key, {}) or {}
                low = fmt_int(limits.get("min"))
                high = fmt_int(limits.get("max"))
                column.metric(labels[key], f"{low}-{high}")

            target1, target2, target3 = st.columns(3)
            target1.metric(
                "Mean Wait Target",
                fmt_minutes(objectives.get("acceptable_mean_wait_min")),
            )
            target2.metric(
                "High-Risk Wait Target",
                fmt_minutes(objectives.get("acceptable_high_risk_wait_min")),
            )
            target3.metric(
                "High-Risk Weighting",
                fmt(objectives.get("high_risk_wait_weight"), 2),
            )

            summary = interpreted_feedback.get("summary")
            if summary:
                st.info(f"Natural-language interpretation: {summary}")

            with st.expander("Detailed applied JSON"):
                st.json(interpreted_feedback)


# ============================================================
# EXPLAINABILITY - INSIDE OPTIMIZE
# ============================================================

with optimize_tab:
    with st.container(border=True):
        st.header("Explainability")

        if not st.session_state.optimization_run:
            st.info(
                "Run an optimization first. The Decision Tree and LLM explanation "
                "will be refreshed for that recommendation."
            )
        else:
            st.caption(
                "Decision Tree metrics describe this interpretable surrogate on "
                "the latest GA policy history; they do not establish clinical validity."
            )
            if not st.session_state.xai_refreshed:
                st.warning(
                    "Explainability was not refreshed for this optimization run; "
                    "saved XAI artifacts are not being treated as current."
                )
            xai_metrics = (
                load_json(XAI_DIR / "tree_metrics.json")
                if st.session_state.xai_refreshed
                else {}
            )
            xai_summary = (
                load_json(XAI_DIR / "best_policy_explanation.json")
                if st.session_state.xai_refreshed
                else {}
            )

            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Accuracy", fmt_pct(xai_metrics.get("accuracy"), 1))
            x2.metric("Precision", fmt_pct(xai_metrics.get("precision"), 1))
            x3.metric("Recall", fmt_pct(xai_metrics.get("recall"), 1))
            x4.metric("F1", fmt_pct(xai_metrics.get("f1"), 1))

            classification = xai_summary.get(
                "decision_tree_class",
                "unknown",
            )
            probability = xai_summary.get(
                "decision_tree_probability_acceptable"
            )

            c1, c2 = st.columns(2)
            c1.metric("Policy Classification", str(classification).upper())
            c2.metric(
                "Acceptable Probability",
                fmt_pct(probability, 1),
                help=(
                    "Probability assigned by the shallow Decision Tree surrogate, "
                    "not a clinical probability."
                ),
            )

            limits = xai_summary.get("acceptability_definition", {})
            st.caption(
                "Current GA acceptability targets: mean wait <= "
                f"{fmt_minutes(limits.get('mean_waiting_time_min_max'))} and "
                "high-risk wait <= "
                f"{fmt_minutes(limits.get('high_risk_mean_waiting_time_min_max'))}."
            )

            st.subheader("Decision Path")
            rules = xai_summary.get("decision_path_rules", [])

            if rules:
                rule_html = '<div class="rule-flow">'
                for idx, rule in enumerate(rules, 1):
                    rule_html += (
                        '<div class="rule-step">'
                        f'<span class="rule-num">{idx}</span>'
                        f'{html.escape(str(rule))}'
                        '</div>'
                    )
                rule_html += (
                    '<div class="rule-step">'
                    '<span class="rule-num">OK</span>'
                    f'{html.escape(str(classification).upper())}'
                    '</div></div>'
                )
                st.markdown(rule_html, unsafe_allow_html=True)
            else:
                st.write("No decision path is available.")

            with st.expander("Full Decision Tree rules"):
                st.code(
                    (
                        load_text(
                            XAI_DIR / "decision_tree_rules.txt",
                            "No rules available.",
                        )
                        if st.session_state.xai_refreshed
                        else "No current rules are available for this run."
                    )
                )

    if st.session_state.optimization_run:
        with st.container(border=True):
            st.header("AI Explanation")
            st.caption(
                "Operational explanation generated from simulation and optimization "
                "results. It is not medical advice."
            )

            st.subheader("AI Summary")
            ai_summary = build_ai_summary(
                best_policy,
                best_metrics,
                xai_summary,
                interpreted_feedback,
                BASE_BOUNDS,
            )
            st.markdown("\n".join(f"- {item}" for item in ai_summary))

            with st.expander("Detailed AI Explanation", expanded=False):
                if st.session_state.llm_generated:
                    llm_text = load_text(
                        LLM_DIR / "llm_explanation.txt",
                        "No LLM explanation was generated.",
                    )
                    st.markdown(llm_text)
                else:
                    st.info(
                        "No current LLM explanation is available for this run. "
                        "The saved explanation from an earlier run is not shown."
                    )


# ============================================================
# REVIEW
# ============================================================

with review_tab:
    with st.container(border=True):
        st.header("Human-in-the-Loop Review")

        if not st.session_state.optimization_run:
            st.info(
                "No recommendation is available for review in this session. "
                "Configure constraints in the sidebar, run optimization, then "
                "return here to record a decision."
            )
        else:
            st.write(
                "Review the recommendation. Accepted or modified decisions remain "
                "human-controlled; the system only provides simulation-based support."
            )
            st.caption(
                "Workflow: review the policy -> record a decision -> optionally "
                "prepare modified values -> rerun optimization."
            )
            if st.session_state.modification_prepared:
                st.info(
                    "Recommended resource values are loaded in the sidebar. "
                    "Adjust them or add an instruction, then select Run Optimization."
                )

            decision = st.radio(
                "Decision",
                ["Accept", "Needs Modification", "Reject"],
                horizontal=True,
                key="review_decision",
            )
            rating = st.slider(
                "Usefulness",
                1,
                5,
                4,
                key="review_rating",
                help="Reviewer assessment of how useful the recommendation is.",
            )
            review_comment = st.text_area(
                "Reviewer comment",
                placeholder=(
                    "Example: Reduce nurses to 7 and keep high-risk waiting "
                    "below 7 minutes."
                ),
                key="review_comment",
            )

            button_left, button_right = st.columns(2)

            with button_left:
                save_review = st.button(
                    "Save Review",
                    width="stretch",
                )

            with button_right:
                st.button(
                    "Prepare Modification",
                    width="stretch",
                    on_click=use_recommended_values,
                    help=(
                        "Loads the recommended resource values into the sidebar. "
                        "Edit them or add a new instruction, then rerun."
                    ),
                )

            if save_review:
                st.session_state.review_status = {
                    "Accept": "Accepted",
                    "Needs Modification": "Needs Modification",
                    "Reject": "Rejected",
                }[decision]

                review = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "decision": decision,
                    "rating": int(rating),
                    "comments": review_comment,
                    "current_policy": best_policy,
                    "current_fitness": best_fitness,
                }

                history_path = FEEDBACK_DIR / "feedback_history.csv"

                row = pd.DataFrame(
                    [
                        {
                            "timestamp": review["timestamp"],
                            "decision": decision,
                            "rating": int(rating),
                            "comments": review_comment,
                            "icu_beds": best_policy.get("icu_beds"),
                            "general_beds": best_policy.get("general_beds"),
                            "doctors": best_policy.get("doctors"),
                            "nurses": best_policy.get("nurses"),
                            "fitness": (
                                round(float(best_fitness), 2)
                                if best_fitness is not None
                                else None
                            ),
                        }
                    ]
                )

                if history_path.exists():
                    previous = pd.read_csv(history_path)
                    row = pd.concat(
                        [previous, row],
                        ignore_index=True,
                    )

                row.to_csv(history_path, index=False)
                st.success("Review saved.")

            history = load_csv(FEEDBACK_DIR / "feedback_history.csv")
            if not history.empty:
                st.subheader("Previous Reviews")
                numeric_cols = history.select_dtypes(include="number").columns
                history[numeric_cols] = history[numeric_cols].round(2)
                st.dataframe(
                    history.tail(10).iloc[::-1],
                    width="stretch",
                    hide_index=True,
                )


# ============================================================
# Developer details
# ============================================================

if st.session_state.run_log:
    with st.expander("Developer Details"):
        st.caption(
            "Technical execution output. This section is not required for normal clinical use."
        )
        st.markdown("#### Pipeline Log")
        st.code(st.session_state.run_log)
        st.markdown("#### Diagnostics")
        st.json(
            {
                "baseline_valid": baseline_valid,
                "optimization_run": st.session_state.optimization_run,
                "latest_successful_optimization": st.session_state.last_run_time,
                "xai_refreshed_for_current_run": st.session_state.xai_refreshed,
                "llm_generated_for_current_run": st.session_state.llm_generated,
            }
        )
        if interpreted_feedback:
            st.markdown("#### Interpreted Feedback")
            st.json(interpreted_feedback)

st.caption(
    "Academic/research prototype. Outputs are simulation-based decision support "
    "and are not autonomous clinical recommendations."
)

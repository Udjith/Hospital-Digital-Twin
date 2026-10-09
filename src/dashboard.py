
from __future__ import annotations

import html
import json
import os
import subprocess
import sys
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from scenario_evaluation import (
    DEFAULT_V2_RESOURCES, ScenarioConfig, HospitalPolicy, load_patient_profiles, evaluate_current_policy,
    save_current_policy_results,
)

from genetic_algorithm_v2 import (
    automatic_bounds, canonical_hash, GENES, GAConfig, run_ga_v2, save_ga_v2,
    validate_bounds, result_is_current, dependency_fingerprints,
)
from decision_tree_xai_v2 import train_xai_v2, save_xai_v2, xai_is_current
from live_dashboard import render_live_twin
from feedback_interpreter import interpret_feedback_v2, apply_feedback_v2, prepare_modification_v2, save_review_v2
from llm_explanation import generate_explanation_v2, explanation_is_current_v2

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
GA_V2_DIR = PROJECT_ROOT / "results" / "genetic_algorithm_v2"
XAI_V2_DIR = PROJECT_ROOT / "results" / "decision_tree_xai_v2"
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
@st.cache_data
def load_source_data(path, modified):
    return load_csv(path)

synthetic_df = load_source_data(DATA_SYNTHETIC, DATA_SYNTHETIC.stat().st_mtime_ns if DATA_SYNTHETIC.exists() else 0)
rf_metrics = load_json(RF_DIR / "metrics.json")
rf_importance = load_csv(RF_DIR / "feature_importance.csv")

# The GA and sidebar share the same baseline-relative search envelope.
BASE_BOUNDS = derive_ga_bounds(baseline_policy)


# ============================================================
# Session state + synchronized controls
# ============================================================

CONTROL_DEFAULTS = {
    **DEFAULT_V2_RESOURCES,
    "mean_wait": 20.0,
    "high_risk_wait": 10.0,
}

RESOURCE_SLIDER_RANGES = {
    "icu_beds": (1, 500),
    "general_beds": (1, 1000),
    "doctors": (1, 200),
    "nurses": (1, 500),
}

# Migrate untouched initial Part 1 defaults once; preserve customized policies.
if st.session_state.get("v2_resource_defaults_version") != 2:
    initial_part1_values = (30, 40, 10, 20)
    previous_values = tuple(st.session_state.get(f"{name}_number", old_value)
                            for name, old_value in zip(DEFAULT_V2_RESOURCES, initial_part1_values))
    if previous_values in (initial_part1_values, (1, 40, 10, 20)):
        for name, value in DEFAULT_V2_RESOURCES.items():
            st.session_state[f"{name}_number"] = value
            st.session_state[f"{name}_slider"] = value
    st.session_state.v2_resource_defaults_version = 2

for control, default in CONTROL_DEFAULTS.items():
    st.session_state.setdefault(f"{control}_slider", default)
    st.session_state.setdefault(f"{control}_number", default)

for resource in ["icu_beds", "general_beds", "doctors", "nurses"]:
    low, high = 1, 10000
    current = st.session_state.get(
        f"{resource}_number",
        CONTROL_DEFAULTS[resource],
    )
    bounded = max(low, min(high, int(current)))
    slider_low, slider_high = RESOURCE_SLIDER_RANGES[resource]
    st.session_state[f"{resource}_slider"] = max(slider_low, min(slider_high, bounded))
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

def sync_from_slider(name):
    st.session_state[f"{name}_number"] = st.session_state[f"{name}_slider"]


def sync_from_number(name):
    value = st.session_state[f"{name}_number"]
    if name in RESOURCE_SLIDER_RANGES:
        low, high = RESOURCE_SLIDER_RANGES[name]
        value = max(low, min(high, value))
    st.session_state[f"{name}_slider"] = value


def set_control(name, value):
    st.session_state[f"{name}_slider"] = value
    st.session_state[f"{name}_number"] = value


def reset_hospital_resources():
    for name, value in DEFAULT_V2_RESOURCES.items():
        set_control(name, value)


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
    if not st.session_state.get("v2_ga_current", False):
        return
    result = st.session_state.get("v2_ga_result", {})
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

def render_v2_comparison(result, details=False):
    current, recommended = result["current_evaluation"], result["verified_evaluation"]
    before, after = current["resource_configuration"], recommended["resource_configuration"]
    st.dataframe(pd.DataFrame([{"Resource": key.replace("_", " ").title(),
        "Current": before[key], "Recommended": after[key], "Change": after[key] - before[key]}
        for key in GENES]), hide_index=True, width="stretch")
    overall, worst = st.columns(2)
    overall.metric("Overall Robustness: Current -> Recommended",
        f"{current['overall_robustness']:.1f}% -> {recommended['overall_robustness']:.1f}%",
        delta=f"{recommended['overall_robustness'] - current['overall_robustness']:+.1f} percentage points")
    old_worst, new_worst = current["scenarios"]["worst"]["scenario_robustness"], recommended["scenarios"]["worst"]["scenario_robustness"]
    worst.metric("Worst Robustness: Current -> Recommended", f"{old_worst:.1f}% -> {new_worst:.1f}%",
        delta=f"{new_worst - old_worst:+.1f} percentage points")
    st.write(f"Overall policy status: {current['overall_policy_status']} -> {recommended['overall_policy_status']}")
    if details:
        for col, name in zip(st.columns(3), ("best", "average", "worst")):
            with col:
                old, new = current["scenarios"][name], recommended["scenarios"][name]
                st.subheader(f"{name.title()} Case")
                st.write(f"Robustness: {old['scenario_robustness']:.1f}% -> {new['scenario_robustness']:.1f}%")
                st.caption(f"Current: {old['scenario_verdict']}; Recommended: {new['scenario_verdict']}")
                rows = []
                for label, key in [("Mean wait (min)", "mean_wait"), ("High-risk wait (min)", "high_risk_mean_wait"),
                    ("P95 (min)", "p95_wait"), ("Unfinished patients", "unfinished_patients"),
                    ("ICU utilization", "icu_utilization"), ("General utilization", "general_utilization"),
                    ("Doctor utilization", "doctor_utilization"), ("Nurse utilization", "nurse_utilization"),
                    ("Throughput (/hour)", "throughput")]:
                    a, b = old["metrics"][key]["mean"], new["metrics"][key]["mean"]
                    rows.append({"Metric": label, "Current": f"{a:.1%}" if key.endswith("utilization") else f"{a:.2f}",
                                 "Recommended": f"{b:.1%}" if key.endswith("utilization") else f"{b:.2f}"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with st.expander("Fully verified policy metrics and condition breakdowns"):
            st.json({"current": current, "recommended": recommended})


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
            max_value=10000,
            step=1,
            key=f"{name}_number",
            on_change=sync_from_number,
            args=(name,),
            help=help_text,
        )
    if st.session_state[f"{name}_number"] > high:
        st.caption(f"{label}: using direct entry {st.session_state[f'{name}_number']:,}; slider range is {low}–{high}.")
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
            help=help_text,
        )
    return float(st.session_state[f"{name}_number"])


with st.sidebar:
    st.markdown("### Hospital Simulation V2")
    sidebar_section("DEMAND CONFIGURATION")
    duration_hours = st.number_input("Simulation Duration (hours)", 1.0, 72.0, 24.0,
        help="Number of simulated hospital hours. The Digital Twin evaluates arrivals, queues, treatment, and resource utilization only within this time window.")
    best_rate = st.number_input("Best Arrival Rate (patients/hour)", 0.0, 1000.0, 12.0,
        help="Average patient arrival rate for the low-demand scenario, measured in patients per hour. Actual arrival times vary stochastically.")
    average_rate = st.number_input("Average Arrival Rate (patients/hour)", 0.0, 1000.0, 22.0,
        help="Average patient arrival rate for the normal-demand scenario, measured in patients per hour. Actual arrival times vary stochastically.")
    worst_rate = st.number_input("Worst Arrival Rate (patients/hour)", 0.0, 1000.0, 35.0,
        help="Average patient arrival rate for the peak-demand scenario, measured in patients per hour. This should be greater than or equal to the average-case rate.")
    replications = st.number_input("Replications Per Scenario", 3, 30, 10,
        help="Number of independent stochastic simulation runs performed for each demand scenario. More replications give more reliable robustness estimates but increase runtime.")

    sidebar_section("HOSPITAL RESOURCES")
    st.caption("Current available capacities; doctors and nurses are simultaneous modeled staff.")
    max_icu = int_control("ICU Beds", "icu_beds", *RESOURCE_SLIDER_RANGES["icu_beds"],
        "Number of ICU beds currently available to the simulated hospital.")
    max_general = int_control("General Beds", "general_beds", *RESOURCE_SLIDER_RANGES["general_beds"],
        "Number of general hospital beds currently available.")
    max_doctors = int_control("Concurrent Doctors", "doctors", *RESOURCE_SLIDER_RANGES["doctors"],
        "Maximum number of doctors that can be simultaneously represented as available resources in the simulation. This is not the total number of hospital employees.")
    max_nurses = int_control("Concurrent Nurses", "nurses", *RESOURCE_SLIDER_RANGES["nurses"],
        "Maximum number of nurses that can be simultaneously represented as available resources in the simulation. This is not the total number of hospital employees.")
    st.button("Reset Hospital Resources", on_click=reset_hospital_resources, width="stretch")

    sidebar_section("Operational Targets")

    mean_wait_target = float_control(
        "Mean wait target (min)",
        "mean_wait",
        1.0,
        120.0,
        1.0,
        "Maximum acceptable average patient waiting time. A simulation run fails this condition if its mean wait exceeds this value.",
    )
    high_risk_wait_target = float_control(
        "High-risk wait target (min)",
        "high_risk_wait",
        1.0,
        120.0,
        1.0,
        "Maximum acceptable average waiting time for patients classified as high risk by the Random Forest model.",
    )

    utilization_target = st.number_input("Maximum Utilization (%)", 0.0, 100.0, 90.0,
        help="Maximum acceptable utilization percentage for critical simulated resources such as beds, doctors, and nurses.")
    robustness_threshold = st.number_input("Robustness Threshold (%)", 0.0, 100.0, 90.0,
        help="Minimum percentage of stochastic simulation runs that must satisfy all operational targets for a scenario to be classified as acceptable.")
    resource_labels = {"icu_beds": "ICU Beds", "general_beds": "General Beds",
                       "doctors": "Concurrent Doctors", "nurses": "Concurrent Nurses"}
    ga_bounds = automatic_bounds(HospitalPolicy(max_icu, max_general, max_doctors, max_nurses))
    sidebar_section("OPTIMIZATION")
    ga_population = st.number_input("GA Population", 2, 100, 12, key="ga_population",
        help="Number of candidate resource configurations evaluated in each Genetic Algorithm generation.")
    ga_generations = st.number_input("GA Generations", 1, 200, 10, key="ga_generations",
        help="Maximum number of evolutionary optimization cycles.")
    ga_replications = st.number_input("GA Replications Per Scenario", 1, 10, 3,
        help="Number of stochastic runs per demand scenario used during GA search. Final recommendations are verified using the full simulation replication count.")
    st.checkbox("Prioritize high-risk patients", key="prioritize_high_risk",
        help="Increases the optimization penalty for excessive waiting among patients classified as high risk.")
    st.checkbox("Prioritize resource efficiency", key="prioritize_efficiency",
        help="Increases the penalty for larger resource configurations when comparing otherwise similar policies.")
    with st.expander("Scenario Fitness Weights"):
        best_weight = st.number_input("Best Fitness Weight", .1, 10., 1.,
            help="Relative contribution of Best demand to GA fitness; weights are normalized internally.")
        average_weight = st.number_input("Average Fitness Weight", .1, 10., 2.,
            help="Relative contribution of Average demand to GA fitness.")
        worst_weight = st.number_input("Worst Fitness Weight", .1, 10., 3.,
            help="Relative contribution of Worst demand to GA fitness. Higher values emphasize peak-demand performance.")
    instruction = st.text_area("Natural-Language Instruction", key="natural_instruction",
        help="Optional instruction that can be interpreted into supported GA constraints or optimization priorities.")
    ai_interpretation = st.toggle("AI Interpretation", key="ai_interpretation",
        help="When enabled, Groq interprets the natural-language instruction into validated optimization constraints. It does not change the authoritative simulation verdict.")
    interpret_clicked = st.button("Preview AI Interpretation", width="stretch", disabled=not (ai_interpretation and instruction.strip()))
    sidebar_section("ADVANCED")
    random_seed = st.number_input("Random Seed", 0, 2147483647, 42,
        help="Controls reproducibility of stochastic simulations. Each replication automatically uses a different derived seed, so the base seed does not need to be changed between runs. Keeping the same seed reproduces the same experiment; changing it creates a new set of random arrival patterns.")
    ga_seed = st.number_input("GA Random Seed", 0, 2147483647, 42,
        help="Controls reproducible population generation, selection, crossover and mutation. The Advanced Random Seed separately controls the shared stochastic demand experiment.")
    with st.expander("Automatic GA Bounds"):
        st.caption("Derived from current resources: floor(70%) to ceil(150%), minimum 1. Explicit availability constraints may override these default ranges.")
        st.dataframe(pd.DataFrame([{"Resource": resource_labels[k], "Minimum": v[0], "Maximum": v[1]} for k, v in ga_bounds.items()]), hide_index=True)
    evaluate_clicked = st.button("Evaluate Current Policy", type="primary", width="stretch")
    sidebar_section("Optimization Actions")
    with st.container(key="optimization_actions"):
        recommendation_action = st.empty()
        run_clicked = st.button("Run Optimization", type="primary", width="stretch")


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

status_panel = st.empty()

if not baseline_valid:
    with st.expander("Legacy GA baseline status"):
        st.info("V1 optimization requires a valid legacy baseline. V2 current-policy evaluation is available independently. Use python src/digital_twin.py --legacy-v1 to regenerate the legacy baseline.")
        for baseline_error in baseline_errors:
            st.write(f"- {baseline_error}")

current_config = ScenarioConfig(
    simulation_duration_hours=float(duration_hours),
    best_case_arrival_rate=float(best_rate), average_case_arrival_rate=float(average_rate),
    worst_case_arrival_rate=float(worst_rate), replications_per_scenario=int(replications),
    mean_wait_target_minutes=float(mean_wait_target), high_risk_wait_target_minutes=float(high_risk_wait_target),
    max_utilization_target=float(utilization_target) / 100,
    robustness_threshold=float(robustness_threshold), random_seed=int(random_seed),
)
current_policy = HospitalPolicy(max_icu, max_general, max_doctors, max_nurses)
configuration_error = None
try:
    current_config.validate()
except ValueError as exc:
    configuration_error = str(exc)
    st.error(configuration_error)

if evaluate_clicked and configuration_error is None:
    try:
        with st.spinner("Evaluating Best, Average and Worst demand..."):
            profiles = load_patient_profiles(DATA_SYNTHETIC, RF_MODEL)
            evaluation = evaluate_current_policy(profiles, current_policy, current_config)
            save_current_policy_results(evaluation, DT_DIR, DATA_SYNTHETIC, RF_MODEL)
            st.session_state.current_policy_evaluation = evaluation
    except (ValueError, OSError) as exc:
        st.error(f"Current policy evaluation failed: {exc}")


# V2 configuration validation, provenance and optimization.
ga_options = GAConfig(population_size=int(ga_population), generations=int(ga_generations),
    ga_replications_per_scenario=int(ga_replications), seed=int(ga_seed),
    prioritize_high_risk=st.session_state.prioritize_high_risk,
    prioritize_efficiency=st.session_state.prioritize_efficiency,
    scenario_weights=(float(best_weight), float(average_weight), float(worst_weight)))

base_config, base_options = current_config, ga_options
auto_bounds = dict(ga_bounds)
feedback_context = {"enabled": bool(ai_interpretation), "instruction": instruction if ai_interpretation else ""}
feedback_key = canonical_hash({**feedback_context, "availability_semantics": 2, "bounds": auto_bounds,
    "configuration": asdict(base_config), "ga_options": asdict(base_options)})
feedback_record = st.session_state.get("v2_feedback_record", {})
feedback_matches = feedback_record.get("key") == feedback_key
if ai_interpretation and instruction.strip() and (interpret_clicked or (run_clicked and not feedback_matches)):
    with st.spinner("Interpreting supported feedback constraints..."):
        feedback_record = {"key": feedback_key, **interpret_feedback_v2(instruction, auto_bounds, base_config)}
    st.session_state.v2_feedback_record = feedback_record
    feedback_matches = True
feedback_error = None
interpreted_feedback = {}
if ai_interpretation and instruction.strip():
    if feedback_matches:
        if feedback_record.get("available"):
            interpreted_feedback = feedback_record["interpretation"]
            try:
                ga_bounds, current_config, ga_options = apply_feedback_v2(
                    interpreted_feedback, auto_bounds, current_policy, base_config, base_options)
            except ValueError as exc:
                feedback_error = str(exc)
        else:
            st.info(feedback_record["message"])
    else:
        st.info("Preview AI Interpretation before optimizing, or Run Optimization to interpret and validate the instruction first.")
availability_override = bool(interpreted_feedback and not feedback_error and any(
    value is not None for limits in interpreted_feedback["resource_constraints"].values() for value in limits.values()))
current_policy_feasible = all(ga_bounds[k][0] <= getattr(current_policy, k) <= ga_bounds[k][1] for k in GENES)
if interpreted_feedback:
    with st.container(border=True):
        st.subheader("Applied AI Interpretation" if not feedback_error else "AI Interpretation - Validation Error")
        st.write(interpreted_feedback["summary"])
        for gene, limits in interpreted_feedback["resource_constraints"].items():
            if any(value is not None for value in limits.values()):
                st.write(resource_labels[gene])
                st.write(f"Automatic range: {auto_bounds[gene][0]}–{auto_bounds[gene][1]}")
                for edge, value in limits.items():
                    if value is not None:
                        st.write(f"User availability constraint: {edge}imum {value}")
                if not feedback_error:
                    st.write(f"Effective GA range: {ga_bounds[gene][0]}–{ga_bounds[gene][1]}")
                st.write(f"Current/reference policy: {getattr(current_policy, gene)}")
                if not feedback_error and ga_bounds[gene] != auto_bounds[gene]:
                    st.caption("Explicit availability overrides the automatic default range.")
        if not feedback_error and not current_policy_feasible:
            st.warning("Current policy is outside the newly stated availability. It remains an unchanged historical/reference comparison, not a feasible operating recommendation. Optimization will search only feasible configurations, starting with the nearest feasible version of the current policy.")
        for priority, value in interpreted_feedback["priorities"].items():
            if value is not None:
                st.write(f"{priority.replace('_', ' ').title()}: {'enabled' if value else 'disabled'}")
        for target, value in interpreted_feedback["target_adjustments"].items():
            if value is not None:
                st.write(f"{target.replace('_', ' ').title()}: {value:g}")
        if feedback_error:
            st.error(feedback_error)
        else:
            st.caption("These effective constraints, priorities and any explicit target adjustments apply to the next GA run and its full verification.")
feedback_context["interpretation"] = interpreted_feedback
feedback_context["available"] = feedback_record.get("available") if feedback_matches and ai_interpretation and instruction.strip() else None

@st.cache_data
def cached_v2_dependencies(signatures):
    return dependency_fingerprints(DATA_SYNTHETIC, RF_MODEL)

optimization_error = configuration_error or feedback_error
try:
    ga_options.validate()
    validate_bounds(ga_bounds, current_policy, require_current_feasible=not availability_override)
except ValueError as exc:
    optimization_error = str(exc)
    st.error(f"GA configuration: {exc}")
try:
    dependency_paths = [DATA_SYNTHETIC, RF_MODEL] + [SRC_DIR / name for name in
        ("digital_twin.py", "scenario_evaluation.py", "genetic_algorithm.py", "genetic_algorithm_v2.py", "decision_tree_xai_v2.py", "feedback_interpreter.py", "llm_explanation.py")]
    dependency_signatures = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in dependency_paths)
    v2_dependencies = cached_v2_dependencies(dependency_signatures)
except OSError as exc:
    v2_dependencies = {}
    optimization_error = f"Required optimization dependency unavailable: {exc}"

v2_dependencies = {**v2_dependencies, "automatic_bounds": auto_bounds, "bound_strategy": "floor(0.70*current), ceil(1.50*current), minimum 1", "feedback": feedback_context}

stored_ga = st.session_state.get("v2_ga_result") or load_json(GA_V2_DIR / "best_policy.json")
ga_stale = bool(stored_ga) and not result_is_current(stored_ga, current_policy, current_config, ga_bounds, ga_options, v2_dependencies, allow_infeasible_current=availability_override)
ga_result = stored_ga if stored_ga and not ga_stale and not optimization_error else {}
if ga_stale:
    st.warning("Previous GA recommendation and XAI explanation are stale because the configuration or a dependency changed. Run optimization to refresh them.")
st.session_state.optimization_run = bool(ga_result)
st.session_state.v2_ga_current = bool(ga_result)
if ga_result:
    st.session_state.v2_ga_result = ga_result

if run_clicked and optimization_error:
    st.error(f"Optimization cannot run: {optimization_error}")
elif run_clicked:
    try:
        with st.status("Running V2 optimization and full final verification...", expanded=True) as optimization_status:
            progress = st.progress(0.)
            def report_generation(record):
                progress.progress((record["generation"] + 1) / ga_options.generations,
                    text=f"Generation {record['generation']} | best fitness {record['best_fitness']:.2f} | search robustness {record['overall_robustness']:.1f}%")
            profiles = load_patient_profiles(DATA_SYNTHETIC, RF_MODEL)
            ga_result = run_ga_v2(profiles, current_policy, ga_bounds, current_config, ga_options,
                progress_callback=report_generation, dependencies=v2_dependencies, allow_infeasible_current=availability_override)
            save_ga_v2(ga_result, GA_V2_DIR)
            st.session_state.v2_ga_result = ga_result
            if current_policy_feasible:
                st.session_state.current_policy_evaluation = ga_result["current_evaluation"]
            st.session_state.optimization_run = True
            st.session_state.v2_ga_current = True
            st.session_state.review_status = None
            st.session_state.modification_prepared = False
            st.session_state.last_run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ga_stale = False
            st.write("Training the Decision Tree explanation from evaluated policy/scenario observations")
            try:
                xai_report, tree, training_data = train_xai_v2(ga_result, seed=ga_options.seed)
                save_xai_v2(xai_report, tree, training_data, XAI_V2_DIR)
                st.session_state.v2_xai_report = xai_report
            except (ValueError, OSError) as exc:
                st.session_state.v2_xai_report = {}
                st.warning(f"Verified recommendation is available, but XAI could not be refreshed: {exc}")
            progress.progress(1., text="Search and full verification complete")
            optimization_status.update(label="V2 optimization complete", state="complete")
    except (ValueError, OSError) as exc:
        st.error(f"Optimization failed: {exc}")

best_policy = ga_result.get("best_policy", {})
best_metrics = ga_result.get("verified_evaluation", {})
best_fitness = ga_result.get("best_fitness")

xai_report = st.session_state.get("v2_xai_report") or load_json(XAI_V2_DIR / "explanation.json")
xai_current = bool(ga_result) and xai_is_current(xai_report, ga_result)
st.session_state.xai_refreshed = bool(xai_current and xai_report.get("status") == "available")
active_xai = xai_report if xai_current else {}
llm_path = PROJECT_ROOT / "results" / "llm_v2" / "explanation.json"
ai_result = st.session_state.get("v2_llm_result") or load_json(llm_path)
ai_stale = bool(ai_result) and (not ga_result or not explanation_is_current_v2(ai_result, ga_result, active_xai))
if ga_result and run_clicked:
    with st.spinner("Preparing grounded explanation..."):
        ai_result = generate_explanation_v2(ga_result, active_xai)
    st.session_state.v2_llm_result = ai_result
    llm_path.parent.mkdir(parents=True, exist_ok=True)
    llm_path.write_text(json.dumps(ai_result, indent=2), encoding="utf-8")
    ai_stale = False
st.session_state.llm_generated = bool(ga_result and ai_result and not ai_stale and ai_result.get("available"))


with recommendation_action.container():
    st.button("Use Recommended Values", width="stretch", on_click=use_recommended_values,
        disabled=not bool(ga_result))

if ga_stale:
    system_status = "Recommendation stale"
elif ga_result:
    system_status = st.session_state.review_status or "Verified recommendation ready"
else:
    status_evaluation = st.session_state.get("current_policy_evaluation", {})
    status_evaluation_matches = (status_evaluation.get("configuration") == asdict(current_config)
        and status_evaluation.get("resource_configuration") == asdict(current_policy))
    system_status = status_evaluation.get("overall_policy_status", "Current policy not evaluated") if status_evaluation_matches else "Current policy not evaluated"
status_run = st.session_state.last_run_time or ("Saved verified run" if ga_result else "Not run")
status_scenario = "Best / Average / Worst"
status_data = f"{duration_hours:g} hours / {replications} runs per scenario" if not synthetic_df.empty else "Data unavailable"
status_panel.markdown(
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


# ============================================================
# Main navigation
# ============================================================

live_tab, dashboard_tab, optimize_tab, review_tab = st.tabs(
    ["Live Twin", "Dashboard", "Optimize", "Review"]
)

with live_tab:
    render_live_twin(DATA_SYNTHETIC, RF_MODEL, current_config)


# ============================================================
# DASHBOARD
# ============================================================

with dashboard_tab:

    with st.container(border=True):
        st.header("Current Hospital Policy")
        cols = st.columns(4)
        for col, label, value in zip(cols, ["ICU Beds", "General Beds", "Concurrent Doctors", "Concurrent Nurses"],
                                     [max_icu, max_general, max_doctors, max_nurses]):
            col.metric(label, value)
        evaluation = st.session_state.get("current_policy_evaluation")
        if evaluation:
            from dataclasses import asdict
            stale = (evaluation["configuration"] != asdict(current_config) or
                     evaluation["resource_configuration"] != asdict(current_policy))
            if stale:
                st.warning("Settings changed. Results below belong to the previous configuration; evaluate again to refresh.")
            overall, status = st.columns(2)
            overall.metric("Overall Robustness", f"{evaluation['overall_robustness']:.1f}%")
            status.metric("Overall Policy Status", evaluation["overall_policy_status"])
            for col, (name, scenario) in zip(st.columns(3), evaluation["scenarios"].items()):
                with col:
                    st.subheader(f"{name.upper()} CASE")
                    st.write(f"Arrival rate: {scenario['arrival_rate']:g}/hour")
                    st.metric("Robustness", f"{scenario['scenario_robustness']:.1f}%")
                    metrics = scenario["metrics"]
                    for label, key in [("Mean wait", "mean_wait"), ("High-risk wait", "high_risk_mean_wait"), ("P95 wait", "p95_wait")]:
                        st.write(f"{label}: {metrics[key]['mean']:.2f} min")
                    st.write("Utilization: " + "; ".join(f"{resource.title()} {metrics[resource + '_utilization']['mean']:.1%}"
                                                        for resource in ["icu", "general", "doctor", "nurse"]))
                    st.write(scenario["scenario_verdict"])
            st.caption("Wait includes elapsed waiting for patients still queued at window end. Utilization uses only the observation window. Scenario statistics average per-run metrics; throughput is completions/hour. No future waiting or discharge tail is counted.")
            with st.expander("Per-run results and configuration"):
                st.json(evaluation)
        else:
            st.info("Select Evaluate Current Policy to test the entered resources under all three demand scenarios. GA is not required.")

    with st.container(border=True):
        st.header("Current Policy vs Recommended Policy")
        if ga_result:
            render_v2_comparison(ga_result, details=False)
        else:
            st.info("Run V2 optimization to compare the current hospital policy with a fully verified recommendation.")

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
        st.header("V2 Optimization")
        st.caption("GA optimizes hospital resource policies through the Digital Twin. It does not optimize the Random Forest; the RF threshold stays at 0.50.")
        if not ga_result:
            st.info("Enter current hospital resources and select Run Optimization. Search bounds update automatically. Generation 0 includes the current policy when feasible, or its nearest feasible version after an availability override.")
            with st.expander("Effective GA Search Bounds"):
                st.dataframe(pd.DataFrame([{"Resource": resource_labels[k], "Minimum": ga_bounds[k][0],
                    "Current": getattr(current_policy, k), "Maximum": ga_bounds[k][1]} for k in GENES]), hide_index=True, width="stretch")
        else:
            fitness_col, generation_col, runtime_col = st.columns(3)
            fitness_col.metric("GA Search Fitness", f"{best_fitness:.2f}",
                help="Search fitness uses shared demand seeds and the smaller GA replication count. Performance below comes from full final verification.")
            generation_col.metric("Generations Completed", ga_result["generations_completed"])
            runtime_col.metric("Optimization Runtime", f"{ga_result['runtime_seconds']:.2f} s")
            st.caption(f"Early stopping: {ga_result['early_stopping_occurred']}. Final verification: {ga_result['final_verification_replications']} replications per scenario. Unique candidates: {ga_result['unique_candidate_evaluations']}; cached comparisons: {ga_result['cache_hits']}.")
            render_v2_comparison(ga_result, details=True)
            generation_data = pd.DataFrame(ga_result["generations"])[["generation", "best_fitness"]].rename(columns={"best_fitness": "fitness"})
            st.altair_chart(monochrome_line_chart(generation_data, "generation", "fitness", "Best Search Fitness by Generation", dark_mode), width="stretch")
            with st.expander("Applied bounds, targets, weights and provenance"):
                st.json(ga_result["provenance"])

    if ga_result:
        with st.container(border=True):
            st.header("Why This Changed")
            st.caption("Deterministic before/after observations from full verification; no LLM interpretation is used.")
            for note in ga_result["change_summary"]:
                st.write(note)

    with st.container(border=True):
        st.header("Decision Tree XAI Explanation")
        st.caption("Learned explanatory patterns from GA-tested scenarios. Authoritative verdicts come from Digital Twin targets and robustness rules.")
        if not ga_result:
            st.info("Run optimization to generate a current explanation.")
        elif not xai_current:
            st.warning("The Decision Tree explanation is unavailable or stale for this recommendation.")
        elif xai_report["status"] != "available":
            st.info(xai_report["message"])
            st.json(xai_report["metrics"])
        else:
            metrics = xai_report["metrics"]
            cols = st.columns(4)
            cols[0].metric("Training Samples", metrics["training_sample_count"])
            cols[1].metric("Tree Depth / Leaves", f"{metrics['tree_depth']} / {metrics['leaf_count']}")
            cols[2].metric("Training Accuracy", f"{metrics['training_accuracy']:.1%}")
            cols[3].metric("Policy-held-out Accuracy", f"{metrics['validation_accuracy']:.1%}" if metrics["validation_accuracy"] is not None else "Unavailable")
            st.caption(f"Class distribution: {metrics['class_distribution']}. Training accuracy measures fit to the observed GA data and does not prove generalization. {metrics['validation_note']}")
            for col, (name, explanation) in zip(st.columns(3), xai_report["explanations"].items()):
                with col:
                    st.subheader(f"{name.title()} Case")
                    st.write(f"AUTHORITATIVE VERDICT: {explanation['authoritative_verdict']}")
                    st.write(f"XAI learned classification: {explanation['surrogate_class']}")
                    st.caption(f"Surrogate acceptable probability: {explanation['surrogate_probability_acceptable']:.1%}")
                    st.code(" AND\n".join(explanation["learned_rules"]) or "Root leaf: no split rule")
            with st.expander("Full learned tree rules"):
                st.code(xai_report["rules_text"])
    with st.container(border=True):
        st.subheader("AI Summary")
        st.caption("AUTHORITATIVE VERDICT: Digital Twin operational criteria. OPTIMIZATION: GA. XAI: learned Decision Tree surrogate. AI EXPLANATION: optional natural-language interpretation.")
        if ai_stale:
            st.warning("Previous AI explanation is stale. Run optimization or regenerate the explanation for the current verified recommendation.")
        if ga_result:
            if st.button("Generate / Refresh AI Explanation"):
                with st.spinner("Preparing grounded explanation..."):
                    ai_result = generate_explanation_v2(ga_result, active_xai)
                st.session_state.v2_llm_result = ai_result
                llm_path.parent.mkdir(parents=True, exist_ok=True)
                llm_path.write_text(json.dumps(ai_result, indent=2), encoding="utf-8")
                ai_stale = False
            if not ai_result or ai_stale:
                from llm_explanation import deterministic_explanation_v2
                summary, detail, _ = deterministic_explanation_v2(ga_result, active_xai)
                st.caption("Deterministic summary of the current verified recommendation.")
            else:
                summary, detail = ai_result["summary"], ai_result["detail"]
                st.info(ai_result["message"])
            for line in summary:
                st.write(line)
            with st.expander("Detailed AI Explanation", expanded=False):
                st.write(detail)
        else:
            st.info("Run optimization to see a summary grounded in verified V2 results.")


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
                "prepare feedback constraints -> rerun optimization."
            )
            if st.session_state.modification_prepared:
                st.info(
                    "Review comments are loaded as the next optimization instruction. Preview AI Interpretation, then select Run Optimization."
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
                    "Example: Do not use more than 120 doctors. Prioritize high-risk patients."
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
                    on_click=lambda: prepare_modification_v2(st.session_state),
                    disabled=decision != "Needs Modification" or not review_comment.strip(),
                    help=(
                        "Transfers review comments into the next optimization instruction and enables AI interpretation. Historical results stay unchanged."
                    ),
                )

            if save_review:
                st.session_state.review_status = {
                    "Accept": "Accepted",
                    "Needs Modification": "Needs Modification",
                    "Reject": "Rejected",
                }[decision]

                try:
                    save_review_v2(ga_result, decision, rating, review_comment, FEEDBACK_DIR / "v2_reviews")
                except (ValueError, OSError):
                    st.error("Review could not be saved. The verified recommendation remains available.")
                else:
                    st.success("Review saved.")

            history = pd.DataFrame([{key: record.get(key) for key in ("timestamp", "ga_run_id", "decision", "rating", "comments", "verified_status")}
                for path in sorted((FEEDBACK_DIR / "v2_reviews").glob("review_*.json"))
                if (record := load_json(path))])
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

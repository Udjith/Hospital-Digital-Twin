# Intelligent Hospital Digital Twin

## Overview

A live, stateful, adaptive hospital **operational simulation prototype** for clinician/administrator decision support. It starts with a seeded operating hospital, predicts stress-event consequences from current state, searches scheduling policies with fixed physical resources, and keeps humans in control.

This is not real EHR integration, clinical decision automation or a validated medical protocol. The authoritative current specification is [Final V3 Architecture](docs/FINAL_V3_ARCHITECTURE.md).

The live dashboard leads with hospital status, scenario forecasts, policy comparisons and human decisions. A sidebar navigation rail links the six main sections; detailed evidence stays available in collapsed panels. See [Dashboard presentation and validation](docs/V3_DASHBOARD_DESIGN.md) and [Navigation and card cleanup](docs/V3_UI_CLEANUP.md).

## Final Architecture

Warm Start ? continuous Live Digital Twin ? manual / automatic / natural-language stress event ? current-state Look-Ahead ? sustained-pressure deterministic diagnosis ? fixed-capacity adaptive GA ? Decision Tree XAI ? Groq explanation ? human Apply / Reject / Modify ? continued operation / temporary policy recovery.

Digital Twin metrics and deterministic rules determine the verdict. The tree explains learned patterns; the LLM communicates validated evidence. Neither overrides the verdict.

## Quick Start

```powershell
python -m pip install -r requirements.txt
python main.py
```

Optional `.env` configuration: `GROQ_API_KEY=your_key`. Existing process variables override `.env`. Without Groq, manual event entry, simulation, optimization, XAI and deterministic explanations continue working.

Required runtime assets are `data/synthetic/synthetic_hospital.csv` and `models/random_forest_pipeline.joblib`. For a fresh checkout install Git LFS and retrieve tracked assets with `git lfs pull`; do not replace them with pointer text.

## Live Digital Twin Workflow

1. Set fixed physical capacity and Warm Start settings; Start Live Simulation.
2. Advance normal operation, manually or using bounded automatic playback.
3. Describe a Hospital Scenario, inspect interpretation/assumptions, and Run Look-Ahead. The Advanced / Manual Event Builder remains available.
4. For AT RISK or CANNOT HANDLE, run Adaptive Optimization and review predicted current vs optimized outcomes.
5. Inspect the main diagnosis and explanation, with engineering evidence in expanders.
6. Apply, Reject, or Modify / Retry. Event application is a separate confirmation. Continue operation and observe temporary policy recovery.

Speeds: 1/5/10/30/60/120/300/600/1200/3600x. Warm Start + Start enables automatic playback; Reset stops it. Fast Forward supports up to30 simulated days without intermediate dashboard rerenders, retaining exact event ordering and integration boundaries. Internal times stay in minutes; displays use readable durations/timestamps.

## Models

- **Random Forest:** cached patient risk prediction, fixed threshold0.50; not optimized by GA.
- **Genetic Algorithm:** current-checkpoint operating-policy search: high-risk priority, wait aging, ICU/doctor/nurse reserve and surge priority. Physical resource counts never change.
- **Decision Tree XAI:** surrogate trained on actual GA-tested observations with deterministic handling labels; explanatory correlation only.
- **Groq LLM:** `openai/gpt-oss-120b` for strict scenario interpretation and validated-result explanation, with nonfatal deterministic/manual fallbacks.

## Supported Stress Events

Patient surge; temporary arrival-rate spike; doctor shortage; nurse shortage; ICU bed outage; General-bed outage. Compound events support explicit delays. Safe temporary outages do not interrupt assigned resources or evict patients. Random synthetic events are optional/off by default.

## Warm Start

Demonstration baseline: **220 ICU /550 General /80 doctors /140 nurses**, **4 patients/hour**. Warm Start defaults create a reproducible existing operating state from real processed/augmented profiles. Normal Demo Mode may enable Low random events. This baseline is a project demonstration, not clinical validation. See [warm-start implementation](docs/V3_WARM_START.md).

## Human-in-the-Loop

No automatic GA application. Human review supports successful, partially improved and unchanged recommendations. Apply affects future queue/allocation decisions only; Reject preserves live policy. Temporary event policy can restore automatically after recovery when chosen. Physical capacity changes require explicit reset/state-event controls.

Default handling targets: mean wait?20min, high-risk mean?10min, time-weighted resource utilization?90%, continuous overload?30min, full saturation?15min, and recovered queue. Brief utilization peaks are diagnostic warnings. Predictive Robustness is handled verified replications /total ?100; default threshold90%. Statuses are HANDLED /AT RISK /CANNOT HANDLE.

## Dataset / Model Pipeline

Original eICU sources, processed datasets and trained model files are preserved. `preprocessing.py`, `synthetic_data.py`, and `random_forest.py` retain the offline preparation/augmentation/training pipeline. Runtime profile/model caches avoid repeated inference. Every patient acquires exactly one ICU OR General bed, one doctor AND one nurse; staff release after treatment, bed after discharge.

## Testing

```powershell
python -m pip install pytest
python -m pytest -q -p no:cacheprovider
```

`tests/test_live_*.py` covers final V3 state, events, interpretation, pressure, GA, diagnosis and initialization. Shared-runtime and retained V2 tests remain required regressions. Compact final evidence is in `results/live_twin/`: warm start, sustained pressure, scenario lock, diagnosis and native browser validation. Phase2 audit results are in `docs/cleanup_phase2/`.

## Historical Implementations

**V1** preserves fixed-cohort experiments and original CLI modules. **V2** preserves repeated Best/Average/Worst scenario experiments and resource-count optimization, XAI, feedback and review. These are historical implementations, not the final live optimizer. Open them only through the collapsed **Legacy V2 Experiments** sidebar entry. Their source, persisted recommendation, models and tests remain intact.

V3_PART1/PART2/PART3, diagnosis and scenario-lock documents record development history; consult [Final V3 Architecture](docs/FINAL_V3_ARCHITECTURE.md) for current behavior. Phase1 safety tag: `v3-final-pre-cleanup`.

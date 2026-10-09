# Final V3 Architecture

This is the authoritative specification of the current **live/adaptive simulated Hospital Digital Twin prototype**. Earlier phase documents are implementation history. This is not production EHR integration, clinical decision automation, or a validated clinical triage protocol.

## 1. Objective

Support clinician/administrator review of operational stress scenarios in an already operating simulated hospital. Predict consequences, search scheduling policies, explain verified evidence, and preserve human control.

## 2. Architecture

`main.py` launches `src/dashboard.py`, the Live Digital Twin environment. `live_dashboard.py` composes the live, scenario and adaptive UI. The default page contains no V2 resource-count optimizer. A collapsed **Legacy V2 Experiments** sidebar entry explicitly opens `legacy_dashboard_v2.py` for historical scenario experiments, optimization and review.

Warm Start ? continuous live hospital state ? manual/automatic/natural-language events ? cloned current-state Look-Ahead ? sustained-pressure deterministic diagnosis ? fixed-capacity Adaptive Optimization ? Decision Tree XAI ? Groq explanation ? human Apply / Reject / Modify ? continued operation / recovery.

`live_hospital_state.py` owns the state; `live_initialization.py` creates incumbents; `live_events.py` and `live_event_context.py` represent disturbances; `live_lookahead.py` and `live_pressure.py` evaluate consequences. `live_policy_ga.py`, `live_policy_diagnosis.py`, `live_policy_explanation.py` (including the tree surrogate) perform search, diagnosis and communication. `live_canonical.py` supplies canonical JSON hashing. Shared profile loading resides in `scenario_evaluation.py`.

## 3. Warm Start

Dashboard Warm Start is enabled by default. Demonstration capacities are **220 ICU beds, 550 General beds, 80 doctors, 140 nurses**, with **4 patients/hour**. Initial occupancy targets are ICU 55%, General 25%, doctors 20%, nurses 20%; Auto queue is zero when all resource types have free capacity. This seeded initialization uses existing RF-profile pools and schedules residual treatment/discharge in the existing engine. Actual staff occupancy respects paired doctor/nurse requirements. Incumbents have synthetic IDs and historical arrival times; subsequent live arrivals are counted separately. Initialization is reproducible, not a measured or calibrated steady state. Reset can reproduce Warm Start or explicitly initialize empty.

## 4. RF risk prediction

The existing cached Random Forest pipeline predicts patient risk from augmented clinical profiles, with threshold **0.50**. It is neither retrained nor optimized by the live GA. Data/model loading and predictions are reused; sampling does not modify source datasets. High-risk overrides in simulated surges sample existing predicted-risk subsets rather than falsifying clinical features or changing the threshold.

## 5. Patient and resource lifecycle

Each patient waits for exactly one bed type (ICU for high-risk routing, otherwise General), one doctor **and** one nurse. Staff remain assigned throughout treatment; the selected bed remains occupied until discharge. Patients transition through waiting, treatment, bed stay and completion. Stable IDs identify ICU/General beds, doctors and nurses; synthetic session IDs identify patients. Assignments are never retroactively reshuffled by policy application.

Persistent session objects retain the clock, queue, event heap, RNGs, ownership, metrics and operating policy across Streamlit reruns. All internal times remain simulated minutes; UI timestamps/durations are human-readable. Playback speeds are **1, 5, 10, 30, 60, 120, 300, 600, 1200, 3600x**. Bounded advancement retains catch-up debt and processes every scheduled event. Warm Start + Start enables automatic playback at the selected speed; Reset stops it and Reset Empty + Start remains manual. Fast Forward supports **43,200 minutes (30 days)** in one unthrottled execution, updating only a progress indicator. It reuses the event-to-event scheduler and retains internal60-minute observation boundaries for bit-for-bit cumulative/rolling-state equality; there is no per-boundary dashboard rerender. Debt is committed before progress callbacks, so rerun interruptions can resume or cancel without replay. See [UX validation](V3_UX_POLISH.md) for timings and equivalence evidence.

## 6. Stress events and scenario entry

Supported types: `PATIENT_SURGE`, `ARRIVAL_RATE_SPIKE`, `DOCTOR_SHORTAGE`, `NURSE_SHORTAGE`, `ICU_BED_OUTAGE`, `GENERAL_BED_OUTAGE`. Manual entry remains available in an expander. Optional seeded random synthetic events are off by default and scheduled in simulated time; Low frequency has a 360-minute mean interval. Normal Demo Mode uses the baseline above and Low events; this is a project demonstration, not clinical validation.

Describe a Hospital Scenario uses the existing Groq infrastructure (`openai/gpt-oss-120b`) solely to interpret text. Strict provider JSON is decoded and normalized before deterministic event-specific, timing and capacity validation. Event-level fields are `event_type`, `duration_minutes`, `start_delay_minutes`; parameters are `count`, `arrival_rate`, `high_risk_proportion` as applicable. Explicit null irrelevant placeholders are removed; non-null unsupported fields fail. Assumptions/warnings are shown, missing outage duration requests clarification, and omitted surge spread uses the disclosed 20-minute default. Compound events share the checkpoint time plus explicit delays. Interpretation, Edit and Cancel do not change hospital state; Look-Ahead and Apply Events are separate human actions. Without Groq, manual events still work.

Outages preserve physical totals and never evict patients or remove assigned staff. Busy selected resources become unavailable on release; temporary effects expire safely. Overlapping arrival spikes use the highest active rate, then restore the base rate.

## 7. Current-state sustained-pressure Look-Ahead

Every future replication starts from a deep, independent checkpoint of the current hospital, including incumbents, queues, ownership, pending events and RNG state. No real clock, RNG or assignments are mutated. Default horizon is 120 minutes, with 5 predictive replications. Metrics distinguish arrivals, completions, waiting, treatment and bed stay; valid ongoing treatment is not a recovery failure.

Resource integration is event-driven: area accumulates `utilization_fraction ? elapsed_minutes`, divided by observation duration for the time-weighted mean. Usable capacity accounts for temporary outages. Peak, total time above target, longest overload, total full saturation and longest full saturation remain available as evidence.

## 8. Deterministic diagnosis

Verified condition breakdowns determine blocking conditions, observed values, targets, failed replication counts and bottleneck categories. Categories include staff, bed, multiple-resource, demand, waiting/process-policy and recovery limitations. Secondary improvements remain visible even when robustness does not recover. Canonical plain JSON with sorted keys and SHA-256 protects stale results across reloads; no pickle metadata determines provenance. Changes to checkpoint, event, targets or search settings mark results stale.

## 9. Adaptive GA

The GA searches operating policy from the same live checkpoint with **fixed physical capacity**. Six discretized genes:

| Gene | Range | Step | Normal default |
| --- | --- | --- | --- |
| high_risk_priority_weight | 0?5 | 0.1 | 1 |
| waiting_time_aging_weight | 0?2 | 0.1 | 0.1 |
| icu_reserve_percentage | 0?0.30 | 0.01 | 0 |
| doctor_reserve_percentage | 0?0.25 | 0.01 | 0 |
| nurse_reserve_percentage | 0?0.25 | 0.01 | 0 |
| surge_priority_strength | 0?5 | 0.1 | 0 |

Queue ranking combines existing clinical priority, high-risk/event priority and elapsed-wait aging. Stress reserves use usable capacity, never preempt ownership, and permit eligible priority/aged patients. ICU routing never permits a General patient to consume an ICU bed. These controls are operational simulation scheduling, not clinical triage.

Common future random numbers are shared across candidates. Search uses fewer replications (default 3); final verification uses the selected full Look-Ahead count on tested candidates. Fitness rewards robustness and penalizes failed conditions, wait violations, sustained pressure, queue growth/non-recovery and unnecessarily restrictive policy. The exact unchanged executable formula is `live_policy_ga.live_fitness`; sustained-pressure development details remain in `V3_LIVE_ACCEPTANCE.md`.

Risky event previews recommend optimization; the user starts search. The best verified policy is returned as **VERIFIED SUCCESS**, **PARTIAL IMPROVEMENT** or **NO MEANINGFUL IMPROVEMENT**. Finite heuristic search is not a mathematical proof of infeasibility.

## 10. Decision Tree XAI

A small surrogate learns from actually evaluated policy/replication observations and deterministic handled/failed labels. It reports real paths, class distribution, training samples and accuracy limitations. One-class data yields an unavailable explanation rather than fabricated rules. Learned correlation cannot override the Digital Twin.

## 11. Groq interpretation and explanation

Both scenario interpretation and result explanation use **Groq / openai/gpt-oss-120b**, with distinct purposes in the existing infrastructure. `.env` is supported; process environment overrides `.env`. Keys are never displayed. Result explanation communicates validated before/after evidence, deterministic diagnosis and XAI limitations. Missing key, provider failure or invalid output yields a deterministic fallback. The LLM cannot decide a verdict, apply events or change policy/capacity.

## 12. Human review

**Apply Optimized Operating Policy**, **Reject Recommendation**, and **Modify / Retry** remain explicit actions, including for partial improvements. Reject changes no live policy. Modify revises experiment/event/target/search settings and reruns evaluation. Apply affects future scheduling only. Temporary event policies restore the previous normal policy after existing event recovery; persistent mode restores only on explicit human action. Policy history and scenario audit are bounded.

## 13. Fixed physical-capacity principle

Live physical ICU, General, doctor and nurse totals are user inputs, locked until Reset. GA never increases/decreases them. Explicit temporary outages change usable capacity, not physical totals. Historical V2 resource-count search remains isolated and does not update live capacity.

## 14. Final acceptance logic

For each predictive run, `HANDLED_RUN` requires mean wait ?20 minutes and high-risk mean wait ?10 minutes by default; each resource must satisfy time-weighted utilization ?90%, longest continuous >target interval ?30 minutes, and longest continuous full saturation ?15 minutes. These targets/grace periods are configurable. A short 91?100% peak is a warning, never an automatic failure.

Recovery acceptance requires end queue ? checkpoint queue +2 and no unresolved event-specific waiting patients. Ongoing treatment/bed stay alone does not fail. Historical recovery-time estimation additionally monitors expired effects and return near the pre-event queue; if not observed it reports no recovery within horizon.

Predictive Robustness = `handled runs / verified runs ?100`. `HANDLED` when robustness reaches the configured threshold (default90%). Otherwise `CANNOT HANDLE` when robustness is zero, or ?50% with mean/high-risk wait more than twice its target; remaining outcomes are `AT RISK`. These are deterministic operational experiment results.

## 15. Final demo workflow

Start Warm Start at baseline capacity ? advance normal operation ? describe a scenario ? interpret and review structured events/assumptions ? run Look-Ahead ? inspect verdict/main reason ? optimize a risky event ? review current/optimized outcomes, diagnosis, XAI and AI Explanation ? Reject or Modify / Retry, or Apply policy ? separately confirm event application ? continue clock ? observe recovery/restoration. No historical V2 tab is needed. Detailed pressure, replications, search, tree paths and provenance remain collapsed evidence.

## 16. Limitations

Synthetic operational stress feed; no EHR/emergency system integration. Patient profiles, treatment/LOS assumptions and seeded initialization are project abstractions. Finite horizons and small replication sets limit inference. Policies are not validated medical protocols or staffing instructions. Clinicians/administrators remain the final decision makers. Raw data, trained RF artifacts, historical implementations and regression tests remain preserved for reproducibility.

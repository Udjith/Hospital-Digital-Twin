# Reproducible warm-start initialization

This initializes a **simulated initial operating state**, not measured hospital occupancy, clinical advice, or a statistically calibrated steady-state hospital. Physical capacity, RF threshold 0.50, patient routing, resource release, event semantics, sustained-pressure acceptance, policy GA chromosome/fitness, diagnosis, XAI, Groq and human approval remain unchanged.

## Architecture and controls

`live_initialization.py` provides immutable `WarmStartSettings` and `initialize_hospital`. The state engine first resets its ordinary counters, resources, queues, RNG streams and policy, then initializes incumbents. All residual events enter the existing heap and use the same treatment-completion/discharge handlers. There is no separate patient schema or simulator, no dataset writes, and no extra RF inference.

The normal dashboard enables warm start. Controls are collapsed under **Advanced Live Twin Initialization Settings** and lock while a session is active. Defaults:

| Setting | Default |
|---|---:|
| Enable warm start | On |
| ICU occupancy target | 55% |
| General occupancy target | 25% |
| Doctor busy target | 20% |
| Nurse busy target | 20% |
| Initial queue | Auto |
| Reset initialization | Reset with Warm Start |

Targets are validated fractions in [0,1]. Queue can be Auto or 0–5. Auto creates zero queued patients when all four resource types have free capacity; otherwise it uses at most three (`min(3, max(1, occupied_beds // 100))`). Explicit queue patients have seeded ages of 0–15 minutes and own no resources initially. They enter the normal dispatcher on the first advance, including a zero-minute advance used by look-ahead.

**Reset Empty** clears patients, queue, ownership, events and clock, retaining entered settings for a later warm reset. **Reset with Warm Start** uses the selected initialization settings. Capacity/feed/seed edits require reset and Start to create the newly configured session. Legacy low-level constructor calls that omit `warm_start` retain empty initialization for compatibility; new callers opt in with `warm_start=WarmStartSettings()`. `reset()` preserves the instance's initialization configuration; `reset(warm_start=False/True)` provides explicit empty/warm API resets.

A session retained across a code reload from before this feature requires an explicit **Reset Legacy Live Session**, preserving input controls. It is never silently replaced or used as a partially initialized warm session.

## Seed, routing and occupancy

Initialization uses `numpy.random.default_rng(numpy.random.SeedSequence([live_seed, 4004]))`. It does not consume the future arrival, profile, stress or random-event generators. Same seed, capacity, settings, profile order/checksum and normal policy reproduce the same initial state and hash. Future arrivals remain ordinary exponential arrivals with mean `60 / patients_per_hour` minutes.

Bed targets use nearest-integer half-up rounding: `floor(capacity * target + 0.5)`. ICU profiles are sampled from existing RF-high-risk records; General profiles from existing lower-risk records. Profiles are resampled with replacement. Treatment and LOS retain the existing V2 normalization and bounds. No source features or RF probabilities are altered, and only synthetic `P000001`-style patient IDs are visible.

Requested staff busy counts use the same rounding. The actual number of treatment patients is `min(requested_doctors, requested_nurses, occupied_beds)`. That many admitted patients are selected without replacement. Every one holds one bed, one doctor and one nurse. Other admitted patients hold a bed only. Unequal requested staff targets produce an explicit warning rather than fictitious staff assignments. Missing required risk-profile pools or BED_STAY profiles with LOS greater than treatment yield a clear initialization error.

## Residual lifecycle and timestamps

For a treatment incumbent, seeded elapsed admission age is `uniform(0.05, 0.95) * treatment_minutes`.

For a bed-stay incumbent, elapsed age is `treatment_minutes + uniform(0.05, 0.95) * (stay_minutes - treatment_minutes)`.

Arrival/admission and treatment start are `-elapsed_age`, reflecting admission before the observed session. Residual discharge is `stay_minutes - elapsed_age`. Treatment incumbents schedule residual staff release at `treatment_minutes - elapsed_age`; bed-stay incumbents already completed treatment and do not schedule another staff release. Ages and residual deadlines remain numeric minutes. The visible clock starts at Day 1 00:00:00; historical timestamps display **Before start: … earlier**.

Patients also record `initialized_patient`, `pre_start_age_minutes`, `initial_elapsed_treatment` and `initial_elapsed_los`. Initial admission wait is assumed zero; queued incumbents instead carry their sampled queue age. Seeded continuous ages stagger releases and avoid a synchronized fresh-LOS discharge wave.

## Accounting, checkpoint and provenance

`total_patients_arrived` counts only arrivals during the live observation, including injected event arrivals. Initial incumbents do not increase it. `completed_patients` and throughput count actual discharges observed during the session, including incumbents. New metrics split those completions into `initial_patients_completed` and `live_arrivals_completed`, and expose initial/remaining incumbent counts.

The accounting invariant is:

`initial_active_patients + live_arrivals = observed_completions + current_active_patients`.

Cumulative/rolling live-arrival waiting metrics exclude historical incumbent waiting. Queued incumbents retain their real elapsed wait and are included in the existing look-ahead waiting cohort. Resource utilization integrates actual occupied time after observation begins. Valid treatment/bed stay is never mistaken for a newly arrived queue.

`initial_state` preserves targets, actual initial occupancy, queue, lifecycle counts, seed strategy, warnings and profile checksum. Snapshots add the complete canonical `live_state_hash`. The existing canonical hash already includes every new patient field, settings, initial-state metadata and residual agenda; no hashing design changes were needed. Look-ahead and GA provenance explicitly copy the compact warm-start metadata alongside their current-state hash. Changing settings or live state invalidates predictions through existing stale checks.

Checkpoints clone the complete initialized state, ownership, queues, residual events and RNG. They never call the initializer. GA candidates therefore see identical incumbents plus common future seeds. Safe outages preserve assigned patients and pending unavailable resources exactly as before.

## Real-profile A–F validation

Baseline: 220 ICU / 550 General / 80 doctors / 140 nurses, 4 arrivals/hour, seed 42. Evidence: `results/live_twin/warm_start_validation.json`.

| Case | Observed result |
|---|---|
| A — initialization | 121 ICU, 138 General; 16 doctors and 16 nurses busy; 259 active (16 treatment, 243 bed stay), queue 0; live arrivals 0 |
| B — 24 hours | 102 live arrivals; 160 observed completions (148 incumbents + 12 live arrivals); 201 active; queue 0; mean/high-risk wait 0 |
| C — +12 patients over 20 minutes | HANDLED, 100% predictive robustness, zero wait; all 12 event patients applied through the ordinary lifecycle |
| D — +60 patients over 25 minutes plus 70-doctor shortage for 120 minutes | Current and optimized CANNOT HANDLE, 0%; finite search classified partial improvement from high-risk wait 81.52 → 80.21 minutes, but mean wait worsened 85.13 → 113.96 minutes and optimized recovery was not reached; capacities stayed fixed |
| E — bus accident +40 over 20 minutes | Existing parser with mocked raw provider JSON, current occupied checkpoint: HANDLED, 100%; no live-state mutation |
| F — reset | Exact original initialization hash reproduced; 259 incumbents, zero live arrivals |

Initial discharge deadlines were all distinct (259), ranging from 3.38 to 17,436.74 minutes. Ownership/accounting invariants were checked after initialization, advancement, manual events and reset. Separate tests cover safe outages, automatic events, queued incumbents, clones, GA and the actual Streamlit controls. The severe result is reported with its adverse tradeoffs; no GA or acceptance changes were made to improve the demonstration outcome.

All 169 tests passed, including the unchanged 155 prior regression tests and 14 new warm-start tests. The native-browser harness launched `python main.py` and passed event preview/apply, policy optimization/apply, stale protection, high-speed pause and reset. The older dashboard validator also passed with explicit Reset Empty selected for its empty-state assertions; cached RF inference occurred once across two starts.

Measured initialization including cached-profile normalization: approximately 0.33–0.36 seconds. A–F validation took approximately 7–10 seconds, including a 5.6–7.7-second small GA search. The final full suite took 45.2 seconds while running alongside browser/demo validation. Evidence is compact JSON; no checkpoint dumps, model copies, dataset copies or screenshots are created. The initializer is a configurable demonstration occupancy model, not an estimate of actual hospital utilization or residual-LOS distributions.

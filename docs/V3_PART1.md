# V3 Part 1: persistent simulated Live Twin

This is a **simulated live hospital feed, not a production EHR/hospital integration**. V2 scenario evaluation, resource GA, deterministic verdicts, Decision Tree XAI, Groq explanations, feedback, provenance and tests remain available and unchanged. V3 adds a separate stateful mode; it does not run GA, adapt capacity or trigger surge reoptimization.

## Files and architecture

- `src/live_hospital_state.py`: `LiveCapacity`, `OperatingPolicy`, `Resource`, `Patient`, `LiveHospitalState` and `LiveClockDriver`.
- `src/live_dashboard.py`: native Streamlit fragment, controls, cards, bounded tables/events and compact snapshot download.
- `src/dashboard.py`: imports the new view and adds **Live Twin** before the preserved **Dashboard**, **Optimize** and **Review** tabs. Dashboard remains the V2 scenario-analysis page.
- `tests/test_live_hospital_state.py`: 17 fast lifecycle/clock/metrics/history tests.
- `tests/validate_live_dashboard.py`: actual cached RF/profile integration, persistent UI state and a real-profile lifecycle demo; one compact JSON evidence file.
- `tests/validate_live_browser.py`: optional real-browser native-fragment playback validation, without screenshots or repository browser caches.
- `README.md` and this document: V3 entry point, assumptions and validation.

The engine uses a deterministic discrete-event heap, free-resource ID heaps and mutable active patient/resource records. It reuses V2 `prepare_profiles` and the existing cached RF profile loader. It contains no GA, RF training, clinical preprocessing or scenario-verdict implementation.

`LiveCapacity` is frozen and validated using the existing resource limits. The current `OperatingPolicy` is a separate frozen object. Individual resource entities are created once at session construction/reset. Active records, counters, rolling history, pending events, RNG state and the operating policy survive Streamlit reruns in `st.session_state.v3_live_hospital`. A separate `v3_clock_driver` adapts wall time to the explicit engine clock.

## Patient and resource lifecycle

The live lifecycle is:

`ARRIVAL -> WAITING_FOR_RESOURCES -> IN_TREATMENT -> BED_STAY -> DISCHARGED`

There are no invented medical stages. Only the next arrival event is scheduled; a patient ID and sampled profile are attached on arrival. A cached RF probability is recorded and classified at the unchanged threshold **0.50**. High-risk patients require ICU, otherwise General. Each feasible patient acquires **exactly one selected bed, one doctor and one nurse atomically**. Waiting patients hold no partial resources.

Doctor and nurse are held for the full normalized treatment duration. At treatment completion both are released, their resource ownership becomes null, and the patient's staff IDs become null. The selected bed remains occupied through the full normalized length of stay. Discharge releases that bed, clears its ownership/patient assignment and increments completed-flow metrics. Freed resources are immediately offered to queued feasible patients. If treatment equals LOS, staff release and discharge occur at the same simulated instant.

Normalization and routing reuse V2: treatment is clipped to 15–720 minutes; bed stay is at least treatment duration and at most 14 days. Explicit existing clinical priority, or V2's RF-derived fallback priority, is retained. Staff are concurrent modeled treatment resources, not total employee counts.

Resource IDs are deterministic and stable during a session: `ICU-001`, `GEN-001`, `DOC-001`, `NUR-001`, etc. Numeric width is a minimum, so IDs also support capacities above 999. Beds use AVAILABLE/OCCUPIED; staff use AVAILABLE/BUSY. Every entity tracks its current synthetic patient ID. Patient IDs are `P000001`, `P000002`, etc.; source eICU identifiers are never retained in the live engine or exposed in tables/events/snapshots.

## Clock and arrivals

The engine clock is elapsed simulated minutes. `advance_simulation(delta_minutes)` processes all scheduled events through the requested instant, then advances idle time to that instant. Each call accepts 0–60 minutes. Equal-time events process staff release before discharge before new arrivals, with a stable sequence tie-breaker. This is a live as-of-now state; V2's fixed observation-window implementation is untouched.

Live demand uses:

```python
arrival_rng = numpy.random.default_rng(seed)
interarrival_minutes = arrival_rng.exponential(60.0 / arrival_rate)
```

The next arrival is scheduled relative to the previous one. Rate zero schedules none. Clinical profiles are resampled with replacement using a separate reproducible profile stream, `default_rng(SeedSequence([seed, 1]))`. Allocation changes and UI refreshes therefore do not alter arrival draws. Same seed/configuration and simulated horizon reproduce the same demand and allocation; different seeds change arrivals. Advancing in smaller partitions produces the same event sequence as a larger equivalent step.

RF inference/data loading use the existing V2 source/model-version cache. Clinical normalization runs once per new live session. Arrivals reuse those normalized profiles and cached probabilities; no RF inference or disk read occurs per arrival/refresh. Neither datasets nor models are mutated or copied.

## Dashboard controls and persistence

Run `python main.py`, then open **Live Twin**. Enter dedicated live ICU/General/doctor/nurse capacities, arrival rate and seed before starting. These are independent of V2 sidebar inputs and V2 GA recommendations. Defaults are **210/280/85/100**, **22 patients/hour**, seed **42**.

Start creates the session engine. Live resource/feed/seed widgets lock until **Reset Live Twin**, including while paused or stopped. The physical totals never change automatically. V2 controls remain usable for independent experiments and cannot change live capacities. Reset stops the session, clears active/completed rows, queues, scheduled events, counters, clock, events and prepared snapshot; all resource entities return to AVAILABLE with the same deterministic IDs. User-entered settings are preserved. Starting after changing settings creates a new session.

Automatic playback defaults **off**; the speed selector defaults to **10x** and offers 1x/5x/10x/30x/60x. A native Streamlit fragment reruns approximately once per second. Playback converts actual elapsed monotonic wall seconds to simulated minutes using the selected multiplier. Speed transitions account for the preceding interval at its preceding speed. The driver accrues elapsed time once, advances at most 60 minutes per update and retains any catch-up debt visibly; throttled refreshes do not drop demand or service events. There is no infinite loop or simulation background thread.

Pause freezes automatic advancement; Resume reanchors wall time so paused time is excluded. Stop freezes state until reset. **Advance Simulation** explicitly advances the chosen 0.1–60 minute step, including while paused for debugging; manual steps are independent of speed. Snapshot preparation and ordinary widget interactions do not recreate the engine. State is persistent per active Streamlit session, not shared between users or durable across server restart/session expiration.

## Visible state and metrics

Four cards report occupied/total beds and busy/total staff, free/available counts and **instantaneous** occupancy percentages. Additional cards report currently waiting, in treatment, bed stay and completed during the session.

The patient table defaults to 50 rows (options 25/50/100/200), includes synthetic ID, RF risk/probability, selected bed type, assigned resource IDs, status, arrival/queue/start times, observed wait, expected staff release/discharge and actual discharge. All numeric times are simulated minutes since start. Optional completed rows fill unused display rows from the capped recent history.

Resource tables are opt-in under an expander, with type/status filters and a 100-row limit. The timeline shows the latest 50 structured events. Each event contains simulated timestamp, type, patient ID, resource IDs and description. In-memory events default to the last **1000** (configurable 1–2000); completed rows default to the last **200**. Completed patients leave the main active mapping. Counters preserve cumulative totals independently of retained rows. No event log or state export is written on refresh.

Cumulative metrics include arrived/completed, current active/waiting/treatment/bed-stay counts, mean observed wait, high-risk mean observed wait and throughput (completed hospital flows per elapsed simulated hour). Served patients retain start-minus-arrival wait; queued patients contribute only now-minus-arrival. Future waiting is never counted. Empty groups return zero.

Current utilization is occupied-or-busy / capacity. Cumulative utilization integrates resource occupancy minutes / (capacity × elapsed minutes). Rolling utilization uses the intersection of those occupancy intervals with the last 60 simulated minutes. Fractions are bounded to [0,1]. Rolling metrics also report arrivals, completions, observed mean wait for the recent arrival cohort, queue length now and time-weighted mean queue length. Before minute 60 they report the shorter observed duration; rolling histories are pruned by simulated time.

## Operating policy and future hooks

The default policy adds no extra weighting to the existing V2 queue order:

| Field | Default | Part 1 role |
|---|---:|---|
| high_risk_priority_weight | 0 | Risk-weight hook |
| waiting_time_aging_weight | 0 | Wait-aging hook |
| icu_reserve_percentage | 0 | Inactive placeholder |
| doctor_reserve_percentage | 0 | Inactive placeholder |
| nurse_reserve_percentage | 0 | Inactive placeholder |
| surge_priority_strength | 0 | Inactive placeholder |

Queue priority is `(existing_clinical_priority - high_risk_weight × high_risk_flag - aging_weight × waiting_minutes, arrival_time, synthetic_patient_id)`, lower first. Feasible patients can proceed when a higher-priority patient needs an unavailable bed type, matching V2. This is an **operational simulation priority mechanism, not a real clinical triage protocol**. Reserve/surge fields must remain zero until implemented; nonzero values are rejected rather than silently ignored. No optimizer changes policy weights or capacities in this part.

## Compact snapshots

`snapshot(patient_limit=50, event_limit=100)` returns schema/version, clock/status, fixed configuration/seed/rate/RF threshold/profile hash, aggregate resources/queue, limited active patient rows with truncation flags, metrics, current policy and bounded recent events. JSON serialization is verified with `allow_nan=False`.

**Prepare Compact Live Snapshot** prepares a 25-patient/50-event JSON download in memory. It is explicitly labeled a historical export until prepared again. It is not a full resumable checkpoint: future look-ahead/restoration will need RNG/pending-event cloning or a complete checkpoint design. Validation writes only two small JSON summaries, not repeated snapshots, datasets, model copies, screenshots or repository cache directories.

## Validation

```powershell
python -m unittest discover -s tests -v
python tests/validate_live_dashboard.py
python tests/validate_v2_dashboard.py
python tests/validate_part3_dashboard.py
# Optional Windows Chrome/websockets browser check:
python tests/validate_live_browser.py
```

All **58 unit tests** passed in **1.41 seconds**: the 41 V2 tests plus 17 V3 tests. They cover IDs/fixed capacities, both bed routes and RF threshold, atomic staff/bed acquisition, queueing, treatment staff release, bed retention/discharge, resource reuse, seeded exponential demand, incremental equivalence, pause/resume/reset, bounded histories, serializable anonymized snapshots, censored waits, integrated/rolling utilization, policy hooks, bounded steps and wall-clock accounting.

Actual Streamlit interaction validation passed: repeated reruns preserved object identity and state; V2 input changes left live capacity untouched; capacity locks, all lifecycle controls, resource inspection, snapshots and reset worked. Two live starts used **one RF inference**, with zero Groq calls. Existing V2 scenario/GA/XAI/feedback/review checks also passed; `python main.py` returned Streamlit HTTP 200.

The actual UI/demo/benchmark validation took **5.57 seconds**. Preparing a new engine from cached real profiles and stepping 24 hours at the default 22/hour took **0.297 seconds**, with **530 arrivals / 51 discharges / 479 active** and an event history held at its **1000-event cap**. This is a single measured prototype run, not a throughput guarantee under arbitrarily large overloaded queues.

The browser check confirmed native automatic fragment updates at 10x: clock 5.00 -> 5.41 minutes over approximately 2.5 wall seconds; Pause held at 5.46, Resume advanced again, Stop kept capacity locked and Reset returned to zero/unlocked. It took **16.0 seconds**, created no screenshots and removed its isolated system-temp browser profile and owned server/browser processes.

### Real-profile lifecycle demo

For clarity the demo selects one existing augmented profile with the shortest bed stay, resampling that actual record without altering it: RF probability **0.393229**, treatment **19.7 minutes**, bed stay **240 minutes**. Capacity is **1 ICU / 1 General / 1 doctor / 1 nurse**, arrivals **12/hour**, seed **42**.

| Simulated minute | Event |
|---:|---|
| 12.021 | P000001 arrives; acquires GEN-001, DOC-001, NUR-001 |
| 23.702 | P000002 arrives; queues with no partial resources |
| 31.721 | DOC-001 and NUR-001 released from P000001; GEN-001 stays occupied |
| 35.626 | P000003 arrives |
| 252.021 | P000001 discharged; GEN-001 released |
| 252.021 | P000002 immediately acquires GEN-001, DOC-001, NUR-001 |

At minute **253.021**, 61 patients had arrived, one completed, one was in treatment and 59 were waiting. This deliberately overloaded miniature hospital demonstrates queues and release/reassignment; it is not a staffing recommendation. Compact evidence is `results/live_twin/validation_summary.json`; browser results are `browser_validation.json`.

## Limits and V3 Part 2

Repository impact is under **0.1 MB**: approximately 75 KB of added source/tests/documentation plus two compact JSON evidence files totaling 4.4 KB, and small README/dashboard edits. No dataset/model copies, screenshots, persistent browser caches or repeated large exports were added. The optional browser test uses and removes an isolated system-temp profile outside the repository.

This prototype assumes cached static clinical profiles, empty initial hospital state and fixed capacities/feed/policy per session. No production data integration, medication/procedure stages, deterioration, abandonment, shift rosters or new clinical triage rules are modeled. The active queue is retained faithfully and can grow under prolonged overload; event/completed display history is capped. Operational use would need validated demand/clinical assumptions, security, persistence and workload limits.

V3 Part 2 can add explicit event injection, surge/look-ahead state handling and richer snapshots. Later process-policy optimization must remain separate from fixed physical capacity, with deterministic criteria still authoritative. No surge-triggered GA, live resource-count changes or adaptive optimization were implemented here.

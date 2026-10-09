# V3 Part 2: stress events and current-state look-ahead

Current live acceptance, revised GA pressure terms and all-candidate verification are documented in [V3_LIVE_ACCEPTANCE.md](V3_LIVE_ACCEPTANCE.md); this supersedes historical peak-only criteria below. State, event and operating-policy mechanics remain unchanged.

This is a **simulated operational stress-event feed, not a production emergency alert system**. V2 scenario analysis, GA, XAI, LLM and review remain available. V3 Part 1's stochastic feed, fixed physical capacities, patient routing and treatment/discharge lifecycle are retained. No adaptive GA or automatic capacity changes are implemented.

## Files and entry points

- `src/live_events.py`: `LiveEvent`, supported types, validation, event application/expiry, independent checkpoints, random scheduling and event-specific counters. `LiveEventSupport` extends the existing state engine.
- `src/live_hospital_state.py`: additive stress-event dispatch, ownership-safe outage release, effective-capacity summaries, source-event patient IDs and snapshot fields.
- `src/live_lookahead.py`: `simulate_lookahead_from_current_state`, `preview_is_current`, deterministic handling verdict and predictive metrics.
- `src/live_event_dashboard.py`: editable presets, proposals, preview/apply separation, active events, random controls and preview audit.
- `src/live_dashboard.py`: integrates event controls after active patients; shows unavailable/pending resources and merges read-only preview audit with the displayed recent timeline. `src/dashboard.py` passes the existing operational targets to this view.
- `tests/test_live_events.py`: 21 additional fast tests. `tests/validate_live_events.py`: real-profile and Streamlit validation, writing one compact JSON summary. `tests/validate_live_event_browser.py` reuses the existing browser harness with an optional event callback; the original browser checks remain unchanged by default.

Run `python main.py` and open **Live Twin**. Start a session, advance to a non-empty state, select/load an editable preset or enter an event, then **Prepare Proposed Event**, **Run Event Look-Ahead**, and **Apply Event**. Previewing alone never applies an event. Pause playback to inspect a preview without it immediately becoming stale. Applying preserves running/paused playback status.

## Event model and lifecycle

Supported types are exactly `PATIENT_SURGE`, `ARRIVAL_RATE_SPIKE`, `DOCTOR_SHORTAGE`, `NURSE_SHORTAGE`, `ICU_BED_OUTAGE`, `GENERAL_BED_OUTAGE`.

Each `LiveEvent` includes a stable `EVT-000001` session ID, type, creation/start time, duration, parameters (magnitude), description, source, status, applied/look-ahead flags, affected IDs and counters. Lifecycle: `PROPOSED -> SCHEDULED -> ACTIVE -> ENDED`. Proposing does not consume the engine's ID counter or mutate its state; application copies the proposal and reserves its ID. Reset clears all stress effects and event state.

Manual events permit a delay and validate finite times, nonnegative spread, positive temporary duration, integer counts, physical totals and overlapping reservations. Maximum duration is 1440 minutes; manual surges contain 1-1000 patients; at most 100 stress events may be active/scheduled. Negative, impossible and unsupported parameters fail clearly. Spike rates are between the base rate and 1000/hour. Sequential outages use half-open intervals; expiring temporary effects restore before another event starts at the same timestamp.

### Surges and clinical profiles

A surge adds exactly N patients, either immediately or at sorted seeded uniform offsets within its spread. Uniform offsets represent a finite incident cohort, separately from the unchanged exponential **background** feed. Surge patients enter the actual queue and normal atomic bed/doctor/nurse acquisition flow. The bed type is still determined by their cached RF probability at threshold **0.50**. Staff release after treatment; the selected bed remains occupied until discharge.

An optional high-risk proportion p selects `floor(N*p + 0.5)` high-risk profiles and the remaining lower-risk profiles, then shuffles that mix with the event RNG. Sampling uses existing RF-classified profile subsets with replacement. Probabilities/features are never edited. A requested class absent from the cached data is rejected. The Traffic Accident preset uses a high-risk proportion 0.20 above the cached population fraction, capped at 1; all values remain editable.

Patient `source_event_id` and per-event counters distinguish incident patients from background arrivals. Event waits include served waits plus elapsed waits of event patients still queued; unfinished counts include treatment and bed-stay patients. Spike-attributable excess patients are not inferred or fabricated.

### Arrival spikes

Physical base rate stays unchanged. Effective rate is `max(base rate, all ACTIVE spike rates)`. On an effective-rate transition, the pending background arrival is replaced by a fresh exponential draw with mean `60/effective_rate` minutes. This uses Poisson memorylessness. Only background arrival scheduling changes; known treatment/discharge and surge events stay intact. Expiry restores the next-highest active override, then the base rate.

### Resource outages

Resources are selected deterministically: unreserved free units first, then busy units, ordered by ID. Free units become `OUT_OF_SERVICE` and leave the free-ID heap. Busy units become `OUT_OF_SERVICE_PENDING`, keep their patient and remain usable for that assignment. Their normal release clears ownership and makes them `OUT_OF_SERVICE`; no patient is interrupted or evicted. Expiry removes the reservation: free affected units return to the heap; occupied affected units resume `BUSY`/`OCCUPIED` until normal release.

Concurrent outages reserve disjoint units. Overlapping demands exceeding the physical total are rejected; random conflicting events are skipped and logged. Capacity summaries distinguish physical `total`, actual `temporarily_unavailable`, `pending_unavailable`, current `usable = total - out_of_service`, planned post-release usable capacity, assigned/busy and free counts. Pending assigned resources remain in current usable capacity until their work finishes, so occupancy does not exceed the denominator. An outage may expire before a long busy assignment releases; in that case it never interrupts or removes that assignment.

Current/peak utilization is assigned units / currently usable units. Zero usable capacity is conservatively recorded as 100% exhausted, even if no patient is currently assigned. Existing cumulative/rolling utilization remains actual occupied resource-minutes divided by physical resource-minutes, with outages excluded from busy-time accounting.

## Random synthetic events

Off by default; opt-in controls specify frequency and allowed supported types. Only one next random-event trigger is scheduled. Mean exponential simulated intervals: Low 360, Medium 180, High 60 minutes. UI reruns with unchanged settings do not schedule or draw anything. Disabling removes future random triggers but does not cancel already applied events.

Seeded synthetic magnitudes: surges 5-30 patients; rate spikes 1.25-2.0 times base (capped at 1000); staff shortages 5-20%; bed outages 5-15%; duration/spread 30-180 minutes. Resource counts use floor with a minimum of one, so a one-unit hospital can lose its only unit. These are demonstration stress inputs, not clinical standards. Presets additionally include ED Surge (1.5x, 90 min), 10% doctor/nurse shortages (120 min) and 10% ICU maintenance (180 min), rounding preset unavailable counts upward.

## Complete checkpoints and independence

`checkpoint()` / `clone()` deep-copy the complete engine in memory. The copy includes the clock, active and recently completed patient objects, queue ordering and shared patient references, resource ownership, free heaps, pending event heap, policies, base/effective-rate state, temporary events, RNGs, patient/event/heap counters, bounded log and cumulative/rolling metric state. Only immutable cached clinical tuples are shared. No data/model reload, disk checkpoint or pickle deserialization occurs.

`state_hash()` hashes an in-memory serialization of the complete mutable state plus profile hash. It is a session provenance fingerprint, not a portable checkpoint format or cryptographic audit attestation. Predicting never advances/mutates the real clock, ownership, heap, counters, logs or RNG. UI look-ahead audit is a separate bounded session deque, merged into the displayed timeline, so it does not mutate the live engine.

## Predictive experiment

Default horizon is 120 minutes (UI options 30/60/120/240/480); default replications 5 (range 1-20). Each replication:

1. Clones the same checkpoint, including known future arrivals and existing stress effects.
2. Derives `seed = SeedSequence([future_base_seed, 3002, replication_index]).generate_state(1)[0]`; default future base seed is the live session seed.
3. Reseeds independent future streams using `SeedSequence([seed, stream])`: 0 background arrivals, 1 profiles, 2 surge scheduling, 3 random events. Known heap entries remain fixed; new future draws vary. A proposed rate spike legitimately replaces its pending background arrival.
4. Applies the identical proposed event to the clone and advances bounded steps at every next event timestamp through the horizon, tracking peak queue and usable-capacity utilization.
5. Reports additional arrivals, completed flows (including incumbent patients), end queue, unfinished patients, observed mean/high-risk/P95 waits, resource peaks, per-run verdict/conditions, event-specific metrics and recovery.

Waiting metrics cover **patients queued at checkpoint time plus new arrivals**, carrying existing elapsed wait. Already-treated incumbents continue consuming capacity and may complete but do not dilute the waiting cohort. Patients still waiting at the horizon contribute elapsed wait only. No drain-to-completion tail is used. Same checkpoint/event/future seed reproduces the experiment; the actual applied event is another valid realization, not necessarily one of the preview replications.

Aggregate output includes means, min/max ranges, all compact per-run results, acceptable count and robustness. Resource utilization values in look-ahead are **peaks**, deliberately stricter than V2's fixed-window average. V2 formulas and scenario behavior are unchanged.

## Exact deterministic verdict and recovery

The reused V2 `policy_verdict` checks mean wait <= mean target, high-risk mean wait <= high-risk target and all four resource peak utilizations <= utilization target. Every condition must pass for a replication to be `ACCEPTABLE`.

`predictive robustness = 100 * acceptable replications / total replications`.

- **HANDLED**: robustness >= configured robustness threshold.
- **CANNOT HANDLE**: otherwise, robustness is zero; OR robustness <=50% and at least one aggregate wait (mean or high-risk mean) exceeds twice its corresponding target.
- **AT RISK**: all other below-threshold outcomes.

HANDLED is checked first (including when a user explicitly sets threshold zero). Neither GA, Decision Tree nor LLM determines this verdict.

Recovery is the first observed event boundary after the proposed event ends where queue length <= checkpoint queue + 2 and none of that event's surge patients remains waiting. Ending an outage restores its own capacity reservation; unrelated pre-existing events may continue. Recovery time is minutes since proposed event start, not since checkpoint. It means queue recovery, not discharge of all surge patients. If absent by the horizon, return null and display **Not recovered within look-ahead horizon**. Aggregate recovery is the mean among recovered replications and always shows the recovered count.

## Provenance, storage and stale previews

Results retain schema/engine version, complete state hash/time, physical and effective capacity, queue, policy, base/effective rate, clinical profile hash, proposed event, horizon, replications, future seed/derivation/seeds, operational targets, per-run metrics and verdicts. A changed clock/state, event form, horizon, replication count or targets marks the displayed prediction **STALE** and disables Apply Event until a current preview exists.

The hospital's existing recent log remains capped (default 1000), UI preview audit at 50, active/scheduled events at 100. The latest 200 ended event records are retained; older records needed by still-active incident patients remain until safe to remove. Tables render limited rows. No per-refresh writes, large history CSVs, browser caches, datasets, model copies or disk checkpoints are created. Validation saves one small Part 2 summary JSON; the optional browser check uses a system-temp profile and no screenshots.

## Validation

`python -m unittest discover -s tests -v`: **79 tests pass**, including all 58 existing V2/V3 Part 1 tests and 21 event tests. Tests cover clone/RNG isolation, reproducibility/variation, exact surge count/routing, spikes/precedence, occupied resource safety, recovery, overlapping/back-to-back outages, conservation during mixed random events, bounded logs, event-specific attribution, serialization and stale results.

`python tests/validate_live_events.py`: Streamlit preset/preview/apply and stale-clock/target protection; no Groq calls. Real cached augmented RF profiles, seed 42, capacities ICU8/General12/Doctors6/Nurses6, rate12/hour, at simulated minute60:

| Current state / prediction | Result |
|---|---:|
| ICU / General / doctor / nurse free | 8 / 6 / 0 / 0 |
| Existing queue | 1 |
| Proposed surge | 12 patients over 20 min |
| Look-ahead | 5 runs, 120 min |
| Mean / high-risk / P95 observed wait | 70.98 / 70.26 / 116.86 min |
| Mean queue peak / end queue | 37.8 / 37.8 |
| Mean ICU / General / doctor / nurse peaks | 35% / 60% / 100% / 100% |
| Predictive robustness / verdict | 0% / CANNOT HANDLE |
| Recovery | Not recovered within horizon |

Preview leaves the real state unchanged and repeats identically. Applying introduces exactly 12 event-tagged patients. At minute77.626 an event patient receives ICU-001/DOC-003/NUR-003. At minute80 a three-doctor shortage preserves all six busy assignments, marks three pending, and expires at minute110 without interrupting treatment. At minute343.226 that event patient's doctor and nurse release normally; its bed remains held. Compact evidence is `results/live_twin/part2_validation.json`. Small-state 5x120 prediction took approximately **0.013 seconds**; complete UI/demo approximately **5.8 seconds**, including initial model/profile load. These measurements are local, not worst-case latency guarantees.

The existing live dashboard validation also passes, preserving session identity, capacity locking, pause/resume/reset, RF cache reuse and no-key operation. Browser validation launches `python main.py`, checks native fragment playback/pause/reset and reports any browser exception; the event browser check also confirms the displayed count defaults to 12, prediction leaves the clock unchanged, and explicit application updates active events. The full browser run took about **18 seconds**; see `results/live_twin/part2_browser_validation.json`. Preserved V2 GA/XAI/feedback/review AppTest also passes (about 12.6 seconds, zero Groq calls without a key).

New source, tests, documentation and Part 2 evidence total approximately **80 KB**, with a few KB of changes to existing files: less than 0.1 MB repository growth. No datasets, models, disk checkpoints or screenshots were copied. Validation-created temporary directories and new bytecode caches were removed.

## Remaining for V3 Part 3

Adaptive operating-policy GA, surge-triggered reoptimization, comparing current/reoptimized process policies and live-specific AI explanations remain future work. Physical resource totals remain fixed. This prototype has no real emergency/EHR integration, patient deterioration, staff shifts, clinical intervention modeling or validated clinical triage. Predictions use the existing simplified resource holding/LOS assumptions and finite sampled replications.

## Live Twin time display

`src/live_time_display.py` provides pure `format_duration`, `format_timestamp` and
`format_time_text` helpers. Patient/event timestamps consistently display
**Day N HH:MM:SS**, with session time zero at **Day 1 00:00:00**. Waits, event
duration/spread/remaining time, look-ahead horizon and recovery display compact
durations. Durations below one hour round to the nearest second; longer durations
round to the nearest minute. Half units round up, carrying into the next unit.
Day durations omit a zero minute component; missing timestamps display an em dash.

| Simulated minutes | Displayed duration |
|---:|---|
| 0 | 0s |
| 0.5 | 30s |
| 12.3 | 12m 18s |
| 59.9 | 59m 54s |
| 60 | 1h 0m |
| 90 | 1h 30m |
| 1439 | 23h 59m |
| 1440 | 1d 0h |
| 4609 | 3d 4h 49m |
| 9859 | 6d 20h 19m |

Formatting applies only to copied table rows and rendered text, including event
descriptions. Stored descriptions, event specifications, metrics, minute values,
RNG/scheduler state, snapshots, provenance and stale-state hashes are unchanged.
Numeric inputs retain explicit minute labels; horizon options also show their
numeric minutes. Raw numeric times remain available in JSON snapshots.

Validation checked all values above plus rounding carry, missing values and the
requested long-wait examples. Streamlit interaction checks confirmed formatted
patient/event/replication tables and wait cards, unchanged state hash and current
preview after rendering, and numeric minute values in snapshot downloads. All
**79 existing tests pass unchanged**; no test or simulation files were modified
for this presentation update.

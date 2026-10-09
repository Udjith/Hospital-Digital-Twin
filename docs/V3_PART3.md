# V3 Part 3: adaptive operating policies with fixed physical capacity

Current live acceptance, revised GA pressure terms and all-candidate verification are documented in [V3_LIVE_ACCEPTANCE.md](V3_LIVE_ACCEPTANCE.md); this supersedes historical peak-only criteria below. State, event and operating-policy mechanics remain unchanged.

V3 Live Twin optimizes an **operating process**, using a checkpoint of the hospital that is already running. It never optimizes ICU/General bed counts or doctor/nurse counts. V2's separate resource GA, scenario experiments, explanations, feedback and review remain available. RF routing remains fixed at probability 0.50; treatment and hospital-stay durations retain the existing model.

**This adaptive policy is a simulated operational scheduling recommendation, not a validated clinical triage protocol.** The live arrival and stress-event feeds are simulated, without production EHR or emergency-alert integration.

## Demonstration baseline

The Live Twin UI initializes to **220 ICU beds, 550 General beds, 80 doctors, 140 nurses, 4 arrivals/hour**, seed 42. Physical inputs lock after Start and require Reset to edit. The explicit normal UI policy is high-risk weight 1.0, waiting-aging weight 0.1, all reserves zero and surge weight zero. Legacy constructors retain their prior defaults for compatibility; new demos pass the baseline explicitly.

Normal Demo Mode uses this capacity/rate with **Low** random-event frequency. Random events still default to **OFF** and must be enabled explicitly. Major disturbances remain manually injectable. Seven simulated days with Low events produced 738 arrivals, 507 completions, 231 active patients, zero current queue and zero cumulative observed wait in the recorded seed-42 experiment. This is demonstration evidence, not clinical validation or a guarantee for other profiles/seeds.

## Architecture and chromosome

`live_policy_control.py` adds scheduling, reserve eligibility, bounded policy history and temporary-policy restoration to the existing live state. `live_policy_ga.py` searches six operating-policy genes using `live_lookahead.py`; it does not duplicate the simulator. `live_skip_control.py` retains manual skip debt. `live_adaptive_dashboard.py` keeps adaptive controls separate from fixed capacity. `live_policy_explanation.py` provides optional learned rules and grounded explanation.

| Gene | Bounds | GA step | Meaning |
| --- | --- | --- | --- |
| `high_risk_priority_weight` | 0–5 | 0.1 | Additional RF-high-risk queue preference |
| `waiting_time_aging_weight` | 0–2 | 0.1 | Preference per minute already waiting |
| `icu_reserve_percentage` | 0–0.30 | 0.01 | Incident ICU reserve; stored as a fraction |
| `doctor_reserve_percentage` | 0–0.25 | 0.01 | Preferential staff reserve |
| `nurse_reserve_percentage` | 0–0.25 | 0.01 | Preferential staff reserve |
| `surge_priority_strength` | 0–5 | 0.1 | Additional incident-patient preference |

The advanced manual editor displays reserves as percentages. The normal UI does not ask users to tune genes. The all-zero `OperatingPolicy()` remains available for legacy behavior and is an explicit search anchor.

## Exact allocation rules

At each allocation opportunity, waiting patients are sorted by:

```text
score = existing_profile_base_priority
        - high_risk_priority_weight * RF_high_risk_flag
        - surge_priority_strength * incident_patient_flag
        - waiting_time_aging_weight * elapsed_wait_minutes
```

Lower score wins; arrival time, then synthetic patient ID, break ties. An incident flag is true for a patient with a source event that is ACTIVE, or an incident patient still waiting during recovery. Positive aging therefore lets a sufficiently old patient outrank a newly arrived patient with bounded priority advantages. It cannot manufacture resources or guarantee service under permanently excessive demand.

Reserve counts are `ceil(current_usable_capacity * reserve_fraction)` and activate only while an event is ACTIVE or incident patients remain queued. Usable capacity excludes resources actually OUT_OF_SERVICE; occupied outage-pending resources remain usable until safely released, as in Part 2.

* Non-priority patients cannot take the last reserved doctors/nurses. RF-high-risk patients, incident patients with positive surge priority, and patients waiting at least **60 minutes** with positive aging weight may borrow staff reserves.
* ICU is already exclusively RF-high-risk; General-routed patients never consume ICU beds. To make the ICU gene meaningful, the explicit operational interpretation is **incident high-risk reservation** during a patient surge: routine high-risk patients cannot take the final reserved slots while an ACTIVE patient surge or queued incident high-risk backlog exists. Incident high-risk patients, and routine high-risk patients aged at least 60 minutes with positive aging weight, may borrow them. A rate spike alone does not artificially change ICU eligibility. This distinction was surfaced during implementation because reserving ICU for all high-risk patients would have no effect under the existing routing.
* A scheduled aging wake-up at arrival +60 minutes permits reserve borrowing even when no new arrival/completion occurs. Already assigned patients are never interrupted. A patient still acquires exactly one routed bed, one doctor and one nurse atomically; staff release after treatment and the bed remains until discharge.
* Reserves do not change resource IDs, totals, ownership or outage behavior. They restrict new allocation only; an existing occupancy above a new reserve target is left intact.

## Triggers, checkpoint evaluation and common random numbers

Optimization runs only after an explicit click on **Optimize Current Operating Policy**, or the prominent **Optimize Operating Policy** button offered after an AT RISK/CANNOT HANDLE event preview. HANDLED previews show that the current policy handles the event. There is no automatic GA execution or application.

Context can be a proposed event, an already applied event, or the current state with existing scheduled/active events. Every candidate starts from the same deep in-memory checkpoint: incumbents, pending arrivals/events, ownership, queue, clock and counters are preserved. Proposed events are injected into clones only. Immutable cached clinical tuples/RF probabilities are shared; models/data are not reloaded during search. Real state, RNG and event agenda remain unchanged.

Default GA settings: population 10, maximum generations 6, search replications 3, patience 3 and seed equal to the live seed. Population 0 contains the exact current policy, an all-zero compatibility policy and a high-risk/staff-reserve boundary policy; remaining candidates are random bounded discrete policies. Tournament selection uses three candidates, crossover selects each gene from either parent, mutation occurs independently with probability 0.25 and shifts up to five steps, and one elite survives. Bounds/steps are enforced after mutation/crossover. Three generations without an improvement greater than 1e-6 trigger early stopping.

Exact seed strategy:

```text
experiment_seed = SeedSequence([live_seed, ga_seed, 3303]).generate_state(1)[0]
replication_seed[i] = SeedSequence([experiment_seed, 3002, i]).generate_state(1)[0]
```

Each future replication uses the existing separate streams 0/1/2/3 for arrivals, profiles, event generation and random events. These **same seeds** are used for every candidate and generation, and the full current/recommended verification uses the same prefix. Known pending events stay fixed. The stable optimization seed does not depend on the checkpoint hash; the complete canonical JSON/SHA-256 state hash is used for stale protection without pickle or class identity. Repeating the same checkpoint, options and seeds reproduces policies and metrics (runtime naturally differs). See [reload-safe hashing and verified bottleneck diagnosis](V3_DIAGNOSIS.md) for the explanation pipeline and regression validation.

Identical policies are memoized within the experiment. Search observations include actual per-replication metrics, deterministic labels, fitness components and generation context. They are retained in memory, not exported as per-generation CSVs.

## Exact fitness

Higher is better. Let `r` be robustness on the 0–100 scale, `M/H/U` the configured mean-wait/high-risk-wait/utilization targets, `m=max(M,1)`, `h=max(H,1)`, `u=max(U,.01)` and `L` the horizon. All wait/recovery quantities remain in minutes.

```text
F = mean number of failed deterministic conditions per replication
V = mean over replications of:
      max(mean_wait-M,0)/m
    + 2*max(high_risk_wait-H,0)/h
    + sum over ICU, General, doctors, nurses of max(peak_util-U,0)/u
N = 1 - recovered_replications / replication_count
T = mean observed recovery_time / L, or 0 if none recovered
R = ICU_reserve/.30 + doctor_reserve/.25 + nurse_reserve/.25
C = high_risk_weight/5 + aging_weight/2 + surge_weight/5

fitness = 100*r - 2000*F - 5000*V
          - 80*mean_high_risk_wait/h - 40*mean_wait/m - 5*mean_P95_wait/m
          - 2*mean_queue_peak - 200*N - 20*T - 30*R - 2*C - D
```

If candidate robustness is no higher than the current search baseline:

```text
D = 500*max(candidate_mean_wait-current_mean_wait,0)/m
    + 10*max(candidate_queue_peak-current_queue_peak,0)
    + 100*max(current_completions-candidate_completions,0)
```

Otherwise `D=0`. This penalizes unnecessary blocking, queue growth and completion loss without a robustness improvement. There is **no physical resource-cost gene or penalty**. Restrictions can still deliberately trade a small routine wait increase for improved utilization/robustness; the UI reports the actual before/after tradeoff.

## Full verification and authoritative result

After search, both the current policy and winner are verified using the full selected look-ahead replication count (default 5), same checkpoint/event/seeds and unchanged capacity. If the winner has worse full-verification fitness, the current policy is retained; the search winner and guard decision are recorded. The recommendation table displays verification, not just the three-run search estimate.

Part 2 criteria remain authoritative: a run passes only when mean wait ≤M, high-risk mean wait ≤H and **each** resource's peak usable-capacity utilization ≤U. Robustness is 100 × acceptable runs / runs. HANDLED means robustness reaches the configured threshold. Below it, zero robustness or robustness ≤50% with mean/high-risk wait >2× its target is CANNOT HANDLE; other outcomes are AT RISK.

The UI distinguishes current status, optimized handling, improvements still at risk, and **NO FEASIBLE POLICY WITH CURRENT PHYSICAL CAPACITY** when the verified recommendation misses the threshold. The accompanying text explicitly qualifies this: a finite heuristic search does not prove mathematical infeasibility. It never recommends imaginary additional staff/beds.

Candidate verification holds the candidate policy throughout the horizon. Actual temporary-policy restoration is a separate human-selected application mode; metrics are not claimed to forecast an identical automatic-restore trajectory.

## Human application and temporary restoration

**Apply Optimized Operating Policy** is explicit. It updates only future queue/allocation rules, stores the prior policy, records POLICY_CHANGED and retains every current assignment. Applying a proposed-event policy does not apply the event: that still requires separate preview/application. The changed checkpoint makes the old preview/recommendation stale.

**Apply Until Event Recovery** defaults for event contexts; a persistent mode and manual normal-policy restore are also offered. Temporary restoration waits for the trigger and every stress event already ACTIVE at application to satisfy the Part 2 recovery test: ENDED, queue ≤ application queue +2, and no event-specific patient waiting. It also waits until there is no other ACTIVE stress event. This prevents a short surge ending from restoring policy while its simultaneous shortage remains active. Protected recovery-event records cannot be pruned before restoration. Recovery does not require all event patients to discharge.

The previous policy is restored with POLICY_RESTORED; nested temporary applications preserve their original restoration target. Persistent policies never auto-restore. Actual apply/restore history and a separate recommendation audit are each capped at 200 entries; the UI renders the last 30. Reset clears both. No disk checkpoint or unbounded history is created.

Provenance includes the complete live-state hash, timestamp, fixed/effective capacities, current policy/queue, source profiles, event, horizon, targets, verification/search replications, GA options, gene bounds and seed strategy. Changing any of these invalidates recommendation/XAI/LLM display and application.

## XAI and optional Groq

The live Decision Tree uses the six policy genes and actual search-replication ACCEPTABLE/UNACCEPTABLE labels. It checks labels against deterministic criteria. Depth is capped at 3; metadata includes sample/class counts, training accuracy/depth/leaves and group-held-out accuracy only when an adequate two-class split is possible. Policies are grouped so identical-policy replications do not leak across a validation split. Single-class results are non-fatal. Rules are learned experiment-specific patterns, never the authoritative verdict.

The default explanation is deterministic, grounded in verified before/after metrics. **Generate Groq Live Explanation** optionally reuses Groq / `openai/gpt-oss-120b`, existing `.env` precedence and a strict schema of supported factual themes. Groq selects which supported facts to emphasize; numeric claims and verdicts are rendered from validated data. Capacity/limitations are always included. Missing keys, malformed output and request errors retain the deterministic explanation. No API key is displayed and no API request is made automatically. Live network success was not exercised in validation; schema/provider/fallback paths were mocked.

## Time controls and bounded execution

Speeds: **1x, 5x, 10x, 30x, 60x, 120x, 300x, 600x, 1200x, 3600x**. The clock driver accrues `elapsed_monotonic_seconds * previous_speed / 60` simulated minutes. Each callback advances at most 60 minutes, retains catch-up debt, and processes the heap in order. The browser refresh schedules bounded work; it does not determine event correctness. Pause stops accrual/advancement and keeps existing debt, with a visible catch-up indicator.

Manual input accepts up to **43,200 minutes (30 days)**. Presets are +5min, +30min, +1h, +6h, +12h, +1d, +3d and +7d. `LiveSkipDriver` processes one ≤60-minute chunk per refresh, retaining remaining debt, progress, pause/resume/cancel. Automatic accrual is suspended during a manual skip so wall playback is not added to an explicit skip. No intermediate giant table/log exports occur. At a nominal one-second refresh, a seven-day UI skip needs about 168 callbacks and a thirty-day skip 720; Python validation's bounded-chunk helper is faster because it does not wait for browser refreshes. Existing patients/arrivals/events/outage expiry remain ordered.

## Validation and limitations

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -v
python tests/validate_live_policy.py
python tests/validate_live_policy_browser.py
python tests/validate_live_dashboard.py
python tests/validate_live_events.py
python tests/validate_part3_dashboard.py
```

The test suite covers fixed resources, gene bounds, high-risk/aging/incident ordering, reserves without interrupted ownership, clone/seed isolation, CRN, stress improvement, no-feasible messaging, full verification, stale checks, single-class XAI, strict Groq/fallback, application/recovery/history, high-speed catch-up/pause, 7-day equivalence and bounded 30-day skips. Original Part 1/2 and V2 tests remain included. One old reserve-validation assertion now checks the fraction-validation message; the event dashboard validation advances 60 rather than 10 minutes to obtain a non-empty state at the lower demonstration arrival rate.

Compact real-profile evidence lives in `results/live_twin/part3_validation.json`; native launcher/browser evidence in `part3_browser_validation.json`. No datasets/models/checkpoints/screenshots are copied.

### Recorded real-profile demonstration

After 12 simulated hours the baseline had 55 arrivals, 3 completions, 52 active patients and no queue. Free resources: ICU 192, General 526, doctors 64, nurses 124. A standalone **30-patient surge over 30 minutes**, with 30% high-risk profiles sampled from existing RF-high-risk/low-risk subsets, was HANDLED at 100% robustness. RF predictions/features were not changed.

For a meaningful combined stress test, **90 nurses were temporarily unavailable for 120 minutes**, leaving 50 usable nurses, and the same-sized surge was proposed. This is a deliberately severe manual synthetic event, beyond ordinary Low random-event magnitudes. All physical totals remained 220/550/80/140.

The default GA searched 51 distinct policies over six generations, using three search replications and five final verification replications over 120 minutes. The recorded recommendation was `(high_risk=.4, aging=1.0, ICU reserve=.03, doctor reserve=0, nurse reserve=.11, surge=1.9)`.

| Full verification metric | Current | Optimized |
| --- | ---: | ---: |
| Handling verdict | AT RISK | AT RISK |
| Robustness | 40% | 80% |
| Mean wait | 0 min | 0.6954 min |
| High-risk mean wait | 0 min | 0 min |
| P95 wait | 0 min | 5.8678 min |
| Mean queue peak | 0 | 1.4 |
| Mean peak ICU utilization | 18.91% | 18.91% |
| Mean peak General utilization | 8.91% | 8.91% |
| Mean peak doctor utilization | 57.75% | 56.50% |
| Mean peak nurse utilization | 92.40% | 90.40% |
| Mean event recovery time | 30 min | 30 min |
| Unfinished at horizon end | 90.6 | 90.6 |

The failure mechanism was nurse peak utilization, rather than waiting-time targets. Reserving staff reduced peaks at the cost of small routine waits/queue growth; neither wait nor recovery improved in this particular case. Four of five optimized runs passed, so the verified policy still missed the 90% threshold. The UI explicitly displays the no-feasible-verified-policy qualification. This result does not establish that every possible policy is infeasible. A separate controlled stress test verifies that the GA can improve 0% to 100% robustness and reduce high-risk wait from 15 to 0 minutes, with fixed capacity.

Application at minute 720 preserved incumbent assignments. Exactly 30 surge patients entered normal resource flow. The surge ended at minute 750; the policy stayed active until the simultaneous shortage ended at minute 840, then logged POLICY_RESTORED. Persistent application is separately tested to remain unchanged after recovery. Fresh-process validation reproduced the exact recommended genes, seeds and predictive metrics.

Recorded GA runtime was approximately 9 seconds; the entire dashboard plus real-profile validation took approximately 18 seconds. A seven-day bounded Python skip took about 0.25 seconds and exactly matched 168 incremental advances. The native `python main.py`/Chrome run passed explicit event/policy application, unchanged preview clock, stale protection, 3600x playback/pause, normal resume and reset, in about 35 seconds. The full suite passed **103 tests**, including all preserved V2/V3 Part 1/2 tests, in about 2.6 seconds after imports. Part 1/event/V2 feedback-review AppTests also passed. Groq validation used mocks, with no live API requests.

Future work: calibrate operating-policy weights/reserve eligibility with domain review, validate across more checkpoints/seeds/horizons, assess fairness and held-out performance, and integrate live human feedback if required. Severe infeasible events remain infeasible with fixed capacity. Queue recovery is not complete patient discharge, and short horizons cannot prove long-run operational safety. Extreme demand/very large queues can make a bounded simulated-time chunk computationally expensive; high-speed UI validation used the documented baseline, not unlimited overload.

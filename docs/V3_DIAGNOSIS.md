# V3 canonical provenance and deterministic optimization diagnosis

Current live resource/recovery acceptance and GA verification are defined in [V3_LIVE_ACCEPTANCE.md](V3_LIVE_ACCEPTANCE.md). That later change replaces the historical peak-only rules described below with sustained-pressure evidence while preserving this canonical hashing and diagnosis architecture.

This correction changes hashing, diagnosis and presentation only. It preserves live allocation, physical capacities, RF threshold 0.50, patient lifecycle, arrivals/events, GA genes/fitness, look-ahead acceptance criteria and robustness formulas.

## Canonical hash and hot reload

`live_events.LiveEventSupport.state_hash()` now hashes `live_canonical.live_hash_input()` using:

```python
hashlib.sha256(json.dumps(
    primitive_state, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode("utf-8")).hexdigest()
```

Only dictionaries with string keys, lists, strings, integers, finite floats, booleans and null enter JSON. Dataclass fields are explicitly read with `dataclasses.fields`: retained objects of old `OperatingPolicy`, patient, resource and event classes are serialized as field dictionaries, without a class/module identity lookup. NumPy scalar/array values become primitives/lists, RNGs contribute their full bit-generator state, and tuples/deques become lists. Unsupported objects/non-finite numbers are rejected rather than represented with `repr()` or object addresses. Pickle is neither imported nor called for state/provenance hashing.

The complete engine attribute state is retained except immutable clinical tuples, which are covered by the existing `profile_sha256`. This includes clock/status, capacities, all active/completed/rolling patient records, resource status and patient ownership, counters, current/normal/previous policies, temporary-policy restore state, policy history, bounded event history, active stress-event objects/statistics, all four RNG states, arrival rates and settings, limits and pending event agenda. Agenda/free-resource heaps are sorted into semantic order; computed queue order, queue insertion order and stress-event order are explicit. The effective arrival rate, RF threshold, step limit and hash-schema version are included. Object aliases and heap array layout do not contribute identity metadata.

GA run IDs also use canonical JSON SHA-256. Future replication seed derivation remains unchanged and independent of state hashing. No checkpoint is written to disk. Old recommendations/explanations with the former hash schema become stale and require regeneration; they are not silently accepted. The tests retain a live state, reload `live_hospital_state`, confirm its old policy class differs from the new class, then hash, clone, optimize, JSON-load a recommendation and check stale/explanation provenance without errors. Streamlit AppTest repeats this with a live session and widget reruns.

## Evidence validation and diagnosis schema

`live_policy_diagnosis.diagnose_live_result()` reads final **verified** current/recommended predictions. It checks configuration/capacity/policy/replication compatibility, recomputes each run's deterministic verdict and condition breakdown, and verifies aggregates, event-specific aggregates, acceptable counts, robustness and handling verdict against the runs. Unsupported fields and non-finite/invalid metrics are rejected before any Groq call. Stored diagnosis input hashes must remain compatible.

The diagnosis has these fields:

```json
{
  "overall_result": "NO_VERIFIED_FEASIBLE_POLICY",
  "authoritative_verdict": "CANNOT HANDLE",
  "robustness": 0.0,
  "blocking_conditions": [
    {
      "metric": "nurse_utilization",
      "observed": 1.0,
      "observed_mean": 1.0,
      "target": 0.9,
      "severity": "critical",
      "failed_replications": 5,
      "total_replications": 5,
      "affects_acceptance": true
    }
  ],
  "likely_constraint_type": "STAFF_BOTTLENECK",
  "primary_limiting_factor": {},
  "secondary_limiting_factors": [],
  "operational_observations": [],
  "secondary_improvements": {},
  "metric_changes": {"improved": {}, "unchanged": {}, "worsened": {}},
  "policy_changes": {},
  "current_policy_diagnosis": {},
  "operational_result": "Operational improvement without feasibility recovery",
  "provenance": {"version": "live-condition-diagnosis-v1", "live_state_hash": "...", "input_sha256": "..."}
}
```

The example is abbreviated; actual records also include labels, observed minimum/statistic and initial utilization when relevant. `observed` is the **maximum verified value**, not the mean of peaks. The comparison table retains mean metrics. Failed counts come directly from the six condition flags: mean wait, high-risk mean wait, ICU/General/doctor/nurse utilization. Robustness below its threshold is a separate blocking record.

Queue growth means a replication's **end queue exceeds the checkpoint's initial queue**. Non-recovery means its recovery time is null within the selected horizon. Both are explicitly marked `affects_acceptance=false`: they diagnose operational pressure but do not add new verdict conditions. A policy meeting the configured robustness threshold has no overall blockers; failed-run exceptions are retained in `residual_failed_conditions`. Recovery observations remain visible even in a HANDLED result.

Severity is an operational diagnostic label, not injury/clinical severity. A core condition is `critical` when every run fails it, a failing resource peak is at least 99%, or a failing wait exceeds twice its target; otherwise it is `major`. Zero robustness is critical. Conditions are ranked by failed-run count, critical severity, normalized excess `(observed-target)/max(target,.01)`, then metric name. Resource categories select their highest-ranked resource as primary; other categories select the highest-ranked failed acceptance condition. Queue/recovery observations never replace the actual acceptance evidence.

## Exact category rules

Rules use verified runs, in this order:

1. Robustness at/above its threshold: `NO_BLOCKING_CONDITION`, with residual failures preserved.
2. Two or more resource types have any failed verified runs: `MULTIPLE_RESOURCE_BOTTLENECK`. It becomes `DEMAND_OVERLOAD` only if at least two resources fail in at least half the runs, at least half the runs have two or more resource peaks ≥95%, and at least half have end-queue growth **and** additional arrivals exceeding completions.
3. One resource repeatedly fails (at least half the runs): `STAFF_BOTTLENECK` for doctors/nurses, otherwise `BED_CAPACITY_BOTTLENECK` for ICU/General.
4. A single resource fails in fewer than half the runs: `PHYSICAL_CAPACITY_BOTTLENECK`, qualified as an intermittent verified limit exceedance.
5. No resource failure, and at least half the runs do not recover: `RECOVERY_FAILURE`.
6. No resource failure, with a waiting-target failure: `WAITING_TIME_TARGET_FAILURE`. It becomes `PROCESS_POLICY_LIMITATION` if end-queue growth is observed and the evaluated policy has a positive reserve, identifying a possible scheduling restriction rather than a proven causal mechanism.
7. Remaining non-recovery: `RECOVERY_FAILURE`; otherwise `PROCESS_POLICY_LIMITATION`.

Non-recovery and wait failures also appear as secondary categories when another primary category dominates. These are finite-experiment diagnostic heuristics, not proof that more physical resources are the only possible solution. The existing HANDLED/AT RISK/CANNOT HANDLE verdict is never changed.

When a resource is already above target at the checkpoint, the diagnosis records `initial_state_utilization` and `already_above_target_at_checkpoint`. The explanation notes that a new queue policy cannot erase that already observed peak under the existing look-ahead metric.

## Improvements versus feasibility

Changes are detected using verified means for waits, P95, queue peak/end, four resource peaks, unfinished patients and recovery time; higher robustness, recovered count and completions count as improvements. Numeric equality uses 1e-9 absolute/relative tolerance. Null recovery becoming measurable is improvement; the reverse is worsening. Recovery counts are reported alongside conditional mean recovery times.

* A new HANDLED result from an unhandled current policy is `VERIFIED FEASIBILITY RECOVERY`.
* An already HANDLED policy staying acceptable is `VERIFIED ACCEPTABLE POLICY`.
* Any metric improvement while still unhandled is **Operational improvement without feasibility recovery**.
* Otherwise the result is **No verified operational improvement**.

The UI lists improved, unchanged and worsened metrics, preserving adverse tradeoffs rather than reporting improvement selectively.

## Decision Tree and structured AI input

Live tree training logic is unchanged. Labels still come from actual GA-tested runs and are checked against Digital Twin conditions. The explanation includes the recommendation's actual rule path, exact thresholds, classification, class probabilities and training/group-validation limitations.

For failed optimization, an additional negative pattern is shown only if an actually tested policy had unacceptable observations and the trained tree classifies that policy as UNACCEPTABLE. Its path, policy, observed unacceptable/tested counts and probabilities are recorded. It is labeled **DECISION TREE XAI PATTERN**, an association rather than an authoritative cause. Single-class data or a tree with no negative path yields an explicit unavailable message; no rule is invented.

`build_explanation_payload()` regenerates and checks XAI against those actual observations. Its primitive-only payload contains:

* event type/parameters/duration, horizon, clock, queue, active-event context, active count, resource state and state hash;
* fixed physical resources and an explicit unchanged-capacity flag;
* current/recommended genes, authoritative verdicts, robustness and verified wait/P95/queue/peak/recovery/unfinished metrics;
* targets, verification count, exact blockers/counts/severity, deterministic categories/primary/secondary factors and initial-peak evidence;
* improved/unchanged/worsened metrics and actual policy changes;
* learned XAI paths/probabilities/limitations;
* capacity interpretation and grounded human-review options.

Groq remains `openai/gpt-oss-120b`, with existing `.env` support/process-environment precedence. It receives **only this validated structured payload**, never raw logs, datasets or an API key in messages. Its strict output selects supported explanation themes. All displayed numerical claims, failed conditions, verdicts and rules are then rendered from validated facts, preventing arbitrary LLM wording from inventing evidence or changing status. Cause/improvement/fixed-capacity/finite-search limitations are mandatory even if Groq omits those themes.

Missing keys, request errors, malformed JSON or invalid schemas use the same deterministic clinician/admin template. API calls remain explicit. Live network success was not exercised; valid/malformed schema/provider behavior was mocked. Invalid evidence is rejected before a provider request; a stale result is hidden pending a fresh evaluation.

**Possible Operational Responses for Human Review** are deterministic templates tied to blockers: temporary/permitted staffing review, approved bed/overflow/transfer review, external support/demand routing review, reserve/queue tradeoff review or extended recovery monitoring. They are options under hospital policy, never automated clinical actions or live-capacity changes.

## Provenance and UI

The diagnosis hashes configuration plus both verified run sets/metrics and policies. Explanations retain the GA provenance and add an input checksum containing event/current state, targets, fixed capacity, both policies/metrics, blockers, diagnosis and regenerated XAI. Generation provider/model/schema/requested state and rendered-output checksum are also covered. Changes to live state/configuration, verified evidence, learned patterns, generation state or rendered prose make old output stale. Older outputs lacking this schema are regenerated from validated evidence or require a new optimization if the GA result itself is stale.

The adaptive UI adds **Why the Optimized Policy Works** / **Why This Policy Still Fails**, primary factor, maximum observed value/target/failing verified count, all blocked conditions, diagnostic observations, improvement table, explanatory tree, AI explanation and human-review options. The existing dark theme and allocation/application controls remain unchanged.

## Validation evidence

`tests/test_live_diagnosis.py` covers canonical ordering/primitives/non-finite rejection, unchanged/cloned/reloaded state, clock/RNG/queue/ownership/agenda/configuration changes, no pickle calls, actual module reload with result loading, all category branches, staff shortage, multi-resource failure, improvements without robustness recovery, successful explanation, initial saturation, strict provider inputs, fabricated-XAI rejection, fallback and stale generation/output checks. The full suite passed **121 tests**, including the original 103 V2/V3 tests.

`tests/validate_live_diagnosis.py` verifies Streamlit retained state across module reload, successful/failing UI sections and stale behavior. Its real-profile example uses the fixed 220/550/80/140 baseline after 12 simulated hours, with 130 temporarily unavailable nurses and a 30-patient surge over 30 minutes. All five verified runs exceed nurse utilization: **100% versus 90%**, `STAFF_BOTTLENECK`; wait targets also fail. The initial nursing peak was already 100%, so the finite small search retains the current policy and reports no improvement rather than inventing one. Capacity/ownership/live clock remain unchanged during diagnosis/search.

A separate controlled test has nurse utilization 100% versus 90% in 5/5 runs, robustness **0%→0%**, but high-risk wait **15m→0s**: operational improvement without feasibility recovery. With that controlled test's utilization target explicitly changed to 100% (other simulation logic unchanged), the same pipeline produces **0%→100%**, HANDLED, and reports that every verified operational condition passed. These controlled examples are not claims about the real-profile hospital.

Compact evidence is stored in `results/live_twin/diagnosis_validation.json`. The real-profile small GA ran in roughly half a second; dashboard/reload/demo validation took about eight seconds on the test machine. Canonical hashing averaged about 12 milliseconds per call in that nonempty-state benchmark. The existing real-profile Part 3 validation also passed its exact comparison to pre-correction genes, replication seeds and metrics. Event preview/outage validation passed. Native Chrome validation launched `python main.py` and passed explicit event/policy application, stale protection, 3600x pause/resume and reset, with no screenshots.

Run with `PYTHONDONTWRITEBYTECODE=1` to avoid new repo bytecode caches. No datasets/models/checkpoints are duplicated and no per-generation disk files are introduced.

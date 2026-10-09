# Live event acceptance based on sustained pressure

This extends V3 Part 3 and the [verified diagnosis pipeline](V3_DIAGNOSIS.md). V2 scenario experiments retain their original window-average rules. Physical capacities, patient lifecycle/ownership, safe outages, RF threshold 0.50, event scheduling, canonical state hashing, six policy genes, genetic operators and common random numbers are unchanged.

## Observation and acceptance

`live_pressure.py` defines live-only `LiveTargets` and `PressureIntegrator`. Default utilization target is 90%, sustained-overload grace 30 simulated minutes, full-saturation grace 15 simulated minutes. The two grace controls are in **Advanced Live Twin Acceptance Settings**; utilization and waiting targets remain in Operational Targets. Limits enter preview/GA/diagnosis/explanation provenance, so changing them makes prior results stale.

Every look-ahead clone integrates between scheduled events, over exactly its horizon H. Occupancy is constant between events; no polling or alteration of the real engine is needed. For resource r and each interval i:

```
u[r,i] = busy[r,i] / usable[r,i]
usable = physical total - OUT_OF_SERVICE resources
area[r] += u[r,i] * delta_minutes[i]
time_weighted_utilization[r] = area[r] / H
```

Busy resources pending outage remain usable until safe release, matching the existing engine. Zero usable capacity retains the engine's conservative utilization of 1.0. Outage start/end and resource releases split intervals exactly. Zero-duration transitions can affect diagnostic peaks but do not contribute time or break a positive-duration streak. Integration starts at the checkpoint; pre-checkpoint occupancy time is not extrapolated.

For each resource the forecast reports:

* time-weighted utilization and peak utilization;
* total minutes above the target and longest continuous interval above target;
* total minutes at full saturation and longest continuous full-saturation interval.

Full saturation means utilization >= 1 - 1e-12, solely to avoid floating-point representation artifacts. Above-target comparison is strictly `u > target`.

A resource passes only if **all three** hold:

```
time_weighted_utilization <= utilization_target
longest_above_target_streak <= sustained_overload_grace_minutes
longest_full_saturation_streak <= critical_saturation_grace_minutes
```

Thus exactly 30/15 minutes passes the respective duration check; longer fails. A brief peak alone never fails. A high average can still fail even when no single continuous interval exceeds grace. A passing resource with a peak above target receives a warning, separate from the authoritative verdict. The retained `*_utilization` fields are diagnostic peaks; authoritative decisions use `resource_pressure`.

## Waiting, recovery and robustness

Each run returns `HANDLED_RUN` or `FAILED_RUN`. It passes when mean/high-risk waits meet targets, all four resources pass, and:

```
end_queue <= checkpoint_queue + 2
unresolved_event_waiting == 0
```

The queue tolerance of two reuses existing V3 queue-recovery semantics. Unresolved event waiting counts patients actually waiting from any event, including automatic events and already-active events. Valid treatment or bed stay, future event arrivals beyond the forecast, and incomplete discharge are not failures. Historical recovery time remains the existing event-expiry/queue-based metric; absence of an expiry-dependent recovery timestamp alone is not failure when the end queue and event waiting recover. The forecast covers only its stated horizon, not later unobserved demand.

Robustness = 100 * handled runs / total runs. Existing aggregate statuses remain: HANDLED at/above robustness threshold; CANNOT HANDLE at zero robustness, or robustness <= 50% with mean/high-risk mean wait more than twice target; AT RISK otherwise.

## Exact revised GA objective

All genes and operators are unchanged. Define `m=max(mean_wait_target,1)`, `h=max(high_risk_wait_target,1)`, `u=max(utilization_target,.01)`, `G=max(overload_grace,1)`, `C=max(saturation_grace,1)`, observation horizon `H`, and `[x]+ = max(x,0)`. Averages below are across runs; resource sums are over ICU/General/doctors/nurses.

```
V = mean_runs(
      [mean_wait - mean_wait_target]+ / m
    + 2*[high_risk_wait - high_risk_wait_target]+ / h
    + sum_resources(
        [time_weighted_utilization - utilization_target]+ / u
      + [longest_above_target_streak - overload_grace]+ / G
      + [longest_full_saturation_streak - saturation_grace]+ / C))
F = mean_runs(number of failed conditions: waits, four resources, recovery)
R = mean_runs(not recovery_pass)
S = sum(reserve gene / its maximum)
K = sum(nonreserve gene / its maximum)
D = 500*[mean_wait - baseline_mean_wait]+/m
    + 10*[queue_peak - baseline_queue_peak]+
    + 100*[baseline_completed - completed]+
    (D applies only when robustness <= baseline robustness)

fitness = 100*robustness_percent
  - 2000*F - 5000*V
  - 80*high_risk_mean_wait/h - 40*mean_wait/m - 5*P95/m
  - 2*queue_peak
  - 10*sum_resources(mean_minutes_above_target
                    + 2*mean_minutes_at_full_saturation)/H
  - 50*[mean_end_queue - checkpoint_queue - 2]+
  - 200*R
  - 20*mean_recovery_time/H (zero if no recovery timestamp exists)
  - 30*S - 2*K - D
```

This preserves the previous objective's reward and penalty weights, substitutes sustained pressure for peak violations, adds small duration/queue-growth terms, and bases non-recovery on the actual recovery condition. Peaks contribute no violation penalty. The small duration term discourages unnecessary pressure without treating a short transient as a major failure. Resource-cost optimization remains absent because physical capacity is fixed.

Search uses the existing shared seeds and smaller search replication count. **Every unique tested candidate is fully verified** from the same checkpoint, with the same full replication seeds. The highest verified fitness wins, including the current policy. Exact evaluations are reused if search and verification replication counts match. This costs more than verifying only one search winner but avoids discarding a useful partial candidate or returning a worse verified result.

Outcome is VERIFIED SUCCESS if threshold is met; otherwise PARTIAL IMPROVEMENT if meaningful verified operational metrics improve; otherwise NO MEANINGFUL IMPROVEMENT. Peak-only improvements do not qualify. Meaningful changes use at least 0.1 percentage point utilization, one simulated second duration/wait, or 0.1 unit for other metrics, with the existing numerical-equality check. A finite heuristic search never proves mathematical infeasibility.

## Diagnosis, XAI and Groq

The existing deterministic diagnosis architecture remains. Resource blockers include all six pressure metrics, the three configured limits, failed counts for each criterion and overall failed verified runs. The main primary-factor card shows a violated continuous-duration limit when average utilization passes. Initial peaks remain contextual evidence, not irreparable failures. Queue and unresolved event waiting now have an authoritative recovery condition; absence of an expiry-dependent historical timestamp remains contextual.

Resource bottleneck classification uses repeated **new resource-condition failures**, not peak crossings. Demand overload uses repeated concurrent high time-weighted utilization plus queue growth. Existing category precedence is retained. Transient warnings are shown separately and explained as nonblocking pressure.

The Decision Tree uses actual GA-tested observations labeled by `live_run_verdict`; its displayed ACCEPTABLE/UNACCEPTABLE classes map to HANDLED_RUN/FAILED_RUN. It remains correlation only, handles single-class observations gracefully and never changes the authoritative result. The existing Groq `openai/gpt-oss-120b` receives the validated structured pressure/diagnosis payload. Its strict theme selection renders only verified facts; malformed responses or no key use deterministic fallback. No second provider or explanation system is introduced.

## Human workflow

Main metrics remain compact, with detailed pressure in the event replication panel and current-versus-optimized resource-pressure expander. After diagnosis, XAI, AI/fallback and human-review responses, the user can:

* **Apply Optimized Operating Policy**, including an explicitly limited partial improvement. Only future operating-policy decisions change; capacity and current assignments remain intact. Existing temporary/persistent application modes remain.
* **Reject Recommendation**: audit rejection, disable application of that recommendation, and leave all live state unchanged.
* **Modify / Retry**: show guidance to adjust event parameters, horizon, targets, grace durations or GA effort, then prepare/preview and run again. A fresh GA clears prior rejection. No physical resources change through retry.

Manual events and seeded random events are preserved. Applied/automatic events can be selected in Adaptive Optimization Context and previewed through **Preview Current / Applied Event** before the same diagnosis/optimization/human-review workflow. Events do not automatically run GA or apply a policy.

## Validation

`tests/test_live_pressure.py` covers integration, transient 91.2%/100% peaks, independent sustained-overload/saturation failures, average failures, grace boundaries, outage boundaries/ownership, genuine recovery failure, reproducibility, full verification/partial selection, fitness, actual XAI labels, validated Groq payload and malformed fallback, physical invariance, automatic/manual events and AppTest Apply/Reject/Modify/stale paths. Existing tests retain their coverage with peak-only expectations updated where the requested semantics intentionally changed.

The complete suite passes **140 tests**, including 19 new pressure/workflow tests and all 121 preceding V2/V3 regressions. The final full suite takes approximately 8.4 seconds on this machine.

`tests/validate_live_pressure.py` writes only compact evidence to `results/live_twin/pressure_validation.json`:

* 31/34 doctors busy for four minutes: 91.2% peak, 3.04% average over 120 minutes, HANDLED.
* One doctor fully busy for four minutes: 100% peak, 3.33% average, HANDLED with ongoing valid bed stay.
* One doctor fully busy for 50 minutes: 41.67% average but continuous pressure exceeds both graces, CANNOT HANDLE.
* Controlled GA: high-risk wait 15 seconds to zero, robustness 0% to 100%, doctor full saturation 30 seconds. The optimized high-risk weight is 0.4; other learned genes are reported in evidence. Tree associations are not causal claims about those genes.
* Occupied-checkpoint pressure GA: ten busy doctors, eleven waiting patients and a one-doctor shortage; high-risk waiting falls 24 to 4 minutes and the longest continuous doctor saturation falls 24 to 4 minutes. Robustness rises 0% to 100%, with existing owners and fixed 10/100/11/100 capacity intact during search.
* Controlled partial GA: high-risk wait 15 minutes to zero, robustness 0% to 0%; nurse average 50%, peak 100%, continuous full saturation 30 minutes exceeds the 15-minute grace. The useful partial policy remains available for human application.
* Real RF-profile baseline 220/550/80/140 at 4/hour after 12 simulated hours, plus 30 patients over 30 minutes: HANDLED, source state/RNG unchanged. This is a demonstration, not clinical validation.

Small controlled GA runs take approximately 0.03 seconds each; the three-replication real-profile forecast approximately 0.08 seconds on this machine. All-candidate final verification scales with unique policies, full replications and event density. Integration retains event-driven processing without high-frequency sampling or disk writes. Live Groq network success was not exercised; provider/schema behavior is mocked. No model/data/checkpoint copies or giant histories are created.

The existing real-profile Part 3 workflow also passes: at baseline fixed capacity with a temporary 90-nurse shortage and 30-patient surge, verified nurse peak averages 92.4%, time-weighted utilization 81.7%, and the mean longest overload is 14.1 minutes. All five runs pass the new criteria with no queue or waiting; ongoing bed stays are not failures. Its 30-candidate/full-five-replication GA takes approximately 10.5 seconds. Explicit application, later temporary-policy restoration and seven-day random-event skip equivalence remain validated. Native browser validation launches `python main.py` and passes event preview/application, policy application, stale protection and high-speed pause.

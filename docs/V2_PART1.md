# V2 Part 1: time-window simulation and current-policy evaluation

Part 2 is now implemented through separate V2 GA/XAI modules; see [V2 Part 2](V2_PART2.md). Part 1 simulator behavior remains unchanged. The legacy-path notes below describe compatibility, not the active dashboard optimizer.

The dashboard can evaluate the entered hospital capacities under Best, Average, and Worst demand without running GA. The observation interval is `[0, simulation_duration_hours * 60)` minutes. It starts empty, stops at the selected end, and never drains patients to completion for primary metrics.

## Files and functions

- `src/scenario_evaluation.py`: `ScenarioConfig` and validation; cached `load_patient_profiles`; `generate_arrivals`; `prepare_profiles`; `simulate_time_window`; `policy_verdict`; `evaluate_current_policy`; `save_current_policy_results`; V2 CLI.
- `src/digital_twin.py`: CLI dispatch defaults to V2; `--legacy-v1` invokes the original CLI. Its legacy simulation, baseline helpers, and RF prediction helper remain available to GA.
- `src/dashboard.py`: manual hospital resources, duration/rates/replications, operational targets, random seed, independent evaluation action, scenario summaries, overall status, changed-settings warning, cached source data, graceful missing/stale legacy artifacts. Existing dark styling remains.
- `tests/test_scenario_evaluation.py`: eight fast unit tests.
- `tests/validate_v2.py`: real-data, seed, resource-response, CLI, and artifact checks.
- `tests/validate_v2_dashboard.py`: Streamlit interaction checks and actual `python main.py` health check; stops its own server after validation.
- `results/digital_twin/current_policy_evaluation.json`: saved V2 evaluation (configuration recorded in the artifact), per-run metrics, aggregates, configuration, resource policy, metric definitions, and data/model/simulator hashes.
- `results/digital_twin/v2_validation.json`: measured validation examples.
- `README.md` and this report: usage and definitions.

`preprocessing.py`, `synthetic_data.py`, `random_forest.py`, GA, Decision Tree XAI, LLM, and model artifacts were not modified.

## Arrivals and patient flow

```python
rng = numpy.random.default_rng(seed)
interarrival_minutes = rng.exponential(60.0 / arrival_rate)
next_arrival_minutes = previous_arrival_minutes + interarrival_minutes
```

The first arrival also follows an exponential interval. An arrival at or beyond window end is excluded. A zero arrival rate produces no arrivals. Rates must be finite and satisfy `0 <= Best <= Average <= Worst <= 1000`; invalid ordering raises a clear error without reordering values.

Patient profiles are sampled with replacement from the existing synthetic/augmented clinical records. RF inference runs once per loaded source/model version, at threshold **0.50**, before replications. All replications reuse those predictions. Original treatment/LOS bounds and clinical priority logic are retained. Source DataFrames and dataset files are not mutated.

V2 admission acquires one appropriate bed, one doctor, and one nurse together, ordered by clinical priority then arrival among patients whose resources are available. This avoids partial resource holdings while queued. Staff are occupied for the treatment duration; the bed remains occupied for the full LOS. A patient who has finished active treatment but still occupies a bed is counted as in treatment at window end. Legacy GA retains its existing sequential acquisition model.

Counts are mutually exclusive: arrived = completed + waiting + in treatment; unfinished = arrived - completed. Completed means the entire hospital flow finished before window end.

Wait metrics include **every arrival**: `service_start - arrival` for admitted patients, or `window_end - arrival` for patients still queued. The latter are right-censored observations, not estimates of their eventual waits. `wait_censored_count` exposes their count. Empty groups, including no high-risk arrivals, have zero wait. Mean, median, and P95 are computed per replication; scenario aggregates report the mean/min/max of these per-run statistics, rather than pooling patients across runs.

```text
resource_utilization = busy_resource_minutes_inside_window
                       / (resource_capacity * window_minutes)
throughput = patients_completed / simulation_duration_hours
```

Unfinished patients' resource occupancy is included and clipped at window end. Utilization is bounded to `[0, 1]`.

## Exact acceptance and robustness rules

A replication is ACCEPTABLE only when all six comparisons pass, using unrounded metrics:

```text
mean_wait <= mean_wait_target_minutes
high_risk_mean_wait <= high_risk_wait_target_minutes
icu_utilization <= max_utilization_target
general_utilization <= max_utilization_target
doctor_utilization <= max_utilization_target
nurse_utilization <= max_utilization_target
```

`condition_breakdown` contains `mean_wait_pass`, `high_risk_wait_pass`, `icu_util_pass`, `general_util_pass`, `doctor_util_pass`, and `nurse_util_pass`. Decision Tree output never determines V2 acceptance.

```text
scenario_robustness = acceptable_runs / replications_per_scenario * 100
scenario_verdict = ACCEPTABLE if scenario_robustness >= robustness_threshold
                   else UNACCEPTABLE
overall_robustness = total_acceptable_runs / (3 * replications_per_scenario) * 100
```

Overall status is ROBUST when Worst robustness meets threshold; otherwise CONDITIONALLY ACCEPTABLE when Average meets threshold; otherwise UNACCEPTABLE. Scenario seeds are `base_seed * 1000 + scenario_index * 100 + run_index`, where Best/Average/Worst indices are 0/1/2 and run indices start at zero. Results contain no changing timestamps, allowing exact same-configuration comparisons. Each replication automatically uses a distinct derived seed: the base seed does not need to be changed between runs. Keep 42 to reproduce an experiment, or intentionally change it for a new experiment set.

## Defaults and bounds

| Input | Default | Validation |
|---|---:|---|
| Simulation duration | 24 hours | 1–72 hours |
| Best / Average / Worst rates | 12 / 22 / 35 patients/hour | Ordered, 0–1000 |
| Replications per scenario | 10 | Integer, 3–30 |
| ICU / General beds | 210 / 280 | Integer, 1–10000 each |
| Concurrent doctors / nurses | 85 / 100 | Integer, 1–10000 each |
| Mean / high-risk wait targets | 20 / 10 minutes | Nonnegative |
| Maximum utilization | 90% | Internally fraction 0–1 |
| Robustness threshold | 90% | 0–100% |
| Random seed | 42 | Nonnegative integer |

Slider ranges remain ICU 1?500, General 1?1000, doctors 1?200, nurses 1?500; numeric entry permits values up to 10000.

Resources are editable demonstration defaults selected by a controlled sweep; see [resource calibration](V2_RESOURCE_CALIBRATION.md). Use **Reset Hospital Resources** to restore them; customized policies survive reruns. No V2 capacity is calculated from workload. The existing dataset's long stays can leave many patients unfinished even with no queues; completion is correctly separated from arrival and admission.

## Usage and validation

```powershell
python main.py
python src/digital_twin.py --simulation-duration-hours 24 --best-case-arrival-rate 5 --average-case-arrival-rate 10 --worst-case-arrival-rate 20 --replications-per-scenario 3 --icu-beds 30 --general-beds 40 --doctors 10 --nurses 20
python -m unittest discover -s tests -v
python tests/validate_v2.py
python tests/validate_v2_dashboard.py
```

Eight fast unit tests passed in approximately **0.46 seconds**. They check 24-hour behavior, 4-hour censoring, partial staff occupancy, all utilization gates, invalid ordering and bounds, resampling, no source mutation, no arrivals/no high-risk arrivals, exact exponential generation, same/different seeds, scenario aggregation, and all three overall statuses.

Real-data examples use 7,500 profiles from the saved dataset and existing RF model. Unless marked otherwise, rates are 5/10/20 per hour and there are three replications per scenario.

| Configuration | Best robustness | Average robustness | Worst robustness | Overall robustness | Status |
|---|---:|---:|---:|---:|---|
| 24h; resources 1/1/1/1 | 0% | 0% | 0% | 0% | UNACCEPTABLE |
| 24h; resources 500/500/100/100 | 100% | 100% | 100% | 100% | ROBUST |
| 4h; resources 30/40/10/20 | 33.33% | 0% | 0% | 11.11% | UNACCEPTABLE |
| Initial Part 1 defaults (30/40/10/20), 10 runs/scenario | 0% | 0% | 0% | 0% | UNACCEPTABLE |

The large resource case is a validation fixture, not a capacity recommendation. For the first two cases, mean arrived counts match exactly at **119.67 / 242.33 / 468.67**, since demand seeds are shared when comparing policies. The large-resource case has zero observed wait; the low-resource case has mean waits **701.47 / 687.66 / 695.05 minutes**. The 4h example has mean waits **21.07 / 42.19 / 87.65 minutes**.

Invalid Best=11, Average=10 produces:

```text
Arrival rates must satisfy 0 <= Best <= Average <= Worst <= 1000 patients/hour.
```

Real-data evaluations reproduced exactly with the same seed and changed arrival statistics with seed 43. RF load/inference took approximately **1.1 seconds**; nine-run evaluations took **0.3–0.4 seconds**, and the initial Part 1 default 30-run evaluation took approximately **2?2.6 seconds**. These are local observations, not performance guarantees; large rates/windows and severe queues take longer. No multiprocessing was added to simulation; the saved RF estimator retains its existing execution behavior.

Streamlit AppTest passed initial rendering, independent current-policy evaluation, and visible invalid-order validation. The actual unchanged `python main.py` launcher started Streamlit and returned HTTP **200** from `/_stcore/health`; the validation server was then stopped.

## Legacy compatibility after Part 2

The preserved V1 GA still uses the fixed-cohort simulator through its original CLI, and its legacy XAI/LLM artifacts remain separate. The main dashboard now uses the V2 GA and Decision Tree described in [Part 2](V2_PART2.md); it does not apply V1 baseline bounds or interpret V1 artifacts as V2 results. LLM/feedback migration is implemented in [Part 3](V2_PART3.md), using automatic GA bounds and validated feedback tightening.

Changing the CLI file changes its provenance hash, so existing V1 baseline artifacts may be rejected as stale. The dashboard handles this gracefully and V2 remains available. Regenerate an optional V1 baseline with `python src/digital_twin.py --legacy-v1` before using legacy optimization. Existing baseline/model/GA/XAI artifacts have not been overwritten. Current-policy results are saved under their own filename.

V2 currently starts with an empty hospital, uses constant Poisson rates per scenario, and reports observed censored waits. Initial occupancy, time-varying demand, confidence intervals, optional draining, and a full dashboard redesign are outside Part 1.

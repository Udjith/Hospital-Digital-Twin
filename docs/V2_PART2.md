# V2 Part 2: scenario GA and Decision Tree XAI

**Part 3 update:** [V2 Part 3](V2_PART3.md) replaces the historical manual bound controls described below with automatic 70%?150% current-policy bounds and validated feedback constraints. Explicit availability overrides now also allow a previously current policy to be infeasible; see [the correction](V2_AVAILABILITY_CONSTRAINTS.md). GA fitness, operators and XAI are unchanged. V2 LLM/feedback are now active.

The main dashboard now optimizes the V2 scenario model. The validated Part 1 simulator was not changed: resource acquisition, exponential arrivals, clinical profile sampling, observation-window metrics, deterministic target checks, scenario robustness, and overall status are unchanged. RF threshold remains 0.50; GA never trains or optimizes RF.

## Files

- `src/genetic_algorithm_v2.py`: V2 configuration validation, initialization, genetic operators, weighted fitness, common-random-number search, memoization, early stopping, full verification, provenance, history, deterministic change summaries, and CLI.
- `src/decision_tree_xai_v2.py`: audit of GA-observed labels, unique policy/scenario training rows, policy-group validation, tree training, verified recommendation explanations, single-class fallback, provenance and CLI.
- `src/dashboard.py`: manually entered optimization bounds, V2 GA controls, progress, current/recommended comparisons, final verification metrics, deterministic summaries, XAI display, stale-result protection and current status/action rendering. Existing dark styling, resource sliders, and Part 1 evaluation remain.
- `tests/test_ga_v2.py`: eleven fast GA/XAI tests, in addition to the eight existing Part 1 tests.
- `tests/validate_ga_v2.py`: real-data search, dashboard interactions, stale configuration rejection, and both V2 CLI checks. `--reuse` reuses a saved search only after checking its full configuration/dependency compatibility.
- `README.md`, `docs/V2_PART1.md`, and this report: active V2 entry points and migration status.
- `results/genetic_algorithm_v2/`: V2 `best_policy.json`, policy/scenario `ga_history.csv`, `best_by_generation.csv`, and `validation_summary.json`.
- `results/decision_tree_xai_v2/`: V2 explanation, training rows, learned rules, and the V2 tree bundle. Legacy model/GA/XAI artifacts are not overwritten.

`genetic_algorithm.py`, `decision_tree_xai.py`, `digital_twin.py`, `scenario_evaluation.py`, preprocessing, synthetic generation, RF, LLM and feedback-interpreter implementations remain unchanged in Part 2. The legacy GA's standard selection and crossover helpers are reused.

## GA architecture

The four integer genes are ICU beds, General beds, concurrent doctors, and concurrent nurses. The initial population's first member is an exact copy of the current policy, in **Generation 0**. Remaining members are random within user-entered bounds using `random.Random(ga_seed)`. Invalid bounds or a current resource outside its range raise a clear error; no resource or bound is silently adjusted.

The generation loop evaluates policies, performs tournament selection, applies uniform gene crossover, mutates integer genes, preserves elites, and advances the population. Mutation uses a local integer step up to 10% of each allowed range, clipped to those bounds. At least one elite is required. Defaults are population 12, maximum 10 generations, mutation probability 0.25 per gene, and two elites. Early stopping uses five successive generations without improvement greater than 0.01; generation count and early-stop status are recorded.

Initial editable bounds are ICU 100–400, General 150–600, doctors 40–180, nurses 60–220. They are static example ranges, not derived from the old workload baseline. Users can enter any validated integer range within 1–10000. The current policy must lie inside every range.

Every candidate calls **the existing `evaluate_current_policy`**. Search uses `ga_replications_per_scenario`, default 3, permitted 1–10. A specialized configuration only changes search replication-count validation; normal Part 1 configuration validation still requires 3–30. No arrival, simulation, metric, acceptance, or aggregation code is duplicated.

The Advanced Random Seed remains the stochastic experiment seed. GA Random Seed controls population and genetic operators separately. Defaults are 42 for both.

## Common random numbers and caching

The existing seed formula is reused unchanged:

```text
replication_seed = base_seed * 1000 + scenario_index * 100 + run_index
scenario_index: Best=0, Average=1, Worst=2
```

Thus base seed 42 and three search replications produce Best 42000–42002, Average 42100–42102, and Worst 42200–42202. Every candidate in **every generation** uses those same streams. Arrival counts, timestamps and sampled clinical profiles are shared across policy comparisons because they are generated before resource-dependent patient flow. Ten-run final verification uses 42000–42009, 42100–42109, and 42200–42209. It extends the fixed experiment; it is not an independent held-out demand experiment.

Candidate results are memoized by resource genes and fixed RF threshold inside one search instance. That instance fixes demand, targets, priorities, weights, bounds, patient profiles and seeds, so identical candidates safely reuse identical evaluations. The cache is not shared across changed experiments. RF inference uses the existing cached profile loader. No multiprocessing or per-replication disk I/O was added.

## Exact fitness formula

Higher fitness is better; negative fitness is allowed. Scenario weights default to **Best 1, Average 2, Worst 3** and are normalized by their sum. All weights are positive and configurable.

For replication `r` in scenario `s`, define:

```text
a_r = 1 if the deterministic run verdict is ACCEPTABLE, otherwise 0
f_r = number of failed conditions among the six deterministic target checks
m_r = mean_wait (minutes)
h_r = high_risk_mean_wait (minutes)
p_r = p95_wait (minutes)
q_r = unfinished_patients / max(1, patients_arrived)

e_m = max(0, m_r - mean_wait_target) / max(1, mean_wait_target)
e_h = max(0, h_r - high_risk_wait_target) / max(1, high_risk_wait_target)
e_u = sum over ICU, General, doctors, nurses of max(0, utilization - U)
      / max(0.01, U)

H = 2 if Prioritize high-risk patients is enabled, otherwise 1
K = 100 if Prioritize resource efficiency is enabled, otherwise 50

run_score_r = 10000*a_r - 2000*f_r
              - 4000*(e_m + H*e_h + e_u)
              - 2*m_r - 4*H*h_r - 0.1*p_r - 5*q_r

scenario_score_s = mean(run_score_r in scenario s)
                   - 2000*max(0, robustness_threshold - scenario_robustness_s)/100

C = 0.35*ICU + 0.15*General + 1.20*doctors + 0.55*nurses
Cmax = same expression using each user-entered upper bound

fitness = 100000*I(all three scenario verdicts are ACCEPTABLE)
          + sum_s(normalized_scenario_weight_s * scenario_score_s)
          - K*C/Cmax
```

`U` is the configured utilization fraction. Resource costs are transparent relative coefficients, not monetary estimates. Every contribution is retained in `fitness_components` for unique evaluated candidates. The normalized resource penalty is at most 100 within bounds, while robustness and target-failure terms operate at thousands and a robust-policy bonus is 100000. Cheap severe queues cannot win by small resource savings; this is explicitly tested. Both priority flags affect fitness only, not the simulator or authoritative targets. P95 is a small secondary penalty. Unfinished fraction has a small bounded penalty because long clinical stays legitimately extend beyond the observation window; throughput is reported without adding another correlated fitness term.

## Search history and final verification

Flat history contains each generation/candidate/scenario observation, all four genes, complete demand/target context, three scenario robustness values, overall robustness/status, scenario verdict, scenario mean metrics (waits, P95, utilizations, throughput, end counts), fitness, and cache-hit status. JSON also preserves each unique policy's full search evaluation, including per-replication metrics, seeds and deterministic condition breakdowns.

After selecting the highest **search** fitness, the GA evaluates that recommendation **once at the full normal replication count**. The current policy is also evaluated at the same full count and streams for comparison; if it is the recommendation, the verified result is reused. Final verification does not re-rank candidates or replace search fitness. The UI prominently displays **verified** metrics and statuses, not search estimates. A verified recommendation may remain unacceptable/conditional; no successful status is forced.

## Decision Tree role and labels

Each training row comes from an actually evaluated unique resource-policy/scenario observation. Labels are its deterministic **scenario ACCEPTABLE/UNACCEPTABLE verdict**, based on observed acceptable run percentage versus the configured robustness threshold. The trainer checks every stored run verdict/condition breakdown against the existing `policy_verdict` and checks scenario robustness before accepting a label. Inconsistent history is rejected. Elite/cache repetitions are deduplicated so they do not inflate the training sample count.

Features are ICU beds, General beds, doctors, nurses, simulation duration and that scenario's arrival rate. No wait/utilization outcome is used as an input, avoiding trivial leakage of the acceptance conditions into the feature set. Targets are fixed within each experiment and bound through provenance.

Both classes are required. With one class, the UI reports: “Decision Tree explanation unavailable because all evaluated policies received the same deterministic class.” No classifier is trained; a V2 bundle with `tree=None` replaces any older V2 tree bundle.

The tree defaults to depth 4, minimum leaf samples 2, balanced classes and a deterministic seed. Reported metrics include unique policy count, sample count, class distribution, actual depth/leaves and training accuracy. A validation split is attempted only with at least four unique policies and both classes present in both train and validation. Entire resource policies are grouped, preventing scenarios from one resource configuration leaking across partitions. Up to twenty deterministic group splits are considered. If no valid split exists, validation accuracy is explicitly unavailable.

The explanatory tree is finally trained on all unique observations; validation accuracy comes from a separate tree fitted on the eligible training partition. Validation is limited to held-out policies within the same demand experiment, not independent stochastic streams or a new hospital. The UI explicitly avoids treating high training accuracy as generalization proof.

After final verification, recommended genes and each demand context pass through the learned tree. Rules and surrogate predictions are shown separately from **AUTHORITATIVE VERDICT**, which always comes from full Digital Twin verification. Surrogate disagreement is allowed and visible.

## Provenance and stale protection

GA provenance includes duration, all rates, normal and GA replications, stochastic base seed and seed strategy, GA seed, all bounds, current policy, wait/utilization targets, robustness threshold, priorities, weights, population, generations, mutation/elites/early-stop settings, and dependency paths/hashes. Dependencies cover patient data, RF model, simulator/helper code, GA and XAI code. An in-memory profile hash additionally audits direct DataFrame callers.

A deterministic `run_id` hashes the complete experiment result before runtime is attached. Repeated identical inputs reproduce all outputs except measured runtime. XAI binds to this run ID, complete GA provenance, and the hash of its audited training data.

The dashboard compares current controls and dependency fingerprints before showing a saved recommendation. Changes mark GA/XAI as stale and hide their metrics, rules and review actions. Reverting exactly to the saved experiment can restore its compatible result. Recommendation buttons and status headers render after compatibility checks, avoiding a stale enabled action or an “Optimizing” header after completion.

## Measured example

Real-data validation used all 7,500 cached RF profiles, unchanged default demand/targets/base seed, current policy **210/280/85/100**, editable default bounds, population **12**, generations **6**, three search replications per scenario and ten final verification replications per scenario. Priority flags were both enabled.

Recommendation: **298 ICU beds, 583 General beds, 136 concurrent doctors, 212 concurrent nurses**.

| Scenario | Current robustness | Verified recommendation robustness | Verified mean wait | Verified high-risk mean wait |
|---|---:|---:|---:|---:|
| Best | 100% | 100% | 0.00 min | 0.00 min |
| Average | 100% | 100% | 0.00 min | 0.00 min |
| Worst | 0% | 50% | 5.74 min | 13.12 min |

Overall robustness improved **66.67% -> 83.33%**. Verified status remained **CONDITIONALLY ACCEPTABLE**, since Worst robustness was below the 90% threshold. The search estimate was 88.89% overall; the UI correctly shows the lower verified result. Search fitness was **6091.25**.

The search completed six generations in approximately **29.99 seconds**, evaluated 46 unique policies, and reused 26 cached comparisons. Full JSON/CSV evidence is under `results/genetic_algorithm_v2/`. An initial smaller 8-policy/4-generation check reduced delay but had not improved discrete robustness; increasing the search budget used the same fixed seeds and objective.

XAI used **138 rows** from 46 unique policies: 76 ACCEPTABLE and 62 UNACCEPTABLE observations. Actual depth was 4 with six leaves. Training accuracy was **98.55%**. Policy-held-out validation accuracy was **97.22%**, on 36 rows with 102 training rows. These are within-experiment fit/validation metrics with the limitations above, not proof that the tree determines policy acceptability.

## Validation and commands

```powershell
python -m unittest discover -s tests -v
python tests/validate_ga_v2.py
python tests/validate_ga_v2.py --reuse
python tests/validate_v2_dashboard.py
python main.py

# V2 entry points; optional JSON configuration uses current_policy, bounds,
# scenario_configuration and ga_configuration.
python src/genetic_algorithm_v2.py --configuration experiment.json
python src/decision_tree_xai_v2.py

# Existing V1 entry points remain unchanged.
python src/genetic_algorithm.py --help
python src/decision_tree_xai.py --help
```

All **19** fast tests passed (about one second): the eight existing Part 1 tests plus eleven GA/XAI tests. They cover Generation 0 current-policy inclusion, 500 bounded integer operator trials, unchanged RF threshold, shared arrival/profile streams, exact reproducibility except runtime, no DataFrame mutation, overloaded-versus-adequate fitness, both priority effects, memoization, early stopping, full verification once, search count 1–10 without relaxing Part 1 validation, bounds errors without policy changes, audited XAI labels, single-class fallback, and GA/XAI compatibility invalidation.

Real-data/dashboard checks passed final verification at ten runs/scenario, current/recommended rendering, demand-change stale-output hiding, explicit invalid-bound rejection, Run Optimization, Use Recommended Values, and both V2 CLIs. The existing dashboard smoke test passed current-policy evaluation and invalid-order validation. The unchanged actual `python main.py` launcher returned HTTP 200 from Streamlit health; its test server was then stopped.

## Part 3

The following was the Part 2 handoff; it is now implemented in [Part 3](V2_PART3.md). The V2 LLM/feedback path consumes verified scenario comparisons, the deterministic change summary, current provenance, and explicitly labeled surrogate rules. It should translate supported constraints into the existing V2 configuration and require normal bound/target validation; it must not replace Digital Twin verdicts. Search-budget tuning and independent stochastic experiment validation can be considered separately; no GA guarantees that a short search will find a robust or globally optimal policy.

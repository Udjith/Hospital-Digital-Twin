# V2 presentation resource calibration

Only presentation resource defaults and Random Seed help were changed. Arrival generation, profile sampling, RF prediction/threshold, observation-window behavior, targets, acceptance rules, robustness formulas, seed derivation, and GA/XAI/LLM behavior remain unchanged. Existing simulation artifacts were preserved.

## Fixed experiment

All 17 policies used the existing V2 simulator and saved RF model with all 7,500 source profiles. Each policy was tested at 24 hours, Best/Average/Worst = 12/22/35 patients per hour, 10 replications per scenario, wait targets 20/10 minutes, utilization limit 90%, robustness threshold 90%, and base seed 42. This is 510 replications in total. The same scenario/run seeds were shared across policies for controlled comparisons.

The sweep started with eight policies in the suggested ranges, then five larger policies after identifying insufficient staff capacity, then four comparisons around the first useful normal-demand result. No rates, targets, probabilities, or acceptance conditions were tuned.

## Policies tested

Resource order is ICU beds / General beds / Concurrent doctors / Concurrent nurses. Robustness columns are percentages. All rows use the fixed experiment above.

| Resources | Best | Average | Worst | Overall |
|---|---:|---:|---:|---:|
| 70/90/14/20 | 0% | 0% | 0% | 0.00% |
| 90/120/20/25 | 0% | 0% | 0% | 0.00% |
| 110/140/25/35 | 0% | 0% | 0% | 0.00% |
| 130/160/30/40 | 0% | 0% | 0% | 0.00% |
| 150/180/35/50 | 10% | 0% | 0% | 3.33% |
| 150/180/25/35 | 0% | 0% | 0% | 0.00% |
| 150/180/30/40 | 0% | 0% | 0% | 0.00% |
| 130/160/35/50 | 10% | 0% | 0% | 3.33% |
| 150/180/50/50 | 100% | 0% | 0% | 33.33% |
| 150/180/75/90 | 100% | 0% | 0% | 33.33% |
| 180/240/80/100 | 100% | 10% | 0% | 36.67% |
| 200/280/85/100 | 100% | 80% | 0% | 60.00% |
| 220/300/90/110 | 100% | 100% | 0% | 66.67% |
| 210/280/85/100 **selected** | 100% | 100% | 0% | 66.67% |
| 210/290/85/100 | 100% | 100% | 0% | 66.67% |
| 220/300/85/100 | 100% | 100% | 0% | 66.67% |
| 210/280/90/110 | 100% | 100% | 0% | 66.67% |

Full measured means and runtimes for every policy are in [V2_RESOURCE_CALIBRATION.json](V2_RESOURCE_CALIBRATION.json).

## Selection and model basis

New defaults: **210 ICU beds, 280 General beds, 85 concurrent doctors, 100 concurrent nurses**. This policy achieves the requested ACCEPTABLE / ACCEPTABLE / UNACCEPTABLE progression with fewer resources than the larger successful policies tested. It is a measured demonstration default, not a claim that these resources are optimal.

The original suggested staffing range could not support the existing records: even 150/180/35/50 scored only 10% Best robustness and 0% Average/Worst. Source treatment duration averages 197.26 minutes (3.29 hours), so Average demand alone implies about 72 concurrent treatment slots under steady offered load; 35 doctors is insufficient. Mean source LOS is 66.45 hours. RF routes 42.11% of source profiles to ICU, which explains the unusually large ICU component of this modeled hospital. These measurements explain the expanded sweep; they were not used to change simulation logic or introduce automatic capacity sizing.

At 200/280/85/100, Average robustness was 80%. Increasing ICU capacity to 210 while keeping the other resources fixed raised Average robustness to 100%. Additional beds or staff in the larger tested policies did not improve the scenario verdict progression, so they were not selected. The policy represents a large hospital under the existing model; concurrent resources are not total employee counts.

## Selected policy results

All waits are minutes and scenario metrics are means of per-run statistics. Utilization uses the unchanged fixed observation window.

| Scenario | Robustness | Verdict | Mean wait | High-risk mean wait | P95 wait |
|---|---:|---|---:|---:|---:|
| Best | 100% | ACCEPTABLE | 0.00 | 0.00 | 0.00 |
| Average | 100% | ACCEPTABLE | 1.51 | 2.40 | 4.91 |
| Worst | 0% | UNACCEPTABLE | 167.89 | 149.17 | 837.74 |

| Scenario | ICU utilization | General utilization | Doctor utilization | Nurse utilization |
|---|---:|---:|---:|---:|
| Best | 29.3% | 27.6% | 43.1% | 36.6% |
| Average | 52.2% | 49.7% | 77.0% | 65.4% |
| Worst | 63.6% | 57.3% | 88.2% | 74.9% |

**Overall robustness: 66.67% (20/30 acceptable runs). Overall status: CONDITIONALLY ACCEPTABLE.**

Best runs all meet every condition with no queues. Average runs all meet every condition, with mean wait 1.51 minutes and about 9.9 patients still queued at window end. Worst runs all fail both wait targets; one also fails doctor utilization. Worst has about 286.8 patients queued and all 490 beds occupied at window end. This preserves a concrete stress condition for later GA work despite lower bed utilization averaged across an initially empty 24-hour window.

## Seeds and validation

No seed strategy was redesigned. It remains `base_seed * 1000 + scenario_index * 100 + run_index`. With base seed 42:

- Best seeds: 42000 through 42009.
- Average seeds: 42100 through 42109.
- Worst seeds: 42200 through 42209.

All 30 seeds were verified distinct. The first arrival offsets in the first three Best runs were 8.6365, 14.1928, and 1.6171 minutes, confirming different stochastic streams. Re-evaluating the same policy/configuration/base seed reproduced the entire result dictionary exactly, including aggregates and per-run metrics. The base seed does not need to be changed between runs.

The Random Seed control stays in Advanced, defaults to 42, and now uses the requested tooltip explaining automatic derived seeds and repeatable experiments. Existing native question-mark styling remains unchanged.

Eight existing unit tests passed (about 0.49 seconds). Dashboard validation passed synchronized default controls, automatic migration of untouched old defaults, preservation of custom policies, resource reset, and exact agreement with direct simulator evaluation. The actual `python main.py` launcher started Streamlit; a headless browser clicked Evaluate Current Policy and displayed 100% / 100% / 0%, overall 66.7%, CONDITIONALLY ACCEPTABLE. Screenshot inspection verified the resource reset button layout. Test servers were stopped afterward. The original simulation artifact was restored byte-for-byte after browser validation.

Full selected per-run results and seed validation are in [V2_DEFAULTS_VALIDATION.json](V2_DEFAULTS_VALIDATION.json). Measured sweep evaluations took approximately 0.8?2.1 seconds per 30-run policy after RF loading.

## Implementation changes

- `src/scenario_evaluation.py`: shared `DEFAULT_V2_RESOURCES` used by the CLI. No simulation function changes.
- `src/dashboard.py`: shared resource defaults; one-time migration of untouched 30/40/10/20 or stale 1/40/10/20 widget state; Reset Hospital Resources callback/button; exact requested Random Seed tooltip. Custom policies and subsequent intentional edits persist.
- `tests/validate_v2.py`: only the default-policy fixture now reads the shared resource defaults. The explicit 4-hour old-policy fixture remains unchanged.
- `README.md` and `docs/V2_PART1.md`: current defaults and seed behavior documented; original validation examples clearly identified as historical.
- This report and the two associated JSON reports: sweep and validation evidence.

Slider ranges, numeric entry up to 10000 and its overflow behavior, dark theme, demand defaults, and target defaults are unchanged. Resource reset changes only the four resources; Load Legacy GA Baseline remains a separate V1 action. Old default values appear only in explicit migration code or intentionally retained historical validation examples. Existing artifacts, artifact formats, and other tests were not changed.

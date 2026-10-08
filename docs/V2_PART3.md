# V2 Part 3: automatic bounds, Groq and human review

**Availability correction:** automatic bounds are defaults. Explicit hard availability can override them and exclude the old current policy; see [implementation and validation](V2_AVAILABILITY_CONSTRAINTS.md). Preferences still affect priorities. The strict schema is unchanged.

The validated arrival model, fixed-window metrics, RF threshold 0.50, deterministic acceptance rules, robustness formulas, GA fitness/operators, and Decision Tree implementation are unchanged. The GA optimizes hospital process resources through simulation; it does not optimize RF. Part 2's manual bounds are historical and no longer appear as normal sidebar controls.

## Bounds and feedback

`automatic_bounds(policy)` calculates each integer range using exact integer arithmetic:

```python
minimum = max(1, 7 * current // 10)     # floor(0.70 * current)
maximum = (3 * current + 1) // 2        # ceil(1.50 * current)
```

For current ICU/General/doctors/nurses 210/280/85/100, the ranges are **147–315 / 196–420 / 59–128 / 70–150**. They update on every resource change. Collapsed informational expanders show automatic/effective bounds; there are no manual min/max widgets. The V2 GA CLI also derives these bounds by default; explicit CLI ranges remain supported for compatibility.

Explicit numeric availability constraints replace the corresponding default bound, including values outside the automatic range. For example, a 110-doctor cap changes 59-128 to **59-110**; a 60-doctor cap gives **59-60** even though the old current value is 85. A nurse cap of 10 changes 70-150 to **1-10**. When a maximum is below the unspecified automatic minimum, the effective minimum becomes 1; when an explicit minimum exceeds the unspecified automatic maximum, the maximum rises to that minimum. If both endpoints are explicit, they are never repaired: minimum > maximum is rejected. Zero, negative, noninteger and unsupported constraints remain invalid.

The current policy need not satisfy new availability. In this explicit override mode, Generation 0 starts with a coordinate-wise clipped, nearest feasible version, followed by random feasible candidates. The exact original current policy remains the reference comparison and historical results are not changed. With no override, original current-policy inclusion and validation remain. The existing 1-10000 capacity validator is unchanged.

The AI toggle defaults off. With it off, instruction text is ignored and no feedback request is sent. With it on, **Preview AI Interpretation** calls Groq once for the current instruction/context and shows the interpreted bounds, priorities and target adjustments. Run Optimization interprets first if no matching preview exists, validates, then runs GA. A matching interpretation is reused on reruns. Failed requests are cached as failures until the preview is explicitly retried or the context changes; no per-render network calls occur.

## Exact supported structured feedback

All objects reject additional properties and require every listed field. All unrequested adjustment values are `null`.

```json
{
  "status": "applied",
  "summary": "Limit doctors and prioritize high-risk waits and peak demand.",
  "resource_constraints": {
    "icu_beds": {"min": null, "max": null},
    "general_beds": {"min": null, "max": null},
    "doctors": {"min": null, "max": 110},
    "nurses": {"min": null, "max": null}
  },
  "priorities": {
    "high_risk_priority": true,
    "resource_efficiency_priority": null,
    "worst_case_priority": true
  },
  "target_adjustments": {
    "mean_wait_target": null,
    "high_risk_wait_target": null,
    "robustness_threshold": null
  }
}
```

`status` is `applied`, `unsupported` or `ambiguous`; the latter two block optimization until clarified. Bounds are nullable integers. Priorities are nullable booleans. Target adjustments are nullable finite numbers, interpreted as minutes/minutes/percentage and validated using the existing ScenarioConfig. Explicit target adjustments are shown in the preview and used for both current/recommended optimization comparisons and full verification; they do not silently mutate sidebar controls. No utilization, RF, arrival, termination or arbitrary objective fields are supported.

High-risk and efficiency flags feed the existing Part 2 flags (high-risk penalty multiplier 2 versus 1; resource-cost multiplier 100 versus 50). Worst-case priority `true` changes its weight to `max(existing_worst, 2 * max(existing_weights))`: default **1:2:3 -> 1:2:6**. False preserves manually configured scenario weights. Efficiency instructions never implicitly relax waiting-time targets. Strict “below 300” is a maximum of 299; “no more than 300” is 300.

Availability wording ("only available", "maximum", "no more than", "at most", "minimum", "at least") maps to explicit bound fields, which may override defaults. Non-numeric preferences ("try to use fewer nurses", "prefer fewer doctors") set resource-efficiency priority and leave all bound fields null.

The feedback path is **text -> strict interpreter -> validated bounds/objectives/explicit targets -> GA -> Digital Twin evaluation**. It never alters RF, the Decision Tree, verdict algorithms or termination logic.

## Explanations and authority

The provider remains **Groq**, model **openai/gpt-oss-120b**. `.env` loading uses `override=False`, so process environment variables take precedence. Credentials are never included in artifacts, UI or error messages. Groq strict JSON-schema output is also validated locally; requests have a 30-second timeout and no automatic SDK retries. See the provider's [structured output documentation](https://console.groq.com/docs/structured-outputs).

`build_payload_v2` supplies current resources, duration/rates/replications/targets, per-scenario current metrics, recommended resources/deltas, search fitness, full final verification, deterministic change summaries and XAI rules/counts/classes/limitations. Per-replication arrays are removed before sending to reduce token use; scenario aggregates retain robustness, verdicts, waits, P95, utilization, unfinished patients and throughput. No patient records are sent.

The visible **AI Summary** uses deterministic verified facts. Numeric comparisons, final status, failed scenarios and learned rules in the detailed explanation are also generated by the application. Groq adds up to 150 words of qualitative interpretation; prose containing numeric claims is rejected and falls back to the deterministic explanation. Additional conservative guards reject unobserved scenario waiting improvements or claims that a far-below-threshold result is borderline; they can also reject valid prose, leaving the factual fallback available. **Detailed AI Explanation** is collapsed. Optional Groq prose must return the exact authoritative status through an enum restricted to the verified result; it never writes a verdict or fitness. The application always adds a resource-cost tradeoff statement and operational-review requirement. Missing/failed/invalid Groq responses display a non-fatal message and a factual deterministic explanation. The deterministic **Why This Changed** section remains independent and visible.

Authority is explicit: **Digital Twin + operational criteria** determine verdicts; **GA** searches; **Decision Tree** supplies learned surrogate patterns; **LLM** translates supplied evidence. Learned rules and training accuracy are not proof of clinical validity or generalization. Concurrent doctors/nurses are modeled concurrent resources, not total employee counts.

Run acceptance requires both mean-wait checks and ICU, General, doctor and nurse utilization checks. Scenario robustness is `100 * acceptable_runs / runs`; the scenario passes at the configured threshold. Overall robustness pools all three scenarios. Overall status is ROBUST when Worst passes; CONDITIONALLY ACCEPTABLE when Average passes and Worst fails; otherwise UNACCEPTABLE. Part 1's implementation is untouched.

## Provenance and human review

GA provenance additionally records automatic bounds, effective bounds, current-policy feasibility, initialization strategy, availability-override mode, bound strategy and the enabled instruction/interpreted feedback context, alongside effective targets, priorities and bounds. Dependency hashes now include both AI modules. AI explanation provenance binds to the GA run ID/full provenance, XAI hash, prompt version, provider and model. Changes to demand, resources, targets, seeds, GA settings, priorities or feedback invalidate the previous recommendation and AI explanation; stale prose is hidden. Refresh can regenerate prose only for a current verified recommendation. Existing V1 artifacts are not consumed by this path.

Save Review creates a new immutable UUID-named JSON record under `results/feedback/v2_reviews/`, containing the decision, usefulness, comments, GA run ID, full provenance, recommended resources and verified status. Prepare Modification requires Needs Modification plus nonempty comments, transfers them into the sidebar instruction, enables interpretation and records the source run ID. It does not edit historical results or current resources. Preview and a new GA run complete the loop. V1 review files remain untouched.

V2 AI artifacts use `results/llm_v2/explanation.json`. Legacy AI/feedback CLI functions remain V1-compatible; V2 dashboard uses `interpret_feedback_v2`, `apply_feedback_v2`, `generate_explanation_v2`, `save_review_v2` and `prepare_modification_v2`.

## Validation

```powershell
python -m unittest discover -s tests -v
python tests/validate_part3_dashboard.py
python tests/validate_v2_dashboard.py
python tests/validate_part3_experiment.py
```

Initial Part 3 validation passed **33 unit tests** (~1.2 seconds). The availability correction expands this to **41 passing tests**; obsolete widening/current-exclusion expectations were updated, while the 19 Part 1/2 tests remain unchanged. Additional dashboard interaction validation used actual RF profiles/simulator/GA/XAI and mocked/disabled network calls (~12.4 seconds across multiple runs). It checked hidden bounds, dynamic recomputation, disabled-toggle behavior, missing-key continuation, preview/application, validation blocking, full verification, stale outputs, collapsed explanation, review provenance and modification transfer. The existing launcher validation returned HTTP 200 from Streamlit and checked current-policy evaluation and invalid demand ordering.

A real 24-hour experiment used unchanged demand/targets/base seed, automatic bounds, population 6, three generations, three search replications and ten final replications. Search/full verification took **7.99 seconds**; including RF loading and two live Groq calls, total runtime was **10.74 seconds**. This small search retained **210/280/85/100**, with Best/Average/Worst robustness **100%/100%/0%**, overall **66.7%**, status **CONDITIONALLY ACCEPTABLE**. It did not find an improved policy; no improvement is claimed. A larger population-12, generation-6 integration run took **25.05 seconds**, evaluated 51 unique policies (21 cache hits), and recommended **267 ICU / 376 General / 122 doctors / 115 nurses**. Full verification produced Best/Average/Worst robustness **100%/100%/10%**, overall **70.0%**, still **CONDITIONALLY ACCEPTABLE**. Worst mean/high-risk waits improved to **25.34/38.27 minutes**; this remains below the robustness target. Its XAI trained on 153 actual policy/scenario observations. A live explanation explicitly discussed the resource-cost tradeoff. A subsequent final-guard validation rejected the optional prose and correctly supplied the factual fallback, including the tradeoff; `resource_heavy_explanation.json` records that fallback. Results and prose are under `results/part3_validation/ga`, `xai` and `resource_heavy_explanation.json`. Neither GA nor AI guarantees improvement.

Live Groq returned a valid 110-doctor cap and enabled high-risk/worst-case priorities. Effective doctor bounds were 59–110 and weights 1:2:6. The live explanation matched the deterministic status. Separate mocked tests covered malformed JSON, missing/extra fields, wrong types, wrong status and no-key behavior. Evidence is in `results/part3_validation/experiment.json` (no credentials).

Example summary: “Current status: CONDITIONALLY ACCEPTABLE. Worst demand was below the robustness threshold. Overall robustness 66.7% -> 66.7%; verified policy CONDITIONALLY ACCEPTABLE. Worst remains below the threshold.” Detailed prose explains the unchanged recommendation, learned rule and simulation limitations instead of inventing an improvement.

Remaining limits: natural-language interpretation remains probabilistic, so administrators should inspect the preview. Schema validation constrains supported fields, not real-world correctness of free prose. Simulation estimates and resource coefficients do not establish clinical staffing or budget feasibility. No Part 1/2 algorithm redesign or real-world policy deployment was performed.

An isolated headless Chrome check launched the actual dashboard, verified native hover help, hidden manual bound controls and no horizontal overflow, and captured `results/part3_validation/dashboard.png` and `optimization_sidebar.png`. The optional Windows preview script needs Chrome and the `websockets` Python package; neither is a new application dependency.

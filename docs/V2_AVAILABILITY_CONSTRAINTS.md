# V2 availability-constraint correction

Automatic ranges remain `max(1, floor(current * 0.70))` to `ceil(current * 1.50)`, but they are default search ranges, not hard resource limits. The strict feedback schema, Groq provider/model, simulator, arrival model, deterministic verdicts, robustness formulas, fitness formula, genetic operators and Decision Tree logic are unchanged.

## Effective ranges

Explicit numeric availability replaces the corresponding endpoint. Unspecified endpoints retain their automatic defaults unless that would produce an empty range:

- An explicit maximum below the automatic minimum uses minimum **1**.
- An explicit minimum above the automatic maximum raises that maximum to the explicit minimum, giving a valid singleton range.
- When both endpoints are explicit, neither is repaired: minimum greater than maximum is rejected.
- Zero, negative, noninteger, unsupported and greater-than-10000 resource values remain invalid under existing capacity validation.

Examples with current policy ICU 210 / General 280 / doctors 85 / nurses 100:

| Instruction | Automatic range | Effective range |
|---|---|---|
| Only 10 nurses are available | 70–150 | **1–10** |
| Do not use more than 60 doctors | 59–128 | **59–60** |
| ICU capacity cannot exceed 150 beds | 147–315 | **147–150** |
| At least 50 General beds must remain available | 196–420 | **50–420** |
| At least 500 General beds | 196–420 | **500–500** |
| Try to use fewer nurses | 70–150 | **70–150**, resource-efficiency priority enabled |
| At least 120 doctors but no more than 100 | 59–128 | **Validation error** |

The interpreter prompt distinguishes explicit availability wording (only available, maximum, minimum, cannot exceed, no more than, at most, at least) from nonnumeric preferences. Preferences never invent a hard resource count. All unrequested resource constraints remain null. The JSON schema itself is unchanged.

## Infeasible old current policy

The dashboard enables `allow_infeasible_current=True` only for successfully validated explicit resource constraints. The GA then validates the effective ranges independently of the old current policy. Its first Generation 0 member is the coordinate-wise nearest feasible version of that policy; remaining members are random feasible policies. With the 10-nurse limit, that first member is **210 / 280 / 85 / 10**. Every evaluated search candidate, crossover result, mutation and recommendation stays within effective bounds.

The original policy remains **210 / 280 / 85 / 100** in provenance and in the reference simulation comparison. No historical result is edited, and the dashboard does not replace its existing current-policy evaluation when the reference is now infeasible. Preview text explicitly shows automatic range, stated availability, effective range, old current value and the warning that optimization searches feasible configurations only. The factual AI summary also marks the old policy as an infeasible reference; the LLM receives this availability context.

Without explicit overrides, original strict validation/current-policy inclusion remains unchanged. Programmatic callers can opt in using the `allow_infeasible_current` keyword on `run_ga_v2` and `result_is_current`; the V2 CLI accepts the same boolean in its configuration JSON. This keeps earlier direct callers/tests compatible.

## Provenance and stale protection

GA provenance now explicitly includes `automatic_bounds`, `effective_bounds` (also retained in the existing `bounds` field), `allow_infeasible_current`, `current_policy_feasible` and `initialization_strategy`. Existing dependencies retain the instruction/interpreted feedback context. Compatibility checks compare effective bounds, mode, resources, settings and feedback. Changing availability or disabling interpretation invalidates the previous recommendation/XAI/AI output. Cached feedback interpretations are versioned so an interpretation from the previous tightening-only prompt is not reused.

## Files changed

- `src/feedback_interpreter.py`: availability/preference prompt and explicit endpoint override validation.
- `src/genetic_algorithm_v2.py`: opt-in infeasible-current validation, feasible initialization, provenance and CLI compatibility. Fitness and evolution loop are unchanged.
- `src/dashboard.py`: override mode, detailed preview/warning, historical-reference preservation and stale compatibility.
- `src/llm_explanation.py`: supplies availability context and an infeasible-reference factual summary; provider/model and explanation logic remain unchanged.
- `tests/test_availability_constraints.py`: eight new fast tests.
- `tests/test_part3.py`: replaces two obsolete rejection expectations with the corrected availability semantics.
- `tests/validate_part3_dashboard.py`: tests the full 10-nurse override workflow.
- `tests/validate_availability_constraints.py`: live interpreter examples plus a small real-data constrained search.
- `README.md`, `docs/V2_PART2.md`, `docs/V2_PART3.md`, and this document: current semantics and validation.

## Validation

```powershell
python -m unittest discover -s tests -v
python tests/validate_part3_dashboard.py
python tests/validate_availability_constraints.py
```

All **41 unit tests** passed in **1.30 seconds**, including all 19 unchanged Part 1/2 tests. New checks cover both override directions, 10 nurses, 60 doctors, preference-only behavior, feasible Generation 0 and evolution, contradictory/zero/negative/unknown constraints, unchanged reference results, both provenance ranges, and effective-bound/feedback/mode staleness.

Streamlit interaction validation passed in **13.59 seconds** across several small real RF/simulation/GA/XAI runs with Groq mocked or disabled. It confirmed the preview/warning, 1–10 nurse range, successful optimization, all feasible candidates, preserved previous evaluation, reference-aware summary, review workflow, stale-output hiding and contradictory-input blocking.

Live Groq interpretation passed all four examples: 10 nurses, 60 doctors, fewer-nurses preference, and contradictory doctor limits. The subsequent real-data 4-hour constrained GA ran successfully in **2.08 seconds**, with **9.81 seconds** total including the four live interpretations and RF loading. Its first candidate and recommendation used **210 / 280 / 85 / 10**, every search candidate was feasible, and the verified status was **UNACCEPTABLE**. This does not guarantee an acceptable policy under such low capacity; the deterministic verdict remains authoritative. Measured output is saved separately in `results/part3_validation/availability_constraints.json` without credentials.

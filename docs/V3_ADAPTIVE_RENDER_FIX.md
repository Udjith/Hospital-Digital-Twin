# V3 adaptive optimization rendering fix

## Reproduction

Before application source edits, native Chrome launched the actual `python main.py` route, started Warm Start/autoplay, interpreted a controlled compound scenario (60 additional patients over 20 minutes plus 70 unavailable doctors for two hours), paused for a stable forecast, ran scenario Look-Ahead and clicked Optimize Scenario Operating Policy. The actual GA executed with default search settings. Chrome reported:

`Error: Bad delta path index 56 (should be between [0, 52])`

The stack contained `visitBlockNode`, `setNodeAtPath`, `addElement` and `applyDelta`. The page body became empty. Server instrumentation showed `SCENARIO_GA_COMPLETED`, outcome PARTIAL IMPROVEMENT, verdict CANNOT HANDLE, before blanking, with the identical live clock/hash before and after search. There was no Python application traceback. The captured fragment ID was `25fb4bdb735a1fca1d62d51c7412eb14`; it is diagnostic, not a hard-coded application identifier. Compact original evidence: `results/live_twin/optimize_browser_reproduction.json`.

## Root cause and structural fix

This was the same frontend render-tree failure class as interpretation: conditional GA progress and large result output added siblings into the live fragment, shifting subsequent blocks onto formerly occupied render paths. The optimization spinner also used transient feedback.

`render_adaptive_optimization` now always creates a keyed permanent parent and these ordered keyed child anchors:

1. action: settings, context, preview, action/status and a synchronous progress placeholder
2. outcome: current vs optimized summary or stale warning
3. diagnosis: deterministic evidence, fixed three-column metrics and diagnostic details
4. xai: learned insight/path or explicit unavailable state
5. explanation: existing AI/fallback summary, detailed prose and human-review responses
6. decision: Apply / Reject / Modify and policy-duration choice
7. technical: verified metric/pressure comparisons and search details
8. controls: manual operating-policy controls and restoration
9. history: provenance/audit and bounded policy/recommendation history

All anchors exist with no result, current/stale results, every result category and action errors. Only their contents vary. The settings keep their original three columns; diagnosis always reserves three columns, with an empty label placeholder when no limiting condition exists. No nested fragments were introduced. Modify/Retry notice also owns a permanent placeholder so it cannot shift the event/adaptive sections.

GA progress, completion and errors replace content at the same action placeholder. Expected/unexpected action exceptions are logged and visibly displayed as Adaptive Optimization Error with details and a retry path. Reset clears these presentation status/error fields.

## Playback and authority

GA, initial XAI/explanation commitment and explicit Groq explanation requests use the existing synchronous checkpoint guard. The driver is temporarily disabled, an independent clone is evaluated, results are committed, and finally the driver is reanchored/restored. Widget-backed autoplay/speed values are untouched. Existing catch-up debt is retained; synchronous computation latency is excluded. The hospital lifecycle state is not paused or mutated by the request. A paused session remains paused; a running session retains its setting.

Algorithms, fitness, genes, common random numbers, fixed capacities, look-ahead, diagnosis, XAI, Groq selection/fallback and authority remain unchanged. A regression compares the UI-generated result with direct unchanged GA output for policy, fitness, verified metrics and provenance. Normal current-state stale checks remain authoritative and prohibit applying an old recommendation.

## Validation

Eight new AppTest tests use actual evaluated policies and cover VERIFIED SUCCESS, PARTIAL IMPROVEMENT, NO MEANINGFUL IMPROVEMENT, single-class XAI, Groq success/failure, guarded running GA, expected/unexpected errors, Retry, Reject, Apply, policy changes and live advancement. Permanent anchor paths are identical in all tested states. Both successful and failed result layouts retain six columns: three settings plus three diagnosis columns.

Native main.py Chrome regression completed Warm Start/autoplay, interpretation, risky Look-Ahead, scenario GA, visible diagnosis/XAI/fallback, Reject, Retry, a second GA run, Apply and +5-minute Fast Forward. No blank page or browser console errors occurred. Search traces prove unchanged actual clock/hash and guarded driver state. The existing paused-for-stable-review workflow was used for forecast/decision review; autoplay was not disabled in the application. RUNNING request preservation is separately tested with the clock guard. Compact evidence: `results/live_twin/optimize_browser_validation.json`.

The native interpretation regression also passed. Complete primary UI/real-profile validation covers Warm Start, look-ahead, GA, XAI/fallback, Reject/Retry/Apply, preserved ownership/capacity, temporary restoration and Reset; shared warm/pressure/legacy regression validators passed. Evidence: `results/live_twin/optimize_workflow_validation.json`.

Run `python tests/reproduce_live_optimize.py --fixture --verify` for the native scenario regression (dedicated ports 18523/19228, controlled provider-shaped scenario output, actual unchanged GA). Temporary browser/server files remain outside the repository and owned processes are cleaned up. No model/dataset copies or screenshots are saved.

Final full regression: **198 tests and 8 equivalence subtests passed in 43.15 seconds** (baseline 190 tests plus eight new UI regressions).

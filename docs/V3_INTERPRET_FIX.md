# V3 scenario interpretation rendering fix

## Reproduction and root cause

The unmodified `python main.py` route was exercised in Chrome with Warm Start and the documented bus scenario. The browser reported `Error: Bad delta path index 3 (should be between [0, 2])` (`visitBlockNode` / `setNodeAtPath` / `addElement`) and the body became empty. The server emitted no Python/Streamlit exception. This was a frontend fragment layout failure, not a schema rejection or a widget-backed session-state write.

The variable-length interpretation section inserted preview/status elements into the same fragment level as the following manual/adaptive sections. Subsequent periodic rerenders shifted these sections, including column blocks, onto previously occupied delta paths. A permanent parent container now isolates scenario content; its insertion/removal cannot move those following sections. A regression asserts their paths remain identical before interpretation, after preview and after Cancel. Synchronous placeholder feedback also avoids delayed spinner timer/clear deltas in this action.

## Playback and read-only boundary

Interpretation captures a complete independent hospital clone. A non-widget synchronous-action flag temporarily guards the clock driver. The request does not call hospital pause/resume, change the playback toggle, advance the hospital, alter ownership or consume its RNG. Finally the driver is reanchored and its previous enabled setting restored, including on errors. Request latency is excluded from wall-clock catch-up; existing debt remains. A paused hospital stays paused. The same boundary protects the confirmed scenario look-ahead.

No simulation, GA, XAI, acceptance, RF, event or Groq model/provider behavior changed. No widget-backed scenario text/speed/playback key is assigned after widget creation.

## Provenance and stale forecasts

The existing interpretation checksum, source text, event creation time and live state hash remain intact. Autoplay can make a proposal stale without making it unusable: unchanged checksum-verified event values are deterministically revalidated against the current checkpoint when Run Look-Ahead is clicked. Relative delays are rebased to that checkpoint. Event capacity/schema validation remains strict. Changed text or a changed payload requires new interpretation.

Forecasts still require exact current-state/event/target provenance before event application or optimization. Advancing live state explicitly marks a forecast stale. Pause remains available for reviewing a stable forecast; no old prediction is silently treated as current.

## Errors

Expected provider and schema failures show Scenario Interpretation Error and, when available, Scenario Interpretation Details. Text and Manual Event Builder remain usable. Provider diagnostic categories exclude credentials/headers. Unexpected interpretation-boundary errors are logged with a traceback and displayed visibly; playback restoration runs in finally.

## Validation

- Native main.py route: running at 10x and paused; fast and delayed strict provider-shaped responses. Preview +40 patients over 20m, fresh look-ahead, Edit and Cancel passed with no browser errors/blank page. Boundary traces proved unchanged actual clock/hash and matching independent checkpoint during interpretation.
- Real configured Groq smoke test: the exact bus scenario produced the correct preview; Look-Ahead, Edit and Cancel passed. No credentials retained.
- Native broader workflow: warm-start autoplay, Fast Forward, manual event preview/application, adaptive GA, explicit policy application, stale protection, 3600x Pause and Reset passed.
- Ten new AppTest regressions: running delayed request, paused request, provider error, malformed response, strict semantic error, unexpected boundary exception, Edit/Cancel, clock-advanced proposal/fresh forecast, stable delta paths, changed text.
- Final suite: 190 tests and 8 equivalence subtests passed (55.06 seconds).

Run `python tests/reproduce_live_interpret.py --fixture` for controlled native regression, add `--paused` for paused playback. `SCENARIO_FIXTURE_DELAY` selects a fast/delayed response. Without `--fixture`, this uses the configured Groq provider. Temporary browser/server files live outside the repository and owned processes are cleaned up. Compact evidence is in results/live_twin/interpret_*.json.

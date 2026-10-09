# Final V3 navigation and card cleanup

This pass changes presentation only. Simulation, RF, warm-start generation,
event scheduling, pressure acceptance, GA, diagnosis, XAI, Groq, playback,
Fast Forward and policy application/restoration retain their existing behavior.

## Layout

- The permanent `v3_top_status` parent is the single status card. Its clock,
  resource bars, counts and keyed **Expand Status** panel share one surface.
  Expanded metrics remain inside that parent; expansion is browser-only.
- **Fast Forward Controls** combines the existing custom duration, preset
  buttons, progress and cancellation controls. Engine inputs and execution
  remain unchanged.
- **Initial Hospital State Details** renders retained configuration, lifecycle
  counts and resource availability as readable groups. Notes preserve the
  paired-staff warning. Seed strategy, checksum and a read-only initialization
  hash are under **Advanced Initialization Provenance**. The entire original
  initialization object remains under **Advanced Raw Initialization Data**.
- Scenario examples are muted inline guidance. Manual and random event tools
  have a separate **Stress Event** destination. **Active Events** shows counts
  and current event status, with full rows/history retained in collapsed panels.
- A lower **Technical Details** area groups forecast, pressure, search, tree,
  audit, patient and resource evidence. Four permanent child slots prevent
  variable-length results from shifting subsequent widget paths.

## Navigation

The sidebar is a borderless navigation rail with six ordinary HTML links:
`#live`, `#stress-event`, `#scenario`, `#policy`, `#active-events`,
`#technical-details`. All six destinations exist before Start and after reruns.
Links target the current browser page and have no callbacks, state writes or
experiment actions. Operational targets use their existing widget keys,
defaults and ranges within Hospital Setup. Historical V2 access remains opt-in
at the bottom of the rail.

## Validation

`tests/test_live_ui_cleanup.py` checks local links, read-only retained values,
permanent anchors, unified controls, collapsed raw evidence, unchanged target
keys and scenario-text persistence across Pause/Resume.

`tests/validate_dashboard_cleanup.py --fixture --verify --paused` launches
`python main.py` and native Chrome. It checks all six jumps without clock
advancement, initialization panels, combined Fast Forward, scenario preview,
Look-Ahead, real adaptive GA, explanation/fallback, Reject/Retry/Apply, status
expansion/collapse and viewport widths 1920, 1366 and 1200 pixels. Scenario
interpretation uses strict provider-shaped JSON fixtures; no external API call
is required. Screenshots and Chrome profiles stay in the system temporary
directory. Compact reports are in `results/live_twin/cleanup_*.json`.

Existing primary-UI validation also checks temporary policy restoration,
ownership, fixed capacities, Reset and retained historical V2 access. Native
playback and exact bus-scenario checks complement the full regression suite.

Final validation: **211 tests passed plus 8 equivalence subtests**, in 71.08
seconds. Native Chrome completed interpretation, forecasting, adaptive GA,
Reject/Retry/Apply, Fast Forward and expanded status with **zero console errors**.
All six sidebar links preserved the paused clock. Document widths matched
1920×1080, 1366×768 and 1200×800 viewports; screenshots were visually inspected.
The exact bus scenario passed under autoplay, including Look-Ahead, Edit and
Cancel, with identical live/checkpoint hashes during interpretation.

The separate native playback/manual-event check passed Start, Pause/Resume,
speed changes, 3600x Pause, event preview/application, adaptive policy application,
Fast Forward and Reset in 48.92 seconds. The primary-UI demonstration passed
temporary restoration, fixed ownership/capacity and historical V2 access in
17.22 seconds. Groq calls used provider-shaped fixtures; the existing mocked
success/error tests passed with the full suite.

## Files for this pass

- UI: `src/dashboard.py`, `src/live_dashboard.py`, `src/live_event_dashboard.py`,
  `src/live_scenario_dashboard.py`, `src/live_adaptive_dashboard.py`.
- Presentation: `src/live_presentation.py`, `src/assets/operations.css`.
- New regression/evidence harnesses: `tests/test_live_ui_cleanup.py`,
  `tests/validate_dashboard_cleanup.py`.
- Existing harness labels/actions updated: `tests/validate_live_browser.py`,
  `tests/validate_live_policy_browser.py`, `tests/validate_dashboard_design.py`,
  `tests/reproduce_live_optimize.py`.
- Documentation: README and this document.
- Compact evidence: `cleanup_browser_validation.json`,
  `cleanup_interpret_validation.json`, `cleanup_playback_validation.json`,
  `cleanup_workflow_validation.json` under `results/live_twin/`.

No backend algorithms, datasets or model artifacts changed in this pass.

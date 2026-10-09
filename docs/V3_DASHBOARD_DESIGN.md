# Final V3 dashboard presentation

The application uses a calm navy operations dashboard. A large status card leads
with the simulated clock, playback state, speed, resource ownership counts,
utilization bars, active patients, waiting patients and effective arrival rate.
These are simulated operational measurements, not an assessment of clinical safety.

**Expand Status** is collapsed by default. It exposes twelve read-only fields in
three groups: patient flow (treatment, bed stay, arrivals, initial incumbents),
performance (completions, mean wait, high-risk wait, throughput), and operating
state (active events, policy mode, warm start, queue). Initialization evidence
remains in **Initial Hospital State Details**.

The page follows this sequence:

1. Hospital status, setup and playback controls.
2. A compact operational overview and warm-start summary.
3. Scenario entry and interpreted proposal; manual controls remain secondary.
4. Look-Ahead verdict, predictive robustness, limiting factor and target comparison.
5. Current versus optimized operating policy, with changed genes emphasized.
6. Deterministic diagnosis, learned XAI insight and AI explanation/fallback.
7. Human Apply / Reject / Modify decision, with temporary application unchanged.
8. Collapsed pressure, replication, search, tree, patient, ownership, history and audit evidence.

## Implementation

`src/live_presentation.py` contains escaped HTML presentation helpers and reads
`src/assets/operations.css`. The stylesheet is local and is applied only to the
V3 page. Historical V2 experiments retain their own presentation. There are no
new packages, remote fonts, CDN assets or trackers.

Cards share spacing, radii, borders and muted colors. Green, amber and red label
HANDLED, AT RISK and CANNOT HANDLE; transient resource peaks do not determine
these colors. Apply is muted green, major experiment actions are blue, and
Reject / Modify remain secondary. Layouts adapt at 1350, 900 and 600 pixels.

## Render stability and authority

The top status, scenario entry/preview/forecast, and all nine adaptive child
regions have permanent structural anchors. Progress uses stable placeholders.
Widget regions are populated in sequence; conditional results stay within their
parent anchors. Column counts do not change between GA result types. The status
expander has an explicit key and does not request a server rerun on expansion.

The forecast comparison reports average wait/pressure values alongside the
existing verdict. Its queue-recovery row uses verified `recovery_pass` results,
not the diagnostic estimate of recovery time. Full per-run conditions remain
available. The diagnosis, XAI and Groq layers retain their existing authority.

No engine, RF, event, acceptance, GA, XAI, provider, provenance, playback driver,
or Fast Forward algorithm was changed for this redesign. Formatting does not
alter stored values or hashes.

## Validation

The presentation suite adds ten checks for status values, semantic colors,
read-only expansion/overview, escaped provider text, local responsive CSS,
verified forecast values, changed-gene summaries, collapsed evidence and reruns.
Existing adaptive tests also verify fixed human-decision columns.

`tests/validate_dashboard_design.py --fixture --verify` launches `python main.py`
and native Chrome, checks interpretation through GA and human decisions, records
console errors, and captures desktop/laptop screenshots outside the repository
in the system temporary directory. The scenario-provider fixture returns strict
raw JSON; forecasting and GA execute normally. Groq success/error rendering is
also covered by the existing regression suite. Compact evidence is stored in
`results/live_twin/design_browser_validation.json` and
`results/live_twin/design_workflow_validation.json`.

Final results: **208 tests passed, plus 8 equivalence subtests** (68.60 seconds).
Native Chrome completed the compound-scenario forecast/GA/Reject/Retry/Apply
workflow with zero console delta-path errors. Screenshots were inspected at
1920×1080, 1366×768 and 1200×800; document width matched viewport width.
Status expansion and collapse passed without clock advancement while paused.
The exact 40-patient/20-minute bus example also passed in running and paused
sessions, including Look-Ahead, Edit and Cancel, with unchanged checkpoint
hashes across interpretation. See `design_interpret_validation.json`.

The separate native playback/manual-event validation passed Start, Resume,
speed changes, 3600x Pause, Fast Forward, explicit event/policy application and
Reset in 50.50 seconds. The primary AppTest demonstration passed temporary
restoration, fixed ownership/capacity and historical V2 access in 22.90 seconds.
Native scenario calls used raw provider-shaped fixtures; Groq explanation
success/failure remains covered by the existing mocked provider tests.

Files changed for this presentation pass:

- UI: `dashboard.py`, `live_dashboard.py`, `live_event_dashboard.py`,
  `live_scenario_dashboard.py`, `live_adaptive_dashboard.py`.
- New presentation: `live_presentation.py`, `assets/operations.css`.
- Tests: new `test_live_presentation.py`, `validate_dashboard_design.py`;
  updated `test_live_adaptive_ui.py`, `validate_live_browser.py`,
  `validate_live_policy_browser.py`, `reproduce_live_interpret.py`.
- Documentation: README and this document.
- Compact evidence: `design_before.json`, `design_browser_validation.json`,
  `design_workflow_validation.json`, `design_interpret_validation.json`.

Screenshots and owned Chrome profiles stay outside the repository. The only
new visual asset is the approximately 9 KB stylesheet. No datasets or models
were copied or modified.

This is a simulated operational decision-support prototype, not clinical advice
or production EHR integration.

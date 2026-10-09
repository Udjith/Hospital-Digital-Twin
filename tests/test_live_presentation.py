"""Read-only dashboard presentation and permanent render anchors."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from streamlit.testing.v1 import AppTest
from live_presentation import (status_html, expanded_status_html, overview_html,
    verdict_class, explanation_html, forecast_html, policy_comparison_html, operations_css)
from live_pressure import LiveTargets
from live_policy_diagnosis import diagnose_prediction
from test_live_ux import baseline, button
from test_live_diagnosis import search


class PresentationTests(unittest.TestCase):
    def test_status_shows_clock_physical_counts_speed_and_live_badge(self):
        state = baseline()
        html = status_html(state, 10)
        for text in ("Hospital Digital Twin", "Day 1", "10x", "LIVE", "121 / 220", "138 / 550", "16 / 80", "16 / 140"):
            self.assertIn(text, html)

    def test_playback_badges_map_to_actual_state(self):
        state = baseline()
        state.pause()
        self.assertIn("PAUSED", status_html(state, 60))
        self.assertNotIn("● LIVE", status_html(state, 60))
        self.assertIn("FAST FORWARDING", status_html(state, 60, skipping=True))

    def test_verdict_colors_are_semantic(self):
        self.assertEqual([verdict_class(v) for v in ("HANDLED", "AT RISK", "CANNOT HANDLE", None)],
                         ["handled", "at-risk", "cannot-handle", "neutral"])

    def test_expanded_status_has_correct_flow_metrics_without_mutation(self):
        state = baseline()
        before = state.state_hash()
        html = expanded_status_html(state)
        self.assertIn("Initial incumbents remaining</span><strong>259", html)
        self.assertIn("In treatment</span><strong>16", html)
        self.assertIn("Bed stay</span><strong>243", html)
        self.assertEqual(state.state_hash(), before)

    def test_overview_is_read_only(self):
        state = baseline(True)
        before = state.state_hash()
        html = overview_html(state)
        for name in ("Patient Flow", "Resource Load", "Operations", "Current Situation"):
            self.assertIn(name, html)
        self.assertEqual(before, state.state_hash())

    def test_ai_text_is_escaped_and_original_summary_preserved(self):
        summary = '<script>alert("bad")</script> verified evidence'
        html = explanation_html(dict(provider="Groq", summary=summary))
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("Groq explanation", html)
        self.assertIn("deterministic fallback", explanation_html(dict(provider="deterministic", summary="Operational evidence")))

    def test_local_css_has_responsive_cards_and_no_remote_assets(self):
        css = operations_css()
        self.assertIn("@media", css)
        self.assertIn(".status-card", css)
        self.assertNotIn("https://", css)
        self.assertNotIn("@import", css)

    def test_forecast_shows_verified_values_without_recomputing_verdict(self):
        state, targets, options, result = search(True)
        prediction = result["optimized_verified"]
        diagnosis = diagnose_prediction(prediction, targets, len(state._waiting))
        before = state.state_hash()
        html = forecast_html(prediction, diagnosis, targets)
        self.assertIn(prediction["verdict"], html)
        self.assertIn(f'{prediction["robustness"]:.1f}%', html)
        self.assertIn("Doctor pressure", html)
        self.assertEqual(state.state_hash(), before)

    def test_comparison_only_emphasizes_changed_genes(self):
        state, targets, options, result = search(True)
        html = policy_comparison_html(result)
        self.assertIn("CURRENT", html)
        self.assertIn("OPTIMIZED", html)
        self.assertIn("Physical resources unchanged", html)
        changes = html.split("Recommended policy changes", 1)[1]
        for gene in result["current_policy"]:
            self.assertEqual(gene.replace("_", " ").title() in changes,
                             result["current_policy"][gene] != result["optimized_policy"][gene])

    @patch("live_dashboard.wall_seconds", new=lambda: 0.)
    def test_status_expander_and_initial_evidence_are_collapsed_and_stable(self):
        app = AppTest.from_file(str(ROOT / "src/dashboard.py"), default_timeout=60).run()
        button(app, "Start Live Simulation").click().run()
        self.assertFalse(app.exception)
        hospital = app.session_state["v3_live_hospital"]
        before = hospital.state_hash()
        expanded = next(x for x in app.expander if x.label == "Expand Status")
        self.assertFalse(expanded.proto.expanded)
        self.assertTrue(any(x.label == "Initial Hospital State Details" and not x.proto.expanded for x in app.expander))
        for label in ("Active Patient Details", "Inspect Live Resources", "Live Operating Policy and Snapshot"):
            self.assertTrue(any(x.label == label for x in app.expander))
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(hospital.state_hash(), before)
        self.assertTrue(any("status-card" in x.value for x in app.markdown))


if __name__ == "__main__":
    unittest.main()

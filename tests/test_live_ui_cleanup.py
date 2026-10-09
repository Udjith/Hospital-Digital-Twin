"""Presentation-only navigation, unified controls and initialization evidence."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from streamlit.testing.v1 import AppTest
from live_presentation import SECTIONS, sidebar_navigation_html, initial_state_html
from test_live_ux import baseline, button


class CleanupPresentationTests(unittest.TestCase):
    def test_navigation_is_six_local_links_without_actions(self):
        html = sidebar_navigation_html()
        self.assertEqual(len(SECTIONS), 6)
        self.assertEqual(html.count('<a '), 6)
        for anchor, label in SECTIONS:
            self.assertIn(f'href="#{anchor}"', html)
            self.assertIn(label, html)
        self.assertNotIn('<button', html)
        self.assertNotIn('onclick', html)
        self.assertNotIn('javascript:', html)

    def test_initial_details_show_retained_values_without_mutation(self):
        hospital = baseline()
        before = hospital.state_hash()
        html = initial_state_html(hospital.initial_state)
        for text in ('Warm Start Configuration', 'Initial Patient State', 'Initial Resource State',
                     '259', '243', '121', '138', '55%', '25%', '20%', 'Auto', 'Out of service'):
            self.assertIn(text, html)
        self.assertEqual(hospital.state_hash(), before)

    @patch('live_dashboard.wall_seconds', new=lambda: 0.)
    def test_permanent_navigation_and_unified_controls(self):
        app = AppTest.from_file(str(ROOT / 'src/dashboard.py'), default_timeout=60).run()
        for anchor, label in SECTIONS:
            self.assertTrue(any(f'id="{anchor}"' in x.value for x in app.markdown), anchor)
        button(app, 'Start Live Simulation').click().run()
        self.assertFalse(app.exception)
        hospital = app.session_state['v3_live_hospital']
        before = hospital.state_hash()
        labels = [x.label for x in app.expander]
        self.assertEqual(labels.count('Fast Forward Controls'), 1)
        self.assertNotIn('Manual Fast Forward Settings', labels)
        self.assertNotIn('Fast Forward Presets', labels)
        self.assertNotIn('Scenario Examples', labels)
        for label in ('Advanced Initialization Provenance', 'Advanced Raw Initialization Data',
                      'Initial Hospital State Details', 'Expand Status'):
            self.assertTrue(any(x.label == label and not x.proto.expanded for x in app.expander))
        self.assertTrue(any('scenario-examples' in x.value for x in app.markdown))
        self.assertTrue(any('Warm Start Configuration' in x.value for x in app.markdown))
        self.assertTrue(any('Technical Details' in x.value for x in app.markdown))
        for key in ('mean_wait_number', 'high_risk_wait_number', 'live_utilization_target', 'live_robustness_threshold'):
            self.assertTrue(any(x.key == key for x in app.number_input), key)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(hospital.state_hash(), before)
        button(app, 'Pause').click().run()
        app.text_area(key='v3_scenario_text').set_value('A bus accident sends 40 patients over 20 minutes').run()
        button(app, 'Resume').click().run()
        self.assertEqual(app.text_area(key='v3_scenario_text').value, 'A bus accident sends 40 patients over 20 minutes')
        for anchor, label in SECTIONS:
            self.assertTrue(any(f'id="{anchor}"' in x.value for x in app.markdown), anchor)


if __name__ == '__main__':
    unittest.main()

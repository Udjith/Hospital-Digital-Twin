"""Fragment-safe scenario actions preserve playback, text and strict provenance."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from streamlit.testing.v1 import AppTest
from test_live_scenarios import event,payload
from test_live_ux import button
TEXT='A bus accident sends 40 patients over 20 minutes'

class InterpretationUITests(unittest.TestCase):
    def setUp(self):
        self.wall=[0.]
        self.clock=patch('live_dashboard.wall_seconds',new=lambda:self.wall[0]);self.clock.start();self.addCleanup(self.clock.stop)
        self.env=patch.dict(os.environ,{'GROQ_API_KEY':'fixture'});self.env.start();self.addCleanup(self.env.stop)
        self.provider=patch('live_scenario_interpreter.Groq');self.groq=self.provider.start();self.addCleanup(self.provider.stop)
        self.groq.return_value.chat.completions.create.return_value.choices=[MagicMock(message=MagicMock(content=json.dumps(payload(event(count=40)))))]
        self.app=AppTest.from_file(str(ROOT/'src/dashboard.py'),default_timeout=60).run()
        button(self.app,'Start Live Simulation').click().run()
        self.state=self.app.session_state['v3_live_hospital']
        self.app.text_area(key='v3_scenario_text').set_value(TEXT).run()

    def interpret(self):
        before=self.state.state_hash()
        button(self.app,'Interpret Scenario').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(before,self.state.state_hash())
        self.assertEqual(self.app.text_area(key='v3_scenario_text').value,TEXT)
        self.assertFalse(self.app.session_state['v3_synchronous_action'])

    def test_running_delayed_request_has_independent_checkpoint_and_restores_playback(self):
        original=self.groq.return_value.chat.completions.create.return_value
        def request(**kwargs):
            self.assertTrue(self.app.session_state['v3_synchronous_action'])
            self.assertFalse(self.app.session_state['v3_clock_driver'].enabled)
            self.assertEqual(self.state.status,'RUNNING')
            self.wall[0]=30.
            return original
        self.groq.return_value.chat.completions.create.side_effect=request
        import live_scenario_dashboard
        real=live_scenario_dashboard.interpret_scenario
        def independent(text,checkpoint):
            self.assertIsNot(checkpoint,self.state)
            self.assertEqual(checkpoint.state_hash(),self.state.state_hash())
            return real(text,checkpoint)
        with patch('live_scenario_dashboard.interpret_scenario',side_effect=independent),patch('live_scenario_dashboard.st.spinner',side_effect=AssertionError('unsafe spinner')):
            self.interpret()
        self.assertTrue(self.app.session_state['v3_autoplay'])
        self.assertTrue(self.app.session_state['v3_clock_driver'].enabled)
        self.app.run();self.assertEqual(self.state.sim_time_minutes,0.)
        self.wall[0]=36.;self.app.run();self.assertEqual(self.state.sim_time_minutes,1.)
        self.app.run();self.assertEqual(self.state.sim_time_minutes,1.)
        self.assertTrue(any('+40 patients over 20m' in e.value for e in self.app.markdown))

    def test_paused_interpretation_remains_paused(self):
        button(self.app,'Pause').click().run();self.interpret()
        self.wall[0]=60.;self.app.run()
        self.assertEqual(self.state.status,'PAUSED');self.assertEqual(self.state.sim_time_minutes,0.)
        self.assertTrue(any('SCENARIO INTERPRETATION' in e.value for e in self.app.markdown))

    def test_provider_error_keeps_page_text_and_manual_builder(self):
        self.groq.return_value.chat.completions.create.side_effect=RuntimeError('provider offline')
        self.interpret()
        self.assertEqual(self.app.session_state['v3_scenario_interpretation']['status'],'UNAVAILABLE')
        self.assertTrue(any(e.label=='Advanced / Manual Event Builder' for e in self.app.expander))
        self.assertTrue(any('Scenario Interpretation Error' in e.value for e in self.app.markdown))
        self.assertTrue(self.app.session_state['v3_clock_driver'].enabled)

    def test_malformed_response_has_visible_precise_error(self):
        self.groq.return_value.chat.completions.create.return_value.choices[0].message.content='not JSON'
        self.interpret()
        self.assertEqual(self.app.session_state['v3_scenario_interpretation']['status'],'INVALID')
        self.assertTrue(any(e.label=='Scenario Interpretation Details' for e in self.app.expander))

    def test_semantic_validation_stays_strict(self):
        invalid=payload(event(count=40));invalid['events'][0]['parameters']['arrival_rate']=2
        self.groq.return_value.chat.completions.create.return_value.choices[0].message.content=json.dumps(invalid)
        self.interpret()
        self.assertIn("unsupported non-null parameter 'arrival_rate'",self.app.session_state['v3_scenario_interpretation']['technical_reason'])

    def test_unexpected_boundary_error_is_visible_and_restores_driver(self):
        with patch('live_scenario_dashboard.interpret_scenario',side_effect=RuntimeError('unexpected fixture error')):
            self.interpret()
        self.assertEqual(self.app.session_state['v3_scenario_interpretation']['status'],'ERROR')
        self.assertIn('RuntimeError',self.app.session_state['v3_scenario_interpretation']['technical_reason'])
        self.assertTrue(self.app.session_state['v3_clock_driver'].enabled)

    def test_edit_and_cancel_preserve_text_and_live_state(self):
        for label in ('Edit Scenario','Cancel'):
            self.interpret();before=self.state.state_hash()
            button(self.app,label).click().run()
            self.assertFalse(self.app.exception);self.assertEqual(before,self.state.state_hash())
            self.assertEqual(self.app.text_area(key='v3_scenario_text').value,TEXT)
            self.assertNotIn('v3_scenario_interpretation',self.app.session_state)
            self.assertTrue(self.app.session_state['v3_autoplay'])

    def test_clock_advance_preserves_proposal_but_requires_current_forecast(self):
        self.interpret();initial=self.app.session_state['v3_scenario_interpretation']
        self.wall[0]=6.;self.app.run()
        self.assertTrue(any('STALE scenario interpretation' in w.value for w in self.app.warning))
        self.assertFalse(button(self.app,'Run Look-Ahead').disabled)
        before=self.state.state_hash()
        button(self.app,'Run Look-Ahead').click().run()
        self.assertFalse(self.app.exception);self.assertEqual(before,self.state.state_hash())
        forecast=self.app.session_state['v3_scenario_preview']
        self.assertEqual(forecast['provenance']['live_state_hash'],before)
        updated=self.app.session_state['v3_scenario_interpretation']
        self.assertNotEqual(initial['interpretation_hash'],updated['interpretation_hash'])
        self.assertEqual(updated['parsed_events'][0]['created_sim_time'],self.state.sim_time_minutes)
        self.assertEqual(updated['parsed_events'][0]['parameters'],{'count':40})
        self.assertEqual(self.groq.return_value.chat.completions.create.call_count,1)
        self.wall[0]=12.;self.app.run()
        self.assertTrue(any('STALE scenario look-ahead' in w.value for w in self.app.warning))
        self.assertFalse(any(b.label=='Apply Scenario Events' for b in self.app.button))

    def test_scenario_results_keep_manual_and_adaptive_delta_paths_stable(self):
        def paths(node,path=()):
            found={}
            if getattr(node,'type',None)=='expander':found[node.label]=path
            children=getattr(node,'children',{})
            if isinstance(children,dict):
                for index,child in children.items():found.update(paths(child,path+(index,)))
            return found
        labels=('Advanced / Manual Event Builder','Current Operating Policy and Live GA Settings','Inspect Live Resources')
        before=paths(self.app.main)
        self.interpret()
        after=paths(self.app.main)
        for label in labels:self.assertEqual(before[label],after[label],label)
        button(self.app,'Cancel').click().run()
        after=paths(self.app.main)
        for label in labels:self.assertEqual(before[label],after[label],label)

    def test_changed_text_requires_interpretation_again(self):
        self.interpret()
        self.app.text_area(key='v3_scenario_text').set_value('different scenario').run()
        self.assertTrue(button(self.app,'Run Look-Ahead').disabled)

if __name__=='__main__':unittest.main()

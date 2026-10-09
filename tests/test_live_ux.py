"""UX clock controls and exact, unthrottled discrete-event Fast Forward equivalence."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from live_hospital_state import LiveHospitalState, LiveCapacity, OperatingPolicy
from live_initialization import WarmStartSettings
from live_skip_control import LiveSkipDriver
from live_policy_control import DEMO_CAPACITY
from test_live_initialization import ownership
from test_live_scenarios import event, payload


def baseline(stress=False):
    profiles=pd.DataFrame(dict(predicted_probability=[.1,.9],treatment_time_min=[15.,30.],length_of_stay_hours=[.5,2.]))
    state=LiveHospitalState(profiles,LiveCapacity(**DEMO_CAPACITY),4.,42,warm_start=WarmStartSettings())
    state.start()
    if stress:
        state.configure_random_events(True,'Low')
        surge=state.propose_event('PATIENT_SURGE',20.,dict(count=100))
        state.apply_event(surge)
        outage=state.propose_event('DOCTOR_SHORTAGE',90.,dict(count=75))
        state.apply_event(outage)
        state.apply_operating_policy(OperatingPolicy(high_risk_priority_weight=2.,waiting_time_aging_weight=.2),event=surge,temporary=True)
    return state


def button(app,label):return next(b for b in app.button if b.label==label)


class FastForwardTests(unittest.TestCase):
    def test_exact_hash_equivalence_one_hour(self): self.compare(60.)
    def test_exact_hash_equivalence_one_day(self): self.compare(1440.)
    def test_exact_hash_equivalence_seven_days(self): self.compare(10080.)
    def test_exact_hash_equivalence_thirty_days(self): self.compare(43200.)

    def compare(self,minutes):
        for stress in (False,True):
            with self.subTest(minutes=minutes,stress=stress):
                original=baseline(stress); a,b=original.clone(),original.clone()
                a.advance_large_skip(minutes)
                report=b.fast_forward(minutes)
                self.assertEqual(a.state_hash(),b.state_hash())
                self.assertEqual(a.metrics(),b.metrics())
                for name in ('_arrival_rng','_profile_rng','_stress_rng','_random_rng'):
                    self.assertEqual(getattr(a,name).bit_generator.state,getattr(b,name).bit_generator.state)
                self.assertEqual(asdict(b.capacity),DEMO_CAPACITY)
                self.assertGreater(report['processed_events'],0)
                ownership(b)
                if stress and minutes>=1440:
                    self.assertTrue(any(h.get('restored') for h in b.policy_history))

    def test_fractional_existing_clock_and_skip_preserve_hash(self):
        a=baseline(True); a.advance_simulation(12.345)
        b=a.clone(); a.advance_large_skip(147.789); b.fast_forward(147.789)
        self.assertEqual(a.state_hash(),b.state_hash())

    def test_fast_forward_validation(self):
        state=baseline()
        for minutes in (-1.,43201.,float('nan'),float('inf')):
            with self.assertRaises(ValueError): state.fast_forward(minutes)
        state.stop()
        with self.assertRaises(ValueError): state.fast_forward(60.)

    def test_debt_survives_interruption_and_resumes_without_replay(self):
        a=baseline(True); b=a.clone(); skip=LiveSkipDriver(b);skip.request(1440.)
        def interrupt(fraction):
            if fraction>=.125: raise RuntimeError('simulated UI rerun')
        with self.assertRaisesRegex(RuntimeError,'UI rerun'):skip.fast_forward(interrupt)
        self.assertEqual(b.sim_time_minutes,180.)
        self.assertEqual(skip.remaining_minutes,1260.)
        skip.fast_forward();a.advance_large_skip(1440.)
        self.assertEqual(a.state_hash(),b.state_hash())

    def test_pause_and_cancel_fast_forward_at_safe_boundary(self):
        state=baseline();skip=LiveSkipDriver(state);skip.request(1440.)
        def pause(fraction):skip.paused=True
        skip.fast_forward(pause)
        self.assertEqual(state.sim_time_minutes,60.)
        self.assertEqual(skip.remaining_minutes,1380.)
        skip.cancel();skip.fast_forward()
        self.assertEqual(state.sim_time_minutes,60.)

    def test_autoplay_start_pause_resume_speed_rerun_scenario_reset(self):
        from streamlit.testing.v1 import AppTest
        wall=[0.]
        with patch('live_dashboard.wall_seconds',new=lambda:wall[0]),patch.dict(os.environ,{'GROQ_API_KEY':'fixture'}),patch('live_scenario_interpreter.Groq') as groq:
            groq.return_value.chat.completions.create.return_value.choices=[MagicMock(message=MagicMock(content=json.dumps(payload(event(count=3)))))]
            app=AppTest.from_file(str(ROOT/'src/dashboard.py'),default_timeout=60).run()
            button(app,'Start Live Simulation').click().run()
            state=app.session_state['v3_live_hospital']
            self.assertTrue(app.session_state['v3_autoplay'])
            self.assertTrue(app.session_state['v3_clock_driver'].enabled)
            wall[0]=6.;app.run();self.assertEqual(state.sim_time_minutes,1.)
            before=state.state_hash();app.run();self.assertEqual(before,state.state_hash())
            app.text_area(key='v3_scenario_text').set_value('3 patients over 20 minutes').run()
            button(app,'Interpret Scenario').click().run()
            self.assertEqual(before,state.state_hash());self.assertTrue(app.session_state['v3_autoplay'])
            button(app,'Pause').click().run();wall[0]=60.;app.run();self.assertEqual(state.sim_time_minutes,1.)
            button(app,'Resume').click().run();wall[0]=66.;app.run();self.assertEqual(state.sim_time_minutes,2.)
            app.selectbox(key='v3_speed').set_value(120).run();wall[0]=67.;app.run();self.assertEqual(state.sim_time_minutes,4.)
            button(app,'Reset Live Twin').click().run();self.assertFalse(app.session_state['v3_autoplay'])
            wall[0]=100.;app.run();self.assertEqual(state.sim_time_minutes,0.)
            button(app,'Start Live Simulation').click().run();self.assertTrue(app.session_state['v3_autoplay'])
            app.selectbox(key='v3_reset_initialization').set_value('Reset Empty').run()
            button(app,'Reset Live Twin').click().run();button(app,'Start Live Simulation').click().run()
            self.assertFalse(app.session_state['v3_autoplay']);self.assertFalse(app.session_state['v3_live_hospital'].initial_state['enabled'])
            self.assertFalse(app.exception)

    @patch('live_dashboard.wall_seconds',new=lambda:0.)
    def test_compact_initial_details_and_fast_forward_single_operation(self):
        from streamlit.testing.v1 import AppTest
        with patch('live_dashboard.render_event_controls') as event_ui:
            app=AppTest.from_file(str(ROOT/'src/dashboard.py'),default_timeout=60).run()
            button(app,'Start Live Simulation').click().run()
            state=app.session_state['v3_live_hospital'];before=state.clone()
            details=next(e for e in app.expander if e.label=='Initial Hospital State Details')
            self.assertFalse(details.proto.expanded)
            self.assertTrue(any('259 initial active patients' in c.value for c in app.caption))
            self.assertEqual(json.loads(details.json[0].value),state.initial_state)
            app.number_input(key='v3_manual_delta').set_value(10080.).run()
            button(app,'Fast Forward').click().run()
            before.advance_large_skip(10080.)
            self.assertEqual(before.state_hash(),state.state_hash())
            self.assertEqual(app.session_state['v3_skip_driver'].remaining_minutes,0.)
            self.assertTrue(app.session_state['v3_autoplay'])
            self.assertFalse(app.exception)

    def test_validation_failure_does_not_enable_playback(self):
        from streamlit.testing.v1 import AppTest
        with patch('live_dashboard.load_patient_profiles',side_effect=ValueError('invalid profiles')):
            app=AppTest.from_file(str(ROOT/'src/dashboard.py'),default_timeout=60).run()
            button(app,'Start Live Simulation').click().run()
            self.assertFalse(app.session_state['v3_autoplay'])
            self.assertTrue(app.error)
            self.assertFalse('v3_live_hospital' in app.session_state)

if __name__=='__main__':unittest.main()

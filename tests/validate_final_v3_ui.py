"""Final primary-UI consolidation demo; provider-shaped fixture, no network calls."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import MagicMock, patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from streamlit.testing.v1 import AppTest
from test_live_scenarios import payload, event
from test_live_initialization import ownership

def button(app,label):
    return next(b for b in app.button if b.label==label)

@patch("live_dashboard.wall_seconds", new=lambda: 0.)
def main():
    started=time.perf_counter()
    with patch.dict(os.environ, {'GROQ_API_KEY':'fixture'}), patch('live_scenario_interpreter.Groq') as groq:
        groq.return_value.chat.completions.create.return_value.choices=[MagicMock(message=MagicMock(content=json.dumps(payload(event(count=60,duration=25.),event('DOCTOR_SHORTAGE',70,120.)))))]
        app=AppTest.from_file(str(ROOT/'src/dashboard.py'),default_timeout=60).run()
        assert not app.exception and not app.tabs
        assert not any(n.key=='icu_beds_number' for n in app.number_input)
        button(app,'Start Live Simulation').click().run()
        state=app.session_state['v3_live_hospital']; capacity=asdict(state.capacity)
        assert state.initial_state['enabled']; ownership(state)
        app.number_input(key='v3_manual_delta').set_value(1440.).run()
        button(app,'Fast Forward').click().run()
        while app.session_state['v3_skip_driver'].remaining_minutes: app.run()
        assert state.sim_time_minutes==1440.; ownership(state)
        app.selectbox(key='v3_lookahead_horizon').set_value(240)
        app.number_input(key='v3_lookahead_reps').set_value(3)
        app.number_input(key='v3_ga_population').set_value(6)
        app.number_input(key='v3_ga_generations').set_value(2)
        app.number_input(key='v3_ga_search_reps').set_value(1)
        app.text_area(key='v3_scenario_text').set_value('60 patients arrive over 25 minutes while 70 doctors are unavailable for two hours.').run()
        before=state.state_hash()
        button(app,'Interpret Scenario').click().run()
        button(app,'Run Look-Ahead').click().run()
        assert before==state.state_hash()
        preview=app.session_state['v3_scenario_preview']
        assert preview['verdict'] in ('AT RISK','CANNOT HANDLE')
        button(app,'Optimize Scenario Operating Policy').click().run()
        assert not app.exception and before==state.state_hash()
        result=app.session_state['v3_live_ga_result']
        assert not app.session_state['v3_live_ai']['provider'].startswith('Groq')
        assert 'v3_live_xai' in app.session_state
        button(app,'Reject Recommendation').click().run()
        assert before==state.state_hash()
        assert button(app,'Apply Optimized Operating Policy').disabled
        button(app,'Modify / Retry').click().run()
        button(app,'Optimize Scenario Operating Policy').click().run()
        previous=state.current_policy; owners={k:{rid:r.patient_id for rid,r in pool.items()} for k,pool in state.resources.items()}
        button(app,'Apply Optimized Operating Policy').click().run()
        assert owners=={k:{rid:r.patient_id for rid,r in pool.items()} for k,pool in state.resources.items()}
        assert asdict(state.capacity)==capacity and state._temporary_policy
        button(app,'Revalidate Unchanged Scenario').click().run()
        button(app,'Run Look-Ahead').click().run()
        button(app,'Apply Scenario Events').click().run()
        app.number_input(key='v3_manual_delta').set_value(720.).run()
        button(app,'Fast Forward').click().run()
        while app.session_state['v3_skip_driver'].remaining_minutes: app.run()
        assert not app.exception and state.current_policy==previous and state._temporary_policy is None
        assert any(e['event_type']=='POLICY_RESTORED' for e in state.events)
        assert asdict(state.capacity)==capacity; ownership(state)
        # Opt-in historical navigation is accessible and preserves the live session.
        app.checkbox(key='show_legacy_v2').set_value(True).run()
        assert not app.exception and [t.label for t in app.tabs]==['Dashboard','Optimize','Review']
        assert app.session_state['v3_live_hospital'] is state
        app.checkbox(key='show_legacy_v2').set_value(False).run()
        assert not app.exception and not app.tabs and app.session_state['v3_live_hospital'] is state
        button(app,'Reset Live Twin').click().run()
        assert state.sim_time_minutes==0 and state.status=='READY'; ownership(state)
        report=dict(primary_ui='PASS', historical_ui_opt_in='PASS', warm_start='PASS',
            advanced_normal_minutes=1440, provider='raw Groq-shaped fixture; no network',
            lookahead_verdict=preview['verdict'], current_robustness=result['current_verified']['robustness'],
            optimized_robustness=result['optimized_verified']['robustness'], ga_outcome=result['outcome'],
            ga_seconds=result['runtime_seconds'], diagnosis=result['diagnosis']['likely_constraint_type'],
            xai_explanation='rendered', deterministic_fallback='rendered', reject_no_mutation=True,
            retry=True, apply_preserves_ownership=True, temporary_restore=True, reset=True,
            fixed_capacity=capacity, runtime_seconds=time.perf_counter()-started)
        (ROOT/'docs/cleanup_phase2/ui_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))

if __name__=='__main__': main()

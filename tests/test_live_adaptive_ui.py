"""Permanent adaptive anchors, guarded GA actions and human decisions."""
from dataclasses import replace,asdict
import json,os,sys,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from streamlit.testing.v1 import AppTest
from test_live_diagnosis import search
from test_live_policy_ga import hospital,waiting
from test_live_ux import button
from live_hospital_state import LiveClockDriver
from live_policy_ga import optimize_live_policy,LiveGAOptions
from live_pressure import LiveTargets
from live_policy_explanation import explain_live_tree,explain_live_policy
SCRIPT='''import streamlit as st
from live_adaptive_dashboard import render_adaptive_optimization
render_adaptive_optimization(st.session_state.v3_live_hospital, None, st.session_state.test_targets, 60., 5)
with st.container(key="following_live_panels"):
    st.caption("Following live panels")
'''
REGIONS=('action','outcome','diagnosis','xai','explanation','decision','technical','controls','history')

def anchors(app):
    found={}
    def visit(node,path=()):
        proto=getattr(node,'proto',None)
        identity=getattr(proto,'id','')
        for name in ('section',)+REGIONS:
            if identity.endswith('-v3_adaptive_'+name):found[name]=path
        if identity.endswith('-following_live_panels'):found['following']=path
        children=getattr(node,'children',{})
        if isinstance(children,dict):
            for index,child in children.items():visit(child,path+(index,))
    visit(app.main)
    assert len(found)==11,found
    return found

class AdaptiveLayoutTests(unittest.TestCase):
    def app(self,state,targets,options,result=None):
        app=AppTest.from_string(SCRIPT,default_timeout=60)
        app.session_state['v3_live_hospital']=state
        app.session_state['v3_clock_driver']=LiveClockDriver(state)
        app.session_state['v3_autoplay']=True
        app.session_state['test_targets']=targets
        for key,value in [('v3_ga_population',options.population),('v3_ga_generations',options.generations),('v3_ga_search_reps',options.search_replications)]:app.session_state[key]=value
        if result:app.session_state['v3_live_ga_result']=result
        app.run();self.assertFalse(app.exception)
        return app

    def test_verified_success_has_fixed_anchors_and_three_diagnosis_columns(self):
        self.check_outcome(True,'VERIFIED SUCCESS')

    def test_partial_improvement_has_fixed_anchors(self):
        self.check_outcome(False,'PARTIAL IMPROVEMENT')

    def check_outcome(self,success,expected):
        state,targets,options,result=search(success)
        self.assertEqual(result['outcome'],expected)
        app=self.app(state,targets,options)
        before=anchors(app)
        app.session_state['v3_live_ga_result']=result;app.run()
        self.assertFalse(app.exception);self.assertEqual(before,anchors(app))
        self.assertTrue(any('Diagnosis:' in e.value for e in app.markdown))
        self.assertTrue(any(e.label=='Decision Tree XAI Details' for e in app.expander))
        self.assertFalse(button(app,'Apply Optimized Operating Policy').disabled)
        for label in ('Reject Recommendation','Modify / Retry'):self.assertTrue(button(app,label))
        # Three settings, diagnosis and human-decision columns are fixed in both outcomes.
        self.assertEqual(len(app.main.get('column')),9)
        state.advance_simulation(1.);app.run()
        self.assertFalse(app.exception);self.assertEqual(before,anchors(app))
        self.assertTrue(any('STALE live policy' in w.value for w in app.warning))

    def test_no_meaningful_improvement_and_single_class_xai(self):
        state=hospital();waiting(state);waiting(state)
        targets=replace(LiveTargets(),mean_wait_target_minutes=1.,max_utilization_target=.1)
        options=LiveGAOptions(population=4,generations=1,search_replications=1)
        result=optimize_live_policy(state,None,60.,5,targets,options)
        self.assertEqual(result['outcome'],'NO MEANINGFUL IMPROVEMENT')
        self.assertFalse(explain_live_tree(result)['available'])
        app=self.app(state,targets,options)
        before=anchors(app);app.session_state['v3_live_ga_result']=result;app.run()
        self.assertFalse(app.exception);self.assertEqual(before,anchors(app))
        self.assertTrue(any('XAI insight unavailable' in e.value for e in app.caption))

    def test_groq_success_and_provider_failure_use_same_explanation_region(self):
        state,targets,options,result=search(True)
        app=self.app(state,targets,options,result);before=anchors(app)
        with patch('live_policy_explanation.Groq') as provider,patch.dict(os.environ,{'GROQ_API_KEY':'fixture'}),patch('live_dashboard.wall_seconds',new=lambda:0.):
            call=provider.return_value.chat.completions.create
            call.return_value.choices=[MagicMock(message=MagicMock(content='{"themes":["diagnosis"]}'))]
            button(app,'Generate Groq Live Explanation').click().run()
            self.assertFalse(app.exception);self.assertEqual(before,anchors(app))
            self.assertTrue(app.session_state['v3_live_ai']['provider'].startswith('Groq'))
            call.side_effect=RuntimeError('offline')
            button(app,'Generate Groq Live Explanation').click().run()
            self.assertFalse(app.exception);self.assertEqual(before,anchors(app))
            self.assertEqual(app.session_state['v3_live_ai']['provider'],'deterministic fallback')

    def test_ga_boundary_preserves_checkpoint_clock_playback_and_outputs(self):
        state,targets,options,expected=search(True)
        app=self.app(state,targets,options)
        driver=app.session_state['v3_clock_driver'];driver.enabled=True
        before=state.state_hash();positions=anchors(app)
        def optimize(checkpoint,*args,**kwargs):
            self.assertIsNot(checkpoint,state);self.assertEqual(checkpoint.state_hash(),before)
            self.assertTrue(app.session_state['v3_synchronous_action']);self.assertFalse(driver.enabled)
            return optimize_live_policy(checkpoint,*args,**kwargs)
        with patch('live_adaptive_dashboard.optimize_live_policy',side_effect=optimize),patch('live_dashboard.wall_seconds',new=lambda:30.),patch('live_adaptive_dashboard.st.spinner',side_effect=AssertionError('transient spinner')):
            button(app,'Optimize Current Operating Policy').click().run()
        self.assertFalse(app.exception);self.assertEqual(before,state.state_hash());self.assertEqual(positions,anchors(app))
        self.assertTrue(driver.enabled);self.assertEqual(driver.last_wall_seconds,30.)
        self.assertFalse(app.session_state['v3_synchronous_action'])
        actual=app.session_state['v3_live_ga_result']
        for key in ('outcome','optimized_policy','search_fitness','verified_fitness','provenance','current_verified','optimized_verified'):self.assertEqual(actual[key],expected[key])

    def test_expected_ga_error_is_visible_inside_same_anchors(self):self.check_error(ValueError('invalid experiment'))
    def test_unexpected_ga_error_is_visible_inside_same_anchors(self):self.check_error(RuntimeError('test failure'))

    def check_error(self,error):
        state,targets,options,_=search()
        app=self.app(state,targets,options);positions=anchors(app);before=state.state_hash()
        driver=app.session_state['v3_clock_driver'];driver.enabled=True
        with patch('live_adaptive_dashboard.optimize_live_policy',side_effect=error),patch('live_dashboard.wall_seconds',new=lambda:0.):
            button(app,'Optimize Current Operating Policy').click().run()
        self.assertFalse(app.exception);self.assertEqual(positions,anchors(app));self.assertEqual(before,state.state_hash())
        self.assertTrue(app.error);self.assertTrue(driver.enabled)
        self.assertTrue(any(e.label=='Adaptive Optimization Error Details' for e in app.expander))
        self.assertTrue(button(app,'Modify / Retry'))
        button(app,'Modify / Retry').click().run()
        self.assertFalse(app.exception);self.assertEqual(positions,anchors(app))
        self.assertNotIn('v3_live_ga_status',app.session_state)

    def test_reject_retry_apply_policy_change_do_not_move_anchors(self):
        state,targets,options,result=search(True)
        app=self.app(state,targets,options,result);positions=anchors(app);before=state.state_hash();capacity=asdict(state.capacity)
        button(app,'Reject Recommendation').click().run()
        self.assertFalse(app.exception);self.assertEqual(positions,anchors(app));self.assertEqual(before,state.state_hash())
        self.assertTrue(button(app,'Apply Optimized Operating Policy').disabled)
        button(app,'Modify / Retry').click().run()
        self.assertFalse(app.exception);self.assertEqual(positions,anchors(app))
        with patch('live_dashboard.wall_seconds',new=lambda:0.):button(app,'Optimize Current Operating Policy').click().run()
        owners={k:{rid:r.patient_id for rid,r in pool.items()} for k,pool in state.resources.items()}
        button(app,'Apply Optimized Operating Policy').click().run()
        self.assertFalse(app.exception);self.assertEqual(positions,anchors(app))
        self.assertEqual(asdict(state.capacity),capacity)
        self.assertEqual(owners,{k:{rid:r.patient_id for rid,r in pool.items()} for k,pool in state.resources.items()})
        state.advance_simulation(1.);app.run();self.assertFalse(app.exception);self.assertEqual(positions,anchors(app))

if __name__=='__main__':unittest.main()

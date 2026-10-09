"""Real warm-start CPU and full-dashboard skip benchmarks; no sleeps/provider calls."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from streamlit.testing.v1 import AppTest
from live_hospital_state import LiveClockDriver
from live_skip_control import LiveSkipDriver

def button(app,label):return next(b for b in app.button if b.label==label)

def main():
    rows=[]
    with patch('live_dashboard.wall_seconds',new=lambda:0.),patch.dict(os.environ,{'GROQ_API_KEY':''}):
        app=AppTest.from_file(str(ROOT/'src/dashboard.py'),default_timeout=120).run()
        button(app,'Start Live Simulation').click().run()
        assert not app.exception
        base=app.session_state['v3_live_hospital'].clone()
        for label,minutes in (('1h',60.),('24h',1440.),('7d',10080.),('30d',43200.)):
            a,b=base.clone(),base.clone()
            started=time.perf_counter();a.advance_large_skip(minutes);ordinary_cpu=time.perf_counter()-started
            started=time.perf_counter();events=b.fast_forward(minutes);fast_cpu=time.perf_counter()-started
            assert a.state_hash()==b.state_hash()
            ordinary_hash=a.state_hash()
            old=base.clone()
            app.session_state['v3_live_hospital']=old
            app.session_state['v3_clock_driver']=LiveClockDriver(old)
            app.session_state['v3_skip_driver']=LiveSkipDriver(old)
            app.run()
            started=time.perf_counter();remaining=minutes
            # Faithfully benchmark the former per-boundary dashboard rerender,
            # without artificially adding the one-second fragment cadence.
            while remaining:
                delta=min(remaining,60.);old.advance_simulation(delta);remaining-=delta;app.run()
                assert not app.exception
            old_ui_seconds=time.perf_counter()-started
            assert old.state_hash()==ordinary_hash
            new=base.clone()
            app.session_state['v3_live_hospital']=new
            app.session_state['v3_clock_driver']=LiveClockDriver(new)
            app.session_state['v3_skip_driver']=LiveSkipDriver(new)
            app.run();app.number_input(key='v3_manual_delta').set_value(minutes).run()
            started=time.perf_counter();button(app,'Fast Forward').click().run();fast_ui_seconds=time.perf_counter()-started
            assert not app.exception and new.state_hash()==ordinary_hash
            row=dict(duration=label,minutes=minutes,ordinary_engine_seconds=ordinary_cpu,
                fast_forward_engine_seconds=fast_cpu,old_dashboard_seconds=old_ui_seconds,
                fast_forward_dashboard_seconds=fast_ui_seconds,processed_events=events['processed_events'],
                ordinary_dashboard_updates=int(minutes/60),fast_forward_completion_updates=1,
                exact_state_hash_equal=True)
            rows.append(row);print(json.dumps(row),flush=True)
    report=dict(baseline=asdict(base.capacity),arrival_rate=base.arrival_rate,warm_start=True,
        benchmark='Same cached real profiles/checkpoint. Engine-only plus actual AppTest presentation; no artificial sleeps. Durations are single-run observations, not universal guarantees.',results=rows)
    (ROOT/'results/live_twin/ux_performance_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()

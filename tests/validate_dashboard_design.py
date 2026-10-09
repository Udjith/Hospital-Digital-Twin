"""Native scenario optimization rendering reproduction and regression."""
import asyncio,json,os,subprocess,sys,tempfile,time,urllib.request
from pathlib import Path
import websockets
ROOT=Path(__file__).resolve().parents[1]
CHROME=Path(r'C:\Program Files\Google\Chrome\Application\chrome.exe')

FIXTURE = r'''
import builtins,json,sys,time
_original_import=builtins.__import__
def _import(name,*args,**kwargs):
    result=_original_import(name,*args,**kwargs)
    if name=='live_policy_ga':
        module=sys.modules.get(name)
        if module and hasattr(module,'optimize_live_policy') and not getattr(module,'_diagnostic_ga',False):
            module._diagnostic_ga=True
            optimize=module.optimize_live_policy
            def run(*a,**kw):
                import streamlit as st
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                live=st.session_state['v3_live_hospital']
                before=dict(hash=live.state_hash(),clock=live.sim_time_minutes,autoplay=st.session_state.get('v3_autoplay'),driver_enabled=st.session_state['v3_clock_driver'].enabled,busy=st.session_state.get('v3_synchronous_action'),fragment_ids=list(get_script_run_ctx().fragment_ids_this_run))
                print('SCENARIO_GA_STARTED '+json.dumps(before),flush=True)
                out=optimize(*a,**kw)
                print('SCENARIO_GA_COMPLETED '+json.dumps(dict(outcome=out['outcome'],verdict=out['optimized_verified']['verdict'],run_id=out['run_id'],hash=live.state_hash(),clock=live.sim_time_minutes)),flush=True)
                return out
            module.optimize_live_policy=run
    if name=='live_scenario_interpreter':
        module=sys.modules.get(name)
        if module and hasattr(module,'interpret_scenario') and not getattr(module,'_diagnostic_fixture',False):
            module._diagnostic_fixture=True
            class Provider:
                def __init__(self,**kwargs):self.chat=self;self.completions=self
                def create(self,**kwargs):
                    from types import SimpleNamespace
                    time.sleep(float(__import__('os').environ.get('SCENARIO_FIXTURE_DELAY','1.2')))
                    data={'events':[{'event_type':'PATIENT_SURGE','duration_minutes':20,'start_delay_minutes':0,'parameters':{'count':60,'arrival_rate':None,'high_risk_proportion':None}}],'assumptions':[],'warnings':[],'clarification':None}
                    data['events'].append({'event_type':'DOCTOR_SHORTAGE','duration_minutes':120,'start_delay_minutes':0,'parameters':{'count':70,'arrival_rate':None,'high_risk_proportion':None}})
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])
            module.Groq=Provider
            original=module.interpret_scenario
            def interpret(text,checkpoint):
                import streamlit as st
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                live=st.session_state['v3_live_hospital']
                def record(stage):
                    driver=st.session_state['v3_clock_driver']
                    print('SCENARIO_BOUNDARY '+json.dumps(dict(stage=stage,clock=live.sim_time_minutes,hash=live.state_hash(),checkpoint_hash=checkpoint.state_hash(),status=live.status,autoplay=st.session_state.get('v3_autoplay'),driver_enabled=driver.enabled,text=st.session_state.get('v3_scenario_text'),busy=st.session_state.get('v3_synchronous_action'),fragment_run=bool(get_script_run_ctx().fragment_ids_this_run))),flush=True)
                record('before');output=original(text,checkpoint);record('after')
                print('SCENARIO_RESULT '+json.dumps(output),flush=True)
                return output
            module.interpret_scenario=interpret
    return result
builtins.__import__=_import
'''

async def inspect(endpoint, paused=False):
    errors=[]
    async with websockets.connect(endpoint,max_size=4_000_000) as ws:
        seq=0
        async def command(method,params=None):
            nonlocal seq
            seq+=1;await ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
            while True:
                msg=json.loads(await ws.recv())
                if msg.get('method') in ('Runtime.exceptionThrown','Log.entryAdded'):
                    errors.append(msg.get('params',{}))
                if msg.get('method')=='Runtime.consoleAPICalled' and msg.get('params',{}).get('type')=='error':
                    errors.append(msg['params'])
                if msg.get('id')==seq:return msg
        await command('Runtime.enable');await command('Log.enable');await command('Page.enable')
        async def evaluate(expression):
            r=await command('Runtime.evaluate',dict(expression=expression,returnByValue=True))
            return r.get('result',{}).get('result',{}).get('value')
        screenshots={}
        async def capture(name,selector=None):
            if selector:
                await evaluate("document.querySelector("+json.dumps(selector)+")?.scrollIntoView({block:'start'})")
                await asyncio.sleep(.3)
            r=await command('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))
            import base64
            folder=Path(os.environ.get('UI_SCREENSHOT_DIR',str(Path(tempfile.gettempdir())/'hospital_v3_design')))
            folder.mkdir(exist_ok=True)
            path=folder/(('before_' if '--before' in sys.argv else 'after_')+name+'.png')
            path.write_bytes(base64.b64decode(r['result']['data']))
            screenshots[name]=str(path)
        async def wait(expr,seconds=30):
            for _ in range(int(seconds*4)):
                if await evaluate(expr):return
                await asyncio.sleep(.25)
            raise AssertionError(json.dumps(dict(wait=expr,dom=await state(),errors=[a.get('description','') for e in errors for a in e.get('args',[]) if a.get('description')])))
        async def click(label):
            for _ in range(60):
                clicked=await evaluate("(()=>{if(document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state')==='running')return false;const b=[...document.querySelectorAll('button')].find(b=>b.innerText.trim()==="+json.dumps(label)+");if(!b||b.disabled)return false;b.click();return true})()")
                if clicked:return True
                await asyncio.sleep(.2)
            return False
        async def state():
            return await evaluate("(()=>{const text=document.body.innerText;const toggle=[...document.querySelectorAll('[data-testid=stToggle], [data-testid=stCheckbox]')].find(e=>e.innerText.includes('Automatic Live Playback'))?.querySelector('input');return {body_length:text.length,clock:text.match(/Elapsed ([0-9.]+) simulated minutes/)?.[1],running:text.includes('| RUNNING'),paused:text.includes('| PAUSED'),autoplay:toggle?.checked,scenario:document.querySelector('textarea[aria-label=\"Hospital Scenario\"]')?.value,preview:text.includes('SCENARIO INTERPRETATION'),stale:text.includes('STALE scenario'),exception:document.querySelector('[data-testid=stException]')?.innerText,text_tail:text.slice(-1000)}})()")
        await wait("[...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Start Live Simulation')")
        await capture('startup')
        assert await click('Start Live Simulation')
        await wait("document.body.innerText.includes('| RUNNING')")
        await asyncio.sleep(2.)
        await capture('warm_running')
        if '--before' in sys.argv:return dict(before=await state(),after=await state(),actions={},minimum_body_length=1000,browser_errors=errors,screenshots=screenshots)
        if paused:
            assert await click('Pause')
            await wait("document.body.innerText.includes('| PAUSED')")
        text='60 patients arrive over 20 minutes while 70 doctors are unavailable for two hours.'
        await evaluate("document.querySelector('textarea[aria-label=\"Hospital Scenario\"]').focus()")
        await command('Input.insertText',dict(text=text))
        await command('Input.dispatchKeyEvent',dict(type='keyDown',key='Tab',code='Tab',windowsVirtualKeyCode=9))
        await command('Input.dispatchKeyEvent',dict(type='keyUp',key='Tab',code='Tab',windowsVirtualKeyCode=9))
        await asyncio.sleep(1.)
        if not paused:
            await evaluate("(()=>{const e=[...document.querySelectorAll('[data-testid=stToggle], [data-testid=stCheckbox]')].find(e=>e.innerText.includes('Automatic Live Playback'))?.querySelector('input');if(e&&!e.checked)e.click()})()")
            await asyncio.sleep(1.)
        before=await state();assert before['scenario']==text
        assert before['paused'] if paused else before['autoplay']
        assert await click('Interpret Scenario')
        samples=[]
        for i in range(60):
            await asyncio.sleep(.5);v=await state();samples.append(v)
            if v['body_length']<100:
                await asyncio.sleep(2.)
                break
            if v.get('preview') or v.get('exception'):
                await asyncio.sleep(3.)
                samples.append(await state())
                break
        after=samples[-1]
        await capture('interpreted','textarea')
        actions={}
        if after.get('preview') and after['body_length']>1000:
            if not paused:
                assert await click('Pause')
                await wait("document.body.innerText.includes('| PAUSED')")
            assert await click('Run Look-Ahead')
            await wait("document.body.innerText.includes('Current scenario verdict:')",60)
            await asyncio.sleep(1.)
            actions['lookahead']=await state()
            await capture('forecast','.st-key-v3_scenario_forecast')
            assert await evaluate("document.body.innerText.includes('Current scenario verdict: CANNOT HANDLE') || document.body.innerText.includes('Current scenario verdict: AT RISK')")
            assert await click('Optimize Scenario Operating Policy')
            for _ in range(180):
                await asyncio.sleep(.5)
                v=await state()
                if v['body_length']<100:
                    await asyncio.sleep(2.)
                    break
                if await evaluate("document.body.innerText.includes('Verified outcome:')"):
                    await asyncio.sleep(3.)
                    break
            actions['optimization']=await state()
            await capture('ga_result','.st-key-v3_adaptive_outcome')
            await capture('explanation','.st-key-v3_adaptive_explanation')
            await evaluate("(()=>{const e=[...document.querySelectorAll('summary')].find(e=>e.innerText.includes('Current vs Optimized Resource Pressure'));if(e&&!e.parentElement.open)e.click()})()")
            await asyncio.sleep(.5)
            await capture('technical','.st-key-v3_adaptive_technical')
            if actions['optimization']['body_length']>1000 and '--verify' in sys.argv:
                await wait("document.body.innerText.includes('Verified outcome:')",90)
                assert await evaluate("document.body.innerText.includes('Diagnosis:') && document.body.innerText.includes('XAI insight') && document.body.innerText.includes('Explanation (deterministic fallback)')")
                assert await evaluate("['Apply Optimized Operating Policy','Reject Recommendation','Modify / Retry'].every(label=>[...document.querySelectorAll('button')].some(b=>b.innerText.trim()===label))")
                assert await click('Reject Recommendation')
                await wait("document.body.innerText.includes('Recommendation rejected.')")
                actions['reject']=await state()
                assert await evaluate("[...document.querySelectorAll('button')].find(b=>b.innerText.trim()==='Apply Optimized Operating Policy').disabled")
                assert await click('Modify / Retry')
                await wait("document.body.innerText.includes('Modify / Retry: adjust')")
                actions['retry']=await state()
                assert await click('Optimize Scenario Operating Policy')
                await wait("document.body.innerText.includes('Verified outcome:') && !document.body.innerText.includes('Recommendation rejected.')",90)
                await asyncio.sleep(3.)
                actions['optimized_again']=await state()
                assert await click('Apply Optimized Operating Policy')
                await wait("document.body.innerText.includes('STALE live policy recommendation')")
                actions['apply']=await state()
                await evaluate("(()=>{const e=[...document.querySelectorAll('summary')].find(e=>e.innerText.includes('Fast Forward Controls'));if(e&&!e.parentElement.open)e.click()})()")
                await wait("[...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Fast Forward')")
                assert await click('Fast Forward')
                await asyncio.sleep(3.)
                actions['advance']=await state()
                for value in actions.values():assert value['body_length']>1000 and not value.get('exception')
        widths={}
        if '--before' not in sys.argv:
            for width,height in ((1920,1080),(1366,768),(1200,800)):
                await command('Emulation.setDeviceMetricsOverride',dict(width=width,height=height,deviceScaleFactor=1,mobile=False))
                await asyncio.sleep(.5)
                widths[str(width)]=await evaluate("({width:innerWidth,scroll:document.documentElement.scrollWidth,visible:document.body.innerText.length})")
                assert widths[str(width)]['scroll']<=width+2,widths
                await capture('responsive_'+str(width),'.st-key-v3_top_status')
            await evaluate("(()=>{const e=[...document.querySelectorAll('summary')].find(e=>e.innerText.includes('Expand Status'));if(e&&!e.parentElement.open)e.click()})()")
            await asyncio.sleep(.5)
            await capture('expanded_status','.st-key-v3_top_status')
            expanded_before=await state()
            assert await evaluate("document.body.innerText.includes('Initial incumbents remaining') && document.body.innerText.includes('Throughput')")
            await evaluate("[...document.querySelectorAll('summary')].find(e=>e.innerText.includes('Expand Status')).click()")
            await wait("![...document.querySelectorAll('summary')].find(e=>e.innerText.includes('Expand Status')).parentElement.open")
            assert (await state())['clock']==expanded_before['clock']
            await evaluate("(()=>{const e=[...document.querySelectorAll('summary')].find(e=>e.innerText.includes('Current vs Optimized Resource Pressure'));if(e&&!e.parentElement.open)e.click()})()")
        return dict(before=before,after=after,actions=actions,minimum_body_length=min(x['body_length'] for x in samples),browser_errors=errors,screenshots=screenshots,responsive=widths)

def main(fixture=False, paused=False):
    env=dict(os.environ,STREAMLIT_SERVER_HEADLESS='true',STREAMLIT_SERVER_PORT='18525',STREAMLIT_BROWSER_GATHER_USAGE_STATS='false',PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='hospital_interpret_repro_') as temp:
        owned=Path(temp);processes=[]
        assert owned.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
        if fixture:
            (owned/'sitecustomize.py').write_text(FIXTURE,encoding='utf-8')
            env['PYTHONPATH']=str(owned)+os.pathsep+env.get('PYTHONPATH','')
            env['GROQ_API_KEY']='fixture-only'
        with (owned/'server.log').open('w') as log:
            try:
                processes.append(subprocess.Popen([sys.executable,'main.py'],cwd=ROOT,env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW))
                for _ in range(100):
                    try:urllib.request.urlopen('http://localhost:18525/_stcore/health',timeout=1).close();break
                    except OSError:time.sleep(.25)
                processes.append(subprocess.Popen([str(CHROME),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=19230','--window-size=1500,1100','--user-data-dir='+str(owned/'chrome'),'http://localhost:18525'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW))
                for _ in range(100):
                    try:
                        pages=json.loads(urllib.request.urlopen('http://localhost:19230/json',timeout=1).read());endpoint=next(p['webSocketDebuggerUrl'] for p in pages if p.get('type')=='page');break
                    except (OSError,StopIteration):time.sleep(.25)
                result=asyncio.run(inspect(endpoint,paused))
            finally:
                for p in reversed(processes):subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True);p.wait(timeout=15)
        logs=(owned/'server.log').read_text(errors='replace')
        # Scrub any key that the local dotenv/process environment could supply.
        from dotenv import dotenv_values
        keys=[os.environ.get('GROQ_API_KEY'),dotenv_values(ROOT/'.env').get('GROQ_API_KEY')]
        encoded=json.dumps(dict(result,server_log=logs[-16000:]),indent=2)
        for key in keys:
            if key:encoded=encoded.replace(key,'[REDACTED]')
        result=json.loads(encoded)
        result['browser_errors']=list(dict.fromkeys(a.get('description','') for e in result['browser_errors'] for a in e.get('args',[]) if a.get('description')))
        result['server_log']='\n'.join(l for l in result['server_log'].splitlines() if l.startswith('SCENARIO_') or 'Traceback' in l)
        for key in ('before','after'):result[key].pop('text_tail',None)
        for value in result['actions'].values():value.pop('text_tail',None)
        records=[json.loads(line.split(' ',1)[1]) for line in logs.splitlines() if line.startswith('SCENARIO_BOUNDARY ')]
        result['interpretation_state_unchanged']=all(records[i]['hash']==records[i+1]['hash'] and records[i]['clock']==records[i+1]['clock'] for i in range(0,len(records),2)) if records else None
        result['request_guarded']=all(r['busy'] and not r['driver_enabled'] for r in records) if records else None
        assert result['interpretation_state_unchanged'] and result['request_guarded'] if fixture and '--before' not in sys.argv else True
        result['ga_completed']=any(l.startswith('SCENARIO_GA_COMPLETED') for l in logs.splitlines())
        result['ga_start_guarded']=all(r['busy'] and not r['driver_enabled'] for r in [json.loads(l.split(' ',1)[1]) for l in logs.splitlines() if l.startswith('SCENARIO_GA_STARTED')])
        print(json.dumps(result,indent=2))
        destination='design_before.json' if '--before' in sys.argv else 'design_browser_validation.json'
        (ROOT/'results/live_twin'/destination).write_text(json.dumps(result,indent=2),encoding='utf-8')
        if '--before' in sys.argv:return
        assert result['actions']['optimization']['body_length']>1000,result
        assert result['after']['preview'],result
        assert result['minimum_body_length']>1000,result
        assert not result['browser_errors'],result
        assert result['after']['paused'] if paused else result['after']['autoplay']
if __name__=='__main__':main(fixture='--fixture' in sys.argv,paused='--paused' in sys.argv)

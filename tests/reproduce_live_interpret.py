"""Native main.py regression with sanitized server/browser diagnostics."""
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
    if name=='live_scenario_interpreter':
        module=sys.modules.get(name)
        if module and hasattr(module,'interpret_scenario') and not getattr(module,'_diagnostic_fixture',False):
            module._diagnostic_fixture=True
            class Provider:
                def __init__(self,**kwargs):self.chat=self;self.completions=self
                def create(self,**kwargs):
                    from types import SimpleNamespace
                    time.sleep(float(__import__('os').environ.get('SCENARIO_FIXTURE_DELAY','1.2')))
                    data={'events':[{'event_type':'PATIENT_SURGE','duration_minutes':20,'start_delay_minutes':0,'parameters':{'count':40,'arrival_rate':None,'high_risk_proportion':None}}],'assumptions':[],'warnings':[],'clarification':None}
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
        await command('Runtime.enable');await command('Log.enable')
        async def evaluate(expression):
            r=await command('Runtime.evaluate',dict(expression=expression,returnByValue=True))
            return r.get('result',{}).get('result',{}).get('value')
        async def wait(expr,seconds=30):
            for _ in range(int(seconds*4)):
                if await evaluate(expr):return
                await asyncio.sleep(.25)
            raise AssertionError(expr)
        async def click(label):
            for _ in range(60):
                clicked=await evaluate("(()=>{if(document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state')==='running')return false;const b=[...document.querySelectorAll('button')].find(b=>b.innerText.trim()==="+json.dumps(label)+");if(!b||b.disabled)return false;b.click();return true})()")
                if clicked:return True
                await asyncio.sleep(.2)
            return False
        async def state():
            # Automatic fragments can replace an st.empty status card between
            # DOM reads without changing the full-app script-state attribute.
            # A persistent blank/error still times out; transient deltas are not
            # evidence that the hospital's actual paused state changed.
            await wait("/\\| (RUNNING|PAUSED)/.test(document.body.innerText) && document.querySelector('textarea[aria-label=\"Hospital Scenario\"]') !== null")
            return await evaluate("(()=>{const text=document.body.innerText;const toggle=[...document.querySelectorAll('[data-testid=stToggle], [data-testid=stCheckbox]')].find(e=>e.innerText.includes('Automatic Live Playback'))?.querySelector('input');return {body_length:text.length,clock:text.match(/Elapsed ([0-9.]+) simulated minutes/)?.[1],running:text.includes('| RUNNING'),paused:text.includes('| PAUSED'),autoplay:toggle?.checked,scenario:document.querySelector('textarea[aria-label=\"Hospital Scenario\"]')?.value,preview:text.includes('SCENARIO INTERPRETATION'),stale:text.includes('STALE scenario'),exception:document.querySelector('[data-testid=stException]')?.innerText,text_tail:text.slice(-1000)}})()")
        await wait("[...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Start Live Simulation')")
        assert await click('Start Live Simulation')
        await wait("document.body.innerText.includes('| RUNNING')")
        await asyncio.sleep(2.)
        if paused:
            assert await click('Pause')
            await wait("document.body.innerText.includes('| PAUSED') && document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state')!=='running'")
        text='A bus accident sends 40 patients over 20 minutes'
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
        actions={}
        if after.get('preview') and after['body_length']>1000:
            assert await evaluate("document.body.innerText.includes('+40 patients over 20m')")
            if not paused:
                assert await click('Pause')
                await wait("document.body.innerText.includes('| PAUSED') && document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state')!=='running'")
            assert await click('Run Look-Ahead')
            await wait("document.body.innerText.includes('Current scenario verdict:')",60)
            await asyncio.sleep(2.)
            actions['lookahead']=await state()
            assert await click('Edit Scenario')
            await wait("!document.body.innerText.includes('SCENARIO INTERPRETATION')")
            await wait("document.body.innerText.includes('| PAUSED') && document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state')!=='running'")
            actions['edit']=await state()
            assert await click('Interpret Scenario')
            await wait("document.body.innerText.includes('SCENARIO INTERPRETATION')")
            assert await click('Cancel')
            await wait("!document.body.innerText.includes('SCENARIO INTERPRETATION')")
            await asyncio.sleep(2.)
            actions['cancel']=await state()
            for value in actions.values():
                assert value['body_length']>1000 and not value.get('exception')
                assert value['paused'] and value['scenario']==text,value
        return dict(before=before,after=after,actions=actions,minimum_body_length=min(x['body_length'] for x in samples),browser_errors=errors)

def main(fixture=False, paused=False):
    env=dict(os.environ,STREAMLIT_SERVER_HEADLESS='true',STREAMLIT_SERVER_PORT='18521',STREAMLIT_BROWSER_GATHER_USAGE_STATS='false',PYTHONDONTWRITEBYTECODE='1')
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
                    try:urllib.request.urlopen('http://localhost:18521/_stcore/health',timeout=1).close();break
                    except OSError:time.sleep(.25)
                processes.append(subprocess.Popen([str(CHROME),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=19226','--window-size=1500,1100','--user-data-dir='+str(owned/'chrome'),'http://localhost:18521'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW))
                for _ in range(100):
                    try:
                        pages=json.loads(urllib.request.urlopen('http://localhost:19226/json',timeout=1).read());endpoint=next(p['webSocketDebuggerUrl'] for p in pages if p.get('type')=='page');break
                    except (OSError,StopIteration):time.sleep(.25)
                result=asyncio.run(inspect(endpoint,paused))
            finally:
                for p in reversed(processes):subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True);p.wait(timeout=15)
        logs=(owned/'server.log').read_text(errors='replace')
        # Scrub any key that the local dotenv/process environment could supply.
        from dotenv import dotenv_values
        keys=[os.environ.get('GROQ_API_KEY'),dotenv_values(ROOT/'.env').get('GROQ_API_KEY')]
        encoded=json.dumps(dict(result,server_log=logs[-7000:]),indent=2)
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
        assert result['interpretation_state_unchanged'] and result['request_guarded'] if fixture else True
        print(json.dumps(result,indent=2))
        destination=('interpret_browser_paused.json' if paused else 'interpret_browser_running.json') if fixture else 'interpret_browser_real.json'
        (ROOT/'results/live_twin'/destination).write_text(json.dumps(result,indent=2),encoding='utf-8')
        assert result['after']['preview'],result
        assert result['minimum_body_length']>1000,result
        assert not result['browser_errors'],result
        assert result['after']['paused'] if paused else result['after']['autoplay']
if __name__=='__main__':main(fixture='--fixture' in sys.argv,paused='--paused' in sys.argv)

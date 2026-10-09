"""Actual native-fragment playback/pause check; no screenshots or repo caches."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

ROOT = Path(__file__).resolve().parents[1]
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


async def inspect(endpoint, event_check=None):
    async with websockets.connect(endpoint, max_size=2_000_000) as ws:
        counter = 0
        async def evaluate(expression):
            nonlocal counter
            counter += 1
            await ws.send(json.dumps({"id": counter, "method": "Runtime.evaluate", "params": {
                "expression": expression, "returnByValue": True}}))
            while True:
                result = json.loads(await ws.recv())
                if result.get("id") == counter:
                    if result.get("error") or result["result"].get("exceptionDetails"):
                        raise AssertionError(result)
                    return result["result"]["result"].get("value")

        async def wait_for(expression):
            for _ in range(120):
                if await evaluate(expression):
                    return
                await asyncio.sleep(.25)
            raise AssertionError("Browser condition failed: " + expression + " | UI: " + str(await evaluate("document.body.innerText.slice(-5500)")))

        async def click(label):
            encoded = json.dumps(label)
            # Native fragments can replace the button between inspection and click.
            for _ in range(40):
                found = await evaluate(f"(() => {{const app=document.querySelector('[data-testid=stApp]');if(app?.getAttribute('data-test-script-state')==='running')return false;const b=[...document.querySelectorAll('button')].find(b=>b.innerText.trim()==={encoded}); if(!b||b.disabled)return false;b.click();return true;}})()")
                if found:
                    return
                await asyncio.sleep(.25)
            raise AssertionError(label + " unavailable: " + str(await evaluate("[...document.querySelectorAll('button')].map(b=>({label:b.innerText,disabled:b.disabled}))")))

        async def clock():
            await wait_for("/Elapsed ([0-9.]+) simulated minutes/.test(document.body.innerText) && document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state') !== 'running'")
            return await evaluate("Number(document.body.innerText.match(/Elapsed ([0-9.]+) simulated minutes/)[1])")

        await wait_for("document.body.innerText.includes('Hospital Digital Twin') && [...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Start Live Simulation')")
        await click("Start Live Simulation")
        await wait_for("document.body.innerText.includes('| RUNNING') && document.querySelector('input[aria-label=\"Live ICU Beds\"]')?.disabled")
        await wait_for("document.body.innerText.includes('Automatic playback on')")
        start_clock = await clock()
        await asyncio.sleep(1.5)
        assert await clock() > start_clock, "Start must enable automatic playback"
        await click("Pause")
        await wait_for("document.body.innerText.includes('| PAUSED')")
        # The top bar updates before the lower elapsed-time caption. Observe a
        # completed render, not two different deltas from the same fragment run.
        await wait_for("document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state') !== 'running'")
        paused_start = await clock()
        await asyncio.sleep(1.)
        assert await clock() == paused_start
        assert await evaluate("![...document.querySelectorAll('summary')].find(e=>e.innerText.includes('Initial Hospital State Details')).parentElement.open")
        await evaluate("[...document.querySelectorAll('summary')].find(e=>e.innerText.includes('Fast Forward Controls')).click()")
        await wait_for("[...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Fast Forward'&&!b.disabled)")
        skip_start = await clock()
        await click("Fast Forward")
        await wait_for(f"Number(document.body.innerText.match(/Elapsed ([0-9.]+) simulated minutes/)[1]) >= {skip_start + 4.99}")
        event_result = await event_check(evaluate, wait_for, click, clock) if event_check else {}
        toggled = await evaluate("""(() => {
            const host=[...document.querySelectorAll('[data-testid="stCheckbox"], [data-testid="stToggle"]')].find(e=>e.innerText.includes('Automatic Live Playback'));
            const input=host?.querySelector('input[type="checkbox"]'); if(!input)return false; if(!input.checked)input.click();return true;
        })()""")
        assert toggled, "Automatic playback toggle not found"
        await wait_for("document.body.innerText.includes('Automatic playback on')")
        if await evaluate("document.body.innerText.includes('| PAUSED')"):
            await click("Resume")
            await wait_for("document.body.innerText.includes('| RUNNING')")
        before = await clock()
        await asyncio.sleep(2.5)
        after = await clock()
        assert after > before, (before, after)
        await click("Pause")
        await wait_for("document.body.innerText.includes('| PAUSED')")
        paused = await clock()
        await asyncio.sleep(2.5)
        assert await clock() == paused
        await click("Resume")
        await wait_for("document.body.innerText.includes('| RUNNING')")
        await asyncio.sleep(1.5)
        assert await clock() > paused
        await click("Stop Live Simulation")
        await wait_for("document.body.innerText.includes('| STOPPED')")
        assert await evaluate("document.querySelector('input[aria-label=\"Live ICU Beds\"]').disabled")
        await click("Reset Live Twin")
        await wait_for("document.body.innerText.includes('| READY') && !document.querySelector('input[aria-label=\"Live ICU Beds\"]').disabled")
        assert await clock() == 0
        assert not await evaluate("document.querySelector('[data-testid=stException]') !== null")
        await click("Start Live Simulation")
        await wait_for("document.body.innerText.includes('| RUNNING') && document.body.innerText.includes('Automatic playback on')")
        restarted = await clock()
        await asyncio.sleep(1.5)
        assert await clock() > restarted
        await click("Reset Live Twin")
        await wait_for("document.body.innerText.includes('| READY') && document.body.innerText.includes('Automatic playback off')")
        assert await clock() == 0
        return dict(auto_start_playback=True, restart_playback=True, initial_details_collapsed=True,
            fast_forward=True, browser="PASS", playback_speed=10, playback_before_minutes=before,
            playback_after_minutes=after, pause_minutes=paused, pause_freezes_clock=True,
            reset_clock_minutes=0, capacity_lock=True, native_fragment_updates=True,
            screenshot_files_created=0, **event_result)


def main(inspector=inspect, artifact_name="browser_validation.json"):
    started = time.perf_counter()
    env = dict(os.environ, STREAMLIT_SERVER_HEADLESS="true", STREAMLIT_SERVER_PORT="18519",
        STREAMLIT_BROWSER_GATHER_USAGE_STATS="false", PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="hospital_live_browser_") as profile:
        # Verify this owned cleanup target is in the system temporary directory.
        assert Path(profile).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
        processes = []
        try:
            processes.append(subprocess.Popen([sys.executable, "main.py"], cwd=ROOT, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW))
            for _ in range(80):
                try:
                    urllib.request.urlopen("http://localhost:18519/_stcore/health", timeout=1).close()
                    break
                except OSError:
                    time.sleep(.25)
            else:
                raise AssertionError("Live server did not start")
            processes.append(subprocess.Popen([str(CHROME), "--headless=new", "--disable-gpu", "--no-first-run",
                "--no-default-browser-check", "--disable-extensions", "--remote-debugging-port=19224",
                "--window-size=1500,1100", "--user-data-dir=" + profile, "http://localhost:18519"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW))
            endpoint = None
            for _ in range(80):
                try:
                    pages = json.loads(urllib.request.urlopen("http://localhost:19224/json", timeout=1).read())
                    endpoint = next(p["webSocketDebuggerUrl"] for p in pages if p.get("type") == "page")
                    break
                except (OSError, StopIteration):
                    time.sleep(.25)
            assert endpoint
            result = asyncio.run(inspector(endpoint))
            result["runtime_seconds"] = time.perf_counter() - started
            directory = ROOT / "results/live_twin"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / artifact_name).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2))
        finally:
            for process in reversed(processes):
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                process.wait(timeout=15)


if __name__ == "__main__":
    main()

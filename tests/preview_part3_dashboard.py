"""Headless Chrome check of actual dark UI/tooltips; owns all launched processes."""
import asyncio
import base64
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
OUTPUT = ROOT / "results/part3_validation"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


async def inspect_browser(endpoint):
    async with websockets.connect(endpoint, max_size=15_000_000) as websocket:
        counter = 0
        async def call(method, params=None):
            nonlocal counter
            counter += 1
            identifier = counter
            await websocket.send(json.dumps(dict(id=identifier, method=method, params=params or {})))
            while True:
                response = json.loads(await websocket.recv())
                if response.get("id") == identifier:
                    if "error" in response:
                        raise AssertionError(response["error"])
                    return response.get("result", {})

        async def evaluate(expression):
            result = await call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
            if "exceptionDetails" in result:
                raise AssertionError(result["exceptionDetails"])
            return result["result"].get("value")

        await call("Emulation.setDeviceMetricsOverride", dict(width=1500, height=1100, deviceScaleFactor=1, mobile=False))
        for _ in range(80):
            if await evaluate("document.body.innerText.includes('GA Population') && document.body.innerText.includes('Current Hospital Policy')"):
                break
            await asyncio.sleep(.25)
        else:
            raise AssertionError("Dashboard did not render")
        for _ in range(80):
            if await evaluate("document.querySelectorAll('[data-testid=stSkeleton]').length === 0"):
                break
            await asyncio.sleep(.25)
        meta = await evaluate("""(() => {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            const icons = [...sidebar.querySelectorAll('[data-testid="stTooltipIcon"]')];
            const first = icons.length ? icons[0].getBoundingClientRect() : {x:0,y:0,width:0,height:0};
            return {tooltip_icons: icons.length, background: getComputedStyle(sidebar).backgroundColor,
                sidebar_ids: [...new Set([...sidebar.querySelectorAll('[data-testid]')].map(n=>n.dataset.testid))],
                sidebar_text: sidebar.innerText.slice(0,200),
                horizontal_overflow: document.documentElement.scrollWidth > innerWidth,
                manual_bounds_visible: /Minimum (ICU|General|Concurrent)|Maximum (ICU|General|Concurrent)/.test(sidebar.innerText),
                hover: {x:first.x+first.width/2,y:first.y+first.height/2},
                exception: document.querySelector('[data-testid="stException"]') !== null};
        })()""")
        print(json.dumps(meta), flush=True)
        assert meta["tooltip_icons"] >= 15 and not meta["manual_bounds_visible"]
        assert not meta["horizontal_overflow"] and not meta["exception"]
        await call("Input.dispatchMouseEvent", {"type": "mouseMoved", **meta["hover"]})
        await asyncio.sleep(.6)
        meta["hover_help_visible"] = await evaluate("document.body.innerText.includes('Number of simulated hospital hours.')")
        assert meta["hover_help_visible"]
        screenshot = await call("Page.captureScreenshot", {"format": "png"})
        (OUTPUT / "dashboard.png").write_bytes(base64.b64decode(screenshot["data"]))
        await call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": 1100, "y": 1000})
        await evaluate("document.querySelector('input[aria-label=\"GA Population\"]').scrollIntoView({block:'center'})")
        await asyncio.sleep(.3)
        screenshot = await call("Page.captureScreenshot", {"format": "png"})
        (OUTPUT / "optimization_sidebar.png").write_bytes(base64.b64decode(screenshot["data"]))
        (OUTPUT / "browser.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, STREAMLIT_SERVER_HEADLESS="true", STREAMLIT_SERVER_PORT="18517",
        STREAMLIT_BROWSER_GATHER_USAGE_STATS="false", PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="chrome_part3_", dir=OUTPUT) as profile:
        assert Path(profile).resolve().is_relative_to(ROOT)
        processes = []
        with (OUTPUT / "browser_launcher.log").open("w") as log:
            try:
                processes.append(subprocess.Popen([sys.executable, "main.py"], cwd=ROOT, env=env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW))
                processes.append(subprocess.Popen([str(CHROME), "--headless=new", "--disable-gpu", "--no-first-run",
                    "--no-default-browser-check", "--disable-extensions", "--remote-debugging-port=19222",
                    "--user-data-dir=" + profile, "http://localhost:18517"], stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW))
                endpoint = None
                for _ in range(80):
                    try:
                        pages = json.loads(urllib.request.urlopen("http://localhost:19222/json", timeout=1).read())
                        endpoint = next(page["webSocketDebuggerUrl"] for page in pages if page.get("type") == "page")
                        break
                    except (OSError, StopIteration):
                        time.sleep(.25)
                assert endpoint
                asyncio.run(inspect_browser(endpoint))
            finally:
                for process in reversed(processes):
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                    process.wait(timeout=15)


if __name__ == "__main__":
    main()

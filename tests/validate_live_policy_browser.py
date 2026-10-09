"""Actual main.py launch, native adaptive UI and high-speed pause check."""
import validate_live_browser as browser
from validate_live_event_browser import check_event


async def check_policy(evaluate, wait_for, click, clock):
    result = await check_event(evaluate, wait_for, click, clock)
    await click("Advance Simulation")
    await wait_for("document.body.innerText.includes('Elapsed 10.00 simulated minutes')")
    before = await clock()
    await click("Optimize Current Operating Policy")
    await wait_for("document.body.innerText.includes('OPTIMIZED POLICY') && [...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Apply Optimized Operating Policy'&&!b.disabled)")
    assert await clock() == before
    await click("Apply Optimized Operating Policy")
    await wait_for("document.body.innerText.includes('STALE live policy recommendation')")
    assert not await evaluate("document.querySelector('[data-testid=stException]') !== null")
    # Pick 3600x in the native select, enable playback, then pause without blocking.
    host = "[...document.querySelectorAll('[data-testid=stSelectbox]')].find(e=>e.innerText.includes('Simulation Speed'))"
    assert await evaluate(f"(() => {{const e={host};const input=e?.querySelector('input'); if(!input)return false;input.focus();input.dispatchEvent(new KeyboardEvent('keydown', {{key:'ArrowDown',code:'ArrowDown',keyCode:40,bubbles:true}}));return true;}})()")
    await wait_for("[...document.querySelectorAll('[role=option]')].some(e=>e.innerText.trim()==='3600x')")
    assert await evaluate("(() => {const e=[...document.querySelectorAll('[role=option]')].find(e=>e.innerText.trim()==='3600x');e.click();return true;})()")
    await wait_for(f"(() => {{const e={host};return e?.innerText.includes('3600x') || e?.querySelector('input')?.value==='3600x';}})()")
    toggle = """(() => {const host=[...document.querySelectorAll('[data-testid=stCheckbox], [data-testid=stToggle]')].find(e=>e.innerText.includes('Automatic Live Playback'));host.querySelector('input[type=checkbox]').click();return true;})()"""
    await evaluate(toggle)
    await wait_for("document.body.innerText.includes('Automatic playback on')")
    await wait_for(f"Number(document.body.innerText.match(/Elapsed ([0-9.]+) simulated minutes/)[1]) > {before + 30}")
    await click("Pause")
    await wait_for("document.body.innerText.includes('| PAUSED')")
    frozen = await clock()
    import asyncio
    await asyncio.sleep(2.)
    assert await clock() == frozen
    await evaluate(toggle)
    await wait_for("document.body.innerText.includes('Automatic playback off')")
    # Return to 10x before the harness's normal playback/resume check.
    await evaluate(f"(() => {{const input=({host}).querySelector('input');input.focus();input.dispatchEvent(new KeyboardEvent('keydown', {{key:'ArrowDown',code:'ArrowDown',keyCode:40,bubbles:true}}));}})()")
    await wait_for("[...document.querySelectorAll('[role=option]')].some(e=>e.innerText.trim()==='10x')")
    await evaluate("[...document.querySelectorAll('[role=option]')].find(e=>e.innerText.trim()==='10x').click()")
    await click("Resume")
    await wait_for("document.body.innerText.includes('| RUNNING')")
    return dict(result, adaptive_ga_read_only=True, explicit_policy_application=True,
                stale_after_policy_change=True, high_speed_pause=3600)


async def inspect(endpoint):
    return await browser.inspect(endpoint, event_check=check_policy)


if __name__ == "__main__":
    browser.main(inspector=inspect, artifact_name="part3_browser_validation.json")

"""Reuse the existing owned-process browser harness for event UI checks."""
import validate_live_browser as browser


async def check_event(evaluate, wait_for, click, clock):
    await evaluate("(() => {const e=[...document.querySelectorAll('summary')].find(e=>e.innerText.includes('Advanced / Manual Event Builder')); if(e&&!e.parentElement.open)e.click();})()")
    count = await evaluate("Number(document.querySelector('input[aria-label=\"Event Patient / Unavailable Resource Count\"]').value)")
    assert count == 12, count
    before = await clock()
    await click("Prepare Proposed Event")
    await wait_for("document.body.innerText.includes('Proposed Event EVT-000001')")
    await click("Run Event Look-Ahead")
    await wait_for("document.body.innerText.includes('Can the current policy handle it?') && [...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Apply Event'&&!b.disabled)")
    assert await clock() == before
    assert not await evaluate("document.querySelector('[data-testid=stException]') !== null")
    await click("Apply Event")
    await wait_for("!document.body.innerText.includes('Proposed Event EVT-000001') && !document.body.innerText.includes('Can the current policy handle it?')")
    assert not await evaluate("document.body.innerText.includes('No active or scheduled stress events.')")
    assert await clock() == before
    return dict(event_default_count=12, event_preview_read_only=True, explicit_event_apply=True)


async def inspect(endpoint):
    return await browser.inspect(endpoint, event_check=check_event)


if __name__ == "__main__":
    browser.main(inspector=inspect, artifact_name="part2_browser_validation.json")

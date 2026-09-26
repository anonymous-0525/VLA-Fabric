import { chromium } from 'playwright';
import assert from 'node:assert/strict';

const base = process.env.SITE_URL || 'http://127.0.0.1:4173';
const browser = await chromium.launch({ args: ['--no-sandbox', '--disable-dev-shm-usage'] });
const reports = [];
async function setup(options = {}) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  await page.route('**/chunks/hero-*.js', route => route.abort());
  if (options.fakeMP4) await page.addInitScript(() => {
    const original = HTMLMediaElement.prototype.canPlayType;
    HTMLMediaElement.prototype.canPlayType = function(type) {
      return type.includes('mp4') ? 'probably' : original.call(this, type);
    };
  });
  if (options.slow) await page.route('**/media/frame4-*.webm', async route => {
    await new Promise(resolve => setTimeout(resolve, route.request().url().includes('wrist') ? 2200 : 1600));
    await route.continue().catch(() => {});
  });
  if (options.fakeMP4) await page.route('**/media/*.mp4', route => route.fulfill({ status: 404, body: '' }));
  if (options.failWrist) await page.route('**/media/frame4-wrist1.*', route => {
    if (route.request().url().endsWith('.jpg')) return route.continue();
    return route.fulfill({status:404,body:''});
  });
  await page.goto(base+'/#physical', { waitUntil: 'domcontentloaded' });
  await page.locator('[data-task=frame4]').scrollIntoViewIfNeeded();
  const item = page.locator('[data-task=frame4]');
  return {page,item};
}
async function running(page, selector, time = .5) {
  await page.waitForFunction(({selector,time}) => {
    const video = document.querySelector(selector);
    return video && !video.paused && !video.seeking && video.readyState >= 3 &&
      video.currentTime > time && video.getVideoPlaybackQuality().totalVideoFrames > 2;
  }, {selector,time}, {timeout:20000});
}
try {
  {
    const {page,item} = await setup({slow:true});
    await item.locator('.video-start').click();
    assert.equal(await item.getAttribute('data-playback'),'loading');
    assert.equal(await item.locator('.video-loading').isVisible(),true);
    await item.locator('.video-play').click();
    assert.equal(await item.getAttribute('data-playback'),'paused');
    await page.waitForTimeout(200);
    assert.equal(await item.locator('.media-error').isVisible(),false);
    await item.locator('.video-play').click();
    await running(page,'[data-task=frame4] .global-video');
    await running(page,'[data-task=frame4] .wrist-slide video');
    await item.locator('.video-play').click();
    await item.locator('.global-video').evaluate(video => { video.currentTime=12; });
    await page.waitForTimeout(800);
    const paused = await item.evaluate(root => ({
      global:root.querySelector('.global-video').currentTime,
      wrist:root.querySelector('.wrist-slide video').currentTime,
      paused: [...root.querySelectorAll('video')].every(video => video.paused),
      controls:root.querySelector('.global-video').controls,
    }));
    assert(paused.paused);
    assert(paused.controls);
    assert(Math.abs(paused.global-paused.wrist)<.3,JSON.stringify(paused));
    reports.push({test:'delayed load, cancellation, resume, paused seek',...paused});
    await page.close();
  }
  {
    const {page,item} = await setup({fakeMP4:true});
    await item.locator('.video-start').click();
    await running(page,'[data-task=frame4] .global-video');
    await running(page,'[data-task=frame4] .wrist-slide video');
    const sources = await item.evaluate(root => [...root.querySelectorAll('video')].map(v=>v.currentSrc));
    assert(sources.every(src=>src.endsWith('.webm')));
    assert.equal(await item.locator('.media-error').isVisible(),false);
    reports.push({test:'unsupported/unavailable preferred format falls back',sources});
    await page.close();
  }
  {
    const {page,item} = await setup({failWrist:true});
    await item.locator('.video-start').click();
    await running(page,'[data-task=frame4] .global-video');
    await item.locator('.media-error').waitFor({state:'visible'});
    const before = await item.locator('.global-video').evaluate(v=>v.currentTime);
    await page.waitForTimeout(900);
    const after = await item.locator('.global-video').evaluate(v=>v.currentTime);
    assert(after>before+.5);
    await item.locator('.wrist-next').click();
    await running(page,'[data-task=frame4] .wrist-slide video');
    reports.push({test:'wrist failure never blocks global; camera switch recovers',before,after});
    await page.close();
  }
} finally { await browser.close(); }
console.log(JSON.stringify(reports,null,2));

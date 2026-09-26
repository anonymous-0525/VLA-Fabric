import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';

const base = process.env.SITE_URL || 'http://127.0.0.1:4173';
await mkdir('site-checks', { recursive: true });
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const report = [];
try {
  for (const [name, width, height] of [['desktop',1440,1000], ['mobile',390,844]]) {
    const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
    const errors = [], requests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => requests.push(request.url()));
    await page.goto(base, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('.physical-item');
    // Page controls initialize without waiting for 3D, and no gallery stream preloads.
    assert.equal(await page.locator('.physical-item').count(),4);
    assert.equal(requests.filter(url => /(?:global|wrist\d|sim-\w+)\.(mp4|webm)/.test(url)).length,0);
    await page.waitForSelector('#robot-stage[data-ready=true]', { timeout:60000 });
    await page.waitForTimeout(1000);
    await page.screenshot({ path:'site-checks/'+name+'-hero.png' });
    const before = await page.locator('#robot-stage').getAttribute('data-frame-height');
    await page.waitForTimeout(1300);
    const after = await page.locator('#robot-stage').getAttribute('data-frame-height');
    assert.notEqual(before,after);
    await page.locator('[data-agent="0"]').click();
    assert.equal(await page.locator('#robot-stage').getAttribute('data-selected'),'0');
    await page.screenshot({ path:'site-checks/'+name+'-highlight.png' });
    await page.locator('[data-agent="0"]').click();
    // Freeze the backdrop so subsequent canvas-pixel checks isolate the model.
    await page.evaluate(() => { document.querySelectorAll('.ambient').forEach(v => v.pause()); document.querySelector('.hero-background').style.visibility='hidden'; });
    await page.locator('#robot-stage canvas').screenshot({ path:'site-checks/'+name+'-canvas-first.png' });
    await page.waitForTimeout(2000);
    await page.locator('#robot-stage canvas').screenshot({ path:'site-checks/'+name+'-canvas-later.png' });
    await page.evaluate(() => { document.querySelector('.hero-background').style.visibility=''; document.documentElement.style.scrollBehavior='auto'; });

    const figureSources = [];
    for (const id of ['discovery','transfer','scale']) {
      await page.locator('[data-chapter="'+id+'"]').click();
      await page.waitForFunction(id => location.hash === '#'+id, id);
      const button = page.locator('#'+id+' .figure-zoom').first();
      await button.scrollIntoViewIfNeeded();
      await button.locator('img').evaluate(img => img.decode());
      figureSources.push(await button.locator('img').getAttribute('src'));
      await button.click();
      assert.equal(await page.locator('#figure-dialog').evaluate(dialog => dialog.open),true);
      await page.keyboard.press('Escape');
      await page.screenshot({ path:'site-checks/'+name+'-'+id+'.png' });
    }
    assert.equal(new Set(figureSources).size,3);
    await page.locator('#simulation').scrollIntoViewIfNeeded();
    await page.screenshot({ path:'site-checks/'+name+'-simulation.png' });
    const simRows = await page.locator('.sim-item').evaluateAll(items => items.map(item => Math.round(item.getBoundingClientRect().top)));
    assert.equal(new Set(simRows).size,name === 'desktop' ? 1 : 2);
    for (const item of await page.locator('.sim-item').all()) {
      await item.scrollIntoViewIfNeeded();
      await item.locator('.video-start').click();
      await item.locator('video').evaluate(async video => {
        await new Promise((resolve,reject) => { const timer=setTimeout(() => reject(new Error('Simulation playback timed out')),10000); const interval=setInterval(() => { if(video.currentTime>.3){clearInterval(interval);clearTimeout(timer);resolve();} },100); });
        video.pause();
      });
    }
    await page.locator('#physical-grid').scrollIntoViewIfNeeded();
    const physicalRows = await page.locator('.physical-item').evaluateAll(items => items.map(item => Math.round(item.getBoundingClientRect().top)));
    assert.equal(new Set(physicalRows).size,name === 'desktop' ? 2 : 4);
    const playback = [];
    for (const item of await page.locator('.physical-item').all()) {
      await item.scrollIntoViewIfNeeded();
      await page.waitForTimeout(200);
      await item.locator('.video-start').click();
      await page.waitForTimeout(1700);
      await item.locator('.video-speed').selectOption('1.5');
      await item.locator('.global-video').evaluate(video => { video.currentTime=5; });
      await item.locator('.wrist-next').click();
      await page.waitForTimeout(2200);
      const timing = await item.evaluate(root => ({
        id: root.dataset.task,
        global: root.querySelector('.global-video').currentTime,
        wrist: root.querySelector('.wrist-slide video').currentTime,
        cameras: root.querySelectorAll('.wrist-slide video').length,
        counter: root.querySelector('.wrist-counter').textContent,
      }));
      assert(timing.global>5);
      assert(Math.abs(timing.global-timing.wrist)<.35,JSON.stringify(timing));
      assert.equal(timing.cameras,1);
      assert(timing.counter.startsWith('02'));
      playback.push(timing);
      await item.locator('.video-play').click();
    }
    await page.locator('#physical-grid').scrollIntoViewIfNeeded();
    await page.screenshot({ path:'site-checks/'+name+'-physical.png' });
    // Load lazy paper artwork before full-page screenshots.
    await page.locator('.figure-zoom img').evaluateAll(async images => {
      await Promise.all(images.map(img => { img.loading='eager'; return img.decode(); }));
    });
    assert.equal(await page.locator('.figure-zoom img').evaluateAll(images => images.some(img => !img.naturalWidth)),false);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth>innerWidth);
    assert.equal(overflow,false);
    assert.deepEqual(errors,[]);
    await page.screenshot({ path:'site-checks/'+name+'-full.png', fullPage:true });
    report.push({name,errors,overflow,before,after,figureSources,simRows,physicalRows,playback});
    await page.close();
  }
  const narrow = await browser.newPage({ viewport:{width:320,height:740}, reducedMotion:'reduce' });
  await narrow.goto(base);
  assert.equal(await narrow.evaluate(() => document.documentElement.scrollWidth>innerWidth),false);
  await narrow.close();
} finally {
  await browser.close();
  await writeFile('site-checks/browser-report.json',JSON.stringify(report,null,2));
}
console.log(JSON.stringify(report,null,2));

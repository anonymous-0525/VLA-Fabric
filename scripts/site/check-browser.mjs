import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';

const base = process.env.SITE_URL || 'http://127.0.0.1:4173';
await mkdir('site-checks', { recursive: true });
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const report = [];
for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#robot-stage[data-ready=true]', { timeout: 45000 });
  await page.waitForTimeout(1800);
  await page.screenshot({ path: `site-checks/${name}-hero.png` });
  const initialFrame = await page.locator('#robot-stage').getAttribute('data-frame');
  await page.waitForTimeout(900);
  const laterFrame = await page.locator('#robot-stage').getAttribute('data-frame');
  await page.locator('[data-agent="0"]').click();
  const selected = await page.locator('#robot-stage').getAttribute('data-selected');
  await page.screenshot({ path: `site-checks/${name}-highlight.png` });
  await page.locator('[data-agent="0"]').click();
  const rect = await page.locator('#robot-stage canvas').boundingBox();
  let hover = '-1';
  for (const [x, y] of [[.32,.40],[.3,.5],[.36,.6],[.65,.4],[.2,.5],[.7,.65]]) {
    await page.mouse.move(rect.x+rect.width*x, rect.y+rect.height*y);
    hover = await page.locator('#robot-stage').getAttribute('data-selected');
    if (hover !== '-1') break;
  }
  await page.mouse.move(0, 0);
  // Keep the video backdrop fixed so pixel differences come from the 3D scene.
  await page.evaluate(() => { document.querySelectorAll('.ambient').forEach(v => v.pause()); document.querySelector('.hero-background').style.visibility='hidden'; });
  const frameHeight = await page.locator('#robot-stage').getAttribute('data-frame-height');
  await page.locator('#robot-stage canvas').screenshot({ path: `site-checks/${name}-canvas-first.png` });
  await page.waitForTimeout(3500);
  await page.locator('#robot-stage canvas').screenshot({ path: `site-checks/${name}-canvas-later.png` });
  const frameHeightLater = await page.locator('#robot-stage').getAttribute('data-frame-height');
  await page.evaluate(() => { document.querySelector('.hero-background').style.visibility=''; });
  await page.locator('#physical').scrollIntoViewIfNeeded();
  await page.waitForFunction(() => [...document.querySelectorAll('#global-video, .wrist-slide video')].every(v => v.readyState >= 2), { timeout: 30000 });
  await page.locator('#video-play').click();
  await page.waitForTimeout(1800);
  await page.locator('#wrist-next').click();
  await page.waitForTimeout(1100);
  const playback = await page.evaluate(() => ({
    global: document.querySelector('#global-video').currentTime,
    wrists: [...document.querySelectorAll('.wrist-slide video')].map(video => video.currentTime),
    current: document.querySelector('#wrist-counter').textContent,
  }));
  await page.locator('#video-play').click();
  await page.screenshot({ path: `site-checks/${name}-physical.png` });
  await page.locator('[data-task="relay3"]').click();
  await page.waitForTimeout(500);
  const task = await page.locator('#physical-task-name').textContent();
  await page.locator('[data-method="transfer"]').click();
  await page.locator('.figure-zoom').click();
  const dialogOpen = await page.locator('#figure-dialog').evaluate(dialog => dialog.open);
  await page.keyboard.press('Escape');
  await page.screenshot({ path: `site-checks/${name}-full.png`, fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  report.push({ name, errors, overflow, initialFrame, laterFrame, frameHeight, frameHeightLater, hover, selected, playback, task, dialogOpen });
  await page.close();
}
await browser.close();
await writeFile('site-checks/browser-report.json', JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
if (report.some(item => item.errors.length || item.overflow || item.initialFrame === item.laterFrame || item.hover === '-1' || item.selected !== '0' || !item.dialogOpen || item.playback.global < .5 || item.playback.wrists.some(t => Math.abs(t-item.playback.global)>.3))) process.exitCode = 1;

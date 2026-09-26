import { createIcons, ArrowUpRight, ArrowDown, ArrowUp, Pause, Play, RotateCcw, ChevronLeft, ChevronRight, Maximize, Expand, X } from 'lucide';
import { startScene } from './hero.js';

const icons = { ArrowUpRight, ArrowDown, ArrowUp, Pause, Play, RotateCcw, ChevronLeft, ChevronRight, Maximize, Expand, X };
const refreshIcons = () => createIcons({ icons });
function setIcon(button, name) { button.innerHTML = `<i data-lucide="${name}"></i>`; refreshIcons(); }
refreshIcons();
startScene(setIcon).catch(() => { document.querySelector('.scene-status').textContent = 'Interactive scene unavailable'; });

const tasks = {
  frame4: { arms: 4, title: 'Four-arm frame installation', description: 'Four arms lift, align, and lower a shared frame over a peg.', full: 60, independent: 36 },
  relay4: { arms: 4, title: 'Parallel object relays', description: 'Two arm pairs hand over vegetables and place them in the basket.', full: 72, independent: 40 },
  relay3: { arms: 3, title: 'Three-arm object relay', description: 'One arm positions the target while two peers coordinate the handover and placement.', full: 56, independent: 16 },
  basket3: { arms: 3, title: 'Shared-basket placement', description: 'Three complete local policies place vegetables into a shared basket.', full: 88, independent: 72 },
};
const globalVideo = document.querySelector('#global-video');
const track = document.querySelector('.wrist-track');
const playButton = document.querySelector('#video-play');
const seek = document.querySelector('#video-seek');
let taskId = 'frame4', wristIndex = 0, wristVideos = [], changing = false, carouselTimer;
const videoExtension = document.createElement('video').canPlayType('video/mp4; codecs="avc1.42E01E"') ? 'mp4' : 'webm';
const media = (task, view, extension = videoExtension) => `static/media/${task}-${view}.${extension}`;
const seconds = time => `${Math.floor((time || 0)/60)}:${String(Math.floor((time || 0)%60)).padStart(2,'0')}`;
const safePlay = video => video.play().catch(() => {});
function sync(force = false) {
  wristVideos.forEach(video => {
    const offset = globalVideo.currentTime-video.currentTime;
    if (video.readyState >= 1 && !video.seeking && (force || Math.abs(offset) > .7)) video.currentTime = globalVideo.currentTime;
    // Small clock corrections avoid repeated seeks and decoder stalls.
    const correction = !globalVideo.paused && !force && Math.abs(offset) > .035 ? Math.max(.9, Math.min(1.1, 1+offset*.65)) : 1;
    const rate = globalVideo.playbackRate * correction;
    if (Math.abs(video.playbackRate-rate) > .005) video.playbackRate = rate;
    if (globalVideo.paused) video.pause(); else if (video.paused) safePlay(video);
  });
}
let syncFrame;
function scheduleSync() {
  cancelAnimationFrame(syncFrame);
  if (globalVideo.paused) return;
  syncFrame = requestAnimationFrame(() => { sync(); scheduleSync(); });
}
globalVideo.addEventListener('play', scheduleSync);
globalVideo.addEventListener('pause', () => cancelAnimationFrame(syncFrame));
function renderWrists() {
  wristVideos.forEach(video => { video.pause(); video.removeAttribute('src'); video.load(); });
  track.replaceChildren(); wristVideos = [];
  for (let slot = -1; slot <= 1; slot++) {
    const camera = (wristIndex + slot + tasks[taskId].arms) % tasks[taskId].arms + 1;
    const slide = document.createElement('div'); slide.className = `wrist-slide${slot === 0 ? ' current' : ''}`;
    const video = document.createElement('video');
    video.src = media(taskId, `wrist${camera}`); video.poster = media(taskId, `wrist${camera}`, 'jpg');
    video.muted = true; video.playsInline = true; video.preload = 'metadata';
    video.setAttribute('aria-label', `${taskId} wrist camera ${camera}`);
    video.addEventListener('loadedmetadata', () => sync(true), { once: true });
    const tag = document.createElement('span'); tag.className = 'view-tag'; tag.textContent = `WRIST 0${camera}`;
    slide.append(video, tag); track.append(slide); wristVideos.push(video);
    if (slot) slide.addEventListener('click', () => changeWrist(slot));
  }
  document.querySelector('#wrist-counter').textContent = `0${wristIndex+1} / 0${tasks[taskId].arms}`;
  centerTrack(); sync(true);
}
function centerTrack() {
  const slide = track.querySelector('.wrist-slide');
  if (!slide) return;
  const gap = parseFloat(getComputedStyle(track).gap), width = slide.getBoundingClientRect().width;
  track.style.left = `${(track.parentElement.clientWidth-width)/2-width-gap}px`;
}
new ResizeObserver(centerTrack).observe(track.parentElement);
function changeWrist(direction) {
  if (changing) return;
  changing = true;
  const width = track.querySelector('.wrist-slide').getBoundingClientRect().width + parseFloat(getComputedStyle(track).gap);
  track.style.transform = `translateX(${-direction*width}px)`;
  carouselTimer = setTimeout(() => {
    wristIndex = (wristIndex+direction+tasks[taskId].arms)%tasks[taskId].arms;
    track.style.transition = 'none'; track.style.transform = ''; renderWrists();
    requestAnimationFrame(() => { track.style.transition = ''; changing = false; });
  }, matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 250);
}
function selectTask(id) {
  clearTimeout(carouselTimer); changing = false; track.style.transform = '';
  const wasPlaying = !globalVideo.paused;
  globalVideo.pause(); taskId = id; wristIndex = 0;
  const task = tasks[id];
  globalVideo.src = media(id, 'global'); globalVideo.poster = media(id, 'global', 'jpg');
  globalVideo.setAttribute('aria-label', `${id} global camera`);
  document.querySelector('#physical-panel').setAttribute('aria-labelledby', `tab-${id}`);
  document.querySelectorAll('[data-task]').forEach(button => { const current = button.dataset.task === id; button.setAttribute('aria-selected', String(current)); button.tabIndex = current ? 0 : -1; });
  document.querySelector('#physical-task-name').textContent = task.title;
  document.querySelector('#physical-task-description').textContent = task.description;
  document.querySelector('#physical-full').textContent = `${task.full}%`;
  document.querySelector('#physical-independent').textContent = `${task.independent}%`;
  document.querySelector('#media-error').hidden = true;
  renderWrists(); if (wasPlaying) safePlay(globalVideo);
}
document.querySelectorAll('[data-task]').forEach(button => button.addEventListener('click', () => selectTask(button.dataset.task)));
document.querySelector('#wrist-prev').addEventListener('click', () => changeWrist(-1));
document.querySelector('#wrist-next').addEventListener('click', () => changeWrist(1));
track.parentElement.addEventListener('keydown', event => { if (['ArrowLeft','ArrowRight'].includes(event.key)) { event.preventDefault(); changeWrist(event.key === 'ArrowRight' ? 1 : -1); } });
let pointerStart = 0;
track.parentElement.addEventListener('pointerdown', event => { pointerStart = event.clientX; });
track.parentElement.addEventListener('pointerup', event => { if (Math.abs(event.clientX-pointerStart)>35) changeWrist(event.clientX<pointerStart ? 1 : -1); });
playButton.addEventListener('click', () => { if (globalVideo.paused) safePlay(globalVideo); else globalVideo.pause(); });
globalVideo.addEventListener('play', () => { setIcon(playButton, 'pause'); playButton.title = 'Pause synchronized cameras'; playButton.setAttribute('aria-label',playButton.title); sync(true); });
globalVideo.addEventListener('pause', () => { setIcon(playButton, 'play'); playButton.title = 'Play synchronized cameras'; playButton.setAttribute('aria-label',playButton.title); sync(); });
globalVideo.addEventListener('ended', () => { wristVideos.forEach(video => video.pause()); });
globalVideo.addEventListener('seeked', () => sync(true));
globalVideo.addEventListener('timeupdate', () => {
  seek.value = globalVideo.duration ? globalVideo.currentTime/globalVideo.duration*1000 : 0;
  document.querySelector('#video-time').textContent = `${seconds(globalVideo.currentTime)} / ${seconds(globalVideo.duration)}`;
  sync();
});
globalVideo.addEventListener('error', () => { const error = document.querySelector('#media-error'); error.textContent = 'This recording could not be loaded. Select another task or reload the page.'; error.hidden = false; });
seek.addEventListener('input', () => { if (Number.isFinite(globalVideo.duration)) globalVideo.currentTime = +seek.value/1000*globalVideo.duration; });
document.querySelector('#video-speed').addEventListener('change', event => { globalVideo.playbackRate = +event.target.value; sync(); });
document.querySelector('#video-fullscreen').addEventListener('click', () => {
  if (globalVideo.requestFullscreen) globalVideo.requestFullscreen().catch(() => {});
  else globalVideo.webkitEnterFullscreen?.();
});
selectTask('frame4');
new IntersectionObserver(entries => { if (!entries[0].isIntersecting) globalVideo.pause(); }, { threshold: .05 }).observe(document.querySelector('#physical-panel'));
document.addEventListener('visibilitychange', () => { if (document.hidden) { globalVideo.pause(); document.querySelectorAll('video').forEach(v => v.pause()); } });

const simulations = [
  { id: 'stack3', name: 'Stack Cube', arms: 3, full: '91.0', independent: '90.5', source: 'Eagle Full rollout', description: 'Sequential placement by three arm-local policies.' },
  { id: 'frame3', name: 'Triangular Frame-over-Peg', arms: 3, full: '79.5', independent: '67.5', source: 'Demonstration replay', description: 'Three contact points, a shared frame, and coupled alignment.' },
  { id: 'frame4', name: 'Frame-over-Peg', arms: 4, full: '89.5', independent: '9.5', source: 'Full policy rollout', description: 'Synchronized manipulation of one object by four agents.' },
  { id: 'arch4', name: 'Arch Building', arms: 4, full: '5.0', independent: '0.0', source: 'Demonstration replay', description: 'A challenging assembly task with support, alignment, and release.' },
];
const grid = document.querySelector('#simulation-grid');
simulations.forEach(task => {
  const article = document.createElement('article'); article.className = 'sim-item';
  article.innerHTML = `<div class="sim-video"><video controls playsinline muted preload="none" poster="static/media/sim-${task.id}.jpg" src="static/media/sim-${task.id}.${videoExtension}" aria-label="${task.name}, ${task.source}"></video><span class="view-tag">${task.source.toUpperCase()}</span></div><header><h3>${task.name}</h3><span>${task.arms} ARMS</span></header><p>${task.description}</p><div class="sim-stats"><span><b>${task.full}%</b> Full</span><span><b>${task.independent}%</b> Independent</span></div>`;
  article.querySelector('video').addEventListener('play', event => { globalVideo.pause(); grid.querySelectorAll('video').forEach(video => { if (video !== event.target) video.pause(); }); });
  grid.append(article);
});
const methods = {
  discovery: ['Interaction structure and adaptation both matter.', 'A controlled two-task study varies peer information, Common aggregation, action interaction, and fine-tuning. Richer information becomes useful when policies learn how to incorporate it.', 'interaction-study.png', 'Controlled architecture and fine-tuning study on Handover Box and Shoes Table'],
  transfer: ['The function transfers. The operator changes.', 'SingleVLA combines action-readout conditions before its local flow-matching action head. The pi0.5 realization uses prefix interactions and gated peer-action attention inside its iterative Action Expert.', 'architecture-transfer.png', 'Three functional roles re-instantiated in SingleVLA and pi0.5'],
  composition: ['An additional arm becomes an additional agent.', 'Teams gain complete local policies and peer interaction relations. The local action dimension stays fixed; each task and team configuration is adapted from pretrained VLAs.', 'team-extension.png', 'Composition of two-, three-, and four-agent teams'],
};
document.querySelectorAll('[data-method]').forEach(button => button.addEventListener('click', () => {
  const id = button.dataset.method, [title, description, image, alt] = methods[id];
  document.querySelectorAll('[data-method]').forEach(tab => { tab.setAttribute('aria-selected', String(tab === button)); tab.tabIndex = tab === button ? 0 : -1; });
  document.querySelector('#method-heading').textContent = title;
  document.querySelector('#method-description').textContent = description;
  const img = document.querySelector('#method-image'); img.src = `static/figures/${image}`; img.alt = alt;
  document.querySelector('#method-panel').setAttribute('aria-labelledby', button.id);
}));
document.querySelectorAll('[role=tablist]').forEach(list => list.addEventListener('keydown', event => {
  if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
  event.preventDefault(); const buttons = [...list.querySelectorAll('[role=tab]')];
  let index = buttons.indexOf(document.activeElement);
  index = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length-1 : (index+(event.key==='ArrowRight'?1:-1)+buttons.length)%buttons.length;
  buttons[index].click(); buttons[index].focus();
}));
const dialog = document.querySelector('#figure-dialog');
document.querySelector('.figure-zoom').addEventListener('click', () => { const source = document.querySelector('#method-image'); dialog.querySelector('img').src = source.src; dialog.querySelector('img').alt = source.alt; dialog.showModal(); });
document.querySelector('#figure-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });

if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const clips = ['frame4','relay4','basket3','relay3'], videos = [...document.querySelectorAll('.ambient')];
  let active = 0, clip = 0, heroVisible = true;
  videos[0].src = media(clips[0], 'ambient'); videos[0].loop = true; safePlay(videos[0]);
  setInterval(() => {
    if (!heroVisible || document.hidden) return;
    const previous = active; active = 1-active; clip = (clip+1)%clips.length;
    videos[active].src = media(clips[clip], 'ambient'); videos[active].loop = true;
    safePlay(videos[active]); videos[active].classList.add('active'); videos[previous].classList.remove('active');
    setTimeout(() => videos[previous].pause(), 2100);
  }, 16000);
  new IntersectionObserver(entries => { heroVisible = entries[0].isIntersecting; if (heroVisible && !document.hidden) safePlay(videos[active]); else videos.forEach(v => v.pause()); }).observe(document.querySelector('.hero'));
}

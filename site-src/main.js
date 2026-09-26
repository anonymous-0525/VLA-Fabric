import { createIcons, ArrowUpRight, ArrowDown, ArrowUp, Pause, Play, RotateCcw, ChevronLeft, ChevronRight, Maximize, Expand, X } from 'lucide';

const icons = { ArrowUpRight, ArrowDown, ArrowUp, Pause, Play, RotateCcw, ChevronLeft, ChevronRight, Maximize, Expand, X };
const refreshIcons = () => createIcons({ icons });
function setIcon(button, name) {
  button.innerHTML = '<i data-lucide="' + name + '"></i>';
  refreshIcons();
}
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const extension = document.createElement('video').canPlayType('video/mp4; codecs="avc1.42E01E"') ? 'mp4' : 'webm';
const media = (id, view, ext = extension) => 'static/media/' + id + '-' + view + '.' + ext;
const seconds = time => Math.floor((time || 0)/60) + ':' + String(Math.floor((time || 0)%60)).padStart(2,'0');
const play = video => video.play().catch(() => {});
const players = [];
function pauseOthers(current) {
  players.forEach(player => { if (player !== current) player.pause(); });
  document.querySelectorAll('.sim-video video').forEach(video => { if (video !== current) video.pause(); });
}
function observePlayback(element, pause) {
  new IntersectionObserver(entries => { if (!entries[0].isIntersecting) pause(); }, { threshold: .05 }).observe(element);
}

const physicalTasks = [
  { id: 'frame4', arms: 4, short: 'Frame4', title: 'Shared-frame installation', description: 'Lift, align, and lower a frame over the central peg.', duration: 56 },
  { id: 'relay4', arms: 4, short: 'Relay4', title: 'Parallel object relays', description: 'Two arm pairs coordinate handover and basket placement.', duration: 54 },
  { id: 'relay3', arms: 3, short: 'Relay3', title: 'Three-arm object relay', description: 'Position the target, hand over, and receive the object.', duration: 62 },
  { id: 'basket3', arms: 3, short: 'Basket3', title: 'Shared-basket placement', description: 'Three local policies place vegetables in a shared basket.', duration: 40 },
];

class CameraPlayer {
  constructor(task) {
    this.task = task;
    this.index = 0;
    this.loaded = false;
    this.root = document.createElement('article');
    this.root.className = 'physical-item';
    this.root.dataset.task = task.id;
    this.root.innerHTML = [
      '<header><div><span class="task-label">' + task.short + ' / ' + task.arms + ' arms</span><h4>' + task.title + '</h4></div></header>',
      '<div class="view-headings"><span>Global</span><div><span>Wrist <b class="wrist-counter">01 / 0' + task.arms + '</b></span><div class="carousel-tools">',
      '<button class="wrist-prev icon-button" title="Previous wrist camera" aria-label="Previous wrist camera"><i data-lucide="chevron-left"></i></button>',
      '<button class="wrist-next icon-button" title="Next wrist camera" aria-label="Next wrist camera"><i data-lucide="chevron-right"></i></button></div></div></div>',
      '<div class="camera-pair"><div class="global-view"><video class="global-video" playsinline muted preload="none" aria-label="' + task.short + ' global camera"></video><button class="video-start" aria-label="Play ' + task.short + ' synchronized cameras"><i data-lucide="play"></i></button></div>',
      '<div class="wrist-window" tabindex="0" aria-label="' + task.short + ' wrist camera carousel"><div class="wrist-track"></div></div></div>',
      '<div class="playback-bar"><button class="video-play icon-button" title="Play synchronized cameras" aria-label="Play synchronized cameras"><i data-lucide="play"></i></button><span class="video-time">0:00 / ' + seconds(task.duration) + '</span>',
      '<input class="video-seek" type="range" min="0" max="1000" value="0" aria-label="' + task.short + ' video position"><select class="video-speed" aria-label="' + task.short + ' playback speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select>',
      '<button class="video-fullscreen icon-button" title="Fullscreen global view" aria-label="Fullscreen global view"><i data-lucide="maximize"></i></button></div>',
      '<p class="task-description">' + task.description + '</p><p class="media-error" role="status" hidden></p>',
    ].join('');
    document.querySelector('#physical-grid').append(this.root);
    this.global = this.root.querySelector('.global-video');
    this.button = this.root.querySelector('.video-play');
    this.seek = this.root.querySelector('.video-seek');
    this.track = this.root.querySelector('.wrist-track');
    this.root.querySelector('.video-start').addEventListener('click', () => this.toggle());
    this.button.addEventListener('click', () => this.toggle());
    this.root.querySelector('.wrist-prev').addEventListener('click', () => this.change(-1));
    this.root.querySelector('.wrist-next').addEventListener('click', () => this.change(1));
    this.track.parentElement.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      event.preventDefault(); this.change(event.key === 'ArrowRight' ? 1 : -1);
    });
    let pointerStart;
    this.track.parentElement.addEventListener('pointerdown', event => { pointerStart = event.clientX; });
    this.track.parentElement.addEventListener('pointerup', event => {
      if (pointerStart !== undefined && Math.abs(event.clientX-pointerStart)>35) this.change(event.clientX<pointerStart ? 1 : -1);
      pointerStart = undefined;
    });
    this.global.addEventListener('play', () => {
      pauseOthers(this);
      this.root.querySelector('.video-start').hidden = true;
      setIcon(this.button, 'pause'); this.button.title = 'Pause synchronized cameras'; this.button.setAttribute('aria-label', this.button.title);
      clearInterval(this.timer); this.timer = setInterval(() => this.sync(), 125);
      this.sync(true);
    });
    this.global.addEventListener('pause', () => {
      clearInterval(this.timer); this.wrist?.pause();
      setIcon(this.button, 'play'); this.button.title = 'Play synchronized cameras'; this.button.setAttribute('aria-label', this.button.title);
      this.root.querySelector('.video-start').hidden = false;
    });
    this.global.addEventListener('ended', () => this.pause());
    this.global.addEventListener('seeked', () => this.sync(true));
    this.global.addEventListener('timeupdate', () => {
      this.seek.value = this.global.duration ? this.global.currentTime/this.global.duration*1000 : 0;
      this.root.querySelector('.video-time').textContent = seconds(this.global.currentTime) + ' / ' + seconds(this.global.duration);
    });
    this.global.addEventListener('error', () => this.error('The recording could not be loaded. Try playing it again.'));
    this.seek.addEventListener('input', () => {
      if (Number.isFinite(this.global.duration)) this.global.currentTime = +this.seek.value/1000*this.global.duration;
    });
    this.root.querySelector('.video-speed').addEventListener('change', event => { this.global.playbackRate = +event.target.value; this.sync(); });
    this.root.querySelector('.video-fullscreen').addEventListener('click', () => {
      if (this.global.requestFullscreen) this.global.requestFullscreen().catch(() => {});
      else this.global.webkitEnterFullscreen?.();
    });
    // Only posters approach the viewport; video bytes wait for an explicit play.
    const posterObserver = new IntersectionObserver(entries => {
      if (!entries[0].isIntersecting) return;
      this.global.poster = media(task.id, 'global', 'jpg');
      this.renderWrists();
      posterObserver.disconnect();
    }, { rootMargin: '250px' });
    posterObserver.observe(this.root);
    new ResizeObserver(() => this.center()).observe(this.track.parentElement);
    observePlayback(this.root, () => this.pause());
  }
  error(message) {
    const error = this.root.querySelector('.media-error'); error.textContent = message; error.hidden = false;
  }
  toggle() {
    if (!this.loaded) {
      this.loaded = true;
      this.global.src = media(this.task.id, 'global');
      this.renderWrists();
    }
    if (this.global.error) this.global.load();
    this.root.querySelector('.media-error').hidden = true;
    if (this.global.paused) {
      this.global.play().catch(() => this.error('Playback could not start. Please try again.'));
    } else this.pause();
  }
  pause() { this.global.pause(); this.wrist?.pause(); clearInterval(this.timer); }
  sync(force = false) {
    const video = this.wrist;
    if (!video || video.readyState < 1) return;
    const offset = this.global.currentTime-video.currentTime;
    if (!video.seeking && (force || Math.abs(offset)>.7)) video.currentTime = this.global.currentTime;
    const correction = !this.global.paused && !force && Math.abs(offset)>.035 ? Math.max(.9,Math.min(1.1,1+offset*.65)) : 1;
    video.playbackRate = this.global.playbackRate*correction;
    if (this.global.paused) video.pause(); else if (video.paused) play(video);
  }
  renderWrists() {
    if (this.wrist) { this.wrist.pause(); this.wrist.removeAttribute('src'); this.wrist.load(); }
    this.track.replaceChildren();
    for (let slot=-1; slot<=1; slot++) {
      const camera = (this.index+slot+this.task.arms)%this.task.arms+1;
      const slide = document.createElement(slot ? 'button' : 'div');
      slide.className = 'wrist-slide' + (slot === 0 ? ' current' : '');
      const poster = media(this.task.id, 'wrist'+camera, 'jpg');
      if (slot) {
        const img = document.createElement('img'); img.src = poster; img.alt = 'Wrist camera ' + camera; img.decoding = 'async';
        slide.append(img); slide.setAttribute('aria-label', 'Select wrist camera '+camera);
        slide.addEventListener('click', () => this.change(slot));
      } else {
        const video = document.createElement('video'); video.playsInline = true; video.muted = true; video.preload = 'none'; video.poster = poster;
        video.setAttribute('aria-label', this.task.short+' wrist camera '+camera);
        if (this.loaded) { video.src = media(this.task.id,'wrist'+camera); video.preload = 'auto'; }
        video.addEventListener('loadedmetadata', () => this.sync(true));
        video.addEventListener('canplay', () => this.sync(true), { once: true });
        video.addEventListener('error', () => this.error('This wrist recording could not be loaded. Select another camera.'));
        slide.append(video); this.wrist = video;
      }
      this.track.append(slide);
    }
    this.root.querySelector('.wrist-counter').textContent = '0'+(this.index+1)+' / 0'+this.task.arms;
    this.center();
  }
  center() {
    const slide = this.track.firstElementChild;
    if (!slide) return;
    const width = slide.getBoundingClientRect().width, gap = parseFloat(getComputedStyle(this.track).gap);
    this.track.style.left = ((this.track.parentElement.clientWidth-width)/2-width-gap)+'px';
  }
  change(direction) {
    if (this.changing) return;
    if (!this.track.firstElementChild) this.renderWrists();
    this.changing = true;
    const width = this.track.firstElementChild.getBoundingClientRect().width + parseFloat(getComputedStyle(this.track).gap);
    this.track.style.transform = 'translateX('+(-direction*width)+'px)';
    setTimeout(() => {
      this.index = (this.index+direction+this.task.arms)%this.task.arms;
      this.track.style.transition = 'none'; this.track.style.transform = ''; this.renderWrists();
      requestAnimationFrame(() => { this.track.style.transition = ''; this.changing = false; });
    }, reduced ? 0 : 220);
  }
}
physicalTasks.forEach(task => players.push(new CameraPlayer(task)));

const simulations = [
  { id: 'stack3', name: 'Stack Cube', short: 'Stack3', description: 'Sequential placement' },
  { id: 'frame3', name: 'Triangular Frame-over-Peg', short: 'Frame3', description: 'Three-point alignment' },
  { id: 'frame4', name: 'Frame-over-Peg', short: 'Frame4', description: 'Shared-object control' },
  { id: 'arch4', name: 'Arch Building', short: 'Arch4', description: 'Multi-stage assembly' },
];
simulations.forEach(task => {
  const article = document.createElement('article'); article.className = 'sim-item';
  article.innerHTML = '<div class="sim-video"><video playsinline muted preload="none" poster="static/media/sim-'+task.id+'.jpg" aria-label="'+task.name+' task illustration"></video><button class="video-start" aria-label="Play '+task.short+' task video"><i data-lucide="play"></i></button></div><header><span class="task-label">'+task.short+'</span><h4>'+task.name+'</h4></header><p>'+task.description+'</p>';
  document.querySelector('#simulation-grid').append(article);
  const video = article.querySelector('video'), button = article.querySelector('button');
  button.addEventListener('click', () => { if (!video.getAttribute('src')) video.src = 'static/media/sim-'+task.id+'.'+extension; video.controls = true; play(video); });
  video.addEventListener('play', () => { button.hidden = true; pauseOthers(video); });
  video.addEventListener('pause', () => { button.hidden = false; });
  observePlayback(article, () => video.pause());
});

// Chapter anchors expose all evidence in the document, without hidden tab panels.
const chapters = [...document.querySelectorAll('.chapter')];
const chapterLinks = [...document.querySelectorAll('[data-chapter]')];
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    chapterLinks.forEach(link => {
      if (link.dataset.chapter === entry.target.id) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  });
}, { rootMargin: '-10% 0px -65% 0px' });
chapters.forEach(section => observer.observe(section));
const dialog = document.querySelector('#figure-dialog');
document.querySelectorAll('.figure-zoom').forEach(button => button.addEventListener('click', () => {
  const source = button.querySelector('img');
  dialog.querySelector('img').src = source.currentSrc || source.src; dialog.querySelector('img').alt = source.alt; dialog.showModal();
}));
document.querySelector('#figure-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) return;
  players.forEach(player => player.pause()); document.querySelectorAll('video').forEach(video => video.pause());
});
refreshIcons();

// Text, navigation, and tables work before the optional 3D bundle is even fetched.
const idle = callback => 'requestIdleCallback' in window ? requestIdleCallback(callback, { timeout: 2000 }) : setTimeout(callback, 300);
let sceneStarted = false;
function bootScene() {
  if (sceneStarted) return;
  sceneStarted = true;
  import('./hero.js').then(module => module.startScene(setIcon)).then(() => {
    if (!reduced && !navigator.connection?.saveData) idle(startAmbient);
  }).catch(() => { document.querySelector('.scene-status').textContent = 'Interactive scene unavailable'; });
}
const sceneObserver = new IntersectionObserver(entries => {
  if (entries[0].isIntersecting) idle(bootScene);
});
sceneObserver.observe(document.querySelector('.hero'));
document.querySelector('.scene-controls').addEventListener('pointerdown', bootScene, { once: true });
document.querySelector('.scene-controls').addEventListener('focusin', bootScene, { once: true });

function startAmbient() {
  const clips = ['frame4','relay4','basket3','relay3'], videos = [...document.querySelectorAll('.ambient')];
  let active = 0, clip = 0, visible = false;
  const start = () => {
    if (!visible || document.hidden) return;
    if (!videos[active].getAttribute('src')) videos[active].src = media(clips[clip], 'ambient');
    videos[active].loop = true; play(videos[active]);
  };
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; if (visible) start(); else videos.forEach(video => video.pause()); }).observe(document.querySelector('.hero'));
  document.addEventListener('visibilitychange', () => { if (!document.hidden) start(); });
  setInterval(() => {
    if (!visible || document.hidden) return;
    const previous = active; active = 1-active; clip = (clip+1)%clips.length;
    const next = videos[active]; next.src = media(clips[clip], 'ambient'); next.loop = true;
    next.addEventListener('playing', () => {
      next.classList.add('active'); videos[previous].classList.remove('active');
      setTimeout(() => { videos[previous].pause(); videos[previous].removeAttribute('src'); videos[previous].load(); }, 2100);
    }, { once: true });
    play(next);
  }, 16000);
}

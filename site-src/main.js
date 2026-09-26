import { createIcons, ArrowUpRight, ArrowDown, ArrowUp, Pause, Play, RotateCcw, ChevronLeft, ChevronRight, Maximize, Expand, X } from 'lucide';

const icons = { ArrowUpRight, ArrowDown, ArrowUp, Pause, Play, RotateCcw, ChevronLeft, ChevronRight, Maximize, Expand, X };
const refreshIcons = () => createIcons({ icons });
function setIcon(button, name) {
  button.innerHTML = '<i data-lucide="' + name + '"></i>';
  refreshIcons();
}
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const codecProbe = document.createElement('video');
const formats = [
  ['mp4', 'video/mp4; codecs="avc1.64001e"'],
  ['webm', 'video/webm; codecs="vp9"'],
].filter(([, type]) => codecProbe.canPlayType(type)).map(([format]) => format);
if (!formats.length) formats.push('mp4', 'webm');
const extension = formats[0];
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
    this.wantPlay = false;
    this.attempt = 0;
    this.root = document.createElement('article');
    this.root.className = 'physical-item';
    this.root.dataset.task = task.id;
    this.root.innerHTML = [
      '<header><div><span class="task-label">' + task.short + ' / ' + task.arms + ' arms</span><h4>' + task.title + '</h4></div></header>',
      '<div class="view-headings"><span>Global</span><div><span>Wrist <b class="wrist-counter">01 / 0' + task.arms + '</b></span><div class="carousel-tools">',
      '<button class="wrist-prev icon-button" title="Previous wrist camera" aria-label="Previous wrist camera"><i data-lucide="chevron-left"></i></button>',
      '<button class="wrist-next icon-button" title="Next wrist camera" aria-label="Next wrist camera"><i data-lucide="chevron-right"></i></button></div></div></div>',
      '<div class="camera-pair"><div class="global-view"><video class="global-video" controls playsinline muted preload="none" aria-label="' + task.short + ' global camera"></video><button class="video-start" aria-label="Play ' + task.short + ' synchronized cameras"><i data-lucide="play"></i></button><span class="video-loading" role="status" hidden>Loading video…</span></div>',
      '<div class="wrist-window" tabindex="0" aria-label="' + task.short + ' wrist camera carousel"><div class="wrist-track"></div><span class="wrist-loading" role="status" hidden>Loading wrist…</span></div></div>',
      '<div class="playback-bar"><button class="video-play icon-button" title="Play synchronized cameras" aria-label="Play synchronized cameras"><i data-lucide="play"></i></button><span class="video-time">0:00 / ' + seconds(task.duration) + '</span>',
      '<input class="video-seek" type="range" min="0" max="1000" value="0" aria-label="' + task.short + ' video position"><select class="video-speed" aria-label="' + task.short + ' playback speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select>',
      '<button class="video-fullscreen icon-button" title="Fullscreen global view" aria-label="Fullscreen global view"><i data-lucide="maximize"></i></button><a class="video-open icon-button" href="' + media(task.id, 'global') + '" target="_blank" rel="noopener" title="Open global video" aria-label="Open global video"><i data-lucide="arrow-up-right"></i></a></div>',
      '<p class="task-description">' + task.description + '</p><p class="media-error" role="status" hidden></p>',
    ].join('');
    document.querySelector('#physical-grid').append(this.root);
    this.global = this.root.querySelector('.global-video');
    this.global.muted = true;
    this.global.defaultMuted = true;
    this.global.playsInline = true;
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
      this.wantPlay = true;
      pauseOthers(this);
      this.setState('loading');
    });
    this.global.addEventListener('playing', () => {
      this.recovering = false;
      this.wantPlay = true;
      this.setState('playing');
      if (this.errorOwner === 'global') this.root.querySelector('.media-error').hidden = true;
      this.ensureWristSource();
      clearInterval(this.timer); this.timer = setInterval(() => this.sync(), 125);
      this.sync(true);
    });
    this.global.addEventListener('waiting', () => {
      if (this.wantPlay) this.setState('loading');
      this.wrist?.pause();
    });
    this.global.addEventListener('pause', () => {
      if (this.recovering) return;
      this.wantPlay = false;
      clearInterval(this.timer); this.wrist?.pause();
      if (this.root.dataset.playback !== 'error') this.setState('paused');
    });
    this.global.addEventListener('ended', () => this.pause());
    this.global.addEventListener('seeked', () => this.sync(true));
    this.global.addEventListener('timeupdate', () => {
      this.seek.value = this.global.duration ? this.global.currentTime/this.global.duration*1000 : 0;
      this.root.querySelector('.video-time').textContent = seconds(this.global.currentTime) + ' / ' + seconds(this.global.duration);
    });
    this.global.addEventListener('error', () => this.sourceError(this.global));
    this.seek.addEventListener('input', () => {
      if (Number.isFinite(this.global.duration)) this.global.currentTime = +this.seek.value/1000*this.global.duration;
    });
    this.root.querySelector('.video-speed').addEventListener('change', event => { this.global.playbackRate = +event.target.value; this.sync(); });
    this.root.querySelector('.video-fullscreen').addEventListener('click', () => {
      if (this.global.requestFullscreen) this.global.requestFullscreen().catch(() => {});
      else this.global.webkitEnterFullscreen?.();
    });
    // Prepare only global metadata near the viewport. Wrist bytes wait for playback.
    const posterObserver = new IntersectionObserver(entries => {
      if (!entries[0].isIntersecting) return;
      this.global.poster = media(task.id, 'global', 'jpg');
      if (!this.track.firstElementChild) this.renderWrists();
      if (!this.loaded) {
        this.loaded = true;
        this.loadSource(this.global, 'global', formats[0], 'metadata');
      }
      posterObserver.disconnect();
    }, { rootMargin: '250px' });
    posterObserver.observe(this.root);
    new ResizeObserver(() => this.center()).observe(this.track.parentElement);
    observePlayback(this.root, () => {
      if (document.fullscreenElement !== this.global && !this.global.webkitDisplayingFullscreen) this.pause();
    });
  }
  setState(state) {
    if (this.root.dataset.playback === state) return;
    this.root.dataset.playback = state;
    const active = state === 'loading' || state === 'playing';
    this.root.querySelector('.video-start').hidden = active;
    this.root.querySelector('.video-loading').hidden = state !== 'loading';
    this.global.setAttribute('aria-busy', String(state === 'loading'));
    setIcon(this.button, active ? 'pause' : 'play');
    this.button.title = state === 'loading' ? 'Cancel loading' : active ? 'Pause synchronized cameras' : 'Play synchronized cameras';
    this.button.setAttribute('aria-label', this.button.title);
  }
  error(message, owner = 'global') {
    this.errorOwner = owner;
    const error = this.root.querySelector('.media-error'); error.textContent = message; error.hidden = false;
  }
  toggle() {
    if (this.wantPlay || !this.global.paused) { this.pause(); return; }
    if (!this.loaded) {
      this.loaded = true;
      this.renderWrists();
      this.loadSource(this.global, 'global');
    }
    if (this.global.error) this.loadSource(this.global, 'global');
    this.root.querySelector('.media-error').hidden = true;
    this.wantPlay = true;
    this.setState('loading');
    this.requestPlay();
  }
  requestPlay() {
    const attempt = ++this.attempt;
    this.global.play().catch(error => {
      if (attempt !== this.attempt || !this.wantPlay || error.name === 'AbortError' || this.global.error) return;
      this.wantPlay = false;
      this.setState('error');
      this.error('Playback was blocked. Use the video controls or open the global video.');
    });
  }
  loadSource(video, view, format = formats[0], preload = 'auto') {
    video.dataset.view = view;
    video.dataset.format = format;
    video.preload = preload;
    video.src = media(this.task.id, view, format);
    video.load();
    if (video === this.global) this.root.querySelector('.video-open').href = video.src;
  }
  sourceError(video) {
    if (video !== this.global && video !== this.wrist) return;
    const next = formats[formats.indexOf(video.dataset.format)+1];
    if (next) {
      const resume = this.wantPlay;
      if (video === this.global) { this.recovering = true; ++this.attempt; }
      this.loadSource(video, video.dataset.view, next);
      if (video === this.global && resume) this.requestPlay();
      return;
    }
    if (video === this.global) {
      this.recovering = false; this.wantPlay = false;
      this.setState('error');
      this.error('The recording could not be loaded. Retry playback or open the global video.');
    } else {
      this.root.querySelector('.wrist-loading').hidden = true;
      this.error('The wrist view could not be loaded. The global video can still play; try another wrist camera.', 'wrist');
    }
  }
  ensureWristSource() {
    if (!this.wrist || this.wrist.getAttribute('src') || !this.loaded) return;
    this.root.querySelector('.wrist-loading').hidden = false;
    this.loadSource(this.wrist, 'wrist'+(this.index+1));
  }
  pause() {
    ++this.attempt;
    this.recovering = false; this.wantPlay = false;
    this.global.pause(); this.wrist?.pause(); clearInterval(this.timer);
    if (this.root.dataset.playback !== 'error') this.setState('paused');
  }
  sync(force = false) {
    const video = this.wrist;
    if (!video || video.readyState < 1 || video.error) return;
    if (this.global.paused || this.global.readyState < 3 || !this.wantPlay) {
      video.pause();
      if (force && !video.seeking && Math.abs(this.global.currentTime-video.currentTime)>.12) video.currentTime = this.global.currentTime;
      return;
    }
    // Seek only on useful data, not on every timer tick while a stream buffers.
    if (video.seeking || (!force && video.readyState < 3)) return;
    const offset = this.global.currentTime-video.currentTime;
    if (Math.abs(offset) > (force ? .12 : .7) && (force || performance.now()-(this.lastWristSeek || 0)>2000)) {
      this.lastWristSeek = performance.now();
      video.currentTime = this.global.currentTime;
      return;
    }
    const correction = !this.global.paused && !force && Math.abs(offset)>.035 ? Math.max(.9,Math.min(1.1,1+offset*.65)) : 1;
    const rate = this.global.playbackRate*correction;
    if (Math.abs(video.playbackRate-rate)>.005) video.playbackRate = rate;
    if (video.paused) play(video);
  }
  renderWrists() {
    if (this.wrist) { const old = this.wrist; this.wrist = null; old.pause(); old.removeAttribute('src'); old.load(); }
    this.track.replaceChildren();
    this.root.querySelector('.wrist-loading').hidden = true;
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
        video.addEventListener('loadedmetadata', () => { if (this.wrist === video) this.sync(true); });
        video.addEventListener('canplay', () => {
          if (this.wrist !== video) return;
          this.root.querySelector('.wrist-loading').hidden = true;
          if (this.errorOwner === 'wrist') this.root.querySelector('.media-error').hidden = true;
          this.sync(true);
        }, { once: true });
        video.addEventListener('seeked', () => { if (this.wrist === video) this.sync(); });
        video.addEventListener('error', () => this.sourceError(video));
        slide.append(video); this.wrist = video;
      }
      this.track.append(slide);
    }
    this.root.querySelector('.wrist-counter').textContent = '0'+(this.index+1)+' / 0'+this.task.arms;
    this.center();
    if (this.wantPlay && this.global.readyState >= 3) this.ensureWristSource();
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

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

export async function startScene(setIcon) {
  const container = document.querySelector('#robot-stage');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'low-power' }); }
  catch { container.querySelector('.scene-status').textContent = 'Interactive scene unavailable'; return; }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.4;
  container.append(renderer.domElement);
  renderer.domElement.setAttribute('aria-label', 'Four agents performing a coordinated frame installation');
  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const environment = pmrem.fromScene(room, .04);
  scene.environment = environment.texture;
  scene.environmentIntensity = .6;
  room.dispose(); pmrem.dispose();
  scene.add(new THREE.HemisphereLight(0xcbe4eb, 0x72827d, 2.3));
  const key = new THREE.DirectionalLight(0xffffff, 3.2); key.position.set(-3, 8, 5); scene.add(key);
  const rim = new THREE.DirectionalLight(0x95c6d5, 1.8); rim.position.set(4, 4, -5); scene.add(rim);
  const camera = new THREE.OrthographicCamera(-6, 6, 3, -3, .1, 100);
  camera.position.set(.7, 7.4, 11.5);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 1, 0); controls.enableDamping = true; controls.enablePan = false;
  controls.enableZoom = false; controls.minPolarAngle = .45; controls.maxPolarAngle = 1.35;
  controls.update(); controls.saveState();
  function resize() {
    const w = container.clientWidth, h = container.clientHeight, aspect = w / h;
    const halfHeight = Math.max(2.7, 4.05 / aspect);
    camera.left = -halfHeight * aspect; camera.right = halfHeight * aspect;
    camera.top = halfHeight; camera.bottom = -halfHeight; camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  new ResizeObserver(resize).observe(container); resize();
  let gltf;
  try { gltf = await new GLTFLoader().loadAsync('static/models/four-arm.glb'); }
  catch { container.querySelector('.scene-status').textContent = 'Scene could not be loaded'; return; }
  const model = gltf.scene; scene.add(model);
  const arms = ['A', 'B', 'C', 'D'].map(letter => model.getObjectByName(`ARM_${letter}`));
  const meshes = [];
  model.traverse(object => {
    if (!object.isMesh) return;
    object.material = object.material.clone();
    object.userData.originalEmissive = object.material.emissive?.clone();
    if (/worktop/i.test(object.material.name)) {
      object.material.color.set('#35464a'); object.material.roughness = .82;
    }
    meshes.push(object);
  });
  arms.forEach((arm, index) => arm.traverse(node => { node.userData.agent = index; }));
  const mixer = new THREE.AnimationMixer(model);
  gltf.animations.forEach(clip => mixer.clipAction(clip).play());
  let selected = -1, pinned = -1, paused = reduced, visible = true, time = 0;
  const labels = arms.map((arm, index) => {
    const label = document.createElement('span'); label.className = 'agent-label';
    label.textContent = `AGENT 0${index + 1}`; container.querySelector('.agent-labels').append(label);
    return label;
  });
  const signals = [];
  const basePoints = arms.map(arm => { const p = new THREE.Vector3(); arm.getWorldPosition(p); p.y += 1.05; return p; });
  for (let a = 0; a < 4; a++) for (let b = a + 1; b < 4; b++) {
    for (let direction = 0; direction < 2; direction++) {
      const start = basePoints[direction ? b : a].clone();
      const end = basePoints[direction ? a : b].clone();
      const middle = start.clone().lerp(end, .5);
      middle.y = 3.15 + .22 * direction;
      const curve = new THREE.QuadraticBezierCurve3(start, middle, end);
      const mat = new THREE.MeshBasicMaterial({ color: 0xa4d4d5, transparent: true, opacity: .11, depthWrite: false });
      const line = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, .009, 5, false), mat);
      scene.add(line);
      const pulse = new THREE.Mesh(new THREE.ConeGeometry(.032, .10, 7),
        new THREE.MeshBasicMaterial({ color: 0xbde9e5, transparent: true, opacity: .55, depthWrite: false }));
      scene.add(pulse); signals.push({ a, b, curve, line, pulse, phase: (a * .17 + b * .22 + direction * .31) % 1 });
    }
  }
  function highlight(index) {
    selected = index; container.dataset.selected = String(index);
    meshes.forEach(mesh => {
      if (mesh.userData.agent === undefined || !mesh.material.emissive) return;
      const active = mesh.userData.agent === selected;
      mesh.material.emissive.copy(mesh.userData.originalEmissive);
      if (active) mesh.material.emissive.set('#328c89');
      mesh.material.emissiveIntensity = active ? .5 : 0;
    });
    signals.forEach(signal => {
      const active = selected >= 0 && (signal.a === selected || signal.b === selected);
      signal.line.material.opacity = active ? .55 : selected >= 0 ? .04 : .11;
      signal.pulse.material.opacity = active ? 1 : selected >= 0 ? .12 : .55;
      signal.line.material.color.set(active ? '#9ae9dc' : '#a4d4d5');
      signal.pulse.scale.setScalar(active ? 1.5 : 1);
    });
    document.querySelectorAll('[data-agent]').forEach(button => button.setAttribute('aria-pressed', String(+button.dataset.agent === selected)));
    labels.forEach((label, index) => label.classList.toggle('selected', index === selected));
  }
  const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();
  renderer.domElement.addEventListener('pointermove', event => {
    if (event.buttons || event.pointerType === 'touch') return;
    const r = renderer.domElement.getBoundingClientRect();
    pointer.set((event.clientX-r.left)/r.width*2-1, -(event.clientY-r.top)/r.height*2+1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(meshes).find(item => item.object.userData.agent !== undefined);
    highlight(hit ? hit.object.userData.agent : pinned);
  });
  renderer.domElement.addEventListener('pointerleave', () => highlight(pinned));
  document.querySelectorAll('[data-agent]').forEach(button => button.addEventListener('click', () => {
    pinned = pinned === +button.dataset.agent ? -1 : +button.dataset.agent; highlight(pinned);
  }));
  const pause = document.querySelector('#scene-pause');
  const updatePause = () => { setIcon(pause, paused ? 'play' : 'pause'); pause.title = paused ? 'Play scene' : 'Pause scene'; pause.setAttribute('aria-label', pause.title); };
  updatePause(); pause.addEventListener('click', () => { paused = !paused; updatePause(); });
  document.querySelector('#scene-reset').addEventListener('click', () => { controls.reset(); pinned = -1; highlight(-1); });
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; }, { threshold: .01 }).observe(container);
  container.querySelector('.scene-poster').hidden = true; container.querySelector('.scene-status').hidden = true;
  container.dataset.ready = 'true';
  let last = performance.now();
  const point = new THREE.Vector3();
  const frame = model.getObjectByName('PROP_four_arm_frame');
  function draw(now) {
    requestAnimationFrame(draw);
    const dt = Math.min((now-last)/1000, .08); last = now;
    if (!visible || document.hidden) return;
    if (!paused) { time += dt; mixer.update(dt); }
    controls.update();
    signals.forEach(signal => {
      const t = (time * .24 + signal.phase) % 1;
      signal.pulse.position.copy(signal.curve.getPoint(t));
      signal.pulse.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), signal.curve.getTangent(t).normalize());
    });
    arms.forEach((arm, index) => {
      arm.getWorldPosition(point); point.y += .2; point.z += .65; point.project(camera);
      labels[index].style.left = `${(point.x*.5+.5)*container.clientWidth}px`;
      labels[index].style.top = `${(-point.y*.5+.5)*container.clientHeight}px`;
    });
    const phase = time % 15;
    document.querySelector('#scene-phase').textContent = phase < 5 ? 'Coordinated lift' : phase < 7 ? 'Shared alignment' : phase < 12 ? 'Synchronized lowering' : 'Reset';
    renderer.render(scene, camera);
    container.dataset.frame = String(Math.floor(time * 30));
    if (frame) container.dataset.frameHeight = String(frame.getWorldPosition(new THREE.Vector3()).y);
  }
  requestAnimationFrame(draw);
}

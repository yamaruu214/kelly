// main.js — bootstrap, fixed-step simulation, render loop, game state machine.
//
// Module contract (each file below is owned independently):
//   core/Renderer.js  WebGL context + adaptive resolution
//   core/PostFX.js    post chain
//   core/Input.js     unified input state
//   core/Audio.js     procedural WebAudio
//   world/Sky.js      atmosphere, sun, IBL
//   world/Level.js    geometry, colliders, spawns
//   player/Controller.js  movement + collision
//   player/Weapons.js     viewmodel, firing, ballistics
//   ai/Enemy.js       bots
//   fx/Particles.js   GPU particles
//   fx/Decals.js      bullet holes
//   ui/HUD.js         hud + menus
//   ui/Touch.js       on-screen controls

import * as THREE from 'three';
import { device, detectTier, presetFor, AdaptiveResolution, TIER } from './core/Perf.js';
import { PostFX } from './core/PostFX.js';
import { Input } from './core/Input.js';
import { AudioEngine } from './core/Audio.js';
import { TextureLibrary } from './world/Textures.js';
import { Sky } from './world/Sky.js';
import { Level } from './world/Level.js';
import { PlayerController } from './player/Controller.js';
import { WeaponSystem } from './player/Weapons.js';
import { EnemyManager } from './ai/Enemy.js';
import { ParticleSystem } from './fx/Particles.js';
import { DecalSystem } from './fx/Decals.js';
import { HUD } from './ui/HUD.js';
import { TouchControls } from './ui/Touch.js';

const FIXED_DT = 1 / 120;          // simulation step; render is decoupled
const MAX_SUBSTEPS = 6;

export class Game {
  constructor() {
    this.state = 'loading';        // loading | menu | playing | paused | dead
    this.accumulator = 0;
    this.clock = null;
    this.elapsed = 0;
  }

  async init(onProgress) {
    const app = document.getElementById('app');

    /* ---------------------------------------------------------- renderer */
    const canvas = document.createElement('canvas');
    app.appendChild(canvas);
    this.canvas = canvas;

    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,            // we resolve with FXAA in the post chain
      alpha: false,
      stencil: false,
      depth: true,
      powerPreference: 'high-performance',
      // iOS Safari drops the context aggressively under memory pressure;
      // preserveDrawingBuffer off keeps our footprint as small as possible.
      preserveDrawingBuffer: false,
      failIfMajorPerformanceCaveat: false,
    });
    this.renderer = renderer;

    const gl = renderer.getContext();
    this.tier = detectTier(gl);
    this.settings = presetFor(this.tier);
    this.settings.maxAnisotropy = Math.min(
      this.settings.anisotropy, renderer.capabilities.getMaxAnisotropy());

    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;   // PostFX applies ACES itself
    renderer.shadowMap.enabled = this.settings.shadows;
    renderer.shadowMap.type = this.settings.softShadows
      ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap;
    renderer.setClearColor(0x0a0d10, 1);
    renderer.info.autoReset = false;

    canvas.addEventListener('webglcontextlost', e => {
      e.preventDefault();
      this.contextLost = true;
      this.hud?.showBanner('GRAPHICS CONTEXT LOST — RELOAD');
    });

    onProgress?.(0.08, 'GPU: ' + this.settings.name);

    /* ------------------------------------------------------------- scene */
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(
      80, 1, 0.05, this.settings.drawDistance);
    this.camera.rotation.order = 'YXZ';    // yaw then pitch — no roll drift

    // Hip-fire FOV. The HUD's FOV slider writes here; ADS and sprint are
    // applied as deltas off it each frame in _postStep.
    this.baseFov = this.settings.fov ?? 80;
    // Measured, not guessed. 0.78 crushed 22-27% of frame to black; 1.40 then
    // overshot to 32-46% clipped. 0.95 balanced the tails but left the frame
    // bimodal, because the real problem was an 11:1 sky-to-ground ratio rather
    // than the exposure point. That ratio is now closed at the source — sky
    // scale down, key light up — so this can sit slightly hot again.
    this.baseExposure = 1.10;

    // The viewmodel lives in its own scene rendered with a narrow FOV so the
    // weapon never clips into world geometry, exactly like every modern FPS.
    this.viewCamera = new THREE.PerspectiveCamera(55, 1, 0.002, 4);

    /* --------------------------------------------------------- resources */
    onProgress?.(0.12, 'SYNTHESIZING MATERIALS');
    this.textures = new TextureLibrary(this.settings.textureSize, this.settings.maxAnisotropy);
    await this.textures.buildAll((p, name) =>
      onProgress?.(0.12 + p * 0.42, 'MATERIAL: ' + name.toUpperCase()));

    onProgress?.(0.56, 'BUILDING ATMOSPHERE');
    this.sky = new Sky(this.scene, this.renderer, this.settings);
    await this.sky.build();

    onProgress?.(0.66, 'CONSTRUCTING LEVEL');
    this.level = new Level(this.scene, this.textures, this.settings, this.sky);
    await this.level.build();

    onProgress?.(0.78, 'LOADING EFFECTS');
    this.particles = new ParticleSystem(this.scene, this.settings);
    this.decals = new DecalSystem(this.scene, this.settings);

    onProgress?.(0.84, 'CALIBRATING AUDIO');
    this.audio = new AudioEngine();

    /* ----------------------------------------------------------- systems */
    this.input = new Input(canvas);
    this.player = new PlayerController(this.camera, this.level, this.input, this.settings);
    this.weapons = new WeaponSystem({
      camera: this.camera, viewCamera: this.viewCamera, scene: this.scene,
      textures: this.textures, audio: this.audio, particles: this.particles,
      decals: this.decals, level: this.level, settings: this.settings,
    });
    this.enemies = new EnemyManager({
      scene: this.scene, level: this.level, textures: this.textures,
      particles: this.particles, audio: this.audio, settings: this.settings,
    });

    onProgress?.(0.92, 'INITIALIZING HUD');
    this.hud = new HUD(app, this);
    this.touch = device.touch ? new TouchControls(app, this) : null;

    /* -------------------------------------------------------------- post */
    this.post = new PostFX(this.renderer, this.scene, this.camera, this.settings);
    this.adaptive = new AdaptiveResolution(this.settings, {
      target: device.mobile ? 60 : 60,
      min: device.mobile ? 0.45 : 0.6,
      max: this.settings.renderScale,
    });
    this.adaptive.onChange = () => this._resize();

    addEventListener('resize', () => this._resize());
    addEventListener('orientationchange', () => setTimeout(() => this._resize(), 250));
    // iOS fires visualViewport resize when the URL bar collapses; without this
    // the canvas keeps the pre-collapse height and letterboxes.
    visualViewport?.addEventListener('resize', () => this._resize());
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && this.state === 'playing') this.pause();
    });

    this._resize();
    this._wireEvents();

    onProgress?.(1.0, 'READY');
    this.state = 'menu';
    this.hud.showMenu();
    return this;
  }

  _wireEvents() {
    // Weapons is built before enemies exist, so the hit-test link is bound here.
    this.weapons.enemies = this.enemies;
    this.enemies.player = this.player;

    this.input.onPointerLockLost = () => {
      if (this.state === 'playing' && !device.touch) this.pause();
    };
    this.weapons.onHit = (hit) => {
      this.hud.hitmarker(hit.headshot, hit.killed);
      if (hit.killed) { this.score += hit.headshot ? 150 : 100; this.kills++; }
    };
    this.player.onDamage = (amount, dir) => {
      this.post.damageAmount = Math.min(1, this.post.damageAmount + amount / 60);
      this.hud.damageIndicator(dir);
      this.audio.play('hurt');
    };
    this.player.onDeath = () => this.die();
  }

  _resize() {
    const w = Math.max(1, visualViewport?.width || innerWidth);
    const h = Math.max(1, visualViewport?.height || innerHeight);
    const dpr = Math.min(device.dpr, this.settings.maxPixelRatio);
    const scale = this.adaptive ? this.adaptive.scale : this.settings.renderScale;

    const rw = Math.floor(w * dpr * scale);
    const rh = Math.floor(h * dpr * scale);

    this.renderer.setPixelRatio(1);            // we do our own scaling
    this.renderer.setSize(rw, rh, false);
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';

    const aspect = w / h;
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    this.viewCamera.aspect = aspect;
    this.viewCamera.updateProjectionMatrix();

    this.post?.setSize(rw, rh);
    this.hud?.resize(w, h);
    this.touch?.resize(w, h);

    document.body.classList.toggle('portrait-lock', device.mobile && h > w * 1.05);
  }

  /* ------------------------------------------------------------- states */
  start() {
    this.score = 0; this.kills = 0; this.wave = 0;
    this.player.reset(this.level.playerSpawn);
    this.enemies.reset();
    this.weapons.reset();
    this.decals.clear();
    this.particles.clear();
    this.state = 'playing';
    this.hud.showGame();
    this.audio.resume();
    this.audio.startAmbience();
    if (!device.touch) this.input.requestPointerLock();
    this.nextWave();
  }

  nextWave() {
    this.wave++;
    const count = Math.min(this.settings.enemyCount, 3 + this.wave * 2);
    this.enemies.spawnWave(count, this.wave);
    this.hud.announce(`WAVE ${this.wave}`, `${count} HOSTILES`);
  }

  pause() {
    if (this.state !== 'playing') return;
    this.state = 'paused';
    this.input.releaseAll();
    this.hud.showPause();
    if (document.pointerLockElement) document.exitPointerLock();
  }

  resume() {
    if (this.state !== 'paused') return;
    this.state = 'playing';
    this.hud.showGame();
    this.audio.resume();
    if (!device.touch) this.input.requestPointerLock();
  }

  die() {
    if (this.state === 'dead') return;
    this.state = 'dead';
    this.input.releaseAll();
    this.audio.play('death');
    this.hud.showDeath(this.score, this.kills, this.wave);
    if (document.pointerLockElement) document.exitPointerLock();
  }

  /* --------------------------------------------------------------- loop */
  run() {
    let last = performance.now();
    const frame = (now) => {
      this.rafId = requestAnimationFrame(frame);
      if (this.contextLost) return;

      const rawDt = (now - last);
      last = now;
      this.adaptive.update(rawDt);

      // A tab that was backgrounded returns a huge delta; clamping stops the
      // player from tunnelling through a wall on the first frame back.
      const dt = Math.min(rawDt / 1000, 0.1);
      this.elapsed += dt;

      if (this.state === 'playing') {
        this.input.beginFrame(dt);
        this.touch?.update(dt);

        this.accumulator += dt;
        let steps = 0;
        while (this.accumulator >= FIXED_DT && steps < MAX_SUBSTEPS) {
          this._step(FIXED_DT);
          this.accumulator -= FIXED_DT;
          steps++;
        }
        if (steps === MAX_SUBSTEPS) this.accumulator = 0;   // shed the backlog

        this._postStep(dt);
        this.input.endFrame();
      } else {
        this.sky.update(dt, this.elapsed);
      }

      this._render(dt);
      this.renderer.info.reset();
    };
    this.rafId = requestAnimationFrame(frame);
  }

  _step(dt) {
    this.player.update(dt);
    this.weapons.update(dt, this.input, this.player);
    this.enemies.update(dt, this.player, this.weapons);
  }

  _postStep(dt) {
    this.sky.update(dt, this.elapsed);
    this.particles.update(dt, this.camera);
    this.decals.update(dt);

    this.post.adsAmount = this.weapons.adsProgress;
    this.post.damageAmount = Math.max(0, this.post.damageAmount - dt * 1.6);
    this.post.flashAmount = Math.max(0, this.post.flashAmount - dt * 3.0);
    // Golden-hour sky is far brighter than the shaded ground; without pulling
    // the base down the whole upper half of frame clips before ACES can roll off.
    this.post.exposure = this.baseExposure * this.weapons.exposureBoost;

    // Aiming narrows the FOV; sprinting widens it. Both are eased, not snapped.
    const targetFov = this.baseFov - this.weapons.adsProgress * this.weapons.adsFovReduction
                         + (this.player.sprinting ? 6 : 0);
    this.camera.fov += (targetFov - this.camera.fov) * Math.min(1, dt * 12);
    this.camera.updateProjectionMatrix();

    this.audio.setListener(this.camera);
    this.hud.update(dt, this);

    if (this.enemies.aliveCount === 0 && this.enemies.spawnedThisWave > 0) {
      this.enemies.spawnedThisWave = 0;
      setTimeout(() => { if (this.state === 'playing') this.nextWave(); }, 2600);
    }
  }

  _render(dt) {
    // PostFX draws the viewmodel into the HDR buffer itself, on a cleared
    // depth range, so the weapon is tone-mapped and blooms like the world.
    const showWeapon = this.state === 'playing' || this.state === 'paused';
    this.post.viewScene = showWeapon ? this.weapons.viewScene : null;
    this.post.viewCamera = this.viewCamera;
    this.post.render(dt);
  }
}

export async function boot() {
  const bootEl = document.getElementById('boot');
  const bar = bootEl.querySelector('#bar i');
  const msg = document.getElementById('bootmsg');

  const game = new Game();
  window.__game = game;   // handy for debugging from the console

  await game.init((p, text) => {
    bar.style.width = Math.round(p * 100) + '%';
    if (text) msg.textContent = text;
  });

  // One frame at full progress before the curtain lifts, so the bar visibly
  // completes instead of jumping from 90% to gone.
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  bootEl.classList.add('hide');
  setTimeout(() => bootEl.remove(), 700);

  game.run();
  return game;
}

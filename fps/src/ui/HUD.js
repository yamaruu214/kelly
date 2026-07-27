// HUD.js — every pixel of 2D: menus, crosshair, readouts, minimap, screens.
//
// The whole layer is DOM + CSS except the crosshair and the minimap, which are
// canvases because they redraw from game state instead of from class changes.
// Nothing here loads an asset: type is the system monospace stack, art is CSS
// gradients and inline SVG.
//
// Two hard rules the rest of the file obeys:
//   1. Every element that is not a button carries pointer-events:none, or it
//      eats the touch-control drags underneath it.
//   2. Every edge-anchored element pads with env(safe-area-inset-*); the iPhone
//      notch and home indicator overlap the corners the HUD wants.

import * as THREE from 'three';
import { device, TIER, presetFor } from '../core/Perf.js';

const STORE_KEY = 'blacksite.prefs.v1';

const DEFAULTS = {
  sens: 1.0,          // multiplier over Input's tuned base, not an absolute
  invertY: false,
  fov: 80,            // matches main.js's base FOV, so 80 is a no-op
  quality: 'auto',
  volume: 0.8,
  mute: false,
  perf: false,
};

const BASE_SENS = 0.0022;
const BASE_TOUCH_SENS = 0.0040;
const BASE_FOV = 80;              // fallback only, for crosshair math pre-boot

const TIER_BY_NAME = { low: TIER.LOW, medium: TIER.MED, high: TIER.HIGH, ultra: TIER.ULTRA };

const MAP_RANGE = 34;             // metres from edge to edge of the minimap disc
const MAP_CULL = 50;              // enemies past this are not drawn at all
const MAP_HZ = 20;
const HIT_LIFE = 0.18;

const CONTROLS = [
  ['MOVE', 'W A S D'], ['LOOK', 'MOUSE'], ['FIRE', 'LMB'], ['AIM', 'RMB'],
  ['RELOAD', 'R'], ['SPRINT', 'SHIFT'], ['CROUCH', 'CTRL / C'], ['JUMP', 'SPACE'],
  ['MELEE', 'F'], ['GRENADE', 'G'], ['LEAN', 'Q / E'], ['WEAPON', '1 2 3'],
  ['PAUSE', 'ESC'], ['PERF', '` BACKQUOTE'],
];

export class HUD {
  constructor(rootEl, game) {
    this.game = game;
    this.w = window.innerWidth;
    this.h = window.innerHeight;
    this.dpr = Math.min(device.dpr, 2);

    this.prefs = loadPrefs();

    this._injectStyle();
    this._buildDOM(rootEl);
    this._bindUI();

    this._hit = null;            // active hitmarker
    this._hitFlash = 0;          // crosshair tint after a confirmed hit
    this._dmg = [];             // directional indicators, world-space
    this._feed = [];
    this._mapAcc = 0;
    this._perfAcc = 0;
    this._crossDirty = true;
    this._lastGap = -1;
    this._cache = {};
    this._mapStatic = null;

    this.applyPrefs();
    this.resize(this.w, this.h);
    this.showMenu();

    addEventListener('keydown', e => {
      if (e.code === 'Backquote') this.setPref('perf', !this.prefs.perf);
    });
  }

  /* ------------------------------------------------------------------ css */
  _injectStyle() {
    if (document.getElementById('hud-style')) return;
    const s = document.createElement('style');
    s.id = 'hud-style';
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  /* ------------------------------------------------------------------ dom */
  _buildDOM(rootEl) {
    const root = document.createElement('div');
    root.id = 'hud';
    root.dataset.screen = 'menu';
    root.innerHTML = MARKUP;
    root.classList.toggle('touch', device.touch);
    rootEl.appendChild(root);
    this.root = root;

    const q = sel => root.querySelector(sel);
    this.el = {
      cross: q('.hud-cross'),
      map: q('.hud-map'),
      mapWrap: q('.hud-mapwrap'),
      feed: q('.hud-feed'),
      ammoMag: q('.hud-ammo .mag'),
      ammoRes: q('.hud-ammo .res'),
      ammoName: q('.hud-ammo .wname'),
      ammoMode: q('.hud-ammo .wmode'),
      ammo: q('.hud-ammo'),
      health: q('.hud-health'),
      hpNum: q('.hud-health .hpnum'),
      hpSegs: q('.hud-health .segs'),
      lowhp: q('.hud-lowhp'),
      dmg: q('.hud-dmg'),
      annTitle: q('.hud-ann .t'),
      annSub: q('.hud-ann .s'),
      ann: q('.hud-ann'),
      banner: q('.hud-banner'),
      perf: q('.hud-perf'),
      menu: q('.hud-menu'),
      settings: q('.hud-settings'),
      controls: q('.hud-controls'),
      death: q('.hud-death'),
      dScore: q('.hud-death .v-score'),
      dKills: q('.hud-death .v-kills'),
      dWave: q('.hud-death .v-wave'),
    };

    this.ctx = this.el.cross.getContext('2d');
    this.mapCtx = this.el.map.getContext('2d');

    const list = root.querySelector('.hud-controls .rows');
    list.innerHTML = CONTROLS.map(([k, v]) =>
      `<div class="row"><span>${k}</span><b>${v}</b></div>`).join('');
  }

  _bindUI() {
    const g = this.game;
    // querySelectorAll, not querySelector: SETTINGS exists on both the menu and
    // the pause screen and both copies must work.
    const on = (sel, fn) => {
      // click (not touchstart) so a drag that started on a button and slid off
      // does not fire; iOS synthesises it fast enough to feel immediate.
      this.root.querySelectorAll(sel).forEach(b =>
        b.addEventListener('click', ev => { ev.preventDefault(); fn(ev); }));
    };

    on('[data-act=play]', () => g.start());
    on('[data-act=settings]', () => this._openSettings());
    on('[data-act=controls]', () => this._panel('controls', true));
    on('[data-act=close-settings]', () => this._closeSettings());
    on('[data-act=close-controls]', () => this._panel('controls', false));
    on('[data-act=resume]', () => g.resume());
    on('[data-act=quit]', () => { this._toMenu(); });
    on('[data-act=retry]', () => g.start());
    on('[data-act=menu]', () => { this._toMenu(); });
    on('[data-act=pause]', () => g.pause());

    const bind = (sel, key, read) => {
      const node = this.root.querySelector(sel);
      if (!node) return;
      const ev = node.type === 'range' ? 'input' : 'change';
      node.addEventListener(ev, () => this.setPref(key, read(node)));
      this._inputs = this._inputs || [];
      this._inputs.push([node, key]);
    };
    bind('[data-pref=sens]', 'sens', n => +n.value);
    bind('[data-pref=fov]', 'fov', n => +n.value);
    bind('[data-pref=volume]', 'volume', n => +n.value);
    bind('[data-pref=invertY]', 'invertY', n => n.checked);
    bind('[data-pref=mute]', 'mute', n => n.checked);
    bind('[data-pref=perf]', 'perf', n => n.checked);
    bind('[data-pref=quality]', 'quality', n => n.value);
  }

  /** Game has no menu() of its own; the state flag plus the screen is enough. */
  _toMenu() {
    this.game.state = 'menu';
    this.game.input?.releaseAll?.();
    if (document.pointerLockElement) document.exitPointerLock();
    this.showMenu();
  }

  /* ------------------------------------------------------------ settings */
  setPref(key, value) {
    this.prefs[key] = value;
    savePrefs(this.prefs);
    this.applyPrefs();
  }

  applyPrefs() {
    const g = this.game, p = this.prefs;

    if (g.input) {
      g.input.sensitivity = BASE_SENS * p.sens;
      g.input.touchSensitivity = BASE_TOUCH_SENS * p.sens;
      g.input.invertY = !!p.invertY;
    }
    g.settings.fov = p.fov;
    g.audio?.setMasterVolume?.(p.mute ? 0 : p.volume);

    if (p.quality !== 'auto' && TIER_BY_NAME[p.quality] !== undefined) {
      const preset = presetFor(TIER_BY_NAME[p.quality]);
      // Only the resolution knobs are safe to swap at runtime — shadow maps,
      // decal pools and particle buffers were sized once at construction.
      g.settings.renderScale = preset.renderScale;
      g.settings.maxPixelRatio = preset.maxPixelRatio;
      g.settings.name = preset.name;
      if (g.adaptive) {
        g.adaptive.max = preset.renderScale;
        g.adaptive.scale = Math.min(g.adaptive.scale, preset.renderScale);
      }
      g._resize?.();
    }

    for (const [node, key] of this._inputs || []) {
      const v = p[key];
      if (node.type === 'checkbox') node.checked = !!v; else node.value = v;
      const out = node.parentElement?.querySelector('.val');
      if (out) out.textContent = formatPref(key, v);
    }

    this.root.classList.toggle('show-perf', !!p.perf);
  }

  _openSettings() { this._panel('settings', true); }
  _closeSettings() { this._panel('settings', false); }

  _panel(name, open) {
    this.root.classList.toggle('panel-' + name, open);
  }

  /* -------------------------------------------------------------- layout */
  resize(w, h) {
    this.w = w; this.h = h;
    this.dpr = Math.min(device.dpr, 2);

    const c = this.el.cross;
    const cs = Math.round(Math.min(300, Math.max(180, Math.min(w, h) * 0.45)));
    c.style.width = cs + 'px'; c.style.height = cs + 'px';
    c.width = Math.round(cs * this.dpr); c.height = Math.round(cs * this.dpr);
    this.crossSize = cs;

    const m = this.el.map;
    const ms = m.getBoundingClientRect().width || 120;
    m.width = Math.round(ms * this.dpr); m.height = Math.round(ms * this.dpr);
    this.mapSize = ms;

    this._crossDirty = true;
    this._mapAcc = 1;
  }

  /* --------------------------------------------------------------- state */
  showMenu() {
    this.root.dataset.screen = 'menu';
    this._panel('settings', false);
    this._panel('controls', false);
    this.showBanner('');
    this._clearTransient();
  }

  showGame() {
    this.root.dataset.screen = 'game';
    this._panel('settings', false);
    this._panel('controls', false);
    this._cache = {};          // force a full re-read after a restart
  }

  showPause() {
    this.root.dataset.screen = 'pause';
    this.el.lowhp.classList.remove('on');
  }

  showDeath(score, kills, wave) {
    this.root.dataset.screen = 'death';
    this._panel('settings', false);
    this._clearTransient();
    this._countUp(this.el.dScore, score, 0.8);
    this._countUp(this.el.dKills, kills, 0.6);
    this._countUp(this.el.dWave, wave, 0.5);
  }

  _clearTransient() {
    this.el.lowhp.classList.remove('on');
    for (const d of this._dmg) d.node.remove();
    this._dmg.length = 0;
    for (const f of this._feed) f.node.remove();
    this._feed.length = 0;
    this._hit = null;
    this._crossDirty = true;
  }

  /** rAF-driven so the numbers land even while the sim is stopped. */
  _countUp(node, target, secs) {
    const t0 = performance.now();
    const step = (now) => {
      const k = Math.min(1, (now - t0) / (secs * 1000));
      const e = 1 - Math.pow(1 - k, 3);
      node.textContent = String(Math.round(target * e)).padStart(2, '0');
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* ------------------------------------------------------- notifications */
  announce(title, subtitle) {
    const { ann, annTitle, annSub } = this.el;
    annTitle.textContent = title || '';
    annSub.textContent = subtitle || '';
    ann.classList.remove('go');
    void ann.offsetWidth;          // restart the animation on a re-announce
    ann.classList.add('go');
    clearTimeout(this._annT);
    this._annT = setTimeout(() => ann.classList.remove('go'), 2900);
  }

  showBanner(text) {
    this.el.banner.textContent = text || '';
    this.el.banner.classList.toggle('on', !!text);
  }

  killfeed(text) {
    const row = document.createElement('div');
    row.className = 'kf';
    row.innerHTML = text;
    this.el.feed.prepend(row);
    void row.offsetWidth;
    row.classList.add('in');
    this._feed.push({ node: row, t: 0 });
    while (this._feed.length > 5) this._feed.shift().node.remove();
  }

  hitmarker(headshot, killed) {
    this._hit = { t: 0, headshot: !!headshot, killed: !!killed };
    this._hitFlash = 0.12;
    this._crossDirty = true;
    if (killed) {
      const w = this.game.weapons?.current?.name || 'MELEE';
      this.killfeed(`<b>YOU</b> <i>&#9656;</i> ${esc(w)}${headshot ? ' <u>HS</u>' : ''} <i>&#9656;</i> <s>HOSTILE</s>`);
    }
  }

  /**
   * dir is a world-space vector pointing from the player toward the hurt
   * source. We keep it in world space and re-project it against the camera
   * yaw every frame, otherwise the arc lies as soon as the player turns.
   */
  damageIndicator(dir) {
    if (!dir) return;
    const x = dir.x || 0, z = dir.z || 0;
    const len = Math.hypot(x, z) || 1;
    const node = document.createElement('div');
    node.className = 'di';
    node.innerHTML = DAMAGE_ARC;
    this.el.dmg.appendChild(node);
    this._dmg.push({ node, x: x / len, z: z / len, t: 0, life: 1.5 });
    while (this._dmg.length > 4) this._dmg.shift().node.remove();
  }

  /* ---------------------------------------------------------------- tick */
  update(dt, game) {
    const playing = game.state === 'playing';

    this._applyFov(game);
    if (playing) {
      this._updateAmmo(game);
      this._updateHealth(game);
      this._updateMap(dt, game);
    }
    this._updateDamage(dt, game);
    this._updateFeed(dt);
    this._updateCrosshair(dt, game);
    this._updatePerf(dt, game);
  }

  /** main.js eases camera.fov toward game.baseFov, applying ADS and sprint as
   *  deltas off it, so the slider only has to move the base. */
  _applyFov(game) {
    game.baseFov = this.prefs.fov;
  }

  _updateAmmo(game) {
    const w = game.weapons?.current;
    if (!w) return;
    const c = this._cache;
    if (w.ammo !== c.ammo) {
      c.ammo = w.ammo;
      this.el.ammoMag.textContent = String(w.ammo ?? 0).padStart(2, '0');
    }
    if (w.reserve !== c.reserve) {
      c.reserve = w.reserve;
      this.el.ammoRes.textContent = String(w.reserve ?? 0).padStart(2, '0');
    }
    if (w.name !== c.wname) { c.wname = w.name; this.el.ammoName.textContent = w.name || ''; }
    if (w.fireMode !== c.mode) { c.mode = w.fireMode; this.el.ammoMode.textContent = w.fireMode || ''; }

    const mag = w.magSize || 1;
    const low = w.ammo <= 0 ? 'empty' : (w.ammo / mag <= 0.25 ? 'low' : '');
    if (low !== c.low) {
      c.low = low;
      this.el.ammo.classList.toggle('low', low === 'low');
      this.el.ammo.classList.toggle('empty', low === 'empty');
    }
  }

  _updateHealth(game) {
    const hp = Math.max(0, Math.round(game.player?.health ?? 100));
    const max = game.player?.maxHealth || 100;
    const c = this._cache;
    if (hp === c.hp) return;
    c.hp = hp;
    this.el.hpNum.textContent = String(hp).padStart(3, '0');

    const frac = Math.max(0, Math.min(1, hp / max));
    const segs = 6, lit = Math.ceil(frac * segs);
    if (!this._segNodes) {
      this._segNodes = [];
      for (let i = 0; i < segs; i++) {
        const d = document.createElement('i');
        this.el.hpSegs.appendChild(d);
        this._segNodes.push(d);
      }
    }
    for (let i = 0; i < segs; i++) this._segNodes[i].classList.toggle('off', i >= lit);

    this.el.health.classList.toggle('hurt', frac <= 0.5);
    this.el.health.classList.toggle('crit', frac <= 0.28);
    // The heartbeat vignette is the primary low-health tell; the numbers are
    // secondary, exactly the CoD read.
    this.el.lowhp.classList.toggle('on', frac <= 0.28 && game.state === 'playing');
  }

  _updateDamage(dt, game) {
    if (!this._dmg.length) return;
    const yaw = game.camera ? game.camera.rotation.y : 0;
    const cos = Math.cos(yaw), sin = Math.sin(yaw);
    for (let i = this._dmg.length - 1; i >= 0; i--) {
      const d = this._dmg[i];
      d.t += dt;
      if (d.t >= d.life || game.state === 'menu') {
        d.node.remove(); this._dmg.splice(i, 1); continue;
      }
      // Rotate the world offset into view space: right = (cos,-sin),
      // forward = (-sin,-cos); screen angle is measured clockwise from up.
      const sx = d.x * cos - d.z * sin;
      const sy = d.x * sin + d.z * cos;
      const ang = Math.atan2(sx, -sy) * 180 / Math.PI;
      const k = 1 - d.t / d.life;
      d.node.style.transform = `rotate(${ang.toFixed(1)}deg)`;
      d.node.style.opacity = (k < 0.25 ? k / 0.25 : 1) * (0.35 + 0.65 * k);
    }
  }

  _updateFeed(dt) {
    for (let i = this._feed.length - 1; i >= 0; i--) {
      const f = this._feed[i];
      f.t += dt;
      if (f.t > 5) { f.node.classList.add('out'); }
      if (f.t > 5.5) { f.node.remove(); this._feed.splice(i, 1); }
    }
  }

  /* ----------------------------------------------------------- crosshair */
  _updateCrosshair(dt, game) {
    if (this._hitFlash > 0) { this._hitFlash -= dt; this._crossDirty = true; }
    if (this._hit) {
      this._hit.t += dt;
      if (this._hit.t >= HIT_LIFE) this._hit = null;
      this._crossDirty = true;
    }

    const gap = this._spreadPixels(game);
    if (Math.abs(gap - this._lastGap) > 0.35) { this._lastGap = gap; this._crossDirty = true; }

    const ads = game.weapons?.adsProgress || 0;
    if (ads !== this._lastAds) { this._lastAds = ads; this._crossDirty = true; }

    if (!this._crossDirty) return;
    this._crossDirty = false;
    this._drawCross(this._lastGap, ads, game.state === 'playing');
  }

  /**
   * Converts the weapon's cone half-angle to a screen radius. A ray at angle a
   * off the axis lands tan(a)/tan(fov/2) of the way from the centre to the top
   * of the frame, so the gap is that fraction of half the viewport height.
   */
  _spreadPixels(game) {
    const spread = game.weapons?.spread || 0;
    const fov = (game.camera?.fov || BASE_FOV) * Math.PI / 180;
    const px = Math.tan(spread) / Math.tan(fov / 2) * (this.h / 2);
    return Math.min(this.crossSize * 0.42, 4 + px);
  }

  _drawCross(gap, ads, playing) {
    const ctx = this.ctx, s = this.crossSize, c = s / 2;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, s, s);
    if (!playing) return;

    // Fully aimed down sights means the sight picture is the reticle.
    if (ads < 0.92) {
      const a = 1 - ads;
      const len = Math.max(5, Math.min(11, s * 0.03));
      const hot = this._hitFlash > 0;
      ctx.strokeStyle = hot ? 'rgba(255,255,255,0.95)' : `rgba(232,230,225,${0.86 * a})`;
      ctx.lineWidth = 2;
      ctx.lineCap = 'butt';
      ctx.beginPath();
      ctx.moveTo(c, c - gap); ctx.lineTo(c, c - gap - len);
      ctx.moveTo(c, c + gap); ctx.lineTo(c, c + gap + len);
      ctx.moveTo(c - gap, c); ctx.lineTo(c - gap - len, c);
      ctx.moveTo(c + gap, c); ctx.lineTo(c + gap + len, c);
      ctx.stroke();

      ctx.fillStyle = hot ? 'rgba(255,122,26,1)' : `rgba(255,122,26,${0.9 * a})`;
      ctx.beginPath();
      ctx.arc(c, c, 1.6, 0, Math.PI * 2);
      ctx.fill();
    }

    const hit = this._hit;
    if (!hit) return;
    const k = hit.t / HIT_LIFE;
    const alpha = 1 - k * k;
    const pop = 1 + 0.55 * (1 - k) * (1 - k);
    const inner = (hit.headshot ? 10 : 6.5) * pop;
    const outer = (hit.headshot ? 22 : 15) * pop;
    ctx.strokeStyle = hit.killed
      ? `rgba(255,59,48,${alpha})` : `rgba(255,255,255,${alpha})`;
    ctx.lineWidth = hit.headshot ? 3 : 2.2;
    ctx.lineCap = 'round';
    ctx.beginPath();
    for (let i = 0; i < 4; i++) {
      const a = Math.PI / 4 + i * Math.PI / 2;
      const dx = Math.cos(a), dy = Math.sin(a);
      ctx.moveTo(c + dx * inner, c + dy * inner);
      ctx.lineTo(c + dx * outer, c + dy * outer);
    }
    ctx.stroke();
  }

  /* ------------------------------------------------------------- minimap */
  _updateMap(dt, game) {
    this._mapAcc += dt;
    if (this._mapAcc < 1 / MAP_HZ) return;
    this._mapAcc = 0;
    if (!this._mapStatic) this._buildMapStatic(game);

    const ctx = this.mapCtx, s = this.mapSize, c = s / 2;
    const p = game.player?.position || game.camera?.position;
    if (!ctx || !p) return;
    const yaw = game.camera ? game.camera.rotation.y : 0;
    const ppm = s / MAP_RANGE;                 // map pixels per metre

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, s, s);
    ctx.save();
    ctx.beginPath();
    ctx.arc(c, c, c - 1, 0, Math.PI * 2);
    ctx.clip();

    ctx.fillStyle = 'rgba(6,8,9,0.72)';
    ctx.fillRect(0, 0, s, s);

    const st = this._mapStatic;
    if (st) {
      ctx.save();
      ctx.translate(c, c);
      ctx.rotate(yaw);
      ctx.scale(ppm / st.ppm, ppm / st.ppm);
      ctx.translate(-(p.x - st.minX) * st.ppm, -(p.z - st.minZ) * st.ppm);
      ctx.drawImage(st.canvas, 0, 0);
      ctx.restore();
    }

    const list = game.enemies?.list || [];
    ctx.save();
    ctx.translate(c, c);
    ctx.rotate(yaw);
    for (let i = 0; i < list.length; i++) {
      const e = list[i];
      if (!isAlive(e)) continue;
      const ep = e.position || e.mesh?.position || e.root?.position || e.object?.position;
      if (!ep) continue;
      const dx = ep.x - p.x, dz = ep.z - p.z;
      const dist = Math.hypot(dx, dz);
      if (dist > MAP_CULL) continue;
      const lim = c - 6;
      let bx = dx * ppm, bz = dz * ppm;
      const bl = Math.hypot(bx, bz);
      const edge = bl > lim;
      if (edge) { bx = bx / bl * lim; bz = bz / bl * lim; }
      ctx.fillStyle = edge ? 'rgba(255,59,48,0.55)' : '#ff3b30';
      ctx.beginPath();
      ctx.arc(bx, bz, edge ? 2 : 3, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
    ctx.restore();

    // Player arrow last and unrotated — it is the fixed reference the rotating
    // map turns around.
    ctx.fillStyle = '#ff7a1a';
    ctx.beginPath();
    ctx.moveTo(c, c - 6);
    ctx.lineTo(c + 4.5, c + 5);
    ctx.lineTo(c, c + 2.5);
    ctx.lineTo(c - 4.5, c + 5);
    ctx.closePath();
    ctx.fill();
  }

  /**
   * The level footprint never changes, so it is projected to top-down once and
   * cached; the per-frame cost is then a single rotated drawImage.
   */
  _buildMapStatic(game) {
    const raw = game.level?.colliders;
    if (!raw || !raw.length) { this._mapStatic = null; return; }

    const boxes = [];
    let minX = Infinity, minZ = Infinity, maxX = -Infinity, maxZ = -Infinity;
    for (const col of raw) {
      const b = toBox(col);
      if (!b) continue;
      const w = b.maxX - b.minX, d = b.maxZ - b.minZ, hgt = b.maxY - b.minY;
      // Floors and the outer shell would flood-fill the disc; only things that
      // stand up like walls and cover read as map geometry.
      if (hgt < 0.55 || w > 300 || d > 300) continue;
      boxes.push(b);
      if (b.minX < minX) minX = b.minX;
      if (b.minZ < minZ) minZ = b.minZ;
      if (b.maxX > maxX) maxX = b.maxX;
      if (b.maxZ > maxZ) maxZ = b.maxZ;
    }
    if (!boxes.length) { this._mapStatic = null; return; }

    const pad = 4;
    minX -= pad; minZ -= pad; maxX += pad; maxZ += pad;
    const span = Math.max(maxX - minX, maxZ - minZ);
    const ppm = Math.max(1, Math.min(8, 1024 / span));
    const cw = Math.ceil((maxX - minX) * ppm), ch = Math.ceil((maxZ - minZ) * ppm);

    const cv = document.createElement('canvas');
    cv.width = cw; cv.height = ch;
    const g = cv.getContext('2d');
    g.fillStyle = 'rgba(232,230,225,0.16)';
    g.strokeStyle = 'rgba(232,230,225,0.42)';
    g.lineWidth = Math.max(1, ppm * 0.14);
    for (const b of boxes) {
      const x = (b.minX - minX) * ppm, y = (b.minZ - minZ) * ppm;
      const w = (b.maxX - b.minX) * ppm, h = (b.maxZ - b.minZ) * ppm;
      g.fillRect(x, y, w, h);
      g.strokeRect(x, y, w, h);
    }
    this._mapStatic = { canvas: cv, ppm, minX, minZ };
  }

  /* ---------------------------------------------------------------- perf */
  _updatePerf(dt, game) {
    if (!this.prefs.perf) return;
    this._perfAcc += dt;
    if (this._perfAcc < 0.25) return;
    this._perfAcc = 0;
    const fps = Math.round(game.adaptive?.fps || 0);
    const scale = (game.adaptive?.scale ?? game.settings.renderScale).toFixed(2);
    const calls = game.renderer?.info?.render?.calls ?? 0;
    const tris = game.renderer?.info?.render?.triangles ?? 0;
    this.el.perf.innerHTML =
      `<b>${fps}</b> FPS<span>SCALE ${scale}</span><span>DRAW ${calls}</span>` +
      `<span>TRI ${(tris / 1000).toFixed(0)}K</span><span>${game.settings.name}</span>`;
  }
}

/* ---------------------------------------------------------------- helpers */

function isAlive(e) {
  if (!e) return false;
  if (e.alive === false || e.dead === true) return false;
  if (typeof e.health === 'number' && e.health <= 0) return false;
  return true;
}

/** Normalises whatever shape Level hands us into flat world-space extents. */
function toBox(col) {
  if (!col) return null;
  const src = col.box || col.aabb || col;
  if (src.min && src.max) {
    return {
      minX: src.min.x, minY: src.min.y ?? 0, minZ: src.min.z,
      maxX: src.max.x, maxY: src.max.y ?? 1, maxZ: src.max.z,
    };
  }
  if (src.isObject3D) {
    const b = new THREE.Box3().setFromObject(src);
    if (!isFinite(b.min.x)) return null;
    return { minX: b.min.x, minY: b.min.y, minZ: b.min.z, maxX: b.max.x, maxY: b.max.y, maxZ: b.max.z };
  }
  const c = src.center || src.position;
  const h = src.halfExtents || src.half || (src.size && { x: src.size.x / 2, y: src.size.y / 2, z: src.size.z / 2 });
  if (c && h) {
    return {
      minX: c.x - h.x, minY: (c.y ?? 0) - (h.y ?? 1), minZ: c.z - h.z,
      maxX: c.x + h.x, maxY: (c.y ?? 0) + (h.y ?? 1), maxZ: c.z + h.z,
    };
  }
  return null;
}

function esc(s) {
  return String(s).replace(/[&<>]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[ch]));
}

function formatPref(key, v) {
  if (key === 'sens') return (+v).toFixed(2) + 'x';
  if (key === 'fov') return Math.round(v) + '°';
  if (key === 'volume') return Math.round(v * 100) + '%';
  return String(v);
}

function loadPrefs() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    return raw ? { ...DEFAULTS, ...JSON.parse(raw) } : { ...DEFAULTS };
  } catch { return { ...DEFAULTS }; }
}

function savePrefs(p) {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(p)); } catch { /* private mode */ }
}

/* -------------------------------------------------------------- fragments */

const DAMAGE_ARC = `<svg viewBox="0 0 100 100" aria-hidden="true">
  <path d="M28 34a30 30 0 0 1 44 0" fill="none" stroke="currentColor" stroke-width="7" stroke-linecap="round"/>
  <path d="M44 22l6-8 6 8z" fill="currentColor"/>
</svg>`;

const MARKUP = `
<div class="hud-grain"></div>
<div class="hud-scan"></div>

<div class="hud-mapwrap"><canvas class="hud-map"></canvas><div class="ring"></div></div>
<div class="hud-feed"></div>

<div class="hud-ammo">
  <div class="nums"><span class="mag">30</span><span class="slash">/</span><span class="res">90</span></div>
  <div class="meta"><span class="wname">—</span><span class="wmode">—</span></div>
</div>

<div class="hud-health">
  <div class="segs"></div>
  <div class="line"><span class="lbl">VITALS</span><span class="hpnum">100</span></div>
</div>

<canvas class="hud-cross"></canvas>
<div class="hud-dmg"></div>
<div class="hud-lowhp"></div>

<div class="hud-ann"><div class="t"></div><div class="s"></div></div>
<div class="hud-banner"></div>
<div class="hud-perf"></div>
<button class="hud-pausebtn" data-act="pause" aria-label="Pause"><i></i><i></i></button>

<div class="hud-screen hud-menu">
  <div class="stack">
    <div class="eyebrow">TACTICAL OPERATIONS</div>
    <h1>BLACKSITE</h1>
    <div class="rule"></div>
    <div class="btns">
      <button class="btn primary" data-act="play"><em></em>PLAY</button>
      <button class="btn" data-act="settings"><em></em>SETTINGS</button>
      <button class="btn" data-act="controls"><em></em>CONTROLS</button>
    </div>
    <div class="foot">WAVE DEFENCE &#183; NO ASSETS &#183; WEBGL</div>
  </div>
</div>

<div class="hud-screen hud-pausescreen">
  <div class="stack">
    <div class="eyebrow">SIMULATION HALTED</div>
    <h2>PAUSED</h2>
    <div class="rule"></div>
    <div class="btns">
      <button class="btn primary" data-act="resume"><em></em>RESUME</button>
      <button class="btn" data-act="settings"><em></em>SETTINGS</button>
      <button class="btn" data-act="quit"><em></em>QUIT</button>
    </div>
  </div>
</div>

<div class="hud-screen hud-death">
  <div class="stack">
    <div class="eyebrow">OPERATOR DOWN</div>
    <h2 class="bad">K.I.A.</h2>
    <div class="rule"></div>
    <div class="stats">
      <div><span>SCORE</span><b class="v-score">0</b></div>
      <div><span>KILLS</span><b class="v-kills">0</b></div>
      <div><span>WAVE</span><b class="v-wave">0</b></div>
    </div>
    <div class="btns">
      <button class="btn primary" data-act="retry"><em></em>RETRY</button>
      <button class="btn" data-act="menu"><em></em>MENU</button>
    </div>
  </div>
</div>

<div class="hud-panel hud-settings">
  <div class="sheet">
    <div class="head"><h3>SETTINGS</h3><button class="x" data-act="close-settings" aria-label="Close">&#10005;</button></div>
    <div class="opts">
      <label class="opt"><span class="k">SENSITIVITY</span><span class="val"></span>
        <input type="range" data-pref="sens" min="0.25" max="3" step="0.05"></label>
      <label class="opt"><span class="k">FIELD OF VIEW</span><span class="val"></span>
        <input type="range" data-pref="fov" min="65" max="110" step="1"></label>
      <label class="opt"><span class="k">MASTER VOLUME</span><span class="val"></span>
        <input type="range" data-pref="volume" min="0" max="1" step="0.02"></label>
      <label class="opt row"><span class="k">INVERT Y</span><input type="checkbox" data-pref="invertY"><i class="sw"></i></label>
      <label class="opt row"><span class="k">MUTE</span><input type="checkbox" data-pref="mute"><i class="sw"></i></label>
      <label class="opt row"><span class="k">PERF OVERLAY</span><input type="checkbox" data-pref="perf"><i class="sw"></i></label>
      <label class="opt row"><span class="k">QUALITY</span>
        <select data-pref="quality">
          <option value="auto">AUTO</option><option value="low">LOW</option>
          <option value="medium">MEDIUM</option><option value="high">HIGH</option>
          <option value="ultra">ULTRA</option>
        </select></label>
      <div class="note">QUALITY CHANGES RESOLUTION IMMEDIATELY; SHADOW AND EFFECT BUDGETS APPLY ON RELOAD.</div>
    </div>
  </div>
</div>

<div class="hud-panel hud-controls">
  <div class="sheet">
    <div class="head"><h3>CONTROLS</h3><button class="x" data-act="close-controls" aria-label="Close">&#10005;</button></div>
    <div class="rows"></div>
    <div class="note">TOUCH: LEFT STICK MOVES, RIGHT SIDE LOOKS, ON-SCREEN KEYS FIRE AND RELOAD.</div>
  </div>
</div>
`;

const CSS = `
#hud{
  --acc:#ff7a1a; --ink:#e8e6e1; --dim:#6d7479; --bad:#ff3b30;
  --bg:rgba(8,9,10,.92);
  --pad-t:max(10px,env(safe-area-inset-top));
  --pad-r:max(12px,env(safe-area-inset-right));
  --pad-b:max(12px,env(safe-area-inset-bottom));
  --pad-l:max(12px,env(safe-area-inset-left));
  position:absolute;inset:0;z-index:500;pointer-events:none;
  font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  color:var(--ink);overflow:hidden;
  -webkit-font-smoothing:antialiased;
}
#hud *{pointer-events:none}
#hud button,#hud input,#hud select,#hud label{pointer-events:auto;touch-action:manipulation}
/* The id in the selector is deliberate: it has to outrank "#hud *" above, or
   an open overlay would let taps fall through to the screen behind it. */
#hud .hud-screen,#hud .hud-panel,#hud .hud-panel .sheet{pointer-events:auto}

/* ---------- ambience: menus only, so gameplay pays nothing ---------- */
.hud-grain,.hud-scan{position:absolute;inset:0;opacity:0;transition:opacity .5s ease}
#hud:not([data-screen=game]) .hud-grain{opacity:.05;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='140' height='140'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3'/></filter><rect width='140' height='140' filter='url(%23n)'/></svg>");
  background-size:140px 140px}
#hud:not([data-screen=game]) .hud-scan{opacity:.5;
  background:repeating-linear-gradient(180deg,rgba(0,0,0,.34) 0 1px,transparent 1px 3px);
  animation:hudScan 8s linear infinite}
@keyframes hudScan{from{background-position-y:0}to{background-position-y:60px}}

/* ---------------------------- minimap ---------------------------- */
.hud-mapwrap{position:absolute;top:var(--pad-t);left:var(--pad-l);
  width:clamp(84px,15vmin,148px);aspect-ratio:1;opacity:0;transition:opacity .3s ease}
.hud-mapwrap .ring{position:absolute;inset:0;border-radius:50%;
  border:1px solid rgba(232,230,225,.22);
  box-shadow:inset 0 0 22px rgba(0,0,0,.7)}
.hud-mapwrap .ring::after{content:"";position:absolute;left:50%;top:-1px;width:1px;height:7px;
  background:var(--acc);transform:translateX(-.5px)}
.hud-map{width:100%;height:100%;border-radius:50%;display:block}

/* ---------------------------- killfeed ---------------------------- */
.hud-feed{position:absolute;top:var(--pad-t);right:var(--pad-r);
  display:flex;flex-direction:column;align-items:flex-end;gap:4px;opacity:0;transition:opacity .3s}
.hud-feed .kf{font-size:clamp(9px,1.5vmin,12px);letter-spacing:.14em;
  background:rgba(8,9,10,.55);padding:4px 8px;border-right:2px solid var(--acc);
  opacity:0;transform:translateX(14px);transition:opacity .18s ease,transform .18s ease;white-space:nowrap}
.hud-feed .kf.in{opacity:1;transform:none}
.hud-feed .kf.out{opacity:0}
.hud-feed .kf b{color:var(--acc);font-weight:700}
.hud-feed .kf i{color:var(--dim);font-style:normal;margin:0 2px}
.hud-feed .kf u{color:var(--bad);text-decoration:none;margin-left:4px}
.hud-feed .kf s{color:var(--ink);text-decoration:none;opacity:.85}

/* ------------------------------ ammo ------------------------------ */
.hud-ammo{position:absolute;right:var(--pad-r);bottom:var(--pad-b);text-align:right;
  opacity:0;transition:opacity .3s ease}
.hud-ammo .nums{display:flex;align-items:baseline;justify-content:flex-end;gap:3px;line-height:.92}
.hud-ammo .mag{font-size:clamp(34px,7vmin,64px);font-weight:800;letter-spacing:.02em;
  text-shadow:0 2px 18px rgba(0,0,0,.8)}
.hud-ammo .slash{font-size:clamp(13px,2.4vmin,22px);color:var(--dim)}
.hud-ammo .res{font-size:clamp(13px,2.4vmin,22px);color:var(--dim)}
.hud-ammo .meta{display:flex;justify-content:flex-end;gap:10px;margin-top:5px;
  font-size:clamp(9px,1.5vmin,12px);letter-spacing:.24em;color:var(--dim);
  border-top:1px solid rgba(232,230,225,.18);padding-top:5px}
.hud-ammo .wname{color:var(--ink)}
.hud-ammo .wmode{color:var(--acc)}
.hud-ammo.low .mag{color:var(--bad)}
.hud-ammo.empty .mag{color:var(--bad);animation:hudBlink .5s steps(2,end) infinite}
@keyframes hudBlink{50%{opacity:.25}}

/* ----------------------------- health ----------------------------- */
.hud-health{position:absolute;left:var(--pad-l);bottom:var(--pad-b);opacity:0;transition:opacity .3s ease}
.hud-health .segs{display:flex;gap:3px}
.hud-health .segs i{display:block;width:clamp(14px,2.6vmin,26px);height:4px;background:var(--ink);
  transition:background .2s ease,opacity .2s ease}
.hud-health .segs i.off{background:rgba(232,230,225,.16)}
.hud-health.hurt .segs i:not(.off){background:#ffb457}
.hud-health.crit .segs i:not(.off){background:var(--bad)}
.hud-health .line{display:flex;align-items:baseline;gap:8px;margin-top:6px;
  font-size:clamp(9px,1.5vmin,12px);letter-spacing:.24em;color:var(--dim)}
.hud-health .hpnum{font-size:clamp(15px,2.8vmin,24px);letter-spacing:.04em;color:var(--ink);font-weight:700}
.hud-health.crit .hpnum{color:var(--bad)}

/* --------------------------- crosshair --------------------------- */
.hud-cross{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);opacity:0}
#hud[data-screen=game] .hud-cross{opacity:1}

/* ---------------- directional damage + low health ---------------- */
.hud-dmg{position:absolute;left:50%;top:50%;width:min(62vmin,520px);aspect-ratio:1;
  transform:translate(-50%,-50%)}
.hud-dmg .di{position:absolute;inset:0;color:var(--bad);filter:drop-shadow(0 0 6px rgba(255,59,48,.55))}
.hud-dmg .di svg{width:100%;height:100%;display:block}
.hud-lowhp{position:absolute;inset:0;opacity:0;
  background:radial-gradient(115% 85% at 50% 50%,transparent 42%,rgba(120,0,0,.5) 100%)}
.hud-lowhp.on{animation:hudBeat 1.15s ease-in-out infinite}
@keyframes hudBeat{0%,100%{opacity:.42}18%{opacity:.9}34%{opacity:.5}52%{opacity:.78}}

/* ------------------------- announcements ------------------------- */
.hud-ann{position:absolute;left:0;right:0;top:22%;text-align:center;opacity:0}
.hud-ann .t{font-size:clamp(22px,6.4vmin,54px);font-weight:800;letter-spacing:.32em;
  text-indent:.32em;text-shadow:0 0 30px rgba(255,122,26,.5)}
.hud-ann .s{margin-top:8px;font-size:clamp(9px,1.7vmin,13px);letter-spacing:.42em;
  text-indent:.42em;color:var(--acc)}
.hud-ann.go{animation:hudAnn 2.9s cubic-bezier(.16,.84,.3,1) forwards}
@keyframes hudAnn{
  0%{opacity:0;transform:translateY(16px) scale(.94);letter-spacing:.6em}
  12%{opacity:1;transform:none}
  74%{opacity:1;transform:none}
  100%{opacity:0;transform:translateY(-10px) scale(1.02)}}

.hud-banner{position:absolute;left:50%;top:calc(var(--pad-t) + 6px);transform:translateX(-50%);
  font-size:clamp(9px,1.6vmin,12px);letter-spacing:.3em;color:var(--acc);
  background:rgba(8,9,10,.7);padding:6px 14px;border:1px solid rgba(255,122,26,.35);
  opacity:0;transition:opacity .25s ease;text-align:center;max-width:70vw}
.hud-banner.on{opacity:1}

/* ---------------------------- perf ---------------------------- */
.hud-perf{position:absolute;left:var(--pad-l);top:calc(var(--pad-t) + clamp(92px,16vmin,158px));
  display:none;gap:10px;font-size:clamp(8px,1.3vmin,11px);letter-spacing:.16em;color:var(--dim);
  background:rgba(8,9,10,.55);padding:4px 8px}
#hud.show-perf .hud-perf{display:flex;flex-wrap:wrap;max-width:44vw}
.hud-perf b{color:var(--acc);font-weight:700}

/* --------------------------- pause button --------------------------- */
.hud-pausebtn{position:absolute;top:var(--pad-t);left:50%;transform:translateX(-50%);
  display:none;width:44px;height:44px;align-items:center;justify-content:center;gap:5px;
  background:rgba(8,9,10,.5);border:1px solid rgba(232,230,225,.2);border-radius:2px}
.hud-pausebtn i{display:block;width:3px;height:14px;background:var(--ink)}
#hud[data-screen=game].touch .hud-pausebtn{display:flex}

/* ------------------------- in-game visibility ------------------------- */
#hud[data-screen=game] .hud-mapwrap,
#hud[data-screen=game] .hud-feed,
#hud[data-screen=game] .hud-ammo,
#hud[data-screen=game] .hud-health,
#hud[data-screen=pause] .hud-mapwrap,
#hud[data-screen=pause] .hud-ammo,
#hud[data-screen=pause] .hud-health{opacity:1}

/* ----------------------------- screens ----------------------------- */
.hud-screen{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
  pointer-events:auto;padding:calc(var(--pad-t) + 8px) calc(var(--pad-r) + 8px)
    calc(var(--pad-b) + 8px) calc(var(--pad-l) + 8px);
  background:radial-gradient(125% 95% at 50% 0%,rgba(20,23,26,.94) 0%,rgba(6,7,8,.96) 60%,rgba(0,0,0,.98) 100%)}
#hud[data-screen=menu] .hud-menu,
#hud[data-screen=pause] .hud-pausescreen,
#hud[data-screen=death] .hud-death{display:flex;animation:hudFade .35s ease both}
@keyframes hudFade{from{opacity:0}to{opacity:1}}
#hud[data-screen=pause] .hud-screen{background:rgba(4,5,6,.72);backdrop-filter:blur(3px)}

.hud-screen .stack{width:min(520px,86vw);display:flex;flex-direction:column;align-items:center;text-align:center}
.hud-screen .eyebrow{font-size:clamp(8px,1.5vmin,11px);letter-spacing:.5em;text-indent:.5em;color:var(--acc)}
.hud-screen h1{margin-top:10px;font-size:clamp(34px,10vmin,88px);font-weight:800;
  letter-spacing:.28em;text-indent:.28em;text-shadow:0 0 40px rgba(255,122,26,.35)}
.hud-screen h2{margin-top:10px;font-size:clamp(26px,7vmin,58px);font-weight:800;
  letter-spacing:.24em;text-indent:.24em}
.hud-screen h2.bad{color:var(--bad);text-shadow:0 0 34px rgba(255,59,48,.35)}
.hud-screen .rule{width:100%;height:1px;margin:18px 0 22px;
  background:linear-gradient(90deg,transparent,rgba(232,230,225,.35),transparent)}
.hud-screen .foot{margin-top:26px;font-size:clamp(8px,1.3vmin,10px);letter-spacing:.34em;color:var(--dim)}

.hud-screen .btns{display:flex;flex-direction:column;gap:10px;width:min(340px,78vw)}
#hud .btn{position:relative;display:flex;align-items:center;min-height:48px;width:100%;
  padding:0 16px 0 22px;background:rgba(232,230,225,.05);border:1px solid rgba(232,230,225,.16);
  color:var(--ink);font:inherit;font-size:clamp(12px,2vmin,15px);font-weight:700;
  letter-spacing:.3em;text-align:left;transition:background .16s ease,border-color .16s ease}
#hud .btn em{position:absolute;left:0;top:0;bottom:0;width:3px;background:transparent;transition:background .16s ease}
#hud .btn.primary{border-color:rgba(255,122,26,.5);background:rgba(255,122,26,.1)}
#hud .btn.primary em{background:var(--acc)}
#hud .btn:active{background:rgba(255,122,26,.24);border-color:var(--acc)}
#hud .btn:active em{background:var(--acc)}
#hud .btn:focus-visible{outline:1px solid var(--acc);outline-offset:2px}

.hud-death .stats{display:flex;gap:clamp(14px,5vw,40px);margin-bottom:26px}
.hud-death .stats div{display:flex;flex-direction:column;gap:6px}
.hud-death .stats span{font-size:clamp(8px,1.4vmin,10px);letter-spacing:.36em;color:var(--dim)}
.hud-death .stats b{font-size:clamp(22px,5.4vmin,42px);font-weight:800;color:var(--acc)}

/* ------------------------------ panels ------------------------------ */
.hud-panel{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
  pointer-events:auto;background:rgba(3,4,5,.86);
  padding:calc(var(--pad-t) + 8px) calc(var(--pad-r) + 8px) calc(var(--pad-b) + 8px) calc(var(--pad-l) + 8px)}
#hud.panel-settings .hud-settings,#hud.panel-controls .hud-controls{display:flex;animation:hudFade .2s ease both}
.hud-panel .sheet{width:min(520px,92vw);max-height:84vh;overflow-y:auto;-webkit-overflow-scrolling:touch;
  background:rgba(10,12,13,.96);border:1px solid rgba(232,230,225,.14);padding:16px}
.hud-panel .head{display:flex;align-items:center;justify-content:space-between;
  border-bottom:1px solid rgba(232,230,225,.14);padding-bottom:12px;margin-bottom:14px}
.hud-panel h3{font-size:clamp(12px,2.2vmin,16px);letter-spacing:.36em;font-weight:700}
.hud-panel .x{width:44px;height:44px;background:none;border:0;color:var(--ink);font:inherit;font-size:16px}
.hud-panel .x:active{color:var(--acc)}
.hud-panel .note{margin-top:16px;font-size:clamp(8px,1.3vmin,10px);line-height:1.9;
  letter-spacing:.16em;color:var(--dim)}

.hud-panel .opt{display:grid;grid-template-columns:1fr auto;gap:6px 12px;align-items:center;
  min-height:48px;padding:8px 0;border-bottom:1px solid rgba(232,230,225,.07)}
.hud-panel .opt .k{font-size:clamp(9px,1.5vmin,11px);letter-spacing:.28em;color:var(--ink)}
.hud-panel .opt .val{font-size:clamp(9px,1.5vmin,11px);letter-spacing:.16em;color:var(--acc)}
.hud-panel .opt input[type=range]{grid-column:1/-1;width:100%;height:28px;appearance:none;background:none}
.hud-panel input[type=range]::-webkit-slider-runnable-track{height:2px;background:rgba(232,230,225,.24)}
.hud-panel input[type=range]::-webkit-slider-thumb{appearance:none;width:20px;height:20px;margin-top:-9px;
  border-radius:50%;background:var(--acc);box-shadow:0 0 10px rgba(255,122,26,.6)}
.hud-panel input[type=range]::-moz-range-track{height:2px;background:rgba(232,230,225,.24)}
.hud-panel input[type=range]::-moz-range-thumb{width:20px;height:20px;border:0;border-radius:50%;background:var(--acc)}
.hud-panel .opt.row{grid-template-columns:1fr auto}
.hud-panel .opt input[type=checkbox]{position:absolute;opacity:0;width:1px;height:1px}
.hud-panel .sw{width:46px;height:24px;border:1px solid rgba(232,230,225,.28);position:relative;display:block}
.hud-panel .sw::after{content:"";position:absolute;top:3px;left:3px;width:16px;height:16px;
  background:rgba(232,230,225,.5);transition:transform .16s ease,background .16s ease}
.hud-panel input[type=checkbox]:checked + .sw{border-color:var(--acc)}
.hud-panel input[type=checkbox]:checked + .sw::after{transform:translateX(22px);background:var(--acc)}
.hud-panel select{min-height:36px;background:rgba(232,230,225,.06);color:var(--ink);
  border:1px solid rgba(232,230,225,.2);font:inherit;font-size:11px;letter-spacing:.2em;padding:0 8px}

.hud-controls .rows{display:grid;grid-template-columns:1fr auto;gap:0 16px}
.hud-controls .row{display:contents}
.hud-controls .row span{font-size:clamp(9px,1.5vmin,11px);letter-spacing:.26em;color:var(--dim);padding:7px 0}
.hud-controls .row b{font-size:clamp(9px,1.5vmin,11px);letter-spacing:.18em;color:var(--ink);
  text-align:right;padding:7px 0}

@media (prefers-reduced-motion:reduce){
  #hud *,#hud *::before,#hud *::after{animation-duration:.01ms !important;animation-iteration-count:1 !important}
}
`;

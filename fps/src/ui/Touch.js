// Touch.js — on-screen controls. This file is the entire mobile game: if it is
// wrong the title is unplayable on a phone, so it owns every iOS quirk too.
//
// Design notes that are not obvious from the code:
//  * All touches are tracked centrally by `identifier` and hit-tested against
//    button rects in JS rather than by putting listeners on each SVG node.
//    Safari's per-element multi-touch dispatch is unreliable once a touch
//    slides off the element that captured it, and the fire-button-drag trick
//    (#3) needs the touch to keep feeding look after it leaves the button.
//  * Listeners live on `document`, not on our own layer, and our layer is
//    pointer-events:none. That way the HUD's real buttons keep working and we
//    never steal a tap meant for a menu — we only accept touches whose target
//    is the canvas or the app root.

import { device } from '../core/Perf.js';

const ACCENT = '#ff7a1a';
const SVGNS = 'http://www.w3.org/2000/svg';
const STORE_KEY = 'blacksite.touch';

const STICK_RADIUS = 62;        // CSS px at scale 1
const STICK_DEAD = 0.08;
const SPRINT_ON = 0.85;         // fraction of the radius that engages sprint
const SPRINT_OFF = 0.72;        // hysteresis, else sprint chatters at the ring
const LOOK_TAU = 0.012;         // ~12ms smoothing: below one frame at 60Hz
const TAP_MS = 220;             // max duration of a "tap to fire"
const TAP_SLOP = 14;            // max travel of a tap, CSS px
const LONG_PRESS_MS = 700;
const IDLE_FADE_AT = 2.5;       // seconds untouched before the UI dims further

// Glyphs are stroked paths in a -10..10 box, scaled to each button's radius.
// Everything is drawn, never loaded — the project ships zero external assets.
const GLYPH = {
  fire:    'M-9,0H-4M4,0H9M0,-9V-4M0,4V9M2.4,0A2.4,2.4 0 1,1 -2.4,0A2.4,2.4 0 1,1 2.4,0',
  ads:     'M8,0A8,8 0 1,1 -8,0A8,8 0 1,1 8,0M-8,0H8M0,-8V8',
  jump:    'M-6,-1L0,-7L6,-1M0,-7V4M-7,8H7',
  crouch:  'M-6,1L0,7L6,1M0,7V-4M-7,-8H7',
  reload:  'M6.5,0A6.5,6.5 0 1,1 -0.5,-6.5M-0.5,-6.5L-4.2,-9.4M-0.5,-6.5L-3.6,-2.6',
  melee:   'M-7,7L1,-1M1,-1L4.5,-8L8,-4.5L1,-1M-7.5,6.5L-4,3',
  grenade: 'M0,8A6.4,6.4 0 1,1 0,-4.8A6.4,6.4 0 1,1 0,8M-2.6,-4.9V-7.6H2.6V-4.9M2.6,-6.6H6.8',
  swap:    'M-8,-3.4H5L1,-7.4M8,3.4H-5L-1,7.4',
  gyro:    'M0,-8A8,8 0 1,1 0,8A8,8 0 1,1 0,-8M-8,0A8,3.4 0 0,0 8,0M-8,0A8,3.4 0 0,1 8,0',
  assist:  'M8,0A8,8 0 1,1 -8,0A8,8 0 1,1 8,0M0,-10V-5M0,5V10M-10,0H-5M5,0H10M1.9,0A1.9,1.9 0 1,1 -1.9,0A1.9,1.9 0 1,1 1.9,0',
  inspect: 'M-9,0C-5,-5.2 5,-5.2 9,0C5,5.2 -5,5.2 -9,0M2.6,0A2.6,2.6 0 1,1 -2.6,0A2.6,2.6 0 1,1 2.6,0',
};

// dx/dy are insets from the anchored corner to the button CENTRE, at scale 1.
// The whole table is multiplied by a viewport-derived scale and then clamped
// on screen, so an iPhone SE in landscape keeps the same reachable shape.
const BUTTONS = [
  { id: 'fire',    anchor: 'br', dx: 84,  dy: 84,  r: 58, glyph: 'fire' },
  { id: 'ads',     anchor: 'br', dx: 206, dy: 92,  r: 33, glyph: 'ads' },
  { id: 'jump',    anchor: 'br', dx: 88,  dy: 200, r: 33, glyph: 'jump' },
  { id: 'crouch',  anchor: 'br', dx: 196, dy: 196, r: 31, glyph: 'crouch' },
  { id: 'reload',  anchor: 'br', dx: 292, dy: 118, r: 30, glyph: 'reload' },
  { id: 'grenade', anchor: 'br', dx: 88,  dy: 300, r: 27, glyph: 'grenade' },
  { id: 'melee',   anchor: 'br', dx: 176, dy: 288, r: 27, glyph: 'melee' },
  { id: 'swap',    anchor: 'br', dx: 282, dy: 216, r: 27, glyph: 'swap' },
  { id: 'gyro',    anchor: 'bl', dx: 46,  dy: 232, r: 21, glyph: 'gyro' },
  { id: 'assist',  anchor: 'bl', dx: 46,  dy: 292, r: 21, glyph: 'assist' },
  { id: 'inspect', anchor: 'bl', dx: 106, dy: 262, r: 21, glyph: 'inspect' },
];

// Momentary actions. Held actions (fire/ads/jump/crouch/sprint) are not here.
const PULSE_ACTION = {
  reload: 'reload', melee: 'melee', grenade: 'grenade', inspect: 'inspect',
};

const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const TAU = Math.PI * 2;
const wrapPi = a => { a %= TAU; if (a > Math.PI) a -= TAU; if (a < -Math.PI) a += TAU; return a; };

function el(tag, attrs) {
  const n = document.createElementNS(SVGNS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}

export class TouchControls {
  constructor(rootEl, game) {
    this.game = game;
    this.input = game.input;
    this.enabled = true;

    this.settings = {
      lookSensX: 1.0,
      lookSensY: 0.86,       // vertical travel is scarcer on a phone; damp it
      adsSens: 0.55,         // separate, lower multiplier while aiming
      assist: 0.75,          // 0 disables aim assist entirely
      autoFire: false,
      tapToFireAnywhere: true,
      adsToggle: false,      // long-press ADS to switch; persisted
      gyro: false,
      haptics: true,
      invertY: false,
      ...this._load(),
    };

    this.w = 1; this.h = 1; this.scale = 1;
    this.insets = { t: 0, r: 0, b: 0, l: 0 };
    this.rect = { left: 0, top: 0 };

    this.touches = new Map();       // identifier -> record
    this.stickId = null;
    this.stick = { ox: 0, oy: 0, x: 0, y: 0, mag: 0, active: false };
    this.sprinting = false;

    this.held = Object.create(null);  // button id -> true while pressed
    this.adsLatched = false;
    this._pulsing = [];
    this._releasing = [];

    this.rawLook = { x: 0, y: 0 };    // pixels accumulated since last update()
    this.smoothLook = { x: 0, y: 0 };
    this.gyroRate = { x: 0, y: 0 };
    this.tapFireT = 0;
    this.idleT = 0;
    this._opacity = -1;
    this._hookedHit = false;
    this._lastVib = 0;

    this._buildDom(rootEl);
    this._bindEvents();
    this.resize(
      Math.max(1, visualViewport?.width || innerWidth),
      Math.max(1, visualViewport?.height || innerHeight));

    // update() only runs while the game is playing, so a pause or a death
    // would otherwise freeze the controls mid-press. This low-rate watchdog
    // clears them and hides the layer whenever we are not in play.
    this._watchdog = setInterval(() => this._syncState(), 240);
  }

  /* --------------------------------------------------------------- public */

  /** Runtime option flip, for a settings menu: setOption('autoFire', true). */
  setOption(key, value) {
    if (!(key in this.settings)) return;
    this.settings[key] = value;
    if (key === 'gyro' && value) this._requestGyro();
    if (key === 'gyro' && !value) this.gyroRate.x = this.gyroRate.y = 0;
    this._save();
    this._paintToggles();
  }

  resize(w, h) {
    this.w = w; this.h = h;
    this.insets = this._readInsets();

    // Small phones get a proportionally smaller cluster so the arc still fits
    // above the home indicator in landscape.
    this.scale = clamp(Math.min(w, h) / 390, 0.76, 1.16);

    this.svg.setAttribute('width', w);
    this.svg.setAttribute('height', h);
    this.svg.setAttribute('viewBox', `0 0 ${w} ${h}`);

    const s = this.scale;
    const padL = this.insets.l, padR = this.insets.r, padB = this.insets.b, padT = this.insets.t;

    for (const b of BUTTONS) {
      const r = b.r * s;
      let cx, cy;
      if (b.anchor === 'br')      { cx = w - padR - b.dx * s; cy = h - padB - b.dy * s; }
      else if (b.anchor === 'bl') { cx = padL + b.dx * s;     cy = h - padB - b.dy * s; }
      else                        { cx = w - padR - b.dx * s; cy = padT + b.dy * s; }

      // Last line of defence: never let a control leave the safe rectangle,
      // whatever the notch and the URL bar have done to the viewport.
      cx = clamp(cx, padL + r + 4, w - padR - r - 4);
      cy = clamp(cy, padT + r + 4, h - padB - r - 4);

      const n = this.nodes[b.id];
      n.hit = { cx, cy, r };
      n.g.setAttribute('transform', `translate(${cx.toFixed(1)},${cy.toFixed(1)})`);
      n.ring.setAttribute('r', r.toFixed(1));
      n.glow.setAttribute('r', r.toFixed(1));
      n.path.setAttribute('transform', `scale(${(r / 17).toFixed(3)})`);
    }

    this.stickR = STICK_RADIUS * s;
    this.stickRing.setAttribute('r', this.stickR.toFixed(1));
    this.stickSprint.setAttribute('r', (this.stickR * SPRINT_ON).toFixed(1));
    this.stickKnob.setAttribute('r', (this.stickR * 0.40).toFixed(1));

    this.rect = this.root.getBoundingClientRect();
  }

  update(dt) {
    if (!this.enabled) return;
    if (this.root.style.display === 'none') this.root.style.display = '';

    const input = this.input;

    // Release last frame's momentary presses first, so the sim saw exactly one
    // frame of them and Input's justPressed/justReleased edges both fire.
    for (let i = 0; i < this._releasing.length; i++) input.set(this._releasing[i], false);
    this._releasing.length = 0;
    for (let i = 0; i < this._pulsing.length; i++) {
      input.set(this._pulsing[i], true);
      this._releasing.push(this._pulsing[i]);
    }
    this._pulsing.length = 0;

    if (!this._hookedHit) this._hookHitFeedback();

    const move = this._stickVector();
    const look = this._lookDelta(dt, move);

    input.applyTouch(move, look);

    input.set('sprint', this.sprinting);
    input.set('jump', !!this.held.jump);
    input.set('crouch', !!this.held.crouch);
    input.set('ads', this.settings.adsToggle ? this.adsLatched : !!this.held.ads);

    this.tapFireT = Math.max(0, this.tapFireT - dt);
    const firing = !!this.held.fire || this.tapFireT > 0 || this._autoFire;
    if (firing && !input.down('fire')) this._vibrate(9);
    input.set('fire', firing);

    this.idleT = this.touches.size ? 0 : this.idleT + dt;
    this._paint(dt);
  }

  dispose() {
    clearInterval(this._watchdog);
    for (const [type, fn, opts] of this._listeners) document.removeEventListener(type, fn, opts);
    removeEventListener('devicemotion', this._onMotion);
    this.root.remove();
    this.styleEl.remove();
  }

  /* ------------------------------------------------------------ movement */

  _stickVector() {
    const st = this.stick;
    if (!st.active) { this.sprinting = false; return { x: 0, y: 0 }; }

    let dx = (st.x - st.ox) / this.stickR;
    let dy = (st.y - st.oy) / this.stickR;

    const len = Math.hypot(dx, dy);
    if (len < STICK_DEAD) { this.sprinting = false; st.mag = 0; return { x: 0, y: 0 }; }

    // Rescale past the dead zone so the very first millimetre of travel is not
    // a jump from 0 to 8% speed.
    const t = Math.min(1, (len - STICK_DEAD) / (1 - STICK_DEAD));
    dx = dx / len * t;
    dy = dy / len * t;
    st.mag = t;

    // Sprint latches past the ring and only drops well inside it, otherwise a
    // thumb resting on the boundary toggles sprint every frame.
    if (len >= SPRINT_ON && dy < -0.25) this.sprinting = true;
    else if (len < SPRINT_OFF) this.sprinting = false;

    return { x: dx, y: -dy };   // screen y is down, input y+ is forward
  }

  /* ---------------------------------------------------------------- look */

  _lookDelta(dt, move) {
    const s = this.settings;
    const target = this._acquireTarget();

    // Sticky aim: cut sensitivity while the crosshair is on or near a body.
    const sticky = (target && target.sticky && s.assist > 0)
      ? 1 - 0.35 * s.assist * target.sticky
      : 1;

    // ADS is a progress value, so the sensitivity change eases in with the
    // scope rather than snapping the instant the button goes down.
    const ads = clamp(this.game.weapons?.adsProgress ?? 0, 0, 1);
    const adsMul = 1 + (s.adsSens - 1) * ads;

    // One-pole smoothing on the accumulated delta. Steady-state gain is 1, so
    // no motion is lost — only a ~12ms ramp, under a single frame at 60Hz.
    const a = 1 - Math.exp(-dt / LOOK_TAU);
    this.smoothLook.x += (this.rawLook.x - this.smoothLook.x) * a;
    this.smoothLook.y += (this.rawLook.y - this.smoothLook.y) * a;
    const rawSpeed = Math.hypot(this.rawLook.x, this.rawLook.y) / Math.max(dt, 1e-4);
    this.rawLook.x = this.rawLook.y = 0;

    const k = adsMul * sticky;
    let lx = this.smoothLook.x * s.lookSensX * k;
    let ly = this.smoothLook.y * s.lookSensY * k * (s.invertY ? -1 : 1);

    // Gyro is additive on top of the thumb, never a replacement for it.
    if (s.gyro) {
      lx += this.gyroRate.x * dt;
      ly += this.gyroRate.y * dt;
    }

    this._autoFire = false;
    if (target && s.assist > 0) {
      const pull = this._rotationalAssist(target, rawSpeed, dt);
      lx += pull.x;
      ly += pull.y;
      if (s.autoFire && target.px < target.fireRadius) this._autoFire = true;
    }

    return { x: lx, y: ly };
  }

  /**
   * Nearest-to-crosshair living enemy, measured in screen pixels so that the
   * assist window tightens automatically as ADS narrows the FOV.
   */
  _acquireTarget() {
    const list = this.game.enemies?.list;
    const cam = this.game.camera;
    if (!list || !list.length || !cam) return null;

    const fov = cam.fov * Math.PI / 180;
    const focal = (this.h * 0.5) / Math.tan(fov * 0.5);
    const stickyR = this.h * 0.085;
    const assistR = this.h * 0.055;
    const fireR = this.h * 0.028;

    const cp = cam.position;
    // Camera forward for YXZ euler order, without allocating a Vector3.
    const cy = cam.rotation.y, cx = cam.rotation.x;
    const cosX = Math.cos(cx);
    const fx = -Math.sin(cy) * cosX, fy = Math.sin(cx), fz = -Math.cos(cy) * cosX;

    let best = null, bestPx = Infinity;
    for (let i = 0; i < list.length; i++) {
      const e = list[i];
      if (!e || e.alive === false || !e.position) continue;
      // The AI publishes visibility when it has already paid for the trace;
      // we never spend a raycast of our own here.
      if (e.visibleToPlayer === false || e.hasLOS === false) continue;

      const ay = typeof e.aimHeight === 'number' ? e.aimHeight
               : (typeof e.height === 'number' ? e.height * 0.62 : 0);
      let dx = e.position.x - cp.x;
      let dy = e.position.y + ay - cp.y;
      let dz = e.position.z - cp.z;
      const dist = Math.hypot(dx, dy, dz);
      if (dist < 0.5 || dist > 70) continue;

      dx /= dist; dy /= dist; dz /= dist;
      const dot = dx * fx + dy * fy + dz * fz;
      if (dot <= 0.2) continue;                     // behind us or far off-axis

      const px = Math.tan(Math.acos(clamp(dot, -1, 1))) * focal;
      if (px > stickyR || px >= bestPx) continue;

      bestPx = px;
      best = { e, px, dx, dy, dz, dist };
    }
    if (!best) return null;

    best.sticky = 1 - clamp(best.px / stickyR, 0, 1);
    best.assist = 1 - clamp(best.px / assistR, 0, 1);
    best.fireRadius = fireR;
    return best;
  }

  /**
   * Gentle rotation toward the target, scaled by how fast the player is
   * already turning — it helps a deliberate flick land, it never drags the
   * camera around while the thumb is still.
   */
  _rotationalAssist(target, rawSpeed, dt) {
    const cam = this.game.camera;
    const s = this.settings;
    if (target.assist <= 0) return { x: 0, y: 0 };

    const targetYaw = Math.atan2(-target.dx, -target.dz);
    const targetPitch = Math.asin(clamp(target.dy, -1, 1));
    const yawErr = wrapPi(targetYaw - cam.rotation.y);
    const pitchErr = targetPitch - cam.rotation.x;

    // 0 when stationary, 1 at a brisk 900 px/s drag. No player input, no pull.
    const intent = clamp(rawSpeed / 900, 0, 1);
    const gain = clamp(2.6 * dt, 0, 0.25) * target.assist * intent * s.assist;

    // Input.applyTouch multiplies by touchSensitivity and subtracts, so the
    // correction is converted back into that pixel space to ride the same
    // path as the thumb — one call, one place where sensitivity is applied.
    const sens = this.input.touchSensitivity || 0.004;
    const yInv = s.invertY || this.input.invertY ? -1 : 1;
    return {
      x: -(yawErr * gain) / sens,
      y: -(pitchErr * gain) / sens * yInv,
    };
  }

  /* ------------------------------------------------------------- gestures */

  _bindEvents() {
    this._listeners = [];
    const on = (type, fn, opts) => {
      document.addEventListener(type, fn, opts);
      this._listeners.push([type, fn, opts]);
    };
    const active = { passive: false, capture: false };

    on('touchstart', e => this._onStart(e), active);
    on('touchmove', e => this._onMove(e), active);
    on('touchend', e => this._onEnd(e, false), active);
    on('touchcancel', e => this._onEnd(e, true), active);

    // iOS raises these for pinch-zoom even with user-scalable=no in some
    // versions, and for the long-press callout over the canvas.
    on('gesturestart', e => e.preventDefault(), active);
    on('gesturechange', e => e.preventDefault(), active);
    on('contextmenu', e => { if (this._ours(e.target)) e.preventDefault(); }, active);
    on('selectstart', e => { if (this._ours(e.target)) e.preventDefault(); }, active);

    this._onMotion = this._onMotion.bind(this);
  }

  /** A touch is ours only if it landed on the game surface, never on the HUD. */
  _ours(node) {
    return node === this.game.canvas || node === this.root ||
           this.root.contains(node) || (node && node.id === 'app');
  }

  _onStart(e) {
    if (!this.enabled || this.game.state !== 'playing') return;

    for (const t of e.changedTouches) {
      if (!this._ours(t.target)) continue;
      e.preventDefault();       // kills double-tap zoom and the scroll chain

      const x = t.clientX - this.rect.left;
      const y = t.clientY - this.rect.top;
      const rec = { id: t.identifier, x, y, sx: x, sy: y, t0: performance.now(), role: 'look', btn: null };

      const btn = this._hitTest(x, y);
      if (btn) {
        rec.btn = btn;
        // The fire button captures the touch but keeps feeding look: this is
        // what makes one-thumbed play possible, and without it the game is
        // simply not playable on a phone.
        rec.role = btn === 'fire' ? 'firedrag' : 'button';
        this._press(btn);
      } else if (x < this.w * 0.5 && this.stickId === null) {
        rec.role = 'stick';
        this.stickId = t.identifier;
        this.stick.ox = this.stick.x = x;
        this.stick.oy = this.stick.y = y;
        this.stick.active = true;
      }

      this.touches.set(t.identifier, rec);
    }
  }

  _onMove(e) {
    if (!this.enabled) return;
    let touched = false;

    for (const t of e.changedTouches) {
      const rec = this.touches.get(t.identifier);
      if (!rec) continue;
      touched = true;

      const x = t.clientX - this.rect.left;
      const y = t.clientY - this.rect.top;
      const dx = x - rec.x, dy = y - rec.y;
      rec.x = x; rec.y = y;

      if (rec.role === 'stick') {
        // Radial clamp against the origin, so the knob follows the thumb but
        // never reports more than full deflection.
        const ox = x - this.stick.ox, oy = y - this.stick.oy;
        const len = Math.hypot(ox, oy);
        if (len > this.stickR) {
          this.stick.x = this.stick.ox + ox / len * this.stickR;
          this.stick.y = this.stick.oy + oy / len * this.stickR;
        } else {
          this.stick.x = x; this.stick.y = y;
        }
      } else if (rec.role === 'look' || rec.role === 'firedrag') {
        // Accumulate: touchmove can fire several times per frame and each one
        // carries motion we must not drop.
        this.rawLook.x += dx;
        this.rawLook.y += dy;
      }
    }

    // Any touch we own must swallow the default, or Safari rubber-bands the
    // page and the left-edge drag triggers swipe-to-go-back mid-firefight.
    if (touched && e.cancelable) e.preventDefault();
  }

  _onEnd(e, cancelled) {
    let touched = false;
    for (const t of e.changedTouches) {
      const rec = this.touches.get(t.identifier);
      if (!rec) continue;
      this.touches.delete(t.identifier);
      touched = true;

      if (rec.role === 'stick') {
        this.stickId = null;
        this.stick.active = false;
        this.stick.mag = 0;
        this.sprinting = false;
      } else if (rec.btn) {
        this._release(rec, cancelled);
      } else if (rec.role === 'look' && !cancelled &&
                 this.settings.tapToFireAnywhere && rec.sx > this.w * 0.5) {
        const dur = performance.now() - rec.t0;
        const travel = Math.hypot(rec.x - rec.sx, rec.y - rec.sy);
        // A quick tap that did not drag is a shot; a drag is purely aiming.
        if (dur < TAP_MS && travel < TAP_SLOP) this.tapFireT = 0.09;
      }
      if (rec.role === 'firedrag') this.held.fire = false;
    }
    // Only swallow ends for touches we actually owned — a HUD button's tap
    // must still turn into a click.
    if (touched && e.cancelable) e.preventDefault();
  }

  _hitTest(x, y) {
    let best = null, bestD = Infinity;
    for (const b of BUTTONS) {
      const n = this.nodes[b.id];
      if (n.hidden) continue;
      const d = Math.hypot(x - n.hit.cx, y - n.hit.cy);
      // A little slop: thumbs are wide and land short of where you think.
      if (d < n.hit.r * 1.18 && d < bestD) { bestD = d; best = b.id; }
    }
    return best;
  }

  _press(id) {
    this.held[id] = true;
    this._vibrate(id === 'fire' ? 9 : 6);

    if (id in PULSE_ACTION) this._pulsing.push(PULSE_ACTION[id]);
    else if (id === 'swap') this._pulsing.push(this._nextSlot());
    else if (id === 'ads' && this.settings.adsToggle) this.adsLatched = !this.adsLatched;
  }

  _release(rec, cancelled) {
    const id = rec.btn;
    this.held[id] = false;
    if (cancelled) return;

    const dur = performance.now() - rec.t0;
    const travel = Math.hypot(rec.x - rec.sx, rec.y - rec.sy);
    const longPress = dur > LONG_PRESS_MS && travel < TAP_SLOP;

    if (id === 'ads' && longPress) {
      // Hold-vs-toggle is a matter of taste and lives on the button itself so
      // it can be changed mid-match without opening a menu.
      this.settings.adsToggle = !this.settings.adsToggle;
      this.adsLatched = false;
      this._vibrate([12, 40, 12]);
      this._save();
      this._paintToggles();
    } else if (id === 'assist' && !longPress) {
      this.settings.assist = this.settings.assist > 0 ? 0 : 0.75;
      this._vibrate(14);
      this._save();
      this._paintToggles();
    } else if (id === 'gyro' && !longPress) {
      this.setOption('gyro', !this.settings.gyro);
      this._vibrate(14);
    }
  }

  _nextSlot() {
    this._slot = ((this._slot ?? 0) + 1) % 3;
    return 'slot' + (this._slot + 1);
  }

  /** Drops every held control. iOS fires touchcancel on an incoming call or
   *  notification banner; without this the player sprints forever. */
  _clearAll() {
    this.touches.clear();
    this.stickId = null;
    this.stick.active = false;
    this.stick.mag = 0;
    this.sprinting = false;
    this.tapFireT = 0;
    this._autoFire = false;
    this.rawLook.x = this.rawLook.y = 0;
    this.smoothLook.x = this.smoothLook.y = 0;
    for (const k in this.held) this.held[k] = false;
  }

  _syncState() {
    const playing = this.enabled && this.game.state === 'playing';
    if (!playing) {
      if (this.touches.size || this.sprinting) this._clearAll();
      this.root.style.display = 'none';
    } else if (this.root.style.display === 'none') {
      this.root.style.display = '';
    }
    // The viewport moves under us when the iOS URL bar collapses; the cached
    // rect is what every hit test is measured against.
    this.rect = this.root.getBoundingClientRect();
  }

  /* -------------------------------------------------------------- haptics */

  _vibrate(pattern) {
    if (!this.settings.haptics) return;
    // iOS Safari has never shipped the Vibration API. Guarded, never assumed.
    if (typeof navigator.vibrate !== 'function') return;
    const now = performance.now();
    if (now - this._lastVib < 28) return;
    this._lastVib = now;
    try { navigator.vibrate(pattern); } catch { /* some UAs throw when denied */ }
  }

  /** main.js assigns weapons.onHit after we are constructed, so the wrap has
   *  to happen on the first frame or we would be overwritten. */
  _hookHitFeedback() {
    const w = this.game.weapons;
    if (!w) return;
    this._hookedHit = true;
    const prev = w.onHit;
    w.onHit = (hit) => {
      prev?.(hit);
      this._vibrate(hit?.killed ? [10, 26, 22] : (hit?.headshot ? 22 : 14));
    };
  }

  /* ----------------------------------------------------------------- gyro */

  _requestGyro() {
    const ask = (Ctor) => {
      if (Ctor && typeof Ctor.requestPermission === 'function') {
        // Must be called synchronously from the user gesture that got us here.
        return Ctor.requestPermission().then(r => r === 'granted').catch(() => false);
      }
      return Promise.resolve(true);
    };
    Promise.all([
      ask(window.DeviceMotionEvent),
      ask(window.DeviceOrientationEvent),
    ]).then(([motion]) => {
      if (!motion) { this.settings.gyro = false; this._paintToggles(); return; }
      addEventListener('devicemotion', this._onMotion);
    }).catch(() => { this.settings.gyro = false; this._paintToggles(); });
  }

  _onMotion(e) {
    const rr = e.rotationRate;
    if (!rr || !this.settings.gyro) return;

    // rotationRate is deg/s about the device axes: beta=x, gamma=y, alpha=z.
    // Which of those is "turn left" depends on how the phone is held, so the
    // screen orientation angle picks the mapping.
    const angle = (screen.orientation?.angle ?? window.orientation ?? 0) | 0;
    const beta = rr.beta || 0, gamma = rr.gamma || 0;
    let yaw, pitch;
    if (angle === 90)       { yaw = -beta;  pitch = -gamma; }
    else if (angle === 270 || angle === -90) { yaw = beta;  pitch = gamma; }
    else if (angle === 180) { yaw = -gamma; pitch = beta; }
    else                    { yaw = gamma;  pitch = -beta; }

    // Fed as pixels-per-second so it rides the same sensitivity as the thumb.
    const k = 4.2;
    this.gyroRate.x = clamp(yaw, -400, 400) * k;
    this.gyroRate.y = clamp(pitch, -400, 400) * k * (this.settings.invertY ? -1 : 1);
  }

  /* ------------------------------------------------------------ rendering */

  _buildDom(rootEl) {
    this.styleEl = document.createElement('style');
    this.styleEl.textContent = `
      #tcx{position:absolute;inset:0;z-index:35;pointer-events:none;
        touch-action:none;-webkit-user-select:none;user-select:none;
        -webkit-touch-callout:none;-webkit-tap-highlight-color:transparent;
        opacity:.35;transition:opacity .45s ease;will-change:opacity}
      #tcx svg{position:absolute;inset:0;display:block;overflow:visible}
      #tcx .ring{fill:rgba(8,10,12,.22);stroke:${ACCENT};stroke-width:1.25;vector-effect:non-scaling-stroke}
      #tcx .glyph{fill:none;stroke:#ffd9b5;stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}
      #tcx .glow{fill:${ACCENT};opacity:0;transition:opacity .12s linear}
      #tcx .on .ring{stroke-width:2.2;fill:rgba(255,122,26,.16)}
      #tcx .on .glow{opacity:.20}
      #tcx .off .ring{stroke:#6d7479}
      #tcx .off .glyph{stroke:#6d7479}
      #tcx .big .glyph{stroke-width:1.9}
      #tcx .latch .ring{stroke-dasharray:2 4}
      #tcx .stick{opacity:0;transition:opacity .16s ease}
      #tcx .stick.live{opacity:1}
      #tcx .sr{fill:none;stroke:${ACCENT};stroke-width:1;stroke-dasharray:3 7;opacity:.5}
      #tcx .knob{fill:rgba(255,122,26,.18);stroke:${ACCENT};stroke-width:1.5}
      #tcx .base{fill:rgba(8,10,12,.18);stroke:#9aa1a6;stroke-width:1.1}
    `;
    document.head.appendChild(this.styleEl);

    this.root = document.createElement('div');
    this.root.id = 'tcx';

    // A probe whose padding resolves the safe-area insets; reading them from
    // JS is otherwise impossible and every control depends on them.
    this._probe = document.createElement('div');
    this._probe.style.cssText =
      'position:absolute;left:0;top:0;width:0;height:0;visibility:hidden;' +
      'padding:env(safe-area-inset-top) env(safe-area-inset-right) ' +
      'env(safe-area-inset-bottom) env(safe-area-inset-left)';
    this.root.appendChild(this._probe);

    this.svg = el('svg', { xmlns: SVGNS, 'aria-hidden': 'true' });
    this.root.appendChild(this.svg);

    const sg = el('g', { class: 'stick' });
    this.stickG = sg;
    this.stickRing = el('circle', { class: 'base', cx: 0, cy: 0, r: 60 });
    this.stickSprint = el('circle', { class: 'sr', cx: 0, cy: 0, r: 50 });
    this.stickKnob = el('circle', { class: 'knob', cx: 0, cy: 0, r: 24 });
    sg.appendChild(this.stickRing);
    sg.appendChild(this.stickSprint);
    sg.appendChild(this.stickKnob);
    this.svg.appendChild(sg);

    this.nodes = Object.create(null);
    for (const b of BUTTONS) {
      const g = el('g', { class: b.id === 'fire' ? 'btn big' : 'btn' });
      const glow = el('circle', { class: 'glow', cx: 0, cy: 0, r: b.r });
      const ring = el('circle', { class: 'ring', cx: 0, cy: 0, r: b.r });
      const path = el('path', { class: 'glyph', d: GLYPH[b.glyph] });
      g.appendChild(glow); g.appendChild(ring); g.appendChild(path);
      this.svg.appendChild(g);
      this.nodes[b.id] = { g, ring, glow, path, hit: { cx: 0, cy: 0, r: b.r }, hidden: false };
    }

    rootEl.appendChild(this.root);
    this._paintToggles();
  }

  _readInsets() {
    const s = getComputedStyle(this._probe);
    const f = v => parseFloat(v) || 0;
    // A floor keeps controls off the rounded corners on devices that report
    // no insets at all, which is most Android phones.
    return {
      t: Math.max(f(s.paddingTop), 12),
      r: Math.max(f(s.paddingRight), 14),
      b: Math.max(f(s.paddingBottom), 14),
      l: Math.max(f(s.paddingLeft), 14),
    };
  }

  _paintToggles() {
    const set = (id, on) => this.nodes[id]?.g.classList.toggle('off', !on);
    set('gyro', this.settings.gyro);
    set('assist', this.settings.assist > 0);
    this.nodes.ads?.g.classList.toggle('latch', this.settings.adsToggle);
  }

  _paint(dt) {
    const st = this.stick;
    this.stickG.classList.toggle('live', st.active);
    if (st.active) {
      this.stickG.setAttribute('transform', `translate(${st.ox.toFixed(1)},${st.oy.toFixed(1)})`);
      this.stickKnob.setAttribute('cx', (st.x - st.ox).toFixed(1));
      this.stickKnob.setAttribute('cy', (st.y - st.oy).toFixed(1));
      this.stickSprint.setAttribute('opacity', this.sprinting ? 0.95 : 0.5);
    }

    for (const b of BUTTONS) {
      const n = this.nodes[b.id];
      const on = b.id === 'ads' && this.settings.adsToggle ? this.adsLatched : !!this.held[b.id];
      if (n.on !== on) { n.on = on; n.g.classList.toggle('on', on); }
    }

    // The reload button pulses when the magazine is low so a player who is not
    // watching the ammo counter still gets the message.
    const low = this._magLow();
    const rl = this.nodes.reload;
    if (low > 0) {
      this._pulseT = (this._pulseT || 0) + dt * (low > 1 ? 9 : 5.5);
      rl.glow.setAttribute('opacity', (0.12 + 0.26 * (0.5 + 0.5 * Math.sin(this._pulseT))).toFixed(3));
    } else if (rl.glow.getAttribute('opacity') !== '0') {
      rl.glow.setAttribute('opacity', 0);
    }

    const o = this.idleT > IDLE_FADE_AT ? 0.17 : 0.35;
    if (o !== this._opacity) { this._opacity = o; this.root.style.opacity = o; }
  }

  /** 0 = fine, 1 = low, 2 = empty. Weapons publishes its ammo under one of a
   *  few names depending on the active weapon shape; all reads are optional. */
  _magLow() {
    const w = this.game.weapons;
    if (!w) return 0;
    const g = w.current || w.active || w.weapon || w;
    const mag = [g.ammo, g.magAmmo, g.rounds, g.mag].find(v => typeof v === 'number');
    if (typeof mag !== 'number') return 0;
    if (mag <= 0) return 2;
    const cap = [g.magSize, g.magCapacity, g.clipSize, g.capacity].find(v => typeof v === 'number' && v > 0);
    if (!cap) return 0;
    return mag / cap <= 0.3 ? 1 : 0;
  }

  /* ------------------------------------------------------------ settings */

  _load() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; }
    catch { return {}; }   // private browsing on iOS throws on localStorage
  }

  _save() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(this.settings)); }
    catch { /* not worth failing a match over */ }
  }
}

// Exported so a settings screen can label the scheme without importing device.
export const touchAvailable = device.touch;

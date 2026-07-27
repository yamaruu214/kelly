// Input.js — one input state, three sources (keyboard+mouse, touch, gamepad).
//
// Gameplay code never asks "are we on mobile"; it reads `input.move`,
// `input.look`, `input.fire`. Everything device-specific is resolved here.

const KEYMAP = {
  KeyW: 'fwd', ArrowUp: 'fwd',
  KeyS: 'back', ArrowDown: 'back',
  KeyA: 'left', ArrowLeft: 'left',
  KeyD: 'right', ArrowRight: 'right',
  Space: 'jump',
  ShiftLeft: 'sprint', ShiftRight: 'sprint',
  ControlLeft: 'crouch', KeyC: 'crouch',
  KeyR: 'reload',
  KeyF: 'melee',
  KeyG: 'grenade',
  KeyQ: 'lean_l', KeyE: 'lean_r',
  KeyV: 'inspect',
  Tab: 'scores',
  Escape: 'pause',
  Digit1: 'slot1', Digit2: 'slot2', Digit3: 'slot3',
};

export class Input {
  constructor(domElement) {
    this.el = domElement;

    this.move = { x: 0, y: 0 };      // -1..1, y+ is forward
    this.look = { x: 0, y: 0 };      // radians consumed per frame
    this.buttons = Object.create(null);
    this.pressed = Object.create(null);   // edge-triggered, cleared each frame
    this.released = Object.create(null);

    this.pointerLocked = false;
    this.sensitivity = 0.0022;
    this.touchSensitivity = 0.0040;
    this.invertY = false;
    this.adsSensScale = 0.62;        // slower look while aiming, as expected

    this._onPointerLockChange = this._onPointerLockChange.bind(this);
    this._bindKeyboard();
    this._bindMouse();
    this._bindGamepad();
  }

  /* ------------------------------------------------------------ queries */
  down(name) { return !!this.buttons[name]; }
  justPressed(name) { return !!this.pressed[name]; }
  justReleased(name) { return !!this.released[name]; }

  set(name, value) {
    const was = !!this.buttons[name];
    const now = !!value;
    if (now && !was) this.pressed[name] = true;
    if (!now && was) this.released[name] = true;
    this.buttons[name] = now;
  }

  /* ------------------------------------------------------------ sources */
  _bindKeyboard() {
    addEventListener('keydown', e => {
      const a = KEYMAP[e.code];
      if (!a) return;
      if (e.code === 'Tab' || e.code === 'Space') e.preventDefault();
      if (e.repeat) return;
      this.set(a, true);
    });
    addEventListener('keyup', e => {
      const a = KEYMAP[e.code];
      if (a) this.set(a, false);
    });
    // A lost focus that leaves keys stuck down is the classic alt-tab bug.
    addEventListener('blur', () => this.releaseAll());
  }

  _bindMouse() {
    document.addEventListener('pointerlockchange', this._onPointerLockChange);
    this.el.addEventListener('mousedown', e => {
      if (!this.pointerLocked) return;
      if (e.button === 0) this.set('fire', true);
      if (e.button === 2) this.set('ads', true);
      if (e.button === 1) { e.preventDefault(); this.set('melee', true); }
    });
    addEventListener('mouseup', e => {
      if (e.button === 0) this.set('fire', false);
      if (e.button === 2) this.set('ads', false);
      if (e.button === 1) this.set('melee', false);
    });
    this.el.addEventListener('contextmenu', e => e.preventDefault());
    addEventListener('mousemove', e => {
      if (!this.pointerLocked) return;
      // movementX/Y can spike enormously on a fast flick or a driver hiccup;
      // clamping stops a single event from spinning the camera wildly.
      const dx = Math.max(-180, Math.min(180, e.movementX || 0));
      const dy = Math.max(-180, Math.min(180, e.movementY || 0));
      this.look.x -= dx * this.sensitivity;
      this.look.y -= dy * this.sensitivity * (this.invertY ? -1 : 1);
    });
    addEventListener('wheel', e => { this._wheel = (this._wheel || 0) + Math.sign(e.deltaY); }, { passive: true });
  }

  _bindGamepad() {
    this.gamepadIndex = null;
    addEventListener('gamepadconnected', e => { this.gamepadIndex = e.gamepad.index; });
    addEventListener('gamepaddisconnected', () => { this.gamepadIndex = null; });
  }

  _pollGamepad(dt) {
    if (this.gamepadIndex === null || !navigator.getGamepads) return;
    const gp = navigator.getGamepads()[this.gamepadIndex];
    if (!gp) return;
    const dz = v => Math.abs(v) < 0.16 ? 0 : (v - Math.sign(v) * 0.16) / 0.84;

    this.move.x += dz(gp.axes[0] || 0);
    this.move.y -= dz(gp.axes[1] || 0);
    const rx = dz(gp.axes[2] || 0), ry = dz(gp.axes[3] || 0);
    // Cubic response on the look stick: fine control near centre, fast at edge.
    this.look.x -= rx * Math.abs(rx) * 3.2 * dt;
    this.look.y -= ry * Math.abs(ry) * 2.4 * dt * (this.invertY ? -1 : 1);

    const b = gp.buttons;
    this.set('fire',   b[7]?.value > 0.35);
    this.set('ads',    b[6]?.value > 0.35);
    this.set('jump',   b[0]?.pressed);
    this.set('crouch', b[1]?.pressed);
    this.set('reload', b[2]?.pressed);
    this.set('sprint', b[10]?.pressed);
    this.set('melee',  b[11]?.pressed);
    this.set('grenade', b[5]?.pressed);
  }

  _onPointerLockChange() {
    this.pointerLocked = document.pointerLockElement === this.el;
    if (!this.pointerLocked) {
      this.set('fire', false);
      this.set('ads', false);
      this.onPointerLockLost?.();
    }
  }

  requestPointerLock() {
    if (this.el.requestPointerLock) {
      const p = this.el.requestPointerLock({ unadjustedMovement: true });
      // Chrome returns a promise; Safari does not. Swallow the rejection that
      // fires when unadjustedMovement is unsupported and retry plainly.
      if (p && p.catch) p.catch(() => this.el.requestPointerLock());
    }
  }

  releaseAll() {
    for (const k in this.buttons) this.set(k, false);
    this.move.x = this.move.y = 0;
  }

  /** Called by the touch layer each frame before update(). */
  applyTouch(moveVec, lookDelta) {
    this.move.x += moveVec.x;
    this.move.y += moveVec.y;
    this.look.x -= lookDelta.x * this.touchSensitivity;
    this.look.y -= lookDelta.y * this.touchSensitivity * (this.invertY ? -1 : 1);
  }

  /** Zeroes accumulators. Call at the very end of a frame. */
  endFrame() {
    this.look.x = 0; this.look.y = 0;
    this.move.x = 0; this.move.y = 0;
    for (const k in this.pressed) delete this.pressed[k];
    for (const k in this.released) delete this.released[k];
    this._wheel = 0;
  }

  beginFrame(dt) {
    // Keyboard contributes to the same move vector the sticks write into.
    if (this.down('fwd')) this.move.y += 1;
    if (this.down('back')) this.move.y -= 1;
    if (this.down('right')) this.move.x += 1;
    if (this.down('left')) this.move.x -= 1;
    this._pollGamepad(dt);

    const len = Math.hypot(this.move.x, this.move.y);
    if (len > 1) { this.move.x /= len; this.move.y /= len; }
  }
}

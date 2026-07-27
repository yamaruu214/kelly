// Audio.js — every sound in the game is synthesized at runtime. No assets, no
// network, no decode step: the whole bank is oscillators, four white-noise
// buffers built once at init, and one procedurally generated impulse response.
//
// Graph:
//   voice -> [lowpass(air) -> panner -> delay(travel time)] -> dryBus ---+
//                                    \-> send -> convolver -> return ----+
//   ambience -----------------------------------------------------------+
//                                                                        v
//                            duck -> compressor -> limiter -> master -> out
//
// The duck node exists so an explosion can sidechain the entire mix; the
// limiter is what keeps eight enemies firing at once from clipping.

import { device } from './Perf.js';

const SPEED_OF_SOUND = 343;          // m/s — drives the travel delay in playAt
const NOISE_VARIANTS = 4;            // enough seeds that a magazine never loops
const NOISE_SECONDS = 1.5;

const rand = (a, b) => a + Math.random() * (b - a);
const clamp = (x, a, b) => x < a ? a : x > b ? b : x;
// ±4% per shot. Without this every round out of a magazine is bit-identical,
// which is the single loudest tell of synthesized weapon audio.
const jitter = (spread) => 1 + (Math.random() * 2 - 1) * spread;

/* Per-sound budget: dry length (the reverb tail outlives it inside the
   convolver), level, reverb send, and steal priority. */
const SPEC = {
  rifle:            { dur: 0.42, amp: 0.80, wet: 0.42, prio: 6 },
  smg:              { dur: 0.30, amp: 0.62, wet: 0.34, prio: 6 },
  sniper:           { dur: 0.70, amp: 1.00, wet: 0.60, prio: 8 },
  enemyFire:        { dur: 0.40, amp: 0.55, wet: 0.50, prio: 5 },
  dryfire:          { dur: 0.14, amp: 0.40, wet: 0.10, prio: 4 },
  reload_magout:    { dur: 0.26, amp: 0.42, wet: 0.14, prio: 4 },
  reload_magin:     { dur: 0.30, amp: 0.48, wet: 0.14, prio: 4 },
  reload_charge:    { dur: 0.34, amp: 0.46, wet: 0.16, prio: 4 },
  shellDrop:        { dur: 0.55, amp: 0.22, wet: 0.30, prio: 1 },
  impact_concrete:  { dur: 0.30, amp: 0.45, wet: 0.30, prio: 3 },
  impact_metal:     { dur: 0.45, amp: 0.45, wet: 0.32, prio: 3 },
  impact_wood:      { dur: 0.28, amp: 0.42, wet: 0.24, prio: 3 },
  impact_sand:      { dur: 0.26, amp: 0.34, wet: 0.16, prio: 2 },
  impact_flesh:     { dur: 0.26, amp: 0.50, wet: 0.14, prio: 4 },
  ricochet:         { dur: 0.45, amp: 0.30, wet: 0.55, prio: 2 },
  explosion:        { dur: 2.40, amp: 1.00, wet: 0.85, prio: 9 },
  grenadeBounce:    { dur: 0.24, amp: 0.34, wet: 0.28, prio: 2 },
  grenadePin:       { dur: 0.20, amp: 0.34, wet: 0.16, prio: 4 },
  melee:            { dur: 0.36, amp: 0.55, wet: 0.22, prio: 5 },
  hurt:             { dur: 0.55, amp: 0.60, wet: 0.18, prio: 7 },
  death:            { dur: 1.60, amp: 0.85, wet: 0.55, prio: 9 },
  enemyDeath:       { dur: 0.90, amp: 0.55, wet: 0.40, prio: 5 },
  footstep_sand:    { dur: 0.20, amp: 0.24, wet: 0.10, prio: 1 },
  footstep_concrete:{ dur: 0.16, amp: 0.26, wet: 0.14, prio: 1 },
  footstep_metal:   { dur: 0.34, amp: 0.24, wet: 0.18, prio: 1 },
  jump:             { dur: 0.26, amp: 0.28, wet: 0.10, prio: 2 },
  land:             { dur: 0.40, amp: 0.40, wet: 0.14, prio: 3 },
  hitmarker:        { dur: 0.10, amp: 0.45, wet: 0.00, prio: 8 },
  headshot:         { dur: 0.16, amp: 0.50, wet: 0.00, prio: 8 },
  killconfirm:      { dur: 0.40, amp: 0.40, wet: 0.10, prio: 8 },
  uiClick:          { dur: 0.09, amp: 0.30, wet: 0.00, prio: 7 },
  uiHover:          { dur: 0.06, amp: 0.14, wet: 0.00, prio: 1 },
  waveStart:        { dur: 1.80, amp: 0.70, wet: 0.45, prio: 9 },
  lowAmmo:          { dur: 0.22, amp: 0.30, wet: 0.00, prio: 6 },
  lowHealth:        { dur: 0.80, amp: 0.40, wet: 0.10, prio: 6 },
  amb_gunfire:      { dur: 1.40, amp: 0.16, wet: 0.90, prio: 1 },
  amb_dog:          { dur: 1.10, amp: 0.10, wet: 0.75, prio: 1 },
  amb_gust:         { dur: 3.20, amp: 0.14, wet: 0.40, prio: 1 },
  _fallback:        { dur: 0.10, amp: 0.25, wet: 0.00, prio: 1 },
};

export class AudioEngine {
  constructor() {
    this.ctx = null;                 // created on first gesture, never at load
    this.masterVolume = 0.9;
    this.muted = false;
    this.enabled = true;
    // iOS starts dropping voices and crackling well before 32; the cap is the
    // difference between a firefight and a stutter on an iPhone.
    this.maxVoices = device.mobile ? 24 : 40;

    this.voices = [];
    this._noiseBufs = [];
    this._pinkBuf = null;
    this._amb = null;
    this._ambTimer = 0;
    this._warned = new Set();
    this._listener = { x: 0, y: 1.7, z: 0 };
    this._hrtf = !(device.mobile && device.cores < 6);
    this._unlockHandler = null;

    this._armUnlock();
  }

  /* ------------------------------------------------------------ lifecycle */

  /**
   * Safe to call from anywhere, but only does something useful inside a real
   * user gesture — that is the only moment Safari will let a context start.
   */
  resume() {
    if (!this.enabled) return;
    if (!this.ctx && !this._create()) return;

    const ctx = this.ctx;
    if (ctx.state !== 'running') {
      // Safari needs an actual scheduled source inside the gesture; resume()
      // alone leaves the graph silent until the next unrelated tap.
      this._silentPing();
      ctx.resume().catch(() => this._armUnlock());
    }
  }

  _create() {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    if (!Ctor) { this.enabled = false; return false; }
    try {
      // 'interactive' asks for the smallest buffer the device will give us;
      // anything larger and the muzzle flash lands before the gunshot.
      this.ctx = new Ctor({ latencyHint: 'interactive' });
    } catch (e) {
      console.warn('[audio] AudioContext unavailable:', e);
      this.enabled = false;
      return false;
    }

    this.ctx.addEventListener?.('statechange', () => this._onStateChange());
    this._buildBuffers();
    this._buildGraph();
    return true;
  }

  _onStateChange() {
    const s = this.ctx.state;
    // 'interrupted' is Safari-only: a call, Siri, or the ringer switch parks
    // the context and it never comes back on its own.
    if (s === 'interrupted' || s === 'suspended') {
      this.ctx.resume().catch(() => {});
      this._armUnlock();
    } else if (s === 'running') {
      this._disarmUnlock();
      if (this._amb) this._scheduleAmbEvent();
    }
  }

  _armUnlock() {
    if (this._unlockHandler) return;
    const h = () => {
      this.resume();
      if (this.ctx && this.ctx.state === 'running') this._disarmUnlock();
    };
    this._unlockHandler = h;
    for (const ev of ['touchend', 'pointerdown', 'mousedown', 'keydown']) {
      addEventListener(ev, h, { passive: true, capture: true });
    }
  }

  _disarmUnlock() {
    const h = this._unlockHandler;
    if (!h) return;
    this._unlockHandler = null;
    for (const ev of ['touchend', 'pointerdown', 'mousedown', 'keydown']) {
      removeEventListener(ev, h, { capture: true });
    }
  }

  _silentPing() {
    try {
      const src = this.ctx.createBufferSource();
      src.buffer = this.ctx.createBuffer(1, 1, this.ctx.sampleRate);
      src.connect(this.ctx.destination);
      src.start(0);
    } catch { /* context is mid-teardown; the next gesture retries */ }
  }

  setMasterVolume(v) {
    this.masterVolume = clamp(+v || 0, 0, 1);
    this._applyGain();
  }

  setMuted(b) {
    this.muted = !!b;
    this._applyGain();
  }

  _applyGain() {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const g = this.master.gain;
    g.cancelScheduledValues(t);
    g.setValueAtTime(g.value, t);
    g.linearRampToValueAtTime(this.muted ? 0 : this.masterVolume, t + 0.03);
  }

  /* ---------------------------------------------------------- master bus */

  _buildGraph() {
    const ctx = this.ctx;

    this.master = ctx.createGain();
    this.master.gain.value = this.muted ? 0 : this.masterVolume;

    const limiter = ctx.createDynamicsCompressor();
    limiter.threshold.value = -2;
    limiter.knee.value = 0;
    limiter.ratio.value = 20;
    limiter.attack.value = 0.001;
    limiter.release.value = 0.06;

    const glue = ctx.createDynamicsCompressor();
    glue.threshold.value = -18;
    glue.knee.value = 14;
    glue.ratio.value = 4;
    glue.attack.value = 0.004;
    glue.release.value = 0.20;

    this.duck = ctx.createGain();     // explosion sidechain target
    this.duck.gain.value = 1;

    this.bus = ctx.createGain();      // dry sum
    this.bus.gain.value = 1;

    this.wet = ctx.createGain();      // reverb send sum
    this.wet.gain.value = 1;

    const conv = ctx.createConvolver();
    conv.normalize = true;
    conv.buffer = this._buildImpulse(1.4);

    // The tail keeps its top end down: a street reflects, it does not sparkle.
    const tailTone = ctx.createBiquadFilter();
    tailTone.type = 'lowpass';
    tailTone.frequency.value = 4200;

    const ret = ctx.createGain();
    ret.gain.value = 0.85;

    this.wet.connect(conv).connect(tailTone).connect(ret).connect(this.duck);
    this.bus.connect(this.duck);
    this.duck.connect(glue).connect(limiter).connect(this.master).connect(ctx.destination);
  }

  /**
   * Exponentially decaying noise with a discrete early-reflection cluster and
   * progressive high-frequency damping. Pure decaying noise reads as a plate;
   * the taps in the first 90ms are what make it read as a street.
   */
  _buildImpulse(seconds) {
    const ctx = this.ctx;
    const sr = ctx.sampleRate;
    const len = Math.max(1, Math.floor(sr * seconds));
    const buf = ctx.createBuffer(2, len, sr);
    const taps = [[0.007, 0.72], [0.013, -0.54], [0.023, 0.46], [0.031, -0.33],
                  [0.047, 0.29], [0.063, -0.21], [0.085, 0.17]];

    for (let ch = 0; ch < 2; ch++) {
      const d = buf.getChannelData(ch);
      let lp = 0;
      for (let i = 0; i < len; i++) {
        const t = i / len;
        const decay = Math.pow(1 - t, 2.6);
        // Damping coefficient climbs with time so the tail darkens as it dies.
        const k = 0.62 - t * 0.42;
        lp += ((Math.random() * 2 - 1) - lp) * k;
        d[i] = lp * decay;
      }
      // Decorrelating the two channels by a few samples widens the tail.
      const skew = ch ? Math.floor(sr * 0.0017) : 0;
      for (const [ms, amp] of taps) {
        const idx = Math.floor(ms * sr) + skew;
        if (idx < len) d[idx] += amp * (ch ? 0.88 : 1);
      }
    }
    return buf;
  }

  /* Built once. Allocating a buffer per shot is what makes browsers hitch. */
  _buildBuffers() {
    const ctx = this.ctx;
    const sr = ctx.sampleRate;
    const n = Math.floor(sr * NOISE_SECONDS);

    for (let v = 0; v < NOISE_VARIANTS; v++) {
      const b = ctx.createBuffer(1, n, sr);
      const d = b.getChannelData(0);
      for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
      this._noiseBufs.push(b);
    }

    // Pink-ish noise for wind: a one-pole cascade is cheap and the -3dB/oct
    // slope is what stops the ambience bed from hissing.
    const p = ctx.createBuffer(1, n, sr);
    const pd = p.getChannelData(0);
    let b0 = 0, b1 = 0, b2 = 0;
    for (let i = 0; i < n; i++) {
      const w = Math.random() * 2 - 1;
      b0 = 0.997 * b0 + w * 0.0555;
      b1 = 0.963 * b1 + w * 0.0750;
      b2 = 0.575 * b2 + w * 0.1538;
      pd[i] = (b0 + b1 + b2 + w * 0.0184) * 3.2;
    }
    this._pinkBuf = p;
  }

  /* ------------------------------------------------------------- voices */

  _reap(now) {
    const v = this.voices;
    for (let i = v.length - 1; i >= 0; i--) {
      if (v[i].end > now) continue;
      for (const n of v[i].chain) { try { n.disconnect(); } catch { /* already gone */ } }
      v.splice(i, 1);
    }
  }

  /** Drops the least valuable voice: quiet and old loses to loud and new. */
  _steal(now) {
    let worst = null, worstScore = Infinity;
    for (const v of this.voices) {
      if (v.dying) continue;
      const score = v.amp * v.prio - (now - v.start) * 1.5;
      if (score < worstScore) { worstScore = score; worst = v; }
    }
    if (!worst) return;
    this._kill(worst, now);
  }

  _kill(v, now, fade = 0.012) {
    v.dying = true;
    try {
      const g = v.input.gain;
      g.cancelScheduledValues(now);
      g.setValueAtTime(g.value, now);
      g.linearRampToValueAtTime(0, now + fade);
    } catch { /* param already detached */ }
    for (const s of v.sources) { try { s.stop(now + fade); } catch { /* not started */ } }
    v.end = now + fade;
  }

  /**
   * Allocates the voice's output chain. Positional voices get air absorption,
   * an inverse-rolloff panner, and a travel delay before they hit the bus.
   */
  _voice(spec, pos, o) {
    const ctx = this.ctx;
    const now = ctx.currentTime;
    this._reap(now);
    if (this.voices.length >= this.maxVoices) this._steal(now);
    if (this.voices.length >= this.maxVoices) return null;

    const amp = spec.amp * (o.volume != null ? o.volume : 1);
    const input = ctx.createGain();
    input.gain.value = amp;

    const chain = [input];
    let tail = input;
    let wetScale = 1;
    let travel = 0;

    if (pos) {
      const dx = pos.x - this._listener.x;
      const dy = pos.y - this._listener.y;
      const dz = pos.z - this._listener.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

      // Air absorption: half the brightness every ~18m. A rifle at 60m must
      // read as a distant thump, not as a quiet rifle.
      const air = ctx.createBiquadFilter();
      air.type = 'lowpass';
      air.frequency.value = clamp(16000 * Math.pow(0.5, dist / 18), 380, 16000);
      air.Q.value = 0.6;

      const pan = ctx.createPanner();
      pan.panningModel = this._hrtf ? 'HRTF' : 'equalpower';
      if (pan.panningModel !== 'HRTF') this._hrtf = false;   // refused: stop asking
      pan.distanceModel = 'inverse';
      pan.refDistance = 5;
      pan.rolloffFactor = 1.15;
      pan.maxDistance = 500;
      this._setPos(pan, pos);

      tail = input.connect(air).connect(pan);
      chain.push(air, pan);

      // Sound arrives late over distance, and the further it is the more of it
      // reaches us as room rather than as source.
      travel = Math.min(0.55, dist / SPEED_OF_SOUND);
      wetScale = 1 + Math.min(2.4, dist / 22);

      if (travel > 0.02) {
        const dly = ctx.createDelay(1);
        dly.delayTime.value = travel;
        tail = tail.connect(dly);
        chain.push(dly);
      }
    } else if (o.pan && ctx.createStereoPanner) {
      const sp = ctx.createStereoPanner();
      sp.pan.value = clamp(o.pan, -1, 1);
      tail = input.connect(sp);
      chain.push(sp);
    }

    tail.connect(this.bus);

    const wet = (o.wet != null ? o.wet : spec.wet) * wetScale;
    if (wet > 0.001) {
      const send = ctx.createGain();
      send.gain.value = Math.min(1.6, wet);
      tail.connect(send).connect(this.wet);
      chain.push(send);
    }

    const v = {
      input, chain, sources: [], amp, prio: spec.prio,
      start: now, end: now + travel + spec.dur + 0.08, dying: false,
    };
    this.voices.push(v);
    return v;
  }

  _setPos(pan, p) {
    if (pan.positionX) {
      pan.positionX.value = p.x;
      pan.positionY.value = p.y;
      pan.positionZ.value = p.z;
    } else {
      pan.setPosition(p.x, p.y, p.z);      // Safari still ships the old API
    }
  }

  setListener(camera) {
    const ctx = this.ctx;
    if (!ctx || !camera) return;
    const e = camera.matrixWorld?.elements;
    if (!e) return;

    this._listener.x = e[12];
    this._listener.y = e[13];
    this._listener.z = e[14];

    const l = ctx.listener;
    // Column 2 of the view matrix points behind the camera in Three's
    // convention, so forward is its negation.
    const fx = -e[8], fy = -e[9], fz = -e[10];
    const ux = e[4], uy = e[5], uz = e[6];

    if (l.positionX) {
      l.positionX.value = e[12];
      l.positionY.value = e[13];
      l.positionZ.value = e[14];
      l.forwardX.value = fx; l.forwardY.value = fy; l.forwardZ.value = fz;
      l.upX.value = ux; l.upY.value = uy; l.upZ.value = uz;
    } else {
      l.setPosition(e[12], e[13], e[14]);
      l.setOrientation(fx, fy, fz, ux, uy, uz);
    }

    // Piggy-backing the per-frame call keeps dead voices from accumulating
    // without a timer of its own.
    this._reap(ctx.currentTime);
  }

  /* ------------------------------------------------------------ playback */

  play(name, opts) { return this._spawn(name, null, opts || {}); }

  playAt(name, position, opts) {
    return this._spawn(name, position || null, opts || {});
  }

  _spawn(name, pos, o) {
    if (!this.enabled) return null;
    if (!this.ctx) { this.resume(); if (!this.ctx) return null; }
    if (this.ctx.state !== 'running') this.resume();

    let spec = SPEC[name];
    if (!spec) {
      if (!this._warned.has(name)) {
        this._warned.add(name);
        console.warn('[audio] unknown sound "' + name + '" — using fallback click');
      }
      spec = SPEC._fallback;
      name = '_fallback';
    }

    const v = this._voice(spec, pos, o);
    if (!v) return null;

    const t = this.ctx.currentTime + Math.max(0, o.delay || 0) + 0.002;
    try {
      this._render(name, v, t, o);
    } catch (e) {
      console.warn('[audio] synth failed for', name, e);
      this._kill(v, this.ctx.currentTime, 0.001);
    }
    return v;
  }

  /* -------------------------------------------------------- synth helpers */

  _osc(v, type, freq, t) {
    const o = this.ctx.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(freq, t);
    v.sources.push(o);
    o.start(t);
    return o;
  }

  _noise(v, t, rate = 1) {
    const ctx = this.ctx;
    const buf = this._noiseBufs[(Math.random() * this._noiseBufs.length) | 0];
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.loop = true;
    src.playbackRate.value = rate;
    v.sources.push(src);
    // A random read offset is a free reseed — no per-shot buffer allocation.
    src.start(t, Math.random() * buf.duration * 0.6);
    return src;
  }

  _filter(type, freq, q = 1) {
    const f = this.ctx.createBiquadFilter();
    f.type = type;
    f.frequency.value = freq;
    f.Q.value = q;
    return f;
  }

  _gain(value = 1) {
    const g = this.ctx.createGain();
    g.gain.value = value;
    return g;
  }

  /** Percussive envelope. Exponential ramps never reach zero, hence the floor. */
  _env(param, t, peak, attack, dur, floor = 0.0006) {
    param.setValueAtTime(floor, t);
    param.exponentialRampToValueAtTime(Math.max(peak, floor * 2), t + attack);
    param.exponentialRampToValueAtTime(floor, t + Math.max(dur, attack + 0.01));
    param.setValueAtTime(0, t + Math.max(dur, attack + 0.01) + 0.001);
  }

  /** Short bright transient — the "snap" that sells any impact. */
  _click(v, t, { hz = 3800, q = 0.9, amp = 0.5, dur = 0.012 } = {}) {
    const n = this._noise(v, t, rand(0.9, 1.15));
    const hp = this._filter('highpass', hz, q);
    const g = this._gain(0);
    this._env(g.gain, t, amp, 0.0008, dur);
    n.connect(hp).connect(g).connect(v.input);
    n.stop(t + dur + 0.02);
  }

  _bandHit(v, t, { hz, q = 8, amp = 0.4, dur = 0.1, rate = 1 } = {}) {
    const n = this._noise(v, t, rate);
    const bp = this._filter('bandpass', hz, q);
    const g = this._gain(0);
    this._env(g.gain, t, amp, 0.0015, dur);
    n.connect(bp).connect(g).connect(v.input);
    n.stop(t + dur + 0.03);
    return bp;
  }

  _tone(v, t, { type = 'sine', from, to, amp = 0.4, attack = 0.004, dur = 0.2 }) {
    const o = this._osc(v, type, from, t);
    if (to && to !== from) o.frequency.exponentialRampToValueAtTime(Math.max(20, to), t + dur);
    const g = this._gain(0);
    this._env(g.gain, t, amp, attack, dur);
    o.connect(g).connect(v.input);
    o.stop(t + dur + 0.03);
    return o;
  }

  _duckMix(t, amount, hold, release) {
    const g = this.duck.gain;
    g.cancelScheduledValues(t);
    g.setValueAtTime(g.value, t);
    g.linearRampToValueAtTime(amount, t + 0.025);
    g.setValueAtTime(amount, t + hold);
    g.linearRampToValueAtTime(1, t + hold + release);
  }

  /* ------------------------------------------------------------- the bank */

  _render(name, v, t, o) {
    switch (name) {
      case 'rifle':  return this._gun(v, t, GUNS.rifle);
      case 'smg':    return this._gun(v, t, GUNS.smg);
      case 'sniper': return this._gun(v, t, GUNS.sniper);
      case 'enemyFire': return this._gun(v, t, GUNS.enemy);

      case 'dryfire':        return this._dryfire(v, t);
      case 'reload_magout':  return this._magOut(v, t);
      case 'reload_magin':   return this._magIn(v, t);
      case 'reload_charge':  return this._charge(v, t);
      case 'shellDrop':      return this._shell(v, t);

      case 'impact_concrete':
      case 'impact_metal':
      case 'impact_wood':
      case 'impact_sand':
      case 'impact_flesh':   return this._impact(v, t, name.slice(7));
      case 'ricochet':       return this._ricochet(v, t);

      case 'explosion':      return this._explosion(v, t, o);
      case 'grenadeBounce':  return this._grenadeBounce(v, t);
      case 'grenadePin':     return this._grenadePin(v, t);
      case 'melee':          return this._melee(v, t);

      case 'hurt':           return this._hurt(v, t);
      case 'death':          return this._death(v, t, 1);
      case 'enemyDeath':     return this._death(v, t, rand(1.25, 1.6));

      case 'footstep_sand':
      case 'footstep_concrete':
      case 'footstep_metal': return this._footstep(v, t, name.slice(9));
      case 'jump':           return this._jump(v, t);
      case 'land':           return this._land(v, t);

      case 'hitmarker':      return this._marker(v, t, 1780, 0.035);
      case 'headshot':       return this._headshot(v, t);
      case 'killconfirm':    return this._killconfirm(v, t);
      case 'uiClick':        return this._marker(v, t, 1050, 0.028);
      case 'uiHover':        return this._marker(v, t, 640, 0.018);
      case 'waveStart':      return this._waveStart(v, t);
      case 'lowAmmo':        return this._lowAmmo(v, t);
      case 'lowHealth':      return this._lowHealth(v, t);

      case 'amb_gunfire':    return this._ambGunfire(v, t);
      case 'amb_dog':        return this._ambDog(v, t);
      case 'amb_gust':       return this._ambGust(v, t);

      default:               return this._marker(v, t, 900, 0.02);
    }
  }

  /**
   * Four layers, because a real gunshot is four events: the supersonic crack,
   * the low-frequency body of the muzzle blast, the ignition transient, and
   * the bolt cycling a moment later.
   */
  _gun(v, t, c) {
    const p = jitter(0.04);

    // (a) crack — noise through a lowpass that slams shut
    const n = this._noise(v, t, rand(0.92, 1.1));
    const lp = this._filter('lowpass', c.crackHz * p, 1.1);
    lp.frequency.exponentialRampToValueAtTime(c.crackHz * 0.14, t + c.crack);
    const hp = this._filter('highpass', 180, 0.7);
    const ng = this._gain(0);
    this._env(ng.gain, t, c.crackAmp, 0.0015, c.crack);
    n.connect(lp).connect(hp).connect(ng).connect(v.input);
    n.stop(t + c.crack + 0.03);

    // (b) body — a detuned pair swept down is the thump you feel
    for (let i = 0; i < 2; i++) {
      const o = this._osc(v, i ? 'triangle' : 'sawtooth',
        c.bodyHz * p * (i ? 1.017 : 0.981), t);
      o.frequency.exponentialRampToValueAtTime(c.bodyHz * 0.4 * p, t + c.body);
      const tone = this._filter('lowpass', 1100, 0.8);
      const g = this._gain(0);
      this._env(g.gain, t, c.bodyAmp * (i ? 0.55 : 1), 0.004, c.body);
      o.connect(tone).connect(g).connect(v.input);
      o.stop(t + c.body + 0.03);
    }

    // (c) ignition transient
    this._click(v, t, { hz: 4200, amp: c.crackAmp * 0.8, dur: 0.009 });

    // (d) action cycling, offset so it reads as mechanism and not as part of
    // the blast
    const mt = t + c.mech * jitter(0.12);
    this._bandHit(v, mt, { hz: c.mechHz * jitter(0.08), q: 6, amp: c.mechAmp, dur: 0.03 });
    this._bandHit(v, mt + 0.018, { hz: c.mechHz * 1.7, q: 9, amp: c.mechAmp * 0.5, dur: 0.02 });
  }

  _dryfire(v, t) {
    this._bandHit(v, t, { hz: 2300 * jitter(0.05), q: 4, amp: 0.6, dur: 0.022 });
    this._click(v, t, { hz: 5200, amp: 0.35, dur: 0.006 });
    // Spring ring: the tell that nothing chambered.
    this._bandHit(v, t + 0.012, { hz: 5400, q: 22, amp: 0.14, dur: 0.075 });
  }

  _magOut(v, t) {
    this._bandHit(v, t, { hz: 1700 * jitter(0.06), q: 5, amp: 0.5, dur: 0.03 });
    this._tone(v, t + 0.004, { type: 'triangle', from: 220, to: 130, amp: 0.22, dur: 0.07 });
    this._bandHit(v, t + 0.09, { hz: 950, q: 7, amp: 0.28, dur: 0.06 });
  }

  _magIn(v, t) {
    this._bandHit(v, t, { hz: 900 * jitter(0.06), q: 8, amp: 0.45, dur: 0.05 });
    this._tone(v, t, { type: 'triangle', from: 150, to: 82, amp: 0.4, dur: 0.1 });
    this._click(v, t + 0.006, { hz: 3200, amp: 0.3, dur: 0.01 });
    this._bandHit(v, t + 0.13, { hz: 2100, q: 6, amp: 0.22, dur: 0.03 });
  }

  _charge(v, t) {
    // Slide: a bandpass climbing over ~90ms, then the bolt slamming home.
    const n = this._noise(v, t, rand(0.9, 1.1));
    const bp = this._filter('bandpass', 1200, 3);
    bp.frequency.exponentialRampToValueAtTime(2700, t + 0.09);
    const g = this._gain(0);
    this._env(g.gain, t, 0.3, 0.01, 0.1);
    n.connect(bp).connect(g).connect(v.input);
    n.stop(t + 0.13);

    const st = t + 0.11;
    this._bandHit(v, st, { hz: 1500 * jitter(0.05), q: 6, amp: 0.55, dur: 0.035 });
    this._tone(v, st, { type: 'triangle', from: 190, to: 95, amp: 0.3, dur: 0.09 });
    this._bandHit(v, st + 0.01, { hz: 4600, q: 18, amp: 0.16, dur: 0.06 });
  }

  _shell(v, t) {
    const n = 2 + (Math.random() * 2 | 0);
    let at = t;
    for (let i = 0; i < n; i++) {
      const hz = rand(2400, 4600);
      const amp = 0.5 * Math.pow(0.62, i);
      this._bandHit(v, at, { hz, q: 24, amp, dur: 0.09 });
      this._bandHit(v, at, { hz: hz * 1.63, q: 30, amp: amp * 0.5, dur: 0.06 });
      this._click(v, at, { hz: 6000, amp: amp * 0.4, dur: 0.005 });
      at += rand(0.045, 0.11);       // bounces crowd together as it settles
    }
  }

  _impact(v, t, surface) {
    switch (surface) {
      case 'concrete': {
        this._click(v, t, { hz: 3000, amp: 0.7, dur: 0.01 });
        this._bandHit(v, t, { hz: 1600 * jitter(0.1), q: 1.4, amp: 0.6, dur: 0.06 });
        // Dust and grit falling away after the strike.
        const n = this._noise(v, t + 0.01, rand(0.8, 1.2));
        const lp = this._filter('lowpass', 2600, 0.7);
        const g = this._gain(0);
        this._env(g.gain, t + 0.01, 0.18, 0.01, 0.2);
        n.connect(lp).connect(g).connect(v.input);
        n.stop(t + 0.24);
        break;
      }
      case 'metal': {
        this._click(v, t, { hz: 5200, amp: 0.6, dur: 0.008 });
        const base = rand(1500, 2600);
        // Inharmonic partials: a struck plate is not a tuned bell.
        for (const [mul, q, amp, dur] of [[1, 26, 0.5, 0.28], [2.41, 34, 0.3, 0.2], [3.87, 40, 0.18, 0.14]]) {
          this._bandHit(v, t, { hz: base * mul, q, amp, dur });
        }
        this._tone(v, t, { type: 'triangle', from: base * 0.5, to: base * 0.47, amp: 0.12, dur: 0.22 });
        break;
      }
      case 'wood': {
        this._click(v, t, { hz: 2600, amp: 0.5, dur: 0.008 });
        this._bandHit(v, t, { hz: 700 * jitter(0.12), q: 3.5, amp: 0.55, dur: 0.09 });
        this._tone(v, t, { type: 'triangle', from: 210 * jitter(0.1), to: 120, amp: 0.35, dur: 0.11 });
        break;
      }
      case 'sand': {
        const n = this._noise(v, t, rand(0.85, 1.15));
        const hp = this._filter('highpass', 900, 0.6);
        const lp = this._filter('lowpass', 5200, 0.7);
        const g = this._gain(0);
        this._env(g.gain, t, 0.55, 0.004, 0.13);
        n.connect(hp).connect(lp).connect(g).connect(v.input);
        n.stop(t + 0.17);
        this._tone(v, t, { type: 'sine', from: 130, to: 70, amp: 0.16, dur: 0.06 });
        break;
      }
      default: {   // flesh
        this._tone(v, t, { type: 'sine', from: 190 * jitter(0.1), to: 62, amp: 0.7, dur: 0.1 });
        const n = this._noise(v, t, rand(0.8, 1.1));
        const lp = this._filter('lowpass', 1400, 1.2);
        const g = this._gain(0);
        this._env(g.gain, t, 0.45, 0.002, 0.07);
        n.connect(lp).connect(g).connect(v.input);
        n.stop(t + 0.1);
        this._bandHit(v, t + 0.004, { hz: 420, q: 2, amp: 0.2, dur: 0.05 });
      }
    }
  }

  _ricochet(v, t) {
    const f0 = rand(1300, 3200);
    const f1 = f0 * rand(0.4, 2.2);      // whines up or down, never the same twice
    const o = this._osc(v, 'sine', f0, t);
    o.frequency.exponentialRampToValueAtTime(f1, t + 0.22);

    // Vibrato is what makes it a whine instead of a test tone.
    const lfo = this._osc(v, 'sine', rand(28, 60), t);
    const lfoAmt = this._gain(f0 * 0.045);
    lfo.connect(lfoAmt).connect(o.frequency);
    lfo.stop(t + 0.3);

    const g = this._gain(0);
    this._env(g.gain, t, 0.5, 0.006, 0.26);
    o.connect(g).connect(v.input);
    o.stop(t + 0.3);

    this._bandHit(v, t, { hz: f0 * 1.4, q: 6, amp: 0.35, dur: 0.05 });
    this._click(v, t, { hz: 4000, amp: 0.4, dur: 0.007 });
  }

  _explosion(v, t, o) {
    // Sub sweep: the part you feel. Sine only — anything richer turns to mud
    // once the limiter grabs it.
    this._tone(v, t, { type: 'sine', from: 110 * jitter(0.1), to: 24, amp: 1.0, attack: 0.012, dur: 0.9 });
    this._tone(v, t, { type: 'triangle', from: 220, to: 48, amp: 0.35, attack: 0.008, dur: 0.5 });

    // Broadband blast with a lowpass collapsing over a second.
    const n = this._noise(v, t, rand(0.85, 1.05));
    const lp = this._filter('lowpass', 3200, 0.9);
    lp.frequency.exponentialRampToValueAtTime(140, t + 1.1);
    const g = this._gain(0);
    this._env(g.gain, t, 0.9, 0.006, 1.2);
    n.connect(lp).connect(g).connect(v.input);
    n.stop(t + 1.35);

    // Debris scatter riding on top of the tail.
    const d = this._noise(v, t + 0.12, rand(0.6, 0.9));
    const dbp = this._filter('bandpass', 2200, 1.2);
    const dg = this._gain(0);
    this._env(dg.gain, t + 0.12, 0.16, 0.06, 0.9);
    d.connect(dbp).connect(dg).connect(v.input);
    d.stop(t + 1.1);

    this._click(v, t, { hz: 5000, amp: 0.9, dur: 0.012 });

    // Ear ringing survives the blast, so it hangs off the master rather than
    // the voice's positional chain.
    const ring = this.ctx.createOscillator();
    ring.type = 'sine';
    ring.frequency.value = 4000 * jitter(0.06);
    const rg = this._gain(0);
    rg.gain.setValueAtTime(0.0006, t);
    rg.gain.exponentialRampToValueAtTime(0.05, t + 0.06);
    rg.gain.exponentialRampToValueAtTime(0.0006, t + 2.0);
    rg.gain.setValueAtTime(0, t + 2.01);
    ring.connect(rg).connect(this.bus);
    ring.start(t);
    ring.stop(t + 2.1);

    const dist = o && o.distance != null ? o.distance : 0;
    const depth = clamp(0.34 + dist / 120, 0.34, 0.9);
    this._duckMix(t, depth, 0.35, 1.15);
  }

  _grenadeBounce(v, t) {
    this._bandHit(v, t, { hz: 780 * jitter(0.15), q: 13, amp: 0.6, dur: 0.11 });
    this._bandHit(v, t, { hz: 2400 * jitter(0.15), q: 20, amp: 0.25, dur: 0.06 });
    this._click(v, t, { hz: 3600, amp: 0.35, dur: 0.006 });
  }

  _grenadePin(v, t) {
    this._bandHit(v, t, { hz: 5200 * jitter(0.05), q: 30, amp: 0.5, dur: 0.07 });
    this._bandHit(v, t + 0.055, { hz: 4300 * jitter(0.05), q: 26, amp: 0.35, dur: 0.06 });
    this._click(v, t, { hz: 6000, amp: 0.25, dur: 0.005 });
  }

  _melee(v, t) {
    // Whoosh first, contact second — the gap is what gives the swing weight.
    const n = this._noise(v, t, rand(0.9, 1.1));
    const bp = this._filter('bandpass', 350, 1.1);
    bp.frequency.exponentialRampToValueAtTime(2400, t + 0.11);
    bp.frequency.exponentialRampToValueAtTime(600, t + 0.2);
    const g = this._gain(0);
    g.gain.setValueAtTime(0.0006, t);
    g.gain.exponentialRampToValueAtTime(0.5, t + 0.09);
    g.gain.exponentialRampToValueAtTime(0.0006, t + 0.21);
    n.connect(bp).connect(g).connect(v.input);
    n.stop(t + 0.24);

    const ht = t + 0.13;
    this._tone(v, ht, { type: 'sine', from: 160, to: 55, amp: 0.6, dur: 0.12 });
    this._bandHit(v, ht, { hz: 900, q: 3, amp: 0.4, dur: 0.06 });
  }

  _hurt(v, t) {
    const p = jitter(0.08);
    // Vocal-ish grunt: a formant band over a falling saw.
    const o = this._osc(v, 'sawtooth', 165 * p, t);
    o.frequency.exponentialRampToValueAtTime(105 * p, t + 0.3);
    const fm = this._filter('bandpass', 620 * p, 3.5);
    const lp = this._filter('lowpass', 1800, 0.8);
    const g = this._gain(0);
    g.gain.setValueAtTime(0.0006, t);
    g.gain.exponentialRampToValueAtTime(0.6, t + 0.03);
    g.gain.exponentialRampToValueAtTime(0.0006, t + 0.34);
    o.connect(fm).connect(lp).connect(g).connect(v.input);
    o.stop(t + 0.38);

    // Breath through the grunt.
    const n = this._noise(v, t, 1);
    const bp = this._filter('bandpass', 1100, 1.1);
    const ng = this._gain(0);
    this._env(ng.gain, t, 0.16, 0.02, 0.3);
    n.connect(bp).connect(ng).connect(v.input);
    n.stop(t + 0.36);
  }

  _death(v, t, pitch) {
    const o = this._osc(v, 'sawtooth', 190 * pitch * jitter(0.06), t);
    o.frequency.exponentialRampToValueAtTime(58 * pitch, t + 0.9);
    const fm = this._filter('bandpass', 520 * pitch, 2.6);
    const lp = this._filter('lowpass', 1400, 0.8);
    lp.frequency.exponentialRampToValueAtTime(320, t + 1.0);
    const g = this._gain(0);
    g.gain.setValueAtTime(0.0006, t);
    g.gain.exponentialRampToValueAtTime(0.55, t + 0.05);
    g.gain.exponentialRampToValueAtTime(0.0006, t + 0.95);
    o.connect(fm).connect(lp).connect(g).connect(v.input);
    o.stop(t + 1.0);

    // Body hitting the ground, late enough to read as a separate event.
    const ft = t + 0.42;
    this._tone(v, ft, { type: 'sine', from: 90, to: 40, amp: 0.5, dur: 0.25 });
    this._bandHit(v, ft, { hz: 500, q: 1.6, amp: 0.3, dur: 0.14 });
  }

  _footstep(v, t, surface) {
    const rate = rand(0.88, 1.14);
    if (surface === 'sand') {
      const n = this._noise(v, t, rate);
      const hp = this._filter('highpass', 1400, 0.5);
      const lp = this._filter('lowpass', 7000, 0.6);
      const g = this._gain(0);
      // Soft attack: sand absorbs the transient entirely.
      g.gain.setValueAtTime(0.0006, t);
      g.gain.exponentialRampToValueAtTime(0.6, t + 0.012);
      g.gain.exponentialRampToValueAtTime(0.0006, t + 0.14);
      n.connect(hp).connect(lp).connect(g).connect(v.input);
      n.stop(t + 0.17);
      this._tone(v, t, { type: 'sine', from: 105, to: 62, amp: 0.18, dur: 0.07 });
    } else if (surface === 'metal') {
      this._bandHit(v, t, { hz: 880 * jitter(0.1), q: 16, amp: 0.5, dur: 0.22, rate });
      this._bandHit(v, t, { hz: 2350 * jitter(0.1), q: 22, amp: 0.28, dur: 0.14, rate });
      this._click(v, t, { hz: 4200, amp: 0.3, dur: 0.006 });
    } else {
      this._bandHit(v, t, { hz: 2100 * jitter(0.12), q: 1.3, amp: 0.55, dur: 0.05, rate });
      this._click(v, t, { hz: 5000, amp: 0.3, dur: 0.005 });
      this._tone(v, t, { type: 'sine', from: 140, to: 80, amp: 0.22, dur: 0.06 });
    }
  }

  _jump(v, t) {
    const n = this._noise(v, t, rand(0.9, 1.1));
    const bp = this._filter('bandpass', 900, 1.0);
    const g = this._gain(0);
    this._env(g.gain, t, 0.4, 0.008, 0.16);
    n.connect(bp).connect(g).connect(v.input);
    n.stop(t + 0.2);
    this._tone(v, t, { type: 'sine', from: 150, to: 95, amp: 0.25, dur: 0.09 });
  }

  _land(v, t) {
    this._tone(v, t, { type: 'sine', from: 95, to: 42, amp: 0.8, dur: 0.24 });
    this._bandHit(v, t, { hz: 1500, q: 1.2, amp: 0.45, dur: 0.09 });
    this._click(v, t, { hz: 4000, amp: 0.3, dur: 0.008 });
    // Gear settling a beat after the boots land.
    this._bandHit(v, t + 0.06, { hz: 2600, q: 9, amp: 0.12, dur: 0.07 });
  }

  _marker(v, t, hz, dur) {
    // Square through a highpass: reads instantly even under a firefight.
    const o = this._osc(v, 'square', hz, t);
    const hp = this._filter('highpass', hz * 0.7, 0.7);
    const g = this._gain(0);
    this._env(g.gain, t, 0.5, 0.001, dur);
    o.connect(hp).connect(g).connect(v.input);
    o.stop(t + dur + 0.02);
    this._click(v, t, { hz: 5000, amp: 0.2, dur: 0.004 });
  }

  _headshot(v, t) {
    // Same shape as the hitmarker but a fifth up and doubled — different at a
    // glance without needing a separate mental category.
    this._marker(v, t, 2640, 0.035);
    this._marker(v, t + 0.045, 3520, 0.045);
  }

  _killconfirm(v, t) {
    const notes = [880, 1320, 1760];
    for (let i = 0; i < notes.length; i++) {
      const at = t + i * 0.06;
      this._tone(v, at, { type: 'triangle', from: notes[i], amp: 0.45, attack: 0.003, dur: 0.09 });
      this._tone(v, at, { type: 'sine', from: notes[i] * 2, amp: 0.14, attack: 0.002, dur: 0.05 });
    }
  }

  _waveStart(v, t) {
    // Low swell under a rising noise bed, resolving on a hit.
    for (const hz of [55, 82.5, 110]) {
      const o = this._osc(v, 'sawtooth', hz * jitter(0.004), t);
      const lp = this._filter('lowpass', 200, 3);
      lp.frequency.exponentialRampToValueAtTime(1800, t + 1.1);
      const g = this._gain(0);
      g.gain.setValueAtTime(0.0006, t);
      g.gain.exponentialRampToValueAtTime(0.22, t + 0.9);
      g.gain.exponentialRampToValueAtTime(0.0006, t + 1.5);
      o.connect(lp).connect(g).connect(v.input);
      o.stop(t + 1.55);
    }

    const n = this._noise(v, t, 1);
    const bp = this._filter('bandpass', 400, 1.4);
    bp.frequency.exponentialRampToValueAtTime(4000, t + 1.05);
    const ng = this._gain(0);
    ng.gain.setValueAtTime(0.0006, t);
    ng.gain.exponentialRampToValueAtTime(0.14, t + 1.0);
    ng.gain.exponentialRampToValueAtTime(0.0006, t + 1.2);
    n.connect(bp).connect(ng).connect(v.input);
    n.stop(t + 1.25);

    const ht = t + 1.05;
    this._tone(v, ht, { type: 'sine', from: 110, to: 38, amp: 0.8, dur: 0.5 });
    this._click(v, ht, { hz: 3000, amp: 0.5, dur: 0.012 });
    this._duckMix(ht, 0.65, 0.15, 0.5);
  }

  _lowAmmo(v, t) {
    this._marker(v, t, 1400, 0.03);
    this._marker(v, t + 0.075, 1180, 0.035);
  }

  _lowHealth(v, t) {
    // Two thumps at resting-heartbeat spacing; the body knows what it means.
    for (const [off, amp] of [[0, 0.9], [0.26, 0.6]]) {
      this._tone(v, t + off, { type: 'sine', from: 78, to: 42, amp: amp * 0.8, attack: 0.012, dur: 0.22 });
      const n = this._noise(v, t + off, 0.6);
      const lp = this._filter('lowpass', 260, 1.2);
      const g = this._gain(0);
      this._env(g.gain, t + off, 0.12 * amp, 0.01, 0.16);
      n.connect(lp).connect(g).connect(v.input);
      n.stop(t + off + 0.2);
    }
  }

  /* ---------------------------------------------------------- ambience */

  startAmbience() {
    this.resume();
    if (!this.ctx || this._amb) return;
    const ctx = this.ctx;

    const out = this._gain(0);
    out.connect(this.bus);
    out.gain.setValueAtTime(0.0001, ctx.currentTime);
    out.gain.exponentialRampToValueAtTime(0.5, ctx.currentTime + 2.5);

    const src = ctx.createBufferSource();
    src.buffer = this._pinkBuf;
    src.loop = true;
    src.playbackRate.value = 0.75;      // stretched, so the loop point is inaudible

    const body = this._filter('lowpass', 420, 0.9);
    const air = this._filter('highpass', 90, 0.7);
    const bodyGain = this._gain(0.5);

    // Hiss layer with its own gust envelope, detuned against the body layer so
    // the two never line up and the bed never reveals its period.
    const hiss = ctx.createBufferSource();
    hiss.buffer = this._noiseBufs[0];
    hiss.loop = true;
    hiss.playbackRate.value = 0.31;
    const hissBp = this._filter('bandpass', 1900, 0.8);
    const hissGain = this._gain(0.05);

    const lfoA = ctx.createOscillator();
    lfoA.frequency.value = 0.061;
    const lfoAAmt = this._gain(260);
    lfoA.connect(lfoAAmt).connect(body.frequency);

    const lfoB = ctx.createOscillator();
    lfoB.frequency.value = 0.037;
    const lfoBAmt = this._gain(0.3);
    lfoB.connect(lfoBAmt).connect(bodyGain.gain);

    const lfoC = ctx.createOscillator();
    lfoC.frequency.value = 0.023;
    const lfoCAmt = this._gain(0.035);
    lfoC.connect(lfoCAmt).connect(hissGain.gain);

    src.connect(air).connect(body).connect(bodyGain).connect(out);
    hiss.connect(hissBp).connect(hissGain).connect(out);

    const t0 = ctx.currentTime;
    src.start(t0); hiss.start(t0); lfoA.start(t0); lfoB.start(t0); lfoC.start(t0);

    this._amb = { out, nodes: [src, hiss, lfoA, lfoB, lfoC] };
    this._scheduleAmbEvent();
  }

  stopAmbience() {
    const a = this._amb;
    if (!a) return;
    this._amb = null;
    clearTimeout(this._ambTimer);
    const t = this.ctx.currentTime;
    a.out.gain.cancelScheduledValues(t);
    a.out.gain.setValueAtTime(a.out.gain.value, t);
    a.out.gain.linearRampToValueAtTime(0, t + 0.6);
    for (const n of a.nodes) { try { n.stop(t + 0.7); } catch { /* already stopped */ } }
    setTimeout(() => { try { a.out.disconnect(); } catch { /* gone */ } }, 900);
  }

  /** Sparse random one-shots are the difference between a bed and a drone. */
  _scheduleAmbEvent() {
    clearTimeout(this._ambTimer);
    this._ambTimer = setTimeout(() => {
      if (!this._amb) return;
      if (this.ctx.state === 'running') {
        const r = Math.random();
        const name = r < 0.45 ? 'amb_gunfire' : r < 0.7 ? 'amb_dog' : 'amb_gust';
        this._spawn(name, null, { pan: rand(-0.9, 0.9) });
      }
      this._scheduleAmbEvent();
    }, rand(5000, 16000));
  }

  _ambGunfire(v, t) {
    const rounds = 2 + (Math.random() * 4 | 0);
    let at = t;
    const gap = rand(0.07, 0.14);
    for (let i = 0; i < rounds; i++) {
      // Distance has already eaten everything above ~700Hz by the time it
      // reaches us, so this is a thump with a click, not a gunshot.
      const n = this._noise(v, at, rand(0.9, 1.1));
      const lp = this._filter('lowpass', rand(520, 820), 1.3);
      const hp = this._filter('highpass', 110, 0.7);
      const g = this._gain(0);
      this._env(g.gain, at, 0.7, 0.004, 0.13);
      n.connect(lp).connect(hp).connect(g).connect(v.input);
      n.stop(at + 0.16);
      this._tone(v, at, { type: 'sine', from: 95, to: 55, amp: 0.25, dur: 0.1 });
      at += gap * jitter(0.25);
    }
  }

  _ambDog(v, t) {
    const barks = 2 + (Math.random() * 2 | 0);
    let at = t;
    for (let i = 0; i < barks; i++) {
      const p = jitter(0.12);
      const o = this._osc(v, 'sawtooth', 340 * p, at);
      o.frequency.exponentialRampToValueAtTime(180 * p, at + 0.13);
      const fm = this._filter('bandpass', 900 * p, 4);
      const lp = this._filter('lowpass', 1600, 0.8);
      const g = this._gain(0);
      this._env(g.gain, at, 0.6, 0.008, 0.14);
      o.connect(fm).connect(lp).connect(g).connect(v.input);
      o.stop(at + 0.17);
      at += rand(0.22, 0.4);
    }
  }

  _ambGust(v, t) {
    const n = this._noise(v, t, rand(0.4, 0.7));
    const bp = this._filter('bandpass', 500, 0.7);
    bp.frequency.exponentialRampToValueAtTime(rand(1400, 2400), t + 1.4);
    bp.frequency.exponentialRampToValueAtTime(400, t + 3.0);
    const g = this._gain(0);
    g.gain.setValueAtTime(0.0006, t);
    g.gain.exponentialRampToValueAtTime(0.9, t + 1.3);
    g.gain.exponentialRampToValueAtTime(0.0006, t + 3.0);
    n.connect(bp).connect(g).connect(v.input);
    n.stop(t + 3.1);
  }
}

/* Weapon character lives in the numbers: a sniper is a low body with a long
   crack, an SMG is a short bright one, and the enemy's rifle is the player's
   heard from across the street. */
const GUNS = {
  rifle:  { crackHz: 5200, crack: 0.11, crackAmp: 0.9, bodyHz: 122, body: 0.17, bodyAmp: 0.55, mech: 0.034, mechHz: 1900, mechAmp: 0.16 },
  smg:    { crackHz: 4300, crack: 0.07, crackAmp: 0.7, bodyHz: 168, body: 0.10, bodyAmp: 0.38, mech: 0.026, mechHz: 2400, mechAmp: 0.14 },
  sniper: { crackHz: 6400, crack: 0.19, crackAmp: 1.0, bodyHz: 76,  body: 0.32, bodyAmp: 0.85, mech: 0.052, mechHz: 1500, mechAmp: 0.20 },
  enemy:  { crackHz: 3400, crack: 0.13, crackAmp: 0.8, bodyHz: 108, body: 0.20, bodyAmp: 0.60, mech: 0.038, mechHz: 1700, mechAmp: 0.10 },
};

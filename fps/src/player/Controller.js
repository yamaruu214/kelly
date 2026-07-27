// Controller.js — player movement, collision, camera.
//
// The player is a vertical cylinder (radius 0.35) whose origin is at the FEET.
// Motion is acceleration-based: a wish direction feeds an accelerate/friction
// pair rather than writing velocity directly, because instant velocity reads as
// a slideshow of positions instead of a body with mass. Collision is an
// axis-separated sweep (X, then Z, then Y) so a blocked axis never cancels the
// others and the player slides along a wall instead of gluing to it.
//
// Everything here runs at the fixed 1/120 step from main.js.

import * as THREE from 'three';

const RADIUS = 0.35;
const STAND_HEIGHT = 1.80;
const CROUCH_HEIGHT = 1.15;
const EYE_RATIO = 0.88;              // eyes sit below the crown, not on it
const STEP_HEIGHT = 0.45;

const SPEED_WALK = 4.2;
const SPEED_SPRINT = 6.8;
const SPEED_CROUCH = 2.1;
const SPEED_ADS = 2.6;

// Real gravity feels like the moon at these speeds and scales: the airtime of a
// 1m jump at 9.81 is over 0.9s, long enough to aim comfortably mid-flight.
// Shooters run 2-2.5x; 22 puts a 1.05m apex at ~0.62s of hang.
const GRAVITY = -22;
const JUMP_APEX = 1.05;
const JUMP_SPEED = Math.sqrt(2 * -GRAVITY * JUMP_APEX);

const GROUND_ACCEL = 14;             // multiplied by wish speed -> ~59 m/s² walking
const AIR_ACCEL = 2.4;               // low, but never zero: air control is expected
const FRICTION = 8.5;
const STOP_SPEED = 1.8;              // below this friction is constant, so we halt

const COYOTE_TIME = 0.10;
const JUMP_BUFFER = 0.12;
const SPRINT_RAMP = 0.22;
const SPRINT_OUT = 0.18;             // post-sprint delay before the gun is usable

const SLIDE_TIME = 0.75;
const SLIDE_COOLDOWN = 0.45;
const SLIDE_DECAY = 1.7;             // e-fold rate: 8.4 m/s -> ~2.3 m/s over a slide
const SLIDE_STEER = 1.2;             // rad/s of course correction while sliding

const MANTLE_TIME = 0.45;
const MANTLE_MIN = 0.50;
const MANTLE_MAX = 1.60;

const FALL_SAFE = 7;                 // m/s of impact absorbed for free
const FALL_LETHAL = 19;              // ~8.2m drop; below that it only hurts
const REGEN_DELAY = 5;
const REGEN_TIME = 4;

const LEAN_OFFSET = 0.35;
const LEAN_ROLL = 8 * Math.PI / 180;
const STRAFE_ROLL = 1.2 * Math.PI / 180;
const PITCH_LIMIT = 89 * Math.PI / 180;

const EPS = 1e-3;

const clamp = (v, a, b) => v < a ? a : (v > b ? b : v);
/** Frame-rate independent lerp factor for an exponential approach. */
const damp = (rate, dt) => 1 - Math.exp(-rate * dt);
const smoothstep = (a, b, x) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

const NO_COLLIDERS = [];

export class PlayerController {
  constructor(camera, level, input, settings) {
    this.camera = camera;
    this.level = level;
    this.input = input;
    this.settings = settings;

    this.position = new THREE.Vector3();
    this.velocity = new THREE.Vector3();
    this.eyeOffset = new THREE.Vector3();
    this.eyePosition = new THREE.Vector3();
    this.forward = new THREE.Vector3(0, 0, -1);

    this.maxHealth = 100;
    this.health = 100;

    this.yaw = 0;
    this.pitch = 0;
    this.speed = 0;

    this.isGrounded = false;
    this.sprinting = false;
    this.crouching = false;
    this.sliding = false;
    this.mantling = false;
    this.dead = false;
    this.canFire = true;        // weapons gate on this: false while sprinting out
    this.sprintOutTimer = 0;

    this.onDamage = null;
    this.onDeath = null;

    this._height = STAND_HEIGHT;
    this._eyeHeight = STAND_HEIGHT * EYE_RATIO;
    this._wish = new THREE.Vector3();
    this._probe = { x: 0, y: 0, z: 0 };
    this._mantleFrom = new THREE.Vector3();
    this._mantleTo = new THREE.Vector3();
    this._damageDir = new THREE.Vector3();

    this.reset(new THREE.Vector3(0, 0, 0));
  }

  reset(spawn) {
    if (spawn) this.position.set(spawn.x, spawn.y, spawn.z);
    this.velocity.set(0, 0, 0);
    this.health = this.maxHealth;
    this.dead = false;

    this.yaw = this.level?.playerSpawnYaw ?? this.yaw;
    this.pitch = 0;

    this.isGrounded = false;
    this.sprinting = false;
    this.crouching = false;
    this.sliding = false;
    this.mantling = false;
    this.canFire = true;
    this.speed = 0;

    this._height = STAND_HEIGHT;
    this._eyeHeight = STAND_HEIGHT * EYE_RATIO;
    this._coyote = 0;
    this._jumpBuffer = 0;
    this._sprintRamp = 0;
    this._sprintOut = 0;
    this._slideTimer = 0;
    this._slideCooldown = 0;
    this._slideRollSign = 1;
    this._mantleT = 0;
    this._bobDist = 0;
    this._bobAmp = 0;
    this._bobX = 0;
    this._bobY = 0;
    this._landDip = 0;
    this._landVel = 0;
    this._roll = 0;
    this._lean = 0;
    this._sinceDamage = REGEN_DELAY + REGEN_TIME;
    this._blockedHoriz = false;

    this._applyCamera(0);
  }

  /* ------------------------------------------------------------- helpers */
  get _colliders() { return this.level?.colliders || NO_COLLIDERS; }

  /** True when the cylinder at (x,y,z) with the given height hits nothing. */
  _capsuleFree(x, y, z, h) {
    const cols = this._colliders;
    for (let i = 0; i < cols.length; i++) {
      const b = cols[i];
      if (y >= b.max.y - EPS || y + h <= b.min.y + EPS) continue;
      const dx = x - clamp(x, b.min.x, b.max.x);
      const dz = z - clamp(z, b.min.z, b.max.z);
      if (dx * dx + dz * dz < RADIUS * RADIUS) return false;
    }
    return true;
  }

  // Each axis sweep resolves against the exact cylinder silhouette: the box is
  // widened on the moving axis by sqrt(r² - offAxisGap²), which rounds the
  // corners. A plain r-wide expansion leaves square nubs you snag on.
  _sweepX(p, dx, h) {
    p.x += dx;
    if (dx === 0) return false;
    const cols = this._colliders;
    const y0 = p.y + 0.02, y1 = p.y + h;
    let hit = false;
    for (let i = 0; i < cols.length; i++) {
      const b = cols[i];
      if (y1 <= b.min.y + EPS || y0 >= b.max.y - EPS) continue;
      const gz = p.z < b.min.z ? b.min.z - p.z : (p.z > b.max.z ? p.z - b.max.z : 0);
      if (gz >= RADIUS) continue;
      const ex = Math.sqrt(RADIUS * RADIUS - gz * gz);
      if (p.x <= b.min.x - ex || p.x >= b.max.x + ex) continue;
      p.x = dx > 0 ? b.min.x - ex - EPS : b.max.x + ex + EPS;
      hit = true;
    }
    return hit;
  }

  _sweepZ(p, dz, h) {
    p.z += dz;
    if (dz === 0) return false;
    const cols = this._colliders;
    const y0 = p.y + 0.02, y1 = p.y + h;
    let hit = false;
    for (let i = 0; i < cols.length; i++) {
      const b = cols[i];
      if (y1 <= b.min.y + EPS || y0 >= b.max.y - EPS) continue;
      const gx = p.x < b.min.x ? b.min.x - p.x : (p.x > b.max.x ? p.x - b.max.x : 0);
      if (gx >= RADIUS) continue;
      const ez = Math.sqrt(RADIUS * RADIUS - gx * gx);
      if (p.z <= b.min.z - ez || p.z >= b.max.z + ez) continue;
      p.z = dz > 0 ? b.min.z - ez - EPS : b.max.z + ez + EPS;
      hit = true;
    }
    return hit;
  }

  /** Returns 0 clear, 1 landed on a top face, 2 hit a ceiling. */
  _sweepY(p, dy, h) {
    p.y += dy;
    const cols = this._colliders;
    let result = 0;
    for (let i = 0; i < cols.length; i++) {
      const b = cols[i];
      const dx = p.x - clamp(p.x, b.min.x, b.max.x);
      const dz = p.z - clamp(p.z, b.min.z, b.max.z);
      if (dx * dx + dz * dz >= RADIUS * RADIUS) continue;
      if (p.y >= b.max.y - EPS || p.y + h <= b.min.y + EPS) continue;
      // Falling normally means landing on the top face, but if the shallower fix
      // is downward it is the head that is inside — a spawn under an overhang
      // must not be flung onto the roof.
      const up = b.max.y - p.y;
      const down = p.y - (b.min.y - h);
      if (dy > 0 || (dy <= 0 && down < up)) { p.y = b.min.y - h; result = 2; }
      else { p.y = b.max.y; result = 1; }
    }
    return result;
  }

  /** Highest surface under the cylinder within `reach` below the feet. */
  _groundBelow(x, y, z, reach) {
    const cols = this._colliders;
    let best = -Infinity;
    for (let i = 0; i < cols.length; i++) {
      const b = cols[i];
      if (b.max.y > y + EPS || b.max.y < y - reach) continue;
      const dx = x - clamp(x, b.min.x, b.max.x);
      const dz = z - clamp(z, b.min.z, b.max.z);
      if (dx * dx + dz * dz >= RADIUS * RADIUS) continue;
      if (b.max.y > best) best = b.max.y;
    }
    return best;
  }

  /* --------------------------------------------------------------- input */
  _readLook() {
    const look = this.input.look;
    // update() runs several times per rendered frame; look is a per-frame delta,
    // so consuming it without zeroing would multiply sensitivity by the substep
    // count and make aim speed depend on frame rate.
    const scale = this.input.down('ads') ? (this.input.adsSensScale ?? 1) : 1;
    this.yaw += look.x * scale;
    this.pitch += look.y * scale;
    look.x = 0; look.y = 0;

    if (this.yaw > Math.PI) this.yaw -= Math.PI * 2;
    else if (this.yaw < -Math.PI) this.yaw += Math.PI * 2;
    this.pitch = clamp(this.pitch, -PITCH_LIMIT, PITCH_LIMIT);

    this.forward.set(-Math.sin(this.yaw), 0, -Math.cos(this.yaw));
  }

  /* ---------------------------------------------------------- simulation */
  update(dt) {
    this._readLook();

    if (this.dead) { this._applyCamera(dt); return; }
    if (this.mantling) { this._updateMantle(dt); this._applyCamera(dt); return; }

    const input = this.input;
    const mx = input.move.x, my = input.move.y;
    const ads = input.down('ads');
    const firing = input.down('fire');

    /* ------------------------------------------------------------ stance */
    this._slideCooldown = Math.max(0, this._slideCooldown - dt);
    const crouchHeld = input.down('crouch');

    if (this.sliding) {
      this._slideTimer -= dt;
      const horiz = Math.hypot(this.velocity.x, this.velocity.z);
      if (this._slideTimer <= 0 || horiz < 1.9 || !this.isGrounded ||
          (!crouchHeld && this._slideTimer < SLIDE_TIME - 0.25)) {
        this.sliding = false;
        this._slideCooldown = SLIDE_COOLDOWN;
        this.crouching = crouchHeld;
      }
    } else if (input.justPressed('crouch') && this.sprinting && this.isGrounded &&
               this._slideCooldown <= 0 && Math.hypot(this.velocity.x, this.velocity.z) > 4.6) {
      this._startSlide(mx);
    } else {
      this.crouching = crouchHeld;
    }

    // Standing back up is only allowed with headroom, otherwise you would pop
    // through the crate you just slid under.
    if (!this.crouching && !this.sliding &&
        !this._capsuleFree(this.position.x, this.position.y, this.position.z, STAND_HEIGHT)) {
      this.crouching = true;
    }

    const wantHeight = (this.crouching || this.sliding) ? CROUCH_HEIGHT : STAND_HEIGHT;
    this._height += (wantHeight - this._height) * damp(14, dt);

    /* ------------------------------------------------------------ sprint */
    const sprintWanted = input.down('sprint') && my > 0.5 && !ads && !firing &&
                         !this.crouching && !this.sliding && this.isGrounded;
    if (sprintWanted) {
      this._sprintRamp = Math.min(1, this._sprintRamp + dt / SPRINT_RAMP);
    } else {
      if (this._sprintRamp > 0.35) this._sprintOut = SPRINT_OUT;
      this._sprintRamp = Math.max(0, this._sprintRamp - dt / 0.15);
    }
    this.sprinting = sprintWanted && this._sprintRamp > 0.05;
    this._sprintOut = Math.max(0, this._sprintOut - dt);
    this.sprintOutTimer = this._sprintOut;
    this.canFire = !this.sprinting && this._sprintOut <= 0 && !this.mantling;

    /* ------------------------------------------------------- wish vector */
    const right = { x: Math.cos(this.yaw), z: -Math.sin(this.yaw) };
    const wish = this._wish.set(
      this.forward.x * my + right.x * mx, 0,
      this.forward.z * my + right.z * mx);
    const wishLen = Math.hypot(wish.x, wish.z);
    if (wishLen > 1e-4) { wish.x /= wishLen; wish.z /= wishLen; }

    let wishSpeed = SPEED_WALK;
    if (this.crouching) wishSpeed = SPEED_CROUCH;
    else if (ads) wishSpeed = SPEED_ADS;
    else if (this._sprintRamp > 0) wishSpeed = SPEED_WALK + (SPEED_SPRINT - SPEED_WALK) * this._sprintRamp;
    wishSpeed *= Math.min(1, wishLen);

    /* ------------------------------------------------------------ ground */
    if (this.sliding) {
      this._slideMove(dt, wish, wishLen);
    } else {
      const before = Math.hypot(this.velocity.x, this.velocity.z);
      if (this.isGrounded) {
        this._friction(dt);
        this._accelerate(dt, wish, wishSpeed, GROUND_ACCEL);
      } else {
        this._accelerate(dt, wish, Math.min(wishSpeed, SPEED_WALK), AIR_ACCEL);
      }
      // Accelerate only limits the component along the wish direction, so running
      // diagonally into a wall (which zeroes one axis every step) would otherwise
      // pump speed up by 1.41x — the Quake wall-strafe. Momentum already earned
      // (a slide, an explosion) is preserved and left to friction.
      const after = Math.hypot(this.velocity.x, this.velocity.z);
      const cap = Math.max(wishSpeed, before);
      if (after > cap + 1e-4) {
        const s = cap / after;
        this.velocity.x *= s;
        this.velocity.z *= s;
      }
    }

    /* -------------------------------------------------------------- jump */
    if (input.justPressed('jump')) this._jumpBuffer = JUMP_BUFFER;
    this._jumpBuffer = Math.max(0, this._jumpBuffer - dt);
    this._coyote = this.isGrounded ? COYOTE_TIME : Math.max(0, this._coyote - dt);

    let jumped = false;
    if (this._jumpBuffer > 0 && this._coyote > 0 &&
        this._capsuleFree(this.position.x, this.position.y + 0.05, this.position.z, this._height)) {
      this.velocity.y = JUMP_SPEED;
      this._jumpBuffer = 0;
      this._coyote = 0;
      this.isGrounded = false;
      jumped = true;
      if (this.sliding) { this.sliding = false; this._slideCooldown = SLIDE_COOLDOWN; }
    }

    this.velocity.y += GRAVITY * dt;

    /* ----------------------------------------------------------- integrate */
    const wasGrounded = this.isGrounded;
    const impact = -this.velocity.y;
    this._moveHorizontal(dt);

    const p = this.position;
    const land = this._sweepY(p, this.velocity.y * dt, this._height);
    if (land === 1) {
      if (!wasGrounded) this._land(impact);
      this.velocity.y = 0;
      this.isGrounded = true;
    } else if (land === 2) {
      this.velocity.y = Math.min(0, this.velocity.y);
      this.isGrounded = false;
    } else {
      this.isGrounded = false;
      // Walking off a stair lip should not launch you: reattach to a surface
      // that is within a step of the feet as long as we are not rising.
      if (wasGrounded && !jumped && this.velocity.y <= 0) {
        const g = this._groundBelow(p.x, p.y, p.z, STEP_HEIGHT * 0.9);
        if (g > -Infinity) { p.y = g; this.velocity.y = 0; this.isGrounded = true; }
      }
    }

    this._clampToBounds();
    this._tryMantle(my);

    this.speed = Math.hypot(this.velocity.x, this.velocity.z);
    this._regen(dt);
    this._applyCamera(dt);
  }

  _accelerate(dt, wish, wishSpeed, accel) {
    if (wishSpeed <= 0) return;
    const current = this.velocity.x * wish.x + this.velocity.z * wish.z;
    const add = wishSpeed - current;
    if (add <= 0) return;
    const step = Math.min(accel * wishSpeed * dt, add);
    this.velocity.x += wish.x * step;
    this.velocity.z += wish.z * step;
  }

  _friction(dt) {
    const speed = Math.hypot(this.velocity.x, this.velocity.z);
    if (speed < 1e-4) { this.velocity.x = this.velocity.z = 0; return; }
    // Below STOP_SPEED friction is applied at a constant rate, which is what
    // actually brings you to a dead stop instead of an infinite crawl.
    const control = Math.max(speed, STOP_SPEED);
    const drop = control * FRICTION * dt;
    const scale = Math.max(0, speed - drop) / speed;
    this.velocity.x *= scale;
    this.velocity.z *= scale;
  }

  _startSlide(strafe) {
    this.sliding = true;
    this.crouching = true;
    this._slideTimer = SLIDE_TIME;
    this._slideRollSign = strafe > 0.2 ? 1 : (strafe < -0.2 ? -1 : 1);

    const speed = Math.hypot(this.velocity.x, this.velocity.z);
    const boosted = Math.min(8.6, Math.max(speed * 1.22, 7.8));
    const s = boosted / Math.max(speed, 1e-4);
    this.velocity.x *= s;
    this.velocity.z *= s;
    this._sprintRamp = 0;
    this._sprintOut = 0;    // sliding out of a sprint is a firing position in CoD
  }

  _slideMove(dt, wish, wishLen) {
    const v = this.velocity;
    let speed = Math.hypot(v.x, v.z);
    if (speed < 1e-4) return;

    // Steering is a bounded rotation of the existing momentum, never an
    // acceleration — that is what keeps a slide committed rather than steerable
    // like a walk.
    if (wishLen > 0.2) {
      const cur = Math.atan2(v.z, v.x);
      const want = Math.atan2(wish.z, wish.x);
      let diff = want - cur;
      while (diff > Math.PI) diff -= Math.PI * 2;
      while (diff < -Math.PI) diff += Math.PI * 2;
      const turn = clamp(diff, -SLIDE_STEER * dt, SLIDE_STEER * dt);
      const a = cur + turn;
      v.x = Math.cos(a) * speed;
      v.z = Math.sin(a) * speed;
    }

    speed *= Math.exp(-SLIDE_DECAY * dt);
    const n = speed / Math.hypot(v.x, v.z);
    v.x *= n;
    v.z *= n;
  }

  _moveHorizontal(dt) {
    const p = this.position;
    const dx = this.velocity.x * dt;
    const dz = this.velocity.z * dt;
    this._blockedHoriz = false;
    if (dx === 0 && dz === 0) return;

    const x0 = p.x, z0 = p.z;
    const hitX = this._sweepX(p, dx, this._height);
    const hitZ = this._sweepZ(p, dz, this._height);

    if (hitX) this.velocity.x = 0;
    if (hitZ) this.velocity.z = 0;
    this._blockedHoriz = hitX || hitZ;

    if (!this._blockedHoriz || !this.isGrounded) return;

    // Step-up: replay the same move from a raised position and drop back down.
    // Cheaper and more predictable than a real capsule sweep, and it never
    // climbs anything the head cannot clear.
    const probe = this._probe;
    probe.x = x0; probe.y = p.y + STEP_HEIGHT; probe.z = z0;
    if (!this._capsuleFree(probe.x, probe.y, probe.z, this._height)) return;

    this._sweepX(probe, dx, this._height);
    this._sweepZ(probe, dz, this._height);

    const gained = (probe.x - x0) ** 2 + (probe.z - z0) ** 2;
    const had = (p.x - x0) ** 2 + (p.z - z0) ** 2;
    if (gained <= had + 1e-4) return;

    const ground = this._groundBelow(probe.x, probe.y, probe.z, STEP_HEIGHT + 0.05);
    if (ground === -Infinity) return;
    p.x = probe.x;
    p.z = probe.z;
    p.y = ground;
    if (hitX) this.velocity.x = dx / dt;
    if (hitZ) this.velocity.z = dz / dt;
  }

  _clampToBounds() {
    const b = this.level?.bounds;
    if (!b) return;
    const p = this.position;
    p.x = clamp(p.x, b.min.x + RADIUS, b.max.x - RADIUS);
    p.z = clamp(p.z, b.min.z + RADIUS, b.max.z - RADIUS);
    // A missing floor collider must not drop the player out of the world.
    if (p.y < b.min.y) { p.y = b.min.y; this.velocity.y = 0; this.isGrounded = true; }
  }

  /* ------------------------------------------------------------- mantle */
  _tryMantle(forwardInput) {
    if (this.mantling || this.sliding) return;
    if (!this._blockedHoriz || forwardInput < 0.4) return;
    if (!this.sprinting && this.isGrounded) return;   // walk-into-wall must not vault

    const p = this.position;
    const fx = this.forward.x, fz = this.forward.z;
    const px = p.x + fx * (RADIUS + 0.35);
    const pz = p.z + fz * (RADIUS + 0.35);

    const cols = this._colliders;
    let top = -Infinity;
    for (let i = 0; i < cols.length; i++) {
      const b = cols[i];
      if (px < b.min.x - 0.05 || px > b.max.x + 0.05) continue;
      if (pz < b.min.z - 0.05 || pz > b.max.z + 0.05) continue;
      const h = b.max.y - p.y;
      if (h < MANTLE_MIN || h > MANTLE_MAX) continue;
      if (b.max.y > top) top = b.max.y;
    }
    if (top === -Infinity) return;
    if (!this._capsuleFree(px, top + 0.02, pz, CROUCH_HEIGHT)) return;

    this.mantling = true;
    this.canFire = false;
    this._mantleT = 0;
    this._mantleFrom.copy(p);
    this._mantleTo.set(px + fx * 0.25, top + 0.02, pz + fz * 0.25);
    this.velocity.set(0, 0, 0);
    this.sprinting = false;
    this._sprintRamp = 0;
  }

  _updateMantle(dt) {
    this._mantleT += dt / MANTLE_TIME;
    const t = Math.min(1, this._mantleT);
    // Rise first, translate second: going up-then-over reads as pulling yourself
    // onto the ledge. A straight lerp looks like floating through the corner.
    const yF = smoothstep(0, 0.62, t);
    const xzF = smoothstep(0.30, 1, t);
    const a = this._mantleFrom, b = this._mantleTo;
    this.position.set(
      a.x + (b.x - a.x) * xzF,
      a.y + (b.y - a.y) * yF,
      a.z + (b.z - a.z) * xzF);

    this.speed = 0;
    if (t >= 1) {
      this.mantling = false;
      this.isGrounded = true;
      this._sprintOut = SPRINT_OUT;
      this._coyote = COYOTE_TIME;
    }
  }

  /* -------------------------------------------------------- damage/health */
  _land(impact) {
    this._landVel -= Math.min(1, impact / 12) * 2.6;
    if (impact > FALL_SAFE) {
      const t = clamp((impact - FALL_SAFE) / (FALL_LETHAL - FALL_SAFE), 0, 1);
      this.takeDamage(Math.min(100, 100 * Math.pow(t, 1.4)), null);
    }
  }

  _regen(dt) {
    this._sinceDamage += dt;
    if (this.dead || this.health >= this.maxHealth) return;
    if (this._sinceDamage < REGEN_DELAY) return;
    this.health = Math.min(this.maxHealth, this.health + (this.maxHealth / REGEN_TIME) * dt);
  }

  takeDamage(amount, fromPosition) {
    if (this.dead || amount <= 0) return;
    this.health -= amount;
    this._sinceDamage = 0;

    const dir = this._damageDir;
    if (fromPosition) dir.copy(fromPosition).sub(this.eyePosition).normalize();
    else dir.copy(this.forward);
    this.onDamage?.(amount, dir);

    if (this.health <= 0) {
      this.health = 0;
      this.dead = true;
      this.velocity.set(0, 0, 0);
      this.onDeath?.();
    }
  }

  /* -------------------------------------------------------------- camera */
  _leanClearance(sign) {
    const rx = Math.cos(this.yaw) * sign, rz = -Math.sin(this.yaw) * sign;
    const p = this.position;
    const y = p.y + this._eyeHeight;
    const cols = this._colliders;
    let clear = 1;
    for (let s = 1; s <= 3; s++) {
      const f = s / 3;
      const x = p.x + rx * (LEAN_OFFSET * f + 0.12);
      const z = p.z + rz * (LEAN_OFFSET * f + 0.12);
      let blocked = false;
      for (let i = 0; i < cols.length; i++) {
        const b = cols[i];
        if (y < b.min.y || y > b.max.y) continue;
        if (x > b.min.x - 0.05 && x < b.max.x + 0.05 &&
            z > b.min.z - 0.05 && z < b.max.z + 0.05) { blocked = true; break; }
      }
      if (blocked) { clear = (s - 1) / 3; break; }
    }
    return clear;
  }

  _applyCamera(dt) {
    const p = this.position;
    const input = this.input;
    const ads = input?.down('ads');

    const targetEye = this._height * EYE_RATIO;
    this._eyeHeight += (targetEye - this._eyeHeight) * damp(16, dt);

    /* ------------------------------------------------------------- bob */
    // Driven by distance travelled, not time, so the head stops dead the instant
    // you do and the cadence always matches the actual stride.
    const horiz = Math.hypot(this.velocity.x, this.velocity.z);
    this._bobDist += horiz * dt;
    const phase = this._bobDist * (Math.PI * 2 / 1.85);   // one cycle per two strides
    let ampTarget = clamp(horiz / SPEED_WALK, 0, 1.4) * 0.042;
    if (ads) ampTarget *= 0.5;
    if (this.crouching || this.sliding) ampTarget *= 0.55;
    if (!this.isGrounded) ampTarget = 0;
    this._bobAmp += (ampTarget - this._bobAmp) * damp(12, dt);
    this._bobX = Math.sin(phase) * this._bobAmp;
    this._bobY = Math.sin(phase * 2) * this._bobAmp * 0.6;   // figure eight

    /* ------------------------------------------------------------ lean */
    const wantLean = (input?.down('lean_l') ? -1 : 0) + (input?.down('lean_r') ? 1 : 0);
    let leanTarget = this.sprinting || this.sliding || this.mantling ? 0 : wantLean;
    if (leanTarget !== 0) leanTarget *= this._leanClearance(Math.sign(leanTarget));
    this._lean += (leanTarget - this._lean) * damp(9, dt);

    /* ------------------------------------------------------- land / roll */
    // Critically damped spring: the dip settles once instead of oscillating.
    this._landVel += (-this._landDip * 170 - this._landVel * 24) * dt;
    this._landDip += this._landVel * dt;

    const strafe = input ? input.move.x : 0;
    let rollTarget = -strafe * STRAFE_ROLL - this._lean * LEAN_ROLL;
    if (this.sliding) rollTarget += this._slideRollSign * 4.5 * Math.PI / 180;
    this._roll += (rollTarget - this._roll) * damp(10, dt);

    const slideDip = this.sliding ? -0.20 * smoothstep(0, 0.12, SLIDE_TIME - this._slideTimer) : 0;

    const rx = Math.cos(this.yaw), rz = -Math.sin(this.yaw);
    const lateral = this._lean * LEAN_OFFSET + this._bobX;
    this.eyeOffset.set(
      rx * lateral, this._eyeHeight + this._bobY + this._landDip + slideDip, rz * lateral);

    const cam = this.camera;
    cam.position.set(p.x + this.eyeOffset.x, p.y + this.eyeOffset.y, p.z + this.eyeOffset.z);
    cam.rotation.set(this.pitch, this.yaw, this._roll);
    this.eyePosition.set(p.x, p.y + this._eyeHeight, p.z);
  }
}

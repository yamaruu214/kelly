// Enemy.js — procedural soldiers: skeleton, math-driven animation, tactical AI.
//
// There are no assets, so the enemy is a bone hierarchy of Object3Ds with simple
// beveled solids parented to it. Every joint has to move independently, so the
// parts cannot be instanced across enemies — instancing would still need a
// per-bone matrix write every frame plus a separate InstancedMesh per body part,
// which is more draw calls than the single shared-material Group we use here.
// What IS shared is the expensive half: one geometry set and one material set
// for the whole population, built once in EnemyAssets.
//
// Animation is entirely arithmetic. The locomotion phase advances with distance
// travelled rather than time, so a soldier slowed by a corner does not moonwalk,
// and the upper body is driven independently of the legs — a body locked to its
// facing is the single loudest tell of a cheap bot.
//
// Death hands the same bone hierarchy to a verlet solver: joints become
// particles, bones become distance constraints, and the corpse falls against the
// ground and the level AABBs. No physics engine, ~15 particles per body.

import * as THREE from 'three';
import { makeStandardMaterial } from '../world/Textures.js';

const UP = new THREE.Vector3(0, 1, 0);
const TAU = Math.PI * 2;

const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const lerp = (a, b, t) => a + (b - a) * t;
const smoothstep = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
/** Shortest signed distance between two angles — keeps turns from going the long way. */
const angleDelta = (a, b) => { let d = (b - a) % TAU; if (d > Math.PI) d -= TAU; if (d < -Math.PI) d += TAU; return d; };
/** Frame-rate independent exponential approach; `rate` is roughly 1/seconds. */
const approach = (cur, target, rate, dt) => cur + (target - cur) * (1 - Math.exp(-rate * dt));

/* Deterministic per-enemy noise — Math.random would make the idle sway jitter
   every frame instead of drifting. */
const hash1 = (n) => { const s = Math.sin(n * 127.1) * 43758.5453; return s - Math.floor(s); };
const snoise = (t, seed) => Math.sin(t * 0.7 + seed * 6.28) * 0.6 + Math.sin(t * 1.9 + seed * 12.9) * 0.4;

/* ------------------------------------------------------------ input coercion */
// Level.js and Controller.js are owned elsewhere and may hand us Vector3s, plain
// objects, arrays or wrappers. Everything crossing the boundary is normalised
// once here so the hot loops can assume Vector3.

function toVec3(p, out = new THREE.Vector3()) {
  if (!p) return null;
  if (p.isVector3) return out.copy(p);
  if (Array.isArray(p)) return out.set(p[0] || 0, p[1] || 0, p[2] || 0);
  if (p.position) return toVec3(p.position, out);
  if (p.pos) return toVec3(p.pos, out);
  if (typeof p.x === 'number') return out.set(p.x, p.y || 0, p.z || 0);
  return null;
}

function toBox3(c) {
  if (!c) return null;
  if (c.isBox3) return c;
  const src = c.box || c.aabb || c;
  const min = toVec3(src.min), max = toVec3(src.max);
  if (min && max) return new THREE.Box3(min, max);
  // {center,size} and {x,y,z,w,h,d} forms both show up in hand-built levels.
  const centre = toVec3(src.center || src.centre);
  const size = toVec3(src.size || src.half);
  if (centre && size) {
    const h = src.half ? size : size.multiplyScalar(0.5);
    return new THREE.Box3(centre.clone().sub(h), centre.clone().add(h));
  }
  return null;
}

/* ------------------------------------------------------------------ geometry */

/**
 * Box with rounded corners. Real gear has no razor edges, and the chamfer is
 * what catches the rim light that makes the silhouette legible against the sky.
 */
function beveledBox(w, h, d, r, seg = 2) {
  const g = new THREE.BoxGeometry(w, h, d, seg, seg, seg);
  const pos = g.attributes.position;
  const hx = Math.max(0, w / 2 - r), hy = Math.max(0, h / 2 - r), hz = Math.max(0, d / 2 - r);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const cx = clamp(x, -hx, hx), cy = clamp(y, -hy, hy), cz = clamp(z, -hz, hz);
    const dx = x - cx, dy = y - cy, dz = z - cz;
    const len = Math.hypot(dx, dy, dz) || 1;
    pos.setXYZ(i, cx + dx / len * r, cy + dy / len * r, cz + dz / len * r);
  }
  g.computeVertexNormals();
  return g;
}

/**
 * One geometry + material set for the entire enemy population. Sixteen soldiers
 * on ULTRA share these, so the memory cost is a single body's worth.
 */
class EnemyAssets {
  constructor(textures, settings) {
    this.settings = settings;
    const cap = (r, l) => new THREE.CapsuleGeometry(r, l, 3, 8);

    this.geo = {
      hips:     beveledBox(0.30, 0.20, 0.21, 0.05),
      belt:     beveledBox(0.31, 0.06, 0.22, 0.025),
      torso:    beveledBox(0.35, 0.36, 0.22, 0.07),
      carrier:  beveledBox(0.38, 0.30, 0.27, 0.045),
      pouch:    beveledBox(0.10, 0.11, 0.07, 0.02),
      neck:     cap(0.055, 0.06),
      head:     beveledBox(0.165, 0.215, 0.19, 0.075),
      helmet:   beveledBox(0.245, 0.17, 0.265, 0.08),
      upperArm: cap(0.055, 0.17),
      foreArm:  cap(0.048, 0.16),
      hand:     beveledBox(0.075, 0.10, 0.085, 0.03),
      thigh:    cap(0.078, 0.27),
      shin:     cap(0.062, 0.28),
      boot:     beveledBox(0.115, 0.115, 0.27, 0.04),
      rifle:    beveledBox(0.055, 0.085, 0.62, 0.02),
      mag:      beveledBox(0.045, 0.16, 0.07, 0.018),
      barrel:   cap(0.017, 0.30),
    };

    // Tiling is per-material rather than per-mesh, so the weave on a 10cm pouch
    // reads at the same physical scale as the weave on the torso.
    this.mat = {
      fabric:   makeStandardMaterial(textures, 'fabric', { repeat: 3 }),
      fabricT:  makeStandardMaterial(textures, 'fabric', { repeat: 4, color: 0xa08a63 }),
      plate:    makeStandardMaterial(textures, 'metalOlive', { repeat: 2 }),
      gun:      makeStandardMaterial(textures, 'gunmetal', { repeat: 2 }),
    };
    for (const m of Object.values(this.mat)) m.shadowSide = THREE.FrontSide;
  }

  dispose() {
    for (const g of Object.values(this.geo)) g.dispose();
    for (const m of Object.values(this.mat)) {
      m.map?.dispose(); m.normalMap?.dispose(); m.ormMap?.dispose(); m.dispose();
    }
  }
}

/* ------------------------------------------------------------------ skeleton */
// `axis` is the direction the bone points at rest, which is what the ragdoll
// aligns to a limb vector. Spine bones run +Y, limbs hang -Y.

const SKELETON = [
  { name: 'pelvis', parent: null,     pos: [0, 0.95, 0],     axis: [0, 1, 0] },
  { name: 'spine',  parent: 'pelvis', pos: [0, 0.16, 0],     axis: [0, 1, 0] },
  { name: 'chest',  parent: 'spine',  pos: [0, 0.20, 0],     axis: [0, 1, 0] },
  { name: 'neck',   parent: 'chest',  pos: [0, 0.20, 0],     axis: [0, 1, 0] },
  { name: 'head',   parent: 'neck',   pos: [0, 0.07, 0],     axis: [0, 1, 0] },

  { name: 'armL',   parent: 'chest',  pos: [ 0.20, 0.15, 0], axis: [0, -1, 0] },
  { name: 'foreL',  parent: 'armL',   pos: [0, -0.28, 0],    axis: [0, -1, 0] },
  { name: 'handL',  parent: 'foreL',  pos: [0, -0.26, 0],    axis: [0, -1, 0] },
  { name: 'armR',   parent: 'chest',  pos: [-0.20, 0.15, 0], axis: [0, -1, 0] },
  { name: 'foreR',  parent: 'armR',   pos: [0, -0.28, 0],    axis: [0, -1, 0] },
  { name: 'handR',  parent: 'foreR',  pos: [0, -0.26, 0],    axis: [0, -1, 0] },

  { name: 'thighL', parent: 'pelvis', pos: [ 0.11, -0.06, 0], axis: [0, -1, 0] },
  { name: 'shinL',  parent: 'thighL', pos: [0, -0.44, 0],     axis: [0, -1, 0] },
  { name: 'footL',  parent: 'shinL',  pos: [0, -0.42, 0],     axis: [0, -1, 0] },
  { name: 'thighR', parent: 'pelvis', pos: [-0.11, -0.06, 0], axis: [0, -1, 0] },
  { name: 'shinR',  parent: 'thighR', pos: [0, -0.44, 0],     axis: [0, -1, 0] },
  { name: 'footR',  parent: 'shinR',  pos: [0, -0.42, 0],     axis: [0, -1, 0] },
];

/* Weapon-ready rest pose. Forward is +Z, so a negative X rotation swings a
   hanging limb forward. Everything the animator does is added on top of this. */
const REST = {
  armL:  [-1.30,  0.10,  0.34],
  foreL: [-1.05,  0.30, -0.10],
  handL: [ 0.20,  0.00,  0.00],
  armR:  [-0.92, -0.16, -0.30],
  foreR: [-1.18, -0.55,  0.05],
  handR: [ 0.10,  0.00,  0.00],
  chest: [ 0.05,  0.00,  0.00],
  spine: [ 0.04,  0.00,  0.00],
  head:  [ 0.02,  0.00,  0.00],
};

const EYE_LOCAL = new THREE.Vector3(0, 1.60, 0.06);

/* ------------------------------------------------------------------ ragdoll */
// Particle indices. Shoulders and hips are particles rather than derived points
// so the torso stays a rigid braced shell instead of folding in half.
const P_PELVIS = 0, P_CHEST = 1, P_HEAD = 2,
      P_SHL = 3, P_ELL = 4, P_HAL = 5,
      P_SHR = 6, P_ELR = 7, P_HAR = 8,
      P_HIL = 9, P_KNL = 10, P_FTL = 11,
      P_HIR = 12, P_KNR = 13, P_FTR = 14;
const PARTICLE_COUNT = 15;

const RAGDOLL_LINKS = [
  [P_PELVIS, P_CHEST, 1], [P_CHEST, P_HEAD, 1],
  [P_CHEST, P_SHL, 1], [P_CHEST, P_SHR, 1], [P_SHL, P_SHR, 1],
  [P_PELVIS, P_SHL, 0.7], [P_PELVIS, P_SHR, 0.7],
  [P_SHL, P_ELL, 1], [P_ELL, P_HAL, 1],
  [P_SHR, P_ELR, 1], [P_ELR, P_HAR, 1],
  [P_PELVIS, P_HIL, 1], [P_PELVIS, P_HIR, 1], [P_HIL, P_HIR, 1],
  [P_CHEST, P_HIL, 0.6], [P_CHEST, P_HIR, 0.6],
  [P_HIL, P_KNL, 1], [P_KNL, P_FTL, 1],
  [P_HIR, P_KNR, 1], [P_KNR, P_FTR, 1],
  // The hands are wired together because the corpse is still holding a rifle.
  [P_HAL, P_HAR, 0.35],
];

/* Which bone tracks which limb vector once the body goes limp. */
const RAGDOLL_BONES = [
  ['pelvis', P_PELVIS, P_CHEST], ['chest', P_CHEST, P_HEAD],
  ['armL', P_SHL, P_ELL], ['foreL', P_ELL, P_HAL],
  ['armR', P_SHR, P_ELR], ['foreR', P_ELR, P_HAR],
  ['thighL', P_HIL, P_KNL], ['shinL', P_KNL, P_FTL],
  ['thighR', P_HIR, P_KNR], ['shinR', P_KNR, P_FTR],
];

/* ------------------------------------------------------------------- states */
const S = {
  IDLE: 'IDLE', PATROL: 'PATROL', ALERT: 'ALERT', SEEK: 'SEEK',
  COMBAT: 'COMBAT', FLANK: 'FLANK', REPOSITION: 'REPOSITION',
  SUPPRESSED: 'SUPPRESSED', DEAD: 'DEAD',
};

const VISION_RANGE = 70;
const VISION_COS = Math.cos(60 * Math.PI / 180);   // 120° total cone
const HEARING_RANGE = 45;

/* --------------------------------------------------------------------- temps */
const _v0 = new THREE.Vector3(), _v1 = new THREE.Vector3(), _v2 = new THREE.Vector3(),
      _v3 = new THREE.Vector3(), _v4 = new THREE.Vector3(), _v5 = new THREE.Vector3();
const _q0 = new THREE.Quaternion(), _q1 = new THREE.Quaternion();
const _e0 = new THREE.Euler();

/* ===========================================================================
   Enemy
   =========================================================================== */

class Enemy {
  constructor(manager, index) {
    this.mgr = manager;
    this.index = index;
    this.seed = hash1(index * 3.77 + 1.3);

    this.position = new THREE.Vector3();     // FEET, matching the player contract
    this.velocity = new THREE.Vector3();
    this.alive = false;
    this.health = 100;
    this.maxHealth = 100;
    this.state = S.IDLE;

    this.group = new THREE.Group();
    this.group.visible = false;
    this.bones = {};
    this._buildBody(manager.assets);

    /* motion */
    this.yaw = 0;
    this.aimYaw = 0;
    this.aimPitch = 0;
    this.stride = 0;          // locomotion phase, advanced by distance
    this.speed = 0;
    this.crouch = 0;
    this.lean = 0;

    /* perception */
    this.canSeePlayer = false;
    this.losTime = 0;         // continuous seconds of LOS — drives accuracy
    this.lastSeenPos = new THREE.Vector3();
    this.timeSinceSeen = 999;
    this.alertLevel = 0;

    /* combat */
    this.burstLeft = 0;
    this.fireCd = 0;
    this.burstCd = 0;
    this.reactionCd = 0;
    this.firstBurst = true;   // the deliberate opening miss
    this.suppression = 0;
    this.stateTime = 0;

    /* navigation */
    this.path = [];
    this.pathIndex = 0;
    this.repathCd = 0;
    this.goal = new THREE.Vector3();
    this.hasGoal = false;
    this.coverNode = -1;
    this.peeking = 0;
    this.isFlanker = false;

    /* animation extras */
    this.flinch = new THREE.Vector3();
    this.flinchDecay = 0;
    this.muzzle = new THREE.Object3D();
    this.bones.handR.add(this.muzzle);
    this.muzzle.position.set(0.02, -0.06, 0.72);

    this.ragdoll = null;
    this.corpseAge = 0;
    this.sinking = 0;
  }

  /* ---------------------------------------------------------------- assembly */

  _buildBody(assets) {
    const { geo, mat } = assets;
    const cast = this.mgr.settings.shadows;

    for (const spec of SKELETON) {
      const b = new THREE.Object3D();
      b.position.fromArray(spec.pos);
      b.userData.axis = new THREE.Vector3().fromArray(spec.axis);
      b.userData.rest = spec.pos.slice();
      const r = REST[spec.name];
      if (r) b.rotation.set(r[0], r[1], r[2]);
      b.userData.restRot = b.rotation.clone();
      (spec.parent ? this.bones[spec.parent] : this.group).add(b);
      this.bones[spec.name] = b;
    }

    const part = (boneName, geometry, material, px, py, pz, rx = 0, ry = 0, rz = 0) => {
      const m = new THREE.Mesh(geometry, material);
      m.position.set(px, py, pz);
      m.rotation.set(rx, ry, rz);
      m.castShadow = cast;
      m.matrixAutoUpdate = false;
      m.updateMatrix();
      this.bones[boneName].add(m);
      return m;
    };

    part('pelvis', geo.hips, mat.fabric, 0, -0.02, 0);
    part('pelvis', geo.belt, mat.fabricT, 0, 0.08, 0);
    part('spine', geo.torso, mat.fabric, 0, 0.10, 0);
    part('chest', geo.carrier, mat.plate, 0, -0.02, 0.005);
    part('chest', geo.pouch, mat.fabricT, 0.11, -0.09, 0.14);
    part('chest', geo.pouch, mat.fabricT, -0.02, -0.09, 0.145);
    part('chest', geo.pouch, mat.fabricT, -0.14, -0.05, 0.13, 0, 0, 0.2);
    part('neck', geo.neck, mat.fabric, 0, 0.02, 0);
    part('head', geo.head, mat.fabric, 0, 0.07, 0.005);
    part('helmet', undefined, undefined, 0, 0, 0);   // placeholder removed below

    // The helmet rides the head bone; there is no separate helmet bone.
    this.bones.head.remove(this.bones.head.children[this.bones.head.children.length - 1]);
    part('head', geo.helmet, mat.plate, 0, 0.145, -0.012);

    for (const s of [['L', 1], ['R', -1]]) {
      const [k, sx] = s;
      part('arm' + k, geo.upperArm, mat.fabric, 0, -0.14, 0);
      part('fore' + k, geo.foreArm, mat.fabric, 0, -0.13, 0);
      part('hand' + k, geo.hand, mat.fabricT, 0, -0.05, 0.01);
      part('thigh' + k, geo.thigh, mat.fabric, 0, -0.22, 0);
      part('shin' + k, geo.shin, mat.fabric, 0, -0.21, 0);
      part('foot' + k, geo.boot, mat.gun, sx * 0.005, -0.05, 0.055);
    }

    // Rifle hangs off the trigger hand so it inherits every arm motion for free.
    const rifle = new THREE.Group();
    rifle.position.set(0.02, -0.05, 0.12);
    rifle.rotation.set(0.06, 0, 0);
    this.bones.handR.add(rifle);
    for (const [g, m, p] of [
      [geo.rifle, mat.gun, [0, 0, 0.06]],
      [geo.mag, mat.gun, [0, -0.10, -0.02]],
      [geo.barrel, mat.gun, [0, 0.012, 0.50]],
    ]) {
      const mesh = new THREE.Mesh(g, m);
      mesh.position.fromArray(p);
      if (g === geo.barrel) mesh.rotation.x = Math.PI / 2;
      mesh.castShadow = cast;
      mesh.matrixAutoUpdate = false;
      mesh.updateMatrix();
      rifle.add(mesh);
    }
    this.rifle = rifle;
  }

  /* ----------------------------------------------------------------- spawning */

  spawn(pos, wave) {
    this.position.copy(pos);
    this.velocity.set(0, 0, 0);
    this.alive = true;

    // Wave scaling stays gentle on health and steep on competence: a bullet
    // sponge feels unfair, a soldier who flanks better does not.
    this.maxHealth = 100 + Math.min(90, (wave - 1) * 11);
    this.health = this.maxHealth;
    this.skill = clamp(0.30 + (wave - 1) * 0.075, 0.3, 0.95);

    this.state = S.IDLE;
    this.stateTime = 0;
    this.yaw = hash1(this.index * 9.1 + wave) * TAU;
    this.aimYaw = this.yaw;
    this.aimPitch = 0;
    this.stride = hash1(this.index * 5.5) * TAU;
    this.speed = 0;
    this.crouch = 0;
    this.lean = 0;
    this.canSeePlayer = false;
    this.losTime = 0;
    this.timeSinceSeen = 999;
    this.alertLevel = 0;
    this.burstLeft = 0;
    this.fireCd = 0;
    this.burstCd = 0.5 + hash1(this.index * 2.2) * 0.7;
    this.firstBurst = true;
    this.suppression = 0;
    this.path.length = 0;
    this.pathIndex = 0;
    this.repathCd = 0;
    this.hasGoal = false;
    this.coverNode = -1;
    this.peeking = 0;
    this.isFlanker = false;
    this.flinch.set(0, 0, 0);
    this.flinchDecay = 0;
    this.ragdoll = null;
    this.corpseAge = 0;
    this.sinking = 0;

    this.group.visible = true;
    this.group.scale.setScalar(1);
    this.group.position.copy(this.position);
    this.group.quaternion.identity();
    for (const spec of SKELETON) {
      const b = this.bones[spec.name];
      b.position.fromArray(spec.userData ? spec.pos : spec.pos);
      b.rotation.copy(b.userData.restRot);
    }
  }

  /* -------------------------------------------------------------- perception */

  eyePosition(out = _v0) {
    return out.set(
      this.position.x + Math.sin(this.yaw) * EYE_LOCAL.z,
      this.position.y + EYE_LOCAL.y - this.crouch * 0.42,
      this.position.z + Math.cos(this.yaw) * EYE_LOCAL.z);
  }

  _updatePerception(dt, playerEye) {
    const eye = this.eyePosition(_v0);
    const toPlayer = _v1.copy(playerEye).sub(eye);
    const dist = toPlayer.length();
    let sees = false;

    if (dist < VISION_RANGE && dist > 0.001) {
      const dir = _v2.copy(toPlayer).divideScalar(dist);
      const facing = _v3.set(Math.sin(this.aimYaw), 0, Math.cos(this.aimYaw));
      const flat = _v4.set(dir.x, 0, dir.z);
      const flatLen = flat.length() || 1;
      const cosang = flat.divideScalar(flatLen).dot(facing);
      // Very close contact bypasses the cone: you notice someone at your elbow.
      if (cosang > VISION_COS || dist < 4) {
        sees = this.mgr.hasLineOfSight(eye, playerEye, dist);
      }
    }

    this.canSeePlayer = sees;
    if (sees) {
      this.losTime += dt;
      this.timeSinceSeen = 0;
      this.lastSeenPos.copy(playerEye);
      this.lastSeenPos.y -= 1.5;
      this.alertLevel = 1;
    } else {
      // Accuracy decays faster than it builds, so breaking LOS actually helps.
      this.losTime = Math.max(0, this.losTime - dt * 2.5);
      this.timeSinceSeen += dt;
      this.alertLevel = Math.max(0, this.alertLevel - dt * 0.08);
    }
  }

  hear(pos, strength) {
    if (!this.alive) return;
    const d = this.position.distanceTo(pos);
    if (d > HEARING_RANGE * strength) return;
    this.lastSeenPos.copy(pos);
    this.timeSinceSeen = Math.min(this.timeSinceSeen, 2.5);
    this.alertLevel = Math.max(this.alertLevel, 0.75);
    if (this.state === S.IDLE || this.state === S.PATROL) this._setState(S.ALERT);
  }

  suppress(amount) {
    this.suppression = Math.min(1.6, this.suppression + amount);
  }

  /* ------------------------------------------------------------ state machine */

  _setState(s) {
    if (this.state === s) return;
    if (this.isFlanker && s !== S.FLANK) { this.mgr.releaseFlank(this); this.isFlanker = false; }
    this.state = s;
    this.stateTime = 0;
  }

  _think(dt, player) {
    this.stateTime += dt;
    this.suppression = Math.max(0, this.suppression - dt * 0.75);
    if (this.repathCd > 0) this.repathCd -= dt;
    if (this.reactionCd > 0) this.reactionCd -= dt;

    const distToPlayer = this.position.distanceTo(player.position);

    // Suppression overrides everything short of death: heads go down first and
    // tactics resume afterwards.
    if (this.suppression > 0.9 && this.state !== S.SUPPRESSED) {
      this._setState(S.SUPPRESSED);
    }

    switch (this.state) {
      case S.IDLE:
        this.desiredSpeed = 0;
        if (this.canSeePlayer) { this.reactionCd = lerp(0.55, 0.16, this.skill); this._setState(S.ALERT); }
        else if (this.stateTime > 1.5 + this.seed * 2) this._setState(S.PATROL);
        break;

      case S.PATROL:
        this.desiredSpeed = 1.5;
        if (this.canSeePlayer) { this.reactionCd = lerp(0.5, 0.14, this.skill); this._setState(S.ALERT); }
        else if (!this.hasGoal || this.position.distanceTo(this.goal) < 1.5) this._pickPatrolGoal();
        break;

      case S.ALERT:
        this.desiredSpeed = 0.6;
        this.aimTarget = this.lastSeenPos;
        if (this.reactionCd <= 0) {
          if (this.canSeePlayer) this._enterCombat(distToPlayer);
          else this._setState(S.SEEK);
        }
        break;

      case S.SEEK:
        this.desiredSpeed = 3.4;
        if (this.canSeePlayer) { this._enterCombat(distToPlayer); break; }
        if (!this.hasGoal || this.position.distanceTo(this.goal) < 1.6) {
          if (this.timeSinceSeen > 14) this._setState(S.PATROL);
          else this._setGoal(this.lastSeenPos);
        }
        break;

      case S.COMBAT:
        this._combat(dt, player, distToPlayer);
        break;

      case S.FLANK: {
        this.desiredSpeed = 3.9;
        const arrived = !this.hasGoal || this.position.distanceTo(this.goal) < 1.8;
        if (arrived || this.stateTime > 11) { this.mgr.releaseFlank(this); this.isFlanker = false; this._setState(S.COMBAT); }
        else if (this.canSeePlayer && this.stateTime > 1.2 && distToPlayer < 16) this._fireLogic(dt, player, distToPlayer, 0.55);
        break;
      }

      case S.REPOSITION:
        this.desiredSpeed = 4.2;
        if (!this.hasGoal || this.position.distanceTo(this.goal) < 1.5 || this.stateTime > 6) this._setState(S.COMBAT);
        break;

      case S.SUPPRESSED:
        this.desiredSpeed = 0;
        this.crouchTarget = 1;
        this.burstLeft = 0;
        if (this.suppression < 0.35) {
          // Being pinned is a good reason to leave: half the time, relocate.
          if (hash1(this.index + this.stateTime) > 0.5) { this._pickCover(player, true); this._setState(S.REPOSITION); }
          else this._setState(S.COMBAT);
        }
        break;
    }
  }

  _enterCombat(dist) {
    this._setState(S.COMBAT);
    this.burstCd = lerp(0.9, 0.28, this.skill) + hash1(this.index * 7.7) * 0.4;
    // Flanking is a squad decision, capped so the player is never surrounded.
    if (dist > 12 && this.mgr.requestFlank(this)) {
      this.isFlanker = true;
      this._pickFlankGoal();
      this._setState(S.FLANK);
    }
  }

  _combat(dt, player, dist) {
    this.aimTarget = null;

    if (!this.canSeePlayer && this.timeSinceSeen > 3.5) {
      this._setState(S.SEEK);
      this._setGoal(this.lastSeenPos);
      return;
    }

    const inCover = this.coverNode >= 0 && this.position.distanceTo(this.mgr.coverNodes[this.coverNode].pos) < 2.0;

    if (!inCover && this.repathCd <= 0) {
      this._pickCover(player, false);
      this.repathCd = 1.2;
    }

    if (inCover) {
      this.desiredSpeed = 0;
      // Peek/re-cover oscillation: stand and shoot, drop and reload the plan.
      this.peeking -= dt;
      if (this.peeking <= 0) {
        const cycle = this.burstLeft > 0 || this.fireCd > 0;
        this.peeking = cycle ? 0.15 : lerp(1.5, 0.7, this.skill) * (0.7 + hash1(this.index + this.stateTime | 0) * 0.6);
        this.exposed = !this.exposed;
      }
      this.crouchTarget = this.exposed ? 0.15 : 0.95;
      this.lean = this.exposed ? (this.index & 1 ? 0.35 : -0.35) : 0;
    } else {
      this.desiredSpeed = dist > 22 ? 3.6 : 2.0;
      this.crouchTarget = 0;
      this.exposed = true;
    }

    if (this.exposed !== false) this._fireLogic(dt, player, dist, 1);

    // Long stalemates get broken by moving, which keeps firefights from
    // settling into two men trading shots from the same two corners.
    if (this.stateTime > 8 + this.seed * 4 && this.mgr.rng() > 0.5) {
      this._pickCover(player, true);
      this._setState(S.REPOSITION);
    }
  }

  /* -------------------------------------------------------------- navigation */

  _setGoal(p) {
    this.goal.copy(p);
    this.hasGoal = true;
    this.path = this.mgr.findPath(this.position, this.goal);
    this.pathIndex = 0;
  }

  _pickPatrolGoal() {
    const n = this.mgr.coverNodes;
    if (!n.length) { this.hasGoal = false; return; }
    const node = n[(Math.random() * n.length) | 0];
    this.coverNode = -1;
    this._setGoal(node.pos);
  }

  _pickFlankGoal() {
    const target = this.mgr.pickFlankNode(this);
    if (target >= 0) { this.coverNode = target; this._setGoal(this.mgr.coverNodes[target].pos); }
    else this.isFlanker = false;
  }

  _pickCover(player, forceNew) {
    const node = this.mgr.pickCoverNode(this, player, forceNew);
    if (node >= 0) { this.coverNode = node; this._setGoal(this.mgr.coverNodes[node].pos); }
    else { this.coverNode = -1; this.hasGoal = false; }
  }

  /** Seek along the path, avoid geometry with three probes, and stay unclumped. */
  _steer(dt, player) {
    const steer = _v0.set(0, 0, 0);
    let want = 0;

    if (this.hasGoal) {
      let target = this.goal;
      if (this.path.length) {
        while (this.pathIndex < this.path.length &&
               this.position.distanceToSquared(this.path[this.pathIndex]) < 1.44) this.pathIndex++;
        target = this.pathIndex < this.path.length ? this.path[this.pathIndex] : this.goal;
      }
      const to = _v1.copy(target).sub(this.position); to.y = 0;
      const d = to.length();
      if (d > 0.001) { steer.add(to.divideScalar(d)); want = this.desiredSpeed; }
      else if (!this.path.length || this.pathIndex >= this.path.length) this.hasGoal = false;
    }

    if (want > 0) {
      const fwd = _v2.copy(steer).normalize();
      const avoid = this.mgr.avoidance(this.position, fwd, _v3);
      steer.addScaledVector(avoid, 2.2);
    }

    const sep = this.mgr.separation(this, _v4);
    steer.add(sep);
    if (sep.lengthSq() > 0.01 && want === 0) want = 1.0;   // shuffle out of a pile

    steer.y = 0;
    const len = steer.length();
    if (len > 0.001 && want > 0) {
      steer.divideScalar(len).multiplyScalar(want);
      // Crouching and being shot at both cost mobility.
      const mob = (1 - this.crouch * 0.55) * (1 - this.suppression * 0.4);
      steer.multiplyScalar(mob);
      this.velocity.x = approach(this.velocity.x, steer.x, 7, dt);
      this.velocity.z = approach(this.velocity.z, steer.z, 7, dt);
    } else {
      this.velocity.x = approach(this.velocity.x, 0, 11, dt);
      this.velocity.z = approach(this.velocity.z, 0, 11, dt);
    }

    this.position.x += this.velocity.x * dt;
    this.position.z += this.velocity.z * dt;
    this.mgr.resolveCollisions(this.position, 0.34);
    this.mgr.clampToBounds(this.position);
    this.position.y = approach(this.position.y, this.mgr.groundHeight(this.position.x, this.position.z, this.position.y), 14, dt);

    this.speed = Math.hypot(this.velocity.x, this.velocity.z);

    // Legs face travel; the torso is free to look elsewhere. That separation is
    // the whole reason the skeleton exists.
    const aimAt = this.aimTarget || (this.canSeePlayer || this.timeSinceSeen < 4 ? this.lastSeenPos : null);
    if (aimAt) {
      const dx = (this.canSeePlayer ? player.position.x : aimAt.x) - this.position.x;
      const dz = (this.canSeePlayer ? player.position.z : aimAt.z) - this.position.z;
      this.aimYaw += angleDelta(this.aimYaw, Math.atan2(dx, dz)) * Math.min(1, dt * lerp(4, 11, this.skill));
      const dy = (this.canSeePlayer ? player.position.y + 1.5 : aimAt.y + 1.5) - (this.position.y + 1.45);
      this.aimPitch = approach(this.aimPitch, clamp(Math.atan2(dy, Math.max(0.5, Math.hypot(dx, dz))), -0.7, 0.7), 8, dt);
    } else {
      this.aimPitch = approach(this.aimPitch, 0, 3, dt);
    }

    const moveYaw = this.speed > 0.35 ? Math.atan2(this.velocity.x, this.velocity.z) : this.aimYaw;
    // If the torso would have to twist past its limit, the hips come around.
    const twist = angleDelta(moveYaw, this.aimYaw);
    const bodyYaw = Math.abs(twist) > 1.15 ? this.aimYaw - Math.sign(twist) * 1.15 : moveYaw;
    this.yaw += angleDelta(this.yaw, bodyYaw) * Math.min(1, dt * (this.speed > 0.35 ? 9 : 4));

    this.crouch = approach(this.crouch, this.crouchTarget || 0, 6, dt);
  }

  /* ------------------------------------------------------------------ combat */

  _fireLogic(dt, player, dist, willingness) {
    if (this.fireCd > 0) this.fireCd -= dt;
    if (this.burstCd > 0) this.burstCd -= dt;
    if (!this.canSeePlayer || this.suppression > 0.6) return;
    if (dist > 62) return;

    if (this.burstLeft > 0) {
      if (this.fireCd <= 0) {
        this._shoot(player, dist);
        this.burstLeft--;
        this.fireCd = lerp(0.13, 0.085, this.skill);
        if (this.burstLeft === 0) {
          this.burstCd = lerp(1.7, 0.65, this.skill) * (0.75 + hash1(this.index * 3.1 + this.stateTime | 0) * 0.6) / willingness;
          this.firstBurst = false;
        }
      }
      return;
    }

    if (this.burstCd <= 0 && this.losTime > lerp(0.55, 0.18, this.skill)) {
      // The squad gate is what stops eight rifles opening up on the same tick.
      if (!this.mgr.requestBurst(this)) return;
      this.burstLeft = 2 + ((hash1(this.index + this.stateTime) * 3) | 0) + (this.skill > 0.7 ? 1 : 0);
      this.fireCd = 0;
    }
  }

  _shoot(player, dist) {
    const origin = _v0.copy(this.muzzle.getWorldPosition(_v5));
    if (!isFinite(origin.x)) origin.copy(this.position).y += 1.4;

    const chest = _v1.copy(player.position); chest.y += 1.25;
    const dir = _v2.copy(chest).sub(origin).normalize();

    // Accuracy model: cone shrinks as LOS persists, and the opening burst is
    // pushed deliberately wide. Getting shot the instant you round a corner is
    // what makes a shooter feel cheap, so the first exchange is a warning.
    let spread = lerp(0.085, 0.022, this.skill) * (1 + dist * 0.008);
    spread *= lerp(1.0, 0.35, smoothstep(0, 2.6, this.losTime));
    spread *= 1 + this.suppression * 1.6;
    if (this.firstBurst) spread += lerp(0.075, 0.035, this.skill);
    if (player.sprinting) spread *= 1.15;

    const a = this.mgr.rng() * TAU;
    const r = Math.sqrt(this.mgr.rng()) * spread;
    const right = _v3.crossVectors(dir, UP).normalize();
    const up = _v4.crossVectors(right, dir);
    dir.addScaledVector(right, Math.cos(a) * r).addScaledVector(up, Math.sin(a) * r).normalize();

    const hitScan = this.mgr.traceShot(origin, dir, 90, player);
    this.mgr.particles?.muzzleFlash?.(origin, dir);
    this.mgr.particles?.tracer?.(origin, hitScan.point, false);
    this.mgr.audio?.playAt?.('enemyFire', origin);
    this.mgr.notifySelfFire(this, origin);

    if (hitScan.hitPlayer) {
      const falloff = lerp(1, 0.55, smoothstep(18, 55, dist));
      const dmg = lerp(6, 13, this.skill) * falloff;
      player.takeDamage?.(dmg, this.position);
    } else {
      this.mgr.particles?.impact?.(hitScan.point, hitScan.normal);
    }

    this.recoil = 1;
  }

  /* ------------------------------------------------------------------ damage */

  /**
   * @returns {{killed:boolean, headshot:boolean, damage:number}}
   */
  takeDamage(amount, point, dir, headshot = false, multiplier = 1) {
    if (!this.alive) return { killed: false, headshot: false, damage: 0 };
    const dealt = amount * multiplier;
    this.health -= dealt;

    const p = toVec3(point) || this.position.clone().setY(this.position.y + 1.3);
    const d = toVec3(dir) || _v0.set(0, 0, 1);
    this.mgr.particles?.blood?.(p, d);

    if (this.health <= 0) {
      this.die(p, d, dealt);
      return { killed: true, headshot, damage: dealt };
    }

    // A flinch that additively rotates the struck side reads as impact without
    // interrupting the aim, so the soldier keeps fighting while being hit.
    const side = p.y > this.position.y + 1.35 ? 1.6 : 1.0;
    this.flinch.set(
      -0.10 * side - Math.min(0.3, dealt * 0.006),
      (hash1(this.health) - 0.5) * 0.30,
      (hash1(this.health * 2.1) - 0.5) * 0.34);
    this.flinchDecay = 1;
    this.suppress(0.35);
    this.alertLevel = 1;
    this.timeSinceSeen = 0;
    if (!this.canSeePlayer && this.mgr.player) {
      this.lastSeenPos.copy(this.mgr.player.position);
      if (this.state === S.IDLE || this.state === S.PATROL) this._setState(S.ALERT);
    }
    this.mgr.audio?.playAt?.('enemyHit', p);
    return { killed: false, headshot, damage: dealt };
  }

  /* Aliases — Weapons.js owns the call site and may reach for any of these. */
  damage(a, p, d, h, m) { return this.takeDamage(a, p, d, h, m); }
  applyDamage(a, p, d, h, m) { return this.takeDamage(a, p, d, h, m); }
  hit(a, p, d, h, m) { return this.takeDamage(a, p, d, h, m); }

  die(point, dir, force = 30) {
    if (!this.alive) return;
    this.alive = false;
    this.health = 0;
    this.state = S.DEAD;
    if (this.isFlanker) { this.mgr.releaseFlank(this); this.isFlanker = false; }
    this.mgr.audio?.playAt?.('enemyDeath', this.position);
    this._startRagdoll(point, dir, force);
    this.mgr.registerCorpse(this);
  }

  /* ----------------------------------------------------------------- ragdoll */

  _startRagdoll(point, dir, force) {
    this.group.updateMatrixWorld(true);

    const pos = [], prev = [], pin = [];
    const grab = (boneName, local) => {
      const b = this.bones[boneName];
      const v = new THREE.Vector3(local[0], local[1], local[2]);
      b.localToWorld(v);
      return v;
    };
    const P = [];
    P[P_PELVIS] = grab('pelvis', [0, 0, 0]);
    P[P_CHEST] = grab('chest', [0, 0.02, 0]);
    P[P_HEAD] = grab('head', [0, 0.12, 0]);
    P[P_SHL] = grab('armL', [0, 0, 0]);
    P[P_ELL] = grab('foreL', [0, 0, 0]);
    P[P_HAL] = grab('handL', [0, 0, 0]);
    P[P_SHR] = grab('armR', [0, 0, 0]);
    P[P_ELR] = grab('foreR', [0, 0, 0]);
    P[P_HAR] = grab('handR', [0, 0, 0]);
    P[P_HIL] = grab('thighL', [0, 0, 0]);
    P[P_KNL] = grab('shinL', [0, 0, 0]);
    P[P_FTL] = grab('footL', [0, 0, 0]);
    P[P_HIR] = grab('thighR', [0, 0, 0]);
    P[P_KNR] = grab('shinR', [0, 0, 0]);
    P[P_FTR] = grab('footR', [0, 0, 0]);

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      pos.push(P[i]);
      prev.push(P[i].clone());
      pin.push(false);
    }

    const links = RAGDOLL_LINKS.map(([a, b, stiff]) => ({
      a, b, stiff, rest: P[a].distanceTo(P[b]),
    }));

    // Verlet stores velocity as (pos - prev), so the bullet impulse is applied
    // by displacing the previous position backwards along the shot vector.
    const impulse = _v0.copy(dir || UP).normalize().multiplyScalar(clamp(force * 0.0011, 0.004, 0.05));
    const hitPoint = point || P[P_CHEST];
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const w = 1 / (1 + pos[i].distanceToSquared(hitPoint) * 3);
      prev[i].sub(_v1.copy(impulse).multiplyScalar(0.4 + w * 2.2));
      // A dead-still fall looks staged; a little scatter makes each one unique.
      prev[i].x -= (this.mgr.rng() - 0.5) * 0.004;
      prev[i].z -= (this.mgr.rng() - 0.5) * 0.004;
    }

    this.ragdoll = { pos, prev, pin, links, boneQ: new Map() };

    // Bones now carry world-space transforms, so the group is flattened to the
    // identity and every joint below it can be written in world coordinates.
    this.group.position.set(0, 0, 0);
    this.group.quaternion.identity();
    this.group.scale.setScalar(1);
  }

  _stepRagdoll(dt) {
    const rd = this.ragdoll;
    const { pos, prev, links } = rd;
    const g = -18.5 * dt * dt;               // heavier than 9.8: corpses read floaty otherwise
    const damp = 0.992;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const p = pos[i], q = prev[i];
      const vx = (p.x - q.x) * damp, vy = (p.y - q.y) * damp, vz = (p.z - q.z) * damp;
      q.copy(p);
      p.x += vx; p.y += vy + g; p.z += vz;
    }

    for (let iter = 0; iter < 5; iter++) {
      for (let i = 0; i < links.length; i++) {
        const l = links[i], a = pos[l.a], b = pos[l.b];
        const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
        const d = Math.hypot(dx, dy, dz) || 1e-5;
        const diff = (d - l.rest) / d * 0.5 * l.stiff;
        const ox = dx * diff, oy = dy * diff, oz = dz * diff;
        a.x += ox; a.y += oy; a.z += oz;
        b.x -= ox; b.y -= oy; b.z -= oz;
      }
      for (let i = 0; i < PARTICLE_COUNT; i++) this.mgr.collideParticle(pos[i], prev[i], 0.10);
    }
  }

  _applyRagdollToBones() {
    const { pos, boneQ } = this.ragdoll;
    this.group.updateMatrixWorld(true);

    this.bones.pelvis.position.copy(pos[P_PELVIS]);
    boneQ.clear();

    for (let i = 0; i < RAGDOLL_BONES.length; i++) {
      const [name, from, to] = RAGDOLL_BONES[i];
      const bone = this.bones[name];
      const dir = _v0.copy(pos[to]).sub(pos[from]);
      if (dir.lengthSq() < 1e-8) continue;
      dir.normalize();

      const worldQ = _q0.setFromUnitVectors(bone.userData.axis, dir);
      const parent = bone.parent;
      const parentQ = boneQ.get(parent) || null;
      if (parentQ) bone.quaternion.copy(_q1.copy(parentQ).invert().multiply(worldQ));
      else bone.quaternion.copy(worldQ);
      boneQ.set(bone, worldQ.clone());
    }

    // The neck, hands and feet have no particle pair; they simply relax.
    for (const n of ['spine', 'neck', 'head', 'handL', 'handR', 'footL', 'footR']) {
      const b = this.bones[n];
      b.quaternion.slerp(_q0.setFromEuler(_e0.set(0, 0, 0)), 0.06);
    }
  }

  /* ---------------------------------------------------------------- animation */

  _animate(dt, elapsed) {
    const b = this.bones;
    const rest = (n) => b[n].userData.restRot;

    // Phase advances with ground covered, so speed changes never slide the feet.
    const strideLength = 1.35;
    this.stride = (this.stride + (this.speed * dt / strideLength) * TAU) % TAU;
    const moving = smoothstep(0.15, 1.2, this.speed);
    const run = smoothstep(2.2, 4.4, this.speed);
    const ph = this.stride;
    const amp = lerp(0.30, 0.72, run) * moving;
    const t = elapsed;

    /* legs */
    const swingL = Math.sin(ph), swingR = Math.sin(ph + Math.PI);
    const kneeL = Math.max(0, -Math.sin(ph - 0.75)) * lerp(0.55, 1.25, run) * moving;
    const kneeR = Math.max(0, -Math.sin(ph + Math.PI - 0.75)) * lerp(0.55, 1.25, run) * moving;
    const crouchBend = this.crouch;

    b.thighL.rotation.set(-swingL * amp - crouchBend * 0.95, 0, 0.04);
    b.thighR.rotation.set(-swingR * amp - crouchBend * 0.95, 0, -0.04);
    b.shinL.rotation.x = kneeL + crouchBend * 1.65;
    b.shinR.rotation.x = kneeR + crouchBend * 1.65;
    // Toes stay level with the ground rather than following the shin.
    b.footL.rotation.x = -(b.thighL.rotation.x + b.shinL.rotation.x) * 0.55 + Math.max(0, swingL) * 0.18 * moving;
    b.footR.rotation.x = -(b.thighR.rotation.x + b.shinR.rotation.x) * 0.55 + Math.max(0, swingR) * 0.18 * moving;

    /* pelvis: two bobs per stride, plus the hip drop over the planted leg */
    const bob = Math.cos(ph * 2) * lerp(0.018, 0.045, run) * moving;
    b.pelvis.position.set(0, 0.95 + bob - crouchBend * 0.40, 0);
    b.pelvis.rotation.set(crouchBend * 0.22, 0, Math.sin(ph) * 0.07 * moving);

    /* torso counter-rotation and breathing */
    const breath = Math.sin(t * 1.9 + this.seed * 9) * 0.016;
    const sway = snoise(t * 0.35, this.seed) * 0.03 * (1 - moving);   // idle weight shift
    const twist = angleDelta(this.yaw, this.aimYaw);

    b.spine.rotation.set(rest('spine').x + breath + crouchBend * 0.18, twist * 0.35 - Math.sin(ph) * 0.10 * moving, sway);
    b.chest.rotation.set(
      rest('chest').x - this.aimPitch * 0.45 + breath * 1.8 + this.flinch.x,
      twist * 0.65 + this.flinch.y,
      -this.lean * 0.55 + this.flinch.z + sway * 0.6);
    b.neck.rotation.set(-this.aimPitch * 0.30, 0, 0);
    b.head.rotation.set(rest('head').x - this.aimPitch * 0.25 + breath * 2, 0, 0);

    /* arms: weapon pose survives the walk cycle, only the swing scales */
    const aiming = this.canSeePlayer || this.timeSinceSeen < 3;
    const armSwing = amp * (aiming ? 0.12 : 0.62);
    const rec = this.recoil ? this.recoil : 0;
    if (this.recoil) this.recoil = Math.max(0, this.recoil - dt * 7);

    const rL = rest('armL'), rR = rest('armR');
    b.armL.rotation.set(rL.x + swingR * armSwing - rec * 0.10, rL.y, rL.z);
    b.armR.rotation.set(rR.x + swingL * armSwing - rec * 0.16, rR.y, rR.z);
    b.foreL.rotation.x = rest('foreL').x - Math.abs(swingR) * armSwing * 0.4 - rec * 0.06;
    b.foreR.rotation.x = rest('foreR').x - rec * 0.12;
    b.handR.rotation.x = rest('handR').x - rec * 0.10;

    /* flinch decays on a spring so the recovery overshoots slightly */
    if (this.flinchDecay > 0) {
      this.flinchDecay = Math.max(0, this.flinchDecay - dt * 4.5);
      this.flinch.multiplyScalar(Math.max(0, 1 - dt * 6));
    }

    this.group.position.copy(this.position);
    this.group.rotation.set(0, this.yaw, this.lean * 0.12);
  }

  /* --------------------------------------------------------------------- tick */

  update(dt, player, elapsed) {
    if (this.alive) {
      this._updatePerception(dt, this.mgr.playerEye(player, _v2.clone()));
      this._think(dt, player);
      this._steer(dt, player);
      this._animate(dt, elapsed);
      return;
    }

    if (this.ragdoll) {
      this.corpseAge += dt;
      if (this.corpseAge < 6) {
        this._stepRagdoll(dt);
        this._applyRagdollToBones();
        this.position.copy(this.ragdoll.pos[P_PELVIS]);
        this.position.y = this.mgr.groundHeight(this.position.x, this.position.z, this.position.y);
      }
      // Fading a corpse would need a per-body material clone; sinking it costs
      // one vector write and hides the pop just as well.
      if (this.sinking > 0) {
        this.sinking -= dt;
        this.group.position.y -= dt * 0.55;
        if (this.sinking <= 0) this.despawn();
      }
    }
  }

  beginSink(seconds = 1.6) { if (this.sinking <= 0) this.sinking = seconds; }

  despawn() {
    this.group.visible = false;
    this.ragdoll = null;
    this.sinking = 0;
  }

  /* -------------------------------------------------------------- hit volumes */

  /** Bounding sphere for the early-out; centre in `out`, radius returned. */
  boundingSphere(out) {
    out.set(this.position.x, this.position.y + 0.95 - this.crouch * 0.35, this.position.z);
    return 1.15;
  }

  /**
   * Head first (it is the smallest and the most rewarding), then torso, then
   * limbs. Each entry is a capsule in world space rebuilt from yaw and crouch
   * rather than read back from the matrices, which keeps the test independent
   * of whether the scene graph has been updated this frame.
   */
  hitVolumes(out) {
    out.length = 0;
    const c = this.crouch;
    const s = Math.sin(this.yaw), co = Math.cos(this.yaw);
    const fx = s, fz = co;                      // forward
    const rx = co, rz = -s;                     // right
    const px = this.position.x, py = this.position.y, pz = this.position.z;
    const drop = c * 0.42;
    const lean = this.lean * 0.20;

    const at = (side, fwd, y) => new THREE.Vector3(
      px + rx * side + fx * fwd + rx * lean,
      py + y,
      pz + rz * side + fz * fwd + rz * lean);

    out.push({ part: 'head', mult: 2.6, headshot: true, r: 0.135,
      a: at(0, 0.02, 1.62 - drop), b: at(0, 0.02, 1.70 - drop) });
    out.push({ part: 'torso', mult: 1.0, headshot: false, r: 0.235,
      a: at(0, 0, 0.98 - drop * 0.7), b: at(0, 0.02, 1.45 - drop) });
    out.push({ part: 'legL', mult: 0.65, headshot: false, r: 0.135,
      a: at(0.11, 0, 0.90 - drop * 0.7), b: at(0.11, c * 0.2, 0.06) });
    out.push({ part: 'legR', mult: 0.65, headshot: false, r: 0.135,
      a: at(-0.11, 0, 0.90 - drop * 0.7), b: at(-0.11, c * 0.2, 0.06) });
    out.push({ part: 'armL', mult: 0.6, headshot: false, r: 0.10,
      a: at(0.22, 0.02, 1.42 - drop), b: at(0.14, 0.34, 1.16 - drop) });
    out.push({ part: 'armR', mult: 0.6, headshot: false, r: 0.10,
      a: at(-0.22, 0.02, 1.42 - drop), b: at(-0.10, 0.30, 1.18 - drop) });
    return out;
  }
}

/* ===========================================================================
   Ray / volume intersection helpers
   =========================================================================== */

/** Ray vs sphere; returns entry distance or -1. */
function raySphere(ro, rd, c, r) {
  const ox = ro.x - c.x, oy = ro.y - c.y, oz = ro.z - c.z;
  const b = ox * rd.x + oy * rd.y + oz * rd.z;
  const cc = ox * ox + oy * oy + oz * oz - r * r;
  if (cc > 0 && b > 0) return -1;
  const disc = b * b - cc;
  if (disc < 0) return -1;
  const t = -b - Math.sqrt(disc);
  return t < 0 ? 0 : t;
}

/**
 * Ray vs capsule (segment a-b, radius r). Solves the infinite-cylinder
 * quadratic in the segment's local frame, then falls back to the end caps.
 */
function rayCapsule(ro, rd, a, b, r, outPoint) {
  const bax = b.x - a.x, bay = b.y - a.y, baz = b.z - a.z;
  const oax = ro.x - a.x, oay = ro.y - a.y, oaz = ro.z - a.z;
  const baba = bax * bax + bay * bay + baz * baz;
  const bard = bax * rd.x + bay * rd.y + baz * rd.z;
  const baoa = bax * oax + bay * oay + baz * oaz;
  const rdoa = rd.x * oax + rd.y * oay + rd.z * oaz;
  const oaoa = oax * oax + oay * oay + oaz * oaz;

  const A = baba - bard * bard;
  const B = baba * rdoa - baoa * bard;
  const C = baba * oaoa - baoa * baoa - r * r * baba;
  let t = -1;
  const h = B * B - A * C;
  if (h >= 0 && Math.abs(A) > 1e-9) {
    const tt = (-B - Math.sqrt(h)) / A;
    const y = baoa + tt * bard;
    if (tt >= 0 && y >= 0 && y <= baba) t = tt;
  }
  if (t < 0) {
    const ta = raySphere(ro, rd, a, r);
    const tb = raySphere(ro, rd, b, r);
    if (ta >= 0 && (tb < 0 || ta < tb)) t = ta;
    else if (tb >= 0) t = tb;
  }
  if (t < 0) return -1;
  if (outPoint) outPoint.set(ro.x + rd.x * t, ro.y + rd.y * t, ro.z + rd.z * t);
  return t;
}

/** Squared distance from point p to segment a-b. */
function pointSegDistSq(p, a, b) {
  const bax = b.x - a.x, bay = b.y - a.y, baz = b.z - a.z;
  const pax = p.x - a.x, pay = p.y - a.y, paz = p.z - a.z;
  const d = bax * bax + bay * bay + baz * baz;
  const t = d > 1e-9 ? clamp((pax * bax + pay * bay + paz * baz) / d, 0, 1) : 0;
  const dx = pax - bax * t, dy = pay - bay * t, dz = paz - baz * t;
  return dx * dx + dy * dy + dz * dz;
}

/** Slab test. Returns entry distance in [0,maxT] or -1. */
function rayBox(ro, rd, box) {
  let tmin = 0, tmax = Infinity;
  for (const ax of ['x', 'y', 'z']) {
    const inv = 1 / (rd[ax] || 1e-12);
    let t1 = (box.min[ax] - ro[ax]) * inv;
    let t2 = (box.max[ax] - ro[ax]) * inv;
    if (t1 > t2) { const s = t1; t1 = t2; t2 = s; }
    if (t1 > tmin) tmin = t1;
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return -1;
  }
  return tmin;
}

/* ===========================================================================
   EnemyManager
   =========================================================================== */

export class EnemyManager {
  constructor({ scene, level, textures, particles, audio, settings }) {
    this.scene = scene;
    this.level = level || {};
    this.particles = particles;
    this.audio = audio;
    this.settings = settings;
    this.player = null;                    // assigned by main.js after construction

    this.assets = new EnemyAssets(textures, settings);
    this.max = Math.max(1, settings.enemyCount | 0);
    this.list = [];
    this.aliveCount = 0;
    this.spawnedThisWave = 0;
    this.wave = 1;
    this.elapsed = 0;

    this.corpses = [];
    this.flankers = new Set();
    this.lastBurstStart = -99;
    this.shotCounterSeen = -1;

    this._seed = 0x2f6e2b1 >>> 0;
    this._ray = new THREE.Raycaster();
    this._ray.firstHitOnly = true;
    this._volumes = [];
    this._tmpHit = new THREE.Vector3();

    this._readLevel();
  }

  /* ------------------------------------------------------------- level intake */

  _readLevel() {
    const lvl = this.level;

    this.colliders = [];
    for (const c of (lvl.colliders || [])) {
      const box = toBox3(c);
      if (box) this.colliders.push(box);
    }

    this.spawns = [];
    for (const s of (lvl.enemySpawns || [])) {
      const v = toVec3(s, new THREE.Vector3());
      if (v) this.spawns.push(v);
    }

    this.raycastTargets = Array.isArray(lvl.raycastTargets) ? lvl.raycastTargets : [];
    this.bounds = toBox3(lvl.bounds);
    this.groundY = this.bounds ? this.bounds.min.y : 0;

    this._buildCoverGraph(lvl.coverPoints || []);
  }

  /**
   * The cover points become a navigation graph: two nodes are neighbours if
   * they are close and can see each other, which is a decent stand-in for
   * "an enemy can walk between them" without building a navmesh.
   */
  _buildCoverGraph(points) {
    this.coverNodes = [];
    for (const p of points) {
      const pos = toVec3(p, new THREE.Vector3());
      if (!pos) continue;
      const normal = p && p.normal ? toVec3(p.normal, new THREE.Vector3()) : null;
      this.coverNodes.push({ pos, normal, links: [], cost: [], owner: -1 });
    }

    const MAX_LINK = 26;
    const a = new THREE.Vector3(), b = new THREE.Vector3();
    for (let i = 0; i < this.coverNodes.length; i++) {
      for (let j = i + 1; j < this.coverNodes.length; j++) {
        const d = this.coverNodes[i].pos.distanceTo(this.coverNodes[j].pos);
        if (d > MAX_LINK) continue;
        a.copy(this.coverNodes[i].pos); a.y += 1.0;
        b.copy(this.coverNodes[j].pos); b.y += 1.0;
        if (!this.hasLineOfSight(a, b, d)) continue;
        this.coverNodes[i].links.push(j); this.coverNodes[i].cost.push(d);
        this.coverNodes[j].links.push(i); this.coverNodes[j].cost.push(d);
      }
    }

    this._open = new Int32Array(Math.max(1, this.coverNodes.length));
    this._g = new Float32Array(Math.max(1, this.coverNodes.length));
    this._f = new Float32Array(Math.max(1, this.coverNodes.length));
    this._from = new Int32Array(Math.max(1, this.coverNodes.length));
    this._closed = new Uint8Array(Math.max(1, this.coverNodes.length));
  }

  /* Deterministic RNG — reproducible firefights are debuggable firefights. */
  rng() {
    this._seed = (this._seed * 1664525 + 1013904223) >>> 0;
    return this._seed / 4294967296;
  }

  /* ------------------------------------------------------------------ queries */

  playerEye(player, out = _v0) {
    const p = player?.position;
    if (!p) return out.set(0, 1.6, 0);
    const h = player.eyeHeight ?? player.viewHeight ?? (player.crouching ? 1.15 : 1.62);
    return out.set(p.x, p.y + h, p.z);
  }

  /** True when nothing solid sits between `from` and `to`. */
  hasLineOfSight(from, to, dist) {
    const d = dist ?? from.distanceTo(to);
    if (d < 0.01) return true;
    const dir = _v5.copy(to).sub(from).divideScalar(d);

    if (this.raycastTargets.length) {
      this._ray.set(from, dir);
      this._ray.near = 0;
      this._ray.far = d - 0.05;
      const hits = this._ray.intersectObjects(this.raycastTargets, false);
      return hits.length === 0;
    }
    // No render targets handed over: the collision boxes are the next best thing.
    for (let i = 0; i < this.colliders.length; i++) {
      const t = rayBox(from, dir, this.colliders[i]);
      if (t >= 0 && t < d - 0.05) return false;
    }
    return true;
  }

  /**
   * Traces an enemy bullet. Returns where it stopped and whether it caught the
   * player, so the shot is resolved against the same world the player sees.
   */
  traceShot(origin, dir, maxDist, player) {
    let best = maxDist;
    let normal = _v3.set(0, 1, 0).clone();
    let hitPlayer = false;

    if (this.raycastTargets.length) {
      this._ray.set(origin, dir);
      this._ray.near = 0; this._ray.far = maxDist;
      const hits = this._ray.intersectObjects(this.raycastTargets, false);
      if (hits.length) {
        best = hits[0].distance;
        if (hits[0].face) normal = hits[0].face.normal.clone();
      }
    } else {
      for (let i = 0; i < this.colliders.length; i++) {
        const t = rayBox(origin, dir, this.colliders[i]);
        if (t >= 0 && t < best) best = t;
      }
    }

    if (player?.position) {
      const feet = _v0.set(player.position.x, player.position.y + 0.35, player.position.z);
      const head = _v1.set(player.position.x, player.position.y + (player.crouching ? 1.15 : 1.68), player.position.z);
      const t = rayCapsule(origin, dir, feet, head, 0.34, null);
      if (t >= 0 && t < best) { best = t; hitPlayer = true; }
    }

    const point = new THREE.Vector3().copy(origin).addScaledVector(dir, Math.min(best, maxDist));
    return { point, normal, distance: best, hitPlayer };
  }

  /** Highest collider top under (x,z) that is not above the current feet. */
  groundHeight(x, z, currentY) {
    let y = this.groundY;
    const head = currentY + 1.2;
    for (let i = 0; i < this.colliders.length; i++) {
      const b = this.colliders[i];
      if (x < b.min.x - 0.25 || x > b.max.x + 0.25) continue;
      if (z < b.min.z - 0.25 || z > b.max.z + 0.25) continue;
      // Only surfaces you could step onto count; a ceiling must not lift a bot.
      if (b.max.y > y && b.max.y <= head) y = b.max.y;
    }
    return y;
  }

  /** Push a point out of any box it has entered, along the shallowest axis. */
  resolveCollisions(p, radius) {
    for (let i = 0; i < this.colliders.length; i++) {
      const b = this.colliders[i];
      if (p.y + 1.5 < b.min.y || p.y + 0.2 > b.max.y) continue;   // stepping over / under
      const minX = b.min.x - radius, maxX = b.max.x + radius;
      const minZ = b.min.z - radius, maxZ = b.max.z + radius;
      if (p.x < minX || p.x > maxX || p.z < minZ || p.z > maxZ) continue;
      const dxl = p.x - minX, dxr = maxX - p.x;
      const dzl = p.z - minZ, dzr = maxZ - p.z;
      const m = Math.min(dxl, dxr, dzl, dzr);
      if (m === dxl) p.x = minX; else if (m === dxr) p.x = maxX;
      else if (m === dzl) p.z = minZ; else p.z = maxZ;
    }
  }

  clampToBounds(p) {
    if (!this.bounds) return;
    p.x = clamp(p.x, this.bounds.min.x + 0.5, this.bounds.max.x - 0.5);
    p.z = clamp(p.z, this.bounds.min.z + 0.5, this.bounds.max.z - 0.5);
  }

  /** Verlet particle vs world, used only by corpses. */
  collideParticle(p, prev, radius) {
    const g = this.groundHeight(p.x, p.z, p.y);
    if (p.y - radius < g) {
      p.y = g + radius;
      // Kill the tangential velocity too, otherwise limbs skate on landing.
      prev.x = lerp(prev.x, p.x, 0.45);
      prev.z = lerp(prev.z, p.z, 0.45);
      prev.y = p.y;
    }
    for (let i = 0; i < this.colliders.length; i++) {
      const b = this.colliders[i];
      if (p.x < b.min.x - radius || p.x > b.max.x + radius) continue;
      if (p.y < b.min.y - radius || p.y > b.max.y + radius) continue;
      if (p.z < b.min.z - radius || p.z > b.max.z + radius) continue;
      const dxl = p.x - (b.min.x - radius), dxr = (b.max.x + radius) - p.x;
      const dyl = p.y - (b.min.y - radius), dyr = (b.max.y + radius) - p.y;
      const dzl = p.z - (b.min.z - radius), dzr = (b.max.z + radius) - p.z;
      const m = Math.min(dxl, dxr, dyl, dyr, dzl, dzr);
      if (m === dxl) p.x = b.min.x - radius; else if (m === dxr) p.x = b.max.x + radius;
      else if (m === dyl) p.y = b.min.y - radius; else if (m === dyr) { p.y = b.max.y + radius; prev.y = p.y; }
      else if (m === dzl) p.z = b.min.z - radius; else p.z = b.max.z + radius;
    }
    if (this.bounds) {
      p.x = clamp(p.x, this.bounds.min.x + 0.2, this.bounds.max.x - 0.2);
      p.z = clamp(p.z, this.bounds.min.z + 0.2, this.bounds.max.z - 0.2);
    }
  }

  /* ------------------------------------------------------------------ steering */

  /**
   * Three probes — straight ahead and ±35° — sampled at two depths. Sampling
   * points against the AABBs beats a real raycast here: it is a handful of
   * comparisons and it degrades into "there is a wall to my left" naturally.
   */
  avoidance(pos, fwd, out) {
    out.set(0, 0, 0);
    const base = Math.atan2(fwd.x, fwd.z);
    const angles = [0, 0.61, -0.61];
    let blockedFwd = 0;
    for (let i = 0; i < 3; i++) {
      const a = base + angles[i];
      const sx = Math.sin(a), sz = Math.cos(a);
      let blocked = 0;
      for (const dist of [1.1, 2.4]) {
        const x = pos.x + sx * dist, z = pos.z + sz * dist;
        if (this._pointBlocked(x, pos.y, z, 0.45)) { blocked = 1 / dist; break; }
      }
      if (!blocked) continue;
      if (i === 0) blockedFwd = blocked;
      out.x -= sx * blocked; out.z -= sz * blocked;
    }
    // Head-on into a wall: the pure repulsion vector cancels, so slide sideways.
    if (blockedFwd && out.lengthSq() < 0.02) {
      out.x += fwd.z * 0.9; out.z -= fwd.x * 0.9;
    }
    out.y = 0;
    return out;
  }

  _pointBlocked(x, y, z, radius) {
    for (let i = 0; i < this.colliders.length; i++) {
      const b = this.colliders[i];
      if (b.max.y < y + 0.35) continue;          // low enough to walk over
      if (x < b.min.x - radius || x > b.max.x + radius) continue;
      if (z < b.min.z - radius || z > b.max.z + radius) continue;
      if (y + 1.5 < b.min.y) continue;
      return true;
    }
    if (this.bounds) {
      if (x < this.bounds.min.x + 0.6 || x > this.bounds.max.x - 0.6) return true;
      if (z < this.bounds.min.z + 0.6 || z > this.bounds.max.z - 0.6) return true;
    }
    return false;
  }

  separation(self, out) {
    out.set(0, 0, 0);
    const R = 1.5;
    for (let i = 0; i < this.list.length; i++) {
      const o = this.list[i];
      if (o === self || !o.alive) continue;
      const dx = self.position.x - o.position.x, dz = self.position.z - o.position.z;
      const d2 = dx * dx + dz * dz;
      if (d2 > R * R || d2 < 1e-6) continue;
      const d = Math.sqrt(d2);
      const w = (1 - d / R) * 1.8;
      out.x += dx / d * w; out.z += dz / d * w;
    }
    return out;
  }

  /* ---------------------------------------------------------------- pathfinding */

  nearestNode(p, maxDist = 40) {
    let best = -1, bd = maxDist * maxDist;
    for (let i = 0; i < this.coverNodes.length; i++) {
      const d = this.coverNodes[i].pos.distanceToSquared(p);
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  /** A* over the cover graph. Returns world-space waypoints, goal included. */
  findPath(from, to) {
    const n = this.coverNodes.length;
    const out = [];
    if (!n) return out;
    // A clear straight line beats any graph detour, and most fights are close.
    if (this._walkable(from, to)) { out.push(to.clone()); return out; }

    const start = this.nearestNode(from), goal = this.nearestNode(to);
    if (start < 0 || goal < 0) { out.push(to.clone()); return out; }
    if (start === goal) { out.push(this.coverNodes[goal].pos.clone(), to.clone()); return out; }

    const g = this._g, f = this._f, from_ = this._from, closed = this._closed, open = this._open;
    g.fill(Infinity); f.fill(Infinity); from_.fill(-1); closed.fill(0);
    let openN = 0;
    g[start] = 0; f[start] = this.coverNodes[start].pos.distanceTo(this.coverNodes[goal].pos);
    open[openN++] = start;

    while (openN > 0) {
      let bi = 0;
      for (let i = 1; i < openN; i++) if (f[open[i]] < f[open[bi]]) bi = i;
      const cur = open[bi];
      open[bi] = open[--openN];
      if (cur === goal) break;
      closed[cur] = 1;

      const node = this.coverNodes[cur];
      for (let k = 0; k < node.links.length; k++) {
        const nx = node.links[k];
        if (closed[nx]) continue;
        const tentative = g[cur] + node.cost[k];
        if (tentative >= g[nx]) continue;
        from_[nx] = cur;
        g[nx] = tentative;
        f[nx] = tentative + this.coverNodes[nx].pos.distanceTo(this.coverNodes[goal].pos);
        let present = false;
        for (let i = 0; i < openN; i++) if (open[i] === nx) { present = true; break; }
        if (!present && openN < open.length) open[openN++] = nx;
      }
    }

    if (from_[goal] < 0 && start !== goal) { out.push(to.clone()); return out; }
    for (let c = goal; c >= 0; c = from_[c]) {
      out.push(this.coverNodes[c].pos.clone());
      if (c === start) break;
    }
    out.reverse();
    out.push(to.clone());
    return out;
  }

  _walkable(from, to) {
    const a = _v0.set(from.x, from.y + 0.9, from.z);
    const b = _v1.set(to.x, to.y + 0.9, to.z);
    const d = a.distanceTo(b);
    if (d > 22) return false;
    const dir = _v2.copy(b).sub(a).divideScalar(d || 1);
    const steps = Math.ceil(d / 1.5);
    for (let i = 1; i <= steps; i++) {
      const t = (i / steps) * d;
      if (this._pointBlocked(a.x + dir.x * t, from.y, a.z + dir.z * t, 0.5)) return false;
    }
    return true;
  }

  /* ------------------------------------------------------------ squad tactics */

  requestFlank(enemy) {
    if (this.flankers.size >= 2) return false;
    this.flankers.add(enemy);
    return true;
  }

  releaseFlank(enemy) { this.flankers.delete(enemy); }

  /** Global trigger discipline: bursts start at least 220ms apart. */
  requestBurst(enemy) {
    if (this.elapsed - this.lastBurstStart < 0.22) return false;
    this.lastBurstStart = this.elapsed;
    return true;
  }

  /** A soldier's own muzzle report should not spook the man beside him. */
  notifySelfFire(shooter, origin) {
    for (let i = 0; i < this.list.length; i++) {
      const e = this.list[i];
      if (e === shooter || !e.alive) continue;
      if (e.state === S.IDLE || e.state === S.PATROL) e.hear(origin, 0.5);
    }
  }

  /**
   * Best cover node: near enough to reach, breaks LOS to the player, and not
   * already claimed. Nodes behind the player are skipped for non-flankers so
   * pressure stays on the front.
   */
  pickCoverNode(enemy, player, forceNew) {
    const eye = this.playerEye(player, _v1.clone());
    let best = -1, bestScore = -Infinity;
    for (let i = 0; i < this.coverNodes.length; i++) {
      const node = this.coverNodes[i];
      if (node.owner >= 0 && node.owner !== enemy.index) continue;
      if (forceNew && i === enemy.coverNode) continue;

      const pos = node.pos;
      const distSelf = pos.distanceTo(enemy.position);
      if (distSelf > 34) continue;
      const distPlayer = pos.distanceTo(player.position);
      if (distPlayer < 6) continue;                       // not in his lap

      const stand = _v2.set(pos.x, pos.y + 1.5, pos.z);
      const covered = !this.hasLineOfSight(stand, eye);
      // Cover that never lets you shoot back is a hiding place, not a position:
      // reward nodes that are covered while crouched but open when peeking.
      const crouched = _v3.set(pos.x, pos.y + 0.9, pos.z);
      const peekable = this.hasLineOfSight(crouched, eye);

      let score = 0;
      if (covered) score += 40;
      if (peekable) score += 12;
      score -= distSelf * 1.4;
      score -= Math.abs(distPlayer - 16) * 0.9;           // prefer a mid-range fight
      if (i === enemy.coverNode) score += 6;              // hysteresis, no dithering
      if (score > bestScore) { bestScore = score; best = i; }
    }
    if (best >= 0) {
      if (enemy.coverNode >= 0 && this.coverNodes[enemy.coverNode]?.owner === enemy.index) {
        this.coverNodes[enemy.coverNode].owner = -1;
      }
      this.coverNodes[best].owner = enemy.index;
    }
    return best;
  }

  /** A flank target sits wide of the player's current facing. */
  pickFlankNode(enemy) {
    const player = this.player;
    if (!player) return -1;
    const pyaw = player.yaw ?? (player.camera ? player.camera.rotation.y : 0);
    const fx = -Math.sin(pyaw), fz = -Math.cos(pyaw);
    let best = -1, bestScore = -Infinity;
    for (let i = 0; i < this.coverNodes.length; i++) {
      const node = this.coverNodes[i];
      if (node.owner >= 0 && node.owner !== enemy.index) continue;
      const dx = node.pos.x - player.position.x, dz = node.pos.z - player.position.z;
      const d = Math.hypot(dx, dz);
      if (d < 7 || d > 30) continue;
      const dot = (dx * fx + dz * fz) / d;
      // dot near -1 is directly behind him; that is the prize.
      const score = -dot * 30 - node.pos.distanceTo(enemy.position) * 0.8;
      if (score > bestScore) { bestScore = score; best = i; }
    }
    if (best >= 0) this.coverNodes[best].owner = enemy.index;
    return best;
  }

  /* --------------------------------------------------------------- public API */

  reset() {
    for (const e of this.list) {
      e.alive = false;
      e.ragdoll = null;
      e.group.visible = false;
      e.state = S.DEAD;
    }
    this.corpses.length = 0;
    this.flankers.clear();
    this.aliveCount = 0;
    this.spawnedThisWave = 0;
    this.wave = 1;
    this.lastBurstStart = -99;
    this.shotCounterSeen = -1;
    for (const n of this.coverNodes) n.owner = -1;
    // The level may have finished building after we were constructed.
    if (!this.coverNodes.length || !this.colliders.length) this._readLevel();
  }

  spawnWave(count, waveNumber) {
    this.wave = waveNumber || 1;
    const n = Math.min(count | 0, this.max);
    const spawns = this.spawns.length ? this.spawns : [new THREE.Vector3(0, 0, -12)];
    const playerPos = this.player?.position;

    // Spawn points are ranked by distance from the player so a wave never
    // materialises in his face; the far half of the list is used first.
    const order = spawns.map((p, i) => i);
    if (playerPos) order.sort((a, b) => spawns[b].distanceToSquared(playerPos) - spawns[a].distanceToSquared(playerPos));

    let placed = 0;
    for (let i = 0; i < n; i++) {
      const e = this._acquire();
      if (!e) break;
      const base = spawns[order[i % order.length]];
      const jitter = (i / order.length) | 0;
      const pos = _v0.copy(base);
      if (jitter > 0) {
        const a = this.rng() * TAU;
        pos.x += Math.cos(a) * (1.2 + jitter * 0.8);
        pos.z += Math.sin(a) * (1.2 + jitter * 0.8);
        this.resolveCollisions(pos, 0.4);
        this.clampToBounds(pos);
      }
      pos.y = this.groundHeight(pos.x, pos.z, pos.y);
      e.spawn(pos, this.wave);
      placed++;
    }

    this.spawnedThisWave = placed;
    this.aliveCount = this._countAlive();
  }

  _acquire() {
    for (const e of this.list) {
      if (!e.alive && !e.ragdoll) return e;
    }
    if (this.list.length < this.max) {
      const e = new Enemy(this, this.list.length);
      this.scene.add(e.group);
      this.list.push(e);
      return e;
    }
    // Everything is a corpse: recycle the oldest one immediately.
    const oldest = this.corpses.shift();
    if (oldest) { oldest.despawn(); return oldest; }
    return null;
  }

  _countAlive() {
    let n = 0;
    for (const e of this.list) if (e.alive) n++;
    return n;
  }

  registerCorpse(enemy) {
    this.corpses.push(enemy);
    const cap = Math.max(1, this.settings.maxCorpses | 0);
    while (this.corpses.length > cap) this.corpses.shift().beginSink();
  }

  /** Called by Weapons.js (or anything else) when a loud noise happens. */
  notifyGunshot(position, loudness = 1) {
    const p = toVec3(position, _v0.clone());
    if (!p) return;
    for (const e of this.list) e.hear(p, loudness);
  }

  update(dt, player, weapons) {
    this.elapsed += dt;
    if (player) this.player = player;

    this._pollWeaponFire(weapons, player);

    let alive = 0;
    for (let i = 0; i < this.list.length; i++) {
      const e = this.list[i];
      if (!e.group.visible) continue;
      e.update(dt, player, this.elapsed);
      if (e.alive) alive++;
    }
    this.aliveCount = alive;
  }

  /**
   * Weapons.js owns the trigger, so rather than require a callback we watch any
   * monotonic shot counter it exposes. If it instead calls notifyGunshot, this
   * quietly finds nothing and costs one property read.
   */
  _pollWeaponFire(weapons, player) {
    if (!weapons || !player?.position) return;
    const n = weapons.shotCounter ?? weapons.shotsFired ?? weapons.roundsFired ?? weapons.shotCount;
    if (typeof n !== 'number') return;
    if (this.shotCounterSeen < 0) { this.shotCounterSeen = n; return; }
    if (n <= this.shotCounterSeen) { this.shotCounterSeen = n; return; }
    this.shotCounterSeen = n;
    const suppressed = weapons.suppressed === true;
    this.notifyGunshot(player.position, suppressed ? 0.35 : 1);
  }

  /**
   * Ray against every living enemy. A bounding sphere rejects almost all of
   * them in three multiplies before any capsule maths runs, and the shot also
   * doubles as a suppression probe for the men it narrowly missed.
   */
  hitTest(origin, dir, maxDist = 300) {
    let best = null;
    let bestT = maxDist;
    const centre = _v1;

    for (let i = 0; i < this.list.length; i++) {
      const e = this.list[i];
      if (!e.alive) continue;

      const r = e.boundingSphere(centre);
      const near = raySphere(origin, dir, centre, r);
      if (near < 0 || near > maxDist) {
        // Near miss inside a 2.5m corridor still makes a man duck.
        const d2 = pointSegDistSq(centre, origin, _v2.copy(origin).addScaledVector(dir, maxDist));
        if (d2 < 6.25) e.suppress(0.55);
        continue;
      }
      if (near > bestT) continue;

      const vols = e.hitVolumes(this._volumes);
      for (let v = 0; v < vols.length; v++) {
        const vol = vols[v];
        const t = rayCapsule(origin, dir, vol.a, vol.b, vol.r, this._tmpHit);
        if (t < 0 || t >= bestT) continue;
        bestT = t;
        best = {
          enemy: e,
          point: this._tmpHit.clone(),
          distance: t,
          headshot: vol.headshot === true,
          part: vol.part,
          multiplier: vol.mult,
          damageMultiplier: vol.mult,
          normal: this._tmpHit.clone().sub(centre).normalize(),
        };
        // Volumes are ordered head-first, so the first hit on this body wins.
        break;
      }
    }
    return best;
  }

  dispose() {
    for (const e of this.list) this.scene.remove(e.group);
    this.list.length = 0;
    this.assets.dispose();
  }
}

export default EnemyManager;

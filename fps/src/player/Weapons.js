// Weapons.js — procedural weapon models, viewmodel rig, ballistics, gunfeel.
//
// Three things happen in here and they are deliberately kept apart:
//   1. MODELLING   — every firearm is assembled from beveled primitives into a
//                    handful of merged meshes (one draw call per material).
//   2. SIMULATION  — fire rate, spread growth, recoil patterns, hitscan.
//   3. PRESENTATION— the viewmodel rig: sway, lag, bob, ADS, recoil springs,
//                    reload/inspect/melee state machines, shells, flashes.
//
// The single most important modelling convention: every weapon is built in a
// local space whose ORIGIN IS THE SIGHT'S OPTICAL CENTRE, pointing down -Z.
// Aiming is then just moving the rig to (0, 0, -eyeRelief) with zero rotation,
// so the sights land dead centre on any aspect ratio without hand-tuned offsets.

import * as THREE from 'three';
import { makeStandardMaterial } from '../world/Textures.js';

const DEG = Math.PI / 180;
const HEADSHOT_MULT = 2.5;
const FLASH_LIFE = 0.045;          // muzzle flash duration, matches real film smear
const SHELL_COUNT = 14;
const GRENADE_POOL = 4;

const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const lerp = (a, b, t) => a + (b - a) * t;
/** Frame-rate independent exponential approach; `rate` is roughly 1/seconds. */
const damp = (a, b, rate, dt) => lerp(a, b, 1 - Math.exp(-rate * dt));
const easeOutBack = t => { const c = 1.70158; const u = t - 1; return 1 + (c + 1) * u * u * u + c * u * u; };

/* Scratch vectors — firing runs up to 20x/second and must not allocate. */
const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _v3 = new THREE.Vector3();
const _v4 = new THREE.Vector3();
const _fwd = new THREE.Vector3();
const _right = new THREE.Vector3();
const _up = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _nmat = new THREE.Matrix3();
const _ray = new THREE.Raycaster();
const SPRING_AXES = [['z', 'vz'], ['p', 'vp'], ['r', 'vr']];

/* ========================================================================== */
/*                              geometry helpers                              */
/* ========================================================================== */

/**
 * A box with chamfered edges. Sharp 90° boxes are the single biggest tell of a
 * programmer-art model: real machined parts always catch a highlight on the
 * break line, and that highlight is what makes the silhouette read as metal.
 */
function bevelBox(w, h, d, bevel = 0.0022) {
  const b = Math.min(bevel, w * 0.3, h * 0.3, d * 0.45);
  const hw = w * 0.5 - b, hh = h * 0.5 - b;
  const s = new THREE.Shape();
  s.moveTo(-hw, -hh - b);
  s.lineTo(hw, -hh - b);
  s.quadraticCurveTo(hw + b, -hh - b, hw + b, -hh);
  s.lineTo(hw + b, hh);
  s.quadraticCurveTo(hw + b, hh + b, hw, hh + b);
  s.lineTo(-hw, hh + b);
  s.quadraticCurveTo(-hw - b, hh + b, -hw - b, hh);
  s.lineTo(-hw - b, -hh);
  s.quadraticCurveTo(-hw - b, -hh - b, -hw, -hh - b);

  const g = new THREE.ExtrudeGeometry(s, {
    depth: Math.max(0.0001, d - b * 2), bevelEnabled: true,
    bevelThickness: b, bevelSize: b, bevelSegments: 1, curveSegments: 1,
  });
  g.translate(0, 0, -(d * 0.5 - b));
  g.computeVertexNormals();
  return g;
}

function cylGeo(rTop, rBot, len, seg = 12, open = false) {
  return new THREE.CylinderGeometry(rTop, rBot, len, seg, 1, open);
}

/** A tapered tube profile revolved — used for scope bells and grips. */
function latheGeo(profile, seg = 16) {
  const pts = profile.map(p => new THREE.Vector2(p[0], p[1]));
  return new THREE.LatheGeometry(pts, seg);
}

/**
 * Concatenates transformed geometries into one buffer.
 * three's mergeGeometries lives in addons, which this build does not vendor, and
 * a viewmodel made of 70 separate meshes would cost 70 draw calls on a phone.
 */
function mergeItems(items) {
  const geos = [];
  let total = 0;
  for (const it of items) {
    const g = it.geo.index ? it.geo.toNonIndexed() : it.geo.clone();
    g.applyMatrix4(it.m);
    if (!g.attributes.normal) g.computeVertexNormals();
    geos.push(g);
    total += g.attributes.position.count;
  }
  const pos = new Float32Array(total * 3);
  const nor = new Float32Array(total * 3);
  const uv = new Float32Array(total * 2);
  let o = 0;
  for (const g of geos) {
    const p = g.attributes.position, n = g.attributes.normal, u = g.attributes.uv;
    pos.set(p.array.subarray(0, p.count * 3), o * 3);
    nor.set(n.array.subarray(0, n.count * 3), o * 3);
    if (u) uv.set(u.array.subarray(0, u.count * 2), o * 2);
    o += p.count;
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  out.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  out.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  out.computeBoundingSphere();
  return out;
}

/**
 * Accumulates primitives tagged by material name, then emits one mesh per tag.
 * Parts that have to animate (magazine, charging handle, bolt) get their own
 * Rack so they stay independently transformable after the merge.
 */
class Rack {
  constructor() { this.bins = new Map(); this._o = new THREE.Object3D(); }

  add(geo, mat, px = 0, py = 0, pz = 0, rx = 0, ry = 0, rz = 0, sx = 1, sy = 1, sz = 1) {
    const o = this._o;
    o.position.set(px, py, pz);
    o.rotation.set(rx, ry, rz);
    o.scale.set(sx, sy, sz);
    o.updateMatrix();
    let bin = this.bins.get(mat);
    if (!bin) this.bins.set(mat, bin = []);
    bin.push({ geo, m: o.matrix.clone() });
    return this;
  }

  box(w, h, d, mat, px, py, pz, rx = 0, ry = 0, rz = 0, bevel = 0.0022) {
    return this.add(bevelBox(w, h, d, bevel), mat, px, py, pz, rx, ry, rz);
  }

  /** Cylinder along Z (the axis every barrel, tube and buffer wants). */
  tube(rTop, rBot, len, mat, px, py, pz, seg = 12, open = false) {
    return this.add(cylGeo(rTop, rBot, len, seg, open), mat, px, py, pz, Math.PI * 0.5, 0, 0);
  }

  pin(rTop, rBot, len, mat, px, py, pz, rx = 0, ry = 0, rz = 0, seg = 10) {
    return this.add(cylGeo(rTop, rBot, len, seg), mat, px, py, pz, rx, ry, rz);
  }

  ring(radius, thick, mat, px, py, pz, seg = 18) {
    return this.add(new THREE.TorusGeometry(radius, thick, 6, seg), mat, px, py, pz);
  }

  /** Evenly spaced copies along Z — picatinny teeth, vent ribs, barrel flutes. */
  repeatZ(n, z0, step, fn) {
    for (let i = 0; i < n; i++) fn(z0 + i * step, i);
    return this;
  }

  build(materials, group = new THREE.Group()) {
    for (const [name, items] of this.bins) {
      const mesh = new THREE.Mesh(mergeItems(items), materials[name] || materials.gun);
      mesh.frustumCulled = false;       // viewmodel is always on screen; culling it is pure risk
      mesh.castShadow = false;
      mesh.receiveShadow = false;
      group.add(mesh);
    }
    return group;
  }
}

/* ========================================================================== */
/*                             procedural textures                            */
/* ========================================================================== */

/** Radial star flare for the muzzle card. No assets ship with this game. */
function flashTexture(size = 64) {
  const d = new Uint8Array(size * size * 4);
  const c = (size - 1) * 0.5;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (x - c) / c, dy = (y - c) / c;
      const r = Math.hypot(dx, dy);
      const ang = Math.atan2(dy, dx);
      const star = 0.55 + 0.45 * Math.pow(Math.abs(Math.cos(ang * 3)), 6);
      const core = Math.pow(clamp(1 - r / 0.28, 0, 1), 1.4);
      const halo = Math.pow(clamp(1 - r, 0, 1), 2.6) * star;
      const a = clamp(core + halo * 0.8, 0, 1);
      const i = (y * size + x) * 4;
      d[i] = 255;
      d[i + 1] = clamp(0.55 + core * 0.45, 0, 1) * 255;
      d[i + 2] = clamp(0.18 + core * 0.72, 0, 1) * 255;
      d[i + 3] = a * 255;
    }
  }
  const t = new THREE.DataTexture(d, size, size, THREE.RGBAFormat);
  t.colorSpace = THREE.SRGBColorSpace;
  t.minFilter = THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.needsUpdate = true;
  return t;
}

/* ========================================================================== */
/*                               weapon models                                */
/* ========================================================================== */

/**
 * M4-pattern carbine. Origin sits inside the red dot, 62mm above the bore,
 * which is the real sight-over-bore for a flat-top AR with a mounted optic.
 */
function buildRifle(mats) {
  const g = new THREE.Group();
  const r = new Rack();
  const BORE = -0.062;

  // Upper receiver + flat-top rail. The rail teeth are individual ribs because
  // the shadow line between them is most of what says "picatinny" at 55° FOV.
  r.box(0.044, 0.052, 0.26, 'gun', 0, BORE + 0.007, -0.14);
  r.box(0.030, 0.007, 0.30, 'gun', 0, -0.026, -0.17);
  r.repeatZ(15, -0.032, -0.019, z => r.box(0.032, 0.0045, 0.007, 'worn', 0, -0.0215, z, 0, 0, 0, 0.0008));
  r.box(0.0045, 0.021, 0.050, 'worn', 0.0235, BORE + 0.010, -0.082);   // ejection port cover
  r.pin(0.0062, 0.0062, 0.016, 'worn', 0.026, BORE, -0.056, 0, 0, Math.PI * 0.5);  // forward assist
  r.pin(0.0075, 0.0075, 0.014, 'gun', -0.024, BORE + 0.004, -0.048, 0, 0, Math.PI * 0.5); // brass deflector

  // Lower receiver, magwell, trigger group.
  r.box(0.038, 0.050, 0.155, 'gun', 0, -0.100, -0.085);
  r.box(0.043, 0.014, 0.054, 'gun', 0, -0.110, -0.101);
  r.box(0.010, 0.006, 0.050, 'gun', 0, -0.133, -0.036);
  r.box(0.010, 0.030, 0.007, 'gun', 0, -0.119, -0.059);
  r.box(0.010, 0.030, 0.007, 'gun', 0, -0.119, -0.013);
  r.box(0.006, 0.021, 0.008, 'worn', 0, -0.124, -0.033, 0.22);
  r.pin(0.008, 0.008, 0.012, 'worn', 0.022, -0.093, -0.021, 0, 0, Math.PI * 0.5);  // safety selector
  r.pin(0.010, 0.010, 0.014, 'worn', -0.022, -0.093, -0.005, 0, 0, Math.PI * 0.5); // mag release

  // Pistol grip: a tapered lathe beats a rotated box because the palm swell
  // catches light along its length instead of flashing one flat facet.
  r.add(latheGeo([[0.0, 0.048], [0.016, 0.046], [0.0185, 0.010], [0.016, -0.030], [0.019, -0.046], [0.0, -0.049]], 12),
    'poly', 0, -0.148, 0.010, -0.30, 0, 0);
  r.box(0.036, 0.010, 0.030, 'poly', 0, -0.196, 0.026, -0.30);

  // Buffer tube + collapsible stock.
  r.tube(0.0155, 0.0155, 0.17, 'worn', 0, BORE, 0.098, 14);
  r.repeatZ(6, 0.045, 0.019, z => r.ring(0.017, 0.0016, 'worn', 0, BORE, z, 12));
  r.box(0.042, 0.056, 0.115, 'poly', 0, BORE - 0.004, 0.110);
  r.box(0.030, 0.014, 0.100, 'poly', 0, -0.030, 0.102);                 // cheek weld
  r.box(0.046, 0.074, 0.016, 'rubber', 0, BORE - 0.006, 0.170, 0.06);   // butt pad
  r.ring(0.011, 0.0025, 'worn', 0.024, -0.088, 0.062, 10);              // sling loop

  // Handguard built as ribs with gaps: the gaps ARE the vent slots, and the
  // barrel visible through them is what sells the depth.
  r.box(0.038, 0.011, 0.175, 'poly', 0, -0.038, -0.385);
  r.box(0.038, 0.011, 0.175, 'poly', 0, -0.086, -0.385);
  for (const sx of [-1, 1]) {
    r.repeatZ(6, -0.312, -0.026, z =>
      r.box(0.007, 0.042, 0.015, 'poly', sx * 0.0195, -0.062, z));
    r.box(0.007, 0.048, 0.176, 'poly', sx * 0.0185, -0.062, -0.385, 0, 0, 0, 0.0015);
  }
  r.ring(0.024, 0.004, 'gun', 0, -0.062, -0.300, 16);
  r.ring(0.021, 0.004, 'gun', 0, -0.062, -0.470, 16);

  // Barrel assembly and A2-style flash hider.
  r.tube(0.0088, 0.0088, 0.245, 'gun', 0, BORE, -0.400, 12);
  r.tube(0.0076, 0.0076, 0.055, 'worn', 0, BORE, -0.500, 12);
  r.box(0.022, 0.026, 0.032, 'gun', 0, -0.050, -0.456);                 // gas block
  r.tube(0.0055, 0.0055, 0.14, 'worn', 0, -0.042, -0.400, 8);           // gas tube
  r.tube(0.0105, 0.0125, 0.032, 'worn', 0, BORE, -0.522, 12);
  r.ring(0.0125, 0.0022, 'worn', 0, BORE, -0.5385, 12);
  r.repeatZ(4, -0.514, -0.008, z => r.box(0.026, 0.0035, 0.004, 'worn', 0, BORE + 0.010, z));

  // Backup iron sights, folded but present — the front tower breaks up the
  // handguard silhouette exactly where the eye expects a step.
  r.box(0.016, 0.020, 0.013, 'worn', 0, -0.030, -0.452);
  r.box(0.004, 0.016, 0.004, 'worn', 0, -0.020, -0.452);

  // Red dot: open tube with an emissive reticle at the front lens.
  r.box(0.032, 0.016, 0.052, 'gun', 0, -0.017, -0.058);
  r.tube(0.0175, 0.0175, 0.056, 'gun', 0, 0, -0.055, 18, true);
  r.tube(0.0155, 0.0155, 0.054, 'poly', 0, 0, -0.055, 18, true);
  r.ring(0.0175, 0.0022, 'worn', 0, 0, -0.0275, 18);
  r.ring(0.0175, 0.0022, 'worn', 0, 0, -0.0825, 18);
  r.pin(0.006, 0.006, 0.010, 'worn', 0.019, 0.004, -0.055, 0, 0, Math.PI * 0.5);  // windage turret
  r.pin(0.006, 0.006, 0.010, 'worn', 0, 0.019, -0.055);                           // elevation turret
  r.add(new THREE.CircleGeometry(0.0155, 18), 'glass', 0, 0, -0.081, 0, 0, 0);
  r.add(new THREE.CircleGeometry(0.0022, 10), 'dot', 0, 0, -0.0795);

  r.build(mats, g);

  const mag = new Rack();
  mag.box(0.031, 0.058, 0.050, 'poly', 0, -0.154, -0.102, 0.06);
  mag.box(0.031, 0.058, 0.048, 'poly', 0, -0.209, -0.093, 0.17);
  mag.box(0.035, 0.011, 0.052, 'worn', 0, -0.240, -0.087, 0.17);
  mag.repeatZ(3, -0.118, -0.012, z => mag.box(0.032, 0.004, 0.006, 'poly', 0, -0.180, z + 0.02, 0.10));
  const magNode = mag.build(mats);
  g.add(magNode);

  const ch = new Rack();
  ch.box(0.038, 0.011, 0.048, 'worn', 0, -0.027, 0.004);
  ch.box(0.058, 0.009, 0.013, 'worn', 0, -0.027, 0.024);
  const chargeNode = ch.build(mats);
  g.add(chargeNode);

  const bolt = new Rack();
  bolt.box(0.010, 0.019, 0.030, 'worn', 0.019, BORE + 0.010, -0.086);
  const boltNode = bolt.build(mats);
  g.add(boltNode);

  return {
    group: g,
    parts: { mag: magNode, charge: chargeNode, bolt: boltNode },
    muzzle: new THREE.Vector3(0, BORE, -0.548),
    eject: new THREE.Vector3(0.028, BORE + 0.010, -0.082),
  };
}

/** MP5-pattern SMG: tubular receiver, drum rear sight, hooded front post. */
function buildSMG(mats) {
  const g = new THREE.Group();
  const r = new Rack();
  const BORE = -0.046;

  r.tube(0.023, 0.023, 0.255, 'gun', 0, BORE, -0.160, 16);
  r.box(0.030, 0.016, 0.230, 'gun', 0, BORE + 0.022, -0.150);           // top rib
  r.tube(0.020, 0.020, 0.070, 'gun', 0, BORE, -0.020, 14);              // receiver end cap
  r.box(0.048, 0.030, 0.055, 'gun', 0, BORE + 0.004, -0.300);           // front trunnion

  // Rear drum sight: origin lives in the aperture.
  r.tube(0.014, 0.014, 0.028, 'worn', 0, 0, -0.012, 14);
  r.ring(0.0135, 0.0035, 'worn', 0, 0, -0.0255, 16);
  r.pin(0.016, 0.016, 0.009, 'worn', 0, 0, -0.012, Math.PI * 0.5, 0, 0, 14);
  r.repeatZ(8, 0, 0, (_, i) => {
    const a = i / 8 * Math.PI * 2;
    r.box(0.0035, 0.004, 0.010, 'worn', Math.sin(a) * 0.016, Math.cos(a) * 0.016, -0.012);
  });

  // Cocking tube on the left, with the handle in its forward notch.
  r.tube(0.011, 0.011, 0.215, 'gun', -0.030, BORE + 0.026, -0.215, 12);
  r.tube(0.013, 0.013, 0.030, 'gun', -0.030, BORE + 0.026, -0.322, 12);

  // Handguard with punched vent slots.
  r.add(latheGeo([[0.0, 0.075], [0.026, 0.070], [0.028, -0.040], [0.023, -0.072], [0.0, -0.075]], 14),
    'poly', 0, BORE - 0.004, -0.290, Math.PI * 0.5, 0, 0);
  for (const sx of [-1, 1]) {
    r.repeatZ(4, -0.245, -0.030, z =>
      r.box(0.008, 0.026, 0.017, 'gun', sx * 0.023, BORE - 0.008, z));
  }

  r.tube(0.0105, 0.0105, 0.075, 'worn', 0, BORE, -0.372, 12);           // exposed barrel
  r.tube(0.0135, 0.0135, 0.030, 'worn', 0, BORE, -0.402, 12);           // muzzle nut
  r.repeatZ(3, -0.396, -0.010, z => r.ring(0.0145, 0.0015, 'worn', 0, BORE, z, 12));

  // Hooded front post — the hood is an open cylinder so you sight through it.
  r.tube(0.015, 0.015, 0.030, 'worn', 0, 0, -0.372, 12, true);
  r.box(0.0035, 0.015, 0.0035, 'worn', 0, -0.008, -0.372);
  r.box(0.034, 0.024, 0.014, 'gun', 0, -0.020, -0.372);                 // sight base

  // Trigger group and grip.
  r.box(0.038, 0.056, 0.105, 'poly', 0, -0.090, -0.095);
  r.box(0.010, 0.006, 0.046, 'poly', 0, -0.122, -0.075);
  r.box(0.010, 0.026, 0.007, 'poly', 0, -0.110, -0.095);
  r.box(0.006, 0.020, 0.008, 'worn', 0, -0.112, -0.070, 0.20);
  r.add(latheGeo([[0.0, 0.042], [0.017, 0.040], [0.019, 0.004], [0.016, -0.030], [0.019, -0.042], [0.0, -0.044]], 12),
    'poly', 0, -0.140, -0.048, -0.26, 0, 0);
  r.pin(0.009, 0.009, 0.012, 'worn', 0.023, -0.078, -0.058, 0, 0, Math.PI * 0.5);

  // Retractable stock: two rails and a butt plate read instantly as "collapsed".
  for (const sx of [-1, 1]) r.tube(0.0085, 0.0085, 0.135, 'worn', sx * 0.019, BORE - 0.002, 0.072, 10);
  r.box(0.052, 0.056, 0.014, 'rubber', 0, BORE - 0.004, 0.140, 0.05);
  r.box(0.026, 0.030, 0.040, 'poly', 0, BORE - 0.006, 0.030);
  r.ring(0.010, 0.0022, 'worn', -0.026, BORE + 0.020, -0.070, 10);

  r.build(mats, g);

  const mag = new Rack();
  mag.box(0.030, 0.070, 0.042, 'gun', 0, -0.108, -0.156, -0.09);
  mag.box(0.030, 0.070, 0.040, 'gun', 0, -0.176, -0.170, -0.20);
  mag.box(0.034, 0.010, 0.044, 'worn', 0, -0.211, -0.178, -0.20);
  const magNode = mag.build(mats);
  g.add(magNode);

  const ch = new Rack();
  ch.box(0.020, 0.020, 0.026, 'worn', -0.038, BORE + 0.026, -0.320);
  ch.box(0.030, 0.012, 0.012, 'worn', -0.044, BORE + 0.026, -0.320);
  const chargeNode = ch.build(mats);
  g.add(chargeNode);

  return {
    group: g,
    parts: { mag: magNode, charge: chargeNode, bolt: null },
    muzzle: new THREE.Vector3(0, BORE, -0.420),
    eject: new THREE.Vector3(0.026, BORE + 0.016, -0.075),
  };
}

/** SR-25-pattern DMR with a variable scope, wood furniture accents, bipod. */
function buildSniper(mats) {
  const g = new THREE.Group();
  const r = new Rack();
  const BORE = -0.078;

  r.box(0.048, 0.060, 0.300, 'gun', 0, BORE + 0.010, -0.150);
  r.box(0.036, 0.010, 0.330, 'gun', 0, -0.038, -0.165);
  r.repeatZ(16, -0.030, -0.020, z => r.box(0.038, 0.005, 0.008, 'worn', 0, -0.032, z, 0, 0, 0, 0.0008));
  r.box(0.005, 0.026, 0.060, 'worn', 0.026, BORE + 0.016, -0.095);      // ejection port
  r.pin(0.009, 0.009, 0.016, 'worn', 0.028, BORE + 0.004, -0.060, 0, 0, Math.PI * 0.5);

  r.box(0.042, 0.054, 0.170, 'gun', 0, -0.120, -0.090);
  r.box(0.048, 0.014, 0.058, 'gun', 0, -0.130, -0.106);
  r.box(0.011, 0.007, 0.054, 'gun', 0, -0.156, -0.040);
  r.box(0.011, 0.032, 0.008, 'gun', 0, -0.141, -0.065);
  r.box(0.011, 0.032, 0.008, 'gun', 0, -0.141, -0.014);
  r.box(0.006, 0.022, 0.008, 'worn', 0, -0.146, -0.036, 0.20);

  r.add(latheGeo([[0.0, 0.052], [0.017, 0.050], [0.020, 0.008], [0.017, -0.034], [0.021, -0.050], [0.0, -0.053]], 12),
    'wood', 0, -0.172, 0.014, -0.32, 0, 0);

  // Fixed rifle stock with an adjustable cheek riser.
  r.box(0.046, 0.070, 0.150, 'wood', 0, BORE + 0.006, 0.130);
  r.box(0.038, 0.020, 0.130, 'wood', 0, -0.036, 0.128);
  for (const sx of [-1, 1]) r.pin(0.005, 0.005, 0.030, 'worn', sx * 0.014, -0.050, 0.105);
  r.box(0.050, 0.090, 0.018, 'rubber', 0, BORE + 0.004, 0.208, 0.05);
  r.box(0.028, 0.026, 0.055, 'gun', 0, BORE - 0.030, 0.170);            // monopod spur

  // Long free-float handguard: full-length slots, barrel visible inside.
  r.box(0.042, 0.012, 0.230, 'gun', 0, -0.048, -0.415);
  r.box(0.042, 0.012, 0.230, 'gun', 0, -0.108, -0.415);
  for (const sx of [-1, 1]) {
    r.box(0.008, 0.062, 0.232, 'gun', sx * 0.020, -0.078, -0.415, 0, 0, 0, 0.0015);
    r.repeatZ(7, -0.320, -0.030, z => r.box(0.011, 0.030, 0.019, 'gun', sx * 0.020, -0.078, z));
  }
  r.ring(0.026, 0.005, 'worn', 0, BORE, -0.302, 16);
  r.ring(0.024, 0.005, 'worn', 0, BORE, -0.528, 16);

  // Heavy fluted barrel + muzzle brake with real port cuts.
  r.tube(0.0125, 0.0125, 0.240, 'gun', 0, BORE, -0.412, 14);
  r.tube(0.0115, 0.0115, 0.080, 'worn', 0, BORE, -0.565, 14);
  r.repeatZ(6, 0, 0, (_, i) => {
    const a = i / 6 * Math.PI * 2;
    r.box(0.004, 0.004, 0.230, 'gun', Math.sin(a) * 0.0125, BORE + Math.cos(a) * 0.0125, -0.412);
  });
  r.tube(0.017, 0.017, 0.062, 'worn', 0, BORE, -0.628, 14);
  r.repeatZ(3, -0.612, -0.018, z => {
    r.box(0.040, 0.006, 0.009, 'worn', 0, BORE + 0.011, z);
    r.box(0.006, 0.030, 0.009, 'poly', 0, BORE, z);
  });

  // Folded bipod under the handguard.
  for (const sx of [-1, 1]) {
    r.pin(0.005, 0.005, 0.115, 'worn', sx * 0.014, -0.140, -0.430, 0.30, 0, sx * 0.16);
    r.pin(0.008, 0.008, 0.020, 'worn', sx * 0.011, -0.106, -0.470, 0, 0, Math.PI * 0.5);
  }

  // Scope: lathed body with ocular/objective bells, rings, turrets, glass.
  const SCOPE = [
    [0.0, 0.075], [0.021, 0.072], [0.0235, 0.058], [0.019, 0.048],
    [0.019, -0.120], [0.0235, -0.135], [0.030, -0.150], [0.032, -0.205],
    [0.030, -0.216], [0.0, -0.218],
  ];
  r.add(latheGeo(SCOPE, 22), 'gun', 0, 0, 0, Math.PI * 0.5, 0, 0);
  r.ring(0.021, 0.005, 'worn', 0, 0, 0.055, 18);                        // eyepiece knurl
  r.ring(0.021, 0.004, 'worn', 0, 0, -0.030, 18);                       // magnification collar
  r.repeatZ(10, 0, 0, (_, i) => {
    const a = i / 10 * Math.PI * 2;
    r.box(0.003, 0.003, 0.018, 'worn', Math.sin(a) * 0.021, Math.cos(a) * 0.021, -0.030);
  });
  for (const z of [-0.010, -0.100]) {
    r.box(0.036, 0.030, 0.024, 'gun', 0, -0.021, z);                    // ring bases
    r.ring(0.021, 0.006, 'gun', 0, 0, z, 16);
  }
  r.pin(0.011, 0.013, 0.020, 'worn', 0, 0.024, -0.055);                 // elevation turret
  r.pin(0.011, 0.013, 0.020, 'worn', 0.024, 0, -0.055, 0, 0, Math.PI * 0.5);
  r.pin(0.009, 0.011, 0.016, 'worn', -0.022, 0, -0.055, 0, 0, Math.PI * 0.5);
  r.add(new THREE.CircleGeometry(0.0185, 20), 'glass', 0, 0, 0.070);
  r.add(new THREE.CircleGeometry(0.029, 20), 'glass', 0, 0, -0.203);

  // Duplex reticle drawn as thin additive bars at the first focal plane.
  r.box(0.0009, 0.030, 0.0006, 'dot', 0, 0.020, -0.150);
  r.box(0.0009, 0.030, 0.0006, 'dot', 0, -0.020, -0.150);
  r.box(0.030, 0.0009, 0.0006, 'dot', 0.020, 0, -0.150);
  r.box(0.030, 0.0009, 0.0006, 'dot', -0.020, 0, -0.150);
  r.box(0.0016, 0.0016, 0.0006, 'dot', 0, 0, -0.150);

  r.build(mats, g);

  const mag = new Rack();
  mag.box(0.034, 0.082, 0.056, 'gun', 0, -0.176, -0.108, 0.05);
  mag.box(0.038, 0.012, 0.058, 'worn', 0, -0.222, -0.106, 0.05);
  const magNode = mag.build(mats);
  g.add(magNode);

  const ch = new Rack();
  ch.box(0.042, 0.012, 0.050, 'worn', 0, -0.038, 0.006);
  ch.box(0.064, 0.010, 0.014, 'worn', 0, -0.038, 0.028);
  const chargeNode = ch.build(mats);
  g.add(chargeNode);

  return {
    group: g,
    parts: { mag: magNode, charge: chargeNode, bolt: null },
    muzzle: new THREE.Vector3(0, BORE, -0.662),
    eject: new THREE.Vector3(0.030, BORE + 0.016, -0.095),
  };
}

/** Combat knife, carried permanently and swung on the melee action. */
function buildKnife(mats) {
  const r = new Rack();
  r.add(latheGeo([[0.0, 0.090], [0.006, 0.080], [0.009, 0.010], [0.007, -0.055], [0.0, -0.060]], 8),
    'rubber', 0, 0, 0, Math.PI * 0.5, 0, 0);
  r.repeatZ(6, 0.020, -0.014, z => r.ring(0.0092, 0.0016, 'rubber', 0, 0, z, 10));
  r.box(0.030, 0.010, 0.012, 'worn', 0, 0, -0.062);                     // guard
  r.box(0.006, 0.026, 0.150, 'worn', 0, 0.002, -0.145, 0, 0, 0, 0.0012);
  r.box(0.003, 0.010, 0.130, 'worn', 0.002, -0.010, -0.150, 0, 0, 0.05, 0.0008);
  r.repeatZ(5, -0.100, -0.014, z => r.box(0.007, 0.006, 0.008, 'worn', 0, 0.012, z));  // serrations
  r.box(0.012, 0.008, 0.020, 'worn', 0, 0, 0.098);                      // pommel
  return r.build(mats);
}

/** Fragmentation grenade — used for both the hand model and the thrown prop. */
function buildGrenade(mats) {
  const r = new Rack();
  r.add(latheGeo([
    [0.0, 0.036], [0.014, 0.034], [0.021, 0.020], [0.023, 0.0],
    [0.021, -0.020], [0.014, -0.033], [0.0, -0.036],
  ], 14), 'poly', 0, 0, 0);
  // Body rings lie in the horizontal plane, so the torus needs laying flat.
  r.repeatZ(4, 0.018, -0.012, y => {
    r.add(new THREE.TorusGeometry(0.0225, 0.0022, 5, 14), 'poly', 0, y, 0, Math.PI * 0.5);
  });
  r.pin(0.010, 0.010, 0.016, 'worn', 0, 0.040, 0);                      // fuse assembly
  r.box(0.010, 0.052, 0.006, 'worn', 0, 0.014, 0.021);                  // spoon
  r.ring(0.008, 0.0018, 'worn', 0.014, 0.044, 0, 10);                   // pull ring
  return r.build(mats);
}

/* ========================================================================== */
/*                              weapon definitions                            */
/* ========================================================================== */

// Recoil patterns are authored, not random: the first eight shots of a mag must
// be identical every time so the pattern is learnable — that is the entire skill
// ceiling of an FPS. Horizontal values are in degrees, sign is the drift side.
const WEAPONS = [
  {
    id: 'ar', name: 'M4A1', build: buildRifle, sound: 'rifle',
    fireMode: 'auto', rpm: 700, magSize: 30, reserve: 210,
    damage: 32, falloff: { start: 22, end: 60, min: 0.55 },
    adsFov: 22, viewFovCut: 6, adsTime: 0.20, eyeRelief: 0.088,
    hipPos: [0.108, -0.082, -0.255], hipRot: [0.03, -0.10, 0.035],
    spread: { hip: 2.4, ads: 0.28, move: 2.6, air: 3.4, perShot: 0.34, max: 5.5, decay: 5.0 },
    recoil: {
      vert: 0.52, vertRamp: 0.055, vertMax: 1.7, horiz: 0.30,
      pattern: [0.05, -0.1, 0.15, 0.45, 0.75, 0.55, -0.25, -0.75],
      kickBack: 0.020, rise: 0.055, roll: 0.030, snap: 0.72,
    },
    reload: 2.10, reloadEmpty: 2.60,
  },
  {
    id: 'smg', name: 'MP5A5', build: buildSMG, sound: 'smg',
    fireMode: 'auto', rpm: 900, magSize: 30, reserve: 240,
    damage: 24, falloff: { start: 12, end: 34, min: 0.40 },
    adsFov: 16, viewFovCut: 4, adsTime: 0.14, eyeRelief: 0.062,
    hipPos: [0.100, -0.076, -0.230], hipRot: [0.035, -0.11, 0.045],
    spread: { hip: 3.4, ads: 0.55, move: 3.0, air: 4.0, perShot: 0.30, max: 6.6, decay: 6.0 },
    recoil: {
      vert: 0.40, vertRamp: 0.042, vertMax: 1.35, horiz: 0.42,
      pattern: [-0.08, 0.18, 0.42, 0.10, -0.35, -0.62, -0.30, 0.35],
      kickBack: 0.015, rise: 0.042, roll: 0.024, snap: 0.66,
    },
    reload: 2.05, reloadEmpty: 2.55,
  },
  {
    id: 'dmr', name: 'SR-25', build: buildSniper, sound: 'sniper',
    fireMode: 'semi', rpm: 260, magSize: 10, reserve: 80,
    damage: 92, falloff: { start: 80, end: 160, min: 0.80 },
    adsFov: 40, viewFovCut: 14, adsTime: 0.30, eyeRelief: 0.148,
    hipPos: [0.135, -0.095, -0.285], hipRot: [0.04, -0.13, 0.05],
    spread: { hip: 6.5, ads: 0.06, move: 4.2, air: 6.0, perShot: 1.60, max: 11.0, decay: 3.2 },
    recoil: {
      vert: 1.55, vertRamp: 0.10, vertMax: 2.6, horiz: 0.45,
      pattern: [0.10, -0.40, 0.55, -0.30, 0.45, -0.55, 0.25, -0.20],
      kickBack: 0.048, rise: 0.135, roll: 0.055, snap: 0.80,
    },
    reload: 2.35, reloadEmpty: 2.85,
  },
];

/* ========================================================================== */
/*                                the system                                  */
/* ========================================================================== */

export class WeaponSystem {
  constructor({ camera, viewCamera, scene, textures, audio, particles, decals, level, settings }) {
    this.camera = camera;
    this.viewCamera = viewCamera;
    this.scene = scene;
    this.textures = textures;
    this.audio = audio;
    this.particles = particles;
    this.decals = decals;
    this.level = level;
    this.settings = settings || {};

    this.enemies = null;          // wired by main.js once the AI exists
    this.onHit = null;

    /* --------------------------------------------------- public HUD state */
    this.adsProgress = 0;
    this.adsFovReduction = WEAPONS[0].adsFov;
    this.exposureBoost = 1.0;
    this.spread = 0;
    this.current = null;
    this.grenades = 3;

    /* -------------------------------------------------------- view scene */
    this.viewScene = new THREE.Scene();
    this._buildViewLights();

    this.rig = new THREE.Group();          // sway, bob, ADS, sprint pose
    this.recoilNode = new THREE.Group();   // kick springs only, so they compose cleanly
    this.rig.add(this.recoilNode);
    this.viewScene.add(this.rig);

    this._materials = this._buildMaterials();
    this._buildWeapons();
    this._buildKnife();
    this._buildFlash();
    this._buildShells();
    this._buildGrenades();

    /* --------------------------------------------------- internal state */
    this.index = 0;
    this.weapon = this.weapons[0];
    this.current = this.weapon.state;

    this._t = 0;
    this._fireTimer = 0;
    this._triggerHeld = false;
    this._shotIndex = 0;           // position in the recoil pattern
    this._spreadHeat = 0;
    this._lastFire = -10;

    this._reloadT = 0; this._reloadDur = 0; this._reloadEmpty = false;
    this._switchT = 0; this._switchTo = -1;
    this._inspectT = 0;
    this._meleeT = 0; this._meleeDone = false;
    this._grenadeCook = 0; this._grenadeHeld = false;

    this._adsRaw = 0;
    this._sprintBlend = 0;
    this._bobPhase = 0;
    this._breathe = 0;

    // Recoil is split into a fast spring (snap, fully recovered) and a slow
    // residual (aim punch). Only the residual can move where the bullets go.
    this._kick = { x: 0, y: 0, vx: 0, vy: 0 };
    this._aimBias = { x: 0, y: 0 };
    this._appliedRot = new THREE.Vector3();
    this._writtenRot = new THREE.Vector3(NaN, NaN, NaN);
    this._prevYaw = 0; this._prevPitch = 0;

    this._lag = { x: 0, y: 0, vx: 0, vy: 0 };
    this._vmKick = { z: 0, vz: 0, p: 0, vp: 0, r: 0, vr: 0 };
    this._shake = 0;
    this._cycleT = 0;
    this._shotDir = new THREE.Vector3();
    this._shotOrigin = new THREE.Vector3();

    this.reset();
  }

  /* ------------------------------------------------------------ assembly */

  _buildViewLights() {
    // The viewmodel scene is rendered standalone with a cleared depth buffer, so
    // it sees none of the world's lights and needs its own three-point set.
    const key = new THREE.DirectionalLight(0xfff2e0, 2.4);
    key.position.set(0.55, 1.0, 0.75);
    const fill = new THREE.DirectionalLight(0x93b0d8, 0.85);
    fill.position.set(-0.9, -0.15, 0.45);
    const rim = new THREE.DirectionalLight(0xffffff, 1.5);
    rim.position.set(-0.35, 0.35, -1.0);
    const amb = new THREE.HemisphereLight(0xa8c4e8, 0x1a1712, 0.55);
    this.viewScene.add(key, fill, rim, amb);
    this.viewScene.environment = this.scene?.environment || null;
  }

  _mat(name, opts) {
    // The library is built before us, but a missing entry must not take the
    // whole game down — a flat standard material still renders a usable gun.
    try {
      return makeStandardMaterial(this.textures, name, opts);
    } catch {
      const { repeat, repeatY, ...rest } = opts || {};
      return new THREE.MeshStandardMaterial({ color: 0x3a3d42, roughness: 0.6, metalness: 0.8, ...rest });
    }
  }

  _buildMaterials() {
    const flash = flashTexture();
    this._flashTex = flash;
    return {
      gun: this._mat('gunmetal', { repeat: 3, envMapIntensity: 1.25 }),
      worn: this._mat('metal', { repeat: 4, color: 0x8d9299, roughness: 0.85, envMapIntensity: 1.5 }),
      // Polymer furniture: same synthesised surface, but forced dielectric and
      // rough so it never picks up the receiver's steel highlight.
      poly: this._mat('gunmetal', { repeat: 5, color: 0x2b2e2c, metalness: 0.0, roughness: 1.25 }),
      wood: this._mat('wood', { repeat: 2, color: 0x9a7a52, metalness: 0.0, roughness: 1.0 }),
      rubber: this._mat('fabric', { repeat: 3, color: 0x2a2a28, metalness: 0.0, roughness: 1.15 }),
      glass: new THREE.MeshStandardMaterial({
        color: 0x142838, roughness: 0.06, metalness: 0.0,
        transparent: true, opacity: 0.42, envMapIntensity: 2.2, side: THREE.DoubleSide,
      }),
      dot: new THREE.MeshBasicMaterial({
        color: 0xff2a12, blending: THREE.AdditiveBlending,
        transparent: true, depthWrite: false, toneMapped: false, side: THREE.DoubleSide,
      }),
      flash: new THREE.MeshBasicMaterial({
        map: flash, color: 0xffd9a0, blending: THREE.AdditiveBlending,
        transparent: true, depthWrite: false, depthTest: false, toneMapped: false, side: THREE.DoubleSide,
      }),
      brass: new THREE.MeshStandardMaterial({ color: 0xb08b3a, roughness: 0.32, metalness: 1.0 }),
    };
  }

  _buildWeapons() {
    this.weapons = WEAPONS.map(def => {
      const built = def.build(this._materials);
      built.group.visible = false;
      this.recoilNode.add(built.group);
      return {
        def,
        group: built.group,
        parts: built.parts,
        muzzle: built.muzzle,
        eject: built.eject,
        state: {
          name: def.name, ammo: def.magSize, reserve: def.reserve,
          magSize: def.magSize, fireMode: def.fireMode,
        },
      };
    });
  }

  _buildKnife() {
    this.knife = buildKnife(this._materials);
    this.knife.visible = false;
    this.rig.add(this.knife);

    this.handGrenade = buildGrenade(this._materials);
    this.handGrenade.visible = false;
    this.rig.add(this.handGrenade);
  }

  _buildFlash() {
    this.flashGroup = new THREE.Group();
    this.flashGroup.visible = false;
    // Two crossed cards instead of a billboard: the flash keeps volume when the
    // recoil roll spins the viewmodel, and costs nothing to update.
    const card = new THREE.Mesh(new THREE.PlaneGeometry(0.24, 0.24), this._materials.flash);
    const cross = new THREE.Mesh(new THREE.PlaneGeometry(0.20, 0.20), this._materials.flash);
    cross.rotation.y = Math.PI * 0.5;
    card.frustumCulled = cross.frustumCulled = false;
    card.renderOrder = cross.renderOrder = 10;
    this.flashGroup.add(card, cross);

    // Range is tiny in view space; the same flash gets a second, world-scale
    // light so the muzzle actually lifts the geometry around the player.
    this.flashLight = new THREE.PointLight(0xffb066, 0, 1.4, 2);
    this.flashGroup.add(this.flashLight);
    this.recoilNode.add(this.flashGroup);

    this.worldFlash = new THREE.PointLight(0xffb066, 0, 16, 2);
    this.worldFlash.visible = false;
    this.scene?.add(this.worldFlash);
    this._flashT = 0;
  }

  _buildShells() {
    const r = new Rack();
    r.tube(0.0043, 0.0047, 0.019, 'brass', 0, 0, 0, 8);
    r.ring(0.0048, 0.0009, 'brass', 0, 0, 0.0085, 8);
    const geo = mergeItems([...r.bins.values()].flat());

    this.shells = [];
    for (let i = 0; i < SHELL_COUNT; i++) {
      const m = new THREE.Mesh(geo, this._materials.brass);
      m.visible = false;
      m.frustumCulled = false;
      this.viewScene.add(m);
      this.shells.push({
        mesh: m, life: 0,
        vel: new THREE.Vector3(), spin: new THREE.Vector3(), bounced: false,
      });
    }
    this._shellCursor = 0;
  }

  _buildGrenades() {
    this.thrown = [];
    for (let i = 0; i < GRENADE_POOL; i++) {
      const mesh = buildGrenade(this._materials);
      mesh.visible = false;
      this.scene?.add(mesh);
      this.thrown.push({
        mesh, active: false, fuse: 0,
        pos: new THREE.Vector3(), vel: new THREE.Vector3(), spin: new THREE.Vector3(),
      });
    }
  }

  /* --------------------------------------------------------------- reset */

  reset() {
    for (const w of this.weapons) {
      w.state.ammo = w.def.magSize;
      w.state.reserve = w.def.reserve;
      w.group.visible = false;
    }
    this.index = 0;
    this.weapon = this.weapons[0];
    this.current = this.weapon.state;
    this.weapon.group.visible = true;
    this.adsFovReduction = this.weapon.def.adsFov;

    this.grenades = 3;
    this.adsProgress = 0; this._adsRaw = 0;
    this.exposureBoost = 1.0;
    this._reloadT = 0; this._switchT = 0; this._inspectT = 0; this._meleeT = 0;
    this._grenadeCook = 0; this._grenadeHeld = false;
    this._shotIndex = 0; this._spreadHeat = 0; this._fireTimer = 0;
    this._kick.x = this._kick.y = this._kick.vx = this._kick.vy = 0;
    this._aimBias.x = this._aimBias.y = 0;
    this._vmKick.z = this._vmKick.p = this._vmKick.r = 0;
    this._vmKick.vz = this._vmKick.vp = this._vmKick.vr = 0;
    this._shake = 0;
    this._flashT = 0;
    this.flashGroup.visible = false;
    this.worldFlash.visible = false;
    this.knife.visible = false;
    this.handGrenade.visible = false;

    for (const s of this.shells) { s.life = 0; s.mesh.visible = false; }
    for (const g of this.thrown) { g.active = false; g.mesh.visible = false; }

    this._writtenRot.set(NaN, NaN, NaN);
    this._appliedRot.set(0, 0, 0);
    this._resetParts();
    this._placeRig();
  }

  _resetParts() {
    for (const w of this.weapons) {
      const p = w.parts;
      if (p.mag) { p.mag.position.set(0, 0, 0); p.mag.rotation.set(0, 0, 0); p.mag.visible = true; }
      if (p.charge) p.charge.position.z = 0;
      if (p.bolt) p.bolt.position.z = 0;
    }
  }

  /* -------------------------------------------------------------- update */

  update(dt, input, player) {
    this._t += dt;
    if (this.viewScene.environment !== this.scene?.environment) {
      this.viewScene.environment = this.scene?.environment || null;
    }

    // Recover the controller's own aim before touching it: if nobody rewrote
    // camera.rotation since our last write, our offset is still baked into it.
    const rot = this.camera.rotation;
    const baseX = rot.x === this._writtenRot.x ? rot.x - this._appliedRot.x : rot.x;
    const baseY = rot.y === this._writtenRot.y ? rot.y - this._appliedRot.y : rot.y;
    const baseZ = rot.z === this._writtenRot.z ? rot.z - this._appliedRot.z : rot.z;

    const yawDelta = this._wrapPi(baseY - this._prevYaw);
    const pitchDelta = baseX - this._prevPitch;
    this._prevYaw = baseY; this._prevPitch = baseX;

    const busy = this._reloadT > 0 || this._switchT > 0 || this._meleeT > 0;
    const speed = this._speedOf(player);
    const grounded = player?.grounded ?? player?.onGround ?? true;
    const sprinting = !!(player?.sprinting) && speed > 1.5;

    this._handleSlots(input);
    this._handleActions(input, player, dt);
    this._updateAds(dt, input, busy, sprinting);
    this._updateFiring(dt, input, player, busy);
    this._updateSpread(dt, speed, grounded);
    this._updateRecoil(dt);
    this._updateTimers(dt);
    this._updateViewmodel(dt, speed, grounded, sprinting, yawDelta, pitchDelta, input);
    this._updateShells(dt);
    this._updateThrown(dt);
    this._updateFlash(dt);

    // Apply recoil last so the viewmodel lag reads the clean camera delta.
    const shakeX = this._shake * Math.sin(this._t * 61.0) * 0.5;
    const shakeY = this._shake * Math.sin(this._t * 47.0) * 0.5;
    this._appliedRot.set(
      this._kick.x + this._aimBias.x + shakeX,
      this._kick.y + this._aimBias.y + shakeY,
      this._vmKick.r * 0.25,
    );
    rot.x = clamp(baseX + this._appliedRot.x, -1.553, 1.553);
    rot.y = baseY + this._appliedRot.y;
    rot.z = baseZ + this._appliedRot.z;
    this._writtenRot.set(rot.x, rot.y, rot.z);

    this.current = this.weapon.state;
  }

  _wrapPi(a) { return a > Math.PI ? a - Math.PI * 2 : a < -Math.PI ? a + Math.PI * 2 : a; }

  _speedOf(player) {
    if (typeof player?.speed === 'number') return player.speed;
    const v = player?.velocity;
    return v ? Math.hypot(v.x || 0, v.z || 0) : 0;
  }

  /* ------------------------------------------------------- weapon switch */

  _handleSlots(input) {
    for (let i = 0; i < 3; i++) {
      if (!input.justPressed('slot' + (i + 1))) continue;
      if (i === this.index || this._switchT > 0 || i >= this.weapons.length) continue;
      this._switchTo = i;
      this._switchT = 0.62;
      this._reloadT = 0;
      this._inspectT = 0;
      this.audio?.play?.('switch');
    }
  }

  _handleActions(input, player, dt) {
    const w = this.weapon;
    const canAct = this._switchT <= 0 && this._meleeT <= 0;

    if (canAct && input.justPressed('reload')) this._startReload();

    if (canAct && input.justPressed('inspect') && this._reloadT <= 0) this._inspectT = 2.4;

    if (input.justPressed('melee') && this._meleeT <= 0 && this._switchT <= 0) {
      this._meleeT = 0.62;
      this._meleeDone = false;
      this._reloadT = 0;
      this.audio?.play?.('melee_swing');
    }

    // Grenades cook while held: releasing early throws long, holding throws a
    // shorter, hotter one. Cooking past the fuse detonates in hand.
    const gDown = input.down('grenade');
    if (gDown && !this._grenadeHeld && this.grenades > 0 && this._switchT <= 0) {
      this._grenadeHeld = true;
      this._grenadeCook = 0;
      this._reloadT = 0;
      this.audio?.play?.('grenade_pin');
    }
    if (this._grenadeHeld) {
      this._grenadeCook += dt;
      if (!gDown) { this._throwGrenade(player, clamp(1 - this._grenadeCook * 0.18, 0.55, 1)); }
      else if (this._grenadeCook >= 3.6) { this._throwGrenade(player, 0.12); }
    }

    if (w.state.ammo <= 0 && w.state.reserve > 0 && this._reloadT <= 0 && canAct &&
        input.down('fire') && this._fireTimer <= 0) {
      this._startReload();
    }
  }

  _startReload() {
    const w = this.weapon;
    if (this._reloadT > 0 || w.state.ammo >= w.def.magSize || w.state.reserve <= 0) return;
    this._reloadEmpty = w.state.ammo <= 0;
    this._reloadDur = this._reloadEmpty ? w.def.reloadEmpty : w.def.reload;
    this._reloadT = this._reloadDur;
    this._ammoLoaded = false;
    this.audio?.play?.('reload_start');
  }

  _finishReload() {
    const w = this.weapon;
    const want = w.def.magSize - w.state.ammo;
    const take = Math.min(want, w.state.reserve);
    w.state.ammo += take;
    w.state.reserve -= take;
    this._shotIndex = 0;
  }

  _updateTimers(dt) {
    if (this._switchT > 0) {
      const prev = this._switchT;
      this._switchT = Math.max(0, this._switchT - dt);
      // Swap models at the bottom of the lower, while the gun is off-screen.
      if (prev > 0.31 && this._switchT <= 0.31 && this._switchTo >= 0) {
        this.weapon.group.visible = false;
        this.index = this._switchTo;
        this.weapon = this.weapons[this.index];
        this.weapon.group.visible = true;
        this.current = this.weapon.state;
        this.adsFovReduction = this.weapon.def.adsFov;
        this._switchTo = -1;
        this._shotIndex = 0;
        this._spreadHeat = 0;
      }
    }

    if (this._reloadT > 0) {
      this._reloadT = Math.max(0, this._reloadT - dt);
      const p = 1 - this._reloadT / this._reloadDur;
      if (!this._ammoLoaded && p > 0.62) { this._ammoLoaded = true; this._finishReload(); }
    }

    if (this._inspectT > 0) this._inspectT = Math.max(0, this._inspectT - dt);

    if (this._meleeT > 0) {
      this._meleeT = Math.max(0, this._meleeT - dt);
      const elapsed = 0.62 - this._meleeT;
      if (!this._meleeDone && elapsed > 0.18) { this._meleeDone = true; this._doMelee(); }
    }

    if (this._cycleT > 0) this._cycleT = Math.max(0, this._cycleT - dt);
    this._shake = Math.max(0, this._shake - dt * 2.4);
    this.exposureBoost = damp(this.exposureBoost, 1.0, 9, dt);
  }

  /* ----------------------------------------------------------------- ADS */

  _updateAds(dt, input, busy, sprinting) {
    const w = this.weapon.def;
    const want = input.down('ads') && !busy && !this._grenadeHeld &&
                 this._inspectT <= 0 && !(sprinting && !input.down('fire'));
    const rate = 1 / Math.max(0.05, w.adsTime);
    this._adsRaw = damp(this._adsRaw, want ? 1 : 0, rate * 3.2, dt);
    if (this._adsRaw > 0.999) this._adsRaw = 1;
    if (this._adsRaw < 0.001) this._adsRaw = 0;
    // Smoothstep on top of the exponential gives the settle a definite end,
    // instead of the asymptotic crawl that makes cheap ADS feel mushy.
    const t = this._adsRaw;
    this.adsProgress = t * t * (3 - 2 * t);
    this.adsFovReduction = w.adsFov;

    if (this.viewCamera) {
      const fov = 55 - this.adsProgress * w.viewFovCut;
      if (Math.abs(this.viewCamera.fov - fov) > 0.01) {
        this.viewCamera.fov = fov;
        this.viewCamera.updateProjectionMatrix();
      }
    }
  }

  /* -------------------------------------------------------------- firing */

  _updateFiring(dt, input, player, busy) {
    const w = this.weapon;
    const def = w.def;
    this._fireTimer -= dt;

    const held = input.down('fire');
    const pressed = input.justPressed('fire');
    const blocked = busy || this._grenadeHeld || this._sprintBlend > 0.45;

    if (!held) this._triggerHeld = false;
    if (blocked) return;

    const wantsShot = def.fireMode === 'auto' ? held : (pressed || (held && !this._triggerHeld));
    if (!wantsShot || this._fireTimer > 0) return;

    if (w.state.ammo <= 0) {
      if (pressed) {
        this.audio?.play?.('dryfire');
        this._fireTimer = 0.22;
        if (w.state.reserve > 0) this._startReload();
      }
      this._triggerHeld = true;
      return;
    }

    this._triggerHeld = true;
    this._fireTimer = 60 / def.rpm;
    if (this._inspectT > 0) this._inspectT = 0;
    this._fire(player);
  }

  _fire(player) {
    const w = this.weapon;
    const def = w.def;
    w.state.ammo--;

    this.camera.getWorldDirection(_fwd);
    _right.set(1, 0, 0).applyQuaternion(this.camera.getWorldQuaternion(_quat));
    _up.crossVectors(_right, _fwd).normalize();

    // Cone sampling: sqrt on the radius keeps the distribution area-uniform, so
    // the crosshair gap the HUD draws actually matches where rounds land.
    const half = this.spread;
    const a = Math.random() * Math.PI * 2;
    const rr = Math.sqrt(Math.random()) * half;
    // Direction and origin live on the instance, not in module scratch: the
    // trace calls into level, particle and enemy code that borrows the scratch.
    this._shotDir.copy(_fwd)
      .addScaledVector(_right, Math.cos(a) * Math.tan(rr))
      .addScaledVector(_up, Math.sin(a) * Math.tan(rr))
      .normalize();
    this._shotOrigin.copy(this.camera.position);
    this._traceShot(this._shotOrigin, this._shotDir, def);

    this._applyRecoil(def.recoil);
    this._spreadHeat = Math.min(def.spread.max, this._spreadHeat + def.spread.perShot);
    this._shotIndex++;
    this._lastFire = this._t;

    this._spawnFlash();
    this._ejectShell();
    this._cycleAction();

    this.audio?.play?.(def.sound + '_fire', { position: this.camera.position });
    this.particles?.muzzleFlash?.(this._worldMuzzle(_v3), _fwd);
  }

  _traceShot(origin, dir, def) {
    let hit = null;
    let hitDist = Infinity;

    const eh = this.enemies?.hitTest?.(origin, dir, 300);
    if (eh) { hit = eh; hitDist = eh.distance ?? origin.distanceTo(eh.point); }

    let world = null;
    const targets = this.level?.raycastTargets;
    if (targets && targets.length) {
      _ray.set(origin, dir);
      _ray.near = 0;
      _ray.far = Math.min(hitDist, 300);
      const list = _ray.intersectObjects(targets, true);
      if (list.length) world = list[0];
    }

    // Whichever surface is nearer stops the round; a wall in front of a bot
    // must eat the bullet or every corner becomes a wallbang.
    if (world && world.distance < hitDist) {
      this._worldImpact(world, dir);
      this.particles?.tracer?.(this._worldMuzzle(_v3), world.point, def.id);
      return;
    }

    if (hit) {
      const dist = hitDist;
      const f = def.falloff;
      const t = clamp((dist - f.start) / Math.max(0.001, f.end - f.start), 0, 1);
      const mult = lerp(1, f.min, t) * (hit.headshot ? HEADSHOT_MULT : 1);
      const killed = this._damageEnemy(hit.enemy, def.damage * mult, hit.point, dir, hit.headshot);

      this.particles?.impact?.(hit.point, dir, 'flesh');
      this.particles?.tracer?.(this._worldMuzzle(_v3), hit.point, def.id);
      this.audio?.play?.(hit.headshot ? 'hit_head' : 'hit_body', { position: hit.point });
      this.onHit?.({ headshot: !!hit.headshot, killed, point: hit.point });
      return;
    }

    this.particles?.tracer?.(this._worldMuzzle(_v3), _v4.copy(origin).addScaledVector(dir, 120), def.id);
  }

  _worldImpact(world, dir) {
    const n = world.face
      ? _v1.copy(world.face.normal)
          .applyNormalMatrix(_nmat.getNormalMatrix(world.object.matrixWorld)).normalize()
      : _v1.copy(dir).negate();
    const surface = world.object?.userData?.surface || 'concrete';
    this.particles?.impact?.(world.point, n, surface);
    this.decals?.add?.(world.point, n, 0.06 + Math.random() * 0.03, surface);
    this.audio?.play?.('impact_' + surface, { position: world.point });
  }

  /**
   * Enemy code is owned elsewhere and its damage entry point has changed shape
   * before; probe the three plausible spellings rather than hard-coupling.
   */
  _damageEnemy(enemy, amount, point, dir, headshot) {
    if (!enemy) return false;
    const before = typeof enemy.health === 'number' ? enemy.health
                 : typeof enemy.hp === 'number' ? enemy.hp : null;
    let ret;
    if (typeof enemy.damage === 'function') ret = enemy.damage(amount, point, dir, headshot);
    else if (typeof enemy.takeDamage === 'function') ret = enemy.takeDamage(amount, point, dir, headshot);
    else if (typeof enemy.hit === 'function') ret = enemy.hit(amount, point, dir, headshot);
    else if (before !== null) {
      if (typeof enemy.health === 'number') enemy.health -= amount; else enemy.hp -= amount;
    }
    if (typeof ret === 'boolean') return ret;
    const after = typeof enemy.health === 'number' ? enemy.health
                : typeof enemy.hp === 'number' ? enemy.hp : null;
    if (enemy.dead === true || enemy.alive === false) return true;
    return after !== null && after <= 0 && (before === null || before > 0);
  }

  /* -------------------------------------------------------------- recoil */

  _applyRecoil(rc) {
    const i = this._shotIndex;
    const vert = Math.min(rc.vertMax, rc.vert * (1 + i * rc.vertRamp)) * DEG;
    // Deterministic for the learnable part of the pattern, random after it.
    const pat = i < rc.pattern.length ? rc.pattern[i] : (Math.random() * 2 - 1) * 1.1;
    const horiz = pat * rc.horiz * DEG;

    const snap = rc.snap;
    this._kick.vx += vert * snap * 34;
    this._kick.vy += horiz * snap * 34;
    this._aimBias.x += vert * (1 - snap);
    this._aimBias.y += horiz * (1 - snap);
    this._aimBias.x = clamp(this._aimBias.x, -0.42, 0.42);

    const ads = this.adsProgress;
    const scale = lerp(1, 0.62, ads);
    this._vmKick.vz += rc.kickBack * scale * 60;
    this._vmKick.vp += rc.rise * scale * 60;
    this._vmKick.vr += (rc.roll * scale * 60) * (Math.random() < 0.5 ? -1 : 1);
    this._shake = Math.min(0.02, this._shake + rc.rise * 0.03);
  }

  _updateRecoil(dt) {
    // Critically damped spring at ~0.25s settle: kick snaps out and returns
    // without overshoot, which is what stops sustained fire feeling floaty.
    const k = 260, c = 2 * Math.sqrt(k);
    this._kick.vx += (-k * this._kick.x - c * this._kick.vx) * dt;
    this._kick.vy += (-k * this._kick.y - c * this._kick.vy) * dt;
    this._kick.x += this._kick.vx * dt;
    this._kick.y += this._kick.vy * dt;

    // The residual settles far slower and only once the trigger is released,
    // so a held burst climbs, but the gun does not drift forever afterwards.
    if (this._t - this._lastFire > 0.22) {
      const r = 1 - Math.exp(-1.6 * dt);
      this._aimBias.x -= this._aimBias.x * r;
      this._aimBias.y -= this._aimBias.y * r;
    }

    const vk = 190, vc = 2 * Math.sqrt(vk) * 0.95;
    for (const [p, v] of SPRING_AXES) {
      this._vmKick[v] += (-vk * this._vmKick[p] - vc * this._vmKick[v]) * dt;
      this._vmKick[p] += this._vmKick[v] * dt;
    }
  }

  /* -------------------------------------------------------------- spread */

  _updateSpread(dt, speed, grounded) {
    const s = this.weapon.def.spread;
    this._spreadHeat = Math.max(0, this._spreadHeat - s.decay * dt);

    const moveFactor = clamp(speed / 6.0, 0, 1);
    let deg = lerp(s.hip, s.ads, this.adsProgress);
    deg += moveFactor * s.move * lerp(1, 0.45, this.adsProgress);
    if (!grounded) deg += s.air;
    deg += this._spreadHeat * lerp(1, 0.7, this.adsProgress);
    if (this._sprintBlend > 0.2) deg += s.move * this._sprintBlend;
    this.spread = Math.min(s.max + s.air, deg) * DEG;
  }

  /* ---------------------------------------------------------- viewmodel */

  _placeRig() {
    const def = this.weapon.def;
    this.rig.position.set(def.hipPos[0], def.hipPos[1], def.hipPos[2]);
    this.rig.rotation.set(def.hipRot[0], def.hipRot[1], def.hipRot[2]);
  }

  _updateViewmodel(dt, speed, grounded, sprinting, yawDelta, pitchDelta, input) {
    const def = this.weapon.def;
    const ads = this.adsProgress;
    const busy = this._reloadT > 0 || this._switchT > 0 || this._meleeT > 0;

    // Holding fire drops the sprint pose; firing itself stays gated on the
    // blend, so pulling the trigger mid-sprint costs the ~0.12s raise time.
    const wantSprint = sprinting && !busy && ads < 0.05 &&
      !input.down('fire') && this._t - this._lastFire > 0.25 ? 1 : 0;
    this._sprintBlend = damp(this._sprintBlend, wantSprint, 9, dt);
    const sp = this._sprintBlend;

    /* --- weapon lag: the rig chases the camera, it does not ride it --------- */
    // Without this the gun is welded to the view and every turn feels weightless.
    const lagK = lerp(120, 260, ads), lagC = 2 * Math.sqrt(lagK) * 1.05;
    const gain = lerp(1.0, 0.28, ads);
    this._lag.vx += (yawDelta * 26 * gain - lagK * this._lag.x * dt - lagC * this._lag.vx * dt);
    this._lag.vy += (pitchDelta * 22 * gain - lagK * this._lag.y * dt - lagC * this._lag.vy * dt);
    this._lag.x = clamp(this._lag.x + this._lag.vx * dt, -0.10, 0.10);
    this._lag.y = clamp(this._lag.y + this._lag.vy * dt, -0.09, 0.09);

    /* --- idle breathing ---------------------------------------------------- */
    this._breathe += dt * (busy ? 1.1 : 0.85);
    const bAmp = lerp(0.0022, 0.0006, ads) * (1 - clamp(speed / 4, 0, 1) * 0.6);
    const breatheY = Math.sin(this._breathe * 1.9) * bAmp;
    const breatheX = Math.sin(this._breathe * 1.3 + 1.1) * bAmp * 0.8;

    /* --- walk / sprint bob ------------------------------------------------- */
    const moveAmt = clamp(speed / 5.5, 0, 1.35);
    this._bobPhase += dt * lerp(7.0, 11.5, clamp(moveAmt - 0.6, 0, 1)) * (grounded ? moveAmt : 0);
    const bobScale = moveAmt * lerp(1, 0.25, ads) * (grounded ? 1 : 0.25);
    const bobX = Math.cos(this._bobPhase) * 0.016 * bobScale;
    const bobY = Math.sin(this._bobPhase * 2) * 0.011 * bobScale;
    const bobRZ = Math.cos(this._bobPhase) * 0.030 * bobScale;

    /* --- target pose ------------------------------------------------------- */
    // ADS overshoot: a touch of easeOutBack past the sight line, then settle —
    // the tiny snap at the end is what makes aiming feel mechanical.
    const over = ads < 1 ? easeOutBack(clamp(ads, 0, 1)) : 1;
    let px = lerp(def.hipPos[0], 0, over);
    let py = lerp(def.hipPos[1], 0, over);
    let pz = lerp(def.hipPos[2], -def.eyeRelief, over);
    let rx = lerp(def.hipRot[0], 0, over);
    let ry = lerp(def.hipRot[1], 0, over);
    let rz = lerp(def.hipRot[2], 0, over);

    px += lerp(0.055, 0.012, ads) * sp;
    py += lerp(-0.045, -0.012, ads) * sp;
    pz += 0.030 * sp;
    rx += 0.16 * sp; ry += -0.52 * sp; rz += 0.62 * sp;

    const anim = this._poseOffsets();
    px += anim.x; py += anim.y; pz += anim.z;
    rx += anim.rx; ry += anim.ry; rz += anim.rz;

    px += (bobX + breatheX + this._lag.x) * (1 - sp * 0.5);
    py += (bobY + breatheY - this._lag.y) * (1 - sp * 0.5);
    rz += bobRZ + this._lag.x * 1.6;
    ry += this._lag.x * 2.2;
    rx += this._lag.y * 1.8;

    /* --- recoil, applied on its own node so it never fights the pose ------- */
    this.recoilNode.position.set(0, this._vmKick.p * 0.10, this._vmKick.z);
    this.recoilNode.rotation.set(-this._vmKick.p, 0, this._vmKick.r * 0.7);

    const follow = lerp(24, 40, ads);
    this.rig.position.x = damp(this.rig.position.x, px, follow, dt);
    this.rig.position.y = damp(this.rig.position.y, py, follow, dt);
    this.rig.position.z = damp(this.rig.position.z, pz, follow, dt);
    this.rig.rotation.x = damp(this.rig.rotation.x, rx, follow, dt);
    this.rig.rotation.y = damp(this.rig.rotation.y, ry, follow, dt);
    this.rig.rotation.z = damp(this.rig.rotation.z, rz, follow, dt);

    this._animateParts();
    this._animateKnife();
    this._animateHandGrenade();
  }

  /** Additive pose offsets from reload / switch / inspect / melee states. */
  _poseOffsets() {
    const o = { x: 0, y: 0, z: 0, rx: 0, ry: 0, rz: 0 };

    if (this._switchT > 0) {
      // One down-and-up arc: 0..0.31 lowers the old gun, 0.31..0.62 raises the new.
      const t = 1 - this._switchT / 0.62;
      const d = t < 0.5 ? t / 0.5 : 1 - (t - 0.5) / 0.5;
      const e = d * d * (3 - 2 * d);
      o.y -= 0.22 * e; o.z += 0.06 * e; o.rx -= 0.95 * e; o.rz += 0.35 * e;
    }

    if (this._reloadT > 0) {
      const p = 1 - this._reloadT / this._reloadDur;
      // Stages: tilt in, mag out, mag in, seat with a tug, (charge), present.
      o.y -= 0.045 * Math.sin(Math.PI * clamp(p * 1.15, 0, 1));
      o.rx += 0.30 * Math.sin(Math.PI * clamp(p * 1.05, 0, 1));
      o.rz += 0.42 * Math.sin(Math.PI * clamp(p * 1.05, 0, 1));
      o.ry += 0.16 * Math.sin(Math.PI * clamp(p * 1.05, 0, 1));
      if (p > 0.52 && p < 0.62) o.y -= 0.012 * Math.sin((p - 0.52) / 0.10 * Math.PI);   // seating tug
      if (this._reloadEmpty && p > 0.72 && p < 0.86) {
        o.rz += 0.10 * Math.sin((p - 0.72) / 0.14 * Math.PI);                            // charging handle yank
      }
    }

    if (this._inspectT > 0) {
      const p = 1 - this._inspectT / 2.4;
      const env = Math.sin(clamp(p, 0, 1) * Math.PI);
      o.z += 0.075 * env;
      o.y -= 0.020 * env;
      o.ry += Math.sin(p * Math.PI * 2) * 1.25 * env;
      o.rz += Math.sin(p * Math.PI * 2 + 1.0) * 0.55 * env;
      o.rx += Math.sin(p * Math.PI * 4) * 0.22 * env;
    }

    if (this._meleeT > 0) {
      const t = 1 - this._meleeT / 0.62;
      const e = Math.sin(clamp(t * 1.2, 0, 1) * Math.PI);
      o.x -= 0.13 * e; o.y -= 0.06 * e; o.rz += 0.85 * e; o.ry -= 0.55 * e;
    }

    if (this._grenadeHeld) {
      o.x -= 0.10; o.y -= 0.10; o.z += 0.04; o.rz += 0.55; o.ry -= 0.30;
    }

    return o;
  }

  /** Magazine, charging handle and bolt, driven off the same reload clock. */
  _animateParts() {
    const w = this.weapon;
    const p = w.parts;
    if (this._reloadT <= 0 || this._switchT > 0) {
      if (p.mag) { p.mag.visible = true; p.mag.position.set(0, 0, 0); p.mag.rotation.set(0, 0, 0); }
      // Per-shot blowback: the bolt travels a fraction of the charging handle's
      // stroke, and both are back in battery before the next round leaves.
      const c = this._cycleT > 0 ? Math.sin(this._cycleT / 0.055 * Math.PI) : 0;
      if (p.bolt) p.bolt.position.z = c * 0.030;
      if (p.charge) p.charge.position.z = c * 0.014;
      return;
    }
    const t = 1 - this._reloadT / this._reloadDur;

    if (p.mag) {
      if (t < 0.20) {
        p.mag.visible = true;
        p.mag.position.set(0, 0, 0);
      } else if (t < 0.34) {
        const k = (t - 0.20) / 0.14;
        p.mag.visible = true;
        p.mag.position.set(0, -0.22 * k * k, 0.02 * k);
        p.mag.rotation.x = 0.4 * k;
      } else if (t < 0.50) {
        p.mag.visible = false;                       // hand is off-screen fetching
      } else if (t < 0.60) {
        const k = (t - 0.50) / 0.10;
        p.mag.visible = true;
        p.mag.position.set(0.02 * (1 - k), -0.20 * (1 - k) * (1 - k), 0);
        p.mag.rotation.x = -0.35 * (1 - k);
      } else {
        p.mag.visible = true;
        p.mag.position.set(0, 0, 0);
        p.mag.rotation.set(0, 0, 0);
      }
    }

    if (p.charge && this._reloadEmpty) {
      const k = t > 0.72 && t < 0.86 ? Math.sin((t - 0.72) / 0.14 * Math.PI) : 0;
      p.charge.position.z = k * 0.075;
      if (p.bolt) p.bolt.position.z = k * 0.045;
    } else if (p.charge) {
      p.charge.position.z = 0;
      if (p.bolt) p.bolt.position.z = 0;
    }
  }

  /** Short bolt/charging-handle blowback on every shot. */
  _cycleAction() { this._cycleT = 0.055; }

  _animateKnife() {
    const show = this._meleeT > 0;
    this.knife.visible = show;
    if (!show) return;
    const t = 1 - this._meleeT / 0.62;
    const swing = Math.sin(clamp(t * 1.15, 0, 1) * Math.PI);
    this.knife.position.set(0.16 - swing * 0.30, -0.16 + swing * 0.12, -0.24 - swing * 0.16);
    this.knife.rotation.set(-0.3 + swing * 0.6, 0.9 - swing * 1.9, 0.7 - swing * 1.5);
  }

  _animateHandGrenade() {
    this.handGrenade.visible = this._grenadeHeld;
    if (!this._grenadeHeld) return;
    const c = clamp(this._grenadeCook / 3.6, 0, 1);
    this.handGrenade.position.set(-0.12, -0.20 + c * 0.02, -0.30);
    this.handGrenade.rotation.set(0.4 + c * 0.5, this._t * 0.6, -0.3);
  }

  /* ------------------------------------------------------- muzzle flash */

  _worldMuzzle(out) {
    // The flash must originate at the barrel in WORLD space, not at the eye,
    // or the light it casts falls on the wrong side of nearby cover.
    this.camera.getWorldDirection(_fwd);
    _right.set(1, 0, 0).applyQuaternion(this.camera.getWorldQuaternion(_quat));
    _up.crossVectors(_right, _fwd).normalize();
    const m = this.weapon.muzzle;
    return out.copy(this.camera.position)
      .addScaledVector(_fwd, -m.z * 0.85)
      .addScaledVector(_right, 0.10)
      .addScaledVector(_up, -0.075);
  }

  _spawnFlash() {
    const m = this.weapon.muzzle;
    this.flashGroup.position.copy(m);
    this.flashGroup.visible = true;
    // Randomised roll and scale per shot: a flash that repeats identically
    // reads as a decal, not as burning gas.
    const s = 0.7 + Math.random() * 0.75;
    this.flashGroup.scale.setScalar(s * lerp(1, 0.55, this.adsProgress));
    this.flashGroup.rotation.z = Math.random() * Math.PI * 2;
    this.flashLight.intensity = 3.2 * s;
    this._flashT = FLASH_LIFE;

    this.worldFlash.position.copy(this._worldMuzzle(_v3));
    this.worldFlash.intensity = 26 * s;
    this.worldFlash.visible = true;
    this.exposureBoost = Math.min(1.35, this.exposureBoost + 0.16);
  }

  _updateFlash(dt) {
    if (this._flashT <= 0) return;
    this._flashT -= dt;
    const k = clamp(this._flashT / FLASH_LIFE, 0, 1);
    this.flashLight.intensity *= k;
    this.worldFlash.intensity *= k;
    if (this._flashT <= 0) {
      this.flashGroup.visible = false;
      this.worldFlash.visible = false;
      this.worldFlash.intensity = 0;
    }
  }

  /* -------------------------------------------------------------- shells */

  _ejectShell() {
    const s = this.shells[this._shellCursor];
    this._shellCursor = (this._shellCursor + 1) % this.shells.length;

    const e = this.weapon.eject;
    s.mesh.position.copy(this.rig.position).add(e);
    s.mesh.rotation.set(Math.random() * 3, Math.random() * 3, Math.random() * 3);
    s.mesh.visible = true;
    s.life = 1.5;
    s.bounced = false;
    s.vel.set(1.35 + Math.random() * 0.5, 0.9 + Math.random() * 0.4, 0.5 + Math.random() * 0.4);
    s.spin.set((Math.random() - 0.5) * 34, (Math.random() - 0.5) * 26, (Math.random() - 0.5) * 30);

    this.particles?.shell?.(this._worldMuzzle(_v3), s.vel, this.weapon.def.id);
  }

  _updateShells(dt) {
    for (const s of this.shells) {
      if (s.life <= 0) continue;
      s.life -= dt;
      if (s.life <= 0) { s.mesh.visible = false; continue; }

      s.vel.y -= 7.2 * dt;
      s.mesh.position.addScaledVector(s.vel, dt);
      s.mesh.rotation.x += s.spin.x * dt;
      s.mesh.rotation.y += s.spin.y * dt;
      s.mesh.rotation.z += s.spin.z * dt;

      // One bounce off an imagined floor plane below the view — brass that just
      // falls out of frame looks weightless; brass that skips reads as metal.
      if (!s.bounced && s.mesh.position.y < -0.42) {
        s.mesh.position.y = -0.42;
        s.vel.y = Math.abs(s.vel.y) * 0.38;
        s.vel.x *= 0.7; s.vel.z *= 0.7;
        s.spin.multiplyScalar(0.55);
        s.bounced = true;
        this.audio?.play?.('shell');
      }
      if (s.life < 0.35) s.mesh.scale.setScalar(clamp(s.life / 0.35, 0, 1));
      else s.mesh.scale.setScalar(1);
    }
  }

  /* ------------------------------------------------------------ grenades */

  _throwGrenade(player, power) {
    this._grenadeHeld = false;
    const cook = this._grenadeCook;
    this._grenadeCook = 0;
    if (this.grenades <= 0) return;
    this.grenades--;

    const g = this.thrown.find(t => !t.active) || this.thrown[0];
    this.camera.getWorldDirection(_fwd);
    _right.set(1, 0, 0).applyQuaternion(this.camera.getWorldQuaternion(_quat));
    _up.crossVectors(_right, _fwd).normalize();

    g.pos.copy(this.camera.position).addScaledVector(_fwd, 0.35).addScaledVector(_right, 0.12);
    // Inherit the thrower's motion, otherwise a grenade thrown while sprinting
    // appears to be left behind in mid-air.
    g.vel.copy(_fwd).multiplyScalar(17 * power).addScaledVector(_up, 3.2 * power);
    const pv = player?.velocity;
    if (pv) g.vel.add(_v1.set(pv.x || 0, 0, pv.z || 0));
    g.spin.set((Math.random() - 0.5) * 12, (Math.random() - 0.5) * 12, (Math.random() - 0.5) * 12);
    g.fuse = Math.max(0.25, 3.6 - cook);
    g.active = true;
    g.mesh.visible = true;
    g.mesh.position.copy(g.pos);

    this.audio?.play?.('grenade_throw');
  }

  _updateThrown(dt) {
    for (const g of this.thrown) {
      if (!g.active) continue;
      g.fuse -= dt;
      if (g.fuse <= 0) { this._explode(g); continue; }

      g.vel.y -= 9.8 * dt;
      _v1.copy(g.vel).multiplyScalar(dt);
      const step = _v1.length();

      const targets = this.level?.raycastTargets;
      if (step > 1e-5 && targets && targets.length) {
        _ray.set(g.pos, _v2.copy(_v1).normalize());
        _ray.near = 0;
        _ray.far = step + 0.06;
        const list = _ray.intersectObjects(targets, true);
        if (list.length) {
          const n = list[0].face
            ? _v3.copy(list[0].face.normal)
                .applyNormalMatrix(_nmat.getNormalMatrix(list[0].object.matrixWorld)).normalize()
            : _v3.set(0, 1, 0);
          g.pos.copy(list[0].point).addScaledVector(n, 0.05);
          g.vel.reflect(n).multiplyScalar(0.38);
          g.spin.multiplyScalar(0.5);
          if (g.vel.lengthSq() > 1) this.audio?.play?.('grenade_bounce', { position: g.pos });
        } else {
          g.pos.add(_v1);
        }
      } else {
        g.pos.add(_v1);
      }

      g.mesh.position.copy(g.pos);
      g.mesh.rotation.x += g.spin.x * dt;
      g.mesh.rotation.y += g.spin.y * dt;
      g.mesh.rotation.z += g.spin.z * dt;
    }
  }

  _explode(g) {
    g.active = false;
    g.mesh.visible = false;
    const RADIUS = 6.5, DAMAGE = 160;

    this.particles?.explosion?.(g.pos, RADIUS);
    this.audio?.play?.('explosion', { position: g.pos });

    const dist = this.camera.position.distanceTo(g.pos);
    if (dist < RADIUS * 2.5) {
      this._shake = Math.min(0.09, this._shake + 0.075 * (1 - dist / (RADIUS * 2.5)));
      this.exposureBoost = Math.min(1.6, this.exposureBoost + 0.5 * (1 - dist / (RADIUS * 2.5)));
    }

    for (const e of this._enemyList()) {
      const p = e.position || e.mesh?.position;
      if (!p) continue;
      const d = p.distanceTo(g.pos);
      if (d > RADIUS) continue;
      const falloff = 1 - d / RADIUS;
      const killed = this._damageEnemy(e, DAMAGE * falloff * falloff, p, _v1.copy(p).sub(g.pos).normalize(), false);
      this.onHit?.({ headshot: false, killed, point: p });
    }
  }

  _enemyList() {
    const m = this.enemies;
    if (!m) return [];
    const list = m.list || m.enemies || m.active || m.alive;
    return Array.isArray(list) ? list : [];
  }

  /* ----------------------------------------------------------------- melee */

  _doMelee() {
    this.camera.getWorldDirection(_fwd);
    const hit = this.enemies?.hitTest?.(this.camera.position, _fwd, 2.4);
    if (hit) {
      const killed = this._damageEnemy(hit.enemy, 150, hit.point, _fwd, false);
      this.particles?.impact?.(hit.point, _fwd, 'flesh');
      this.audio?.play?.('melee_hit', { position: hit.point });
      this.onHit?.({ headshot: false, killed, point: hit.point });
      return;
    }
    const targets = this.level?.raycastTargets;
    if (targets && targets.length) {
      _ray.set(this.camera.position, _fwd);
      _ray.near = 0; _ray.far = 2.0;
      const list = _ray.intersectObjects(targets, true);
      if (list.length) this._worldImpact(list[0], _fwd);
    }
  }

  /* ---------------------------------------------------------------- teardown */

  dispose() {
    this.viewScene.traverse(o => {
      if (o.isMesh) o.geometry?.dispose();
    });
    for (const m of Object.values(this._materials)) m.dispose?.();
    this._flashTex?.dispose();
    this.scene?.remove(this.worldFlash);
    for (const g of this.thrown) this.scene?.remove(g.mesh);
  }
}

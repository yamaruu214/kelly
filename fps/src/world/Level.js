// Level.js — the arena: procedural geometry, merged draw calls, collision AABBs.
//
// Nothing here is authored by hand in a DCC tool; every wall, stair and barrel
// is generated from a small table of world-space rectangles. Two constraints
// shape the whole file:
//
//   1. Draw calls. Static geometry is accumulated into per-material buckets and
//      concatenated into one BufferGeometry each, so the entire outpost is ~20
//      opaque draws. Repeated props go through InstancedMesh instead.
//   2. Texel density. UVs are written in *metres*, not 0..1, so a 24 m wall and
//      a 1 m crate show the same grain size. Every material declares the world
//      size of one texture tile in TILE and the geometry helpers divide by it.
//
// Collision is AABB-only, so every structure is axis-aligned. Window openings
// deliberately do not split colliders — a 1 m sill is not walkable anyway, and
// one box per wall instead of seven is the difference between 90 colliders and
// 600 for the same building.

import * as THREE from 'three';
import { makeStandardMaterial, fbm } from './Textures.js';

/* World size, in metres, of one repetition of each texture. Chosen so the
   synthesised detail lands at its real scale: 12 brick courses over 2 m gives
   16 cm courses, 16 corrugations over 1.05 m gives a 6.5 cm pitch. */
const TILE = {
  concrete: 2.6, brick: 2.0, sand: 6.0, asphalt: 6.0, wood: 1.3,
  corrugated: 1.05, metal: 1.6, metalOlive: 2.2, fabric: 0.9, gunmetal: 0.7,
};

/* Impact FX only care about the physical family, not which of the ten library
   materials was used, so several materials collapse onto one tag. Sandbags
   report 'sand' because a hit should throw a sand puff, not a cloth ripple. */
const SURFACE = {
  concrete: 'concrete', brick: 'concrete', asphalt: 'concrete',
  sand: 'sand', wood: 'wood', fabric: 'sand',
  metal: 'metal', metalOlive: 'metal', corrugated: 'metal', gunmetal: 'metal',
  glass: 'glass',
};

const HALF = 70;            // playable half-extent; colliders seal the boundary
const STEP_RISE = 0.36;     // conservative — any capsule controller steps 0.4
const STEP_RUN = 0.60;

const clamp01 = v => v < 0 ? 0 : v > 1 ? 1 : v;
const smoothstep = (e0, e1, x) => { const t = clamp01((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t); };

/** Deterministic PRNG — prop scatter must be identical on every device so that
    a collider generated on a phone matches the one a desktop player sees. */
function mulberry32(a) {
  return function () {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ------------------------------------------------------------ merge helper */

/**
 * Concatenates BufferGeometries that share the position/normal/uv layout.
 * BufferGeometryUtils lives in the addons bundle, which is not vendored, so
 * this is the 40-line subset the level actually needs. Sources are disposed
 * because the merged copy is the only one that reaches the GPU.
 */
function mergeGeometries(geoms) {
  let vCount = 0, iCount = 0;
  for (const g of geoms) {
    if (!g.attributes.normal) g.computeVertexNormals();
    vCount += g.attributes.position.count;
    iCount += g.index ? g.index.count : g.attributes.position.count;
  }
  const pos = new Float32Array(vCount * 3);
  const nor = new Float32Array(vCount * 3);
  const uvs = new Float32Array(vCount * 2);
  const idx = vCount > 65535 ? new Uint32Array(iCount) : new Uint16Array(iCount);

  let vo = 0, io = 0;
  for (const g of geoms) {
    const p = g.attributes.position, n = g.attributes.normal, t = g.attributes.uv;
    const c = p.count;
    for (let i = 0; i < c; i++) {
      pos[(vo + i) * 3] = p.getX(i); pos[(vo + i) * 3 + 1] = p.getY(i); pos[(vo + i) * 3 + 2] = p.getZ(i);
      nor[(vo + i) * 3] = n.getX(i); nor[(vo + i) * 3 + 1] = n.getY(i); nor[(vo + i) * 3 + 2] = n.getZ(i);
      if (t) { uvs[(vo + i) * 2] = t.getX(i); uvs[(vo + i) * 2 + 1] = t.getY(i); }
    }
    if (g.index) {
      const gi = g.index;
      for (let i = 0; i < gi.count; i++) idx[io + i] = gi.getX(i) + vo;
      io += gi.count;
    } else {
      for (let i = 0; i < c; i++) idx[io + i] = vo + i;
      io += c;
    }
    vo += c;
    g.dispose();
  }

  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  out.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  out.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  out.setIndex(new THREE.BufferAttribute(idx, 1));
  out.computeBoundingSphere();
  out.computeBoundingBox();
  return out;
}

/* -------------------------------------------------------- geometry helpers */

/**
 * Box with per-face metric UVs. BoxGeometry emits its six faces in the fixed
 * order +X,-X,+Y,-Y,+Z,-Z with four vertices each, so each face can be given
 * the two world dimensions that actually lie in its plane.
 */
function boxGeo(w, h, d, tile, opt = {}) {
  const g = new THREE.BoxGeometry(w, h, d);
  const uv = g.attributes.uv;
  const dims = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]];
  for (let f = 0; f < 6; f++) {
    // Container corrugations run vertically, so the side faces swap u and v.
    const swap = opt.swapSides && f !== 2 && f !== 3;
    const a = dims[f][swap ? 1 : 0], b = dims[f][swap ? 0 : 1];
    for (let i = 0; i < 4; i++) {
      const k = f * 4 + i;
      const u = uv.getX(k), v = uv.getY(k);
      uv.setXY(k, (swap ? v : u) * a / tile, (swap ? u : v) * b / tile);
    }
  }
  return g;
}

function planeGeo(w, d, tile, segW = 1, segD = 1) {
  const g = new THREE.PlaneGeometry(w, d, segW, segD).rotateX(-Math.PI / 2);
  const uv = g.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * w / tile, uv.getY(i) * d / tile);
  return g;
}

function cylGeo(rTop, rBot, h, seg, tile, opt = {}) {
  const g = new THREE.CylinderGeometry(rTop, rBot, h, seg, 1, !!opt.open);
  const uv = g.attributes.uv;
  const torso = (seg + 1) * 2;
  const circ = Math.PI * (rTop + rBot);
  const capD = Math.max(rTop, rBot) * 2;
  for (let i = 0; i < uv.count; i++) {
    const su = i < torso ? circ / tile : capD / tile;
    const sv = i < torso ? h / tile : capD / tile;
    uv.setXY(i, uv.getX(i) * su, uv.getY(i) * sv);
  }
  return g;
}

function scaleUV(g, su, sv) {
  const uv = g.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * su, uv.getY(i) * sv);
  return g;
}

/* ------------------------------------------------------------ site plan */
// Footprints are declared up front because the terrain has to be flattened
// under them before a single vertex of ground is written.

const BUILDINGS = {
  warehouse: { x: -40, z: -34, w: 24, d: 16 },
  apartment: { x: -38, z: 28, w: 16, d: 13 },
  admin:     { x: 36, z: 34, w: 13, d: 13 },
  garage:    { x: 30, z: -30, w: 18, d: 12 },
  shops:     { x: -58, z: -4, w: 10, d: 26 },
  guard:     { x: 54, z: 30, w: 8, d: 8 },
  depot:     { x: -20, z: 48, w: 12, d: 9 },
  ruin:      { x: -8, z: -50, w: 16, d: 11 },
  shed:      { x: 54, z: -48, w: 9, d: 7 },
};

const ROADS = [
  { x0: -66, z0: 10, x1: 66, z1: 20, y: 0.045 },     // the long east-west lane
  { x0: 6, z0: -66, x1: 16, z1: 10, y: 0.050 },
  { x0: 6, z0: 20, x1: 16, z1: 66, y: 0.050 },
  { x0: 18, z0: -40, x1: 44, z1: -20, y: 0.055 },    // garage apron
  { x0: -56, z0: -26, x1: -26, z1: -17, y: 0.055 },  // warehouse loading yard
  { x0: 20, z0: -20, x1: 36, z1: -2, y: 0.060 },     // container yard
];

const FLAT = [];
for (const b of Object.values(BUILDINGS)) {
  FLAT.push({ x0: b.x - b.w / 2 - 3, z0: b.z - b.d / 2 - 3, x1: b.x + b.w / 2 + 3, z1: b.z + b.d / 2 + 3 });
}
for (const r of ROADS) FLAT.push({ x0: r.x0 - 2, z0: r.z0 - 2, x1: r.x1 + 2, z1: r.z1 + 2 });

/** 0 inside a slab or road (so foundations sit flush), 1 out in open sand. */
function flatMask(x, z) {
  let m = 1;
  for (const r of FLAT) {
    const inside = Math.min(
      smoothstep(r.x0 - 4, r.x0, x) * smoothstep(r.x1 + 4, r.x1, x),
      smoothstep(r.z0 - 4, r.z0, z) * smoothstep(r.z1 + 4, r.z1, z));
    m = Math.min(m, 1 - inside);
  }
  return m;
}

/**
 * Ground height. The arena floor stays within ±0.25 m so a single flat ground
 * collider is honest; past the boundary the berm climbs hard, which is what
 * visually justifies the invisible wall at the edge of the playable square.
 */
function terrainH(x, z) {
  const r = Math.max(Math.abs(x), Math.abs(z));
  const berm = smoothstep(69, 82, r) * 13 + smoothstep(80, 100, r) * 6;
  const dunes = fbm(x * 0.014 + 11.3, z * 0.014 - 4.7, 3) * 0.42 +
                fbm(x * 0.06, z * 0.06, 2) * 0.10;
  return dunes * flatMask(x, z) + berm;
}

/* ---------------------------------------------------------------- foliage */

/** Dry grass tuft, drawn as strokes into an RGBA buffer for alpha testing. */
function makeTuftTexture() {
  const S = 64, data = new Uint8Array(S * S * 4);
  const rnd = mulberry32(0x7a17);
  for (let b = 0; b < 15; b++) {
    const x0 = 8 + rnd() * 48, lean = (rnd() - 0.5) * 34, len = 0.55 + rnd() * 0.45;
    const shade = 0.55 + rnd() * 0.45;
    for (let t = 0; t <= 40; t++) {
      const f = t / 40;
      const x = x0 + lean * f * f;
      const y = (S - 2) - f * len * (S - 4);
      const wpx = 1.6 * (1 - f * 0.85);
      for (let dx = -2; dx <= 2; dx++) {
        const px = Math.round(x + dx), py = Math.round(y);
        if (px < 0 || px >= S || py < 0 || py >= S) continue;
        if (Math.abs(dx) > wpx) continue;
        const i = (py * S + px) * 4;
        // Tips bleach out; the base keeps the olive the shrub grew from.
        const dry = 0.35 + f * 0.5;
        data[i] = 255 * shade * (0.42 + dry * 0.34);
        data[i + 1] = 255 * shade * (0.38 + dry * 0.28);
        data[i + 2] = 255 * shade * (0.22 + dry * 0.16);
        data[i + 3] = 255;
      }
    }
  }
  const tex = new THREE.DataTexture(data, S, S, THREE.RGBAFormat);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

/* ------------------------------------------------------------------ level */

export class Level {
  constructor(scene, textures, settings, sky) {
    this.scene = scene;
    this.textures = textures;
    this.settings = settings;
    this.sky = sky;                       // lighting is Sky's to own, not ours

    this.playerSpawn = new THREE.Vector3(0, 0.05, 58);
    this.enemySpawns = [];
    this.coverPoints = [];
    this.colliders = [];
    this.raycastTargets = [];
    this.bounds = {
      min: new THREE.Vector3(-HALF, 0, -HALF),
      max: new THREE.Vector3(HALF, 26, HALF),
    };

    this.meshes = [];
    this._mats = new Map();
    this._batches = new Map();
    // Emplacements are authored while buildings go up but instanced in one pass.
    this._crates = []; this._smallCrates = []; this._bags = [];
    this._rng = mulberry32(0x5eed17);
    this._low = settings.name === 'LOW';
    this.stats = { drawCalls: 0, triangles: 0, colliders: 0 };
  }

  /* ------------------------------------------------------------ plumbing */

  _material(name, extra) {
    const key = name + (extra ? '#' + JSON.stringify(extra) : '');
    let m = this._mats.get(key);
    if (!m) {
      m = makeStandardMaterial(this.textures, name, { repeat: 1, ...extra });
      this._mats.set(key, m);
    }
    return m;
  }

  _bucket(mat) {
    let b = this._batches.get(mat);
    if (!b) { b = []; this._batches.set(mat, b); }
    return b;
  }

  _geo(mat, g) { this._bucket(mat).push(g); return g; }

  /** Geometry only — used for trim and anything the player cannot touch. */
  _box(mat, x, y, z, w, h, d, rotY = 0) {
    const g = boxGeo(w, h, d, TILE[mat]);
    if (rotY) g.rotateY(rotY);
    g.translate(x, y, z);
    return this._geo(mat, g);
  }

  _collider(x, y, z, w, h, d, rotY = 0) {
    let ex = w / 2, ez = d / 2;
    if (rotY) {
      const c = Math.abs(Math.cos(rotY)), s = Math.abs(Math.sin(rotY));
      ex = (w * c + d * s) / 2; ez = (w * s + d * c) / 2;
    }
    this.colliders.push({
      min: new THREE.Vector3(x - ex, y - h / 2, z - ez),
      max: new THREE.Vector3(x + ex, y + h / 2, z + ez),
    });
  }

  /** Geometry plus its collision box. */
  _solid(mat, x, y, z, w, h, d, rotY = 0) {
    this._box(mat, x, y, z, w, h, d, rotY);
    this._collider(x, y, z, w, h, d, rotY);
  }

  _cover(x, z, y = 0.05) { this.coverPoints.push(new THREE.Vector3(x, y, z)); }

  /* --------------------------------------------------------------- walls */

  /**
   * One wall panel with punched openings.
   *
   * Geometry is emitted per solid rectangle so windows read as real holes with
   * a sill and a lintel. Colliders are emitted from a much coarser pass: only
   * openings flagged `walk` (doors, vehicle bays) split the wall, because a
   * window with a 1 m sill is impassable to a capsule anyway. That single
   * decision keeps a nine-building outpost under a hundred boxes.
   *
   * @param {object} o axis:'x'|'z', cx, cz, len, height, thick, y0,
   *                   openings:[{at,w,y0,y1,walk,glass}], trim, foundation
   */
  _wall(mat, o) {
    const { axis, cx, cz, len, height, thick } = o;
    const y0 = o.y0 ?? 0, top = y0 + height;
    // Timber reveals on brick, steel on sheet metal, cast surrounds on concrete.
    const trim = o.trim ?? (mat === 'brick' ? 'wood' : mat === 'corrugated' ? 'metal' : 'concrete');
    // Only walls that meet grade get a buried foot; repeating it on an upper
    // storey would hang a half-metre downstand into the room below.
    const foot = y0 <= 0.001 ? (o.foundation ?? 0.5) : 0.06;
    const ops = (o.openings || []).slice().sort((a, b) => a.at - b.at);

    const put = (u0, u1, v0, v1) => {
      const w = u1 - u0, h = v1 - v0;
      if (w < 0.02 || h < 0.02) return;
      // Sink anything that lands on grade so a dune crest never shows daylight.
      const grounded = v0 <= y0 + 1e-4;
      const hh = grounded ? h + foot : h;
      const vy = grounded ? (v0 - foot + hh / 2) : (v0 + h / 2);
      const uu = (u0 + u1) / 2;
      if (axis === 'x') this._box(mat, cx + uu, vy, cz, w, hh, thick);
      else this._box(mat, cx, vy, cz + uu, thick, hh, w);
    };

    let cursor = -len / 2;
    for (const op of ops) {
      const a = op.at - op.w / 2, b = op.at + op.w / 2;
      put(cursor, a, y0, top);
      if (op.y0 > y0) put(a, b, y0, op.y0);
      if (op.y1 < top) put(a, b, op.y1, top);
      cursor = b;
      this._opening(mat, trim, o, op);
    }
    put(cursor, len / 2, y0, top);

    /* Coarse collision pass. */
    const walk = ops.filter(op => op.walk);
    const seg = (u0, u1, v0, v1) => {
      const w = u1 - u0, h = v1 - v0;
      if (w < 0.05 || h < 0.05) return;
      const uu = (u0 + u1) / 2, vy = v0 + h / 2;
      if (axis === 'x') this._collider(cx + uu, vy, cz, w, h, thick);
      else this._collider(cx, vy, cz + uu, thick, h, w);
    };
    let c = -len / 2;
    for (const op of walk) {
      seg(c, op.at - op.w / 2, y0, top);
      // The band above a door still has to stop anyone standing on the floor above.
      if (op.y1 < top - 0.4) seg(op.at - op.w / 2, op.at + op.w / 2, op.y1, top);
      c = op.at + op.w / 2;
    }
    seg(c, len / 2, y0, top);
  }

  /** Reveal, lintel and sill trim around one opening — pure silhouette work. */
  _opening(mat, trim, o, op) {
    const { axis, cx, cz, thick } = o;
    const t = thick + 0.16, oh = op.y1 - op.y0, yc = (op.y0 + op.y1) / 2;
    const at = op.at;
    const place = (u, y, w, h, d) => {
      if (axis === 'x') this._box(trim, cx + u, y, cz, w, h, d);
      else this._box(trim, cx, y, cz + u, d, h, w);
    };
    place(at - op.w / 2 - 0.07, yc, 0.16, oh, t);
    place(at + op.w / 2 + 0.07, yc, 0.16, oh, t);
    place(at, op.y1 + 0.11, op.w + 0.32, 0.22, t + 0.06);
    if (op.y0 > 0.3) place(at, op.y0 - 0.07, op.w + 0.36, 0.14, t + 0.12);

    if (op.glass) {
      const g = new THREE.PlaneGeometry(op.w - 0.05, oh - 0.05);
      if (axis === 'x') g.translate(cx + at, yc, cz);
      else { g.rotateY(Math.PI / 2); g.translate(cx, yc, cz + at); }
      this._geo('glass', g);
    }
  }

  /* --------------------------------------------------------------- shells */

  /**
   * A rectangular building: four walls per storey, a slab per storey, roof and
   * parapet. Windows are laid out on a fixed pitch and identical on every floor
   * so the facade reads as one building rather than a pile of random holes.
   */
  _shell(cfg) {
    const {
      x, z, w, d, mat, storyH, floors,
      thick = 0.34, doors = {}, roof = true, parapet = 0.7, parapetSkip = '',
      slabMat = 'concrete', roofMat = 'concrete', hole = null, interiorSlabs = true,
      windowPitch = 4.6, glass = false, topFloorWindows = true,
    } = cfg;
    const H = storyH * floors;

    const sides = [
      { k: 'n', axis: 'x', cx: x, cz: z - d / 2 + thick / 2, len: w },
      { k: 's', axis: 'x', cx: x, cz: z + d / 2 - thick / 2, len: w },
      { k: 'w', axis: 'z', cx: x - w / 2 + thick / 2, cz: z, len: d - thick * 2 },
      { k: 'e', axis: 'z', cx: x + w / 2 - thick / 2, cz: z, len: d - thick * 2 },
    ];

    for (const s of sides) {
      for (let f = 0; f < floors; f++) {
        const base = f * storyH;
        const openings = [];
        if (f < floors - 1 || topFloorWindows) {
          const n = Math.max(1, Math.min(3, Math.floor(s.len / windowPitch)));
          for (let i = 0; i < n; i++) {
            const at = -s.len / 2 + s.len * (i + 0.5) / n;
            openings.push({
              at, w: 1.5, y0: base + 1.0, y1: base + 2.35,
              glass: glass && (f > 0 || i !== 1),
            });
          }
        }
        for (const dr of (doors[s.k] || [])) {
          if (dr.floor !== undefined ? dr.floor !== f : f !== 0) continue;
          const a = dr.at - dr.w / 2 - 0.7, b = dr.at + dr.w / 2 + 0.7;
          for (let i = openings.length - 1; i >= 0; i--) {
            if (openings[i].at > a && openings[i].at < b) openings.splice(i, 1);
          }
          openings.push({
            at: dr.at, w: dr.w, y0: base + (dr.y0 ?? 0),
            y1: base + (dr.h ?? 2.4), walk: true,
          });
        }
        this._wall(mat, {
          axis: s.axis, cx: s.cx, cz: s.cz, len: s.len,
          height: storyH, thick, y0: base, openings,
        });
      }
    }

    /* Slabs. The ground pad sits 4 cm proud so it never z-fights the sand. */
    // Ground pad carries no collider: the world floor already sits at y=0 and
    // the 4 cm the pad stands proud is not worth 9 more boxes to test per step.
    this._box(slabMat, x, -0.21, z, w, 0.5, d);
    if (interiorSlabs) {
      for (let f = 1; f < floors; f++) this._slab(slabMat, x, f * storyH, z, w, d, 0.3, hole);
    }
    if (roof) {
      this._slab(roofMat, x, H, z, w + 0.24, d + 0.24, 0.3, hole);
      if (parapet > 0) this._parapet(mat, x, z, w, d, H, parapet, thick * 0.72, parapetSkip);
    }
    return H;
  }

  /**
   * Floor slab, optionally with a stairwell void. Four strips around the hole
   * beat one slab plus a hole in the collider, which AABBs cannot express.
   */
  _slab(mat, cx, yTop, cz, w, d, t, hole) {
    const y = yTop - t / 2;
    if (!hole) { this._solid(mat, cx, y, cz, w, t, d); return; }
    const x0 = cx - w / 2, x1 = cx + w / 2, z0 = cz - d / 2, z1 = cz + d / 2;
    const h = hole;
    const strip = (a0, b0, a1, b1) => {
      const sw = a1 - a0, sd = b1 - b0;
      if (sw > 0.05 && sd > 0.05) this._solid(mat, (a0 + a1) / 2, y, (b0 + b1) / 2, sw, t, sd);
    };
    strip(x0, z0, h.x0, z1);
    strip(h.x1, z0, x1, z1);
    strip(h.x0, z0, h.x1, h.z0);
    strip(h.x0, h.z1, h.x1, z1);
  }

  /** Roof parapet — falls off the roof edge become a choice, not an accident. */
  _parapet(mat, x, z, w, d, y, h, t, skip = '') {
    const yc = y + h / 2;
    // A skipped side is where a catwalk or stair lands; a continuous parapet
    // there would turn the arrival into an un-jumpable step.
    if (!skip.includes('n')) this._solid(mat, x, yc, z - d / 2 + t / 2, w, h, t);
    if (!skip.includes('s')) this._solid(mat, x, yc, z + d / 2 - t / 2, w, h, t);
    if (!skip.includes('w')) this._solid(mat, x - w / 2 + t / 2, yc, z, t, h, d - t * 2);
    if (!skip.includes('e')) this._solid(mat, x + w / 2 - t / 2, yc, z, t, h, d - t * 2);
    // A parapet is chest-high cover, so each face earns an AI slot.
    this._cover(x, z - d / 2 + 1.2, y + 0.05);
    this._cover(x, z + d / 2 - 1.2, y + 0.05);
    this._cover(x - w / 2 + 1.2, z, y + 0.05);
    this._cover(x + w / 2 - 1.2, z, y + 0.05);
  }

  /* -------------------------------------------------------------- stairs */

  /**
   * A straight flight. Each tread is one collider, which is the only way an
   * AABB world climbs; the rise is held at 36 cm so any controller with a
   * conventional step-up clears it without a jump.
   */
  _stairs(mat, opt) {
    const { x, z, y0, y1, dir, width = 1.6, rail = true } = opt;
    const n = Math.max(1, Math.ceil((y1 - y0) / STEP_RISE));
    const rise = (y1 - y0) / n;
    const dx = dir === '+x' ? 1 : dir === '-x' ? -1 : 0;
    const dz = dir === '+z' ? 1 : dir === '-z' ? -1 : 0;
    for (let i = 0; i < n; i++) {
      const top = y0 + rise * (i + 1);
      const h = top - (y0 - 0.4);
      const cx = x + dx * STEP_RUN * (i + 0.5);
      const cz = z + dz * STEP_RUN * (i + 0.5);
      const w = dx ? STEP_RUN : width, dd = dx ? width : STEP_RUN;
      this._solid(mat, cx, top - h / 2, cz, w, h, dd);
    }
    const runLen = n * STEP_RUN;
    if (rail) this._stairRail(x, z, dx, dz, y0, y1, runLen, width);
    return { x: x + dx * runLen, z: z + dz * runLen, len: runLen };
  }

  _stairRail(x, z, dx, dz, y0, y1, runLen, width) {
    for (const side of [-1, 1]) {
      const ox = dz ? side * width / 2 : 0, oz = dx ? side * width / 2 : 0;
      const midX = x + dx * runLen / 2 + ox, midZ = z + dz * runLen / 2 + oz;
      const ang = Math.atan2(y1 - y0, runLen);
      const g = boxGeo(Math.hypot(runLen, y1 - y0), 0.09, 0.09, TILE.metal);
      g.rotateZ(dx ? dx * ang : 0);
      if (dz) { g.rotateZ(ang); g.rotateY(dz > 0 ? -Math.PI / 2 : Math.PI / 2); }
      g.translate(midX, (y0 + y1) / 2 + 1.0, midZ);
      this._geo('metal', g);
      for (let p = 0; p <= 3; p++) {
        const f = p / 3;
        const px = x + dx * runLen * f + ox, pz = z + dz * runLen * f + oz;
        this._box('metal', px, y0 + (y1 - y0) * f + 0.5, pz, 0.07, 1.0, 0.07);
      }
    }
  }

  /** Landing platform between flights. */
  _landing(mat, x, y, z, w, d) {
    this._solid(mat, x, y - 0.14, z, w, 0.28, d);
  }

  /* ----------------------------------------------------------- instancing */

  _instanced(matName, geo, list, opt = {}) {
    if (!list.length) return null;
    const mesh = new THREE.InstancedMesh(geo, opt.material || this._material(matName), list.length);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion();
    const e = new THREE.Euler(), p = new THREE.Vector3(), s = new THREE.Vector3();
    for (let i = 0; i < list.length; i++) {
      const it = list[i];
      e.set(it.rx || 0, it.ry || 0, it.rz || 0);
      q.setFromEuler(e);
      p.set(it.x, it.y, it.z);
      s.set(it.sx ?? it.s ?? 1, it.sy ?? it.s ?? 1, it.sz ?? it.s ?? 1);
      mesh.setMatrixAt(i, m.compose(p, q, s));
      if (opt.color) mesh.setColorAt(i, opt.color(i, this._rng));
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.castShadow = !!opt.cast && !(this._low && opt.smallProp);
    mesh.receiveShadow = opt.receive !== false;
    mesh.userData.surface = opt.surface || SURFACE[matName];
    mesh.computeBoundingSphere();
    this.scene.add(mesh);
    this.meshes.push(mesh);
    if (opt.raycast !== false) this.raycastTargets.push(mesh);
    this.stats.drawCalls++;
    this.stats.triangles += (geo.index ? geo.index.count : geo.attributes.position.count) / 3 * list.length;
    return mesh;
  }

  /* ---------------------------------------------------------------- build */

  /**
   * Chunked so the loading bar keeps painting; every await yields a frame to
   * Safari, which otherwise shows a frozen bar for the whole construction.
   */
  async build() {
    const yield_ = () => new Promise(r => setTimeout(r, 0));

    this._buildTerrain();
    this._buildRoads();
    await yield_();

    this._buildPerimeter();
    await yield_();

    this._buildWarehouse();
    this._buildApartment();
    await yield_();

    this._buildAdmin();
    this._buildGarage();
    this._buildShops();
    await yield_();

    this._buildGuard();
    this._buildDepot();
    this._buildRuin();
    this._buildShed();
    await yield_();

    this._buildWaterTower();
    this._buildContainerYard();
    this._buildScaffold();
    this._buildTruck();
    this._buildCables();
    await yield_();

    this._buildProps();
    await yield_();

    this._buildFoliage();
    this._finalize();
    await yield_();
    return this;
  }

  /* ------------------------------------------------------------- terrain */

  _buildTerrain() {
    const SIZE = 200, SEG = 112;
    const g = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG).rotateX(-Math.PI / 2);
    const pos = g.attributes.position, uv = g.attributes.uv;
    const col = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), z = pos.getZ(i);
      pos.setY(i, terrainH(x, z));
      uv.setXY(i, x / TILE.sand, z / TILE.sand);
      // Large-scale tint break-up stops 200 m of one texture reading as tiling.
      const n = fbm(x * 0.05, z * 0.05, 3) * 0.5 + 0.5;
      const gravel = smoothstep(0.56, 0.82, fbm(x * 0.11 + 31, z * 0.11 - 17, 3) * 0.5 + 0.5);
      const t = 0.84 + n * 0.30 - gravel * 0.24;
      col[i * 3] = t * 1.03; col[i * 3 + 1] = t * 0.99; col[i * 3 + 2] = t * 0.92;
    }
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.computeVertexNormals();

    const mesh = new THREE.Mesh(g, this._material('sand', { vertexColors: true }));
    mesh.receiveShadow = true;          // ground never casts: it is the floor
    mesh.castShadow = false;
    mesh.userData.surface = 'sand';
    mesh.matrixAutoUpdate = false;
    this.scene.add(mesh);
    this.meshes.push(mesh);
    this.raycastTargets.push(mesh);
    this.stats.drawCalls++;
    this.stats.triangles += SEG * SEG * 2;

    // One slab under the whole arena; the ±0.25 m of dune is inside step-up.
    this.colliders.push({
      min: new THREE.Vector3(-HALF - 6, -3, -HALF - 6),
      max: new THREE.Vector3(HALF + 6, 0, HALF + 6),
    });
  }

  _buildRoads() {
    for (const r of ROADS) {
      const w = r.x1 - r.x0, d = r.z1 - r.z0;
      const g = planeGeo(w, d, TILE.asphalt,
        Math.max(1, Math.round(w / 4)), Math.max(1, Math.round(d / 4)));
      g.translate((r.x0 + r.x1) / 2, 0, (r.z0 + r.z1) / 2);
      const pos = g.attributes.position, uv = g.attributes.uv;
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i), z = pos.getZ(i);
        pos.setY(i, terrainH(x, z) + r.y);
        uv.setXY(i, x / TILE.asphalt, z / TILE.asphalt);   // world-locked seams
      }
      g.computeVertexNormals();
      this._geo('asphalt', g);
    }
  }

  /* ----------------------------------------------------------- perimeter */

  _buildPerimeter() {
    // Four sealed sides. The berm behind them climbs 13 m in 13 m, so the
    // invisible half of this reads as an un-climbable bank rather than a bug.
    const E = HALF + 0.5, T = 3;
    for (const s of [[0, -E, 2 * E + T * 2, T], [0, E, 2 * E + T * 2, T],
                     [-E, 0, T, 2 * E], [E, 0, T, 2 * E]]) {
      this.colliders.push({
        min: new THREE.Vector3(s[0] - s[2] / 2, 0, s[1] - s[3] / 2),
        max: new THREE.Vector3(s[0] + s[2] / 2, 16, s[1] + s[3] / 2),
      });
    }

    /* T-wall runs. Each panel is a slab plus a splayed foot; the run gets one
       collider rather than one per panel. Gaps are deliberate — they frame the
       berm and give the silhouette a rhythm instead of a fence. */
    const runs = [
      { a: [-68, -68], b: [-68, 6], },
      { a: [-68, 22], b: [-68, 68] },
      { a: [-56, -68], b: [12, -68] },
      { a: [26, -68], b: [68, -68] },
      { a: [68, -34], b: [68, 6] },
      { a: [68, 22], b: [68, 60] },
      { a: [-40, 68], b: [2, 68] },
      { a: [22, 68], b: [68, 68] },
    ];
    const panel = [];
    for (const r of runs) {
      const dx = r.b[0] - r.a[0], dz = r.b[1] - r.a[1];
      const len = Math.hypot(dx, dz), rot = dz !== 0 ? Math.PI / 2 : 0;
      const n = Math.floor(len / 1.55);
      for (let i = 0; i < n; i++) {
        const f = (i + 0.5) / n;
        panel.push({
          x: r.a[0] + dx * f, y: 0, z: r.a[1] + dz * f, ry: rot,
          sy: 0.92 + this._rng() * 0.16,
        });
      }
      const cx = (r.a[0] + r.b[0]) / 2, cz = (r.a[1] + r.b[1]) / 2;
      this._collider(cx, 1.8, cz, dz !== 0 ? 0.6 : len, 3.6, dz !== 0 ? len : 0.6);
      // Behind a T-wall is the safest lane on the map; the AI should know.
      for (let i = 0; i < Math.max(2, Math.floor(len / 14)); i++) {
        const f = (i + 0.5) / Math.max(2, Math.floor(len / 14));
        const px = r.a[0] + dx * f, pz = r.a[1] + dz * f;
        this._cover(px - Math.sign(px) * 1.6, pz - Math.sign(pz) * 1.6);
      }
    }

    const wall = boxGeo(1.5, 3.2, 0.32, TILE.concrete).translate(0, 1.6, 0);
    const foot = boxGeo(1.5, 0.28, 1.1, TILE.concrete).translate(0, 0.14, 0);
    this._instanced('concrete', mergeGeometries([wall, foot]), panel,
      { cast: true, surface: 'concrete' });

    /* Rock debris along the toe of the berm, tying wall to terrain. */
    const rocks = [];
    for (let i = 0; i < 140; i++) {
      const a = this._rng() * Math.PI * 2;
      const r = 66 + this._rng() * 8;
      const x = Math.cos(a) * r * (1 + this._rng() * 0.2);
      const z = Math.sin(a) * r * (1 + this._rng() * 0.2);
      if (Math.abs(x) > 96 || Math.abs(z) > 96) continue;
      rocks.push({
        x, y: terrainH(x, z) - 0.1, z, s: 0.5 + this._rng() * 1.5,
        ry: this._rng() * 6.28, rx: this._rng() * 0.6,
      });
    }
    this._instanced('concrete', this._rockGeo(), rocks,
      { cast: !this._low, smallProp: true, surface: 'concrete' });
  }

  _rockGeo() {
    const g = new THREE.IcosahedronGeometry(0.6, 0);
    const pos = g.attributes.position;
    const rnd = mulberry32(0x1234);
    for (let i = 0; i < pos.count; i++) {
      const f = 0.65 + rnd() * 0.7;
      pos.setXYZ(i, pos.getX(i) * f, pos.getY(i) * f * 0.7, pos.getZ(i) * f);
    }
    g.computeVertexNormals();
    return scaleUV(g, 1.2 / TILE.concrete, 1.2 / TILE.concrete);
  }

  /* ----------------------------------------------------------- buildings */

  /** Warehouse — the western anchor: open floor, L mezzanine, external stair. */
  _buildWarehouse() {
    const B = BUILDINGS.warehouse, H = 7.2;
    this._shell({
      x: B.x, z: B.z, w: B.w, d: B.d, mat: 'concrete', storyH: 3.6, floors: 2,
      interiorSlabs: false, parapet: 0.75, parapetSkip: 'e', windowPitch: 5.2,
      doors: {
        n: [{ at: -2, w: 4.2, h: 4.4 }],
        e: [{ at: 4, w: 1.7, h: 2.4 }, { at: -0.5, w: 1.5, h: 2.4, floor: 1 }],
        s: [{ at: 6, w: 1.7, h: 2.4 }],
      },
    });

    /* Mezzanine, an L hugging the north and east walls. Its open edges are the
       whole point: everything on the floor below is exposed from up here. */
    this._solid('concrete', B.x, 3.45, -40.0, 23.2, 0.3, 4.0);
    this._solid('concrete', -30.2, 3.45, -32.2, 3.6, 0.3, 11.6);
    for (const r of [[-51.5, -38.0, -28.6, -38.0], [-28.4, -38.0, -28.4, -26.4]]) {
      this._railing(r[0], r[1], r[2], r[3], 3.6);
    }
    this._cover(-44, -39.4, 3.6); this._cover(-36, -39.4, 3.6); this._cover(-30.2, -34, 3.6);

    /* Switchback stair on the east flank, then a bridge onto the roof. */
    this._stairs('concrete', { x: -26.8, z: -41.5, y0: 0, y1: 3.6, dir: '+z' });
    this._landing('concrete', -26.0, 3.6, -34.5, 3.4, 2.0);
    this._stairs('concrete', { x: -24.9, z: -33.5, y0: 3.6, y1: 7.2, dir: '-z' });
    this._solid('metal', -25.9, 7.06, -39.9, 4.6, 0.28, 1.7);
    this._railing(-28.2, -39.05, -23.6, -39.05, 7.2);
    this._railing(-28.2, -40.75, -23.6, -40.75, 7.2);

    // Roof furniture: two AC plants and a header tank read at 200 m.
    this._acUnit(-46, H, -38);
    this._acUnit(-43.5, H, -38);
    this._roofTank(-33, H, -30, 1.5, 2.2);
    this._pipe(-52.1, 0, -30, H);
    this._pipe(-27.9, 0, -38, H);
    this._cover(-46, -30, 0.05); this._cover(-38, -30, 0.05);
    this.enemySpawns.push(new THREE.Vector3(-46, 0.05, -33), new THREE.Vector3(-33, 3.65, -30));
  }

  /** Apartment block — three storeys, real internal switchback, balconies. */
  _buildApartment() {
    const B = BUILDINGS.apartment, H = 9.0;
    const hole = { x0: -45.2, z0: 22.2, x1: -41.5, z1: 27.6 };
    this._shell({
      x: B.x, z: B.z, w: B.w, d: B.d, mat: 'brick', storyH: 3.0, floors: 3,
      hole, parapet: 0.8, windowPitch: 4.4, thick: 0.32,
      doors: { s: [{ at: -3, w: 1.8, h: 2.4 }], n: [{ at: 4, w: 1.6, h: 2.4 }] },
    });

    this._stairs('concrete', { x: -44.3, z: 22.2, y0: 0, y1: 3.0, dir: '+z', width: 1.5, rail: false });
    this._stairs('concrete', { x: -42.4, z: 27.6, y0: 3.0, y1: 6.0, dir: '-z', width: 1.5, rail: false });
    this._stairs('concrete', { x: -44.3, z: 22.2, y0: 6.0, y1: 9.0, dir: '+z', width: 1.5, rail: false });

    /* South balconies — the cheapest possible way to break a flat brick face,
       and they double as firing positions over the plaza. */
    for (let f = 1; f <= 2; f++) {
      const y = f * 3.0;
      this._solid('concrete', -38, y - 0.1, 35.6, 9.0, 0.24, 1.9);
      this._solid('concrete', -38, y + 0.45, 36.5, 9.0, 0.9, 0.16);
      this._box('concrete', -42.4, y + 0.45, 35.6, 0.16, 0.9, 1.9);
      this._box('concrete', -33.6, y + 0.45, 35.6, 0.16, 0.9, 1.9);
      this._cover(-38, 35.4, y + 0.02);
      this.enemySpawns.push(new THREE.Vector3(-40, y + 0.05, 26));
    }
    this._acUnit(-42, H, 24);
    this._roofTank(-34, H, 31, 1.3, 1.8);
    this._pipe(-46.2, 0, 31, H);
    this._cover(-47.5, 24, 0.05); this._cover(-29, 30, 0.05);
    this.enemySpawns.push(new THREE.Vector3(-38, 0.05, 24));
  }

  /** Admin tower — tallest built volume, glazed, fire-escape to the roof. */
  _buildAdmin() {
    const B = BUILDINGS.admin, H = 9.45;
    this._shell({
      x: B.x, z: B.z, w: B.w, d: B.d, mat: 'concrete', storyH: 3.15, floors: 3,
      hole: { x0: 37.5, z0: 30.0, x1: 41.0, z1: 34.0 }, glass: true,
      parapet: 0.85, parapetSkip: 'w', windowPitch: 3.6,
      doors: {
        s: [{ at: 0, w: 1.8, h: 2.4 }],
        w: [{ at: 1.4, w: 1.5, h: 2.3, floor: 1 }, { at: -5.0, w: 1.5, h: 2.3, floor: 2 }],
      },
    });

    this._stairs('concrete', { x: 28.2, z: 29.0, y0: 0, y1: 3.15, dir: '+z', width: 1.5 });
    this._landing('concrete', 27.3, 3.15, 35.4, 3.6, 2.0);
    this._stairs('concrete', { x: 26.4, z: 34.4, y0: 3.15, y1: 6.3, dir: '-z', width: 1.5 });
    this._landing('concrete', 27.3, 6.3, 28.0, 3.6, 2.6);
    this._stairs('concrete', { x: 28.2, z: 29.0, y0: 6.3, y1: 9.45, dir: '+z', width: 1.5 });

    // Mast and dish: the second landmark, and it tells you where north is.
    this._box('metal', 36, H + 3.2, 34, 0.22, 6.4, 0.22);
    for (let i = 0; i < 3; i++) {
      const a = i * 2.094;
      this._cable(36, H + 6.0, 34, 36 + Math.cos(a) * 4.4, H + 0.2, 34 + Math.sin(a) * 4.4, 0.25);
    }
    this._box('metal', 33.4, H + 1.4, 34, 0.14, 2.8, 0.14);
    this._geo('metal', cylGeo(1.0, 1.0, 0.18, 12, TILE.metal).rotateX(1.2).translate(33.4, H + 2.3, 33.4));
    this._acUnit(40, H, 30.5);
    this._pipe(42.2, 0, 39.8, H);
    this._cover(29.5, 40, 0.05); this._cover(43, 28, 0.05);
    this.enemySpawns.push(new THREE.Vector3(36, 0.05, 34), new THREE.Vector3(36, 3.2, 32),
                          new THREE.Vector3(45, 0.05, 43));
  }

  /** Vehicle garage — open bays west, roof reached from the container stack. */
  _buildGarage() {
    const B = BUILDINGS.garage, H = 5.4;
    this._shell({
      x: B.x, z: B.z, w: B.w, d: B.d, mat: 'concrete', storyH: H, floors: 1,
      roofMat: 'corrugated', parapet: 0.55, parapetSkip: 's', windowPitch: 5.0,
      doors: {
        w: [{ at: -3, w: 4.0, h: 4.3 }, { at: 3, w: 4.0, h: 4.3 }],
        e: [{ at: 0, w: 1.7, h: 2.4 }],
      },
    });
    // Corrugated sheeting laid over the slab, with a proper eave overhang.
    this._box('corrugated', B.x, H + 0.4, B.z, B.w + 1.0, 0.14, B.d + 1.0);
    for (let i = 0; i < 5; i++) {
      this._box('metal', B.x - 7 + i * 3.5, H - 0.35, B.z, 0.2, 0.5, B.d - 0.6);
    }
    this._awning(21.0, 4.6, B.z, 2.4, B.d - 1.5, 'w');
    this._acUnit(34, H + 0.5, -34);
    this._cover(20, -33, 0.05); this._cover(20, -27, 0.05); this._cover(40, -30, 0.05);
    this.enemySpawns.push(new THREE.Vector3(30, 0.05, -30), new THREE.Vector3(38, 0.05, -34));
  }

  /** Shophouse terrace — a long, tight north-south wall on the west lane. */
  _buildShops() {
    const B = BUILDINGS.shops, H = 6.8;
    this._shell({
      x: B.x, z: B.z, w: B.w, d: B.d, mat: 'brick', storyH: 3.4, floors: 2,
      parapet: 0.85, windowPitch: 4.0, thick: 0.32, interiorSlabs: true,
      hole: null,
      doors: {
        e: [{ at: -8, w: 1.9, h: 2.5 }, { at: 0, w: 1.9, h: 2.5 }, { at: 8, w: 1.9, h: 2.5 }],
      },
    });
    // Arcade awning: the shade line under it is where players stop being seen.
    this._awning(-52.9, 3.1, B.z, 2.6, B.d - 2, 'e');
    for (let i = -2; i <= 2; i++) this._box('wood', -50.6, 1.55, B.z + i * 5.4, 0.16, 3.1, 0.16);
    this._pipe(-53.2, 0, -16.2, H);
    this._pipe(-53.2, 0, 8.2, H);
    for (const z of [-14, -6, 2, 8]) this._cover(-51.5, z, 0.05);
    this.enemySpawns.push(new THREE.Vector3(-58, 0.05, -10), new THREE.Vector3(-58, 0.05, 4),
                          new THREE.Vector3(-64, 0.05, -20));
  }

  /** Guard post — small, glazed, roof taken by parkour rather than stairs. */
  _buildGuard() {
    const B = BUILDINGS.guard, H = 3.4;
    this._shell({
      x: B.x, z: B.z, w: B.w, d: B.d, mat: 'concrete', storyH: H, floors: 1,
      parapet: 0.9, glass: true, windowPitch: 3.4,
      doors: { w: [{ at: 0, w: 1.6, h: 2.3 }] },
    });
    this._sandbagRing(B.x, H + 0.05, B.z, 2.6);
    this._cover(B.x - 5.5, B.z, 0.05);
    this.enemySpawns.push(new THREE.Vector3(54, 0.05, 30), new THREE.Vector3(60, 0.05, 22));
  }

  /** Supply depot on the north edge, the player's first piece of hard cover. */
  _buildDepot() {
    const B = BUILDINGS.depot, H = 4.0;
    this._shell({
      x: B.x, z: B.z, w: B.w, d: B.d, mat: 'concrete', storyH: H, floors: 1,
      roofMat: 'corrugated', parapet: 0, windowPitch: 4.2,
      doors: { s: [{ at: 0, w: 3.0, h: 3.0 }], e: [{ at: 0, w: 1.6, h: 2.3 }] },
    });
    this._box('corrugated', B.x, H + 0.45, B.z, B.w + 1.2, 0.14, B.d + 1.2);
    this._awning(B.x, 3.3, B.z + 4.5, 2.2, B.w - 2, 's');
    this._cover(-27, 48, 0.05); this._cover(-13, 48, 0.05);
    this.enemySpawns.push(new THREE.Vector3(-20, 0.05, 48));
  }

  /** Bombed-out shell: jagged brick stubs, no roof, rubble spilling out. */
  _buildRuin() {
    const B = BUILDINGS.ruin, t = 0.36;
    const x0 = B.x - B.w / 2, x1 = B.x + B.w / 2;
    const z0 = B.z - B.d / 2, z1 = B.z + B.d / 2;
    const seg = [
      { axis: 'x', cx: B.x - 4, cz: z0, len: 8, h: 4.2 },
      { axis: 'x', cx: B.x + 5.5, cz: z0, len: 5, h: 1.4 },
      { axis: 'z', cx: x0, cz: B.z - 2, len: 7, h: 3.6 },
      { axis: 'z', cx: x0, cz: B.z + 4, len: 3, h: 1.2 },
      { axis: 'x', cx: B.x - 5, cz: z1, len: 6, h: 2.4 },
      { axis: 'z', cx: x1, cz: B.z, len: 10, h: 3.0 },
      { axis: 'x', cx: B.x + 2, cz: B.z, len: 9, h: 1.9 },   // collapsed partition
    ];
    for (const s of seg) {
      const w = s.axis === 'x' ? s.len : t, d = s.axis === 'x' ? t : s.len;
      this._solid('brick', s.cx, s.h / 2 - 0.3, s.cz, w, s.h + 0.6, d);
      // Ragged top course so the break never reads as a clean saw cut.
      const n = Math.max(2, Math.round(s.len / 1.1));
      for (let i = 0; i < n; i++) {
        const f = (i + 0.5) / n, hh = 0.2 + this._rng() * 0.55;
        const px = s.axis === 'x' ? s.cx - s.len / 2 + s.len * f : s.cx;
        const pz = s.axis === 'x' ? s.cz : s.cz - s.len / 2 + s.len * f;
        this._box('brick', px, s.h + hh / 2 - 0.05, pz,
          s.axis === 'x' ? s.len / n * 0.9 : t, hh, s.axis === 'x' ? t : s.len / n * 0.9);
      }
      this._cover(s.cx + (s.axis === 'x' ? 0 : 1.5), s.cz + (s.axis === 'x' ? 1.5 : 0));
    }
    this._solid('concrete', B.x, -0.1, B.z, B.w, 0.4, B.d);
    this.enemySpawns.push(new THREE.Vector3(B.x, 0.05, B.z), new THREE.Vector3(B.x - 10, 0.05, B.z + 4));
  }

  /** Corrugated lean-to in the south-east dead ground. */
  _buildShed() {
    const B = BUILDINGS.shed, H = 3.0;
    this._shell({
      x: B.x, z: B.z, w: B.w, d: B.d, mat: 'corrugated', storyH: H, floors: 1,
      thick: 0.16, parapet: 0, roofMat: 'corrugated', windowPitch: 4.5,
      doors: { n: [{ at: 0, w: 2.2, h: 2.4 }] },
    });
    // Single-pitch roof: a thin slab tilted just enough to read as drainage.
    const g = boxGeo(B.w + 0.9, 0.12, B.d + 0.9, TILE.corrugated);
    g.rotateX(0.14).translate(B.x, H + 0.5, B.z);
    this._geo('corrugated', g);
    for (let i = 0; i < 3; i++) this._box('wood', B.x - 3 + i * 3, H + 0.2, B.z, 0.14, 0.5, B.d);
    this._cover(48, -48, 0.05); this._cover(54, -43, 0.05);
    this.enemySpawns.push(new THREE.Vector3(54, 0.05, -48), new THREE.Vector3(62, 0.05, -56));
  }

  /* ------------------------------------------------------------- details */

  _railing(x0, z0, x1, z1, y) {
    const len = Math.hypot(x1 - x0, z1 - z0);
    const ang = Math.atan2(-(z1 - z0), x1 - x0);
    const cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
    for (const h of [1.05, 0.55]) {
      this._box('metal', cx, y + h, cz, len, 0.07, 0.07, ang);
    }
    const n = Math.max(2, Math.round(len / 2.0));
    for (let i = 0; i <= n; i++) {
      const f = i / n;
      this._box('metal', x0 + (x1 - x0) * f, y + 0.55, z0 + (z1 - z0) * f, 0.08, 1.1, 0.08);
    }
  }

  /** Rooftop condenser: box, fan cowl and fins. Reads as clutter from below. */
  _acUnit(x, y, z) {
    this._box('metal', x, y + 0.45, z, 1.15, 0.9, 0.95);
    this._geo('metal', cylGeo(0.34, 0.34, 0.12, 10, TILE.metal).translate(x, y + 0.96, z));
    for (let i = 0; i < 4; i++) this._box('metal', x - 0.45 + i * 0.3, y + 0.45, z + 0.5, 0.08, 0.8, 0.06);
    this._box('metal', x, y + 0.05, z, 1.3, 0.1, 1.1);
  }

  _roofTank(x, y, z, r, h) {
    this._geo('metal', cylGeo(r, r, h, 14, TILE.metal).translate(x, y + h / 2 + 0.5, z));
    this._collider(x, y + h / 2 + 0.5, z, r * 2, h, r * 2);
    for (let i = 0; i < 4; i++) {
      const a = i * 1.5708;
      this._box('metal', x + Math.cos(a) * r * 0.8, y + 0.25, z + Math.sin(a) * r * 0.8, 0.1, 0.5, 0.1);
    }
  }

  /** Downpipe with brackets — vertical relief on an otherwise flat corner. */
  _pipe(x, y0, z, y1) {
    this._geo('metal', cylGeo(0.075, 0.075, y1 - y0, 6, TILE.metal).translate(x, (y0 + y1) / 2, z));
    for (let h = 1.2; h < y1; h += 2.4) this._box('metal', x, h, z, 0.22, 0.08, 0.22);
  }

  /** Sloped awning on one face, plus its tie rods. */
  _awning(x, y, z, out, len, side) {
    const dx = side === 'e' ? 1 : side === 'w' ? -1 : 0;
    const dz = side === 's' ? 1 : side === 'n' ? -1 : 0;
    const g = boxGeo(dx ? out : len, 0.1, dx ? len : out, TILE.corrugated);
    g.rotateZ(dx ? -dx * 0.22 : 0);
    if (dz) g.rotateX(dz * 0.22);
    g.translate(x + dx * out / 2, y, z + dz * out / 2);
    this._geo('corrugated', g);
    for (const s of [-1, 1]) {
      const px = x + dx * out * 0.9 + (dx ? 0 : s * len * 0.45);
      const pz = z + dz * out * 0.9 + (dz ? 0 : s * len * 0.45);
      this._box('metal', px, y - 0.9, pz, 0.09, 1.8, 0.09);
    }
  }

  /** Catenary cable between two anchor points. */
  _cable(x0, y0, z0, x1, y1, z1, sag = 1.2) {
    const pts = [];
    for (let i = 0; i <= 8; i++) {
      const t = i / 8;
      pts.push(new THREE.Vector3(
        x0 + (x1 - x0) * t,
        y0 + (y1 - y0) * t - Math.sin(t * Math.PI) * sag,
        z0 + (z1 - z0) * t));
    }
    const curve = new THREE.CatmullRomCurve3(pts);
    const g = new THREE.TubeGeometry(curve, 10, 0.035, 3, false);
    scaleUV(g, curve.getLength() / TILE.gunmetal, 0.2 / TILE.gunmetal);
    this._geo('gunmetal', g);
  }

  /* ---------------------------------------------------------- structures */

  /**
   * Water tower, near the centre and 18 m tall. Deliberately not climbable —
   * its whole job is to be visible from every corner so the player always knows
   * which way they are facing. Nothing else on the map is that tall.
   */
  _buildWaterTower() {
    const cx = 2, cz = -2, legR = 3.6, legH = 12;
    for (let i = 0; i < 4; i++) {
      const a = i * 1.5708 + 0.785;
      const lx = cx + Math.cos(a) * legR, lz = cz + Math.sin(a) * legR;
      // Legs splay outward: a straight-sided tower looks like a table.
      const g = cylGeo(0.16, 0.24, legH, 8, TILE.metal);
      g.translate(0, legH / 2, 0);
      g.applyMatrix4(new THREE.Matrix4().makeTranslation(lx, 0, lz));
      this._geo('metal', g);
      this._collider(lx, legH / 2, lz, 0.55, legH, 0.55);
    }
    for (let ring = 0; ring < 3; ring++) {
      const y = 3.0 + ring * 3.2;
      for (let i = 0; i < 4; i++) {
        const a0 = i * 1.5708 + 0.785, a1 = a0 + 1.5708;
        const p0 = [cx + Math.cos(a0) * legR, cz + Math.sin(a0) * legR];
        const p1 = [cx + Math.cos(a1) * legR, cz + Math.sin(a1) * legR];
        const len = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
        const ang = Math.atan2(-(p1[1] - p0[1]), p1[0] - p0[0]);
        this._box('metal', (p0[0] + p1[0]) / 2, y, (p0[1] + p1[1]) / 2, len, 0.12, 0.12, ang);
        for (const s of [-1, 1]) {
          const g = boxGeo(Math.hypot(len, 3.2), 0.09, 0.09, TILE.metal);
          g.rotateZ(s * Math.atan2(3.2, len)); g.rotateY(ang);
          g.translate((p0[0] + p1[0]) / 2, y + 1.6, (p0[1] + p1[1]) / 2);
          this._geo('metal', g);
        }
      }
    }
    this._geo('metal', cylGeo(4.3, 4.3, 0.14, 16, TILE.metal).translate(cx, legH - 0.1, cz));
    for (let i = 0; i < 12; i++) {
      const a = i * 0.5236;
      this._box('metal', cx + Math.cos(a) * 4.1, legH + 0.5, cz + Math.sin(a) * 4.1, 0.07, 1.1, 0.07);
    }
    const tank = cylGeo(3.3, 3.3, 4.8, 16, TILE.corrugated);
    this._geo('corrugated', tank.translate(cx, legH + 2.4, cz));
    this._collider(cx, legH + 2.4, cz, 6.6, 4.8, 6.6);
    this._geo('corrugated', cylGeo(0.2, 3.7, 1.6, 16, TILE.corrugated).translate(cx, legH + 5.6, cz));
    for (let i = 0; i < 4; i++) {
      const a = i * 1.5708;
      this._box('metal', cx + Math.cos(a) * 3.35, legH + 2.4, cz + Math.sin(a) * 3.35, 0.1, 4.8, 0.1);
    }
    // Access ladder, purely to explain how the tank is ever serviced.
    for (const s of [-0.28, 0.28]) this._box('metal', cx + s, legH / 2, cz - legR - 0.3, 0.06, legH, 0.06);
    for (let r = 0; r < 26; r++) this._box('metal', cx, r * 0.45 + 0.4, cz - legR - 0.3, 0.62, 0.05, 0.05);
  }

  /**
   * Container yard: the only route from ground to the garage roof, gated by a
   * crate parkour stack so the climb costs time and exposes you while you take it.
   */
  _buildContainerYard() {
    const boxes = [
      { x: 24, z: -16, y: 0, ry: 0 }, { x: 24, z: -13.2, y: 0, ry: 0 },
      { x: 31.5, z: -16, y: 0, ry: 0 }, { x: 24, z: -14.6, y: 2.59, ry: 0 },
      { x: 31.5, z: -16, y: 2.59, ry: 0 },
      { x: 44, z: 14, y: 0, ry: Math.PI / 2 },      // road blocker, east lane
      { x: -20, z: 16, y: 0, ry: Math.PI / 2 },     // road blocker, west lane
      { x: -34, z: -20, y: 0, ry: 0 },
      { x: 8, z: -40, y: 0, ry: Math.PI / 2 },
      { x: 60, z: 4, y: 0, ry: 0 }, { x: 60, z: 4, y: 2.59, ry: 0 },
      { x: -50, z: 42, y: 0, ry: Math.PI / 2 },
    ];
    const palette = [0x8c5a3c, 0x4a6a78, 0x6f7a5a, 0x8a8578, 0x7a4a44, 0x5c6663];
    this._instanced('corrugated', this._containerGeo(),
      boxes.map(b => ({ x: b.x, y: b.y + 1.295, z: b.z, ry: b.ry })), {
      cast: true, surface: 'metal',
      color: (i) => new THREE.Color(palette[i % palette.length]),
    });
    for (const b of boxes) {
      this._collider(b.x, b.y + 1.295, b.z, 6.06, 2.59, 2.44, b.ry);
      this._cover(b.x + (b.ry ? 2.2 : 0), b.z + (b.ry ? 0 : 2.2));
      this._cover(b.x - (b.ry ? 2.2 : 0), b.z - (b.ry ? 0 : 2.2));
    }

    /* Ground -> A/B tops -> D top -> garage roof. */
    this._stairs('metal', { x: 21.6, z: -12.8, y0: 2.59, y1: 5.18, dir: '+x', width: 1.4 });
    this._landing('metal', 26.9, 5.18, -13.3, 2.2, 1.8);
    this._solid('metal', 24, 5.16, -19.8, 1.9, 0.24, 9.0);      // catwalk to the roof
    this._railing(23.05, -24.2, 23.05, -15.4, 5.28);
    this._railing(24.95, -24.2, 24.95, -15.4, 5.28);
    for (const z of [-18, -22]) this._box('metal', 24, 2.6, z, 0.16, 5.2, 0.16);
    this._cover(24, -17, 5.3); this._cover(31.5, -14, 5.25);
    this.enemySpawns.push(new THREE.Vector3(34, 0.05, -8), new THREE.Vector3(24, 5.3, -18));
  }

  _containerGeo() {
    const parts = [boxGeo(6.06, 2.59, 2.44, TILE.corrugated, { swapSides: true })];
    // Corner castings and the door end — the details that read as "container".
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
      parts.push(boxGeo(0.3, 0.22, 2.5, TILE.metal).translate(sx * 2.88, sy * 1.19, 0));
    }
    for (const sz of [-1, 1]) {
      parts.push(boxGeo(6.1, 0.2, 0.2, TILE.metal).translate(0, 1.2, sz * 1.21));
      parts.push(boxGeo(6.1, 0.2, 0.2, TILE.metal).translate(0, -1.2, sz * 1.21));
    }
    for (const ox of [-0.55, 0.55]) {
      parts.push(boxGeo(0.09, 2.3, 0.09, TILE.metal).translate(3.04, 0, ox));
    }
    parts.push(boxGeo(0.06, 2.4, 1.1, TILE.metal).translate(3.05, 0, 0.6));
    parts.push(boxGeo(0.06, 2.4, 1.1, TILE.metal).translate(3.05, 0, -0.6));
    return mergeGeometries(parts);
  }

  /** Timber scaffold — mid-height perch that owns the northern approach. */
  _buildScaffold() {
    const x = 14, z = 40, w = 6.4, d = 3.6, y = 3.3;
    this._solid('wood', x, y - 0.12, z, w, 0.24, d);
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      this._box('wood', x + sx * (w / 2 - 0.2), y / 2, z + sz * (d / 2 - 0.2), 0.22, y, 0.22);
      const g = boxGeo(Math.hypot(w, y), 0.12, 0.12, TILE.wood);
      g.rotateZ(sz * 0.5); g.translate(x, y / 2, z + sz * (d / 2 - 0.2));
      this._geo('wood', g);
    }
    this._railing(x - w / 2, z - d / 2, x + w / 2, z - d / 2, y);
    this._railing(x - w / 2, z + d / 2, x + w / 2, z + d / 2, y);
    this._stairs('wood', { x: x + w / 2 + 0.4, z: z - 1.8, y0: 0, y1: y, dir: '+z', width: 1.4 });
    this._box('corrugated', x, y + 2.3, z, w + 0.8, 0.1, d + 0.8);
    for (const sx of [-1, 1]) this._box('wood', x + sx * (w / 2 - 0.3), y + 1.2, z, 0.14, 2.4, 0.14);
    this._cover(x, z - 1.4, y + 0.02);
  }

  /** Burnt-out truck: breaks the long east-west lane and gives low cover. */
  _buildTruck() {
    const x = -14, z = 15, r = 0.32;
    const put = (mat, ox, oy, oz, w, h, d, rot = 0) => {
      const g = boxGeo(w, h, d, TILE[mat]);
      g.rotateY(rot); g.rotateY(r);
      g.translate(x + ox * Math.cos(r) - oz * Math.sin(r), oy, z + ox * Math.sin(r) + oz * Math.cos(r));
      this._geo(mat, g);
    };
    put('metal', -1.9, 1.35, 0, 2.4, 1.9, 2.3);            // cab shell
    put('metal', -1.9, 2.4, 0, 2.2, 0.2, 2.4);
    put('metal', 0.6, 0.75, 0, 3.4, 0.5, 2.2);             // chassis rails
    for (let i = 0; i < 6; i++) put('wood', 0.6, 1.15, -1.0 + i * 0.4, 3.2, 0.16, 0.28);
    put('metal', 2.3, 1.5, 0, 0.16, 1.4, 2.2);
    for (const sz of [-1, 1]) put('metal', 0.6, 1.6, sz * 1.05, 3.3, 1.2, 0.14);
    for (const ax of [-1.9, 1.4]) {
      const g = cylGeo(0.22, 0.22, 2.1, 8, TILE.gunmetal).rotateZ(Math.PI / 2).rotateY(r);
      g.translate(x + ax * Math.cos(r), 0.5, z + ax * Math.sin(r));
      this._geo('gunmetal', g);
    }
    this._collider(x - 1.9 * Math.cos(r), 1.3, z - 1.9 * Math.sin(r), 2.9, 2.6, 2.9);
    this._collider(x + 0.6 * Math.cos(r), 1.1, z + 0.6 * Math.sin(r), 4.0, 2.2, 2.8);
    this._cover(x, z + 2.4); this._cover(x, z - 2.4); this._cover(x - 4, z);
  }

  /**
   * Overhead lines. Sag is what sells scale: straight cables read as wires in a
   * CAD file, drooping ones read as a place people actually wired up badly.
   */
  _buildCables() {
    const poles = [[-24, 8.5], [4, 8.5], [26, 8.5], [50, 8.5], [12, -24], [12, -50], [12, 34]];
    for (const [px, pz] of poles) {
      this._geo('wood', cylGeo(0.16, 0.22, 8.2, 6, TILE.wood).translate(px, 4.1, pz));
      this._collider(px, 4.1, pz, 0.45, 8.2, 0.45);
      this._box('wood', px, 7.5, pz, 1.8, 0.14, 0.14);
      this._box('wood', px, 6.9, pz, 1.3, 0.12, 0.12);
    }
    for (let i = 0; i < poles.length - 1; i++) {
      const a = poles[i], b = poles[i + 1];
      if (Math.hypot(a[0] - b[0], a[1] - b[1]) > 40) continue;
      for (const o of [-0.7, 0, 0.7]) this._cable(a[0] + o, 7.5, a[1], b[0] + o, 7.5, b[1], 1.1);
    }
    this._cable(-28, 7.0, -30, -24, 7.5, 8, 1.4);          // warehouse to the line
    this._cable(-46, 8.6, 26, -24, 8.6, 46, 2.6);          // apartment to depot
    this._cable(29.6, 9.0, 34, 50, 7.5, 8.5, 3.0);         // admin to the line
    this._cable(-53, 6.4, -4, -24, 7.5, 8, 2.2);           // shops to the line
    this._cable(5.3, 11.6, -2, 26, 7.5, 8.5, 2.4);         // tower to the line
  }

  /* ---------------------------------------------------------------- props */

  /** One sandbag row; bags are instanced, the run gets a single collider. */
  _sandbagWall(x0, z0, x1, z1, rows = 3) {
    const len = Math.hypot(x1 - x0, z1 - z0);
    const ang = Math.atan2(-(z1 - z0), x1 - x0);
    const n = Math.max(1, Math.round(len / 0.56));
    for (let r = 0; r < rows; r++) {
      const off = (r & 1) ? 0.5 : 0;                 // stretcher bond, like real bags
      for (let i = 0; i < n - (r & 1); i++) {
        const f = (i + 0.5 + off) / n;
        this._bags.push({
          x: x0 + (x1 - x0) * f, y: 0.17 + r * 0.3, z: z0 + (z1 - z0) * f,
          ry: ang + (this._rng() - 0.5) * 0.25,
          s: 0.92 + this._rng() * 0.18,
        });
      }
    }
    const h = rows * 0.3;
    const cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
    const nx = -Math.sin(ang), nz = -Math.cos(ang);
    this._collider(cx, h / 2, cz, Math.abs(x1 - x0) + 0.5, h, Math.abs(z1 - z0) + 0.5);
    this._cover(cx + nx * 1.1, cz + nz * 1.1);
    this._cover(cx - nx * 1.1, cz - nz * 1.1);
  }

  _sandbagRing(x, y, z, r) {
    for (let row = 0; row < 2; row++) {
      for (const s of [-1, 1]) {
        for (let i = 0; i < 9; i++) {
          const f = (i + 0.5) / 9;
          this._bags.push({ x: x - r + 2 * r * f, y: y + 0.17 + row * 0.3, z: z + s * r, ry: 0, s: 1 });
          this._bags.push({ x: x + s * r, y: y + 0.17 + row * 0.3, z: z - r + 2 * r * f, ry: 1.5708, s: 1 });
        }
      }
    }
    this._cover(x, z, y + 0.02);
  }

  /** Jumpable crate ladder — how the roofs and stacks are actually reached. */
  _stack(x, z, n) {
    for (let i = 0; i < n; i++) {
      this._crates.push({ x, y: 0.45 + i * 0.9, z, ry: (this._rng() - 0.5) * 0.2 });
    }
    this._collider(x, n * 0.45, z, 1.05, n * 0.9, 1.05);   // one box for the column
    this._cover(x + 1.4, z);
  }

  _buildProps() {
    const barrels = [], jersey = [], hesco = [], pallets = [], rubble = [];
    const R = this._rng;

    /* Sandbag emplacements: the plaza around the tower, and firing points that
       cover each approach into it. */
    this._sandbagWall(-4, 4, 4, 4);
    this._sandbagWall(8, 2, 8, -6);
    this._sandbagWall(-5, -8, 1, -8);
    this._sandbagWall(-8, 3, -8, -4);
    this._sandbagWall(-2, 26, 6, 26);
    this._sandbagWall(38, -6, 38, 2);
    this._sandbagWall(-24, -12, -24, -5);
    this._sandbagWall(46, 44, 54, 44);
    this._sandbagWall(-16, 34, -8, 34);
    this._sandbagWall(18, 52, 18, 60);
    this._sandbagWall(-40, -8, -34, -8);
    this._sandbagWall(60, -20, 60, -12);

    /* Jersey barriers channel the two road lanes without sealing them. */
    const runs = [
      [-30, 9.2, -18, 9.2], [-6, 9.2, 6, 9.2], [20, 9.2, 32, 9.2], [40, 20.8, 52, 20.8],
      [-46, 20.8, -34, 20.8], [5.2, -30, 5.2, -18], [16.8, -12, 16.8, 0],
      [5.2, 30, 5.2, 42], [56, 12, 56, 22], [-60, 36, -50, 36],
    ];
    for (const [x0, z0, x1, z1] of runs) {
      const len = Math.hypot(x1 - x0, z1 - z0);
      const along = x1 === x0 ? 'z' : 'x';
      const n = Math.max(1, Math.round(len / 3.1));
      for (let i = 0; i < n; i++) {
        const f = (i + 0.5) / n;
        jersey.push({
          x: x0 + (x1 - x0) * f, y: 0, z: z0 + (z1 - z0) * f,
          ry: along === 'z' ? Math.PI / 2 : 0,
        });
      }
      this._collider((x0 + x1) / 2, 0.43, (z0 + z1) / 2,
        along === 'x' ? len : 0.62, 0.86, along === 'x' ? 0.62 : len);
      this._cover((x0 + x1) / 2 + (along === 'x' ? 0 : 1.2), (z0 + z1) / 2 + (along === 'x' ? 1.2 : 0));
      this._cover((x0 + x1) / 2 - (along === 'x' ? 0 : 1.2), (z0 + z1) / 2 - (along === 'x' ? 1.2 : 0));
    }

    /* HESCO bastion — chest-high, indestructible-looking, and the only cover
       that works out in the open ground north of the plaza. */
    const bast = [
      [-14, 4, 4], [10, 44, 4], [44, -6, 3], [-30, 40, 3], [30, 12, 3], [-46, 12, 3],
    ];
    for (const [bx, bz, n] of bast) {
      const horiz = (bx + bz) % 2 === 0;
      for (let i = 0; i < n; i++) {
        const ox = horiz ? (i - (n - 1) / 2) * 1.52 : 0;
        const oz = horiz ? 0 : (i - (n - 1) / 2) * 1.52;
        hesco.push({ x: bx + ox, y: 0.75, z: bz + oz });
      }
      this._collider(bx, 0.75, bz, horiz ? n * 1.52 : 1.5, 1.5, horiz ? 1.5 : n * 1.52);
      this._cover(bx + (horiz ? 0 : 1.4), bz + (horiz ? 1.4 : 0));
      this._cover(bx - (horiz ? 0 : 1.4), bz - (horiz ? 1.4 : 0));
    }

    /* Parkour stacks. Each is the entry to a piece of verticality. */
    this._stack(19.4, -16.2, 2);        // onto the container yard
    this._stack(49.6, 30, 3);           // onto the guard post roof
    this._stack(-24.6, 46.4, 2);        // onto the depot awning
    this._stack(41.8, -22.6, 2);

    /* Loose crates and pallets — clutter with a purpose: every one of these is
       a waist-high shooting rest somewhere an AI can actually use. */
    const clusters = [
      [-46, -22], [-33, -25], [26, -22], [36, -18], [-56, 14], [-52, -32],
      [8, -46], [-2, 30], [22, 34], [56, -30], [40, 48],
      [0, -20], [-18, -36], [62, -4], [30, 58], [-36, 6],
    ];
    for (const [cx, cz] of clusters) {
      const n = 1 + Math.floor(R() * 3);
      for (let i = 0; i < n; i++) {
        const x = cx + (R() - 0.5) * 3.4, z = cz + (R() - 0.5) * 3.4;
        const stacked = R() < 0.3;
        this._crates.push({ x, y: 0.45, z, ry: (R() - 0.5) * 1.2 });
        if (stacked) this._crates.push({ x: x + (R() - 0.5) * 0.2, y: 1.35, z, ry: (R() - 0.5) * 1.2 });
        this._collider(x, stacked ? 0.9 : 0.45, z, 1.15, stacked ? 1.8 : 0.9, 1.15);
      }
      for (let i = 0; i < 2; i++) {
        this._smallCrates.push({
          x: cx + (R() - 0.5) * 4.5, y: 0.31, z: cz + (R() - 0.5) * 4.5, ry: R() * 3.14,
        });
      }
      if (R() < 0.6) pallets.push({ x: cx + (R() - 0.5) * 5, y: 0.07, z: cz + (R() - 0.5) * 5, ry: R() * 3.14 });
      this._cover(cx + 1.6, cz + 1.6);
      this._cover(cx - 1.6, cz - 1.6);
    }

    /* Oil drums: singles read as junk, groups of four read as a fuel point. */
    const drums = [[-44, -20], [28, -8], [-54, 20], [14, -34], [46, 20], [-6, 44],
                   [34, 4], [-28, -44], [58, -38], [-64, -12], [20, 24], [-12, -14]];
    for (const [cx, cz] of drums) {
      const n = 2 + Math.floor(R() * 3);
      for (let i = 0; i < n; i++) {
        const a = R() * 6.28, r = R() * 1.3;
        const x = cx + Math.cos(a) * r, z = cz + Math.sin(a) * r;
        const tipped = R() < 0.15;
        barrels.push({
          x, y: tipped ? 0.29 : 0.44, z, ry: R() * 3.14,
          rz: tipped ? Math.PI / 2 : 0,
        });
      }
      this._collider(cx, 0.45, cz, 2.6, 0.9, 2.6);
      this._cover(cx + 2.0, cz);
    }

    /* Rubble against the ruin and along the blast walls. */
    for (let i = 0; i < 150; i++) {
      const near = i < 90;
      const x = near ? -8 + (R() - 0.5) * 26 : (R() - 0.5) * 130;
      const z = near ? -50 + (R() - 0.5) * 20 : (R() - 0.5) * 130;
      if (flatMask(x, z) < 0.25 && !near) continue;
      rubble.push({
        x, y: terrainH(x, z) + 0.05, z, ry: R() * 6.28, rx: R() * 0.8,
        s: 0.2 + R() * 0.45,
      });
    }

    for (const l of [this._crates, this._smallCrates, pallets, barrels, jersey, hesco, this._bags]) this._settle(l);

    this._instanced('wood', this._crateGeo(0.9), this._crates, { cast: true, smallProp: true });
    this._instanced('wood', this._crateGeo(0.62), this._smallCrates, { cast: !this._low, smallProp: true });
    this._instanced('wood', this._palletGeo(), pallets, { cast: false, smallProp: true });
    this._instanced('metalOlive', this._barrelGeo(), barrels, { cast: true, smallProp: true });
    this._instanced('concrete', this._jerseyGeo(), jersey, { cast: true });
    this._instanced('sand', this._hescoGeo(), hesco, { cast: true, surface: 'sand' });
    this._instanced('fabric', this._bagGeo(), this._bags, { cast: !this._low, smallProp: true, surface: 'sand' });
    this._instanced('concrete', this._rockGeo(), rubble, { cast: false, smallProp: true, raycast: false });
  }

  /**
   * Props rest on the collision floor at y=0, but the sand dips as much as
   * 25 cm below it. Dropping each instance into the dip trades a floating crate
   * — which the eye catches instantly — for a slightly buried one, which it does not.
   */
  _settle(list) {
    for (const p of list) p.y += Math.min(0, terrainH(p.x, p.z));
  }

  _crateGeo(s) {
    const parts = [boxGeo(s, s, s, TILE.wood)];
    const b = s * 0.09, h = s / 2 + 0.005;
    for (const sy of [-1, 1]) for (const sz of [-1, 1]) parts.push(boxGeo(s + 0.01, b, b, TILE.wood).translate(0, sy * h, sz * h));
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) parts.push(boxGeo(b, s + 0.01, b, TILE.wood).translate(sx * h, 0, sz * h));
    for (const sx of [-1, 1]) parts.push(boxGeo(b, b * 0.9, s, TILE.wood).translate(sx * h, 0, 0));
    return mergeGeometries(parts);
  }

  _palletGeo() {
    const parts = [];
    for (let i = 0; i < 5; i++) parts.push(boxGeo(1.2, 0.035, 0.12, TILE.wood).translate(0, 0.06, -0.4 + i * 0.2));
    for (const sz of [-0.4, 0, 0.4]) parts.push(boxGeo(1.2, 0.09, 0.14, TILE.wood).translate(0, 0, sz));
    return mergeGeometries(parts);
  }

  _barrelGeo() {
    const parts = [cylGeo(0.29, 0.29, 0.88, 10, TILE.metalOlive)];
    for (const y of [-0.24, 0.24]) parts.push(cylGeo(0.305, 0.305, 0.07, 10, TILE.metalOlive).translate(0, y, 0));
    parts.push(cylGeo(0.12, 0.12, 0.03, 6, TILE.metalOlive).translate(0.1, 0.45, 0.05));
    return mergeGeometries(parts);
  }

  _jerseyGeo() {
    return mergeGeometries([
      boxGeo(3.05, 0.24, 0.61, TILE.concrete).translate(0, 0.12, 0),
      boxGeo(3.05, 0.36, 0.42, TILE.concrete).translate(0, 0.42, 0),
      boxGeo(3.05, 0.28, 0.22, TILE.concrete).translate(0, 0.72, 0),
    ]);
  }

  _hescoGeo() {
    const parts = [boxGeo(1.5, 1.5, 1.5, TILE.sand)];
    for (const sy of [-1, 1]) for (const sz of [-1, 1]) parts.push(boxGeo(1.54, 0.06, 0.06, TILE.metal).translate(0, sy * 0.76, sz * 0.76));
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) parts.push(boxGeo(0.06, 1.54, 0.06, TILE.metal).translate(sx * 0.76, 0, sz * 0.76));
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) parts.push(boxGeo(0.06, 0.06, 1.54, TILE.metal).translate(sx * 0.76, sy * 0.76, 0));
    return mergeGeometries(parts);
  }

  _bagGeo() {
    const g = new THREE.SphereGeometry(0.5, 7, 4);
    const pos = g.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      pos.setXYZ(i, pos.getX(i) * 1.15, pos.getY(i) * 0.42, pos.getZ(i) * 0.72);
    }
    g.computeVertexNormals();
    return scaleUV(g, 1.6 / TILE.fabric, 0.5 / TILE.fabric);
  }

  /* -------------------------------------------------------------- foliage */

  _buildFoliage() {
    if (!this.settings.foliage) return;
    const count = this.settings.drawDistance > 300 ? 700 : 380;
    const list = [];
    for (let i = 0; i < count * 3 && list.length < count; i++) {
      const x = (this._rng() - 0.5) * 152, z = (this._rng() - 0.5) * 152;
      // Nothing grows on a road, a slab, or inside a building footprint.
      if (flatMask(x, z) < 0.85) continue;
      list.push({
        x, y: terrainH(x, z) - 0.05, z, ry: this._rng() * 3.14,
        s: 0.7 + this._rng() * 0.8,
      });
    }
    const quad = [];
    for (let i = 0; i < 3; i++) {
      const g = new THREE.PlaneGeometry(1.0, 0.85).translate(0, 0.42, 0).rotateY(i * 1.047);
      quad.push(g);
    }
    const geo = mergeGeometries(quad);
    // Flat upward normals: a cross-quad lit by its own facing goes black on the
    // shaded side, which no dry shrub in direct desert sun ever does.
    const nrm = geo.attributes.normal;
    for (let i = 0; i < nrm.count; i++) nrm.setXYZ(i, 0, 1, 0);

    this._tuft = makeTuftTexture();
    this._tuft.anisotropy = this.settings.maxAnisotropy || 4;
    const mat = new THREE.MeshStandardMaterial({
      map: this._tuft, alphaTest: 0.5, side: THREE.DoubleSide,
      roughness: 0.95, metalness: 0, color: 0xd9cba8,
    });
    this._instanced('sand', geo, list, {
      material: mat, cast: false, receive: false, raycast: false, surface: 'sand',
    });
  }

  /* ------------------------------------------------------------- finalize */

  _finalize() {
    for (const [name, geoms] of this._batches) {
      if (!geoms.length) continue;
      const geo = mergeGeometries(geoms);
      let mat;
      if (name === 'glass') {
        // No glazing in the texture library, and none is wanted: a thin tinted
        // pane with depthWrite off stays cheap and never fights the bloom pass.
        mat = new THREE.MeshStandardMaterial({
          color: 0x17242a, roughness: 0.08, metalness: 0.1,
          transparent: true, opacity: 0.34, depthWrite: false,
          side: THREE.DoubleSide,
        });
        this._mats.set('glass', mat);
      } else {
        mat = this._material(name);
      }
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = name !== 'glass';
      mesh.receiveShadow = name !== 'glass';
      mesh.userData.surface = SURFACE[name] || 'concrete';
      mesh.matrixAutoUpdate = false;
      mesh.updateMatrix();
      this.scene.add(mesh);
      this.meshes.push(mesh);
      this.raycastTargets.push(mesh);
      this.stats.drawCalls++;
      this.stats.triangles += geo.index.count / 3;
    }
    this._batches.clear();

    this.playerSpawn.set(0, terrainH(0, 58) + 0.05, 58);

    /* Spawns sit behind hard geometry relative to the player start, so a wave
       never materialises in the open in front of the camera. */
    const extra = [
      [-58, 30], [-64, -34], [-6, -62], [40, -52], [64, 40], [16, -56],
      [-34, -14], [56, -12], [-44, 44], [-56, 58], [-66, 6], [62, 50],
    ];
    for (const [x, z] of extra) this.enemySpawns.push(new THREE.Vector3(x, terrainH(x, z) + 0.05, z));
    for (const s of this.enemySpawns) {
      s.x = THREE.MathUtils.clamp(s.x, -HALF + 3, HALF - 3);
      s.z = THREE.MathUtils.clamp(s.z, -HALF + 3, HALF - 3);
    }

    for (const c of this.coverPoints) {
      if (c.y < 0.2) c.y = terrainH(c.x, c.z) + 0.05;
    }

    /* Placement is authored by hand, so a later layout edit can quietly bury a
       spawn inside a wall. Rejecting anything that intersects a collider is a
       lot cheaper than discovering it as an enemy stuck in the brickwork. */
    const clear = (p, pad) => {
      for (const c of this.colliders) {
        if (c.max.y < 0.06) continue;                 // the ground slab is not an obstacle
        if (p.x > c.min.x - pad && p.x < c.max.x + pad &&
            p.z > c.min.z - pad && p.z < c.max.z + pad &&
            p.y + 1.7 > c.min.y && p.y < c.max.y - 0.06) return false;
      }
      return Math.abs(p.x) < HALF - 1 && Math.abs(p.z) < HALF - 1;
    };
    this.enemySpawns = this.enemySpawns.filter(s => clear(s, 0.45));
    this.coverPoints = this.coverPoints.filter(c => clear(c, 0.05));

    // Backstop: the AI contract needs 30 cover slots even if a layout edit
    // removes props, so pad from open ground rather than fail late.
    for (let i = 0; this.coverPoints.length < 30; i++) {
      const a = i * 0.7, r = 20 + i;
      const p = new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r);
      p.y = terrainH(p.x, p.z) + 0.05;
      if (clear(p, 0.05)) this.coverPoints.push(p);
    }

    this.stats.colliders = this.colliders.length;
    this.stats.triangles = Math.round(this.stats.triangles);
  }

  /* -------------------------------------------------------------- queries */

  /** Surface family of a hit mesh, for impact particles, decals and sound. */
  materialAt(mesh) {
    if (!mesh) return 'concrete';
    return mesh.userData?.surface || mesh.parent?.userData?.surface || 'concrete';
  }

  heightAt(x, z) { return terrainH(x, z); }

  dispose() {
    for (const m of this.meshes) {
      this.scene.remove(m);
      m.geometry.dispose();
    }
    for (const mat of this._mats.values()) mat.dispose();
    this._tuft?.dispose();
    this.meshes.length = 0;
    this._mats.clear();
  }
}

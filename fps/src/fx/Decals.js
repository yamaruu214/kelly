// Decals.js — pooled surface decals (bullet holes, blood, scorch marks).
//
// One InstancedBufferGeometry of quads, one draw call, no per-frame CPU work:
// each decal uploads its hit point, surface normal, roll and birth time once,
// and the vertex shader rebuilds the oriented quad and the lifetime fade from
// there. Projected decals (clipping a mesh against the hit volume) look better
// on curved geometry but cost a geometry rebuild per shot, which is exactly the
// spike a phone cannot absorb mid-firefight.
//
// The decals multiply into the frame (dst * colour) instead of being alpha
// composited. That single blend choice is why they sit *in* the lighting: a
// bullet hole in shadow darkens what is already there rather than glowing as an
// unlit sprite, and it needs no lighting code of its own.

import * as THREE from 'three';
import { fbm, voronoi } from '../world/Textures.js';

const clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);
const smoothstep = (e0, e1, x) => { const t = clamp01((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t); };
const rnd = (a, b) => a + Math.random() * (b - a);

/* ------------------------------------------------------------------ atlas */

const TILE = 0.5;   // 2x2 cell grid
// Two bullet variants: a wall taking twenty hits with one texture reads as a
// rubber stamp no matter how much the rotation is randomised.
const CELLS = {
  bullet: [[0.0, 0.0], [0.5, 0.5]],
  blood:  [[0.5, 0.0]],
  scorch: [[0.0, 0.5]],
};

let _atlas = null;

function bulletCell(seed) {
  return (nx, ny, r, o) => {
    const v = voronoi(nx * 0.5 + 0.5, ny * 0.5 + 0.5, 9);
    const n = fbm(nx * 4 + seed, ny * 4 - seed * 1.7, 5);
    const holeR = 0.26 + 0.05 * n;
    const rimR = 0.44 + 0.10 * n - 0.06 * v.edge;

    const centre = smoothstep(holeR, holeR - 0.05, r);
    const rimBand = smoothstep(rimR, rimR - 0.17, r) * (1 - centre);

    // Radial cracks keyed off the Voronoi cell id so each variant fractures
    // differently rather than sharing one starburst.
    const ang = Math.atan2(ny, nx);
    const spoke = Math.pow(Math.abs(Math.sin(ang * (4.0 + seed) + 6.0 * v.id)), 14.0);
    const crack = spoke * smoothstep(0.95, 0.28, r) * (0.55 + 0.45 * (n * 0.5 + 0.5));

    // Baked bevel: a fixed top-left key light across the rim band reads as a
    // raised lip. A real tangent-space normal map would need a second sampler
    // and a lighting pass this decal deliberately does not have.
    const lip = smoothstep(holeR - 0.02, rimR - 0.02, r) * smoothstep(rimR + 0.10, rimR - 0.08, r);
    const key = -(nx * 0.62 + ny * 0.78);

    let b = 0.44 - 0.39 * centre - 0.16 * crack;
    b *= 1 + lip * key * 0.55;

    o[0] = b * 1.02; o[1] = b * 0.99; o[2] = b * 0.95;
    o[3] = clamp01(centre + rimBand * 0.8 + crack * 0.5);
  };
}

function buildAtlas(cell = 256) {
  const w = cell * 2, h = cell * 2;
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(w, h);
  const d = img.data;

  const o = [0, 0, 0, 0];
  const put = (col, row, fn) => {
    const ox = col * cell, oy = row * cell;
    for (let y = 0; y < cell; y++) {
      const ny = ((y + 0.5) / cell) * 2 - 1;
      for (let x = 0; x < cell; x++) {
        const nx = ((x + 0.5) / cell) * 2 - 1;
        fn(nx, ny, Math.sqrt(nx * nx + ny * ny), o);
        const i = ((oy + y) * w + (ox + x)) * 4;
        d[i]     = clamp01(o[0]) * 255;
        d[i + 1] = clamp01(o[1]) * 255;
        d[i + 2] = clamp01(o[2]) * 255;
        d[i + 3] = clamp01(o[3]) * 255;
      }
    }
  };

  put(0, 0, bulletCell(3));
  put(1, 1, bulletCell(7));

  // Blood: irregular Voronoi-bitten edge, runs below the pool, satellite drops.
  // Canvas +y is the decal's local down, so drips are drawn toward the bottom.
  put(1, 0, (nx, ny, r, o) => {
    const v = voronoi(nx * 0.5 + 0.5, ny * 0.5 + 0.5, 6);
    const n = fbm(nx * 2.6 + 17, ny * 2.6 - 5, 5);
    const rad = 0.48 + 0.17 * n - 0.10 * smoothstep(0.0, 0.22, v.edge);
    let a = smoothstep(rad, rad - 0.07, r);

    for (let k = 0; k < 4; k++) {
      const cx = -0.52 + k * 0.35 + 0.10 * Math.sin(k * 7.3);
      const len = 0.55 + 0.40 * Math.abs(Math.sin(k * 4.7 + 1.2));
      const wd = 0.040 + 0.018 * Math.sin(k * 2.1);
      if (ny <= 0.05 || ny > len) continue;
      const t = ny / len;
      const taper = Math.max(wd * (1 - t * 0.85), 1e-4);
      const dx = (nx - cx) / taper;
      a = Math.max(a, Math.exp(-dx * dx * 1.4) * smoothstep(1.0, 0.72, t));
    }
    for (let k = 0; k < 8; k++) {
      const ak = k * 2.3 + 0.4, dk = 0.62 + 0.28 * Math.sin(k * 5.9);
      const dx = nx - Math.cos(ak) * dk, dy = ny - Math.sin(ak) * dk;
      a = Math.max(a, Math.exp(-(dx * dx + dy * dy) * (240 + k * 90)) * 0.9);
    }

    // Thin blood is bright and orange; pooled blood is nearly black. Grading
    // the multiply colour by coverage is what stops it reading as red paint.
    const thin = 1 - smoothstep(rad - 0.28, rad, r);
    o[0] = 0.28 + 0.44 * (1 - thin);
    o[1] = 0.030 + 0.135 * (1 - thin);
    o[2] = 0.030 + 0.110 * (1 - thin);
    o[3] = clamp01(a);
  });

  put(0, 1, (nx, ny, r, o) => {
    const n1 = fbm(nx * 2.0 + 61, ny * 2.0 + 29, 5);
    const n2 = fbm(nx * 6.0 - 11, ny * 6.0 + 53, 4);
    const rr = r * (1 + 0.35 * n1);
    const core = smoothstep(0.72, 0.05, rr);
    const b = 0.09 + 0.44 * (1 - core) + 0.10 * (n2 * 0.5 + 0.5);
    o[0] = b * 1.06; o[1] = b * 0.97; o[2] = b * 0.88;
    o[3] = clamp01(smoothstep(1.0, 0.12, rr) * (0.6 + 0.4 * (n2 * 0.5 + 0.5)) * 0.95);
  });

  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(cv);
  tex.flipY = false;          // canvas row 0 stays at v=0, matching CELLS
  tex.colorSpace = THREE.NoColorSpace;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

/* ----------------------------------------------------------------- shaders */

const VERT = /* glsl */`
uniform float uTime;
uniform float uTile;

attribute vec3 aPos;
attribute vec3 aNrm;
attribute vec4 aParams;   // roll, size, birthTime, life
attribute vec4 aSprite;   // uOffset, vOffset, normalOffset, peakAlpha
attribute vec3 aTint;

varying vec2 vUv;
varying vec3 vTint;
varying float vAlpha;

void main(){
  float life = max(aParams.w, 1e-4);
  float t = (uTime - aParams.z) / life;
  if (t < 0.0 || t > 1.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }

  vec3 n = normalize(aNrm);
  // Reference vector swaps near the poles so the basis never degenerates on a
  // floor or ceiling hit; elsewhere world-up keeps local +Y upright, which is
  // what lets the blood texture's drips actually run downhill.
  vec3 ref = abs(n.y) > 0.94 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
  vec3 tx = normalize(cross(ref, n));
  vec3 ty = cross(n, tx);

  float c = cos(aParams.x), s = sin(aParams.x);
  vec3 ax = tx * c + ty * s;
  vec3 ay = ty * c - tx * s;

  // Brief scale-in so a decal appears rather than pops, and blood spreads.
  float grow = 0.72 + 0.28 * smoothstep(0.0, 0.05 / life, t);
  float size = aParams.y * grow;

  vec3 world = aPos + n * aSprite.z + ax * (position.x * size) + ay * (position.y * size);

  // Inset by half a percent of the cell: at coarse mip levels neighbouring
  // atlas cells would otherwise bleed a grey halo around every hole.
  vec2 quv = vec2(position.x + 0.5, 0.5 - position.y);
  vUv = aSprite.xy + uTile * (quv * 0.988 + 0.006);
  vTint = aTint;
  vAlpha = aSprite.w * (1.0 - smoothstep(0.78, 1.0, t));

  gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
}
`;

const FRAG = /* glsl */`
precision highp float;

uniform sampler2D uAtlas;
uniform float uAlphaTest;

varying vec2 vUv;
varying vec3 vTint;
varying float vAlpha;

void main(){
  vec4 tex = texture2D(uAtlas, vUv);
  float a = tex.a * vAlpha;
  if (a < uAlphaTest) discard;

  // Premultiplied against DST_COLOR / ONE_MINUS_SRC_ALPHA gives
  // dst * (1 - a + a*colour): a true multiply weighted by coverage, so fading
  // alpha to zero returns the surface exactly to its original shade.
  gl_FragColor = vec4(tex.rgb * vTint * a, a);
}
`;

/* Multiply tints per type, jittered per decal so no two hits grade identically.
   Values above 1 brighten, which is what the bullet hole's bevel lip rides on. */
const TINTS = {
  bullet: [1.00, 0.99, 0.97],
  blood:  [1.00, 0.94, 0.92],
  scorch: [1.00, 0.98, 0.95],
};
const LIVES = { bullet: 55, blood: 28, scorch: 60 };
const ALPHAS = { bullet: 1.0, blood: 0.92, scorch: 0.85 };

/* ------------------------------------------------------------------ system */

export class DecalSystem {
  constructor(scene, settings = {}) {
    this.scene = scene;
    this.settings = settings;
    this.time = 0;
    this.enabled = true;

    this.max = Math.max(8, settings.maxDecals | 0 || 96);
    this._cursor = 0;
    this._used = 0;
    this._dirtyMin = Infinity;
    this._dirtyMax = -Infinity;

    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0,
    ]), 3));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    geo.instanceCount = 0;

    const mk = (name, items) => {
      const a = new THREE.InstancedBufferAttribute(new Float32Array(this.max * items), items);
      a.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute(name, a);
      return a;
    };
    this.aPos = mk('aPos', 3);
    this.aNrm = mk('aNrm', 3);
    this.aParams = mk('aParams', 4);
    this.aSprite = mk('aSprite', 4);
    this.aTint = mk('aTint', 3);
    this._attrs = [this.aPos, this.aNrm, this.aParams, this.aSprite, this.aTint];

    for (let i = 0; i < this.max; i++) {
      this.aParams.array[i * 4 + 2] = -1e6;   // born long ago, so already dead
      this.aParams.array[i * 4 + 3] = 1;
    }

    _atlas = _atlas || buildAtlas();
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uTime: { value: 0 },
        uTile: { value: TILE },
        uAtlas: { value: _atlas },
        uAlphaTest: { value: 0.04 },
      },
      transparent: true,
      depthTest: true,
      // Depth write off plus a depth-slope bias: two decals overlapping on the
      // same wall are exactly coplanar, and either alone still z-fights.
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -4,
      polygonOffsetUnits: -8,
      side: THREE.FrontSide,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.DstColorFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.ZeroFactor,
      blendDstAlpha: THREE.OneFactor,
    });

    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 2;         // after opaque, before the particle pass
    this.mesh.matrixAutoUpdate = false;
    this.scene.add(this.mesh);
  }

  /**
   * `size` is the decal's width in metres. Rotation and scale are jittered per
   * decal so repeated hits on one wall never line up into a visible grid.
   */
  add(position, normal, type = 'bullet', size = 0.14) {
    if (!this.enabled) return;
    const cells = CELLS[type] || CELLS.bullet;
    const cell = cells[(Math.random() * cells.length) | 0];

    const i = this._cursor;
    this._cursor = (this._cursor + 1) % this.max;
    if (this._used < this.max) this._used++;

    const i3 = i * 3, i4 = i * 4;
    this.aPos.array[i3] = position.x;
    this.aPos.array[i3 + 1] = position.y;
    this.aPos.array[i3 + 2] = position.z;

    const nl = Math.hypot(normal.x, normal.y, normal.z) || 1;
    this.aNrm.array[i3] = normal.x / nl;
    this.aNrm.array[i3 + 1] = normal.y / nl;
    this.aNrm.array[i3 + 2] = normal.z / nl;

    // Blood keeps its up axis so the painted drips run downward; everything
    // else spins freely.
    const roll = type === 'blood' ? rnd(-0.3, 0.3) : Math.random() * 6.283185;
    this.aParams.array[i4] = roll;
    this.aParams.array[i4 + 1] = size * rnd(0.80, 1.30);
    this.aParams.array[i4 + 2] = this.time;
    this.aParams.array[i4 + 3] = LIVES[type] || 40;

    this.aSprite.array[i4] = cell[0];
    this.aSprite.array[i4 + 1] = cell[1];
    // Stagger the lift per decal so stacked decals also separate from each other.
    this.aSprite.array[i4 + 2] = 0.006 + (i % 7) * 0.0009;
    this.aSprite.array[i4 + 3] = ALPHAS[type] ?? 1;

    const tint = TINTS[type] || TINTS.bullet;
    const shade = rnd(0.86, 1.10);
    this.aTint.array[i3] = tint[0] * shade;
    this.aTint.array[i3 + 1] = tint[1] * shade;
    this.aTint.array[i3 + 2] = tint[2] * shade;

    if (i < this._dirtyMin) this._dirtyMin = i;
    if (i > this._dirtyMax) this._dirtyMax = i;
  }

  update(dt) {
    this.time += dt;
    this.material.uniforms.uTime.value = this.time;
    this.mesh.geometry.instanceCount = this._used;

    if (this._dirtyMax < this._dirtyMin) return;
    const start = this._dirtyMin, count = this._dirtyMax - start + 1;
    for (const a of this._attrs) {
      if (a.clearUpdateRanges && a.addUpdateRange) {
        a.clearUpdateRanges();
        a.addUpdateRange(start * a.itemSize, count * a.itemSize);
      }
      a.needsUpdate = true;
    }
    this._dirtyMin = Infinity; this._dirtyMax = -Infinity;
  }

  clear() {
    const p = this.aParams.array;
    for (let i = 0; i < this.max; i++) p[i * 4 + 2] = -1e6;
    this.aParams.needsUpdate = true;
    this._cursor = 0; this._used = 0;
    this._dirtyMin = Infinity; this._dirtyMax = -Infinity;
    this.mesh.geometry.instanceCount = 0;
  }

  dispose() {
    this.scene.remove(this.mesh);
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}

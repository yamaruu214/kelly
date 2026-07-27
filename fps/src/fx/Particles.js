// Particles.js — one GPU-integrated particle pool for every effect in the game.
//
// The whole system is a single InstancedBufferGeometry of quads drawn in one
// call. A particle's entire trajectory is a closed-form function of its age, so
// the CPU writes 26 floats once at spawn and never touches that particle again;
// the vertex shader evaluates position, size, spin, colour and fade every frame.
// Updating 2400 positions in JS per frame is what kills a phone here — this
// keeps the per-frame CPU cost of the FX layer at literally zero.
//
// Additive and alpha-blended particles share that one draw call by writing
// premultiplied colour with ONE / ONE_MINUS_SRC_ALPHA blending: the alpha
// channel alone chooses the mode (0 = additive, a = normal), so a spark burst
// and the dust cloud it sits inside cost one state change between them.

import * as THREE from 'three';
import { fbm, voronoi } from '../world/Textures.js';

const clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);
const smoothstep = (e0, e1, x) => { const t = clamp01((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t); };
const rnd = (a, b) => a + Math.random() * (b - a);

/* ------------------------------------------------------------------ atlas */

// 4x2 cells. UV offsets are the cell's origin; the shader adds uTile * corner.
const TILE_U = 0.25, TILE_V = 0.5;
const SPR = {
  PUFF:  [0.00, 0.0],   // soft round smoke/dust body
  SPARK: [0.25, 0.0],   // streak, brighter toward the leading edge
  SMOKE: [0.50, 0.0],   // fbm-billowed wisp
  CHIP:  [0.75, 0.0],   // angular debris fragment with baked shading
  BLOOD: [0.00, 0.5],   // splat with satellite droplets
  FLASH: [0.25, 0.5],   // six-spike muzzle/explosion star
  DUST:  [0.50, 0.5],   // grainy, softer and dirtier than PUFF
  GLOW:  [0.75, 0.5],   // tight core + wide halo, used for tracer glow
};

let _atlas = null;

function buildAtlas(cell = 128) {
  const cols = 4, rows = 2, w = cols * cell, h = rows * cell;
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(w, h);
  const d = img.data;

  // One shared output tuple for ~130k pixel evaluations — returning a fresh
  // array per pixel would hand the GC a hundred thousand objects at boot.
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

  put(0, 0, (nx, ny, r, o) => {
    const n = fbm(nx * 2.2 + 11, ny * 2.2 + 7, 4);
    o[0] = o[1] = o[2] = 1;
    o[3] = Math.exp(-r * r * 2.9) * (0.82 + 0.18 * n) * smoothstep(1.06, 0.70, r);
  });

  put(1, 0, (nx, ny, r, o) => {
    // Width grows toward the head so the streak reads as a trail, not a bar.
    const head = nx * 0.5 + 0.5;
    const wd = 0.055 + 0.10 * head;
    const core = Math.exp(-(ny * ny) / (wd * wd));
    const along = smoothstep(-1.0, -0.55, nx) * smoothstep(1.0, 0.80, nx);
    const hot = Math.exp(-(ny * ny) / (wd * wd * 0.22)) * along;
    o[0] = 1; o[1] = 0.55 + 0.45 * hot; o[2] = 0.22 + 0.78 * hot;
    o[3] = core * along;
  });

  put(2, 0, (nx, ny, r, o) => {
    const n1 = fbm(nx * 1.8 + 31, ny * 1.8 + 19, 5);
    const n2 = fbm(nx * 4.5 - 13, ny * 4.5 + 5, 4);
    // Perturbing the radius rather than the alpha keeps the silhouette ragged,
    // which is the difference between smoke and an out-of-focus circle.
    const rr = r * (1 + 0.42 * n1);
    o[0] = o[1] = o[2] = 0.80 + 0.20 * (n1 * 0.5 + 0.5);
    o[3] = smoothstep(1.0, 0.10, rr) * (0.55 + 0.45 * (n2 * 0.5 + 0.5)) * 0.95;
  });

  put(3, 0, (nx, ny, r, o) => {
    const ang = Math.atan2(ny, nx);
    const rad = 0.52 + 0.20 * Math.sin(ang * 2.0 + 1.1) + 0.10 * Math.sin(ang * 5.0);
    // Fixed top-left key baked into the chip so tumbling debris shows facets
    // even though the particle pass has no lighting of its own.
    const l = 0.45 + 0.55 * clamp01(0.5 - nx * 0.45 - ny * 0.55);
    o[0] = o[1] = o[2] = l;
    o[3] = smoothstep(rad, rad - 0.07, r);
  });

  put(0, 1, (nx, ny, r, o) => {
    const v = voronoi(nx * 0.5 + 0.5, ny * 0.5 + 0.5, 7);
    const rad = 0.56 + 0.16 * fbm(nx * 3 + 5, ny * 3 - 9, 4) - 0.10 * v.edge;
    let a = smoothstep(rad, rad - 0.09, r);
    for (let k = 0; k < 5; k++) {
      const ak = k * 1.7 + 0.6, dk = 0.72 + 0.22 * Math.sin(k * 3.1);
      const dx = nx - Math.cos(ak) * dk, dy = ny - Math.sin(ak) * dk;
      a = Math.max(a, Math.exp(-(dx * dx + dy * dy) * (120 + k * 40)));
    }
    const l = 0.55 + 0.45 * smoothstep(0.0, 0.55, r);
    o[0] = l; o[1] = l * 0.86; o[2] = l * 0.84; o[3] = a;
  });

  put(1, 1, (nx, ny, r, o) => {
    const ang = Math.atan2(ny, nx);
    const spikes = Math.pow(Math.abs(Math.cos(ang * 3.0)), 6.0);
    const a = Math.exp(-r * r * 26) + 0.55 * Math.exp(-r * r * 4.5)
            + 0.85 * spikes * Math.exp(-r * r * 2.2);
    o[0] = 1; o[1] = 0.92; o[2] = 0.78;
    o[3] = clamp01(a) * smoothstep(1.05, 0.85, r);
  });

  put(2, 1, (nx, ny, r, o) => {
    const n = fbm(nx * 3.1 - 21, ny * 3.1 + 43, 5);
    const g = fbm(nx * 9.0 + 7, ny * 9.0 - 3, 3);
    o[0] = o[1] = o[2] = 0.95 + 0.05 * n;
    o[3] = smoothstep(1.0, 0.05, r * (1 + 0.30 * n)) * (0.45 + 0.55 * (g * 0.5 + 0.5)) * 0.8;
  });

  put(3, 1, (nx, ny, r, o) => {
    o[0] = o[1] = o[2] = 1;
    o[3] = clamp01(Math.exp(-r * r * 22) + 0.30 * Math.exp(-r * r * 2.6)) * smoothstep(1.05, 0.90, r);
  });

  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(cv);
  // flipY off keeps canvas row 0 at v=0, so the SPR table reads like the grid.
  tex.flipY = false;
  tex.colorSpace = THREE.NoColorSpace;   // masks, not colour — no sRGB decode
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
uniform vec2  uTile;

attribute vec3 aOrigin;
attribute vec3 aVel;
attribute vec4 aParams;   // spawnTime, life, size0, size1
attribute vec4 aPhys;     // gravity, drag, spinRate, roll0
attribute vec4 aSprite;   // uOffset, vOffset, fadeIn, stretch(m)
attribute vec4 aColorA;   // rgb at birth, blendMode (0 additive / 1 alpha)
attribute vec4 aColorB;   // rgb at death, peak alpha

varying vec2 vUv;
varying vec3 vColor;
varying vec3 vMeta;       // alpha, blendMode, view depth

void main(){
  float life = max(aParams.y, 1e-4);
  float age  = uTime - aParams.x;
  float t    = age / life;
  // Dead and unborn particles are pushed behind the near plane; clipping them
  // costs one vertex each and no fragments at all.
  if (t < 0.0 || t > 1.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }

  // Linear drag has a closed form: v(t) = v0*exp(-k t), so displacement is
  // v0*(1 - exp(-k t))/k. Gravity is integrated undamped, which is both cheaper
  // and closer to how heavy debris actually behaves in air.
  float k = aPhys.y;
  float f = k > 1e-4 ? (1.0 - exp(-k * age)) / k : age;
  vec3 world = aOrigin + aVel * f - vec3(0.0, 0.5 * aPhys.x * age * age, 0.0);

  vec4 viewPos = viewMatrix * vec4(world, 1.0);

  float size = mix(aParams.z, aParams.w, t);
  float roll = aPhys.w + aPhys.z * age;

  vec2 axisX = vec2(cos(roll), sin(roll));
  float lengthScale = 1.0;
  if (aSprite.w > 0.0) {
    // Stretch along screen-space velocity: a spark that keeps its round shape
    // reads as a floating dot no matter how fast it is actually moving.
    vec3 vv = mat3(viewMatrix) * aVel;
    axisX = length(vv.xy) > 1e-5 ? normalize(vv.xy) : vec2(1.0, 0.0);
    lengthScale = (size + aSprite.w) / max(size, 1e-4);
  }
  vec2 axisY = vec2(-axisX.y, axisX.x);
  viewPos.xy += (axisX * position.x * lengthScale + axisY * position.y) * size;

  float fade = smoothstep(0.0, max(aSprite.z, 1e-4), t) * pow(1.0 - t, 1.3);

  vUv    = aSprite.xy + uTile * ((position.xy + 0.5) * 0.94 + 0.03);
  vColor = mix(aColorA.rgb, aColorB.rgb, t);
  vMeta  = vec3(aColorB.w * fade, aColorA.w, -viewPos.z);

  gl_Position = projectionMatrix * viewPos;
}
`;

const FRAG = /* glsl */`
precision highp float;

uniform sampler2D uAtlas;
uniform sampler2D uDepth;
uniform vec2  uResolution;
uniform float uNear, uFar;
uniform float uSoft;      // fade distance in metres; 0 disables the depth read

varying vec2 vUv;
varying vec3 vColor;
varying vec3 vMeta;

float linearDepth(float d){
  float z = d * 2.0 - 1.0;
  return (2.0 * uNear * uFar) / (uFar + uNear - z * (uFar - uNear));
}

void main(){
  vec4 tex = texture2D(uAtlas, vUv);
  float a = tex.a * vMeta.x;

  // Soft particles: without this a smoke card intersecting a wall draws a
  // razor-straight seam that instantly reads as a billboard.
  if (uSoft > 0.0) {
    float scene = linearDepth(texture2D(uDepth, gl_FragCoord.xy / uResolution).r);
    a *= clamp((scene - vMeta.z) / uSoft, 0.0, 1.0);
  }
  if (a <= 0.003) discard;

  gl_FragColor = vec4(vColor * tex.rgb * a, a * vMeta.y);
}
`;

const RING_FRAG = /* glsl */`
uniform vec3  uColor;
uniform float uEdge, uWidth, uOpacity;
varying vec2 vUv;
void main(){
  float r = length(vUv - 0.5) * 2.0;
  float ring = exp(-pow((r - uEdge) / max(uWidth, 1e-3), 2.0));
  gl_FragColor = vec4(uColor, ring * uOpacity * step(r, 1.0));
}
`;

const RING_VERT = /* glsl */`
varying vec2 vUv;
void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
`;

/* ------------------------------------------------- per-surface impact recipes */

// Every entry layers at least two of: sparks (additive, fast), a dust body
// (alpha, slow, expanding), solid debris under gravity, and a lingering wisp.
// A single grey puff is what an impact looks like when nobody layered it.
const SURFACES = {
  concrete: {
    sparks: 5, sparkCol: [3.2, 1.5, 0.45], sparkSpeed: [3.5, 8], sparkLife: [0.06, 0.16],
    dust: 4, dustCol: [0.46, 0.44, 0.40], dustEnd: [0.28, 0.27, 0.26], dustSize: [0.10, 0.62],
    chips: 4, chipCol: [0.30, 0.29, 0.27], chipSize: 0.030,
    wisp: 1, wispCol: [0.34, 0.33, 0.31],
  },
  metal: {
    // No dust at all — metal throws hot fragments, and adding grey to it is the
    // single most common tell of a stock particle preset.
    sparks: 16, sparkCol: [5.5, 2.4, 0.5], sparkSpeed: [5, 13], sparkLife: [0.12, 0.42],
    dust: 0, dustCol: [0, 0, 0], dustEnd: [0, 0, 0], dustSize: [0, 0],
    chips: 2, chipCol: [0.42, 0.40, 0.38], chipSize: 0.020,
    wisp: 1, wispCol: [0.30, 0.29, 0.28],
  },
  wood: {
    sparks: 0, sparkCol: [0, 0, 0], sparkSpeed: [0, 0], sparkLife: [0, 0],
    dust: 3, dustCol: [0.44, 0.33, 0.21], dustEnd: [0.30, 0.23, 0.15], dustSize: [0.08, 0.44],
    chips: 7, chipCol: [0.40, 0.28, 0.16], chipSize: 0.042,
    wisp: 1, wispCol: [0.32, 0.26, 0.19],
  },
  sand: {
    sparks: 0, sparkCol: [0, 0, 0], sparkSpeed: [0, 0], sparkLife: [0, 0],
    dust: 7, dustCol: [0.62, 0.53, 0.37], dustEnd: [0.44, 0.38, 0.28], dustSize: [0.14, 0.95],
    chips: 5, chipCol: [0.50, 0.43, 0.30], chipSize: 0.018,
    wisp: 0, wispCol: [0.5, 0.45, 0.34],
  },
  glass: {
    sparks: 7, sparkCol: [2.6, 3.4, 4.4], sparkSpeed: [3, 9], sparkLife: [0.10, 0.30],
    dust: 1, dustCol: [0.60, 0.66, 0.70], dustEnd: [0.44, 0.48, 0.52], dustSize: [0.06, 0.26],
    chips: 9, chipCol: [0.66, 0.74, 0.80], chipSize: 0.024,
    wisp: 0, wispCol: [0.5, 0.55, 0.6],
  },
  flesh: {
    sparks: 0, sparkCol: [0, 0, 0], sparkSpeed: [0, 0], sparkLife: [0, 0],
    dust: 4, dustCol: [0.34, 0.020, 0.020], dustEnd: [0.13, 0.012, 0.012], dustSize: [0.09, 0.40],
    chips: 8, chipCol: [0.30, 0.018, 0.016], chipSize: 0.020,
    wisp: 0, wispCol: [0.2, 0.02, 0.02],
  },
};

/* ------------------------------------------------------------------ system */

export class ParticleSystem {
  constructor(scene, settings = {}) {
    this.scene = scene;
    this.settings = settings;
    this.time = 0;
    this.enabled = true;

    this.max = Math.max(64, settings.maxParticles | 0 || 700);
    // Every effect scales its emitter counts by this, so a phone gets the same
    // layered composition at a lower density rather than a different effect.
    this.quality = Math.min(1.25, Math.max(0.35, this.max / 900));

    this._cursor = 0;
    this._used = 0;          // high-water mark; caps instanceCount before the pool fills
    this._dirtyMin = Infinity;
    this._dirtyMax = -Infinity;
    this._idleUntil = 0;

    this._buildPool();
    this._buildRings();
    this._buildShells();

    this._v = new THREE.Vector3();
    this._n = new THREE.Vector3();
    this._t = new THREE.Vector3();
    this._q = new THREE.Quaternion();
    this._m = new THREE.Matrix4();
    this._scale = new THREE.Vector3(1, 1, 1);

    this._p = {
      x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0,
      life: 1, size0: 0.2, size1: 0.2,
      r0: 1, g0: 1, b0: 1, r1: 1, g1: 1, b1: 1,
      alpha: 1, blend: 1, gravity: 0, drag: 0, spin: 0, roll: 0,
      su: 0, sv: 0, fadeIn: 0.06, stretch: 0,
    };
  }

  /* ------------------------------------------------------------- resources */

  _buildPool() {
    const n = this.max;
    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0,
    ]), 3));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    geo.instanceCount = 0;

    const mk = (name, items) => {
      const a = new THREE.InstancedBufferAttribute(new Float32Array(n * items), items);
      a.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute(name, a);
      return a;
    };
    this.aOrigin = mk('aOrigin', 3);
    this.aVel    = mk('aVel', 3);
    this.aParams = mk('aParams', 4);
    this.aPhys   = mk('aPhys', 4);
    this.aSprite = mk('aSprite', 4);
    this.aColorA = mk('aColorA', 4);
    this.aColorB = mk('aColorB', 4);
    this._attrs = [this.aOrigin, this.aVel, this.aParams, this.aPhys,
                   this.aSprite, this.aColorA, this.aColorB];

    // Spawn times start far in the past so nothing draws before the first emit.
    for (let i = 0; i < n; i++) this.aParams.array[i * 4] = -1e6;

    _atlas = _atlas || buildAtlas();
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uTime: { value: 0 },
        uTile: { value: new THREE.Vector2(TILE_U, TILE_V) },
        uAtlas: { value: _atlas },
        uDepth: { value: null },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uNear: { value: 0.05 }, uFar: { value: 500 },
        uSoft: { value: 0 },
      },
      transparent: true,
      depthTest: true,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
    });

    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;   // instances live anywhere; the pool is the bound
    this.mesh.renderOrder = 10;
    this.mesh.matrixAutoUpdate = false;
    this.scene.add(this.mesh);
  }

  _buildRings() {
    this.rings = [];
    const geo = new THREE.PlaneGeometry(1, 1);
    for (let i = 0; i < 3; i++) {
      const mat = new THREE.ShaderMaterial({
        vertexShader: RING_VERT,
        fragmentShader: RING_FRAG,
        uniforms: {
          uColor: { value: new THREE.Color(1.6, 1.2, 0.9) },
          uEdge: { value: 0.5 }, uWidth: { value: 0.12 }, uOpacity: { value: 0 },
        },
        transparent: true, depthWrite: false, depthTest: true,
        blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      });
      const m = new THREE.Mesh(geo, mat);
      m.frustumCulled = false;
      m.visible = false;
      m.renderOrder = 9;
      this.scene.add(m);
      this.rings.push({ mesh: m, mat, age: 0, life: 0, radius: 1 });
    }
  }

  _buildShells() {
    this.shellMax = Math.max(8, Math.min(28, Math.round(this.max / 60)));
    // Slight taper reads as a case rather than a rod at the size these tumble by.
    const geo = new THREE.CylinderGeometry(0.0048, 0.0056, 0.024, 7, 1, false);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xb08a3c, metalness: 0.95, roughness: 0.32,
    });
    this.shellMesh = new THREE.InstancedMesh(geo, mat, this.shellMax);
    this.shellMesh.frustumCulled = false;
    this.shellMesh.castShadow = false;
    this.shellMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.scene.add(this.shellMesh);

    const n = this.shellMax;
    this.shells = {
      pos: new Float32Array(n * 3), vel: new Float32Array(n * 3),
      quat: new Float32Array(n * 4), spin: new Float32Array(n * 3),
      age: new Float32Array(n), live: new Uint8Array(n), cursor: 0, count: 0,
    };
    // Ground plane for the bounce. FX has no collider access, so the level's
    // floor height is published here for whoever spawns the shells.
    this.shellGroundY = 0;
    this._hideAllShells();
  }

  _hideAllShells() {
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let i = 0; i < this.shellMax; i++) this.shellMesh.setMatrixAt(i, zero);
    this.shellMesh.instanceMatrix.needsUpdate = true;
  }

  /* -------------------------------------------------------------- emission */

  /** Resets the scratch descriptor. Effects override only what they care about. */
  _reset() {
    const p = this._p;
    p.x = p.y = p.z = 0; p.vx = p.vy = p.vz = 0;
    p.life = 1; p.size0 = 0.2; p.size1 = 0.2;
    p.r0 = p.g0 = p.b0 = 1; p.r1 = p.g1 = p.b1 = 1;
    p.alpha = 1; p.blend = 1; p.gravity = 0; p.drag = 0;
    p.spin = 0; p.roll = Math.random() * 6.283185;
    p.su = SPR.PUFF[0]; p.sv = SPR.PUFF[1];
    p.fadeIn = 0.06; p.stretch = 0;
    return p;
  }

  _sprite(s) { const p = this._p; p.su = s[0]; p.sv = s[1]; return p; }

  /** Writes the scratch descriptor into the pool, recycling the oldest slot. */
  _emit() {
    const p = this._p;
    const i = this._cursor;
    this._cursor = (this._cursor + 1) % this.max;
    if (this._used < this.max) this._used++;

    const i3 = i * 3, i4 = i * 4;
    const o = this.aOrigin.array, v = this.aVel.array, pr = this.aParams.array;
    const ph = this.aPhys.array, sp = this.aSprite.array;
    const ca = this.aColorA.array, cb = this.aColorB.array;

    o[i3] = p.x; o[i3 + 1] = p.y; o[i3 + 2] = p.z;
    v[i3] = p.vx; v[i3 + 1] = p.vy; v[i3 + 2] = p.vz;
    pr[i4] = this.time; pr[i4 + 1] = p.life; pr[i4 + 2] = p.size0; pr[i4 + 3] = p.size1;
    ph[i4] = p.gravity; ph[i4 + 1] = p.drag; ph[i4 + 2] = p.spin; ph[i4 + 3] = p.roll;
    sp[i4] = p.su; sp[i4 + 1] = p.sv; sp[i4 + 2] = p.fadeIn; sp[i4 + 3] = p.stretch;
    ca[i4] = p.r0; ca[i4 + 1] = p.g0; ca[i4 + 2] = p.b0; ca[i4 + 3] = p.blend;
    cb[i4] = p.r1; cb[i4 + 1] = p.g1; cb[i4 + 2] = p.b1; cb[i4 + 3] = p.alpha;

    if (i < this._dirtyMin) this._dirtyMin = i;
    if (i > this._dirtyMax) this._dirtyMax = i;
    this._idleUntil = Math.max(this._idleUntil, this.time + p.life);
  }

  _count(base) { return Math.max(1, Math.round(base * this.quality)); }

  /** Unit vector `spread` radians-ish off `n` — 0 is exact, 1 is a wide cone. */
  _cone(out, n, spread) {
    const u = Math.random() * 2 - 1, th = Math.random() * 6.283185;
    const s = Math.sqrt(Math.max(0, 1 - u * u));
    out.set(s * Math.cos(th), s * Math.sin(th), u);
    out.multiplyScalar(spread).add(n);
    if (out.lengthSq() < 1e-8) out.copy(n);
    return out.normalize();
  }

  /* ---------------------------------------------------------------- public */

  setDepthTexture(tex) {
    const u = this.material.uniforms;
    u.uDepth.value = tex || null;
    // Soft fade needs a depth buffer that is not the one being written to, i.e.
    // a prepass or a copy. Without one we simply draw hard-edged cards.
    u.uSoft.value = tex ? 0.45 : 0;
    if (tex?.image) u.uResolution.value.set(tex.image.width || 1, tex.image.height || 1);
  }

  update(dt, camera) {
    this.time += dt;
    const u = this.material.uniforms;
    u.uTime.value = this.time;
    if (camera) { u.uNear.value = camera.near; u.uFar.value = camera.far; }

    // Once every particle has expired the pool draws nothing at all.
    this.mesh.geometry.instanceCount = this.time < this._idleUntil ? this._used : 0;
    this._upload();

    this._updateRings(dt, camera);
    this._updateShells(dt);
  }

  clear() {
    const pr = this.aParams.array;
    for (let i = 0; i < this.max; i++) pr[i * 4] = -1e6;
    this.aParams.needsUpdate = true;
    this._cursor = 0; this._used = 0; this._idleUntil = 0;
    this._dirtyMin = Infinity; this._dirtyMax = -Infinity;
    this.mesh.geometry.instanceCount = 0;

    for (const r of this.rings) { r.life = 0; r.mesh.visible = false; }
    this.shells.live.fill(0);
    this.shells.cursor = 0; this.shells.count = 0;
    this._hideAllShells();
  }

  muzzleFlash(position, direction) {
    if (!this.enabled) return;
    const dir = this._n.copy(direction).normalize();
    const px = position.x, py = position.y, pz = position.z;
    let p;

    // Star + two glow cards. The star carries the shape, the glows carry the
    // bloom energy; one sprite doing both always looks like a decal on the gun.
    p = this._reset(); this._sprite(SPR.FLASH);
    p.x = px + dir.x * 0.06; p.y = py + dir.y * 0.06; p.z = pz + dir.z * 0.06;
    p.life = 0.055; p.size0 = 0.30; p.size1 = 0.52; p.blend = 0; p.fadeIn = 0.004;
    p.r0 = 7.0; p.g0 = 5.4; p.b0 = 2.6; p.r1 = 3.0; p.g1 = 1.4; p.b1 = 0.4;
    this._emit();

    for (let i = 0; i < 2; i++) {
      p = this._reset(); this._sprite(SPR.GLOW);
      const d = 0.05 + i * 0.10;
      p.x = px + dir.x * d; p.y = py + dir.y * d; p.z = pz + dir.z * d;
      p.life = 0.07 + i * 0.03; p.size0 = 0.22 + i * 0.10; p.size1 = 0.40 + i * 0.16;
      p.blend = 0; p.fadeIn = 0.004; p.alpha = 0.9 - i * 0.35;
      p.r0 = 5.5; p.g0 = 3.4; p.b0 = 1.2; p.r1 = 2.0; p.g1 = 0.8; p.b1 = 0.2;
      this._emit();
    }

    const sparks = this._count(8);
    for (let i = 0; i < sparks; i++) {
      const d = this._cone(this._v, dir, 0.30);
      const sp = rnd(4, 12);
      p = this._reset(); this._sprite(SPR.SPARK);
      p.x = px + dir.x * 0.08; p.y = py + dir.y * 0.08; p.z = pz + dir.z * 0.08;
      p.vx = d.x * sp; p.vy = d.y * sp; p.vz = d.z * sp;
      p.life = rnd(0.08, 0.26); p.size0 = 0.016; p.size1 = 0.006;
      p.blend = 0; p.fadeIn = 0.004; p.gravity = 6; p.drag = 5.5; p.stretch = rnd(0.10, 0.30);
      p.r0 = 5.0; p.g0 = 2.2; p.b0 = 0.5; p.r1 = 1.4; p.g1 = 0.30; p.b1 = 0.05;
      this._emit();
    }

    const wisps = this._count(2);
    for (let i = 0; i < wisps; i++) {
      const d = this._cone(this._v, dir, 0.55);
      p = this._reset(); this._sprite(SPR.SMOKE);
      p.x = px + dir.x * 0.12; p.y = py + dir.y * 0.12; p.z = pz + dir.z * 0.12;
      p.vx = d.x * 1.6; p.vy = d.y * 1.6 + 0.5; p.vz = d.z * 1.6;
      p.life = rnd(0.5, 1.1); p.size0 = 0.07; p.size1 = 0.42;
      p.alpha = 0.20; p.fadeIn = 0.14; p.drag = 2.4; p.gravity = -0.35;
      p.spin = rnd(-1.6, 1.6);
      p.r0 = 0.34; p.g0 = 0.33; p.b0 = 0.31; p.r1 = 0.20; p.g1 = 0.20; p.b1 = 0.19;
      this._emit();
    }
  }

  impact(position, normal, surfaceTag = 'concrete') {
    if (!this.enabled) return;
    const cfg = SURFACES[surfaceTag] || SURFACES.concrete;
    const n = this._n.copy(normal).normalize();
    const px = position.x, py = position.y, pz = position.z;
    let p;

    for (let i = 0, c = cfg.sparks ? this._count(cfg.sparks) : 0; i < c; i++) {
      const d = this._cone(this._v, n, 0.85);
      const sp = rnd(cfg.sparkSpeed[0], cfg.sparkSpeed[1]);
      p = this._reset(); this._sprite(SPR.SPARK);
      p.x = px + n.x * 0.02; p.y = py + n.y * 0.02; p.z = pz + n.z * 0.02;
      p.vx = d.x * sp; p.vy = d.y * sp; p.vz = d.z * sp;
      p.life = rnd(cfg.sparkLife[0], cfg.sparkLife[1]);
      p.size0 = 0.015; p.size1 = 0.004;
      p.blend = 0; p.fadeIn = 0.004; p.gravity = 8.5; p.drag = 3.2;
      p.stretch = rnd(0.08, 0.28);
      p.r0 = cfg.sparkCol[0]; p.g0 = cfg.sparkCol[1]; p.b0 = cfg.sparkCol[2];
      p.r1 = cfg.sparkCol[0] * 0.18; p.g1 = cfg.sparkCol[1] * 0.10; p.b1 = cfg.sparkCol[2] * 0.06;
      this._emit();
    }

    for (let i = 0, c = cfg.dust ? this._count(cfg.dust) : 0; i < c; i++) {
      const d = this._cone(this._v, n, 0.95);
      const sp = rnd(0.5, 2.4);
      p = this._reset();
      this._sprite(surfaceTag === 'flesh' ? SPR.BLOOD : SPR.DUST);
      p.x = px + n.x * 0.03; p.y = py + n.y * 0.03; p.z = pz + n.z * 0.03;
      p.vx = d.x * sp; p.vy = d.y * sp + 0.35; p.vz = d.z * sp;
      p.life = rnd(0.45, 1.15);
      p.size0 = cfg.dustSize[0] * rnd(0.8, 1.3);
      p.size1 = cfg.dustSize[1] * rnd(0.8, 1.3);
      p.alpha = surfaceTag === 'flesh' ? 0.55 : 0.42;
      p.fadeIn = 0.09; p.drag = 3.0; p.gravity = surfaceTag === 'flesh' ? 1.2 : -0.2;
      p.spin = rnd(-1.2, 1.2);
      p.r0 = cfg.dustCol[0]; p.g0 = cfg.dustCol[1]; p.b0 = cfg.dustCol[2];
      p.r1 = cfg.dustEnd[0]; p.g1 = cfg.dustEnd[1]; p.b1 = cfg.dustEnd[2];
      this._emit();
    }

    for (let i = 0, c = cfg.chips ? this._count(cfg.chips) : 0; i < c; i++) {
      const d = this._cone(this._v, n, 0.7);
      const sp = rnd(2.5, 7.5);
      const flesh = surfaceTag === 'flesh';
      p = this._reset();
      this._sprite(flesh ? SPR.BLOOD : SPR.CHIP);
      p.x = px + n.x * 0.02; p.y = py + n.y * 0.02; p.z = pz + n.z * 0.02;
      p.vx = d.x * sp; p.vy = d.y * sp + 1.0; p.vz = d.z * sp;
      p.life = rnd(0.6, 1.4);
      p.size0 = cfg.chipSize * rnd(0.6, 1.5);
      // Wood splinters are long slivers, not cubes; stretching sells the shape.
      p.size1 = p.size0 * (surfaceTag === 'wood' ? 1.0 : 0.85);
      p.stretch = surfaceTag === 'wood' ? rnd(0.02, 0.06) : (flesh ? rnd(0.01, 0.04) : 0);
      p.alpha = 1.0; p.fadeIn = 0.01; p.gravity = 11; p.drag = 0.6;
      p.spin = rnd(-14, 14);
      p.r0 = cfg.chipCol[0]; p.g0 = cfg.chipCol[1]; p.b0 = cfg.chipCol[2];
      p.r1 = cfg.chipCol[0] * 0.7; p.g1 = cfg.chipCol[1] * 0.7; p.b1 = cfg.chipCol[2] * 0.7;
      this._emit();
    }

    for (let i = 0, c = cfg.wisp ? this._count(cfg.wisp) : 0; i < c; i++) {
      p = this._reset(); this._sprite(SPR.SMOKE);
      p.x = px + n.x * 0.05; p.y = py + n.y * 0.05; p.z = pz + n.z * 0.05;
      p.vx = n.x * 0.35; p.vy = n.y * 0.35 + 0.45; p.vz = n.z * 0.35;
      p.life = rnd(1.1, 2.2); p.size0 = 0.10; p.size1 = rnd(0.55, 0.9);
      p.alpha = 0.16; p.fadeIn = 0.28; p.drag = 1.6; p.gravity = -0.28;
      p.spin = rnd(-0.8, 0.8);
      p.r0 = cfg.wispCol[0]; p.g0 = cfg.wispCol[1]; p.b0 = cfg.wispCol[2];
      p.r1 = cfg.wispCol[0] * 0.55; p.g1 = cfg.wispCol[1] * 0.55; p.b1 = cfg.wispCol[2] * 0.55;
      this._emit();
    }
  }

  blood(position, direction) {
    if (!this.enabled) return;
    const dir = this._n.copy(direction).normalize();
    const px = position.x, py = position.y, pz = position.z;
    let p;

    // Mist first: the low-alpha cloud is what the eye reads as "wet", the
    // droplets on top are what make the direction of the hit legible.
    const mist = this._count(5);
    for (let i = 0; i < mist; i++) {
      const d = this._cone(this._v, dir, 0.7);
      p = this._reset(); this._sprite(SPR.BLOOD);
      p.x = px; p.y = py; p.z = pz;
      p.vx = d.x * rnd(0.8, 3.0); p.vy = d.y * rnd(0.8, 3.0) + 0.4; p.vz = d.z * rnd(0.8, 3.0);
      p.life = rnd(0.35, 0.8); p.size0 = rnd(0.06, 0.12); p.size1 = rnd(0.30, 0.55);
      p.alpha = 0.5; p.fadeIn = 0.07; p.drag = 3.6; p.gravity = 1.6;
      p.spin = rnd(-2, 2);
      p.r0 = 0.36; p.g0 = 0.022; p.b0 = 0.020;
      p.r1 = 0.12; p.g1 = 0.010; p.b1 = 0.010;
      this._emit();
    }

    const drops = this._count(12);
    for (let i = 0; i < drops; i++) {
      const d = this._cone(this._v, dir, 0.55);
      const sp = rnd(2.5, 9);
      p = this._reset(); this._sprite(SPR.BLOOD);
      p.x = px; p.y = py; p.z = pz;
      p.vx = d.x * sp; p.vy = d.y * sp + rnd(0.5, 2.5); p.vz = d.z * sp;
      p.life = rnd(0.5, 1.3); p.size0 = rnd(0.010, 0.026); p.size1 = p.size0 * 0.85;
      p.alpha = 0.95; p.fadeIn = 0.01; p.gravity = 12; p.drag = 0.7;
      p.stretch = rnd(0.02, 0.07);
      p.r0 = 0.30; p.g0 = 0.016; p.b0 = 0.014;
      p.r1 = 0.16; p.g1 = 0.010; p.b1 = 0.008;
      this._emit();
    }
  }

  explosion(position, radius = 2.5) {
    if (!this.enabled) return;
    const px = position.x, py = position.y, pz = position.z;
    const R = Math.max(0.5, radius);
    let p;

    // 1. Core: gone in three frames, but it is what makes the blast feel violent.
    p = this._reset(); this._sprite(SPR.FLASH);
    p.x = px; p.y = py; p.z = pz;
    p.life = 0.09; p.size0 = R * 0.7; p.size1 = R * 1.9;
    p.blend = 0; p.fadeIn = 0.005;
    p.r0 = 9.0; p.g0 = 7.5; p.b0 = 4.5; p.r1 = 4.0; p.g1 = 1.6; p.b1 = 0.35;
    this._emit();

    // 2. Fireball: additive orange that cools into near-black as it rises.
    const fire = this._count(12);
    for (let i = 0; i < fire; i++) {
      const d = this._cone(this._v, this._t.set(0, 1, 0), 1.6);
      p = this._reset(); this._sprite(SPR.PUFF);
      p.x = px + d.x * R * 0.25; p.y = py + d.y * R * 0.25; p.z = pz + d.z * R * 0.25;
      p.vx = d.x * rnd(2, 7); p.vy = d.y * rnd(2, 7) + 1.5; p.vz = d.z * rnd(2, 7);
      p.life = rnd(0.35, 0.85); p.size0 = R * rnd(0.3, 0.6); p.size1 = R * rnd(0.9, 1.6);
      p.blend = 0; p.alpha = 0.9; p.fadeIn = 0.03; p.drag = 3.2; p.gravity = -1.2;
      p.spin = rnd(-2, 2);
      p.r0 = 5.0; p.g0 = 2.0; p.b0 = 0.35; p.r1 = 0.35; p.g1 = 0.10; p.b1 = 0.03;
      this._emit();
    }

    // 3. Black smoke: alpha-blended, slower and much longer lived than the fire.
    const smoke = this._count(11);
    for (let i = 0; i < smoke; i++) {
      const d = this._cone(this._v, this._t.set(0, 1, 0), 1.3);
      p = this._reset(); this._sprite(SPR.SMOKE);
      p.x = px + d.x * R * 0.3; p.y = py + d.y * R * 0.3; p.z = pz + d.z * R * 0.3;
      p.vx = d.x * rnd(1, 4); p.vy = d.y * rnd(1, 4) + 1.2; p.vz = d.z * rnd(1, 4);
      p.life = rnd(1.8, 3.4); p.size0 = R * rnd(0.35, 0.6); p.size1 = R * rnd(1.6, 2.6);
      p.alpha = 0.55; p.fadeIn = 0.16; p.drag = 1.5; p.gravity = -0.55;
      p.spin = rnd(-0.9, 0.9);
      p.r0 = 0.16; p.g0 = 0.145; p.b0 = 0.135;
      p.r1 = 0.10; p.g1 = 0.095; p.b1 = 0.09;
      this._emit();
    }

    // 4. Ground ring: dust thrown outward along the floor, not up with the ball.
    const ring = this._count(12);
    for (let i = 0; i < ring; i++) {
      const a = (i / ring) * 6.283185 + rnd(-0.2, 0.2);
      const cx = Math.cos(a), cz = Math.sin(a);
      const sp = rnd(5, 11);
      p = this._reset(); this._sprite(SPR.DUST);
      p.x = px + cx * R * 0.3; p.y = py - R * 0.35; p.z = pz + cz * R * 0.3;
      p.vx = cx * sp; p.vy = rnd(0.3, 1.4); p.vz = cz * sp;
      p.life = rnd(1.0, 2.0); p.size0 = R * 0.30; p.size1 = R * rnd(1.2, 1.9);
      p.alpha = 0.34; p.fadeIn = 0.12; p.drag = 2.6; p.gravity = -0.1;
      p.spin = rnd(-1, 1);
      p.r0 = 0.50; p.g0 = 0.46; p.b0 = 0.40; p.r1 = 0.30; p.g1 = 0.28; p.b1 = 0.26;
      this._emit();
    }

    // 5. Debris + trailing sparks.
    const chunks = this._count(14);
    for (let i = 0; i < chunks; i++) {
      const d = this._cone(this._v, this._t.set(0, 1, 0), 1.9);
      const sp = rnd(6, 18);
      p = this._reset(); this._sprite(SPR.CHIP);
      p.x = px; p.y = py; p.z = pz;
      p.vx = d.x * sp; p.vy = Math.abs(d.y) * sp + 2.0; p.vz = d.z * sp;
      p.life = rnd(1.0, 2.2); p.size0 = rnd(0.03, 0.075); p.size1 = p.size0;
      p.gravity = 13; p.drag = 0.35; p.spin = rnd(-18, 18);
      p.r0 = 0.24; p.g0 = 0.22; p.b0 = 0.20; p.r1 = 0.14; p.g1 = 0.13; p.b1 = 0.12;
      this._emit();
    }
    const sparks = this._count(16);
    for (let i = 0; i < sparks; i++) {
      const d = this._cone(this._v, this._t.set(0, 1, 0), 2.0);
      const sp = rnd(8, 22);
      p = this._reset(); this._sprite(SPR.SPARK);
      p.x = px; p.y = py; p.z = pz;
      p.vx = d.x * sp; p.vy = d.y * sp + 1.0; p.vz = d.z * sp;
      p.life = rnd(0.25, 0.8); p.size0 = 0.020; p.size1 = 0.005;
      p.blend = 0; p.fadeIn = 0.005; p.gravity = 10; p.drag = 1.8;
      p.stretch = rnd(0.15, 0.5);
      p.r0 = 6.0; p.g0 = 2.6; p.b0 = 0.5; p.r1 = 1.0; p.g1 = 0.2; p.b1 = 0.03;
      this._emit();
    }

    this._spawnRing(px, py, pz, R);
  }

  tracer(fromVec3, toVec3, speed = 340) {
    if (!this.enabled) return;
    this._v.copy(toVec3).sub(fromVec3);
    const dist = this._v.length();
    if (dist < 0.05) return;
    const life = Math.min(dist / Math.max(speed, 1), 1.2);
    this._v.multiplyScalar(1 / dist);

    // The vertex shader already integrates constant velocity, so a tracer is
    // just one particle whose life happens to end exactly at the impact point.
    let p = this._reset(); this._sprite(SPR.SPARK);
    p.x = fromVec3.x; p.y = fromVec3.y; p.z = fromVec3.z;
    p.vx = this._v.x * speed; p.vy = this._v.y * speed; p.vz = this._v.z * speed;
    p.life = life; p.size0 = 0.028; p.size1 = 0.016;
    p.blend = 0; p.fadeIn = 0.02; p.stretch = 3.2;
    p.r0 = 6.0; p.g0 = 3.6; p.b0 = 1.2; p.r1 = 2.2; p.g1 = 0.9; p.b1 = 0.25;
    this._emit();

    p = this._reset(); this._sprite(SPR.GLOW);
    p.x = fromVec3.x; p.y = fromVec3.y; p.z = fromVec3.z;
    p.vx = this._v.x * speed; p.vy = this._v.y * speed; p.vz = this._v.z * speed;
    p.life = life; p.size0 = 0.10; p.size1 = 0.06;
    p.blend = 0; p.alpha = 0.30; p.fadeIn = 0.02; p.stretch = 1.6;
    p.r0 = 1.8; p.g0 = 0.95; p.b0 = 0.30; p.r1 = 0.7; p.g1 = 0.28; p.b1 = 0.08;
    this._emit();
  }

  smoke(position, amount = 4) {
    if (!this.enabled) return;
    const c = this._count(amount);
    for (let i = 0; i < c; i++) {
      const p = this._reset(); this._sprite(SPR.SMOKE);
      p.x = position.x + rnd(-0.12, 0.12);
      p.y = position.y + rnd(-0.06, 0.10);
      p.z = position.z + rnd(-0.12, 0.12);
      p.vx = rnd(-0.35, 0.35); p.vy = rnd(0.35, 0.95); p.vz = rnd(-0.35, 0.35);
      p.life = rnd(1.6, 3.2); p.size0 = rnd(0.12, 0.24); p.size1 = rnd(0.9, 1.6);
      p.alpha = 0.24; p.fadeIn = 0.22; p.drag = 1.1; p.gravity = -0.45;
      p.spin = rnd(-0.7, 0.7);
      p.r0 = 0.30; p.g0 = 0.295; p.b0 = 0.285;
      p.r1 = 0.18; p.g1 = 0.178; p.b1 = 0.175;
      this._emit();
    }
  }

  shell(position, velocityVec3) {
    if (!this.enabled) return;
    const s = this.shells;
    const i = s.cursor;
    s.cursor = (s.cursor + 1) % this.shellMax;
    if (s.count < this.shellMax) s.count++;

    const i3 = i * 3, i4 = i * 4;
    s.pos[i3] = position.x; s.pos[i3 + 1] = position.y; s.pos[i3 + 2] = position.z;
    s.vel[i3] = velocityVec3.x; s.vel[i3 + 1] = velocityVec3.y; s.vel[i3 + 2] = velocityVec3.z;
    s.spin[i3] = rnd(-26, 26); s.spin[i3 + 1] = rnd(-26, 26); s.spin[i3 + 2] = rnd(-26, 26);
    this._q.set(Math.random(), Math.random(), Math.random(), Math.random()).normalize();
    s.quat[i4] = this._q.x; s.quat[i4 + 1] = this._q.y;
    s.quat[i4 + 2] = this._q.z; s.quat[i4 + 3] = this._q.w;
    s.age[i] = 0; s.live[i] = 1;
  }

  /* --------------------------------------------------------------- private */

  _upload() {
    if (this._dirtyMax < this._dirtyMin) return;
    const start = this._dirtyMin, count = this._dirtyMax - start + 1;
    for (const a of this._attrs) {
      // A partial range matters here: a burst touches ~40 slots out of 2400 and
      // re-uploading the whole buffer would cost more than the effect itself.
      if (a.clearUpdateRanges && a.addUpdateRange) {
        a.clearUpdateRanges();
        a.addUpdateRange(start * a.itemSize, count * a.itemSize);
      }
      a.needsUpdate = true;
    }
    this._dirtyMin = Infinity; this._dirtyMax = -Infinity;
  }

  _spawnRing(x, y, z, radius) {
    let slot = this.rings.find(r => r.life <= 0) || this.rings[0];
    slot.age = 0; slot.life = 0.36; slot.radius = radius * 5.5;
    slot.mesh.position.set(x, y, z);
    slot.mesh.visible = true;
    slot.mat.uniforms.uOpacity.value = 0.85;
  }

  _updateRings(dt, camera) {
    for (const r of this.rings) {
      if (r.life <= 0) continue;
      r.age += dt;
      const t = r.age / r.life;
      if (t >= 1) { r.life = 0; r.mesh.visible = false; continue; }
      // Sqrt easing: the shock front decelerates, and a linear ring reads as a
      // growing circle rather than a pressure wave.
      const s = r.radius * Math.sqrt(t);
      r.mesh.scale.setScalar(Math.max(0.001, s));
      if (camera) r.mesh.quaternion.copy(camera.quaternion);
      r.mat.uniforms.uEdge.value = 0.80;
      r.mat.uniforms.uWidth.value = 0.07 + 0.10 * t;
      r.mat.uniforms.uOpacity.value = 0.9 * (1 - t) * (1 - t);
    }
  }

  _updateShells(dt) {
    const s = this.shells;
    if (!s.count) return;
    let any = false;

    for (let i = 0; i < this.shellMax; i++) {
      if (!s.live[i]) continue;
      any = true;
      const i3 = i * 3, i4 = i * 4;
      s.age[i] += dt;

      s.vel[i3 + 1] -= 9.81 * dt;
      s.pos[i3] += s.vel[i3] * dt;
      s.pos[i3 + 1] += s.vel[i3 + 1] * dt;
      s.pos[i3 + 2] += s.vel[i3 + 2] * dt;

      const rest = this.shellGroundY + 0.006;
      if (s.pos[i3 + 1] <= rest && s.vel[i3 + 1] < 0) {
        s.pos[i3 + 1] = rest;
        s.vel[i3 + 1] *= -0.30;                 // brass barely bounces
        s.vel[i3] *= 0.55; s.vel[i3 + 2] *= 0.55;
        s.spin[i3] *= 0.4; s.spin[i3 + 1] *= 0.4; s.spin[i3 + 2] *= 0.4;
        if (Math.abs(s.vel[i3 + 1]) < 0.35) s.vel[i3 + 1] = 0;
      }

      // Quaternion derivative q' = 0.5 * omega * q, renormalised each step.
      const q = this._q.set(s.quat[i4], s.quat[i4 + 1], s.quat[i4 + 2], s.quat[i4 + 3]);
      const wx = s.spin[i3] * dt * 0.5, wy = s.spin[i3 + 1] * dt * 0.5, wz = s.spin[i3 + 2] * dt * 0.5;
      const nx = q.x + (wx * q.w + wy * q.z - wz * q.y);
      const ny = q.y + (wy * q.w + wz * q.x - wx * q.z);
      const nz = q.z + (wz * q.w + wx * q.y - wy * q.x);
      const nw = q.w - (wx * q.x + wy * q.y + wz * q.z);
      q.set(nx, ny, nz, nw).normalize();
      s.quat[i4] = q.x; s.quat[i4 + 1] = q.y; s.quat[i4 + 2] = q.z; s.quat[i4 + 3] = q.w;

      const age = s.age[i];
      if (age >= 4.0) { s.live[i] = 0; this._scale.setScalar(0); }
      else this._scale.setScalar(age > 3.6 ? 1 - (age - 3.6) / 0.4 : 1);

      this._v.set(s.pos[i3], s.pos[i3 + 1], s.pos[i3 + 2]);
      this._m.compose(this._v, q, this._scale);
      this.shellMesh.setMatrixAt(i, this._m);
    }

    if (any) this.shellMesh.instanceMatrix.needsUpdate = true;
  }

  dispose() {
    this.scene.remove(this.mesh, this.shellMesh);
    this.mesh.geometry.dispose();
    this.material.dispose();
    this.shellMesh.geometry.dispose();
    this.shellMesh.material.dispose();
    this.shellMesh.dispose();
    for (const r of this.rings) { this.scene.remove(r.mesh); r.mat.dispose(); }
    this.rings[0]?.mesh.geometry.dispose();
  }
}

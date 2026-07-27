// Textures.js — runtime procedural PBR texture synthesis.
//
// Every surface in the game is generated here into typed arrays and uploaded as
// DataTextures. Nothing is downloaded, so the whole game stays a ~800KB payload
// while still getting albedo + normal + roughness + AO per material.
//
// Normals are derived from the same height field that drives AO and roughness
// variation, which is what keeps a surface reading as one coherent material
// instead of three unrelated noise layers stacked on top of each other.

import * as THREE from 'three';

/* ------------------------------------------------------------------ noise */

const P = new Uint8Array(512);
(function seedPermutation(seed = 1337) {
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  let s = seed;
  for (let i = 255; i > 0; i--) {
    s = (s * 1664525 + 1013904223) >>> 0;      // LCG — deterministic across devices
    const j = s % (i + 1);
    const t = p[i]; p[i] = p[j]; p[j] = t;
  }
  P.set(p); P.set(p, 256);
})();

const fade = t => t * t * t * (t * (t * 6 - 15) + 10);
const lerp = (a, b, t) => a + (b - a) * t;

function grad2(h, x, y) {
  switch (h & 7) {
    case 0: return  x + y; case 1: return  x - y;
    case 2: return -x + y; case 3: return -x - y;
    case 4: return  x;     case 5: return -x;
    case 6: return  y;     default: return -y;
  }
}

/**
 * Perlin noise on a tiling lattice. The result repeats every `per` units in x
 * and `perY` in y, so a caller gets a seamless tile only by passing the period
 * that matches the span it samples: sampling `u * 24` needs `per = 24`.
 *
 * `perY` exists because several materials are deliberately anisotropic — rain
 * streaks and machining marks sample far more cycles across than down — and a
 * single period cannot divide both spans without collapsing to their GCD.
 *
 * Periods must be whole numbers: the lattice index is an integer, so a
 * fractional period aliases neighbouring cells onto each other and the field
 * stops repeating at all.
 */
export function noise2(x, y, per = 256, perY = per) {
  const X0 = Math.floor(x), Y0 = Math.floor(y);
  const xf = x - X0, yf = y - Y0;
  const u = fade(xf), v = fade(yf);
  const xi  = (((X0 % per) + per) % per) & 255;
  const yi  = (((Y0 % perY) + perY) % perY) & 255;
  const xi1 = ((((X0 + 1) % per) + per) % per) & 255;
  const yi1 = ((((Y0 + 1) % perY) + perY) % perY) & 255;

  const aa = P[P[xi] + yi],  ab = P[P[xi] + yi1];
  const ba = P[P[xi1] + yi], bb = P[P[xi1] + yi1];
  return lerp(
    lerp(grad2(aa, xf, yf),     grad2(ba, xf - 1, yf),     u),
    lerp(grad2(ab, xf, yf - 1), grad2(bb, xf - 1, yf - 1), u), v);
}

/**
 * Octave i samples at `freq` times the base coordinates, so its period has to
 * scale by `freq` too — that is why the base span is the right thing to pass and
 * every octave lands back on the tile edge together.
 *
 * Note the output is a sum of signed Perlin values divided by total amplitude,
 * which does NOT span [-1,1]: over the windows sampled here it reaches roughly
 * ±0.3 (2 octaves) narrowing to ±0.22 (5 octaves). Thresholds applied to
 * `fbm(...) * 0.5 + 0.5` must be set against that measured range, not against a
 * nominal 0..1, or the mask silently evaluates to zero.
 */
export function fbm(x, y, octaves = 5, lac = 2, gain = 0.5, per = 256, perY = per) {
  let sum = 0, amp = 1, freq = 1, norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += amp * noise2(x * freq, y * freq, per * freq, perY * freq);
    norm += amp; amp *= gain; freq *= lac;
  }
  return sum / norm;
}

/** Ridged fbm — the sharp creases read as cracks and rock strata. */
export function ridged(x, y, octaves = 5, per = 256, perY = per) {
  let sum = 0, amp = 0.5, freq = 1, norm = 0;
  for (let i = 0; i < octaves; i++) {
    const n = 1 - Math.abs(noise2(x * freq, y * freq, per * freq, perY * freq));
    sum += amp * n * n; norm += amp; amp *= 0.5; freq *= 2;
  }
  return sum / norm;
}

/**
 * Tiling Voronoi. Returns F1 distance, the cell id, and F2-F1 (the edge mask,
 * which is what gives mortar lines and cracked-mud borders).
 */
export function voronoi(x, y, cells = 8) {
  const gx = Math.floor(x * cells), gy = Math.floor(y * cells);
  const fx = x * cells - gx, fy = y * cells - gy;
  let f1 = 8, f2 = 8, id = 0;
  for (let j = -1; j <= 1; j++) {
    for (let i = -1; i <= 1; i++) {
      const cx = ((gx + i) % cells + cells) % cells;
      const cy = ((gy + j) % cells + cells) % cells;
      const h = P[(P[cx & 255] + (cy & 255)) & 255];
      const ox = (h & 15) / 15, oy = ((h >> 4) & 15) / 15;
      const dx = i + ox - fx, dy = j + oy - fy;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < f1) { f2 = f1; f1 = d; id = h; }
      else if (d < f2) { f2 = d; }
    }
  }
  return { f1, f2, edge: f2 - f1, id: id / 255 };
}

const clamp01 = v => v < 0 ? 0 : v > 1 ? 1 : v;
const smoothstep = (e0, e1, x) => { const t = clamp01((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t); };

/* ------------------------------------------------------------- texture I/O */

function makeTexture(data, size, { srgb = false, repeat = 1, aniso = 8 } = {}) {
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(repeat, repeat);
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = aniso;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Sobel-filters a height field into a tangent-space normal map.
 * `strength` is in height-units-per-texel; higher values read as deeper relief.
 */
function heightToNormal(height, size, strength = 2.2) {
  const out = new Uint8Array(size * size * 4);
  const at = (x, y) => height[((y & (size - 1)) * size) + (x & (size - 1))];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const tl = at(x - 1, y - 1), t = at(x, y - 1), tr = at(x + 1, y - 1);
      const l  = at(x - 1, y),                       r = at(x + 1, y);
      const bl = at(x - 1, y + 1), b = at(x, y + 1), br = at(x + 1, y + 1);
      const dx = (tr + 2 * r + br) - (tl + 2 * l + bl);
      const dy = (bl + 2 * b + br) - (tl + 2 * t + tr);
      let nx = -dx * strength, ny = -dy * strength, nz = 1;
      const inv = 1 / Math.sqrt(nx * nx + ny * ny + 1);
      nx *= inv; ny *= inv; nz *= inv;
      const i = (y * size + x) * 4;
      out[i]     = (nx * 0.5 + 0.5) * 255;
      out[i + 1] = (ny * 0.5 + 0.5) * 255;
      out[i + 2] = (nz * 0.5 + 0.5) * 255;
      out[i + 3] = 255;
    }
  }
  return out;
}

/**
 * Cheap screen-space-style cavity AO computed directly on the height field:
 * a texel darker than its neighbourhood average sits in a crevice.
 */
function heightToAO(height, size, radius = 3, strength = 1.0) {
  const out = new Float32Array(size * size);
  const at = (x, y) => height[((y & (size - 1)) * size) + (x & (size - 1))];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const h = at(x, y);
      let occ = 0, n = 0;
      for (let j = -radius; j <= radius; j += 1) {
        for (let i = -radius; i <= radius; i += 1) {
          if (!i && !j) continue;
          occ += Math.max(0, at(x + i, y + j) - h);
          n++;
        }
      }
      out[y * size + x] = clamp01(1 - (occ / n) * strength * 6);
    }
  }
  return out;
}

/**
 * Packs three grayscale channels into one RGBA texture.
 * The engine reads R as AO, G as roughness, B as metalness — the same ORM
 * layout glTF uses, so one sampler covers all three lookups.
 */
function packORM(ao, rough, metal, size) {
  const out = new Uint8Array(size * size * 4);
  for (let i = 0, n = size * size; i < n; i++) {
    out[i * 4]     = clamp01(ao[i]) * 255;
    out[i * 4 + 1] = clamp01(rough[i]) * 255;
    out[i * 4 + 2] = clamp01(metal ? metal[i] : 0) * 255;
    out[i * 4 + 3] = 255;
  }
  return out;
}

function packAlbedo(rgb, size) {
  const out = new Uint8Array(size * size * 4);
  for (let i = 0, n = size * size; i < n; i++) {
    out[i * 4]     = clamp01(rgb[i * 3])     * 255;
    out[i * 4 + 1] = clamp01(rgb[i * 3 + 1]) * 255;
    out[i * 4 + 2] = clamp01(rgb[i * 3 + 2]) * 255;
    out[i * 4 + 3] = 255;
  }
  return out;
}

/* ---------------------------------------------------------------- palette */
// Values are linear-ish sRGB reference albedos. Real-world materials almost
// never sit below 0.03 or above 0.85 luminance; staying inside that band is a
// large part of why a render reads as photographic rather than synthetic.

const MIX = (a, b, t) => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

/* --------------------------------------------------------------- builders */

/**
 * Each builder fills a height field plus per-texel albedo/roughness/metalness,
 * then the shared tail converts height into normal + AO. Adding a material
 * means writing one function of (x, y) -> surface properties.
 */
function build(size, fn, { normalStrength = 2.2, aoRadius = 3, aoStrength = 1 } = {}) {
  const n = size * size;
  const height = new Float32Array(n);
  const rgb = new Float32Array(n * 3);
  const rough = new Float32Array(n);
  const metal = new Float32Array(n);
  const s = { height, rgb, rough, metal };

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      fn(x / size, y / size, (y * size + x), s);
    }
  }
  const normal = heightToNormal(height, size, normalStrength);
  const ao = heightToAO(height, size, aoRadius, aoStrength);
  return {
    albedo: packAlbedo(rgb, size),
    normal,
    orm: packORM(ao, rough, metal, size),
    size,
  };
}

export const MATERIALS = {

  /* Poured concrete: aggregate speckle, form-tie pocks, hairline cracks,
     and long vertical water staining under the panel joints. */
  concrete(size) {
    return build(size, (u, v, i, s) => {
      const grain = fbm(u * 24, v * 24, 5, 2, 0.5, 24);
      const agg   = fbm(u * 90, v * 90, 3, 2, 0.5, 90);
      const cell  = voronoi(u, v, 14);
      const pit   = smoothstep(0.14, 0.0, cell.f1) * (cell.id > 0.72 ? 1 : 0);
      // Cracks follow a narrow band *around* a ridge level, not everything above
      // it — without the abs() the mask floods half the surface with blotches.
      const crack = smoothstep(0.031, 0.0, Math.abs(ridged(u * 6, v * 6, 4, 6) - 0.82));

      let h = 0.5 + grain * 0.22 + agg * 0.10 - pit * 0.42 - crack * 0.30;
      s.height[i] = h;

      // Stretched along v: rain runs down the panel, so the streak must too, so
      // the two axes need separate periods to wrap. Upper edge sits at the
      // field's measured ceiling (~0.71 here) — the old 1.0 put it past what fbm
      // can reach, throttling the streaks to a third of their intended depth.
      const stain = smoothstep(0.55, 0.74, fbm(u * 11, v * 3, 4, 2, 0.5, 11, 3) * 0.5 + 0.5);
      const base = MIX([0.44, 0.435, 0.425], [0.30, 0.298, 0.292], stain * 0.75);
      const c = MIX(base, [0.22, 0.22, 0.225], pit * 0.6 + crack * 0.5);
      const spec = agg * 0.06;
      s.rgb[i * 3]     = clamp01(c[0] + spec);
      s.rgb[i * 3 + 1] = clamp01(c[1] + spec);
      s.rgb[i * 3 + 2] = clamp01(c[2] + spec);

      s.rough[i] = clamp01(0.82 + grain * 0.14 - stain * 0.10 + pit * 0.08);
      s.metal[i] = 0;
    }, { normalStrength: 2.6, aoRadius: 3, aoStrength: 1.1 });
  },

  /* Painted steel over a rusting substrate — the paint chips where the
     underlying corrosion has lifted it, which is the tell that sells metal. */
  paintedMetal(size, tint = [0.30, 0.34, 0.38]) {
    return build(size, (u, v, i, s) => {
      const dent  = fbm(u * 7, v * 7, 4, 2, 0.5, 7) * 0.5 + 0.5;
      const rustF = smoothstep(0.52, 0.86, fbm(u * 9, v * 9, 5, 2, 0.5, 9) * 0.5 + 0.5);
      const chip  = smoothstep(0.60, 0.78, fbm(u * 34, v * 34, 4, 2, 0.5, 34) * 0.5 + 0.5) * rustF;
      // A 3-octave fbm only reaches ~0.85 once remapped to 0..1, so the old 0.86
      // threshold clipped this mask to zero everywhere — with metalness now
      // driven by it, the bare-steel scratches have to actually appear.
      const scratch = smoothstep(0.655, 0.76, fbm(u * 140, v * 6, 3, 2, 0.5, 140, 6) * 0.5 + 0.5);

      s.height[i] = 0.5 + dent * 0.08 - chip * 0.16 - scratch * 0.05;

      const rustCol = MIX([0.36, 0.17, 0.075], [0.52, 0.28, 0.13], fbm(u * 40, v * 40, 3, 2, 0.5, 40) * 0.5 + 0.5);
      let c = MIX(tint, rustCol, chip);
      c = MIX(c, [0.55, 0.56, 0.58], scratch * 0.5);
      s.rgb[i * 3] = c[0]; s.rgb[i * 3 + 1] = c[1]; s.rgb[i * 3 + 2] = c[2];

      s.rough[i] = clamp01(0.45 + chip * 0.50 + dent * 0.10 - scratch * 0.18);
      // Paint and rust are both dielectrics — only bare steel is metallic, so a
      // metallic base would strip the diffuse response and leave these surfaces
      // reflecting nothing but the blue upper hemisphere of the environment.
      s.metal[i] = clamp01(scratch * 0.7 + chip * 0.12);
    }, { normalStrength: 1.8, aoRadius: 2, aoStrength: 0.9 });
  },

  /* Running-bond brick with recessed mortar. */
  brick(size) {
    // Over the 2 m brick tile: 8.3 cm courses and 22 cm faces. ROWS has to stay
    // even, or the running-bond offset butts two identically aligned courses
    // together where the tile wraps in v.
    const ROWS = 24, COLS = 9;
    const BRICK_H = 1 / ROWS, BRICK_W = 1 / COLS, MORTAR = 0.0045;
    return build(size, (u, v, i, s) => {
      const row = Math.floor(v / BRICK_H);
      const offset = (row & 1) ? BRICK_W * 0.5 : 0;
      const bu = ((u + offset) % BRICK_W) / BRICK_W;
      const bv = (v % BRICK_H) / BRICK_H;
      const mortarMask = 1 - smoothstep(0, MORTAR / BRICK_W, Math.min(bu, 1 - bu)) *
                             smoothstep(0, MORTAR / BRICK_H * 0.6, Math.min(bv, 1 - bv));
      // Column index wraps at COLS so the half-brick courses, whose offset
      // straddles u = 0, keep one shade across the tile seam.
      const col = Math.floor((u + offset) / BRICK_W) % COLS;
      const id = ((row * 31 + col * 17) % 97) / 97;
      const grit = fbm(u * 110, v * 110, 3, 2, 0.5, 110) * 0.5 + 0.5;
      const wear = fbm(u * 20, v * 20, 4, 2, 0.5, 20) * 0.5 + 0.5;

      s.height[i] = (1 - mortarMask) * 0.55 + grit * 0.06 + wear * 0.05;

      const brickCol = MIX([0.34, 0.15, 0.11], [0.50, 0.26, 0.19], id * 0.8 + wear * 0.2);
      const mortarCol = MIX([0.52, 0.51, 0.48], [0.40, 0.395, 0.375], grit);
      const c = MIX(brickCol, mortarCol, mortarMask);
      s.rgb[i * 3] = c[0]; s.rgb[i * 3 + 1] = c[1]; s.rgb[i * 3 + 2] = c[2];

      s.rough[i] = clamp01(0.78 + mortarMask * 0.14 + grit * 0.08);
      s.metal[i] = 0;
    }, { normalStrength: 3.0, aoRadius: 4, aoStrength: 1.35 });
  },

  /* Wind-rippled sand with a coarse shell/pebble layer. */
  sand(size) {
    return build(size, (u, v, i, s) => {
      // The dune band is 1.5 m across inside a 6 m tile, so at any real viewing
      // distance it is the feature the eye recognises as the repeat — keep it as
      // a faint tonal drift only. Matching `per` to the frequency also stops the
      // band from stepping at the tile edge.
      const dune   = fbm(u * 4, v * 4, 4, 2, 0.5, 4);
      // Crest heading wanders instead of running dead straight along world Z.
      // The turn is applied as a phase offset rather than a rotated uv because
      // only whole cycle counts in u and v survive the wrap; 45/10 keeps the
      // wave-vector length (and so the 13 cm pitch) within 0.2% of the old 46.
      const bend   = fbm(u * 2, v * 2, 3, 2, 0.5, 2) * 8;
      const ripple = Math.sin((u * 45 + v * 10 + bend) * Math.PI * 2) * 0.5 + 0.5;
      const grit   = fbm(u * 170, v * 170, 2, 2, 0.5, 170) * 0.5 + 0.5;
      // Thresholds bracket the top of this field's measured range (~0.77 peak);
      // the old 0.80..0.92 window sat entirely above it, so the pebble layer
      // never contributed a single texel of height, colour or roughness.
      const peb    = smoothstep(0.63, 0.74, fbm(u * 60, v * 60, 3, 2, 0.5, 60) * 0.5 + 0.5);

      s.height[i] = 0.5 + dune * 0.04 + ripple * 0.055 + grit * 0.03 + peb * 0.10;

      const c = MIX(MIX([0.52, 0.44, 0.32], [0.63, 0.55, 0.41], ripple * 0.62 + dune * 0.20),
                    [0.40, 0.37, 0.32], peb);
      const sparkle = grit * 0.05;
      s.rgb[i * 3] = c[0] + sparkle; s.rgb[i * 3 + 1] = c[1] + sparkle; s.rgb[i * 3 + 2] = c[2] + sparkle;

      s.rough[i] = clamp01(0.90 - peb * 0.22 + grit * 0.06);
      s.metal[i] = 0;
    }, { normalStrength: 1.5, aoRadius: 2, aoStrength: 0.6 });
  },

  /* Asphalt with aggregate, tar seams and polished wheel tracks. */
  asphalt(size) {
    return build(size, (u, v, i, s) => {
      const agg  = fbm(u * 130, v * 130, 3, 2, 0.5, 130) * 0.5 + 0.5;
      const cell = voronoi(u, v, 26);
      const chunk = smoothstep(0.30, 0.06, cell.f1);
      const crack = smoothstep(0.05, 0.0, Math.abs(ridged(u * 5, v * 5, 4, 5) - 0.55));
      // Two polished bands where tyres have burnished the aggregate smooth.
      const track = Math.max(smoothstep(0.09, 0.0, Math.abs(u - 0.30)),
                             smoothstep(0.09, 0.0, Math.abs(u - 0.70)));

      s.height[i] = 0.5 + agg * 0.09 + chunk * 0.10 - crack * 0.34 - track * 0.05;

      const c = MIX(MIX([0.075, 0.075, 0.080], [0.155, 0.155, 0.162], agg),
                    [0.20, 0.20, 0.205], chunk * 0.5);
      const t = MIX(c, [0.13, 0.13, 0.135], track * 0.6);
      s.rgb[i * 3] = t[0]; s.rgb[i * 3 + 1] = t[1]; s.rgb[i * 3 + 2] = t[2];

      s.rough[i] = clamp01(0.86 - track * 0.34 + agg * 0.08 + crack * 0.06);
      s.metal[i] = 0;
    }, { normalStrength: 2.2, aoRadius: 3, aoStrength: 1.0 });
  },

  /* Weathered plank timber. */
  wood(size) {
    return build(size, (u, v, i, s) => {
      const PLANKS = 7, pw = 1 / PLANKS;
      const idx = Math.floor(u / pw);
      const pu = (u % pw) / pw;
      const seam = 1 - smoothstep(0, 0.055, Math.min(pu, 1 - pu));
      const shift = ((idx * 37) % 61) / 61;

      // Rings: a stretched noise field pushed through fract() to make bands.
      // The ring count over v has to be a whole number — fract() of 5.5 cycles
      // leaves a half-band step across the tile edge that reads as a hard line
      // through every plank. The warp period must be integral for the same
      // reason, so the vertical stretch is 2 rather than 2.2.
      const warp = fbm(u * 8, v * 2, 4, 2, 0.5, 8, 2) * 1.6;
      // JS % keeps the sign of its operand, and warp is signed, so near v = 0 the
      // phase goes negative — which both breaks the wrap on the first plank and
      // lets rings run past 1 into the height and roughness terms.
      const phase = (v * 6 + shift * 4 + warp) % 1;
      const rings = Math.abs((phase < 0 ? phase + 1 : phase) - 0.5) * 2;
      const fiber = fbm(u * 200, v * 14, 3, 2, 0.5, 200, 14) * 0.5 + 0.5;
      // Knots vary per plank because each plank samples its own stretch of the
      // field; the old per-plank phase offset only added a wrap discontinuity on
      // top of that. Window tracks the measured ceiling — at 0.86 it was dead.
      const knot  = smoothstep(0.66, 0.72, fbm(u * 6, v * 6, 3, 2, 0.5, 6) * 0.5 + 0.5);

      s.height[i] = 0.5 + rings * 0.11 + fiber * 0.05 - seam * 0.55 - knot * 0.12;

      const c0 = MIX([0.235, 0.155, 0.090], [0.395, 0.285, 0.180], rings);
      const c1 = MIX(c0, [0.14, 0.09, 0.05], knot);
      const c  = MIX(c1, [0.07, 0.05, 0.035], seam);
      const g = fiber * 0.045;
      s.rgb[i * 3] = c[0] + g; s.rgb[i * 3 + 1] = c[1] + g; s.rgb[i * 3 + 2] = c[2] + g;

      s.rough[i] = clamp01(0.72 + rings * 0.16 + knot * 0.10);
      s.metal[i] = 0;
    }, { normalStrength: 2.0, aoRadius: 3, aoStrength: 1.1 });
  },

  /* Corrugated sheet — the profile is geometric, the decay is not. */
  corrugated(size) {
    return build(size, (u, v, i, s) => {
      const wave = Math.sin(u * Math.PI * 2 * 16) * 0.5 + 0.5;
      const rust = smoothstep(0.45, 0.85, fbm(u * 7, v * 7, 5, 2, 0.5, 7) * 0.5 + 0.5);
      // v * 2 rather than 2.5: a fractional period cannot wrap, and the streaks
      // want the longer vertical run anyway.
      const streak = smoothstep(0.5, 1.0, fbm(u * 12, v * 2, 4, 2, 0.5, 12, 2) * 0.5 + 0.5) * rust;
      const dent = fbm(u * 18, v * 18, 3, 2, 0.5, 18) * 0.5 + 0.5;

      s.height[i] = wave * 0.72 + dent * 0.06 - rust * 0.05;

      const steel = MIX([0.42, 0.44, 0.46], [0.30, 0.32, 0.34], dent);
      const rusty = MIX([0.40, 0.19, 0.08], [0.28, 0.13, 0.06], streak);
      const c = MIX(steel, rusty, rust);
      s.rgb[i * 3] = c[0]; s.rgb[i * 3 + 1] = c[1]; s.rgb[i * 3 + 2] = c[2];

      s.rough[i] = clamp01(0.38 + rust * 0.48 + dent * 0.08);
      s.metal[i] = clamp01(0.92 - rust * 0.80);
    }, { normalStrength: 2.4, aoRadius: 3, aoStrength: 0.9 });
  },

  /* Ballistic nylon / webbing for gear and enemy kit. */
  fabric(size, tint = [0.20, 0.21, 0.17]) {
    return build(size, (u, v, i, s) => {
      // Over-under weave: two offset square waves multiplied together.
      const wu = Math.sin(u * Math.PI * 2 * 130);
      const wv = Math.sin(v * Math.PI * 2 * 130);
      const weave = (wu * wv) * 0.5 + 0.5;
      const fuzz = fbm(u * 220, v * 220, 2, 2, 0.5, 220) * 0.5 + 0.5;
      const wear = smoothstep(0.6, 0.95, fbm(u * 11, v * 11, 4, 2, 0.5, 11) * 0.5 + 0.5);

      s.height[i] = 0.5 + weave * 0.30 + fuzz * 0.06;

      const c = MIX(tint, MIX(tint, [0.55, 0.54, 0.50], 0.5), wear * 0.6 + weave * 0.2);
      s.rgb[i * 3] = c[0]; s.rgb[i * 3 + 1] = c[1]; s.rgb[i * 3 + 2] = c[2];
      s.rough[i] = clamp01(0.88 - weave * 0.10 + fuzz * 0.08);
      s.metal[i] = 0;
    }, { normalStrength: 1.2, aoRadius: 2, aoStrength: 0.8 });
  },

  /* Gunmetal — parkerised finish with edge polish from handling. */
  gunmetal(size) {
    return build(size, (u, v, i, s) => {
      const tool = fbm(u * 260, v * 8, 3, 2, 0.5, 260, 8) * 0.5 + 0.5;   // machining marks
      const grain = fbm(u * 60, v * 60, 4, 2, 0.5, 60) * 0.5 + 0.5;
      const wear = smoothstep(0.66, 0.94, fbm(u * 14, v * 14, 4, 2, 0.5, 14) * 0.5 + 0.5);

      s.height[i] = 0.5 + grain * 0.05 + tool * 0.03 - wear * 0.02;

      const base = MIX([0.085, 0.088, 0.094], [0.135, 0.138, 0.145], grain);
      const c = MIX(base, [0.46, 0.47, 0.49], wear);   // rubbed-through steel
      s.rgb[i * 3] = c[0]; s.rgb[i * 3 + 1] = c[1]; s.rgb[i * 3 + 2] = c[2];

      s.rough[i] = clamp01(0.52 - wear * 0.34 + grain * 0.10 + tool * 0.05);
      s.metal[i] = 1;
    }, { normalStrength: 1.0, aoRadius: 2, aoStrength: 0.7 });
  },
};

/* ---------------------------------------------------------------- library */

/**
 * Builds the whole material set, yielding to the event loop between entries so
 * the loading bar keeps painting on a phone instead of locking the main thread.
 */
export class TextureLibrary {
  constructor(size = 512, anisotropy = 8) {
    this.size = size;
    this.aniso = anisotropy;
    this.cache = new Map();
  }

  /** @returns {{map:THREE.Texture, normalMap:THREE.Texture, ormMap:THREE.Texture}} */
  get(name) { return this.cache.get(name); }

  _store(name, res, repeat) {
    const entry = {
      map:       makeTexture(res.albedo, res.size, { srgb: true, repeat, aniso: this.aniso }),
      normalMap: makeTexture(res.normal, res.size, { repeat, aniso: this.aniso }),
      ormMap:    makeTexture(res.orm,    res.size, { repeat, aniso: this.aniso }),
    };
    this.cache.set(name, entry);
    return entry;
  }

  async buildAll(onProgress) {
    const jobs = [
      ['concrete',   () => MATERIALS.concrete(this.size),                        1],
      ['brick',      () => MATERIALS.brick(this.size),                           1],
      ['sand',       () => MATERIALS.sand(this.size),                            1],
      ['asphalt',    () => MATERIALS.asphalt(this.size),                         1],
      ['wood',       () => MATERIALS.wood(this.size),                            1],
      ['corrugated', () => MATERIALS.corrugated(this.size),                      1],
      ['metal',      () => MATERIALS.paintedMetal(this.size),                    1],
      ['metalOlive', () => MATERIALS.paintedMetal(this.size, [0.20, 0.22, 0.15]),1],
      ['fabric',     () => MATERIALS.fabric(this.size),                          1],
      ['gunmetal',   () => MATERIALS.gunmetal(Math.min(this.size, 512)),         1],
    ];
    for (let i = 0; i < jobs.length; i++) {
      const [name, fn, repeat] = jobs[i];
      this._store(name, fn(), repeat);
      onProgress?.((i + 1) / jobs.length, name);
      await new Promise(r => setTimeout(r, 0));
    }
    return this;
  }

  dispose() {
    for (const e of this.cache.values()) {
      e.map.dispose(); e.normalMap.dispose(); e.ormMap.dispose();
    }
    this.cache.clear();
  }
}

/**
 * Builds a MeshStandardMaterial wired to a library entry.
 *
 * three reads aoMap from .r, roughnessMap from .g and metalnessMap from .b, so
 * the single packed ORM texture serves all three slots. aoMap defaults to UV
 * set 1, which our geometry does not have — forcing channel 0 avoids having to
 * duplicate the uv attribute on every mesh in the level.
 *
 * @param {TextureLibrary} lib
 * @param {string} name        key in the library
 * @param {object} [opts]      repeat, plus any MeshStandardMaterial override
 */
export function makeStandardMaterial(lib, name, opts = {}) {
  const entry = lib.get(name);
  if (!entry) throw new Error(`Unknown material "${name}"`);
  const { repeat = 1, repeatY, ...rest } = opts;

  // Each mesh usually wants its own tiling rate, so the shared library
  // textures are cloned per-material; clones share the GPU upload.
  const map = entry.map.clone();
  const normalMap = entry.normalMap.clone();
  const ormMap = entry.ormMap.clone();
  const ry = repeatY ?? repeat;
  for (const t of [map, normalMap, ormMap]) {
    t.repeat.set(repeat, ry);
    t.needsUpdate = true;
  }
  ormMap.channel = 0;

  return new THREE.MeshStandardMaterial({
    map, normalMap,
    aoMap: ormMap, roughnessMap: ormMap, metalnessMap: ormMap,
    roughness: 1, metalness: 1,       // scaled by the packed channels
    normalScale: new THREE.Vector2(1, 1),
    envMapIntensity: 1.0,
    ...rest,
  });
}

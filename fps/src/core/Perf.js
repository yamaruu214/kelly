// Perf.js — device capability detection, quality tiers, adaptive resolution.
// Everything downstream reads its budgets from the active tier so the game
// degrades gracefully from desktop dGPU down to an iPhone SE in Safari.

export const TIER = { LOW: 0, MED: 1, HIGH: 2, ULTRA: 3 };

const PRESETS = {
  [TIER.LOW]: {
    name: 'LOW',
    renderScale: 0.62, maxPixelRatio: 1.5,
    shadows: true, shadowMapSize: 1024, shadowCascades: 1, softShadows: false,
    bloom: true, bloomIterations: 4,
    ssao: false, motionBlur: false, chromatic: false, grain: true,
    aaMode: 'fxaa',
    maxDecals: 48, maxParticles: 320, maxCorpses: 3,
    anisotropy: 4, textureSize: 512,
    enemyCount: 7, drawDistance: 190, foliage: false, volumetrics: false,
  },
  [TIER.MED]: {
    name: 'MEDIUM',
    renderScale: 0.78, maxPixelRatio: 2.0,
    shadows: true, shadowMapSize: 1536, shadowCascades: 2, softShadows: true,
    bloom: true, bloomIterations: 5,
    ssao: true, motionBlur: false, chromatic: true, grain: true,
    aaMode: 'fxaa',
    maxDecals: 96, maxParticles: 700, maxCorpses: 5,
    anisotropy: 8, textureSize: 1024,
    enemyCount: 10, drawDistance: 280, foliage: true, volumetrics: true,
  },
  [TIER.HIGH]: {
    name: 'HIGH',
    renderScale: 1.0, maxPixelRatio: 2.0,
    shadows: true, shadowMapSize: 2048, shadowCascades: 3, softShadows: true,
    bloom: true, bloomIterations: 6,
    ssao: true, motionBlur: true, chromatic: true, grain: true,
    aaMode: 'taa',
    maxDecals: 160, maxParticles: 1400, maxCorpses: 8,
    anisotropy: 16, textureSize: 1024,
    enemyCount: 14, drawDistance: 420, foliage: true, volumetrics: true,
  },
  [TIER.ULTRA]: {
    name: 'ULTRA',
    renderScale: 1.0, maxPixelRatio: 2.0,
    shadows: true, shadowMapSize: 3072, shadowCascades: 4, softShadows: true,
    bloom: true, bloomIterations: 7,
    ssao: true, motionBlur: true, chromatic: true, grain: true,
    aaMode: 'taa',
    maxDecals: 256, maxParticles: 2400, maxCorpses: 12,
    anisotropy: 16, textureSize: 2048,
    enemyCount: 18, drawDistance: 600, foliage: true, volumetrics: true,
  },
};

export const device = (() => {
  const ua = navigator.userAgent;
  const touch = matchMedia('(pointer: coarse)').matches || navigator.maxTouchPoints > 0;
  // iPadOS 13+ reports as Macintosh; the touch-point count is the giveaway.
  const iOS = /iPad|iPhone|iPod/.test(ua) ||
              (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  const android = /Android/.test(ua);
  const safari = /^((?!chrome|android|crios|fxios).)*safari/i.test(ua);
  return {
    ua, touch, iOS, android, safari,
    mobile: touch && (iOS || android || Math.min(screen.width, screen.height) < 820),
    cores: navigator.hardwareConcurrency || 4,
    memory: navigator.deviceMemory || (iOS ? 4 : 8),
    dpr: window.devicePixelRatio || 1,
  };
})();

/** Reads the unmasked GPU string when the driver exposes it. */
function gpuName(gl) {
  try {
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    if (ext) return String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) || '');
    return String(gl.getParameter(gl.RENDERER) || '');
  } catch { return ''; }
}

export function detectTier(gl) {
  const g = gpuName(gl).toLowerCase();
  const { mobile, iOS, cores, memory } = device;

  if (mobile) {
    // Apple's GPU generations are legible from the renderer string on iOS.
    const m = g.match(/apple a(\d+)/);
    const gen = m ? +m[1] : 0;
    if (iOS) {
      if (/apple (m\d|a1[7-9]|a2\d)/.test(g) || gen >= 15) return TIER.HIGH;
      if (gen >= 12 || /apple a1[2-4]/.test(g)) return TIER.MED;
      if (gen > 0) return TIER.LOW;
      // Unknown Apple GPU: infer from core count. A11 and earlier ship 2 cores.
      return cores >= 6 ? TIER.MED : TIER.LOW;
    }
    if (/adreno (7|8)\d\d/.test(g) || /mali-g7\d\d/.test(g)) return TIER.HIGH;
    if (/adreno (6)\d\d/.test(g) || cores >= 8) return TIER.MED;
    return TIER.LOW;
  }

  if (/rtx|radeon rx (6|7|9)|apple m[1-9]|arc a7/.test(g)) return TIER.ULTRA;
  if (/gtx 1[06]|radeon rx (5)|iris xe/.test(g)) return TIER.HIGH;
  if (cores >= 8 && memory >= 8) return TIER.HIGH;
  return TIER.MED;
}

export function presetFor(tier) { return { ...PRESETS[tier] }; }

/**
 * Rolling frame-time monitor that nudges render scale to hold a target FPS.
 * Scale moves in small steps and only after a sustained trend, so the image
 * never visibly pulses during a brief spike (a grenade, a spawn burst).
 */
export class AdaptiveResolution {
  constructor(settings, { target = 60, min = 0.5, max = 1.0 } = {}) {
    this.settings = settings;
    this.targetMs = 1000 / target;
    this.min = min; this.max = max;
    this.scale = settings.renderScale;
    this.samples = []; this.cooldown = 0; this.onChange = null;
    this.fps = target;
  }
  update(dtMs) {
    this.samples.push(dtMs);
    if (this.samples.length > 60) this.samples.shift();
    if (this.cooldown > 0) { this.cooldown--; return; }
    if (this.samples.length < 45) return;

    // Median is immune to the odd GC pause that would otherwise drag the mean.
    const sorted = [...this.samples].sort((a, b) => a - b);
    const med = sorted[sorted.length >> 1];
    this.fps = 1000 / med;

    const prev = this.scale;
    if (med > this.targetMs * 1.28) this.scale = Math.max(this.min, this.scale - 0.08);
    else if (med < this.targetMs * 0.82) this.scale = Math.min(this.max, this.scale + 0.05);

    if (Math.abs(this.scale - prev) > 0.001) {
      this.cooldown = 45;
      this.samples.length = 0;
      this.onChange?.(this.scale);
    }
  }
}

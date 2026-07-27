// Sky.js — analytic atmosphere, procedural cloud deck, sun rig and IBL.
//
// The sky is the single largest contributor to how expensive the scene reads,
// so it is a real single-scattering model (Rayleigh + Mie, Preetham-style
// closed form) rather than a gradient. Closed form matters: a ray-marched sky
// at 60fps on an iPhone would eat the entire frame budget, while this is a
// handful of transcendentals per pixel and only runs on background fragments.
//
// Everything the rest of the game needs off this file — key light, fill light,
// fog, environment map — is derived from the same sun direction and the same
// shader, so the lighting can never drift out of agreement with the backdrop.

import * as THREE from 'three';

const D2R = Math.PI / 180;

/* Golden hour. What reaches flat ground is the key times sin(elevation), so 15°
   delivered 0.26 of it and left sunlit sand a stop and a half under while the
   sky above it clipped. 26° gives 0.44, still rakes hard (a caster throws two of
   its own heights) and is still low enough for a warm, reddened key. */
const SUN_ELEVATION = 26.0;
const SUN_AZIMUTH = 118.0;

/* ~0.2°/min. A ten-minute session loses two degrees, which nobody consciously
   notices, but it stops the scene from reading as a still frame. */
const SUN_DRIFT_PER_SEC = -0.0034;

/* Rebuilding the IBL is a PMREM pass; at this threshold it happens roughly
   once every seven minutes instead of every frame. */
const ENV_REBUILD_DEG = 1.5;

/* Cascades folded into one map, indexed by the tier's cascade allowance. A
   second shadow-casting light would re-draw every caster and add a second PCF
   fetch to every lit fragment, which MEDIUM cannot pay for, so the allowance is
   spent as area on a single ortho instead. Each entry is paired with its tier's
   shadowMapSize to hold the texel footprint near 7cm everywhere (78/1024,
   152/2048, 196/3072), which is what lets the bias and the texel snapping below
   stay valid as the span grows. HIGH and up now reach across the whole 140m map;
   at 76m the far half of every wide shot had no contact shadows at all. */
const SHADOW_SPAN_BY_CASCADES = { 1: 78, 2: 112, 3: 152, 4: 196 };

/* -------------------------------------------------------------- shader src */

const SKY_VERT = /* glsl */`
varying vec3 vRay;

void main(){
  vRay = position;
  #ifdef SKY_INFINITE
    // Translation stripped so the dome is welded to the camera, then z forced
    // to w so the fragment lands exactly on the far plane. That is what lets
    // camera.far stay at 190 on a phone without the sky being clipped away.
    vec4 clip = projectionMatrix * mat4(mat3(modelViewMatrix)) * vec4(position, 1.0);
    gl_Position = clip.xyww;
  #else
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  #endif
}
`;

const SKY_FRAG = /* glsl */`
varying vec3 vRay;

uniform vec3  uSunDir;
uniform float uTime;
uniform float uRayleigh, uTurbidity, uMieCoeff, uMieG;
uniform float uSkyScale, uSunDisc, uSunRadius;
uniform vec3  uGroundColor, uSunTint, uZenithTint;
uniform float uCloudCover, uCloudSoft, uCloudScale, uCloudHeight;
uniform float uCloudOpacity, uCloudWind, uSilver;

const float PI = 3.141592653589793;

// Rayleigh cross-sections at 680/550/440nm and the Mie constant term
// pi * (2pi/lambda)^2 * K, both at sea level. Baking them avoids three pow()
// calls per pixel for values that never change.
const vec3  BETA_R = vec3(5.804542996261093e-6, 1.3562911419845635e-5, 3.0265902468824876e-5);
const vec3  MIE_K  = vec3(1.8399918514433978e14, 2.7798023919660528e14, 4.0790479543861094e14);

// Effective atmosphere thickness at the zenith, in metres.
const float R_DEPTH = 8.4e3;
const float M_DEPTH = 1.25e3;

const float SUN_E  = 1000.0;
const float CUTOFF = 1.6110731556870734;   // zenith angle where the sun sets
const float STEEP  = 1.5;

const vec3 UP = vec3(0.0, 1.0, 0.0);

float rayleighPhase(float c){ return (3.0 / (16.0 * PI)) * (1.0 + c * c); }

float hg(float c, float g){
  float g2 = g * g;
  return (1.0 / (4.0 * PI)) * ((1.0 - g2) / pow(max(1e-4, 1.0 - 2.0 * g * c + g2), 1.5));
}

/* --------------------------------------------------------------- fbm noise */

float hash21(vec2 p){
  // Multiply-fract hash rather than the usual sin() one: sin loses all its
  // entropy on mobile GPUs once the argument gets large, and the cloud plane
  // coordinates near the horizon get very large indeed.
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Rotating each octave stops the lattice from producing the axis-aligned
// cross-hatch that gives cheap procedural clouds away instantly.
const mat2 OCT = mat2(0.80, 0.60, -0.60, 0.80);

float fbm(vec2 p){
  float sum = 0.0, amp = 0.5, norm = 0.0;
  for (int i = 0; i < CLOUD_OCTAVES; i++){
    sum += amp * vnoise(p);
    norm += amp;
    p = OCT * p * 2.03;
    amp *= 0.5;
  }
  return sum / norm;
}

/* ---------------------------------------------------------- the atmosphere */

/**
 * Single-scattering radiance for a view ray.
 * fex returns the extinction along that ray and sunE the sun's irradiance,
 * both of which the cloud and sun-disc code reuse instead of recomputing.
 */
vec3 atmosphere(vec3 rd, vec3 sd, out vec3 fex, out float sunE){
  vec3 betaR = BETA_R * uRayleigh;
  vec3 betaM = (0.434 * (0.2 * uTurbidity * 1e-17) * MIE_K) * uMieCoeff;

  // Optical depth via the standard secant approximation, which stays finite at
  // the horizon where a naive 1/cos(theta) would blow up.
  float zenith = acos(max(0.0, rd.y));
  float denom = cos(zenith) + 0.15 * pow(max(1e-3, 93.885 - zenith * (180.0 / PI)), -1.253);
  fex = exp(-(betaR * (R_DEPTH / denom) + betaM * (M_DEPTH / denom)));

  sunE = SUN_E * max(0.0, 1.0 - exp(-((CUTOFF - acos(clamp(sd.y, -1.0, 1.0))) / STEEP)));

  float cosT = dot(rd, sd);
  // Remapping cosT into [0,1] before the Rayleigh phase is Preetham's fudge for
  // the missing multiple-scattering term; without it the sky opposite the sun
  // comes out far too dark.
  vec3 betaRt = betaR * rayleighPhase(cosT * 0.5 + 0.5);
  vec3 betaMt = betaM * hg(cosT, uMieG);
  vec3 ratio = (betaRt + betaMt) / (betaR + betaM);

  vec3 lin = pow(sunE * ratio * (1.0 - fex), vec3(1.5));
  // Near a low sun the in-scattered term needs the extinction folded back in,
  // otherwise the horizon saturates to a flat orange band with no gradient.
  lin *= mix(vec3(1.0), pow(sunE * ratio * fex, vec3(0.5)),
             clamp(pow(1.0 - dot(UP, sd), 5.0), 0.0, 1.0));

  return lin * uSkyScale;
}

vec3 sunDisc(vec3 rd, vec3 sd, vec3 fex, float sunE){
  float ang = acos(clamp(dot(rd, sd), -1.0, 1.0));
  float r = ang / uSunRadius;
  if (r > 1.02) return vec3(0.0);

  // Limb darkening: the photosphere is optically thicker toward the edge, so
  // the rim is dimmer and redder. Per-channel exponents give that for free and
  // are the difference between a sun and a white circle.
  float mu = sqrt(max(0.0, 1.0 - min(r, 1.0) * min(r, 1.0)));
  vec3 limb = vec3(1.0) - 0.85 * (vec3(1.0) - pow(vec3(mu), vec3(0.397, 0.503, 0.652)));

  float edge = 1.0 - smoothstep(0.90, 1.01, r);
  return fex * limb * (edge * sunE * uSunDisc * uSkyScale);
}

/* -------------------------------------------------------------- cloud deck */

vec4 cloudLayer(vec3 rd, vec3 sd, vec3 skyCol, float sunE){
  if (rd.y < 0.012) return vec4(0.0);

  // Intersect a flat deck. Clamping the distance keeps the noise coordinates
  // bounded near the horizon (where t explodes) and doubles as the compressed
  // "deck runs out" look real cloud layers have at grazing angles.
  float t = min(uCloudHeight / rd.y, 42000.0);
  vec2 p = rd.xz * t * uCloudScale + vec2(uTime * uCloudWind, uTime * uCloudWind * 0.37);

  #ifdef CLOUD_WARP
    // Domain warp. Feeding fbm back into its own coordinates is what turns
    // round blobs into the sheared, billowing silhouettes of real cumulus.
    vec2 q = vec2(fbm(p + vec2(1.7, 9.2)), fbm(p + vec2(8.3, 2.8)));
    float dens = fbm(p + 2.2 * q);
  #else
    float dens = fbm(p * 1.15 + 3.7);
  #endif

  float cover = smoothstep(uCloudCover, uCloudCover + uCloudSoft, dens);
  if (cover < 0.002) return vec4(0.0);

  // One tap toward the sun stands in for a light march: where density falls off
  // sunward the cloud is facing the light, where it rises the cloud is behind
  // its own bulk. That single difference gives lit tops and shadowed bases.
  vec2 sunStep = normalize(sd.xz + vec2(1e-4)) * 0.6;
  float densL = fbm(p + sunStep);
  float lit = clamp((dens - densL) * 3.2 + 0.55, 0.0, 1.0);

  // Powder term — multiple scattering brightens thin edges relative to cores.
  float powder = 1.0 - exp(-cover * 3.4);

  float cosT = dot(rd, sd);
  // Peaks where the deck is thin *and* the sun is behind it: the silver lining.
  float silver = hg(cosT, 0.80) * cover * (1.0 - cover) * uSilver;

  vec3 sunLight = uSunTint * (sunE * uSkyScale * 0.18);
  vec3 ambient = mix(skyCol * 0.9, uZenithTint, 0.35);

  vec3 col = ambient * mix(0.30, 0.95, lit)
           + sunLight * (lit * powder * 0.55 + 0.05)
           + sunLight * min(silver, 4.0);

  // Aerial perspective on the deck itself, so distant clouds sit behind the
  // same haze as distant geometry instead of floating in front of it.
  col = mix(col, skyCol * 1.05, clamp(t / 42000.0, 0.0, 1.0) * 0.8);

  float alpha = cover * uCloudOpacity * smoothstep(0.015, 0.17, rd.y);
  return vec4(col, alpha);
}

/* --------------------------------------------------------------------- main */

void main(){
  vec3 rd = normalize(vRay);
  vec3 sd = normalize(uSunDir);

  vec3 fex; float sunE;
  vec3 col = atmosphere(rd, sd, fex, sunE);

  vec4 cl = cloudLayer(rd, sd, col, sunE);
  // Disc first, then clouds over it, so a cloud crossing the sun occludes it.
  col += sunDisc(rd, sd, fex, sunE);
  col = mix(col, cl.rgb, cl.a);

  // The analytic model just mirrors the horizon band below y=0; fade to a dark
  // scattered-ground tone so the dome never shows an upside-down sunset.
  col = mix(col, uGroundColor * (0.25 + 0.75 * dot(col, vec3(0.33))), smoothstep(0.0, -0.10, rd.y));

  col += vec3(0.0005, 0.0009, 0.0016);   // never let the sky reach true black

  // Triangular-PDF dither. The gradient from horizon to zenith is the classic
  // 8-bit banding offender and it survives all the way through the post chain,
  // so it has to be broken up here, at the source, before quantisation.
  float d1 = hash21(gl_FragCoord.xy);
  float d2 = hash21(gl_FragCoord.yx + 17.3);
  col += (d1 + d2 - 1.0) * (0.0018 + 0.004 * dot(col, vec3(0.2126, 0.7152, 0.0722)));

  // Half-float render targets overflow at 65504 and the bloom bright-pass would
  // carry an Inf straight through the whole pyramid.
  gl_FragColor = vec4(min(max(col, vec3(0.0)), vec3(220.0)), 1.0);
}
`;

const SHAFT_VERT = /* glsl */`
varying vec2 vUv;
void main(){
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const SHAFT_FRAG = /* glsl */`
varying vec2 vUv;
uniform vec3  uColor;
uniform float uTime, uIntensity;

void main(){
  vec2 d = vUv - 0.5;
  float r = length(d) * 2.0;
  if (r > 1.0) discard;

  float a = atan(d.y, d.x);
  // Three incommensurate harmonics read as irregular shafts; a single one
  // would look like a starburst decal.
  float shafts = 0.50
               + 0.30 * sin(a *  6.0 + uTime * 0.13)
               + 0.22 * sin(a * 11.0 - uTime * 0.09)
               + 0.16 * sin(a * 19.0 + uTime * 0.05);
  shafts = pow(clamp(shafts, 0.0, 1.4), 2.6);

  float radial = exp(-r * 2.3) * smoothstep(1.0, 0.12, r);
  float glow = exp(-r * 4.5);

  // Feathered to zero at the quad edge — otherwise the billboard's silhouette
  // is visible as a square of haze the moment the sun is off-centre.
  float mask = smoothstep(1.0, 0.55, r);

  vec3 col = uColor * (glow * 0.55 + shafts * radial * 0.85) * uIntensity * mask;
  gl_FragColor = vec4(col, 1.0);
}
`;

/* Only patched once per page; re-running it on an already-patched chunk would
   nest the mix and wash the fog out. */
let fogChunkPatched = false;

function patchAerialFog(warmColor, coolColor, sunDir, farDistance) {
  if (fogChunkPatched) return;
  fogChunkPatched = true;
  const c = v => v.toFixed(5);

  /* The far tint has to know which way the fragment is being viewed from, and
     nothing in the stock fragment prefix locates the fragment. Carrying the ray
     down as a varying costs one interpolator; reconstructing world position from
     depth would cost a matrix per fogged fragment on every tier. */
  THREE.ShaderChunk.fog_pars_vertex = /* glsl */`
#ifdef USE_FOG
  varying float vFogDepth;
  varying vec3 vFogRay;
#endif
`;
  THREE.ShaderChunk.fog_vertex = /* glsl */`
#ifdef USE_FOG
  vFogDepth = - mvPosition.z;
  // The camera sits at the origin in view space, so mvPosition is already the
  // camera-to-fragment direction. Right-multiplying by the view rotation applies
  // its transpose, which for a camera basis is its inverse — no inverse() call,
  // which also keeps this compiling under GLSL ES 1.00.
  vFogRay = mvPosition.xyz * mat3( viewMatrix );
#endif
`;
  THREE.ShaderChunk.fog_pars_fragment = /* glsl */`
#ifdef USE_FOG
  uniform vec3 fogColor;
  varying float vFogDepth;
  varying vec3 vFogRay;
  #ifdef FOG_EXP2
    uniform float fogDensity;
  #else
    uniform float fogNear;
    uniform float fogFar;
  #endif
#endif
`;
  /* The sun direction is baked rather than passed: the fog uniform block is
     merged into every built-in material at three's module init, so a new uniform
     added afterwards would be declared but never uploaded. Elevation drifts ~2°
     across a session, which moves this dot product by 0.03. */
  THREE.ShaderChunk.fog_fragment = /* glsl */`
#ifdef USE_FOG
  #ifdef FOG_EXP2
    float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
  #else
    float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
  #endif
  // Aerial perspective is the sky seen through the air the geometry stands in,
  // so it has to swing warm looking into the sun and cool looking away from it.
  // One view-independent far colour is what welds a hue seam along the horizon:
  // blue-hazed sand meeting a warm sky along a hard line.
  float fogSun = smoothstep( -0.35, 0.85, dot( normalize( vFogRay ),
                             vec3( ${c(sunDir.x)}, ${c(sunDir.y)}, ${c(sunDir.z)} ) ) );
  vec3 fogHaze = mix( vec3( ${c(coolColor.r)}, ${c(coolColor.g)}, ${c(coolColor.b)} ),
                      vec3( ${c(warmColor.r)}, ${c(warmColor.g)}, ${c(warmColor.b)} ), fogSun );
  vec3 fogTint = mix( fogColor, fogHaze, smoothstep( 0.0, ${c(farDistance)}, vFogDepth ) );
  gl_FragColor.rgb = mix( gl_FragColor.rgb, fogTint, fogFactor );
#endif
`;
}

export class Sky {
  constructor(scene, renderer, settings) {
    this.scene = scene;
    this.renderer = renderer;
    this.settings = settings;

    this.elevation = SUN_ELEVATION;
    this.azimuth = SUN_AZIMUTH;
    this.sunDirection = new THREE.Vector3();
    this._applySunAngles();

    this.envMap = null;
    this.time = 0;
    this._envElevation = this.elevation;

    /* ------------------------------------------------------------- lights */

    // Warm key / cool fill. The split is what stops shadowed faces from going
    // to dead black and is doing more for the "expensive" read than any
    // individual effect in the post chain.
    // 3.0 rather than 3.2 because the elevation change already multiplied what
    // lands on the ground plane by 1.7x, and sun-facing walls took none of that
    // increase — they would be the first thing to clip if the key went up too.
    this.sunLight = new THREE.DirectionalLight(0xffd9a8, 3.0);
    this.sunLight.castShadow = !!settings.shadows;
    scene.add(this.sunLight);
    scene.add(this.sunLight.target);

    // Metres covered by the ortho frustum, from the tier's cascade allowance.
    this._shadowSpan = SHADOW_SPAN_BY_CASCADES[settings.shadowCascades] || 112;
    this._sunDistance = 165;
    this._configureShadow();

    // Desaturated because scene.environment already supplies sky ambient from
    // the IBL and the two stack; a saturated blue here turns every shadowed
    // surface visibly cyan. The level is one term though, not two — at 0.26 the
    // shadows sat below 4/255 and the frame had no midtones between them and
    // the sunlit sand.
    this.fillLight = new THREE.HemisphereLight(0x9db4c6, 0x4a4238, 0.45);
    scene.add(this.fillLight);

    // Light-space basis, used to snap the shadow frustum to whole texels.
    this._sunRight = new THREE.Vector3();
    this._sunUp = new THREE.Vector3();
    this._rebuildSunBasis();

    this._tmpA = new THREE.Vector3();
    this._tmpB = new THREE.Vector3();
  }

  _applySunAngles() {
    const el = this.elevation * D2R, az = this.azimuth * D2R;
    this.sunDirection.set(
      Math.sin(az) * Math.cos(el),
      Math.sin(el),
      Math.cos(az) * Math.cos(el)).normalize();
  }

  _rebuildSunBasis() {
    this._sunRight.crossVectors(this.sunDirection, new THREE.Vector3(0, 1, 0)).normalize();
    this._sunUp.crossVectors(this._sunRight, this.sunDirection).normalize();
  }

  _configureShadow() {
    const s = this.sunLight.shadow;
    const size = this.settings.shadowMapSize || 1024;
    s.mapSize.set(size, size);

    const h = this._shadowSpan * 0.5;
    s.camera.left = -h; s.camera.right = h;
    s.camera.top = h; s.camera.bottom = -h;
    // The depth range has to swallow the box's own spread along the light as well
    // as the casters: at 26° the far edge of a 196m box sits ~100m up- or
    // down-sun of the centre, so a tight near plane would clip the casters
    // nearest the light out of the map entirely.
    s.camera.near = 15;
    s.camera.far = this._sunDistance * 2.1;
    s.camera.updateProjectionMatrix();

    // Grazing light is the worst case for acne, and normalBias has to be sized
    // in world units against the texel footprint or it either does nothing at
    // 3072 or peter-pans everything at 1024.
    const texel = this._shadowSpan / size;
    // Constant bias is depth-range-relative, and this ortho spans ~330m: -0.0005
    // was 16cm of push, which detached every wall foot and pole base from its
    // own shadow. Keep it small enough to be invisible and let normalBias, which
    // is a world-space offset along the normal and so cannot peter-pan a contact
    // point, carry the acne suppression instead.
    s.bias = -0.00008;
    s.normalBias = texel * 0.7;
    s.radius = this.settings.softShadows ? 2.5 : 1.0;
    s.blurSamples = this.settings.softShadows ? 8 : 4;
  }

  /* ------------------------------------------------------------------ build */

  async build() {
    const { renderer, scene, settings } = this;
    const low = settings.name === 'LOW';
    const med = settings.name === 'MEDIUM';

    this.uniforms = {
      uSunDir: { value: this.sunDirection },
      uTime: { value: 0 },
      uRayleigh: { value: 2.4 },
      uTurbidity: { value: 5.2 },
      uMieCoeff: { value: 0.0045 },
      uMieG: { value: 0.80 },
      // Pulled down from 0.042 to hold the dome's absolute radiance across the
      // elevation change: sunE rises by half again between 15° and 26°, and the
      // sky was already clipping. The exposure lift the post chain applies is
      // meant to land on the ground plane, not on a sky that has no headroom.
      uSkyScale: { value: 0.030 },
      // 0.27° is the real solar disc; at 0.9° it covered eleven times the solid
      // angle and bloom smeared it over a third of the sky. Doubling the radiance
      // keeps the peak reading as the sun while the total flux into the bright
      // -pass drops, which is the whole point.
      uSunDisc: { value: 26.0 },
      uSunRadius: { value: 0.27 * D2R },
      uGroundColor: { value: new THREE.Color(0x2b2721) },
      uSunTint: { value: new THREE.Color(1.0, 0.84, 0.62) },
      uZenithTint: { value: new THREE.Color(0.32, 0.44, 0.62) },
      uCloudCover: { value: 0.50 },
      // Tightened with the octave count: more octaves raise the fbm normaliser
      // and so flatten the low-frequency contrast the cover threshold cuts
      // against, which would have made the extra detail read as more fog rather
      // than as harder cloud tops.
      uCloudSoft: { value: 0.20 },
      uCloudScale: { value: 0.0011 },
      uCloudHeight: { value: 1800 },
      uCloudOpacity: { value: 0.94 },
      uCloudWind: { value: 0.010 },
      uSilver: { value: 3.0 },
    };

    /* Three octaves is a blob with nothing inside it. The deck only ever shades
       background fragments and the warp path already costs three fbm calls, so
       the extra taps buy internal structure and crisp tops for a rounding error
       on anything with a desktop GPU; MEDIUM is held at four because it covers
       mid-range phones, and LOW pays for none of it. */
    const defines = {
      CLOUD_OCTAVES: low ? 2 : (med ? 4 : 5),
      ...(low ? {} : { CLOUD_WARP: 1 }),
    };

    const common = {
      vertexShader: SKY_VERT,
      fragmentShader: SKY_FRAG,
      uniforms: this.uniforms,      // shared by reference across both materials
      side: THREE.BackSide,
      depthWrite: false,
      fog: false,
    };

    this.material = new THREE.ShaderMaterial({
      ...common,
      defines: { ...defines, SKY_INFINITE: 1 },
    });

    // A unit box, not a sphere: 12 triangles, no pole pinching, and the object
    // -space vertex position is already the view ray.
    this._geometry = new THREE.BoxGeometry(2, 2, 2);
    this.mesh = new THREE.Mesh(this._geometry, this.material);
    this.mesh.frustumCulled = false;
    // Drawn last among opaques so it only shades pixels the level did not
    // already cover — on a phone that is most of the sky's cost recovered.
    this.mesh.renderOrder = 1000;
    scene.add(this.mesh);

    /* Offscreen twin used for the IBL capture and the fog colour probe. It
       needs real projection, so it drops the SKY_INFINITE path. */
    this._envMaterial = new THREE.ShaderMaterial({ ...common, defines: { ...defines } });
    this._envScene = new THREE.Scene();
    this._envMesh = new THREE.Mesh(this._geometry, this._envMaterial);
    this._envMesh.scale.setScalar(20);
    this._envMesh.frustumCulled = false;
    this._envScene.add(this._envMesh);

    const prevTarget = renderer.getRenderTarget();

    /* ------------------------------------------------- fog matched to sky */

    const horizonWarm = this._probeSky(this.sunDirection.x, this.sunDirection.z, 0.03);
    const horizonCross = this._probeSky(-this.sunDirection.z, this.sunDirection.x, 0.03);
    const horizonAway = this._probeSky(-this.sunDirection.x, -this.sunDirection.z, 0.03);

    // Near haze is heading-independent — it is the air a few metres out — so it
    // is the mean of all three probes.
    const near = horizonWarm.clone().lerp(horizonCross, 0.5)
      .lerp(horizonAway, 1 / 3).multiplyScalar(1.05);

    // The two ends the far tint swings between, each taken from the sky the
    // horizon actually meets in that direction. Only the anti-sun end is pushed
    // toward Rayleigh blue and only slightly: 45% of the way to a saturated blue
    // was enough to turn yellow sand into olive-teal mud.
    const warmFar = horizonWarm.clone().multiplyScalar(1.04);
    const coolFar = horizonAway.clone().lerp(new THREE.Color(0.30, 0.44, 0.64), 0.18);

    // 0.0042: at 0.0062 the far half of the map was three-quarters haze, so the
    // sand's own hue was gone long before the horizon and any mismatch in the
    // haze tint became the entire colour of the distance.
    scene.fog = new THREE.FogExp2(0x000000, 0.0042);
    scene.fog.color.copy(near);
    this.fog = scene.fog;
    patchAerialFog(warmFar, coolFar, this.sunDirection,
                   Math.min(settings.drawDistance, 260) * 0.8);

    this.horizonColor = near;

    /* --------------------------------------------------------------- IBL */

    this._cubeRT = new THREE.WebGLCubeRenderTarget(low || med ? 128 : 256, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      generateMipmaps: false,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    });
    this._cubeCam = new THREE.CubeCamera(0.5, 200, this._cubeRT);
    this._pmrem = new THREE.PMREMGenerator(renderer);
    this._pmrem.compileCubemapShader();

    this._renderEnvironment();

    renderer.setRenderTarget(prevTarget);

    /* -------------------------------------------------------- god rays */

    if (settings.volumetrics) this._buildShafts();

    /* Three hands the active camera to the scene here, which is the only way
       to reach it under the update(dt, elapsed) contract. Chained rather than
       assigned so whoever else wants the hook still gets it. */
    const prevHook = scene.onBeforeRender;
    scene.onBeforeRender = (r, s, camera, rt) => {
      prevHook?.call(scene, r, s, camera, rt);
      if (camera?.isPerspectiveCamera) this._trackCamera(camera);
    };

    scene.environment = this.envMap;
    // The IBL is the only thing lighting a surface that faces neither the key nor
    // much of the upper hemisphere. At 0.6 those surfaces were black holes.
    scene.environmentIntensity = 0.9;
  }

  /**
   * Averages a small render of the sky looking at the horizon along (dx, dz).
   * RGBA8 is deliberate — half-float readback is not portable, and fog colour
   * lives in [0,1] anyway.
   */
  _probeSky(dx, dz, y) {
    const fallback = new THREE.Color(0.42, 0.46, 0.52);
    try {
      if (!this._probeRT) {
        this._probeRT = new THREE.WebGLRenderTarget(16, 16, {
          type: THREE.UnsignedByteType, format: THREE.RGBAFormat,
          depthBuffer: false, stencilBuffer: false, generateMipmaps: false,
        });
        this._probeRT.texture.colorSpace = THREE.NoColorSpace;
        this._probeCam = new THREE.PerspectiveCamera(75, 1, 0.1, 120);
        this._probeBuf = new Uint8Array(16 * 16 * 4);
      }

      // The disc would blow out one sample and skew the average by 100x.
      const disc = this.uniforms.uSunDisc.value;
      this.uniforms.uSunDisc.value = 0;

      this._probeCam.position.set(0, 0, 0);
      this._probeCam.lookAt(dx, y, dz);
      this._probeCam.updateMatrixWorld(true);

      this.renderer.setRenderTarget(this._probeRT);
      this.renderer.clear();
      this.renderer.render(this._envScene, this._probeCam);
      this.renderer.readRenderTargetPixels(this._probeRT, 0, 0, 16, 16, this._probeBuf);
      this.uniforms.uSunDisc.value = disc;

      let r = 0, g = 0, b = 0;
      for (let i = 0; i < this._probeBuf.length; i += 4) {
        r += this._probeBuf[i]; g += this._probeBuf[i + 1]; b += this._probeBuf[i + 2];
      }
      const n = (this._probeBuf.length / 4) * 255;
      const c = new THREE.Color(r / n, g / n, b / n);
      return (c.r + c.g + c.b) > 0.02 ? c : fallback;
    } catch {
      return fallback;
    }
  }

  _renderEnvironment() {
    const r = this.renderer;
    const prev = r.getRenderTarget();

    // At 0.27° the disc no longer fills even one texel of a 128px cube face, so
    // whatever it lands on is a single sample several hundred times its
    // neighbours' brightness — exactly the input PMREM's box-filtered mips turn
    // into a crawling firefly. The directional light carries the sun's energy
    // anyway; the capture only needs enough of it for a plausible specular.
    const disc = this.uniforms.uSunDisc.value;
    this.uniforms.uSunDisc.value = disc * 0.10;
    this._cubeCam.update(r, this._envScene);
    this.uniforms.uSunDisc.value = disc;

    const rt = this._pmrem.fromCubemap(this._cubeRT.texture);
    this._envRT?.dispose();
    this._envRT = rt;
    this.envMap = rt.texture;
    this.scene.environment = this.envMap;
    this._envElevation = this.elevation;

    r.setRenderTarget(prev);
  }

  _buildShafts() {
    this._shaftMaterial = new THREE.ShaderMaterial({
      vertexShader: SHAFT_VERT,
      fragmentShader: SHAFT_FRAG,
      uniforms: {
        uColor: { value: new THREE.Color(1.0, 0.78, 0.50) },
        uTime: { value: 0 },
        // Additive, so it scales with exposure for free; 0.30 was set against a
        // frame the post chain now renders most of a stop brighter.
        uIntensity: { value: 0.18 },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      // Depth-tested but not depth-writing: the level occludes the billboard
      // per-pixel, which is real occlusion for the cost of one quad. Nothing
      // else here would give shafts that break behind a wall.
      depthTest: true,
      depthWrite: false,
      fog: false,
    });
    this._shaftGeo = new THREE.PlaneGeometry(1, 1);
    this.shafts = new THREE.Mesh(this._shaftGeo, this._shaftMaterial);
    this.shafts.frustumCulled = false;
    this.shafts.renderOrder = 1100;
    this.scene.add(this.shafts);
  }

  /* --------------------------------------------------------------- runtime */

  _trackCamera(camera) {
    const centre = this._tmpA.copy(camera.position);
    centre.y = 0;

    // Push the box toward where the player is looking; a frustum centred on the
    // player spends half its texels on shadows behind them. Capped in metres
    // rather than left proportional, because once the span is wide enough to
    // reach the far side of the map the lead stops buying coverage and starts
    // throwing away the ground under the player's feet.
    camera.getWorldDirection(this._tmpB);
    this._tmpB.y = 0;
    if (this._tmpB.lengthSq() > 1e-6) {
      const lead = Math.min(this._shadowSpan * 0.22, 30);
      centre.addScaledVector(this._tmpB.normalize(), lead);
    }

    // Snap the centre to whole shadow texels along the light's own axes. Without
    // this the depth samples slide under static geometry as the player walks and
    // every shadow edge crawls — the single most visible shadow artefact there is.
    const texel = this._shadowSpan / (this.settings.shadowMapSize || 1024);
    const a = Math.round(centre.dot(this._sunRight) / texel) * texel;
    const b = Math.round(centre.dot(this._sunUp) / texel) * texel;
    const c = centre.dot(this.sunDirection);
    centre.set(0, 0, 0)
      .addScaledVector(this._sunRight, a)
      .addScaledVector(this._sunUp, b)
      .addScaledVector(this.sunDirection, c);

    this.sunLight.target.position.copy(centre);
    this.sunLight.position.copy(centre).addScaledVector(this.sunDirection, this._sunDistance);
    this.sunLight.target.updateMatrixWorld();
    this.sunLight.updateMatrixWorld();

    if (this.shafts) {
      // Far enough out that level geometry sits in front of it, near enough to
      // stay inside camera.far on the tightest preset that enables volumetrics.
      const d = Math.min(camera.far * 0.78, 240);
      this.shafts.position.copy(camera.position).addScaledVector(this.sunDirection, d);
      this.shafts.quaternion.copy(camera.quaternion);
      this.shafts.scale.setScalar(d * 1.5);
      this.shafts.updateMatrixWorld();
    }
  }

  update(dt, elapsed) {
    this.time += dt;
    if (this.uniforms) this.uniforms.uTime.value = this.time;
    if (this._shaftMaterial) this._shaftMaterial.uniforms.uTime.value = this.time;

    if (SUN_DRIFT_PER_SEC !== 0) {
      this.elevation = Math.max(6, this.elevation + SUN_DRIFT_PER_SEC * dt);
      this._applySunAngles();

      if (this._pmrem && Math.abs(this.elevation - this._envElevation) > ENV_REBUILD_DEG) {
        this._rebuildSunBasis();
        this._renderEnvironment();
      }
    }
  }

  dispose() {
    this._geometry?.dispose();
    this._shaftGeo?.dispose();
    this.material?.dispose();
    this._envMaterial?.dispose();
    this._shaftMaterial?.dispose();
    this._cubeRT?.dispose();
    this._envRT?.dispose();
    this._probeRT?.dispose();
    this._pmrem?.dispose();
    this.scene.remove(this.mesh, this.sunLight, this.sunLight.target, this.fillLight);
    if (this.shafts) this.scene.remove(this.shafts);
  }
}

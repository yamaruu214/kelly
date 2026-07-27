// PostFX.js — hand-rolled post chain tuned for mobile WebGL2.
//
// Order: scene -> SSAO(applied to lighting) -> bright-pass -> bloom pyramid ->
//        composite(tonemap + bloom + CA + vignette + grain) -> FXAA -> screen.
//
// Three's EffectComposer would work, but each of its passes allocates a
// full-res target; on a phone the bandwidth is the whole budget. This chain
// shares two half-res ping-pong targets across every blur step instead.

import * as THREE from 'three';

const FULLSCREEN_VERT = /* glsl */`
varying vec2 vUv;
void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

/* ACES filmic, Narkowicz fit. Cheap, and the shoulder is what stops muzzle
   flashes and sky from clipping to flat white the way Reinhard does. */
const ACES = /* glsl */`
vec3 aces(vec3 x){
  const float a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
  return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}
`;

class Pass {
  constructor(fragmentShader, uniforms) {
    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: `precision highp float;\nin vec3 position;\nin vec2 uv;\nout vec2 vUv;\nvoid main(){ vUv=uv; gl_Position=vec4(position.xy,0.0,1.0); }`,
      fragmentShader: `precision highp float;\nprecision highp sampler2D;\nin vec2 vUv;\nout vec4 fragColor;\n${fragmentShader}`,
      uniforms,
      depthTest: false, depthWrite: false,
    });
  }
  dispose() { this.material.dispose(); }
}

export class PostFX {
  constructor(renderer, scene, camera, settings) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.settings = settings;
    this.enabled = true;

    this.time = 0;
    this.exposure = 1.0;
    this.flashAmount = 0;      // white-out on nearby explosions
    this.damageAmount = 0;     // red edge pulse when hit
    this.adsAmount = 0;        // drives depth-of-field strength when aiming

    this._quad = new THREE.Mesh(new THREE.BufferGeometry(), null);
    // A single oversized triangle beats a quad: no diagonal seam, one less
    // vertex, and the GPU never shades the same pixel twice at the split.
    const g = this._quad.geometry;
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0, 0, 2, 0, 0, 2]), 2));
    this._quad.frustumCulled = false;

    this._orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    this._buildTargets(1, 1);
    this._buildPasses();
  }

  _makeRT(w, h, { depth = false, type = THREE.HalfFloatType } = {}) {
    const rt = new THREE.WebGLRenderTarget(Math.max(1, w | 0), Math.max(1, h | 0), {
      type,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: depth,
      stencilBuffer: false,
      generateMipmaps: false,
    });
    rt.texture.colorSpace = THREE.NoColorSpace;
    if (depth) {
      // Reading depth back for SSAO/DOF needs an explicit depth texture.
      rt.depthTexture = new THREE.DepthTexture(rt.width, rt.height);
      rt.depthTexture.type = THREE.UnsignedIntType;
    }
    return rt;
  }

  _buildTargets(w, h) {
    this.width = w; this.height = h;
    this.sceneRT = this._makeRT(w, h, { depth: true });

    // Bloom pyramid: successive halves, reused for both down and up passes.
    const levels = Math.max(2, Math.min(this.settings.bloomIterations || 5, 8));
    this.mips = [];
    let mw = w >> 1, mh = h >> 1;
    for (let i = 0; i < levels && mw > 4 && mh > 4; i++) {
      this.mips.push({ rt: this._makeRT(mw, mh), w: mw, h: mh });
      mw >>= 1; mh >>= 1;
    }
    this.aoRT = this.settings.ssao ? this._makeRT(w >> 1, h >> 1, { type: THREE.UnsignedByteType }) : null;
    this.aoBlurRT = this.settings.ssao ? this._makeRT(w >> 1, h >> 1, { type: THREE.UnsignedByteType }) : null;
    this.compositeRT = this._makeRT(w, h, { type: THREE.UnsignedByteType });
  }

  _disposeTargets() {
    this.sceneRT?.dispose();
    this.mips?.forEach(m => m.rt.dispose());
    this.aoRT?.dispose(); this.aoBlurRT?.dispose();
    this.compositeRT?.dispose();
  }

  _buildPasses() {
    const U = THREE.UniformsUtils;

    /* ---- SSAO: hemisphere sampling against the depth buffer ---- */
    this.aoPass = new Pass(/* glsl */`
      uniform sampler2D tDepth;
      uniform vec2  uRes;
      uniform mat4  uProjInv;
      uniform mat4  uProj;
      uniform float uRadius, uBias, uIntensity, uNear, uFar;

      vec3 viewPos(vec2 uv){
        float d = texture(tDepth, uv).r;
        vec4 c = uProjInv * vec4(uv*2.0-1.0, d*2.0-1.0, 1.0);
        return c.xyz / c.w;
      }
      float hash(vec2 p){ return fract(sin(dot(p, vec2(41.3, 289.1))) * 43758.5453); }

      void main(){
        vec3 p = viewPos(vUv);
        // Sky writes no depth, so those texels sit at the far plane. Reject
        // relative to uFar — a fixed cutoff would miss them entirely on mobile,
        // where the far plane is only 190 units out.
        if (-p.z > uFar * 0.97) { fragColor = vec4(1.0); return; }

        // Reconstruct the normal from depth derivatives — no normal buffer.
        vec3 dx = dFdx(p), dy = dFdy(p);
        vec3 n = normalize(cross(dx, dy));

        float ang = hash(gl_FragCoord.xy) * 6.2831853;
        float occ = 0.0;
        const int SAMPLES = 12;
        for (int i = 0; i < SAMPLES; i++){
          float fi = float(i);
          // Golden-angle spiral: even coverage without a sample-kernel texture.
          float r = uRadius * sqrt((fi + 0.5) / float(SAMPLES));
          float a = ang + fi * 2.39996;
          vec3 dir = vec3(cos(a), sin(a), 0.0);
          dir = normalize(dir + n * 0.7);
          vec3 sp = p + dir * r;

          vec4 clip = uProj * vec4(sp, 1.0);
          vec2 suv = (clip.xy / clip.w) * 0.5 + 0.5;
          if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) continue;

          float sampleDepth = viewPos(suv).z;
          float diff = sampleDepth - sp.z;
          // Range check stops distant geometry haloing over near surfaces.
          float rangeCheck = smoothstep(0.0, 1.0, uRadius / max(0.0001, abs(p.z - sampleDepth)));
          occ += step(uBias, diff) * rangeCheck;
        }
        fragColor = vec4(vec3(clamp(1.0 - (occ / float(SAMPLES)) * uIntensity, 0.0, 1.0)), 1.0);
      }
    `, {
      tDepth: { value: null }, uRes: { value: new THREE.Vector2() },
      uProjInv: { value: new THREE.Matrix4() }, uProj: { value: new THREE.Matrix4() },
      uRadius: { value: 0.55 }, uBias: { value: 0.025 }, uIntensity: { value: 1.15 },
      uNear: { value: 0.1 }, uFar: { value: 500 },
    });

    /* ---- cross bilateral blur for the AO buffer ---- */
    this.aoBlurPass = new Pass(/* glsl */`
      uniform sampler2D tAO; uniform vec2 uDir; uniform vec2 uRes;
      void main(){
        vec2 texel = uDir / uRes;
        float sum = 0.0, wsum = 0.0;
        for (int i = -3; i <= 3; i++){
          float w = exp(-float(i*i) * 0.18);
          sum += texture(tAO, vUv + texel * float(i)).r * w;
          wsum += w;
        }
        fragColor = vec4(vec3(sum / wsum), 1.0);
      }
    `, { tAO: { value: null }, uDir: { value: new THREE.Vector2(1, 0) }, uRes: { value: new THREE.Vector2() } });

    /* ---- bright pass with soft knee ---- */
    this.brightPass = new Pass(/* glsl */`
      uniform sampler2D tScene; uniform float uThreshold, uKnee, uExposure;
      void main(){
        vec3 c = texture(tScene, vUv).rgb * uExposure;
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        // Soft knee: quadratic ramp through the threshold so bloom fades in
        // rather than popping on as a pixel crosses the cut.
        float soft = clamp(l - uThreshold + uKnee, 0.0, 2.0 * uKnee);
        soft = soft * soft / (4.0 * uKnee + 0.0001);
        float contrib = max(soft, l - uThreshold) / max(l, 0.0001);
        fragColor = vec4(c * contrib, 1.0);
      }
    `, { tScene: { value: null }, uThreshold: { value: 1.05 }, uKnee: { value: 0.6 }, uExposure: { value: 1 } });

    /* ---- 13-tap downsample (Jimenez / COD Advanced Warfare filter) ---- */
    this.downPass = new Pass(/* glsl */`
      uniform sampler2D tSrc; uniform vec2 uTexel;
      void main(){
        vec2 t = uTexel;
        vec3 a = texture(tSrc, vUv + vec2(-2,-2)*t).rgb;
        vec3 b = texture(tSrc, vUv + vec2( 0,-2)*t).rgb;
        vec3 c = texture(tSrc, vUv + vec2( 2,-2)*t).rgb;
        vec3 d = texture(tSrc, vUv + vec2(-1,-1)*t).rgb;
        vec3 e = texture(tSrc, vUv + vec2( 1,-1)*t).rgb;
        vec3 f = texture(tSrc, vUv + vec2(-2, 0)*t).rgb;
        vec3 g = texture(tSrc, vUv               ).rgb;
        vec3 h = texture(tSrc, vUv + vec2( 2, 0)*t).rgb;
        vec3 i = texture(tSrc, vUv + vec2(-1, 1)*t).rgb;
        vec3 j = texture(tSrc, vUv + vec2( 1, 1)*t).rgb;
        vec3 k = texture(tSrc, vUv + vec2(-2, 2)*t).rgb;
        vec3 l = texture(tSrc, vUv + vec2( 0, 2)*t).rgb;
        vec3 m = texture(tSrc, vUv + vec2( 2, 2)*t).rgb;
        vec3 res = (d+e+i+j) * 0.125
                 + (a+b+g+f) * 0.03125
                 + (b+c+h+g) * 0.03125
                 + (f+g+l+k) * 0.03125
                 + (g+h+m+l) * 0.03125
                 + g * 0.125;
        fragColor = vec4(res, 1.0);
      }
    `, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() } });

    /* ---- 3x3 tent upsample, additively blended into the larger mip ---- */
    this.upPass = new Pass(/* glsl */`
      uniform sampler2D tSrc; uniform vec2 uTexel; uniform float uRadius;
      void main(){
        vec2 t = uTexel * uRadius;
        vec3 s = texture(tSrc, vUv + vec2(-1,-1)*t).rgb * 1.0
               + texture(tSrc, vUv + vec2( 0,-1)*t).rgb * 2.0
               + texture(tSrc, vUv + vec2( 1,-1)*t).rgb * 1.0
               + texture(tSrc, vUv + vec2(-1, 0)*t).rgb * 2.0
               + texture(tSrc, vUv               ).rgb * 4.0
               + texture(tSrc, vUv + vec2( 1, 0)*t).rgb * 2.0
               + texture(tSrc, vUv + vec2(-1, 1)*t).rgb * 1.0
               + texture(tSrc, vUv + vec2( 0, 1)*t).rgb * 2.0
               + texture(tSrc, vUv + vec2( 1, 1)*t).rgb * 1.0;
        fragColor = vec4(s / 16.0, 1.0);
      }
    `, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() }, uRadius: { value: 1.0 } });

    /* ---- composite ---- */
    this.compositePass = new Pass(/* glsl */`
      ${ACES}
      uniform sampler2D tScene, tBloom, tAO, tDepth;
      uniform vec2  uRes;
      uniform float uExposure, uBloomStrength, uTime;
      uniform float uVignette, uGrain, uChroma, uSaturation, uContrast;
      uniform float uFlash, uDamage, uAO;
      uniform float uDofStrength, uDofFocus, uNear, uFar;
      uniform int   uUseAO;

      float linearDepth(float d){
        float z = d * 2.0 - 1.0;
        return (2.0 * uNear * uFar) / (uFar + uNear - z * (uFar - uNear));
      }

      void main(){
        vec2 uv = vUv;
        vec2 fromCenter = uv - 0.5;
        float r2 = dot(fromCenter, fromCenter);

        // Lateral chromatic aberration: sample the channels along slightly
        // different radii. Scales with r^2 so the centre stays clean.
        vec3 col;
        if (uChroma > 0.0){
          float amt = uChroma * r2;
          col.r = texture(tScene, uv - fromCenter * amt * 1.0).r;
          col.g = texture(tScene, uv).g;
          col.b = texture(tScene, uv + fromCenter * amt * 1.0).b;
        } else {
          col = texture(tScene, uv).rgb;
        }

        // Depth of field: a cheap 6-tap ring, only meaningful while aiming.
        // The 0.6m near guard keeps the viewmodel out of the blur; it shares
        // this buffer and would otherwise defocus completely while aiming.
        if (uDofStrength > 0.001 && linearDepth(texture(tDepth, uv).r) > 0.6){
          float d = linearDepth(texture(tDepth, uv).r);
          float coc = clamp(abs(d - uDofFocus) / max(uDofFocus, 1.0), 0.0, 1.0) * uDofStrength;
          if (coc > 0.004){
            vec3 acc = col; float w = 1.0;
            for (int i = 0; i < 6; i++){
              float a = float(i) * 1.0472;
              vec2 o = vec2(cos(a), sin(a)) * coc * 0.02;
              acc += texture(tScene, uv + o).rgb; w += 1.0;
            }
            col = acc / w;
          }
        }

        if (uUseAO == 1){
          float ao = texture(tAO, uv).r;
          col *= mix(1.0, ao, uAO);
        }

        col *= uExposure;
        col += texture(tBloom, uv).rgb * uBloomStrength;
        col += uFlash;

        col = aces(col);

        // Grade in display space: saturation then contrast around 0.5 pivot.
        float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
        col = mix(vec3(lum), col, uSaturation);
        col = clamp((col - 0.5) * uContrast + 0.5, 0.0, 1.0);

        // Slight teal push in shadows, warm in highlights — filmic split tone.
        col = mix(col * vec3(0.94, 0.99, 1.06), col * vec3(1.05, 1.00, 0.94), smoothstep(0.25, 0.85, lum));

        col *= smoothstep(uVignette, uVignette - 0.55, r2);

        if (uDamage > 0.001){
          float edge = smoothstep(0.02, 0.18, r2);
          col = mix(col, vec3(0.42, 0.02, 0.02), edge * uDamage);
        }

        // Animated film grain, scaled down in highlights the way real grain is.
        if (uGrain > 0.0){
          float n = fract(sin(dot(uv * uRes + uTime * 91.7, vec2(12.9898, 78.233))) * 43758.5453);
          col += (n - 0.5) * uGrain * (1.0 - smoothstep(0.35, 1.0, lum));
        }

        fragColor = vec4(col, 1.0);
      }
    `, {
      tScene: { value: null }, tBloom: { value: null }, tAO: { value: null }, tDepth: { value: null },
      uRes: { value: new THREE.Vector2() },
      uExposure: { value: 1.0 }, uBloomStrength: { value: 0.55 }, uTime: { value: 0 },
      uVignette: { value: 0.62 }, uGrain: { value: 0.035 }, uChroma: { value: 0.0022 },
      uSaturation: { value: 1.06 }, uContrast: { value: 1.05 },
      uFlash: { value: 0 }, uDamage: { value: 0 }, uAO: { value: 0.85 },
      uDofStrength: { value: 0 }, uDofFocus: { value: 12 }, uNear: { value: 0.1 }, uFar: { value: 500 },
      uUseAO: { value: 0 },
    });

    /* ---- FXAA 3.11 (console quality preset) ---- */
    this.fxaaPass = new Pass(/* glsl */`
      uniform sampler2D tSrc; uniform vec2 uTexel;
      float luma(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }
      void main(){
        vec3 rgbM = texture(tSrc, vUv).rgb;
        float lM  = luma(rgbM);
        float lNW = luma(texture(tSrc, vUv + vec2(-1,-1)*uTexel).rgb);
        float lNE = luma(texture(tSrc, vUv + vec2( 1,-1)*uTexel).rgb);
        float lSW = luma(texture(tSrc, vUv + vec2(-1, 1)*uTexel).rgb);
        float lSE = luma(texture(tSrc, vUv + vec2( 1, 1)*uTexel).rgb);

        float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
        float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
        float range = lMax - lMin;
        // Flat areas are left untouched; blurring them just softens texture.
        if (range < max(0.0312, lMax * 0.125)) { fragColor = vec4(rgbM, 1.0); return; }

        vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)), ((lNW + lSW) - (lNE + lSE)));
        float reduce = max((lNW + lNE + lSW + lSE) * 0.03125, 0.0078125);
        float rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
        dir = clamp(dir * rcp, -8.0, 8.0) * uTexel;

        vec3 rgbA = 0.5 * (texture(tSrc, vUv + dir * (1.0/3.0 - 0.5)).rgb +
                           texture(tSrc, vUv + dir * (2.0/3.0 - 0.5)).rgb);
        vec3 rgbB = rgbA * 0.5 + 0.25 * (texture(tSrc, vUv - dir * 0.5).rgb +
                                         texture(tSrc, vUv + dir * 0.5).rgb);
        float lB = luma(rgbB);
        fragColor = vec4((lB < lMin || lB > lMax) ? rgbA : rgbB, 1.0);
      }
    `, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() } });
  }

  setSize(w, h) {
    if (w === this.width && h === this.height) return;
    this._disposeTargets();
    this._buildTargets(w, h);
  }

  _blit(pass, target) {
    this._quad.material = pass.material;
    this.renderer.setRenderTarget(target);
    this.renderer.render(this._quad, this._orthoCam);
  }

  render(dt) {
    const r = this.renderer;
    this.time += dt;

    // 1. Scene into the HDR buffer.
    r.setRenderTarget(this.sceneRT);
    r.clear();
    r.render(this.scene, this.camera);

    // The viewmodel renders into the same HDR buffer on a cleared depth range,
    // so it is tone-mapped and bloomed with everything else. Rendering it after
    // the composite instead would write raw HDR straight to the back buffer,
    // where anything above 1.0 clips to flat white.
    if (this.viewScene && this.viewCamera) {
      r.autoClear = false;
      r.clearDepth();
      r.render(this.viewScene, this.viewCamera);
      r.autoClear = true;
    }

    if (!this.enabled) {
      r.setRenderTarget(null);
      this._quad.material = this.compositePass.material;
      this.compositePass.material.uniforms.tScene.value = this.sceneRT.texture;
      r.render(this._quad, this._orthoCam);
      return;
    }

    // 2. SSAO.
    let useAO = 0;
    if (this.settings.ssao && this.aoRT) {
      const u = this.aoPass.material.uniforms;
      u.tDepth.value = this.sceneRT.depthTexture;
      u.uRes.value.set(this.aoRT.width, this.aoRT.height);
      u.uProj.value.copy(this.camera.projectionMatrix);
      u.uProjInv.value.copy(this.camera.projectionMatrixInverse);
      u.uNear.value = this.camera.near; u.uFar.value = this.camera.far;
      this._blit(this.aoPass, this.aoRT);

      const bu = this.aoBlurPass.material.uniforms;
      bu.uRes.value.set(this.aoRT.width, this.aoRT.height);
      bu.tAO.value = this.aoRT.texture;  bu.uDir.value.set(1, 0);
      this._blit(this.aoBlurPass, this.aoBlurRT);
      bu.tAO.value = this.aoBlurRT.texture; bu.uDir.value.set(0, 1);
      this._blit(this.aoBlurPass, this.aoRT);
      useAO = 1;
    }

    // 3. Bloom: bright pass, then down the pyramid and back up.
    if (this.settings.bloom && this.mips.length) {
      const bu = this.brightPass.material.uniforms;
      bu.tScene.value = this.sceneRT.texture;
      bu.uExposure.value = this.exposure;
      this._blit(this.brightPass, this.mips[0].rt);

      for (let i = 1; i < this.mips.length; i++) {
        const src = this.mips[i - 1], dst = this.mips[i];
        this.downPass.material.uniforms.tSrc.value = src.rt.texture;
        this.downPass.material.uniforms.uTexel.value.set(1 / src.w, 1 / src.h);
        this._blit(this.downPass, dst.rt);
      }
      // Additive up-chain so each level contributes its own frequency band.
      const prevAutoClear = r.autoClear;
      r.autoClear = false;
      this.upPass.material.blending = THREE.AdditiveBlending;
      for (let i = this.mips.length - 1; i > 0; i--) {
        const src = this.mips[i], dst = this.mips[i - 1];
        this.upPass.material.uniforms.tSrc.value = src.rt.texture;
        this.upPass.material.uniforms.uTexel.value.set(1 / src.w, 1 / src.h);
        this._blit(this.upPass, dst.rt);
      }
      r.autoClear = prevAutoClear;
    }

    // 4. Composite.
    const cu = this.compositePass.material.uniforms;
    cu.tScene.value = this.sceneRT.texture;
    cu.tBloom.value = this.mips.length ? this.mips[0].rt.texture : null;
    cu.tAO.value = this.aoRT ? this.aoRT.texture : null;
    cu.tDepth.value = this.sceneRT.depthTexture;
    cu.uUseAO.value = useAO;
    cu.uRes.value.set(this.width, this.height);
    cu.uExposure.value = this.exposure;
    cu.uBloomStrength.value = this.settings.bloom ? 0.5 : 0;
    cu.uTime.value = this.time;
    cu.uGrain.value = this.settings.grain ? 0.035 : 0;
    cu.uChroma.value = this.settings.chromatic ? 0.0022 : 0;
    cu.uFlash.value = this.flashAmount;
    cu.uDamage.value = this.damageAmount;
    cu.uNear.value = this.camera.near;
    cu.uFar.value = this.camera.far;
    cu.uDofStrength.value = this.adsAmount * 1.6;
    cu.uDofFocus.value = 14;

    const wantFXAA = this.settings.aaMode !== 'none';
    this._blit(this.compositePass, wantFXAA ? this.compositeRT : null);

    if (wantFXAA) {
      this.fxaaPass.material.uniforms.tSrc.value = this.compositeRT.texture;
      this.fxaaPass.material.uniforms.uTexel.value.set(1 / this.width, 1 / this.height);
      this._blit(this.fxaaPass, null);
    }
    this.renderer.setRenderTarget(null);
  }

  dispose() {
    this._disposeTargets();
    [this.aoPass, this.aoBlurPass, this.brightPass, this.downPass,
     this.upPass, this.compositePass, this.fxaaPass].forEach(p => p?.dispose());
    this._quad.geometry.dispose();
  }
}

# BLACKSITE

A tactical FPS built in Three.js that runs in a mobile browser. No engine, no
asset pipeline, no build step — every texture, model, sound and animation is
generated procedurally at runtime.

## Playing it on an iPhone

The game is a static site, so GitHub Pages will host it as-is:

1. In this repository, open **Settings → Pages**.
2. Under *Build and deployment*, set **Source** to `Deploy from a branch`,
   pick the branch this lives on, folder `/ (root)`, and save.
3. Wait for the deploy, then open `https://<user>.github.io/kelly/fps/` in
   Safari on the phone.
4. Tap the share icon → **Add to Home Screen**. Launching from the home screen
   runs it fullscreen with no browser chrome, which is both the intended
   presentation and a measurable framerate win.

Hold the phone in landscape. The game asks you to rotate if you don't.

To run it locally instead, any static server works — the ES modules and the
importmap need a real HTTP origin, so opening `index.html` from the filesystem
will not work:

```sh
cd fps && python3 -m http.server 8080
```

## Controls

**Touch** — left thumb drags a floating stick to move (push to the edge to
sprint), right thumb drags anywhere to aim. The fire button also aims while
held, so the whole game is playable with two thumbs. Remaining actions sit in
thumb-reach arcs in the bottom corners.

**Keyboard/mouse** — `WASD` move, `Shift` sprint, `Ctrl`/`C` crouch (crouch
while sprinting to slide), `Space` jump, `R` reload, `F` melee, `G` grenade,
`Q`/`E` lean, `1`–`3` weapons, right mouse to aim, `Esc` to pause.

**Gamepad** — any standard controller is picked up automatically.

## How it fits together

```
index.html          shell, importmap, loading screen, error surface
vendor/             Three.js r0.185, vendored so there is no CDN dependency
src/main.js         bootstrap, fixed-timestep sim, render loop, state machine
src/core/
  Perf.js           device tiering + adaptive resolution
  PostFX.js         SSAO, bloom, ACES composite, DOF, grain, FXAA
  Input.js          unified keyboard/mouse/touch/gamepad state
  Audio.js          Web Audio synthesis — every sound is generated
src/world/
  Textures.js       procedural PBR material synthesis
  Sky.js            atmospheric scattering, sun, IBL, fog
  Level.js          level geometry, colliders, spawns
src/player/
  Controller.js     movement, collision, slide/mantle, view bob
  Weapons.js        procedural weapon models, gunplay, ballistics
src/ai/Enemy.js     procedural soldiers, animation, squad AI
src/fx/             GPU particles and decals
src/ui/             HUD, menus, and the touch control layer
```

The simulation runs at a fixed 120 Hz and is decoupled from rendering, so
physics and recoil behave identically whether the device is holding 120 fps or
struggling at 30. Quality tier is detected once at startup and render
resolution then floats to hold the frame budget.

## Scope, honestly

This targets what a browser can actually do on a phone: WebGL2, no compute
shaders, a ~2 GB memory ceiling in Safari, and a payload measured in hundreds
of kilobytes rather than the ~150 GB of authored assets a current Call of Duty
ships. It is built to be the best-looking thing that fits in those limits, not
to be mistaken for one of those games.

# RapidRetouch: development notes

What we decided, why, what was tried and rejected, and what's next, so the project can
be picked up without the conversations it came from. The original brief is
[RETOUCH_APP_BRIEF.md](../RETOUCH_APP_BRIEF.md); where they differ, this file is newer.

Last updated: 2026-10-03.

---

## 1. Standing rules

- **Commercial-safe by default.** Users are photographers doing paid client work. Default
  models must allow commercial use (MIT / Apache). Non-commercial models are opt-in only,
  behind a licence-acceptance prompt (the registry supports this).
- **Never commit model weights.** Download on first use from the official source, verified
  by sha256 (`source.kind: url`) or a pinned Hugging Face revision.
- **Record provenance.** Every export's `.retouch.json` lists each step, its settings,
  area/mask edits and the model versions used.
- **Skin retouching stays local, always.** Optional online backends (fal.ai / Replicate,
  FLUX Kontext) were planned in the brief only for big generative fills, sending masked
  crops, never whole photos. None have been built.
- **Never redraw faces.** Sliders are classic image processing; generative models only
  touch what the user paints (Remove brush).
- **Don't detect sex/gender** (Evoto does). It misclassifies people, relies on
  non-commercial data, and raises privacy issues. Detect the features that matter instead
  (stubble/beard, texture, wrinkle depth). Skin evening must leave stubble alone.
- **CPU must stay fully usable;** the GPU is an optional extra. Low-spec machines matter.
- **Installation must be as simple as possible on all Linux distros:** no asking users to
  install CUDA, ROCm, Python…
- **Don't use client photos** unless Chris points to them.

## 2. Working with Chris

- Commit only when asked; push only when asked. All commits are on `master`, remote
  `origin` = private repo github.com/crispin81/RapidRetouch.
- Prefers Tauri for desktop apps; the UI matches RapidCulling's look (title bar, footer,
  yellow accents, "New user? Watch this first!" banner).
- Likes being shown before/after numbers, not just impressions. Two lessons from this
  project: **verify visual changes numerically** (side-by-side JPEG judgements misled
  twice), and a "wide taper" read to him as the opposite of "reaching the edge".
- Sign-offs to respect: don't retune these without being asked:
  Eye bags ("perfect"), Backdrop Brightness ("brilliant"), Eye whites strength
  ("works good"), the speed work ("I love this").
- No "coming soon" text anywhere in the UI.
- The next phase (from 2026-10-03) is **polishing only, towards the first AppImage
  (1.0.0)**. No new features unless asked.

## 3. Architecture

```
Tauri 2 + React UI (src/, src-tauri/)
   │  JSON lines over stdin/stdout (engine_call / engine-event)
   ▼
Python engine (engine/, uv, Python 3.12, torch cu128)
   ├─ server.py: Engine: state, caches, all methods
   ├─ tools/*.py: one module per tool
   ├─ registry.py + manifests/ + adapters/: AI models
   └─ imageio.py: RAW/TIFF/JPEG in, 16-bit TIFF / JPEG out
```

- Why a Python sidecar: every model works in Python and tuning is fast there; the UI
  never needs rewriting. Cost: bundling Python + torch is several GB (see §9).
- Requests are serial except `thumbnail`, which runs on one background thread (two threads
  slowed renders).
- **Preview** is 2048 px on the long edge (`PREVIEW_EDGE`). Zooming past it asks
  `render_region` for full-resolution pixels of the view.
- **Settings are per photo** (`PhotoSettings` in App.tsx), kept in memory for the film
  strip. The engine keeps only edit lists (strokes, mask/area edits, faces) for photos
  that aren't open (`sessions`).
- **Presets** (`presets.py`, `~/.config/rapidretouch/presets/*.json`) hold every slider
  except Backdrop/Outdoor mode and the crop. Brush work is per photo. `withPreset()` fills
  sliders added later with defaults and maps legacy keys (`blemishes`→`acne`, a single
  `dodgeBurn` number → highlights & shadows).
- **Copy/paste** carries all sliders, the mode and the crop, but no brush work.

### Pipeline order (preview, zoom and export are the same)

1. Removals (Remove brush, Patch, Glasses reflections), applied first.
2. Backdrop smoothing (skipped in Outdoor mode).
3. Face skin → Neck & Body skin → Dodge & burn → Eyes → Mouth (each a cached stage).
4. Clothes creases (last, measured once at full strength; the slider only scales it).
5. Tone (per pixel; cached around, so tone sliders never re-render the rest).
6. Crop & straighten: export only.

### Caching (server.py)

- `_staged`: LRU of each face stage's output, keyed by its settings and all earlier ones,
  so moving one slider reruns only that tool and those after it. float32 on purpose:
  float16 caused up to 4.5/255 differences depending on slider history.
- `_pretone`: the preview before tone (2 slots, so a hold-for-before doesn't evict).
- Full resolution: `_full` (edited photo, subject mask, backdrop lighting), `_zoom_tiles`
  (last 4 zoomed areas), `_face_window_cache` (8 face-stage results on the face crop).
- Frontend: detail tiles are only requested 400 ms after the last preview change, so a
  slow zoom render never blocks slider previews.

## 4. Tools: how they work and why

**Backdrop smoothing** (`backdrop_smooth.py`)
- Robust IRLS lighting fit (creases drop out instead of smearing).
- "Evenness" blends toward an asymmetric-least-squares degree-4 polynomial "intended
  lighting" surface. It's lopsided on purpose: it climbs out of shadows.
- Near hair: the matting equation I = aF + (1−a)B with B swapped for the smoothed
  backdrop, so strands keep their detail and there are no halos.
- Grain is measured, then re-synthesised and added in quadrature.
- Brightness: a linear-light gain on the backdrop's share of each pixel.
- Known limit, by design: darkening in a corner is indistinguishable from light falloff.
- Grain is generated in fixed 256 px blocks seeded by position, so any zoomed area gets
  exactly the export's grain. `lighting()` is the whole-photo part; `prepare(box=…)` /
  `smooth_area()` work on any area.

**Scene detection** (`scene.py`): border strips; WEIGHTS (0.2, 0.3, 0.5), COLOUR ramp
(6, 9). All 72 test photos are correct (studio colour spread ≤ 5.7, outdoor ≥ 10.4).
When unsure, it picks backdrop. An "auto" tag shows on the mode button; a click or paste
clears it.

**Skin** (`skin.py`)
- The skin map is the face oval ∩ a SkinModel (chromaticity learnt from cheeks and
  forehead), minus eyes, brows and lips. Stubble is found by texture, only below the
  cheekbones.
- Every "surrounding skin" estimate is edge-aware (guided filter), or the shadow line of
  split lighting gets "fixed".
- Even tone works in linear chromaticity, keeping brightness; it covers the face region
  (no colour test) × not hair × lit, because a red nose fails a colour test.
- **Acne** (rebuilt 2026-10-03, replacing Blemishes):
  - A spot scores as darker or redder than a local median, in units of the skin's own
    variation. The slider sets the threshold: 7σ (clearest) to 3σ (faintest).
  - Size, shape, edge, mole and "ring of skin around it" tests reject false alarms.
  - Healing: inpaint the tone layer, put the texture back.
  - Found at full detail even for the preview (`acne_masks`), so the preview and zoom
    agree and spots don't flicker.
  - Tested on DSCF7253 (chin spot).
- **Texture** reduces pores in a band (0.0015–0.015 face widths) and keeps the finest grain
  and highlights. It keeps hair strands (`_strands`) and is 25% stronger on the chin.
- **Moles are kept** by default (strong, round, brown not red).
- Wrinkle zones: forehead, frown, smile (+ cheek lines for older subjects), chin, neck. A
  Hessian line detector fills lines by closing, and deep smile folds are only softened.
- **Neck & Body:**
  - Person's colour × subject mask, minus clothes (person-parts model).
  - Even tone aims at the region's own average; aiming at the face turned a chest orange.
  - Bearded faces make the neck stubble test far more sensitive.
- The person-parts **hair** map keeps Acne, Even tone and Dodge & burn off a fringe.
- **Refine area:** per-photo brush corrections to the face, neck, body and clothes areas
  (`region_edit.py`; strokes in image fractions).

**Dodge & burn** (`dodge_burn.py`), replacing "Sculpt", which was weak and left a line
across the forehead:
- Contour: landmark zones blurred well past their size, light-aware: no dodging in shade,
  no burning in highlights.
- Highlights and Shadows strengthen the face's own light shape through a roll-off curve.
- Applied in linear light.
- Keeps off hair (parts map) and single strands near the hairline (strand tracing is
  limited to there, because forehead lines look the same).

**Eyes** (`eyes.py`)
- Landmarks come from two passes: detect at 2048, then refine each face on a full-res
  1024 crop. The single pass put the iris about 20 px high on a squinting face.
- Shading lifts must be linear-light gains (`_relight`). Lab-L-only turns skin grey;
  scaling a/b blows up red noise.
- Whites:
  - Strength 0.625, tapering from beside the iris toward the corners (taper 1.4–2.4 iris
    radii, corner keep 0.1).
  - Lid fade 0.085 eye widths.
- Veins: detected veins are replaced fully, and the white's diffuse redness is pulled
  toward its 20th-percentile a*.
- Iris:
  - The pupil is found by a radial brightness profile; thresholding swallowed dark
    irises.
  - The limbal ring is kept for detail.
- **Iris hue:**
  - Range 150° (doubled so green can turn blue).
  - The chroma floor stays off near-black pixels to avoid red-eye.
  - Its colour area reaches nearly to the white (0.9–1.02 r) and fades right across the
    pupil edge.
  - The colour bar is measured from each photo's own irises (`iris_scale`).
- Iris saturation uses the same area.

**Mouth** (`mouth.py`)
- Lip mask: the outer outline is grown 0.012 mouth widths (landmarks sit inside the
  visible edge), with a 0.018 feather. The teeth edge stays put.
- Lip hue (±25°) and Lip saturation; muting goes toward the surrounding skin colour,
  not grey.
- Teeth whitening: bright, not red, inside the inner lip.

**Clothes creases** (`fabric.py`): shadow softening ("Fine creases"), not removal. Light
fabric flattens up to its lit level, crease lines are filled like wrinkles, and stripes are
protected by line coverage. The panel note says to use Patch for real removal.

**Tone** (`tone.py`)
- Order: Temperature/Tint (linear channel gains, brightness kept) → Exposure → Vibrance
  → Luminosity curve.
- Vibrance:
  - Skin hues are protected and dull colours gain most.
  - At full: ×1.69 (Chris asked for +25%, then settled on +15%).
- The curve changes Lab L only, never colour.

**Remove** (`inpaint.py`, `patch.py`, `reflection.py`): LaMa per context crop; Patch is a
frequency-separated heal; Glasses reflections is a least-squares tint split, still labelled
"(alpha)" and leaving a faint trace.

**Crop & straighten** (`crop.py`, `CropOverlay.tsx`)
- Rotate about the centre on a same-size canvas, then a fractional crop. Export only.
- Drag outside the frame to rotate; aspect presets.
- While dragging, the frame is redrawn locally and handed to the app on release.
  Re-rendering App on every move was the stutter.
- Keep the last frame while a new preview loads; clearing caused flicker.

**Export** (`ExportDialog.tsx`, `server.export`)
- Batch by default: the film-strip selection, or the current photo when nothing is
  selected. Each photo is exported with its own settings.
- Per-photo progress bars driven by `progress` events.
- 16-bit TIFF (deflate) or JPEG quality 100, 4:4:4, ICC profile embedded.
- Names are `name_retouched.tif/.jpg`; a re-export overwrites.

## 5. Performance (benchmark: 100 MP GFX, 11662×8746, DSCF7253)

| | Time |
|---|---|
| Tone / Mouth / Eyes slider | 0.04–0.05 s |
| Skin / Backdrop / Dodge & burn slider | 0.1–0.4 s |
| First zoom on the face / on the backdrop / pan | 4 s / 0.2 s / 0.01 s |
| Zoomed, after an Eyes / Dodge / Skin / Backdrop change | 0.5 / 1.0 / 2.4 / 2.4 s |
| Export TIFF | ~9 s |
| First preview of a photo (models + mask) | ~4.3 s |

How:
- Zooming renders only the area in view (+25% spare):
  - The backdrop is done for that area alone.
  - Face tools run on a crop around the faces, with coordinates translated by
    `_frame_faces`, `_frame_map` and `_frame_edits`.
  - Neck & Body and Creases work on the whole person, so with those it falls back to the
    whole photo.
- Grain is measured on 24 sampled strips and generated on all cores.
- `parallel.py` (`by_rows`, `by_bands`) runs per-pixel work on all cores with identical
  results. Nested splitting is guarded, since it would deadlock the pool.

Rejected for speed (don't retry):
- `cv2.randn`: 15% too weak on 100 MP arrays.
- Lookup-table colour conversion: slower than the exact formula on threads.
- Parallel bands for the whole-photo backdrop: slower, because OpenCV already
  multithreads and the margins repeat work.

**GPU advice given:** not worth it for the pixel maths any more; the AI models are where a
GPU matters.

## 6. Tried and rejected (quality)

- **LaMa** for creases, eye veins and blemishes: all measured worse. It continues the
  surroundings, fills with already-pink white, and invents freckles. LaMa is for the
  Remove brush only.
- Blemishes over the face-region mask caught mouth-corner shadows and hair; replaced by
  Acne.
- Bigger blemish scales: these healed broad shading.
- Sculpt (single slider): replaced by Contour / Highlights / Shadows.
- Wide colour tapers at iris and lip edges: read as pulling back from the edges.
- Glasses: dark-channel-only estimate, fixed near-black eye floor, broad-brightness term.
  All rejected.
- Flyaway hairs (`tools/flyaways.py`, not wired up): three rounds failed. Strays connect to
  the hair mass through faint pixels. The next idea is per-strand centre-line tracing.
  **Paused.**
- SAM click-select and auto spot detection: not wanted for now; the brush is enough.

## 7. Test photos (`~/Pictures/Studio test images/`, read-only)

- Studio, Panasonic RW2:
  - P1167822: striped shirt, red nose.
  - P1256049: crow's feet.
  - P1246006 / P1256101: under-eye lines, veins.
  - P1256063: white shirt creases.
  - P1256027/28/76/78/83: glasses.
  - P1256060: two people.
- Body skin: DSC_2376-2 (neck, chest, hands), DSC_2509 / DSC_2552 (skin-tone dresses;
  DSC_2552's strapless dress is a known limit), DSC_2383-2 (hat brim).
- Outdoor 100 MP Fuji RAF:
  - DSCF7253: chin spot, fringe, blue-grey eyes; the main benchmark.
  - DSCF7266 / DSCF7323: fringe.
  - DSCF7094–7100: bluebells, no detectable face.

## 8. Gotchas

- `pkill -f`/`pgrep -f` with a pattern in the command kills the calling shell (exit 144).
  Kill by PID: `ps -eo pid,args | grep "[p]attern"`. If vite is left on port 1420, the
  next `tauri dev` fails.
- `mediapipe<1` is required: 1.0.1 is SIGKILLed creating the landmarker. mediapipe pulls in
  opencv-contrib, which is overridden in pyproject. If cv2 vanishes:
  `uv sync --reinstall-package opencv-python-headless`.
- GTK file dialog filters are case-sensitive: list both `.RW2` and `.rw2`.
- RAW: neutral develop (camera WB, sRGB, 16-bit, no auto-bright; auto-bright clips white
  backdrops). Thumbnails use the same develop at half size, so colours don't jump on open.
  Read `raw.sizes` before the half-size postprocess.
- Keep the WebKit/Wayland env workaround in `src-tauri/src/main.rs`
  (GDK_BACKEND=x11,wayland; WEBKIT_DISABLE_DMABUF_RENDERER; WEBKIT_DISABLE_COMPOSITING_MODE,
  never overriding the user's own values).
- Dev machine: an RTX 5070 Ti (Blackwell, needs CUDA 12.8+ builds). The Hyprland rule
  `^[Rr]apid[Rr]etouch$` keeps the window opaque.
- The BiRefNet adapter empties the CUDA cache after each run and falls back to the CPU on
  out-of-memory (the GPU is shared).

## 9. Release plan

- **Versioning** (agreed for all three apps, copying RapidRAW):
  - Plain numbers, no "beta" anywhere ("open source is permanently beta").
  - Every release bumps package.json, Cargo.toml, tauri.conf.json and the tag: last digit
    for fixes and small features, middle for bigger batches, major only if old
    presets/settings break.
  - Turn off `prerelease: true` in release workflows.
  - RapidRetouch goes 0.1.0 → **1.0.0** at its first public release. RapidCulling and
    RapidTimelapse go to 1.0.1 at their next release.
- **First release: AppImage** (then .deb/.rpm like RapidCulling).
  - Copy RapidCulling's patched `linuxdeploy-plugin-gtk.sh` (GDK_BACKEND=wayland,x11)
    and release workflow. There's no `.github/` here yet.
  - Packaging direction: ONNX Runtime instead of PyTorch, a CPU baseline in the
    installer (~1–1.5 GB target, against ~7.8 GB today), models downloaded on first run,
    and a hardware-matched GPU pack (CUDA / ROCm / OpenVINO; maybe Vulkan) downloaded on
    first run.
  - LaMa (FFTs) and BiRefNet (transformer) need op-coverage testing in ONNX.
- Before going public:
  - Make the repo public (AGPL; the About page links the source).
  - Add the About photo (600×600 from Chris) and the tutorial video URL
    (`TUTORIAL_VIDEO_URL`).
  - Remove any "beta" from the title bar.

## 10. Backlog (not started unless asked)

- Background preparation: preload models at start-up and find mask, faces and hair while
  browsing (first preview 4.3 s → ~instant).
- Time BiRefNet CPU-only; consider ONNX Runtime with DirectML / Core ML / OpenVINO for
  AMD/Intel GPUs.
- Flyaway hairs (strand tracing).
- Per-person settings for group shots (deferred to a later version).
- Acne tuning on a real acne photo, when Chris has one.
- Opt-in online generative fill for big removals (FLUX Kontext via fal.ai), never for skin.

## 11. History

| Commit | Contents |
|---|---|
| e78761e … beff0d2 | Engine, backdrop, removals, mask brush, eyes, skin, film strip, chrome, AGPL, presets |
| 04582f7 | Rename to RapidRetouch; neck & body, refine area, three-slider dodge & burn |
| 13f3c43 | 16-bit TIFF export, About page, coffee note, social links |
| 4fed7cd | Acne rebuild, crop & straighten, batch export, Temperature/Tint/Vibrance, iris & lip hue |
| a071e5c | Speed work (area-only zoom, parallel tone/grain), iris hue ×2, colour edges, crop flicker |

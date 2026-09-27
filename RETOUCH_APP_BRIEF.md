# Linux AI Retouch App (Evoto-style, open source)

## Goal
An open-source Linux desktop app for AI-assisted studio portrait retouching, similar to Evoto.
Local AI by default, with optional online AI backends for better results on some tasks.
Privacy (no uploads unless the user opts in) and no per-image credits are key selling points.

## First-feature priorities
1. **Backdrop smoothing**: segment the subject (BiRefNet), feather the mask around hair,
   then split the backdrop into two layers. Keep the low-frequency lighting gradient/falloff,
   suppress the high-frequency creases, seams and dust, and add matched grain back.
   Watch for halos around hair and grain mismatch.
2. **Backdrop distraction removal**, in three tiers:
   - Small spots (dust, tape, scuffs, cables): auto-detect anything that breaks from a fitted
     smooth backdrop model, then inpaint with LaMa.
   - Named objects (light stands, sandbags, reflector edges): open-vocabulary detection
     (Grounding DINO / OWLv2) with a preset list of studio clutter, then LaMa.
   - Large areas (end of roll, wall visible, floor seams): generative fill, either a local
     large model or an optional online backend.
   - Plus a click-to-remove brush using SAM.
3. **Dark circles**: MediaPipe Face Mesh landmarks locate the under-eye area. Lift luminance
   and shift the purple/blue cast toward the cheek tone (Lab colour space), at low frequency
   only so texture is preserved.
4. **Eye bags**: automated dodge & burn. Lighten the shadow crease and reduce the bulge's local
   contrast. Needs a strength slider; 100% removal looks unnatural.
5. **Skin colour evening**: skin mask covering face, neck and arms. Pull blotchy/red patches
   toward a target tone at low frequency only. Lock the target tone per person across a batch
   for session consistency.

Suggested build order: backdrop cleanup first (smoothing + auto small-spot removal + SAM brush),
then the face-landmark stage, which the eye and skin tools share.

## Model policy (open source)
Defaults must be permissively licensed, because users are photographers doing paid client
work (commercial use).

- **Default / bundled (permissive):**
  - LaMa (Apache 2.0): object removal
  - MI-GAN (MIT): fast previews
  - BiRefNet (MIT): masking
  - SAM / SAM 2 (Apache 2.0): click-to-select
  - MediaPipe Face Mesh (Apache 2.0): face landmarks
  - Grounding DINO / OWLv2 (Apache 2.0): detection
  - Qwen-Image-Edit (Apache 2.0) or SDXL inpainting (OpenRAIL++): large generative fills
- **Optional add-ons, user opt-in:**
  - ObjectClear: better object removal, also removes shadows and reflections; non-commercial
  - OmniEraser and FLUX.1-dev-based models: non-commercial
  - RMBG-2.0: non-commercial
- **Grey area:** face-parsing and retouch models trained on CelebAMask-HQ / FFHQ-Retouch
  (non-commercial training data). Prefer MediaPipe landmarks plus a skin-colour mask for now.
- Verify every licence on the model's official page before shipping.
- Never commit weights to the repo. Download them on first run from the official source.

## Model registry design
Each backend (local default, optional local, online) is described by a manifest plus a thin
adapter, so new models plug in without core changes. Example:

```yaml
id: objectclear
task: object_removal
licence: S-Lab (non-commercial)
commercial_use: false
bundled_default: false
quality_note: "Better results, also removes shadows and reflections"
weights_url: <official source>
min_vram_gb: 12
```

- UI dropdowns show badges, e.g. "Better results · Non-commercial licence".
- Choosing a non-commercial model the first time shows its licence and requires acceptance
  before download.
- Record which model was used in each export's settings.

## Online backends (optional, opt-in per job)
- Send only masked crops, never full images, then blend results back at full resolution.
  Generative models redraw at low resolution and can alter faces.
- Integrate via fal.ai or Replicate (one key, many models).
- Default online model: FLUX Kontext, which is strongest at preserving texture and identity.
- Keep skin retouching local.

## Technical notes
- Models run at ~512–1024px, so tile/crop and blend back to preserve pore detail on 24–60MP files.
- Candidate stacks (undecided): a Python + ONNX Runtime core with a PySide6 UI, or Rust/Tauri
  with an ONNX backend (the same approach as RapidRAW).
- Input: 16-bit TIFF exports first; possibly raw via LibRaw later.
- GPU: NVIDIA via CUDA, AMD via ROCm/ONNX Runtime; CPU fallback for batch jobs.
- IOPaint (formerly Lama Cleaner) is useful as a testbed for comparing inpainting models.

## Open questions
- Which GPU the developer has (this decides model choices and defaults).
- A sample studio portrait for the first prototype.
- App licence: GPL-3.0 (keeps forks open) or Apache 2.0/MIT (maximum adoption).
- UI stack choice.

## Next step
Build a prototype script running all first-priority tools with adjustable strengths, with the
model registry in place from the start.

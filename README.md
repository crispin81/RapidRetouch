# RapidRetouch

AI-assisted portrait retouching for photographers, in the spirit of Evoto, but open
source and entirely on your own computer. No uploads, no accounts, no per-image
credits. Built for studio and outdoor portraits, from 24 MP up to 100 MP medium format.

Part of a family of apps by Chris Cork Photography, alongside RapidCulling and
RapidTimelapse.

> Status: in development towards the first public release (1.0.0, AppImage first).
> The repository is private until then. AGPL-3.0 requires the source to be public once
> the app is distributed.

## What it does

Every adjustment is a slider (or a brush where a slider can't know what you mean),
applied non-destructively and live on a 2048 px preview, with full-resolution detail
when you zoom in.

| Panel | Tools |
|---|---|
| **Remove** | Remove brush (LaMa inpainting), Patch (Photoshop-style lasso + drag), Glasses reflections (alpha) |
| **Tone** | Temperature, Tint, Exposure (±2.5 EV), Vibrance (skin protected), Luminosity curve (light only, never colour) |
| **Backdrop** | Smoothing (creases, seams, dust) with matched grain, Evenness, Brightness (±2.5 EV), edge protection for hair; Backdrop / Outdoor mode, detected automatically |
| **Skin** (Face / Neck / Body tabs) | Acne, Smooth, Even tone, Texture, Shine (matte ↔ gloss); Wrinkles: forehead, frown, smile and cheek lines, chin, neck lines. Moles are kept. Refine area brush per tab |
| **Dodge & burn** | Contour, Highlights, Shadows (keeps off hair) |
| **Eyes** | Dark circles, Eye bags, Wrinkles, Eye whites, Eye veins, Iris, Iris saturation, Iris hue, Catch light, Eyelashes |
| **Mouth** | Lip hue, Lip saturation, Lip smoothing, Teeth whitening |
| **Clothes** | Fine creases (shadow softening; use Patch for full removal) |

Plus: a film strip with multi-select, copy/paste of settings, presets, a per-panel
"Hold for before", crop & straighten, and batch export to 16-bit TIFF or
maximum-quality JPEG with a progress bar per photo. Each export gets a
`<file>.retouch.json` sidecar recording every step, its settings and the exact AI
model versions used.

## AI models

Only permissively licensed models are used by default, because the people using this
do paid client work. Weights are never stored in this repository: they're downloaded
on first use from their official source and verified by checksum.

| Model | Used for | Licence |
|---|---|---|
| BiRefNet HR | Subject mask (backdrop tools) | MIT |
| LaMa | Remove brush | Apache-2.0 |
| MediaPipe Face Landmarker | Face, eye and mouth positions | Apache-2.0 |
| MediaPipe Multi-class Selfie Segmentation | Hair, clothes and skin areas | Apache-2.0 |

All the retouching sliders themselves are classic image processing (frequency
separation, edge-aware filters, linear-light dodge & burn), not generative AI, so
faces are never redrawn. Models with non-commercial licences may be offered later
as opt-in extras, behind a licence prompt.

## Running from source

Requirements: Linux, Node.js 20+, Rust (stable), [uv](https://docs.astral.sh/uv/),
and the Tauri 2 system packages (WebKitGTK 4.1 etc.). An NVIDIA GPU is used when
present (the AI models); everything works on the CPU too.

```sh
npm install
(cd engine && uv sync)   # Python 3.12 engine: PyTorch (CUDA 12.8 build), OpenCV, rawpy, MediaPipe
npm run tauri dev        # starts the UI; the UI starts the engine
```

Engine tests: `cd engine && uv run pytest -q`. Type-check the UI: `npx tsc --noEmit`.

Models download to `~/.local/share/rapidretouch/` the first time they're needed.
Presets live in `~/.config/rapidretouch/presets/`.

## How it's built

- **UI:** Tauri 2 + React + TypeScript (`src/`, `src-tauri/`).
- **Engine:** a Python process (`engine/`) the UI starts and talks to over JSON lines
  on stdin/stdout. It does all image processing and model inference.
- Developer notes, design decisions, tuning history and the release plan are in
  [docs/DEVELOPMENT_NOTES.md](docs/DEVELOPMENT_NOTES.md).

## Licence

[AGPL-3.0-only](LICENSE). Copyright Chris Cork.

If you find it useful: [buy me a coffee](https://buymeacoffee.com/chriscorkphotography).
· [chriscorkphotography.co.uk](https://www.chriscorkphotography.co.uk)
· [Instagram](https://www.instagram.com/chriscorkphotography)
· [Facebook](https://www.facebook.com/chriscorkphoto)

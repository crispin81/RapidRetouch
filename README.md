# RapidRetouch

AI-assisted portrait retouching for photographers, in the spirit of Evoto, but open
source and entirely on your own computer. No uploads, no accounts, no per-image
credits. Built for studio and outdoor portraits, from 24 MP up to 100 MP medium format.

Part of a family of apps by Chris Cork Photography, alongside RapidCulling and
RapidTimelapse.

> **Actively developed: expect frequent updates.**

▶ **New to RapidRetouch? [Watch the tutorial video](https://youtu.be/dvJXRj3YTuc).**

## What it does

Every adjustment is a slider (or a brush where a slider can't know what you mean),
applied non-destructively and live on a 2048 px preview, with full-resolution detail
when you zoom in.

| Panel | Tools |
|---|---|
| **Remove** | Remove brush (LaMa inpainting), Patch (Photoshop-style lasso + drag), Glasses reflections (alpha) |
| **Tone** | Temperature, Tint, Exposure (±2.5 EV), Dehaze, Vibrance (skin protected), Luminosity curve (light only, never colour) |
| **Backdrop** | Smoothing (creases, seams, dust) with matched grain, Evenness, Brightness (±2.5 EV), edge protection for hair; Backdrop / Outdoor mode, detected automatically |
| **Skin** (Face / Neck / Body tabs) | Acne, Blemishes, Smooth, Even tone, Texture, Pores, Shine (matte ↔ gloss); Wrinkles: forehead, frown, smile and cheek lines, chin, neck lines. Moles are kept. Refine area brush per tab |
| **Dodge & burn** | Contour, Highlights, Shadows (keeps off hair) |
| **Relight** | A radial light placed over the face: Exposure, Warmth and Feather, with the circle moved, stretched, turned and feathered by hand |
| **Eyes** | Dark circles, Eye bags, Wrinkles, Eye whites, Eye veins, Iris, Iris saturation, Iris hue, Catch light, Eyelashes |
| **Mouth** | Lip hue, Lip saturation, Lip smoothing, Teeth whitening |
| **Clothes** | Fine creases (shadow softening; use Patch for full removal) |

Skin, Dodge & burn, Eyes and Mouth each have an Opacity that fades the whole panel together.

Plus: a film strip with multi-select, copy/paste of settings, presets, a per-panel
"Hold for before", crop & straighten, and batch export to 16-bit TIFF or
maximum-quality JPEG (optionally scaled to a long edge) with a progress bar per photo.

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

## Installing

Download the installer for your computer from
[Releases](https://github.com/crispin81/RapidRetouch/releases), install it, and open
RapidRetouch. There's nothing to set up.

- **Windows** (64-bit): `RapidRetouch_…_x64-setup.exe` (or the `.msi`).
- **Mac** (Apple Silicon: M1 and later): `RapidRetouch_…_aarch64.dmg`. Open it and drag
  RapidRetouch into Applications. Intel Macs aren't supported: the AI library it uses
  no longer makes Intel Mac versions.
- **Linux**: `RapidRetouch_…_amd64.AppImage` runs on most distributions (make it
  executable, then open it), or install the `.deb` / `.rpm` for your distribution.

**First launch:** RapidRetouch downloads its AI engine once, showing its progress as it
goes: about 3 GB on a PC with an NVIDIA graphics card (which it then uses for speed),
about 1 GB otherwise. It needs an internet connection for that, and about 7 GB of disk
space with an NVIDIA card (2–3 GB otherwise). After that it starts straight away and
works offline, apart from downloading each AI model the first time a tool needs it.

**The installers aren't code-signed yet** (that needs a paid developer certificate), so
your computer will warn that the publisher is unverified the first time:

- **Windows**: SmartScreen says "Windows protected your PC". Click **More info**, then
  **Run anyway**.
- **Mac**: macOS refuses to open it the first time. Either run this once in Terminal,
  then open it as normal:
  ```
  xattr -d com.apple.quarantine /Applications/RapidRetouch.app
  ```
  or try to open it once, then go to **System Settings → Privacy & Security**, scroll to
  the note about RapidRetouch and click **Open Anyway**.

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

Photos never leave your computer. The app only goes online to download a model the first
time it's needed and to read `links.json` from this repository (the tutorial video's link).

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

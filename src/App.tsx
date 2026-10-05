import { useCallback, useEffect, useRef, useState, type PointerEvent } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import { getCurrentWindow, UserAttentionType } from "@tauri-apps/api/window";
import {
  FolderOpen,
  Download,
  Eye,
  Layers,
  Paintbrush,
  Undo2,
  Trash2,
  Plus,
  Minus,
  ZoomIn,
  Maximize,
  Coffee,
  Smile,
  Glasses,
  RotateCcw,
  ChevronRight,
  ChevronsDownUp,
  ChevronsUpDown,
  X,
  Sun,
  Lasso,
  Crop as CropIcon,
  GripVertical,
  Circle,
} from "lucide-react";
import {
  BackdropParams,
  EngineError,
  EyesParams,
  MouthParams,
  SkinParams,
  SkinRegion,
  ModelInfo,
  OpenResult,
  call,
  jpegSrc,
  onEngineEvent,
} from "./api";
import About from "./About";
import SetupScreen from "./SetupScreen";
import ExportDialog, { ExportFormat, ExportRow, ExportSize } from "./ExportDialog";
import BrushBar from "./BrushBar";
import OpacityGroup from "./OpacityGroup";
import PaintOverlay from "./PaintOverlay";
import PatchOverlay from "./PatchOverlay";
import CropOverlay, { Crop, NO_CROP, cropToRatio, fitCrop, isNoCrop } from "./CropOverlay";
import RelightOverlay, { RelightShape } from "./RelightOverlay";
import CurveEditor, { IDENTITY, Point } from "./CurveEditor";
import PresetMenu, { Preset } from "./PresetMenu";
import Slider from "./Slider";
import Viewer, { Detail, ViewerHandle } from "./Viewer";
import TitleBar from "./TitleBar";
import FilmStrip, { StripItem } from "./FilmStrip";

interface PhotoSettings {
  backdrop: BackdropParams;
  outdoor: boolean; // no backdrop step: for portraits not shot on a backdrop
  eyes: EyesParams;
  skin: SkinParams;
  mouth: MouthParams;
  dodgeBurn: DodgeBurnParams;
  creases: number; // clothes crease smoothing 0..1
  tone: Tone;
  crop: Crop; // straighten and crop, applied at export; not part of presets
  opacity: Opacity; // each face panel's overall amount
  relight: Relight | null; // placed on this photo's face, so not part of presets
  off: string[]; // sliders switched off by their eye (see withOff), e.g. "skin.face.smooth"
}
/** Relight: a radial gradient over the face (see RelightOverlay and the
 * engine's tools/relight.py). */
type Relight = RelightShape & { feather: number; exposure: number; warmth: number; invert: boolean };
const RELIGHT_LOOK = { feather: 0.6, exposure: 0.5, warmth: 0, invert: false };
/** The engine's relight setting: null while it would change nothing. */
const relightLook = (r: Relight | null) => (r && (r.exposure !== 0 || r.warmth !== 0) ? r : null);
/** Each face panel's overall amount (0..1): its whole result faded toward the
 * photo before it, like a layer's opacity. */
interface Opacity {
  skin: number;
  dodge_burn: number;
  eyes: number;
  mouth: number;
}
const OPACITY_DEFAULTS: Opacity = { skin: 1, dodge_burn: 1, eyes: 1, mouth: 1 };
interface Tone {
  temperature: number; // -1 bluer .. +1 warmer
  tint: number; // -1 greener .. +1 more magenta
  ev: number;
  dehaze: number; // -1 hazier .. +1 clearer (the photo's own haze veil taken off)
  vibrance: number; // -1 .. +1, skin tones protected, dull colours gain most (like Lightroom's)
  curve: Point[]; // luminosity curve
}
const TONE_DEFAULTS: Tone = { temperature: 0, tint: 0, ev: 0, dehaze: 0, vibrance: 0, curve: IDENTITY };
// Colour bars for the white balance sliders.
const TEMPERATURE_SCALE = "linear-gradient(to right, #4a86d8, #b9c3cf, #e0a74a)";
const TINT_SCALE = "linear-gradient(to right, #5fae58, #c0c0c0, #c45cb8)";
const signed = (v: number) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2));
const sameValues = (a: object, b: object) => JSON.stringify(a) === JSON.stringify(b);
const sameSettings = (a: PhotoSettings, b: PhotoSettings) => sameValues(a, b);

const COFFEE_URL = "https://buymeacoffee.com/chriscorkphotography";
// The tutorial video's link lives in links.json in the GitHub repo, read at
// start-up, so it can be set or changed after a release without a new one.
// The last link read is remembered, for offline starts. No link: no banner.
// While developing, the repo's own copy is read (served by Vite), so a link
// can be tried before it's published.
const LINKS_URL = import.meta.env.DEV
  ? "/links.json"
  : "https://raw.githubusercontent.com/crispin81/RapidRetouch/master/links.json";
const LINKS_TIMEOUT_MS = 5000;
const TUTORIAL_URL_KEY = "rapidretouch.tutorialUrl";
const TUTORIAL_DISMISSED_KEY = "rapidretouch.tutorialDismissed";
const EXPORT_FORMAT_KEY = "rapidretouch.exportFormat";

/** Bring the window to the front (after the first-launch setup, when About
 * opens by itself), or where the system won't allow that (Windows sometimes),
 * flash it in the taskbar / bounce it in the Dock. */
function bringToFront(): void {
  const win = getCurrentWindow();
  win
    .unminimize()
    .then(() => win.show())
    .then(() => win.setFocus())
    .catch(() => undefined);
  win.isFocused().then((focused) => {
    if (!focused) win.requestUserAttention(UserAttentionType.Informational).catch(() => undefined);
  }).catch(() => undefined);
}

const EXPORT_SIZE_KEY = "rapidretouch.exportSize";

const PREVIEW_EDGE = 2048; // keep in sync with the engine's server.PREVIEW_EDGE
// Scanning: a photo is only opened and processed after this long on it (or as
// soon as it's edited), so stepping through a shoot isn't slowed down.
const OPEN_DELAY_MS = 3000;
// The photo-switch progress bar: where it starts, and each engine step moves
// it this share of the way to PREP_DONE_AT (it fills when the photo's ready).
const PREP_START = 0.08;
const PREP_PREVIEW = 0.3; // the camera's preview is showing, the retouched version still to come
const PREP_STEP = 0.3;
const PREP_DONE_AT = 0.92;
import "./App.css";

// Off until the user raises Strength (or Brightness): no subject finding, the
// slow part on a computer without a graphics card, until it's wanted.
const DEFAULTS: BackdropParams = {
  strength: 0,
  smoothness: 1.5,
  evenness: 0,
  grain: 1,
  edge_protect: 0.4,
  exposure: 0,
};

const SLIDERS: {
  key: keyof BackdropParams;
  label: string;
  min: number;
  max: number;
  step: number;
  hint: string;
  format?: (v: number) => string;
}[] = [
  { key: "strength", label: "Strength", min: 0, max: 1, step: 0.01, hint: "Overall amount" },
  {
    key: "exposure",
    label: "Brightness",
    min: -2.5,
    max: 2.5,
    step: 0.05,
    hint: "Backdrop exposure in stops; the subject is untouched (e.g. turn a white backdrop grey)",
    format: (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(2)} EV`,
  },
  { key: "smoothness", label: "Crease size", min: 0.3, max: 5, step: 0.1, hint: "Largest crease or fold to remove" },
  { key: "evenness", label: "Evenness", min: 0, max: 1, step: 0.01, hint: "Even out blotchy shadows; keeps the light's falloff" },
  { key: "grain", label: "Grain", min: 0, max: 2, step: 0.05, hint: "Matched grain added back (1 = as shot)" },
  { key: "edge_protect", label: "Edge protection", min: 0.1, max: 2, step: 0.05, hint: "Width of the hair transition zone" },
];

// Off until asked for: faces aren't changed unless the photographer chooses to.
const EYE_DEFAULTS: EyesParams = {
  dark_circles: 0,
  eye_bags: 0,
  wrinkles: 0,
  whites: 0,
  iris: 0,
  iris_saturation: 0,
  iris_hue: 0,
  catchlight: 0,
  veins: 0,
  lashes: 0,
};

// Iris hue's colour scale until the photo's own irises are measured, as it
// moves blue eyes: teal-green to the left, violet to the right.
const IRIS_HUE_SCALE = "linear-gradient(to right, #3f9e7d, #3f8fa8, #4a78c0, #6c68c4, #9160bd)";
const EYE_SLIDERS: { key: keyof EyesParams; label: string; hint: string; centred?: boolean; scale?: string }[] = [
  {
    key: "dark_circles",
    label: "Dark circles",
    hint: "Lift under-eye darkness and colour toward the surrounding skin; texture is kept",
  },
  {
    key: "eye_bags",
    label: "Eye bags",
    hint: "Soften the bag's crease shadow and bulge (dodge & burn); 1 still keeps some shape",
  },
  {
    key: "wrinkles",
    label: "Wrinkles",
    hint: "Soften fine lines under the eyes and crow's feet; pores and skin texture are kept",
  },
  {
    key: "whites",
    label: "Eye whites",
    hint: "Clear redness, yellowing and veins from the whites; lashes and the inner corner are left alone",
  },
  {
    key: "veins",
    label: "Eye veins",
    hint: "Remove red veins from the whites; each takes the colour of the clean white beside it",
  },
  {
    key: "iris",
    label: "Iris",
    hint: "Bring out iris detail and colour; the pupil and the dark outer ring are kept",
  },
  {
    key: "iris_saturation",
    label: "Iris saturation",
    hint: "Left mutes the iris colour, right makes it richer; pupils, catchlights and whites are left alone",
    centred: true,
  },
  {
    key: "iris_hue",
    label: "Iris hue",
    hint: "Turns the iris colour around the colour wheel: blue eyes toward violet (right) or teal-green (left), brown toward gold (right) or red-brown (left). Pupils, catchlights and whites are left alone",
    centred: true,
    scale: IRIS_HUE_SCALE,
  },
  {
    key: "catchlight",
    label: "Catch light",
    hint: "Brighten and crisp up the existing catchlights (none are added)",
  },
  {
    key: "lashes",
    label: "Eyelashes",
    hint: "Deepen the lashes and add clarity; the lid skin and brows are left alone",
  },
];

// "compare": holding a panel's before button — the result without that step.
// "area": a Refine area brush — a tool's area (face, neck, body skin or
// clothes) shown in blue over the photo, corrected by painting.
type View = "result" | "before" | "mask" | "compare" | "area";
type Area = "face" | "neck" | "body" | "clothes" | "dodge_burn";
const AREA_NAMES: Record<Area, string> = {
  face: "Face skin",
  neck: "Neck skin",
  body: "Body skin",
  clothes: "Clothes",
  dodge_burn: "Dodge & Burn",
};
// Whose tools the blue area is, for the badge while refining it.
const AREA_TOOLS: Record<Area, string> = {
  face: "Skin",
  neck: "Skin",
  body: "Skin",
  clothes: "Clothes",
  dodge_burn: "Dodge & Burn",
};
type Step =
  | "removals"
  | "tone"
  | "backdrop"
  | "clothes"
  | "skin"
  | "skin_face"
  | "skin_neck"
  | "skin_body"
  | "dodge_burn"
  | "eyes"
  | "mouth"
  | "relight";
const STEP_NAMES: Record<Step, string> = {
  removals: "removals",
  backdrop: "backdrop",
  skin: "skin",
  skin_face: "face skin",
  skin_neck: "neck skin",
  skin_body: "body skin",
  eyes: "eyes",
  mouth: "mouth",
  clothes: "clothes",
  tone: "tone",
  dodge_burn: "dodge & burn",
  relight: "relight",
};

/** Press-and-hold handlers. The pointer is captured while pressed, so only
 * releasing it ends the hold: when zoomed in, the full-resolution tile landing
 * a few seconds later updated the page, which WebKit could report as the
 * pointer leaving the button, and the view reverted mid-hold. */
const holdHandlers = (start: () => void, end: () => void) => ({
  onPointerDown: (e: PointerEvent<HTMLButtonElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    start();
  },
  onPointerUp: end,
  onPointerCancel: end,
  onLostPointerCapture: end,
});

/** A look with one step left out, for that step's hold-for-before view. The
 * Skin panel compares one region at a time (Face, Neck or Body). */
const withoutStep = (look: Record<string, unknown>, step: Step): Record<string, unknown> => {
  if (step === "removals") return { ...look, removals: false };
  if (step === "skin_face" || step === "skin_neck" || step === "skin_body") {
    const region = step.slice("skin_".length);
    return { ...look, skin: { ...((look.skin as Record<string, unknown>) ?? {}), [region]: null } };
  }
  return { ...look, [step]: null };
};

// Brush modes: the Remove tools in the top bar (LaMa fill, glasses reflection, patch).
type BrushMode = "fill" | "reflection" | "patch";
const BRUSH_NAMES: Record<BrushMode, string> = { fill: "brush", reflection: "glasses", patch: "patch" };
const BRUSH_COLOURS: Record<BrushMode, string | undefined> = {
  fill: undefined, // PaintOverlay's red
  reflection: "rgba(64, 200, 255, 0.45)",
  patch: undefined,
};

type DodgeBurnParams = { contour: number; highlights: number; shadows: number };
const DODGE_BURN_DEFAULTS: DodgeBurnParams = { contour: 0, highlights: 0, shadows: 0 };
const DODGE_BURN_SLIDERS: { key: keyof DodgeBurnParams; label: string; hint: string }[] = [
  {
    key: "contour",
    label: "Contour",
    hint: "Shape the face: brighten the forehead, nose bridge, cheekbones and chin, and deepen under the cheekbones, beside the nose and along the jaw and temples",
  },
  { key: "highlights", label: "Highlights", hint: "Brighten the highlights already on the face" },
  { key: "shadows", label: "Shadows", hint: "Deepen the shadows already on the face: the contours, not the whole shaded side" },
];
const SKIN_REGION_DEFAULTS: SkinRegion = {
  acne: 0,
  blemishes: 0,
  smooth: 0,
  even: 0,
  shine: 0,
  texture: 0,
  pores: 0,
  forehead_lines: 0,
  frown_lines: 0,
  smile_lines: 0,
  chin_lines: 0,
  neck_lines: 0,
};
const WRINKLE_SLIDERS: { key: keyof SkinRegion; label: string; hint: string }[] = [
  { key: "forehead_lines", label: "Forehead lines", hint: "Soften horizontal lines across the forehead" },
  { key: "frown_lines", label: "Frown lines", hint: "Soften the vertical lines between the brows" },
  {
    key: "smile_lines",
    label: "Smile lines",
    hint: "Soften the folds from nose to mouth; even at 1 some fold is kept, as removing it looks unnatural",
  },
  { key: "chin_lines", label: "Chin lines", hint: "Soften lines around the chin and below the mouth corners" },
];
const SKIN_DEFAULTS: SkinParams = {
  face: SKIN_REGION_DEFAULTS,
  neck: SKIN_REGION_DEFAULTS,
  body: SKIN_REGION_DEFAULTS,
};
type SkinTab = keyof SkinParams;
const MOUTH_DEFAULTS: MouthParams = { lip_saturation: 0, lip_hue: 0, lip_smooth: 0, teeth_whiten: 0 };
// Lip hue's colour scale: cooler, pinker to the left, warmer, more coral to the right.
const LIP_HUE_SCALE = "linear-gradient(to right, #b8326f, #c73a52, #c8322f, #d9533a, #e8763c)";
const MOUTH_SLIDERS: {
  key: keyof MouthParams;
  label: string;
  hint: string;
  centred?: boolean;
  scale?: string;
}[] = [
  {
    key: "lip_hue",
    label: "Lip hue",
    hint: "Left makes the lips cooler and pinker, right warmer and more coral; their brightness is kept",
    centred: true,
    scale: LIP_HUE_SCALE,
  },
  {
    key: "lip_saturation",
    label: "Lip saturation",
    hint: "Left mutes the lips toward the skin's colour, right makes them richer",
    centred: true,
  },
  { key: "lip_smooth", label: "Lip smoothing", hint: "Soften dry lines, flakes and spots on the lips; the sheen is kept" },
  { key: "teeth_whiten", label: "Teeth whitening", hint: "Take out yellow and brighten the teeth a little; gums and lips are left alone" },
];
// Crop shapes; "Original" is the photo's own.
const CROP_ASPECTS = ["Free", "Original", "1:1", "4:5", "5:4", "2:3", "3:2", "16:9"];
/** Width / height for a crop shape, or null for free. */
function aspectRatio(aspect: string, width: number, height: number): number | null {
  if (aspect === "Free") return null;
  if (aspect === "Original") return width / height;
  const [w, h] = aspect.split(":").map(Number);
  return w / h;
}

/** The backdrop step's settings for the engine, or null to skip it: in
 * Outdoor mode, or while it would change nothing (no Strength, no
 * Brightness), so the subject isn't looked for until it's needed. */
function backdropLook(outdoor: boolean, p: BackdropParams): BackdropParams | null {
  return outdoor || (p.strength === 0 && p.exposure === 0) ? null : p;
}

const ALL_DEFAULTS: PhotoSettings = {
  backdrop: DEFAULTS,
  outdoor: false,
  eyes: EYE_DEFAULTS,
  skin: SKIN_DEFAULTS,
  mouth: MOUTH_DEFAULTS,
  dodgeBurn: DODGE_BURN_DEFAULTS,
  creases: 0,
  tone: TONE_DEFAULTS,
  crop: NO_CROP,
  opacity: OPACITY_DEFAULTS,
  relight: null,
  off: [],
};
/** A preset's settings as full PhotoSettings: anything the preset doesn't have
 * (a slider added since it was saved) takes its default. */
function withPreset(
  saved: Record<string, unknown>,
  outdoor: boolean,
  crop: Crop = NO_CROP,
  relight: Relight | null = null,
): PhotoSettings {
  const obj = (v: unknown) => (v && typeof v === "object" ? (v as Record<string, unknown>) : {});
  const num = (v: unknown, d: number) => (typeof v === "number" ? v : d);
  const skin = obj(saved.skin);
  const region = (defaults: SkinRegion, savedRegion: unknown): SkinRegion =>
    ({ ...defaults, ...obj(savedRegion) }) as SkinRegion;
  const tone = obj(saved.tone);
  return {
    backdrop: { ...ALL_DEFAULTS.backdrop, ...obj(saved.backdrop) },
    outdoor,
    eyes: { ...ALL_DEFAULTS.eyes, ...obj(saved.eyes) },
    skin: {
      face: region(ALL_DEFAULTS.skin.face, skin.face),
      neck: region(ALL_DEFAULTS.skin.neck, skin.neck),
      body: region(ALL_DEFAULTS.skin.body, skin.body),
    },
    mouth: { ...ALL_DEFAULTS.mouth, ...obj(saved.mouth) },
    // Presets saved before the split had one Sculpt number: it was the
    // strength of today's Highlights and Shadows together.
    dodgeBurn:
      typeof saved.dodgeBurn === "number"
        ? { ...DODGE_BURN_DEFAULTS, highlights: saved.dodgeBurn, shadows: saved.dodgeBurn }
        : { ...DODGE_BURN_DEFAULTS, ...obj(saved.dodgeBurn) },
    creases: num(saved.creases, 0),
    tone: {
      temperature: num(tone.temperature, 0),
      tint: num(tone.tint, 0),
      ev: num(tone.ev, 0),
      dehaze: num(tone.dehaze, 0),
      vibrance: num(tone.vibrance, 0),
      curve: Array.isArray(tone.curve) ? (tone.curve as Point[]) : IDENTITY,
    },
    crop,
    opacity: { ...OPACITY_DEFAULTS, ...(obj(saved.opacity) as Partial<Opacity>) },
    relight,
    off: Array.isArray(saved.off) ? saved.off.filter((k): k is string => typeof k === "string") : [],
  };
}

/** ``st`` as the engine should see it: each slider switched off by its eye
 * at its default (its own value is kept, for switching it back on). */
function withOff(st: PhotoSettings): PhotoSettings {
  if (!st.off?.length) return st;
  const out: PhotoSettings = {
    ...st,
    skin: { face: { ...st.skin.face }, neck: { ...st.skin.neck }, body: { ...st.skin.body } },
    eyes: { ...st.eyes },
    mouth: { ...st.mouth },
    dodgeBurn: { ...st.dodgeBurn },
    relight: st.relight && { ...st.relight },
  };
  for (const id of st.off) {
    const [group, a, b] = id.split(".");
    if (group === "skin" && (a === "face" || a === "neck" || a === "body") && b in SKIN_REGION_DEFAULTS) {
      out.skin[a][b as keyof SkinRegion] = SKIN_REGION_DEFAULTS[b as keyof SkinRegion];
    } else if (group === "eyes" && a in EYE_DEFAULTS) {
      out.eyes[a as keyof EyesParams] = EYE_DEFAULTS[a as keyof EyesParams];
    } else if (group === "mouth" && a in MOUTH_DEFAULTS) {
      out.mouth[a as keyof MouthParams] = MOUTH_DEFAULTS[a as keyof MouthParams];
    } else if (group === "dodgeBurn" && a in DODGE_BURN_DEFAULTS) {
      out.dodgeBurn[a as keyof DodgeBurnParams] = 0;
    } else if (group === "relight" && out.relight && (a === "exposure" || a === "warmth")) {
      out.relight[a] = 0;
    } else if (group === "creases") {
      out.creases = 0;
    }
  }
  return out;
}

const SKIN_TABS: { key: SkinTab; label: string }[] = [
  { key: "face", label: "Face" },
  { key: "neck", label: "Neck" },
  { key: "body", label: "Body" },
];
const SKIN_SLIDERS: {
  key: keyof SkinRegion;
  label: string;
  hint: string;
  min: number;
  centred?: boolean;
}[] = [
  {
    key: "acne",
    label: "Acne",
    hint: "Heal spots and acne: low takes only the clearest, high fainter ones too. Moles and pores are kept",
    min: 0,
  },
  {
    key: "blemishes",
    label: "Blemishes",
    hint: "Even out large pores and small marks, each to the skin around it; the finest texture is kept. Moles are kept",
    min: 0,
  },
  { key: "smooth", label: "Smooth", hint: "Even out blotchy light and shade; pores and fine texture are kept", min: 0 },
  {
    key: "texture",
    label: "Texture",
    hint: "Soften pores, most visible ones most; the skin's finest grain is kept so it never looks plastic",
    min: 0,
  },
  {
    key: "pores",
    label: "Pores",
    hint: "Take pores out, pits and raised bumps alike, even where they catch the light; the finest grain is kept. Stronger than Texture",
    min: 0,
  },
  {
    key: "even",
    label: "Even tone",
    hint: "Move red or blotchy patches toward the skin's own tone (on the neck and body, toward that area's average, so a made-up face isn't copied onto them)",
    min: 0,
  },
  {
    key: "shine",
    label: "Shine",
    hint: "Left for matte, right for gloss, middle for natural",
    min: -1,
    centred: true,
  },
];

// Linux file dialogs match extensions case-sensitively (cameras write "P1167822.RW2"),
// so every filter lists both cases.
const bothCases = (exts: string[]) => exts.flatMap((e) => [e, e.toUpperCase()]);

const IMAGE_EXTENSIONS = ["tif", "tiff", "jpg", "jpeg", "png"];
// Developed by LibRaw in the engine; keep in sync with engine imageio.RAW_EXTENSIONS.
const RAW_EXTENSIONS = [
  "rw2", "nef", "nrw", "cr2", "cr3", "crw", "arw", "srf", "sr2", "raf", "orf", "dng",
  "pef", "srw", "rwl", "3fr", "fff", "iiq", "erf", "mef", "mos", "x3f", "kdc", "dcr",
];

// Brush radius as a fraction of the image's long edge.
const COLLAPSED_KEY = "rapidretouch.collapsedPanels";
// Hold-for-before steps in the sidebar's order, top to bottom.
const PANEL_STEPS: Step[] = ["removals", "tone", "backdrop", "skin", "dodge_burn", "relight", "eyes", "mouth", "clothes"];
const PANEL_IDS = ["tone", "backdrop", "skin", "dodge_burn", "relight", "eyes", "mouth", "clothes"];

// Zoomed in: wait this long after the preview last changed before asking for
// the sharp full-resolution tile (see the detail effect).
const DETAIL_AFTER_EDIT_MS = 400;
const SLOW_MS = 300; // an update taking longer than this shows the circle over the photo
const BRUSH = { min: 0.002, max: 0.05, step: 0.001, default: 0.01 };
const clampBrush = (r: number) => Math.min(BRUSH.max, Math.max(BRUSH.min, r));

// Remembered across launches; storage can be unavailable, so never let it throw.
const LAST_DIR_KEY = "rapidretouch.lastDir";
const readLastDir = (): string | null => {
  try {
    return localStorage.getItem(LAST_DIR_KEY);
  } catch {
    return null;
  }
};
const saveLastDir = (dir: string) => {
  try {
    localStorage.setItem(LAST_DIR_KEY, dir);
  } catch {
    // not remembered; harmless
  }
};

function errorMessage(e: unknown): string {
  if (e && typeof e === "object" && "message" in e) return String((e as EngineError).message);
  return String(e);
}

function ResetButton({
  disabled,
  onClick,
  what,
}: {
  disabled: boolean;
  onClick: () => void;
  what: string;
}) {
  return (
    <button
      className="reset-button"
      disabled={disabled}
      onClick={onClick}
      title={`Reset ${what} to defaults`}
    >
      <RotateCcw size={13} />
    </button>
  );
}

/** A sidebar panel that folds away to its header. The header is its title
 * (or ``head``, for a panel whose header is a control) plus ``actions``. */
function Panel({
  title,
  head,
  actions,
  open,
  onToggle,
  children,
}: {
  title?: string;
  head?: React.ReactNode;
  actions?: React.ReactNode;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <section className={`panel${open ? "" : " panel--collapsed"}`}>
      <div className="panel__head">
        <button
          className="panel__toggle"
          onClick={onToggle}
          aria-expanded={open}
          title={open ? "Collapse" : "Expand"}
        >
          <ChevronRight size={14} className="panel__chevron" />
          {title && <h2>{title}</h2>}
        </button>
        {head}
        {actions}
      </div>
      {open && children}
    </section>
  );
}

export default function App() {
  const [engineReady, setEngineReady] = useState(false);
  const [status, setStatus] = useState("Starting the AI engine…");
  const [error, setError] = useState<string | null>(null);
  const [image, setImage] = useState<(OpenResult & { path: string }) | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [mask, setMask] = useState<string | null>(null);
  const [view, setView] = useState<View>("result");
  const [params, setParams] = useState<BackdropParams>(DEFAULTS);
  const [eyesParams, setEyesParams] = useState<EyesParams>(EYE_DEFAULTS);
  const [skinParams, setSkinParams] = useState<SkinParams>(SKIN_DEFAULTS);
  const [mouthParams, setMouthParams] = useState<MouthParams>(MOUTH_DEFAULTS);
  const [dodgeBurn, setDodgeBurn] = useState<DodgeBurnParams>(DODGE_BURN_DEFAULTS);
  const [creases, setCreases] = useState(0);
  // Folded sidebar panels, remembered between sessions.
  const [collapsed, setCollapsed] = useState<Set<string>>(() => {
    try {
      return new Set(JSON.parse(localStorage.getItem(COLLAPSED_KEY) ?? "[]"));
    } catch {
      return new Set();
    }
  });
  const saveCollapsed = (next: Set<string>) => {
    try {
      localStorage.setItem(COLLAPSED_KEY, JSON.stringify([...next]));
    } catch {
      // storage unavailable: folding still works for this session
    }
    return next;
  };
  const panel = (id: string) => ({
    open: !collapsed.has(id),
    onToggle: () =>
      setCollapsed((c) => {
        const next = new Set(c);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return saveCollapsed(next);
      }),
  });
  const allCollapsed = PANEL_IDS.every((id) => collapsed.has(id));
  const toggleAllPanels = () =>
    setCollapsed(saveCollapsed(allCollapsed ? new Set() : new Set(PANEL_IDS)));
  const [toneParams, setToneParams] = useState<Tone>(TONE_DEFAULTS);
  const [skinTab, setSkinTab] = useState<SkinTab>("face");
  // Opening the Skin panel always starts on Face, the tab used most.
  const skinOpen = !collapsed.has("skin");
  useEffect(() => {
    if (skinOpen) setSkinTab("face");
  }, [skinOpen]);
  const [faces, setFaces] = useState<number | null>(null);
  const [maskModel, setMaskModel] = useState<ModelInfo | null>(null);
  const [licencePrompt, setLicencePrompt] = useState<EngineError | null>(null);
  const [busy, setBusy] = useState(false);
  // The photo's lightness, behind the tone curve (from each render).
  const [histogram, setHistogram] = useState<number[] | null>(null);
  // A circle over the photo once an update has taken a moment (none for the
  // quick ones, so dragging a fast slider doesn't flicker it).
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    if (!busy) {
      setSlow(false);
      return;
    }
    const t = window.setTimeout(() => setSlow(true), SLOW_MS);
    return () => window.clearTimeout(t);
  }, [busy]);
  const [brushOn, setBrushOn] = useState(false);
  const [brushRadius, setBrushRadius] = useState(BRUSH.default);
  // The mask brush has its own size: mask edits (long soft strokes along hair)
  // usually want a different size from spot removals.
  const [maskBrushRadius, setMaskBrushRadius] = useState(BRUSH.default * 2);
  const [removals, setRemovals] = useState(0);
  // How many removals each Remove tool has made on this photo (its Clear button).
  const [removalKinds, setRemovalKinds] = useState<Record<string, number>>({});
  const [brushMode, setBrushMode] = useState<BrushMode>("fill");
  const [reflectionStrength, setReflectionStrength] = useState(1);
  const [removing, setRemoving] = useState(false);
  const [maskMode, setMaskMode] = useState<"add" | "subtract">("add");
  const [maskEdits, setMaskEdits] = useState(0);
  const [maskPending, setMaskPending] = useState(false);
  const [aboutModels, setAboutModels] = useState<ModelInfo[] | null>(null);
  const [area, setArea] = useState<Area | null>(null);
  const [areaImg, setAreaImg] = useState<string | null>(null);
  const [areaEdits, setAreaEdits] = useState(0);
  const [areaMode, setAreaMode] = useState<"add" | "remove">("remove");
  const [areaPending, setAreaPending] = useState(false);
  const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null);
  const [compareImg, setCompareImg] = useState<string | null>(null);
  const [compareStep, setCompareStep] = useState<Step | null>(null);
  const [zoomTool, setZoomTool] = useState(false);
  const [zoomLabel, setZoomLabel] = useState("Fit");
  const [detail, setDetail] = useState<Detail | null>(null);
  // A zoomed-in view's full-resolution detail is being made: until it comes,
  // the viewer shows the preview enlarged, too soft to show the retouching.
  const [detailLoading, setDetailLoading] = useState(false);
  const viewerRef = useRef<ViewerHandle>(null);
  // Film strip: every photo opened this session, each with its own settings.
  const [strip, setStrip] = useState<StripItem[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [copied, setCopied] = useState<PhotoSettings | null>(null);
  const [presets, setPresets] = useState<Preset[]>([]);
  const settingsByPath = useRef(new Map<string, PhotoSettings>());
  // Switching photo: a progress bar (on its thumbnail and along the top of the
  // viewer) while it's prepared, nudged on by each step the engine reports.
  const [prep, setPrep] = useState<{ path: string; fraction: number } | null>(null);
  // First-launch setup of the AI engine (released app only), while it runs.
  const [setup, setSetup] = useState<{ message: string; fraction: number | null; error: string | null } | null>(
    null,
  );
  const [tutorialUrl, setTutorialUrl] = useState<string | null>(() => {
    try {
      return localStorage.getItem(TUTORIAL_URL_KEY);
    } catch {
      return null;
    }
  });
  useEffect(() => {
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => ctrl.abort(), LINKS_TIMEOUT_MS);
    fetch(LINKS_URL, { signal: ctrl.signal, cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((links: { tutorial_video?: unknown }) => {
        const url = typeof links.tutorial_video === "string" ? links.tutorial_video.trim() : "";
        const valid = /^https:\/\//.test(url) ? url : null;
        setTutorialUrl(valid);
        try {
          if (valid) localStorage.setItem(TUTORIAL_URL_KEY, valid);
          else localStorage.removeItem(TUTORIAL_URL_KEY);
        } catch {
          // storage unavailable: the link is just read again next start
        }
      })
      .catch(() => undefined) // offline or unreachable: keep the remembered link
      .finally(() => window.clearTimeout(timer));
    return () => ctrl.abort();
  }, []);
  const [tutorialDismissed, setTutorialDismissed] = useState(() => {
    try {
      return localStorage.getItem(TUTORIAL_DISMISSED_KEY) === "1";
    } catch {
      return false;
    }
  });

  const dismissTutorial = () => {
    setTutorialDismissed(true);
    try {
      localStorage.setItem(TUTORIAL_DISMISSED_KEY, "1");
    } catch {
      // storage unavailable: dismissal just won't persist across restarts
    }
  };
  const compareToken = useRef(0);

  // Only one preview render at a time; while one runs, remember that the
  // sliders moved and render once more with the latest values afterwards.
  const paramsRef = useRef(params);
  const eyesRef = useRef(eyesParams);
  const skinRef = useRef(skinParams);
  const mouthRef = useRef(mouthParams);
  const dodgeBurnRef = useRef(dodgeBurn);
  const creasesRef = useRef(creases);
  const toneRef = useRef(toneParams);
  const [crop, setCrop] = useState<Crop>(NO_CROP);
  const [opacity, setOpacity] = useState<Opacity>(OPACITY_DEFAULTS);
  const opacityRef = useRef(opacity);
  const [relight, setRelight] = useState<Relight | null>(null);
  const relightRef = useRef(relight);
  // Sliders switched off by their eye (see withOff).
  const [off, setOff] = useState<string[]>([]);
  const offRef = useRef(off);
  // Showing the ellipse's handles to move, stretch and turn it.
  const [relightEditing, setRelightEditing] = useState(false);
  // Iris hue's colour bar for the open photo's eyes (null: the blue-eye default).
  const [irisScale, setIrisScale] = useState<string[] | null>(null);
  const irisScaleFor = useRef<string | null>(null);
  const [exportRows, setExportRows] = useState<ExportRow[] | null>(null);
  const [exportRunning, setExportRunning] = useState(false);
  const [exportCancelling, setExportCancelling] = useState(false);
  const exportCancel = useRef(false);
  const exportingPath = useRef<string | null>(null);
  // Exports being written in the background, by output path: settled by the
  // engine's "written" event.
  const pendingWrites = useRef(new Map<string, (error?: string) => void>());
  // "written" events that came in before their export call returned (a small
  // file can be written that fast): picked up when the call does.
  const writtenEarly = useRef(new Map<string, string | undefined>());
  const [exportFormat, setExportFormat] = useState<ExportFormat>(() => {
    try {
      return localStorage.getItem(EXPORT_FORMAT_KEY) === "jpeg" ? "jpeg" : "tiff";
    } catch {
      return "tiff";
    }
  });
  const [exportFolder, setExportFolder] = useState<string | null>(null);
  const [exportSize, setExportSize] = useState<ExportSize>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(EXPORT_SIZE_KEY) ?? "null");
      if (saved && (saved.mode === "full" || saved.mode === "long") && typeof saved.px === "number") return saved;
    } catch {
      // unreadable: the default
    }
    return { mode: "full", px: 2048 };
  });
  const cropRef = useRef(crop);
  const [cropping, setCropping] = useState(false);
  const [cropAspect, setCropAspect] = useState<string>("Free");
  // Where the crop bar's been dragged to (by its grip), in the viewer; null:
  // its default place, centred near the top.
  const [cropBarPos, setCropBarPos] = useState<{ left: number; top: number } | null>(null);
  const cropBarRef = useRef<HTMLDivElement>(null);
  const dragCropBar = (e: PointerEvent<HTMLElement>) => {
    const bar = cropBarRef.current;
    const area = bar?.offsetParent as HTMLElement | null;
    if (!bar || !area) return;
    e.preventDefault();
    const grip = e.currentTarget;
    grip.setPointerCapture(e.pointerId);
    const start = { x: e.clientX, y: e.clientY, left: bar.offsetLeft, top: bar.offsetTop };
    const move = (ev: globalThis.PointerEvent) => {
      const left = Math.min(Math.max(0, start.left + ev.clientX - start.x), area.clientWidth - bar.offsetWidth);
      const top = Math.min(Math.max(0, start.top + ev.clientY - start.y), area.clientHeight - bar.offsetHeight);
      setCropBarPos({ left, top });
    };
    const up = () => {
      grip.removeEventListener("pointermove", move);
      grip.removeEventListener("pointerup", up);
      grip.removeEventListener("pointercancel", up);
    };
    grip.addEventListener("pointermove", move);
    grip.addEventListener("pointerup", up);
    grip.addEventListener("pointercancel", up);
  };
  const [outdoor, setOutdoor] = useState(false);
  // Photos whose Backdrop/Outdoor mode was set by detection, not by the user
  // (shown as "auto"); any click on the mode buttons makes it the user's.
  const autoMode = useRef(new Set<string>());
  // Photos whose studio/outdoor detection has come in. Until it has, a photo
  // whose mode isn't the user's choice gets no backdrop work (see backdropArg):
  // clicked too soon, an outdoor shot was smoothed as a backdrop, then put back.
  const detectedScene = useRef(new Set<string>());
  const modeUnknown = (path: string) =>
    !detectedScene.current.has(path) && (autoMode.current.has(path) || !settingsByPath.current.has(path));
  const [modeIsAuto, setModeIsAuto] = useState(false);
  const outdoorRef = useRef(outdoor);
  /** Backdrop settings to send: none when the photo is set to Outdoor. */
  const backdropArg = () => {
    const path = activePath.current;
    if (path && modeUnknown(path)) return null; // studio or outdoor? not known yet
    return backdropLook(outdoorRef.current, paramsRef.current);
  };
  const inFlight = useRef(false);
  const dirty = useRef(false);

  /** Every tool's settings for an engine call: the photo's whole look. */
  const lookArgs = (): Record<string, unknown> => {
    const st = withOff(currentSettings());
    return {
      backdrop: backdropArg(),
      eyes: st.eyes,
      skin: st.skin,
      mouth: st.mouth,
      dodge_burn: sameValues(st.dodgeBurn, DODGE_BURN_DEFAULTS) ? null : st.dodgeBurn,
      clothes: st.creases > 0 ? { creases: st.creases } : null,
      tone: sameValues(toneRef.current, TONE_DEFAULTS) ? null : toneRef.current,
      opacity: opacityRef.current,
      relight: relightLook(st.relight),
    };
  };
  /** The same, for any photo's settings (batch export). */
  const lookArgsFor = (photo: PhotoSettings): Record<string, unknown> => {
    const st = withOff(photo);
    return {
    backdrop: backdropLook(st.outdoor, st.backdrop),
    eyes: st.eyes,
    skin: st.skin,
    mouth: st.mouth,
    dodge_burn: sameValues(st.dodgeBurn, DODGE_BURN_DEFAULTS) ? null : st.dodgeBurn,
    clothes: st.creases > 0 ? { creases: st.creases } : null,
    tone: sameValues(st.tone, TONE_DEFAULTS) ? null : st.tone,
    opacity: st.opacity,
    relight: relightLook(st.relight),
    };
  };

  useEffect(() => {
    const unlisten = onEngineEvent((e) => {
      if (e.event === "status") {
        setStatus(e.message);
        setPrep((p) => (p ? { ...p, fraction: p.fraction + (PREP_DONE_AT - p.fraction) * PREP_STEP } : p));
      } else if (e.event === "setup") {
        if (e.error) setSetup({ message: "", fraction: null, error: e.error });
        else if (e.fraction === 1) {
          // The engine was just set up: a first launch or an update. About
          // opens by itself then (and only then), in front: the long first
          // download may have left the window behind others.
          setSetup(null);
          showAbout().then(() => bringToFront());
        }
        else setSetup({ message: e.message ?? "", fraction: e.fraction ?? null, error: null });
      } else if (e.event === "written") {
        const settle = pendingWrites.current.get(e.path);
        pendingWrites.current.delete(e.path);
        if (settle) settle(e.error);
        else writtenEarly.current.set(e.path, e.error);
      } else if (e.event === "progress") {
        const path = exportingPath.current;
        if (path)
          setExportRows((rows) =>
            rows && rows.map((r) => (r.path === path ? { ...r, fraction: Math.max(r.fraction, e.fraction) } : r)),
          );
      } else if (e.event === "stopped") {
        setEngineReady(false);
        setError("The engine stopped. Restart the app.");
      }
    });
    // Ask rather than wait for the engine's "ready" event, which can fire
    // before this page has started listening. Requests queue until it's up.
    call<{ device: string; cuda: boolean }>("ping")
      .then((r) => {
        setEngineReady(true);
        call<Preset[]>("presets").then(setPresets).catch(() => undefined);
        setStatus(r.device !== "CPU" ? `Ready · ${r.device}` : "Ready · no GPU found, running on CPU (slower)");
      })
      .catch((e) => setError(`The engine didn't start: ${errorMessage(e)}`));
    return () => {
      unlisten.then((f) => f());
    };
  }, []);

  useEffect(() => {
    if (!engineReady) return;
    call<ModelInfo[]>("models").then((models) =>
      setMaskModel(models.find((m) => m.task === "subject_mask" && m.default) ?? null),
    );
  }, [engineReady]);

  const handleError = useCallback((e: unknown) => {
    const err = e as EngineError;
    if (err && err.kind === "licence") setLicencePrompt(err);
    else setError(errorMessage(e));
  }, []);

  // Which photo is being looked at, and which one the engine has open. While
  // scanning they differ: the viewer shows the camera's embedded preview and
  // nothing is processed until ensureOpen() (the timer, or any edit).
  const activePath = useRef<string | null>(null);
  const openedPath = useRef<string | null>(null);
  const openedInfo = useRef<(OpenResult & { path: string }) | null>(null);
  const opening = useRef<{ path: string; done: Promise<boolean> } | null>(null);
  const openTimer = useRef<number | undefined>(undefined);
  // "Retouching in 3, 2, 1" in the footer while a photo waits to be opened.
  const [countdown, setCountdown] = useState<number | null>(null);
  const countdownTimer = useRef<number | undefined>(undefined);
  const stopCountdown = () => {
    window.clearInterval(countdownTimer.current);
    setCountdown(null);
  };
  const startCountdown = () => {
    window.clearInterval(countdownTimer.current);
    setCountdown(Math.round(OPEN_DELAY_MS / 1000));
    countdownTimer.current = window.setInterval(
      () => setCountdown((c) => (c !== null && c > 1 ? c - 1 : c)),
      1000,
    );
  };

  /** Open the photo being looked at in the engine, if it isn't already.
   * Concurrent callers share one open. Returns false if it failed. */
  const ensureOpen = useCallback(async (): Promise<boolean> => {
    const path = activePath.current;
    if (!path) return false;
    if (openedPath.current === path) return true;
    if (opening.current?.path === path) return opening.current.done;
    window.clearTimeout(openTimer.current);
    stopCountdown();
    const done = (async () => {
      try {
        const r = await call<OpenResult>("open", { path });
        openedPath.current = path;
        openedInfo.current = { ...r, path };
        if (activePath.current === path) {
          setImage({ ...r, path });
          setRemovals(r.removals);
          setRemovalKinds(r.removal_kinds ?? {});
          setMaskEdits(r.mask_edits);
        }
        saveLastDir(path.replace(/\/[^/]*$/, ""));
        return true;
      } catch (e) {
        handleError(e);
        return false;
      } finally {
        if (opening.current?.path === path) opening.current = null;
      }
    })();
    opening.current = { path, done };
    return done;
  }, [handleError]);

  const render = useCallback(async () => {
    if (inFlight.current) {
      dirty.current = true;
      return;
    }
    inFlight.current = true;
    setBusy(true);
    const path = activePath.current;
    if (path && openedPath.current !== path) setPrep((p) => (p?.path === path ? p : { path, fraction: PREP_START }));
    try {
      if (!(await ensureOpen())) return;
      do {
        dirty.current = false;
        const forPath = openedPath.current;
        const r = await call<{ preview: string; faces: number | null; histogram: number[] | null }>("render", {
          ...lookArgs(),
        });
        // Moved on to another photo meanwhile: don't show this one's result.
        if (forPath !== activePath.current) break;
        setResult(r.preview);
        setHistogram(r.histogram ?? null);
        setPrep((p) => (p?.path === forPath ? null : p)); // the retouched version is on screen
        setFaces(r.faces);
      } while (dirty.current);
      setStatus("Ready");
      // Iris hue's colour bar, from this photo's own irises (once per photo).
      const forPath = openedPath.current;
      if (forPath && irisScaleFor.current !== forPath) {
        irisScaleFor.current = forPath;
        call<{ colours: string[] | null }>("iris_scale")
          .then(({ colours }) => {
            if (activePath.current === forPath) setIrisScale(colours);
          })
          .catch(() => undefined);
      }
    } catch (e) {
      handleError(e);
    } finally {
      setPrep((p) => (p?.path === path ? null : p));
      inFlight.current = false;
      setBusy(false);
    }
  }, [handleError, ensureOpen]);

  const updateParam = (key: keyof BackdropParams, value: number) => {
    const next = { ...paramsRef.current, [key]: value };
    paramsRef.current = next;
    setParams(next);
    render();
  };

  const updateSkin = (region: SkinTab, key: keyof SkinRegion, value: number) => {
    const next = { ...skinRef.current, [region]: { ...skinRef.current[region], [key]: value } };
    skinRef.current = next;
    setSkinParams(next);
    render();
  };

  // Reset buttons beside each group's heading.
  const toggleOutdoor = () => {
    const path = activePath.current;
    if (path) {
      // The user's choice: a detection still to come mustn't override it, and
      // the backdrop needn't wait for it.
      autoMode.current.delete(path);
      detectedScene.current.add(path);
    }
    setModeIsAuto(false);
    const next = !outdoorRef.current;
    outdoorRef.current = next;
    setOutdoor(next);
    if (next && (view === "mask" || view === "area")) setView("result");
    render();
  };

  const resetBackdrop = () => {
    paramsRef.current = DEFAULTS;
    setParams(DEFAULTS);
    render();
  };
  const updateOpacity = (group: keyof Opacity, value: number, rerender = true) => {
    const next = { ...opacityRef.current, [group]: value };
    opacityRef.current = next;
    setOpacity(next);
    if (rerender) render();
  };
  const resetSkin = () => {
    skinRef.current = SKIN_DEFAULTS;
    setSkinParams(SKIN_DEFAULTS);
    updateOpacity("skin", 1, false);
    render();
  };
  const resetWrinkles = () => {
    const face = { ...skinRef.current.face };
    for (const s of WRINKLE_SLIDERS) face[s.key] = 0;
    const next = { ...skinRef.current, face };
    skinRef.current = next;
    setSkinParams(next);
    render();
  };
  const resetEyes = () => {
    eyesRef.current = EYE_DEFAULTS;
    setEyesParams(EYE_DEFAULTS);
    updateOpacity("eyes", 1, false);
    render();
  };

  const updateMouth = (key: keyof MouthParams, value: number) => {
    const next = { ...mouthRef.current, [key]: value };
    mouthRef.current = next;
    setMouthParams(next);
    render();
  };
  const updateTone = (next: Tone) => {
    toneRef.current = next;
    setToneParams(next);
    render();
  };

  /** A slider's eye: switch it off (its value is kept) or back on. */
  const toggleOff = (id: string, on?: boolean) => {
    const isOff = offRef.current.includes(id);
    if (on === true && !isOff) return;
    const next = isOff ? offRef.current.filter((k) => k !== id) : [...offRef.current, id];
    offRef.current = next;
    setOff(next);
    render();
  };
  /** The eye props for a slider: ``id`` as in withOff. */
  const eye = (id: string) => ({
    off: off.includes(id),
    onToggleOff: () => toggleOff(id),
  });

  const updateRelight = (next: Relight | null) => {
    relightRef.current = next;
    setRelight(next);
    if (!next) setRelightEditing(false);
    render();
  };
  /** A new relight, over the face (the engine finds it). */
  const addRelight = async () => {
    try {
      if (!(await ensureOpen())) return;
      const shape = await call<RelightShape>("relight_default");
      setBrushOn(false);
      setRelightEditing(true);
      updateRelight({ ...shape, ...RELIGHT_LOOK });
    } catch (e) {
      handleError(e);
    }
  };

  // Crop only changes the export, so there's nothing to re-render.
  const updateCrop = (next: Crop) => {
    cropRef.current = next;
    setCrop(next);
  };
  const startCropping = () => {
    setBrushOn(false);
    setZoomTool(false);
    setView("result");
    setCropping(true);
  };

  const updateCreases = (value: number) => {
    creasesRef.current = value;
    setCreases(value);
    render();
  };

  const updateDodgeBurn = (key: keyof DodgeBurnParams, value: number) => {
    const next = { ...dodgeBurnRef.current, [key]: value };
    dodgeBurnRef.current = next;
    setDodgeBurn(next);
    render();
  };
  const resetDodgeBurn = () => {
    dodgeBurnRef.current = DODGE_BURN_DEFAULTS;
    setDodgeBurn(DODGE_BURN_DEFAULTS);
    updateOpacity("dodge_burn", 1, false);
    render();
  };

  const resetMouth = () => {
    mouthRef.current = MOUTH_DEFAULTS;
    setMouthParams(MOUTH_DEFAULTS);
    updateOpacity("mouth", 1, false);
    render();
  };

  const updateEyes = (key: keyof EyesParams, value: number) => {
    const next = { ...eyesRef.current, [key]: value };
    eyesRef.current = next;
    setEyesParams(next);
    render();
  };

  // Removal calls return the whole pipeline's preview (removals, then backdrop).
  const removalCall = useCallback(
    async (method: string, extra: Record<string, unknown> = {}) => {
      setRemoving(true);
      setBusy(true);
      try {
        if (!(await ensureOpen())) return;
        const r = await call<{ preview: string; removals: number; removal_kinds: Record<string, number> }>(method, {
          ...extra,
          ...lookArgs(),
        });
        setResult(r.preview);
        setRemovals(r.removals);
        setRemovalKinds(r.removal_kinds ?? {});
        if (openedInfo.current) {
          openedInfo.current = { ...openedInfo.current, removals: r.removals, removal_kinds: r.removal_kinds ?? {} };
        }
        setStatus("Ready");
      } catch (e) {
        handleError(e);
      } finally {
        setRemoving(false);
        setBusy(false);
      }
    },
    [handleError, ensureOpen],
  );

  const onStroke = (points: [number, number][], radius: number) =>
    removalCall(
      "remove",
      brushMode === "reflection"
        ? { points, radius, kind: "reflection", strength: reflectionStrength }
        : { points, radius },
    );

  const onPatch = (outline: [number, number][], offset: [number, number]) =>
    removalCall("remove", { points: outline, radius: 0, kind: "patch", offset });

  // The tool buttons pick the brush mode; clicking the active one turns it off.
  // The top bar shows one tool at a time while it's in use, with its controls.
  const activeTool: "crop" | BrushMode | null = cropping
    ? "crop"
    : brushOn && view === "result"
      ? brushMode
      : null;
  const pickBrush = (mode: BrushMode) => {
    setBrushOn((on) => !(on && brushMode === mode));
    setBrushMode(mode);
    setZoomTool(false);
    setRelightEditing(false);
  };
  const undoRemove = () => removalCall("undo_remove");
  // Clear: only the selected Remove tool's removals (Undo still steps back through all of them).
  const clearRemovals = () => removalCall("clear_removals", { kind: brushMode });

  // Keyboard: Ctrl+Z undoes a removal, B toggles the brush, [ and ] resize it.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Crop mode: Enter (or Esc) is Done, even straight after using the
      // Straighten slider, which still has the keyboard's focus.
      if (image && cropping && (e.key === "Enter" || e.key === "Escape")) {
        e.preventDefault();
        setCropping(false);
        return;
      }
      // Esc closes the Brush or Glasses tool (with Patch it drops the selection).
      if (brushOn && brushMode !== "patch" && view === "result" && e.key === "Escape") {
        e.preventDefault();
        setBrushOn(false);
        return;
      }
      if (relightEditing && (e.key === "Enter" || e.key === "Escape")) {
        e.preventDefault();
        setRelightEditing(false);
        return;
      }
      if (!image || e.target instanceof HTMLInputElement) return;
      if (cropping) {
        if (e.key === "c" || e.key === "C") {
          e.preventDefault();
          setCropping(false);
        }
        return;
      }
      if (!e.ctrlKey && !e.metaKey && (e.key === "c" || e.key === "C") && view !== "mask" && view !== "area") {
        startCropping();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (view === "mask") {
          if (maskEdits > 0 && !maskPending) undoMaskEdit();
        } else if (view === "area") {
          if (areaEdits > 0 && !areaPending) undoAreaEdit();
        } else if (removals > 0 && !removing) undoRemove();
      } else if (e.key === "\\" && view !== "mask" && view !== "area") {
        e.preventDefault();
        setView((v) => (v === "before" ? "result" : "before"));
      } else if (view === "mask" && (e.key === "x" || e.key === "X")) {
        setMaskMode((m) => (m === "add" ? "subtract" : "add"));
      } else if (view === "area" && (e.key === "x" || e.key === "X")) {
        setAreaMode((m) => (m === "add" ? "remove" : "add"));
      } else if ((e.ctrlKey || e.metaKey) && e.key === "0") {
        e.preventDefault();
        viewerRef.current?.fit();
      } else if ((e.ctrlKey || e.metaKey) && e.key === "1") {
        e.preventDefault();
        viewerRef.current?.actualPixels();
      } else if ((e.ctrlKey || e.metaKey) && (e.key === "=" || e.key === "+")) {
        e.preventDefault();
        viewerRef.current?.zoomBy(1.5);
      } else if ((e.ctrlKey || e.metaKey) && e.key === "-") {
        e.preventDefault();
        viewerRef.current?.zoomBy(1 / 1.5);
      } else if (!e.ctrlKey && !e.metaKey && (e.key === "z" || e.key === "Z")) {
        setZoomTool((on) => !on);
        setBrushOn(false);
      } else if (view !== "mask" && (e.key === "b" || e.key === "B")) {
        setBrushOn((on) => !on);
        setZoomTool(false);
      } else if (view !== "mask" && view !== "area" && !e.ctrlKey && !e.metaKey && (e.key === "p" || e.key === "P")) {
        pickBrush("patch"); // on, or off again if it's on
      } else if (view !== "mask" && view !== "area" && !e.ctrlKey && !e.metaKey && (e.key === "g" || e.key === "G")) {
        pickBrush("reflection");
      } else if (!e.ctrlKey && !e.metaKey && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
        e.preventDefault();
        step(e.key === "ArrowRight" ? 1 : -1);
      } else if (e.key === "[" || e.key === "]") {
        // Resize whichever brush is in use.
        const factor = e.key === "]" ? 1.2 : 1 / 1.2;
        const setRadius = view === "mask" || view === "area" ? setMaskBrushRadius : setBrushRadius;
        setRadius((r) => clampBrush(r * factor));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const currentSettings = (): PhotoSettings => ({
    backdrop: paramsRef.current,
    outdoor: outdoorRef.current,
    eyes: eyesRef.current,
    skin: skinRef.current,
    mouth: mouthRef.current,
    dodgeBurn: dodgeBurnRef.current,
    creases: creasesRef.current,
    tone: toneRef.current,
    crop: cropRef.current,
    opacity: opacityRef.current,
    relight: relightRef.current,
    off: offRef.current,
  });

  const applySettings = (st: PhotoSettings) => {
    paramsRef.current = st.backdrop;
    outdoorRef.current = st.outdoor;
    setOutdoor(st.outdoor);
    eyesRef.current = st.eyes;
    skinRef.current = st.skin;
    setSkinParams(st.skin);
    setParams(st.backdrop);
    setEyesParams(st.eyes);
    mouthRef.current = st.mouth;
    setMouthParams(st.mouth);
    dodgeBurnRef.current = st.dodgeBurn;
    setDodgeBurn(st.dodgeBurn);
    creasesRef.current = st.creases;
    setCreases(st.creases);
    toneRef.current = st.tone;
    setToneParams(st.tone);
    cropRef.current = st.crop;
    setCrop(st.crop);
    opacityRef.current = st.opacity;
    setOpacity(st.opacity);
    relightRef.current = st.relight ?? null;
    setRelight(st.relight ?? null);
    offRef.current = st.off ?? [];
    setOff(st.off ?? []);
  };

  const markEdited = (path: string, st: PhotoSettings) =>
    setStrip((items) =>
      items.map((i) =>
        i.path === path
          ? { ...i, edited: !sameSettings(st, ALL_DEFAULTS) }
          : i,
      ),
    );

  /** Make ``path`` the photo being edited, keeping the current one's settings
   * (unless it's being taken out of the strip). Shows the camera's embedded
   * preview straight away; the engine opens it after OPEN_DELAY_MS, or as soon
   * as it's edited. */
  const loadImage = async (path: string, keepCurrent = true) => {
    const current = activePath.current;
    if (current && keepCurrent) {
      settingsByPath.current.set(current, currentSettings());
      markEdited(current, currentSettings());
    }
    activePath.current = path;
    if (current !== path) {
      setHistogram(null);
      setSkinTab("face"); // Skin opens on Face for every photo
      setRelightEditing(false);
    }
    setPrep({ path, fraction: PREP_START });
    if (irisScaleFor.current !== path) setIrisScale(null);
    window.clearTimeout(openTimer.current);
    stopCountdown();
    setError(null);
    setResult(null);
    setMask(null);
    setView("result");
    setFaces(null);
    setRemovals(0);
    setRemovalKinds({});
    setMaskEdits(0);
    const saved = settingsByPath.current.get(path);
    applySettings(saved ?? ALL_DEFAULTS);
    setModeIsAuto(autoMode.current.has(path));

    // Back to the photo the engine already has: no need to wait.
    if (openedPath.current === path && openedInfo.current) {
      setImage(openedInfo.current);
      setRemovals(openedInfo.current.removals);
      setRemovalKinds(openedInfo.current.removal_kinds ?? {});
      setMaskEdits(openedInfo.current.mask_edits);
      render();
      return;
    }
    try {
      const q = await call<{
        image: string;
        width: number;
        height: number;
        scene: "backdrop" | "outdoor";
      }>("thumbnail", {
        path,
        edge: PREVIEW_EDGE,
      });
      if (activePath.current !== path) return;
      // Its camera preview is showing; the bar carries on until the
      // retouched version replaces it.
      setPrep((p) => (p?.path === path ? { path, fraction: Math.max(p.fraction, PREP_PREVIEW) } : p));
      if (!saved) applyDetectedScene(path, q.scene);
      setImage({
        path,
        width: q.width,
        height: q.height,
        bit_depth: 0, // not known until opened
        preview: q.image,
        removals: 0,
        removal_kinds: {},
        mask_edits: 0,
      });
    } catch (e) {
      handleError(e);
    }
    startCountdown();
    openTimer.current = window.setTimeout(async () => {
      if (activePath.current === path && (await ensureOpen())) render();
    }, OPEN_DELAY_MS);
  };

  const openImage = async () => {
    // Start in the last image's folder. Otherwise GTK may open on its "Recent"
    // view, which lists files flat and makes them look as if they've moved.
    const lastDir = image?.path.replace(/\/[^/]*$/, "") ?? readLastDir();
    const picked = await open({
      multiple: true,
      defaultPath: lastDir ?? undefined,
      filters: [
        { name: "Images and RAW", extensions: bothCases([...IMAGE_EXTENSIONS, ...RAW_EXTENSIONS]) },
        { name: "RAW", extensions: bothCases(RAW_EXTENSIONS) },
        { name: "Images", extensions: bothCases(IMAGE_EXTENSIONS) },
      ],
    });
    const paths = picked ? (Array.isArray(picked) ? picked : [picked]) : [];
    if (!paths.length) return;
    const known = new Set(strip.map((i) => i.path));
    const added = paths.filter((p) => !known.has(p));
    setStrip((items) => [...items, ...added.map((path) => ({ path, edited: false }))]);
    const first = added[0] ?? paths[0];
    setSelected(new Set([first]));
    if (first !== image?.path) await loadImage(first);
  };

  /** Set Backdrop/Outdoor from detection, for a photo the user hasn't set up. */
  const applyDetectedScene = (path: string, detected: "backdrop" | "outdoor") => {
    const wasUnknown = modeUnknown(path);
    // Chosen by hand before the detection came in: keep the choice.
    if (detectedScene.current.has(path) && !autoMode.current.has(path)) return;
    detectedScene.current.add(path);
    if (settingsByPath.current.has(path) && !autoMode.current.has(path)) return;
    const wantOutdoor = detected === "outdoor";
    autoMode.current.add(path);
    if (activePath.current === path) {
      setModeIsAuto(true);
      const changed = outdoorRef.current !== wantOutdoor;
      if (changed) {
        outdoorRef.current = wantOutdoor;
        setOutdoor(wantOutdoor);
      }
      // Rendered without its backdrop while the mode was unknown: now with it.
      if ((changed || wasUnknown) && openedPath.current === path) render();
    } else {
      const st = settingsByPath.current.get(path) ?? ALL_DEFAULTS;
      settingsByPath.current.set(path, { ...st, outdoor: wantOutdoor });
    }
  };

  // Thumbnails, one at a time so they don't hold up the photo being edited,
  // and none while it's being rendered: on a laptop they took its CPU.
  const loadingThumb = useRef(false);
  useEffect(() => {
    const next = strip.find((i) => !i.thumb);
    if (!next || loadingThumb.current || !engineReady || busy) return;
    loadingThumb.current = true;
    call<{ image: string; scene: "backdrop" | "outdoor" }>("thumbnail", { path: next.path })
      .then((r) => {
        setStrip((items) =>
          items.map((i) => (i.path === next.path ? { ...i, thumb: r.image, scene: r.scene } : i)),
        );
        applyDetectedScene(next.path, r.scene);
      })
      .catch(() =>
        // Leave a placeholder rather than retrying forever.
        setStrip((items) => items.map((i) => (i.path === next.path ? { ...i, thumb: "" } : i))),
      )
      .finally(() => {
        loadingThumb.current = false;
      });
  }, [strip, engineReady, busy]);

  const removeFromStrip = async (path: string) => {
    const idx = strip.findIndex((i) => i.path === path);
    const rest = strip.filter((i) => i.path !== path);
    setStrip(rest);
    setSelected((sel) => new Set([...sel].filter((p) => p !== path)));
    settingsByPath.current.delete(path);
    if (image?.path === path) {
      const neighbour = rest[Math.min(idx, rest.length - 1)];
      if (neighbour) {
        // Switch first, without keeping this photo's settings; the engine
        // stashes the photo it's leaving, so forget it only afterwards.
        await loadImage(neighbour.path, false);
        setSelected(new Set([neighbour.path]));
      } else {
        activePath.current = null;
        window.clearTimeout(openTimer.current);
        stopCountdown();
        setImage(null);
        setResult(null);
      }
    }
    await call("forget", { path }).catch(() => undefined);
  };

  // About: the app, its licence and the AI models it uses (from the engine's
  // own list, so it's always accurate).
  const showAbout = async (): Promise<boolean> => {
    try {
      setAboutModels(await call<ModelInfo[]>("models"));
      return true;
    } catch (e) {
      handleError(e);
      return false;
    }
  };


  const copySettings = () => {
    setCopied(currentSettings());
    setStatus("Settings copied");
  };

  const pasteSettings = () => {
    if (!copied) return;
    let pasted = 0;
    for (const path of selected) {
      autoMode.current.delete(path); // a pasted mode is the user's choice
      if (path === image?.path) setModeIsAuto(false);
      if (path === image?.path) {
        applySettings(copied);
        render();
      } else {
        settingsByPath.current.set(path, copied);
      }
      markEdited(path, copied);
      pasted++;
    }
    setStatus(`Settings pasted to ${pasted} photo${pasted === 1 ? "" : "s"}`);
  };

  // Presets: every slider setting, but not the Backdrop/Outdoor mode (each
  // photo keeps the one detected or chosen for it) and not brush work, which
  // belongs to the photo it was painted on. They apply to the film-strip
  // selection, like paste.
  const presetTargets = (): string[] => {
    const paths = [...selected].filter((p) => strip.some((i) => i.path === p));
    return paths.length ? paths : image ? [image.path] : [];
  };
  const savePreset = async (name: string) => {
    const { outdoor: _mode, crop: _crop, relight: _relight, ...settings } = currentSettings();
    try {
      setPresets(await call<Preset[]>("save_preset", { name, settings }));
      setStatus(`Saved preset "${name}"`);
    } catch (e) {
      handleError(e);
    }
  };
  const deletePreset = async (name: string) => {
    try {
      setPresets(await call<Preset[]>("delete_preset", { name }));
      setStatus(`Deleted preset "${name}"`);
    } catch (e) {
      handleError(e);
    }
  };
  const applyPreset = (preset: Preset) => {
    const targets = presetTargets();
    for (const path of targets) {
      // A photo never opened has no settings yet: keep the mode detected on
      // import (else a preset turned an outdoor photo's backdrop step on),
      // and keep it automatic, so a detection still to come sets it.
      const fresh = path !== image?.path && !settingsByPath.current.has(path);
      if (fresh) autoMode.current.add(path);
      const current =
        path === image?.path
          ? currentSettings()
          : (settingsByPath.current.get(path) ?? {
              ...ALL_DEFAULTS,
              outdoor: strip.find((i) => i.path === path)?.scene === "outdoor",
            });
      const st = withPreset(preset.settings, current.outdoor, current.crop, current.relight ?? null);
      if (path === image?.path) {
        applySettings(st);
        render();
      } else {
        settingsByPath.current.set(path, st);
      }
      markEdited(path, st);
    }
    setStatus(`Applied "${preset.name}" to ${targets.length} photo${targets.length === 1 ? "" : "s"}`);
  };

  const step = (delta: number) => {
    if (!image || strip.length < 2) return;
    const idx = strip.findIndex((i) => i.path === image.path);
    const next = strip[idx + delta];
    if (next) {
      setSelected(new Set([next.path]));
      loadImage(next.path);
    }
  };

  // Mask view: the photo with the backdrop tinted red, where the brush corrects
  // the subject mask. Fetched fresh each time, since removals change the photo.
  const maskCall = useCallback(
    async (method: string, extra: Record<string, unknown> = {}) => {
      setMaskPending(true);
      try {
        if (!(await ensureOpen())) return false;
        const r = await call<{ preview: string; mask_edits: number }>(method, extra);
        setMask(r.preview);
        setMaskEdits(r.mask_edits);
        setStatus("Ready");
        return true;
      } catch (e) {
        handleError(e);
        return false;
      } finally {
        setMaskPending(false);
      }
    },
    [handleError, ensureOpen],
  );

  const showMask = async () => {
    if (view === "mask") {
      setView("result");
      render(); // the backdrop may need redoing with the corrected mask
      return;
    }
    if (await maskCall("mask")) setView("mask");
  };

  // Refine area: a tool's area shown in blue, corrected by painting. The
  // brush size is shared with the subject mask brush.
  const areaCall = async (method: string, which: Area, extra: Record<string, unknown> = {}) => {
    setAreaPending(true);
    try {
      if (!(await ensureOpen())) return false;
      const r = await call<{ preview: string; edits: number }>(method, { region: which, ...extra });
      setAreaImg(r.preview);
      setAreaEdits(r.edits);
      setStatus("Ready");
      return true;
    } catch (e) {
      handleError(e);
      return false;
    } finally {
      setAreaPending(false);
    }
  };
  const showArea = async (which: Area) => {
    if (view === "area" && area === which) {
      closeArea();
      return;
    }
    setArea(which);
    if (await areaCall("region_view", which)) setView("area");
  };
  const closeArea = () => {
    setView("result");
    render(); // the tools redo their work with the corrected area
  };
  const onAreaStroke = (points: [number, number][], radius: number) =>
    area && areaCall("region_paint", area, { mode: areaMode, points, radius });
  const undoAreaEdit = () => area && areaCall("undo_region_edit", area);
  const clearAreaEdits = () => area && areaCall("clear_region_edits", area);

  const onMaskStroke = (points: [number, number][], radius: number) =>
    maskCall("mask_paint", { mode: maskMode, points, radius });
  const undoMaskEdit = () => maskCall("undo_mask_edit");
  const clearMaskEdits = () => maskCall("clear_mask_edits");

  // Each panel's hold-for-before shows everything except that panel's step, so
  // each edit can be judged on its own. Releasing before the render returns
  // must not leave the comparison showing, hence the token.
  // Before views are rendered ahead on hover and remembered until the photo
  // or its settings change, so pressing shows them at once. Requests are
  // shared: pressing while the hover render is still running waits for it.
  const compareCache = useRef(new Map<string, Promise<string>>());
  const beforeView = (step: Step): Promise<string> => {
    const args = withoutStep(lookArgs(), step);
    const key = `${openedPath.current}|${JSON.stringify(args)}`;
    let pending = compareCache.current.get(key);
    if (!pending) {
      pending = call<{ preview: string }>("render", args).then((r) => r.preview);
      pending.catch(() => compareCache.current.delete(key));
      compareCache.current.set(key, pending);
    }
    return pending;
  };
  const prefetchWithout = (step: Step) => {
    // Only when idle and open: never queue ahead of a slider's own render.
    if (inFlight.current || !activePath.current || openedPath.current !== activePath.current) return;
    beforeView(step).catch(() => undefined);
  };
  // Which panels have edits to compare, in the sidebar's top-to-bottom order.
  const stepIsEdited = (step: Step): boolean => {
    switch (step) {
      case "removals":
        return removals > 0;
      case "tone":
        return !sameValues(toneRef.current, TONE_DEFAULTS);
      case "backdrop":
        return !outdoorRef.current;
      case "skin":
        return !sameValues(skinRef.current, SKIN_DEFAULTS);
      case "skin_face":
      case "skin_neck":
      case "skin_body":
        return !sameValues(skinRef.current[step.slice("skin_".length) as SkinTab], SKIN_REGION_DEFAULTS);
      case "dodge_burn":
        return !sameValues(dodgeBurnRef.current, DODGE_BURN_DEFAULTS);
      case "eyes":
        return !sameValues(eyesRef.current, EYE_DEFAULTS);
      case "mouth":
        return !sameValues(mouthRef.current, MOUTH_DEFAULTS);
      case "clothes":
        return creasesRef.current > 0;
      case "relight":
        return relightLook(relightRef.current) !== null;
    }
  };
  const holdWithout = async (step: Step) => {
    const token = ++compareToken.current;
    setCompareStep(step);
    setView("compare");
    try {
      if (!(await ensureOpen())) return;
      const preview = await beforeView(step);
      if (token === compareToken.current) setCompareImg(preview);
      // Photographers tend to check panels top to bottom: get the next edited
      // panel's before view ready while this one is being looked at.
      const panelStep: Step = step.startsWith("skin_") ? "skin" : step;
      const next = PANEL_STEPS.slice(PANEL_STEPS.indexOf(panelStep) + 1).find(stepIsEdited);
      if (next) beforeView(next).catch(() => undefined);
    } catch (e) {
      handleError(e);
    }
  };
  useEffect(() => {
    compareCache.current.clear();
  }, [result]);
  const releaseCompare = () => {
    compareToken.current++;
    setCompareImg(null);
    setView((v) => (v === "compare" ? "result" : v));
  };

  // Full-resolution detail when zoomed in. The tile must show exactly what the
  // viewer shows (result, original, or a per-step before), so each request uses
  // that view's render arguments, and a tile for anything else is dropped.
  const detailArgs = (): Record<string, unknown> | null => {
    if (view === "mask" || view === "area") return null;
    if (view === "before") return { removals: false };
    if (view === "compare" && compareStep) return withoutStep(lookArgs(), compareStep);
    return lookArgs();
  };
  const detailArgsRef = useRef(detailArgs);
  detailArgsRef.current = detailArgs;
  const wantedRegion = useRef<{ region: [number, number, number, number]; scale: number } | null>(
    null,
  );
  const contentVersion = useRef(0);
  const detailInFlight = useRef(false);
  const detailDirty = useRef(false);

  const requestDetail = useCallback(async () => {
    if (detailInFlight.current) {
      detailDirty.current = true;
      return;
    }
    detailInFlight.current = true;
    try {
      do {
        detailDirty.current = false;
        const want = wantedRegion.current;
        setDetailLoading(Boolean(want));
        const args = detailArgsRef.current();
        // Zoom detail waits for the real open rather than forcing one, so
        // scanning while zoomed in stays fast; it's re-asked once opened.
        if (!want || !args || openedPath.current !== activePath.current) break;
        const version = contentVersion.current;
        const r = await call<Detail>("render_region", {
          region: want.region,
          scale: want.scale,
          ...args,
        });
        if (version === contentVersion.current && wantedRegion.current === want) setDetail(r);
      } while (detailDirty.current);
    } catch (e) {
      if ((e as EngineError)?.kind === "superseded") {
        // The engine dropped it for a slider change: ask again once the
        // sliders are still (its finished steps are kept).
        const again = () => {
          if (inFlight.current || dirty.current) window.setTimeout(again, DETAIL_AFTER_EDIT_MS);
          else requestDetailRef.current();
        };
        window.setTimeout(again, DETAIL_AFTER_EDIT_MS);
      } else handleError(e);
    } finally {
      detailInFlight.current = false;
      setDetailLoading(false);
      // The engine's last message ("Rendering full-resolution detail") stayed
      // in the footer through every later zoom step, which only reuse it.
      if (!inFlight.current) setStatus("Ready");
    }
  }, [handleError]);
  const requestDetailRef = useRef(requestDetail);
  requestDetailRef.current = requestDetail;

  const onDetailNeeded = useCallback(
    (region: [number, number, number, number] | null, scale: number) => {
      wantedRegion.current = region ? { region, scale } : null;
      if (!region) setDetail(null);
      else requestDetail();
    },
    [requestDetail],
  );

  // Whenever the picture changes, the old tile is stale: drop it and re-ask,
  // but only once the sliders have been still for a moment. The engine does
  // one thing at a time, so a full-resolution tile asked for mid-drag would
  // hold up the next preview and make the slider lag.
  const detailTimer = useRef<number | undefined>(undefined);
  useEffect(() => {
    contentVersion.current++;
    setDetail(null);
    window.clearTimeout(detailTimer.current);
    if (!wantedRegion.current) return;
    const ask = () => {
      if (inFlight.current || dirty.current) {
        detailTimer.current = window.setTimeout(ask, DETAIL_AFTER_EDIT_MS);
      } else {
        requestDetail();
      }
    };
    detailTimer.current = window.setTimeout(ask, DETAIL_AFTER_EDIT_MS);
    return () => window.clearTimeout(detailTimer.current);
  }, [result, compareImg, view, compareStep, requestDetail]);

  // Export: the film-strip selection (or the photo being edited), each photo
  // with its own settings, one after another, each with its own progress bar.
  const openExport = () => {
    if (!image) return;
    setExportRows(
      presetTargets().map((path) => ({
        path,
        thumb: strip.find((i) => i.path === path)?.thumb,
        state: "waiting",
        fraction: 0,
      })),
    );
  };
  const chooseExportFolder = async () => {
    const dir = await open({ directory: true, defaultPath: exportFolder ?? readLastDir() ?? undefined });
    if (typeof dir === "string") setExportFolder(dir);
  };
  const runExport = async () => {
    if (!exportRows) return;
    const update = (path: string, patch: Partial<ExportRow>) =>
      setExportRows((rows) => rows && rows.map((r) => (r.path === path ? { ...r, ...patch } : r)));
    exportCancel.current = false;
    setExportCancelling(false);
    setExportRunning(true);
    setBusy(true);
    const ext = exportFormat === "tiff" ? "tif" : "jpg";
    const writes: Promise<void>[] = [];
    // The photo the engine has open goes first: it's ready to go.
    const order = exportRows
      .map((r) => r.path)
      .sort((a, b) => Number(b === openedPath.current) - Number(a === openedPath.current));
    for (const path of order) {
      if (exportCancel.current) {
        update(path, { state: "cancelled" });
        continue;
      }
      exportingPath.current = path;
      update(path, { state: "exporting", fraction: 0.02 });
      try {
        let st = path === activePath.current ? currentSettings() : settingsByPath.current.get(path);
        if (path !== activePath.current && (!st || autoMode.current.has(path))) {
          // Its mode wasn't chosen by hand: the detected one, detected now if
          // its thumbnail hasn't come in yet.
          let scene = strip.find((i) => i.path === path)?.scene;
          if (!scene) scene = (await call<{ scene: "backdrop" | "outdoor" }>("thumbnail", { path })).scene;
          st = { ...(st ?? ALL_DEFAULTS), outdoor: scene === "outdoor" };
        }
        if (!st) st = ALL_DEFAULTS;
        if (openedPath.current !== path) {
          const r = await call<OpenResult>("open", { path });
          openedPath.current = path;
          openedInfo.current = { ...r, path };
        }
        const stem = path.replace(/\.[^./]+$/, "");
        const out = exportFolder
          ? `${exportFolder}/${stem.split("/").pop()}_retouched.${ext}`
          : `${stem}_retouched.${ext}`;
        const r = await call<{ path: string; pending: boolean }>("export", {
          path: out,
          ...lookArgsFor(st),
          crop: isNoCrop(st.crop) ? null : st.crop,
          long_edge: exportFormat === "jpeg" && exportSize.mode === "long" ? exportSize.px : null,
          background: true,
        });
        // Written while the next photo is retouched: done once the file is.
        update(path, { fraction: 0.95, message: "Writing…" });
        writes.push(
          new Promise<void>((resolve) => {
            const settle = (error?: string) => {
              update(path, error ? { state: "failed", message: error } : { state: "done", fraction: 1, message: r.path });
              resolve();
            };
            if (!r.pending) settle();
            else if (writtenEarly.current.has(r.path)) {
              const error = writtenEarly.current.get(r.path);
              writtenEarly.current.delete(r.path);
              settle(error);
            } else pendingWrites.current.set(r.path, settle);
          }),
        );
      } catch (e) {
        update(path, { state: "failed", message: errorMessage(e) });
      }
    }
    exportingPath.current = null;
    await Promise.all(writes); // the last files may still be being written
    setExportRunning(false);
    setBusy(false);
    // Back to the photo being edited.
    if (activePath.current && openedPath.current !== activePath.current && (await ensureOpen())) render();
  };

  const acceptLicence = async () => {
    if (!licencePrompt?.model) return;
    await call("accept_licence", { model_id: licencePrompt.model });
    setLicencePrompt(null);
    render();
  };

  const shown =
    view === "mask"
      ? mask
      : view === "area"
        ? areaImg
        : view === "compare"
        ? (compareImg ?? result)
        : view === "before" || !result
          ? image?.preview
          : result;

  return (
    <div className="app">
      <TitleBar
        detail={
          image
            ? `${image.path.split(/[\\/]/).pop()} · ${image.width}×${image.height} · ${image.bit_depth ? `${image.bit_depth}-bit` : "preview"}`
            : null
        }
      />
      <header className="toolbar">
        {!tutorialDismissed && tutorialUrl && (
          <div className="video-link">
            <a
              className="video-link__cta"
              href={tutorialUrl}
              onClick={(e) => {
                e.preventDefault();
                openUrl(tutorialUrl);
              }}
              title="Watch the tutorial video on YouTube"
            >
              ▶ New user? Watch this first!
            </a>
            <button
              type="button"
              className="video-link__close"
              onClick={dismissTutorial}
              title="Dismiss"
            >
              ×
            </button>
          </div>
        )}
        <button onClick={openImage} disabled={!engineReady}>
          <FolderOpen size={16} /> Open
        </button>
        <button
          onClick={openExport}
          disabled={!image || busy}
          title="Export the photos selected in the film strip, or the one being edited"
        >
          <Download size={16} /> Export
        </button>
        <button
          disabled={!image || view === "mask"}
          {...holdHandlers(
            () => setView("before"),
            () => setView((v) => (v === "before" ? "result" : v)),
          )}
          title="Hold to see the untouched original"
        >
          <Eye size={16} /> Original
        </button>
        <div className="toolbar__group">
          <button disabled={!image} onClick={() => viewerRef.current?.fit()} title="Fit (Ctrl+0)">
            <Maximize size={16} /> Fit
          </button>
          <button
            disabled={!image}
            onClick={() => {
              setZoomTool((on) => !on);
              setBrushOn(false);
            }}
            className={zoomTool ? "active" : ""}
            title="Zoom tool (Z): click to zoom in, Alt+click to zoom out. Wheel zooms, Space+drag pans"
          >
            <ZoomIn size={16} />
          </button>
          <button
            disabled={!image}
            onClick={() => viewerRef.current?.actualPixels()}
            title="100%: one image pixel per screen pixel (Ctrl+1)"
          >
            100%
          </button>
          <span className="toolbar__zoom">{image ? zoomLabel : ""}</span>
        </div>
        <div className="toolbar__group toolbar__group--tools">
          {(!activeTool || activeTool === "crop") && (
            <button
            disabled={!image || view === "mask" || view === "area"}
            onClick={() => (cropping ? setCropping(false) : startCropping())}
            className={cropping ? "active" : ""}
            data-tip="Crop & straighten (C)" aria-label="Crop & straighten (C). Applied when you export"
          >
            <CropIcon size={16} />
          </button>
          )}
          {(!activeTool || activeTool === "fill") && (
            <button
            disabled={!image || cropping || view === "mask" || view === "area"}
            onClick={() => pickBrush("fill")}
            className={brushOn && brushMode === "fill" ? "active" : ""}
            data-tip="Remove brush (B)" aria-label="Remove brush (B): paint over anything to remove it"
          >
            <Paintbrush size={16} />
          </button>
          )}
          {(!activeTool || activeTool === "reflection") && (
            <button
            disabled={!image || cropping || view === "mask" || view === "area"}
            onClick={() => pickBrush("reflection")}
            className={brushOn && brushMode === "reflection" ? "active" : ""}
            data-tip="Glasses reflection (G)" aria-label="Glasses reflection (G): paint over a reflection on a lens to remove it"
          >
            <Glasses size={16} />
          </button>
          )}
          {(!activeTool || activeTool === "patch") && (
            <button
            disabled={!image || cropping || view === "mask" || view === "area"}
            onClick={() => pickBrush("patch")}
            className={brushOn && brushMode === "patch" ? "active" : ""}
            data-tip="Patch (P)" aria-label="Patch (P): draw around an area, then drag it onto clean skin or backdrop to take its texture"
          >
            <Lasso size={16} />
          </button>
          )}
          {activeTool && activeTool !== "crop" && (
            <>
              {activeTool !== "patch" && (
                <span className="toolbar__range-wrap" data-tip="Brush size ([ and ])">
                  <input
                    type="range"
                    className="toolbar__range"
                    min={BRUSH.min}
                    max={BRUSH.max}
                    step={BRUSH.step}
                    value={brushRadius}
                    onChange={(e) => setBrushRadius(Number(e.target.value))}
                    onDoubleClick={() => setBrushRadius(BRUSH.default)}
                    aria-label="Brush size ([ and ] keys)"
                  />
                </span>
              )}
              {activeTool === "reflection" && (
                <span className="toolbar__range-wrap" data-tip="Reflection strength">
                  <input
                    type="range"
                    className="toolbar__range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={reflectionStrength}
                    onChange={(e) => setReflectionStrength(Number(e.target.value))}
                    onDoubleClick={() => setReflectionStrength(1)}
                    aria-label="Reflection strength: how much of the reflection each new stroke removes"
                  />
                </span>
              )}
              <button
                disabled={removals === 0 || removing}
                onClick={undoRemove}
                data-tip="Undo last removal (Ctrl+Z)"
                aria-label="Undo the last removal (Ctrl+Z)"
              >
                <Undo2 size={16} />
              </button>
              <button
                disabled={!removalKinds[brushMode] || removing}
                onClick={clearRemovals}
                data-tip={`Clear ${BRUSH_NAMES[brushMode]}`}
                aria-label={`Undo every ${BRUSH_NAMES[brushMode]} removal on this photo, keeping the others`}
              >
                <Trash2 size={16} />
              </button>
              <button
                disabled={removals === 0 || removing}
                onPointerEnter={() => prefetchWithout("removals")}
                {...holdHandlers(() => holdWithout("removals"), releaseCompare)}
                data-tip="Hold for before"
                aria-label="Hold to see the photo without your removals"
              >
                <Eye size={16} />
              </button>
              <button
                className="toolbar__close"
                onClick={() => setBrushOn(false)}
                data-tip={activeTool === "patch" ? "Close" : "Close (Esc)"}
                aria-label="Close the tool"
              >
                <X size={16} />
              </button>
            </>
          )}
        </div>
        <div className="toolbar__spacer" />
        <button className="toolbar__about" onClick={showAbout} title="About RapidRetouch">
          About
        </button>
      </header>

      <main className="viewer" onDragStart={(e) => e.preventDefault()}>
        {prep && prep.path === activePath.current && (
          <div className="viewer__progress" title="Preparing this photo">
            <div style={{ width: `${prep.fraction * 100}%` }} />
          </div>
        )}
        <Viewer
          ref={viewerRef}
          src={shown ? jpegSrc(shown) : null}
          size={image ? { width: image.width, height: image.height } : null}
          previewEdge={PREVIEW_EDGE}
          zoomTool={zoomTool}
          detail={detail}
          imgRef={setImgEl}
          onDetailNeeded={onDetailNeeded}
          onZoomChange={setZoomLabel}
          // Straightened and cropped as it will export, except while cropping
          // (the crop tool shows the whole photo itself).
          rotation={cropping ? 0 : crop.angle}
          frame={cropping || isNoCrop(crop) ? null : crop}
          empty={<div className="viewer__empty">Open a RAW file, 16-bit TIFF or JPEG to start</div>}
        >
          {({ panning }) => (
            <>
              {image && (view === "result" || view === "before" || view === "compare") && (cropping || !isNoCrop(crop)) && (
                <CropOverlay
                  image={imgEl}
                  width={image.width}
                  height={image.height}
                  crop={crop}
                  editing={cropping && !panning}
                  ratio={cropping ? aspectRatio(cropAspect, image.width, image.height) : null}
                  onChange={updateCrop}
                />
              )}
              {image && view === "result" && !cropping && relight && relightEditing && (
                <RelightOverlay
                  image={imgEl}
                  width={image.width}
                  height={image.height}
                  shape={relight}
                  feather={relight.feather}
                  editing={!panning && !zoomTool}
                  onChange={(change) => updateRelight({ ...relightRef.current!, ...change })}
                />
              )}
              {image && view === "result" && !cropping && brushMode !== "patch" && (
                <PaintOverlay
                  image={imgEl}
                  active={brushOn && !panning && !zoomTool}
                  radius={brushRadius}
                  pending={removing}
                  onStroke={onStroke}
                  colour={BRUSH_COLOURS[brushMode]}
                />
              )}
              {image && view === "result" && !cropping && brushOn && brushMode === "patch" && (
                <PatchOverlay
                  image={imgEl}
                  active={!panning && !zoomTool}
                  pending={removing}
                  onPatch={onPatch}
                />
              )}
              {image && view === "mask" && (
                <PaintOverlay
                  image={imgEl}
                  active={!panning && !zoomTool}
                  radius={maskBrushRadius}
                  pending={maskPending}
                  onStroke={onMaskStroke}
                  colour={maskMode === "add" ? "rgba(255, 255, 255, 0.5)" : "rgba(255, 40, 40, 0.6)"}
                />
              )}
              {image && view === "area" && (
                <PaintOverlay
                  image={imgEl}
                  active={!panning && !zoomTool}
                  radius={maskBrushRadius}
                  pending={areaPending}
                  onStroke={onAreaStroke}
                  colour={areaMode === "add" ? "rgba(60, 140, 255, 0.55)" : "rgba(255, 255, 255, 0.5)"}
                />
              )}
            </>
          )}
        </Viewer>
        {view === "before" && <div className="viewer__badge">Original</div>}
        {activeTool && activeTool !== "crop" && !removing && (
          <div className="viewer__badge viewer__badge--low">
            {activeTool === "patch"
              ? "Patch: draw around the area to fix, then drag it onto clean skin or backdrop. Esc drops the selection"
              : activeTool === "reflection"
                ? "Glasses: paint just the reflection, a little past its edges. Paint again to take more"
                : "Brush: paint over a distraction; it's filled when you let go"}
          </div>
        )}
        {slow && !(detailLoading && !detail) && (
          <div className="viewer__busy" title="Updating the preview">
            <span className="viewer__busy-circle" />
          </div>
        )}
        {detailLoading && !detail && (
          <div className="viewer__badge viewer__badge--detail">
            Rendering full detail…
            <span className="app-footer__busy" />
          </div>
        )}
        {cropping && image && (
          <div
            ref={cropBarRef}
            className="brushbar cropbar"
            style={cropBarPos ? { left: cropBarPos.left, top: cropBarPos.top, transform: "none" } : undefined}
          >
            <span className="cropbar__grip" onPointerDown={dragCropBar} title="Drag to move">
              <GripVertical size={14} />
            </span>
            <span className="brushbar__what">Crop</span>
            <label className="brushbar__size" title="Straighten: turn the photo (double-click to reset)">
              Straighten
              <input
                type="range"
                min={-45}
                max={45}
                step={0.1}
                value={crop.angle}
                onDoubleClick={() => updateCrop(fitCrop({ ...cropRef.current, angle: 0 }, image.width, image.height))}
                onChange={(e) => {
                  const next = fitCrop({ ...cropRef.current, angle: Number(e.target.value) }, image.width, image.height);
                  const r = aspectRatio(cropAspect, image.width, image.height);
                  updateCrop(r === null ? next : cropToRatio(next, r, image.width, image.height));
                }}
              />
              <span className="cropbar__angle">{crop.angle > 0 ? "+" : ""}{crop.angle.toFixed(1)}°</span>
            </label>
            <div className="tabs">
              {CROP_ASPECTS.map((a) => (
                <button
                  key={a}
                  className={cropAspect === a ? "active" : ""}
                  onClick={() => {
                    setCropAspect(a);
                    const r = aspectRatio(a, image.width, image.height);
                    if (r !== null) updateCrop(cropToRatio(cropRef.current, r, image.width, image.height));
                  }}
                >
                  {a}
                </button>
              ))}
            </div>
            <button onClick={() => updateCrop(NO_CROP)} title="Back to the whole photo, unturned">
              <RotateCcw size={14} /> Reset
            </button>
            <button className="brushbar__done" onClick={() => setCropping(false)} title="Done (Enter)">
              Done
            </button>
          </div>
        )}
        {view === "mask" && (
          <BrushBar
            what="Subject mask"
            mode={maskMode === "add" ? "add" : "remove"}
            addLabel="Subject"
            removeLabel="Backdrop"
            onMode={(m) => setMaskMode(m === "add" ? "add" : "subtract")}
            radius={maskBrushRadius}
            min={BRUSH.min}
            max={BRUSH.max}
            step={BRUSH.step}
            onRadius={setMaskBrushRadius}
            edits={maskEdits}
            pending={maskPending}
            onUndo={undoMaskEdit}
            onClear={clearMaskEdits}
            onDone={showMask}
          />
        )}
        {view === "area" && area && (
          <BrushBar
            what={`${AREA_NAMES[area]} area`}
            mode={areaMode}
            addLabel="Add"
            removeLabel="Remove"
            onMode={setAreaMode}
            radius={maskBrushRadius}
            min={BRUSH.min}
            max={BRUSH.max}
            step={BRUSH.step}
            onRadius={setMaskBrushRadius}
            edits={areaEdits}
            pending={areaPending}
            onUndo={undoAreaEdit}
            onClear={clearAreaEdits}
            onDone={closeArea}
          />
        )}
        {view === "area" && area && (
          <div className="viewer__badge viewer__badge--low">
            {AREA_NAMES[area]} · blue is what the {AREA_TOOLS[area]} tools work on ·
            painting {areaMode === "add" ? "it in" : "it out"}
          </div>
        )}
        {image && view !== "mask" && view !== "area" && (
          <div className="tabs before-after" title="Before / after (\ key)">
            <button
              className={view === "before" ? "active" : ""}
              onClick={() => setView("before")}
            >
              Before
            </button>
            <button
              className={view !== "before" ? "active" : ""}
              onClick={() => setView("result")}
            >
              After
            </button>
          </div>
        )}
        {view === "compare" && compareImg && (
          <div className="viewer__badge">Before · {compareStep && STEP_NAMES[compareStep]}</div>
        )}
        {view === "mask" && (
          <div className="viewer__badge">
            Subject mask · red is backdrop · painting {maskMode === "add" ? "subject back in" : "backdrop"}
          </div>
        )}
      </main>

      <aside className="sidebar">
        <div className="sidebar__bar">
          <PresetMenu
            presets={presets}
            disabled={!engineReady || !image}
            targets={presetTargets().length}
            onApply={applyPreset}
            onSave={savePreset}
            onDelete={deletePreset}
          />
          <button
            className="sidebar__fold"
            onClick={toggleAllPanels}
            title={allCollapsed ? "Expand all panels" : "Collapse all panels"}
          >
            {allCollapsed ? <ChevronsUpDown size={14} /> : <ChevronsDownUp size={14} />}
            {allCollapsed ? "Expand all" : "Collapse all"}
          </button>
        </div>

        <Panel
          title="Tone"
          actions={
            <ResetButton
              disabled={!image || sameValues(toneParams, TONE_DEFAULTS)}
              onClick={() => updateTone(TONE_DEFAULTS)}
              what="Tone"
            />
          }
          {...panel("tone")}
        >
          <Slider
            label="Temperature"
            hint="White balance: left bluer, right warmer (amber); brightness is kept"
            min={-1}
            max={1}
            step={0.01}
            centred
            scale={TEMPERATURE_SCALE}
            value={toneParams.temperature}
            defaultValue={0}
            format={signed}
            disabled={!image}
            onChange={(v) => updateTone({ ...toneRef.current, temperature: v })}
          />
          <Slider
            label="Tint"
            hint="White balance: left greener, right more magenta; brightness is kept"
            min={-1}
            max={1}
            step={0.01}
            centred
            scale={TINT_SCALE}
            value={toneParams.tint}
            defaultValue={0}
            format={signed}
            disabled={!image}
            onChange={(v) => updateTone({ ...toneRef.current, tint: v })}
          />
          <Slider
            label="Exposure"
            hint="Brighten or darken the whole photo, in stops (EV)"
            min={-2.5}
            max={2.5}
            step={0.05}
            centred
            value={toneParams.ev}
            defaultValue={0}
            format={(v) => `${v > 0 ? "+" : ""}${v.toFixed(2)} EV`}
            disabled={!image}
            onChange={(v) => updateTone({ ...toneRef.current, ev: v })}
          />
          <Slider
            label="Dehaze"
            hint="Right: clear haze or a washed-out look (deeper blacks, more contrast and colour). Left: add a soft, misty haze"
            min={-1}
            max={1}
            step={0.01}
            centred
            value={toneParams.dehaze}
            defaultValue={0}
            format={signed}
            disabled={!image}
            onChange={(v) => updateTone({ ...toneRef.current, dehaze: v })}
          />
          <Slider
            label="Vibrance"
            hint="Richer or more muted colour across the photo, like Lightroom's Vibrance: skin tones are protected and dull colours gain most"
            min={-1}
            max={1}
            step={0.01}
            centred
            value={toneParams.vibrance}
            defaultValue={0}
            format={signed}
            disabled={!image}
            onChange={(v) => updateTone({ ...toneRef.current, vibrance: v })}
          />
          <CurveEditor
            points={toneParams.curve}
            histogram={histogram}
            disabled={!image}
            onChange={(curve) => updateTone({ ...toneRef.current, curve })}
          />
          <div className="panel__buttons">
            <button
              disabled={!image || (view === "mask" || view === "area") || sameValues(toneParams, TONE_DEFAULTS)}
              onPointerEnter={() => prefetchWithout("tone")}
              {...holdHandlers(() => holdWithout("tone"), releaseCompare)}
              title="Hold to see the photo without the exposure and curves"
            >
              <Eye size={15} /> Hold for before
            </button>
          </div>
        </Panel>

        <Panel
          head={
            <div className="tabs panel__modes">
              <button
                className={!outdoor ? "active" : ""}
                disabled={!image}
                onClick={() => outdoor && toggleOutdoor()}
                title="Studio backdrop: smooth, even out and adjust the backdrop"
              >
                <Layers size={13} /> Backdrop
                {!outdoor && modeIsAuto && <span className="auto-tag">auto</span>}
              </button>
              <button
                className={outdoor ? "active" : ""}
                disabled={!image}
                onClick={() => !outdoor && toggleOutdoor()}
                title="Outdoor: no backdrop step, for portraits not shot on a backdrop"
              >
                <Sun size={13} /> Outdoor
                {outdoor && modeIsAuto && <span className="auto-tag">auto</span>}
              </button>
            </div>
          }
          actions={
            <ResetButton
              disabled={!image || sameValues(params, DEFAULTS)}
              onClick={resetBackdrop}
              what="Backdrop"
            />
          }
          {...panel("backdrop")}
        >
          {SLIDERS.map((s) => (
            <Slider
              key={s.key}
              label={s.label}
              hint={s.hint}
              min={s.min}
              max={s.max}
              step={s.step}
              value={params[s.key]}
              defaultValue={DEFAULTS[s.key]}
              disabled={!image || outdoor}
              format={s.format}
              onChange={(v) => updateParam(s.key, v)}
            />
          ))}
          <div className="panel__buttons">
            <button
              disabled={!image || view === "mask" || outdoor}
              onPointerEnter={() => prefetchWithout("backdrop")}
              {...holdHandlers(() => holdWithout("backdrop"), releaseCompare)}
              title="Hold to see the photo without backdrop smoothing"
            >
              <Eye size={15} /> Hold for before
            </button>
            <button
              disabled={!image || outdoor}
              onClick={showMask}
              className={view === "mask" ? "active" : ""}
            >
              <Layers size={15} /> Mask
            </button>
          </div>
          {view === "mask" && (
            <>
              <div className="panel__buttons">
                <button
                  className={maskMode === "add" ? "active" : ""}
                  onClick={() => setMaskMode("add")}
                  title="Paint to mark subject, protected from smoothing (X swaps)"
                >
                  <Plus size={15} /> Add
                </button>
                <button
                  className={maskMode === "subtract" ? "active" : ""}
                  onClick={() => setMaskMode("subtract")}
                  title="Paint to mark backdrop, so it gets smoothed (X swaps)"
                >
                  <Minus size={15} /> Subtract
                </button>
                <button
                  disabled={maskEdits === 0 || maskPending}
                  onClick={undoMaskEdit}
                  title="Undo the last mask edit (Ctrl+Z)"
                >
                  <Undo2 size={15} />
                </button>
                <button
                  disabled={maskEdits === 0 || maskPending}
                  onClick={clearMaskEdits}
                  title="Back to the AI's original mask"
                >
                  <Trash2 size={15} />
                </button>
              </div>
              <Slider
                label="Brush size"
                hint="Mask brush size ([ and ] keys)"
                min={BRUSH.min}
                max={BRUSH.max}
                step={BRUSH.step}
                value={maskBrushRadius}
                defaultValue={BRUSH.default * 2}
                format={(v) => (v * 100).toFixed(1)}
                onChange={setMaskBrushRadius}
              />
              <p className="panel__model">
                Paint on the photo to correct the mask. [ and ] resize the brush.
                {maskEdits > 0 && ` ${maskEdits} edit${maskEdits === 1 ? "" : "s"}.`}
              </p>
            </>
          )}
          {outdoor ? (
            <p className="panel__model">
              Outdoor: the backdrop step is off for this photo. Removals, skin and eyes still work.
            </p>
          ) : (
            maskModel && (
              <p className="panel__model">
                Subject mask: {maskModel.name} · {maskModel.licence}
              </p>
            )
          )}
        </Panel>

        <Panel
          title="Skin"
          actions={
            <ResetButton
              disabled={!image || (sameValues(skinParams, SKIN_DEFAULTS) && opacity.skin === 1)}
              onClick={resetSkin}
              what="Skin (Face, Neck and Body)"
            />
          }
          {...panel("skin")}
        >
          <OpacityGroup
            value={opacity.skin}
            onChange={(v) => updateOpacity("skin", v)}
            disabled={!image}
            what="Skin (Face, Neck and Body)"
          >
            <div className="tabs">
              {SKIN_TABS.map((t) => (
                <button
                  key={t.key}
                  className={skinTab === t.key ? "active" : ""}
                  onClick={() => setSkinTab(t.key)}
                >
                  {t.label}
                </button>
              ))}
            </div>
            {SKIN_SLIDERS.map((s) => (
              <Slider
                key={`${skinTab}-${s.key}`}
                label={s.label}
                hint={s.hint}
                min={s.min}
                max={1}
                step={0.01}
                centred={s.centred}
                value={skinParams[skinTab][s.key]}
                defaultValue={SKIN_REGION_DEFAULTS[s.key]}
                {...eye(`skin.${skinTab}.${s.key}`)}
                format={
                  s.key === "shine"
                    ? (v) => (v < -0.005 ? `Matte ${(-v).toFixed(2)}` : v > 0.005 ? `Gloss ${v.toFixed(2)}` : "Natural")
                    : undefined
                }
                disabled={!image}
                onChange={(v) => updateSkin(skinTab, s.key, v)}
              />
            ))}
            {skinTab === "neck" && (
              <Slider
                label="Neck lines"
                hint="Soften the horizontal creases across the neck; some of a deep fold is always kept"
                min={0}
                max={1}
                step={0.01}
                value={skinParams.neck.neck_lines}
                defaultValue={0}
                {...eye("skin.neck.neck_lines")}
                disabled={!image}
                onChange={(v) => updateSkin("neck", "neck_lines", v)}
              />
            )}
            {skinTab === "face" && (
              <>
                <div className="panel__head">
                  <h3 className="panel__subhead">Wrinkles</h3>
                  <ResetButton
                    disabled={!image || WRINKLE_SLIDERS.every((s) => skinParams.face[s.key] === 0)}
                    onClick={resetWrinkles}
                    what="Wrinkles"
                  />
                </div>
                {WRINKLE_SLIDERS.map((s) => (
                  <Slider
                    key={s.key}
                    label={s.label}
                    hint={s.hint}
                    min={0}
                    max={1}
                    step={0.01}
                    value={skinParams.face[s.key]}
                    defaultValue={0}
                    {...eye(`skin.face.${s.key}`)}
                    disabled={!image}
                    onChange={(v) => updateSkin("face", s.key, v)}
                  />
                ))}
              </>
            )}
            <div className="panel__buttons">
              <button
                disabled={
                  !image || (view === "mask" || view === "area") || Object.values(skinParams[skinTab]).every((v) => v === 0)
                }
                onPointerEnter={() => prefetchWithout(`skin_${skinTab}`)}
                {...holdHandlers(() => holdWithout(`skin_${skinTab}`), releaseCompare)}
                title={`Hold to see the photo without the ${skinTab} skin retouching`}
              >
                <Eye size={15} /> Hold for before
              </button>
              <button
                disabled={!image || view === "mask"}
                className={view === "area" && area === skinTab ? "active" : ""}
                onClick={() => showArea(skinTab)}
                title={`Show and correct where the ${skinTab} skin tools work`}
              >
                <Layers size={15} /> Refine area
              </button>
            </div>
            <p className="panel__model">
              Facial hair, eyes, brows and lips are left alone automatically. Eye wrinkles and crow's
              feet are in the Eyes panel. Neck and Body find the skin by the person's own colour, so
              clothes close to skin colour may be retouched too.
            </p>
          </OpacityGroup>
        </Panel>

        <Panel
          title="Dodge & Burn"
          actions={
            <ResetButton
              disabled={!image || (sameValues(dodgeBurn, DODGE_BURN_DEFAULTS) && opacity.dodge_burn === 1)}
              onClick={resetDodgeBurn}
              what="Dodge & Burn"
            />
          }
          {...panel("dodge_burn")}
        >
          <OpacityGroup
            value={opacity.dodge_burn}
            onChange={(v) => updateOpacity("dodge_burn", v)}
            disabled={!image}
            what="Dodge & Burn"
          >
            {DODGE_BURN_SLIDERS.map((s) => (
              <Slider
                key={s.key}
                label={s.label}
                hint={s.hint}
                min={0}
                max={1}
                step={0.01}
                value={dodgeBurn[s.key]}
                defaultValue={0}
                {...eye(`dodgeBurn.${s.key}`)}
                disabled={!image}
                onChange={(v) => updateDodgeBurn(s.key, v)}
              />
            ))}
            <div className="panel__buttons">
              <button
                disabled={!image || (view === "mask" || view === "area") || sameValues(dodgeBurn, DODGE_BURN_DEFAULTS)}
                onPointerEnter={() => prefetchWithout("dodge_burn")}
                {...holdHandlers(() => holdWithout("dodge_burn"), releaseCompare)}
                title="Hold to see the photo without dodge & burn"
              >
                <Eye size={15} /> Hold for before
              </button>
              <button
                disabled={!image || view === "mask"}
                className={view === "area" && area === "dodge_burn" ? "active" : ""}
                onClick={() => showArea("dodge_burn")}
                title="Show and correct where Dodge & Burn works"
              >
                <Layers size={15} /> Refine area
              </button>
            </div>
            <p className="panel__model">
              Follows the light already on the face, so nothing is painted against it. Skin texture and colour are kept.
            </p>
          </OpacityGroup>
        </Panel>

        <Panel
          title="Relight"
          actions={
            <ResetButton disabled={!image || !relight} onClick={() => updateRelight(null)} what="Relight" />
          }
          {...panel("relight")}
        >
          {!relight ? (
            <>
              <p className="panel__model">
                A radial light over the face: brighten, darken, warm or cool it, fading out toward the
                edge.
              </p>
              <div className="panel__buttons">
                <button disabled={!image} onClick={addRelight} title="Place a radial light over the face">
                  <Circle size={15} /> Add on face
                </button>
              </div>
            </>
          ) : (
            <>
              <Slider
                label="Exposure"
                hint="Brighten (or darken) the light, in stops (EV)"
                min={-2}
                max={2}
                step={0.05}
                centred
                value={relight.exposure}
                defaultValue={0}
                {...eye("relight.exposure")}
                format={(v) => `${v > 0 ? "+" : ""}${v.toFixed(2)} EV`}
                disabled={!image}
                onChange={(v) => updateRelight({ ...relightRef.current!, exposure: v })}
              />
              <Slider
                label="Warmth"
                hint="Warmer (amber) or cooler (blue) light; brightness is kept"
                min={-1}
                max={1}
                step={0.01}
                centred
                scale={TEMPERATURE_SCALE}
                value={relight.warmth}
                defaultValue={0}
                {...eye("relight.warmth")}
                format={signed}
                disabled={!image}
                onChange={(v) => updateRelight({ ...relightRef.current!, warmth: v })}
              />
              <Slider
                label="Feather"
                hint="How gradually the light fades out toward the edge of the circle: the dashed ring is where the fade starts (drag it in Edit circle)"
                min={0}
                max={1}
                step={0.01}
                value={relight.feather}
                defaultValue={RELIGHT_LOOK.feather}
                format={(v) => `${Math.round(v * 100)}`}
                disabled={!image}
                onChange={(v) => updateRelight({ ...relightRef.current!, feather: v })}
              />
              <div className="panel__buttons">
                <button
                  className={relightEditing ? "active" : ""}
                  disabled={!image || view !== "result"}
                  onClick={() => {
                    if (!relightEditing) setBrushOn(false);
                    setRelightEditing(!relightEditing);
                  }}
                  title="Show the circle: drag inside to move it, the edge handles to stretch it, the top handle to turn it, the dashed ring to change the feather (Enter when done)"
                >
                  <Circle size={15} /> {relightEditing ? "Done" : "Edit circle"}
                </button>
                <button
                  disabled={!image || (view === "mask" || view === "area") || !relightLook(relight)}
                  onPointerEnter={() => prefetchWithout("relight")}
                  {...holdHandlers(() => holdWithout("relight"), releaseCompare)}
                  title="Hold to see the photo without the relight"
                >
                  <Eye size={15} /> Hold for before
                </button>
              </div>
            </>
          )}
        </Panel>

        <Panel
          title="Eyes"
          actions={
            <ResetButton
              disabled={!image || (sameValues(eyesParams, EYE_DEFAULTS) && opacity.eyes === 1)}
              onClick={resetEyes}
              what="Eyes"
            />
          }
          {...panel("eyes")}
        >
          <OpacityGroup
            value={opacity.eyes}
            onChange={(v) => updateOpacity("eyes", v)}
            disabled={!image}
            what="Eyes"
          >
            {EYE_SLIDERS.map((s) => (
              <Slider
                key={s.key}
                label={s.label}
                hint={s.hint}
                min={s.centred ? -1 : 0}
                max={1}
                step={0.01}
                centred={s.centred}
                scale={s.key === "iris_hue" && irisScale ? `linear-gradient(to right, ${irisScale.join(", ")})` : s.scale}
                format={s.centred ? signed : undefined}
                value={eyesParams[s.key]}
                defaultValue={EYE_DEFAULTS[s.key]}
                {...eye(`eyes.${s.key}`)}
                disabled={!image}
                onChange={(v) => updateEyes(s.key, v)}
              />
            ))}
            <div className="panel__buttons">
              <button
                disabled={
                  !image || (view === "mask" || view === "area") || Object.values(eyesParams).every((v) => v === 0)
                }
                onPointerEnter={() => prefetchWithout("eyes")}
                {...holdHandlers(() => holdWithout("eyes"), releaseCompare)}
                title="Hold to see the photo without the eye edits"
              >
                <Eye size={15} /> Hold for before
              </button>
            </div>
            <p className="panel__model">
              {faces === 0
                ? "No faces found in this photo."
                : faces
                  ? `${faces} face${faces === 1 ? "" : "s"} · MediaPipe Face Landmarker · Apache-2.0`
                  : "Faces are found when you first move a slider. MediaPipe · Apache-2.0"}
            </p>
          </OpacityGroup>
        </Panel>

        <Panel
          title="Mouth"
          actions={
            <ResetButton
              disabled={!image || (sameValues(mouthParams, MOUTH_DEFAULTS) && opacity.mouth === 1)}
              onClick={resetMouth}
              what="Mouth"
            />
          }
          {...panel("mouth")}
        >
          <OpacityGroup
            value={opacity.mouth}
            onChange={(v) => updateOpacity("mouth", v)}
            disabled={!image}
            what="Mouth"
          >
            {MOUTH_SLIDERS.map((s) => (
              <Slider
                key={s.key}
                label={s.label}
                hint={s.hint}
                min={s.centred ? -1 : 0}
                max={1}
                step={0.01}
                centred={s.centred}
                scale={s.scale}
                value={mouthParams[s.key]}
                defaultValue={MOUTH_DEFAULTS[s.key]}
                {...eye(`mouth.${s.key}`)}
                format={s.centred ? (v) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2)) : undefined}
                disabled={!image}
                onChange={(v) => updateMouth(s.key, v)}
              />
            ))}
            <div className="panel__buttons">
              <button
                disabled={!image || (view === "mask" || view === "area") || sameValues(mouthParams, MOUTH_DEFAULTS)}
                onPointerEnter={() => prefetchWithout("mouth")}
                {...holdHandlers(() => holdWithout("mouth"), releaseCompare)}
                title="Hold to see the photo without the mouth edits"
              >
                <Eye size={15} /> Hold for before
              </button>
            </div>
          </OpacityGroup>
        </Panel>

        <Panel title="Clothes" {...panel("clothes")}>
          <Slider
            label="Fine creases"
            hint="Soften creases by evening out their shadows; use Patch to remove one completely"
            min={0}
            max={1}
            step={0.01}
            value={creases}
            defaultValue={0}
            {...eye("creases")}
            disabled={!image}
            onChange={updateCreases}
          />
          <div className="panel__buttons">
            <button
              disabled={!image || (view === "mask" || view === "area") || creases === 0}
              onPointerEnter={() => prefetchWithout("clothes")}
              {...holdHandlers(() => holdWithout("clothes"), releaseCompare)}
              title="Hold to see the photo without crease smoothing"
            >
              <Eye size={15} /> Hold for before
            </button>
            <button
              disabled={!image || view === "mask"}
              className={view === "area" && area === "clothes" ? "active" : ""}
              onClick={() => showArea("clothes")}
              title="Show and correct where the Clothes tool works"
            >
              <Layers size={15} /> Refine area
            </button>
          </div>
          <p className="panel__model">
            Softens creases by evening out their shadows, so they look less obvious. For creases that need to
            go completely, use the Patch tool.
          </p>
        </Panel>

      </aside>


      {strip.length > 0 && (
        <FilmStrip
          progress={prep}
          items={strip.map((i) =>
            // The photo being edited: its dot follows the sliders live.
            i.path === image?.path
              ? {
                  ...i,
                  edited: !sameSettings(
                    { backdrop: params, outdoor, eyes: eyesParams, skin: skinParams, mouth: mouthParams, dodgeBurn, creases, tone: toneParams, crop, opacity, relight, off },
                    ALL_DEFAULTS,
                  ),
                }
              : i,
          )}
          active={image?.path ?? null}
          selected={selected}
          canPaste={copied !== null}
          onActivate={(p) => loadImage(p)}
          onSelect={setSelected}
          onRemove={removeFromStrip}
          onCopy={copySettings}
          onPaste={pasteSettings}
        />
      )}

      <footer className="app-footer">
        <span className="oss-note">
          RapidRetouch is free and open-source software, licensed AGPL-3.0. Developed and maintained
          by Chris Cork Photography.
        </span>
        <span className="app-footer__status">
          {error ? (
            <span className="statusbar__error" onClick={() => setError(null)} title="Click to dismiss">
              {error}
            </span>
          ) : !engineReady && !setup ? (
            <span className="app-footer__starting">
              Starting the AI engine…
              <span className="app-footer__busy" />
            </span>
          ) : countdown !== null ? (
            <span>
              Retouching in{" "}
              <span key={countdown} className="countdown">
                {countdown}
              </span>
            </span>
          ) : (
            <span>
              {busy && <span className="spinner" />} {status}
            </span>
          )}
        </span>
        <a
          className="coffee-link"
          href={COFFEE_URL}
          onClick={(e) => {
            e.preventDefault();
            openUrl(COFFEE_URL);
          }}
          title="Buy Chris a coffee"
        >
          Feed Chris' coffee addiction <Smile size={16} color="#ffcc33" />{" "}
          <Coffee size={16} color="#ffcc33" />
        </a>
      </footer>

      {exportRows && (
        <ExportDialog
          rows={exportRows}
          format={exportFormat}
          size={exportSize}
          onSize={(sz) => {
            setExportSize(sz);
            try {
              localStorage.setItem(EXPORT_SIZE_KEY, JSON.stringify(sz));
            } catch {
              // only a convenience
            }
          }}
          folder={exportFolder}
          running={exportRunning}
          cancelling={exportCancelling}
          onFormat={(f) => {
            setExportFormat(f);
            try {
              localStorage.setItem(EXPORT_FORMAT_KEY, f);
            } catch {
              // only a convenience
            }
          }}
          onChooseFolder={chooseExportFolder}
          onSameFolder={() => setExportFolder(null)}
          onStart={runExport}
          onCancel={() => {
            exportCancel.current = true;
            setExportCancelling(true);
          }}
          onClose={() => setExportRows(null)}
        />
      )}
      {setup && <SetupScreen {...setup} />}
      {aboutModels && <About models={aboutModels} onClose={() => setAboutModels(null)} />}
      {licencePrompt && (
        <div className="modal">
          <div className="modal__box">
            <h2>Licence required</h2>
            <p>
              <strong>{licencePrompt.model}</strong> is licensed <strong>{licencePrompt.licence}</strong>.
              It may not be usable for paid client work.
            </p>
            {licencePrompt.licence_url && (
              <p className="modal__link">Licence: {licencePrompt.licence_url}</p>
            )}
            <div className="modal__buttons">
              <button onClick={() => setLicencePrompt(null)}>Cancel</button>
              <button className="primary" onClick={acceptLicence}>
                Accept and download
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

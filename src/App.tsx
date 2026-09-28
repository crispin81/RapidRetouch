import { useCallback, useEffect, useRef, useState } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
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
} from "lucide-react";
import {
  BackdropParams,
  EngineError,
  EyesParams,
  SkinParams,
  SkinRegion,
  ModelInfo,
  OpenResult,
  call,
  jpegSrc,
  onEngineEvent,
} from "./api";
import PaintOverlay from "./PaintOverlay";
import Slider from "./Slider";
import Viewer, { Detail, ViewerHandle } from "./Viewer";
import TitleBar from "./TitleBar";
import FilmStrip, { StripItem } from "./FilmStrip";

interface PhotoSettings {
  backdrop: BackdropParams;
  eyes: EyesParams;
  skin: SkinParams;
}
const sameSettings = (a: PhotoSettings, b: PhotoSettings) =>
  JSON.stringify(a) === JSON.stringify(b);

const COFFEE_URL = "https://buymeacoffee.com/chriscorkphotography";
// No tutorial video yet: the banner does nothing until this is set.
const TUTORIAL_VIDEO_URL: string | null = null;
const TUTORIAL_DISMISSED_KEY = "retouch.tutorialDismissed";

const PREVIEW_EDGE = 2048; // keep in sync with the engine's server.PREVIEW_EDGE
// Scanning: a photo is only opened and processed after this long on it (or as
// soon as it's edited), so stepping through a shoot isn't slowed down.
const OPEN_DELAY_MS = 3000;
import "./App.css";

const DEFAULTS: BackdropParams = {
  strength: 1,
  smoothness: 1.5,
  evenness: 0.75,
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
    min: -3,
    max: 1,
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
  catchlight: 0,
  veins: 0,
};

const EYE_SLIDERS: { key: keyof EyesParams; label: string; hint: string }[] = [
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
    key: "catchlight",
    label: "Catch light",
    hint: "Brighten and crisp up the existing catchlights (none are added)",
  },
];

// "compare": holding a panel's before button — the result without that step.
type View = "result" | "before" | "mask" | "compare";
type Step = "removals" | "backdrop" | "skin" | "eyes";

const SKIN_REGION_DEFAULTS: SkinRegion = { blemishes: 0, smooth: 0, even: 0, shine: 0 };
const SKIN_DEFAULTS: SkinParams = {
  face: SKIN_REGION_DEFAULTS,
  neck: SKIN_REGION_DEFAULTS,
  body: SKIN_REGION_DEFAULTS,
};
type SkinTab = keyof SkinParams;
const ALL_DEFAULTS = { backdrop: DEFAULTS, eyes: EYE_DEFAULTS, skin: SKIN_DEFAULTS };
const SKIN_TABS: { key: SkinTab; label: string; ready: boolean }[] = [
  { key: "face", label: "Face", ready: true },
  { key: "neck", label: "Neck", ready: false },
  { key: "body", label: "Body", ready: false },
];
const SKIN_SLIDERS: {
  key: keyof SkinRegion;
  label: string;
  hint: string;
  min: number;
  centred?: boolean;
}[] = [
  { key: "blemishes", label: "Blemishes", hint: "Heal small spots and marks; skin texture is kept", min: 0 },
  { key: "smooth", label: "Smooth", hint: "Even out blotchy light and shade; pores and fine texture are kept", min: 0 },
  { key: "even", label: "Even tone", hint: "Move red or blotchy patches toward the person's own skin tone", min: 0 },
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
const BRUSH = { min: 0.002, max: 0.05, step: 0.001, default: 0.01 };
const clampBrush = (r: number) => Math.min(BRUSH.max, Math.max(BRUSH.min, r));

// Remembered across launches; storage can be unavailable, so never let it throw.
const LAST_DIR_KEY = "retouch.lastDir";
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

export default function App() {
  const [engineReady, setEngineReady] = useState(false);
  const [status, setStatus] = useState("Starting engine…");
  const [error, setError] = useState<string | null>(null);
  const [image, setImage] = useState<(OpenResult & { path: string }) | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [mask, setMask] = useState<string | null>(null);
  const [view, setView] = useState<View>("result");
  const [params, setParams] = useState<BackdropParams>(DEFAULTS);
  const [eyesParams, setEyesParams] = useState<EyesParams>(EYE_DEFAULTS);
  const [skinParams, setSkinParams] = useState<SkinParams>(SKIN_DEFAULTS);
  const [skinTab, setSkinTab] = useState<SkinTab>("face");
  const [faces, setFaces] = useState<number | null>(null);
  const [maskModel, setMaskModel] = useState<ModelInfo | null>(null);
  const [licencePrompt, setLicencePrompt] = useState<EngineError | null>(null);
  const [busy, setBusy] = useState(false);
  const [brushOn, setBrushOn] = useState(false);
  const [brushRadius, setBrushRadius] = useState(BRUSH.default);
  // The mask brush has its own size: mask edits (long soft strokes along hair)
  // usually want a different size from spot removals.
  const [maskBrushRadius, setMaskBrushRadius] = useState(BRUSH.default * 2);
  const [removals, setRemovals] = useState(0);
  // Remove-panel brush mode: LaMa fill, or glasses reflection (alpha).
  const [removeMode, setRemoveMode] = useState<"fill" | "reflection">("fill");
  const [reflectionStrength, setReflectionStrength] = useState(1);
  const [removing, setRemoving] = useState(false);
  const [maskMode, setMaskMode] = useState<"add" | "subtract">("add");
  const [maskEdits, setMaskEdits] = useState(0);
  const [maskPending, setMaskPending] = useState(false);
  const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null);
  const [compareImg, setCompareImg] = useState<string | null>(null);
  const [compareStep, setCompareStep] = useState<Step | null>(null);
  const [zoomTool, setZoomTool] = useState(false);
  const [zoomLabel, setZoomLabel] = useState("Fit");
  const [detail, setDetail] = useState<Detail | null>(null);
  const viewerRef = useRef<ViewerHandle>(null);
  // Film strip: every photo opened this session, each with its own settings.
  const [strip, setStrip] = useState<StripItem[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [copied, setCopied] = useState<PhotoSettings | null>(null);
  const settingsByPath = useRef(new Map<string, PhotoSettings>());
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
  const inFlight = useRef(false);
  const dirty = useRef(false);

  useEffect(() => {
    const unlisten = onEngineEvent((e) => {
      if (e.event === "status") {
        setStatus(e.message);
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
        setStatus(r.cuda ? `Ready · ${r.device}` : "Ready · no GPU found, running on CPU (slow)");
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
    try {
      if (!(await ensureOpen())) return;
      do {
        dirty.current = false;
        const forPath = openedPath.current;
        const r = await call<{ preview: string; faces: number | null }>("render", {
          backdrop: paramsRef.current,
          eyes: eyesRef.current,
          skin: skinRef.current,
        });
        // Moved on to another photo meanwhile: don't show this one's result.
        if (forPath !== activePath.current) break;
        setResult(r.preview);
        setFaces(r.faces);
      } while (dirty.current);
      setStatus("Ready");
    } catch (e) {
      handleError(e);
    } finally {
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
        const r = await call<{ preview: string; removals: number }>(method, {
          ...extra,
          backdrop: paramsRef.current,
          eyes: eyesRef.current,
          skin: skinRef.current,
        });
        setResult(r.preview);
        setRemovals(r.removals);
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
      removeMode === "reflection"
        ? { points, radius, kind: "reflection", strength: reflectionStrength }
        : { points, radius },
    );

  // Brush and Glasses pick the brush mode; clicking the active one turns it off.
  const pickRemoveBrush = (mode: "fill" | "reflection") => {
    setBrushOn((on) => !(on && removeMode === mode));
    setRemoveMode(mode);
    setZoomTool(false);
  };
  const undoRemove = () => removalCall("undo_remove");
  const clearRemovals = () => removalCall("clear_removals");

  // Keyboard: Ctrl+Z undoes a removal, B toggles the brush, [ and ] resize it.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!image || e.target instanceof HTMLInputElement) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (view === "mask") {
          if (maskEdits > 0 && !maskPending) undoMaskEdit();
        } else if (removals > 0 && !removing) undoRemove();
      } else if (view === "mask" && (e.key === "x" || e.key === "X")) {
        setMaskMode((m) => (m === "add" ? "subtract" : "add"));
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
      } else if (!e.ctrlKey && !e.metaKey && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
        e.preventDefault();
        step(e.key === "ArrowRight" ? 1 : -1);
      } else if (e.key === "[" || e.key === "]") {
        // Resize whichever brush is in use.
        const factor = e.key === "]" ? 1.2 : 1 / 1.2;
        const setRadius = view === "mask" ? setMaskBrushRadius : setBrushRadius;
        setRadius((r) => clampBrush(r * factor));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const currentSettings = (): PhotoSettings => ({
    backdrop: paramsRef.current,
    eyes: eyesRef.current,
    skin: skinRef.current,
  });

  const applySettings = (st: PhotoSettings) => {
    paramsRef.current = st.backdrop;
    eyesRef.current = st.eyes;
    skinRef.current = st.skin;
    setSkinParams(st.skin);
    setParams(st.backdrop);
    setEyesParams(st.eyes);
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
    window.clearTimeout(openTimer.current);
    stopCountdown();
    setError(null);
    setResult(null);
    setMask(null);
    setView("result");
    setFaces(null);
    setRemovals(0);
    setMaskEdits(0);
    applySettings(settingsByPath.current.get(path) ?? ALL_DEFAULTS);

    // Back to the photo the engine already has: no need to wait.
    if (openedPath.current === path && openedInfo.current) {
      setImage(openedInfo.current);
      setRemovals(openedInfo.current.removals);
      setMaskEdits(openedInfo.current.mask_edits);
      render();
      return;
    }
    try {
      const q = await call<{ image: string; width: number; height: number }>("thumbnail", {
        path,
        edge: PREVIEW_EDGE,
      });
      if (activePath.current !== path) return;
      setImage({
        path,
        width: q.width,
        height: q.height,
        bit_depth: 0, // not known until opened
        preview: q.image,
        removals: 0,
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

  // Thumbnails, one at a time so they don't hold up the photo being edited.
  const loadingThumb = useRef(false);
  useEffect(() => {
    const next = strip.find((i) => !i.thumb);
    if (!next || loadingThumb.current || !engineReady) return;
    loadingThumb.current = true;
    call<{ image: string }>("thumbnail", { path: next.path })
      .then((r) =>
        setStrip((items) => items.map((i) => (i.path === next.path ? { ...i, thumb: r.image } : i))),
      )
      .catch(() =>
        // Leave a placeholder rather than retrying forever.
        setStrip((items) => items.map((i) => (i.path === next.path ? { ...i, thumb: "" } : i))),
      )
      .finally(() => {
        loadingThumb.current = false;
      });
  }, [strip, engineReady]);

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

  const copySettings = () => {
    setCopied(currentSettings());
    setStatus("Settings copied");
  };

  const pasteSettings = () => {
    if (!copied) return;
    let pasted = 0;
    for (const path of selected) {
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

  const onMaskStroke = (points: [number, number][], radius: number) =>
    maskCall("mask_paint", { mode: maskMode, points, radius });
  const undoMaskEdit = () => maskCall("undo_mask_edit");
  const clearMaskEdits = () => maskCall("clear_mask_edits");

  // Each panel's hold-for-before shows everything except that panel's step, so
  // each edit can be judged on its own. Releasing before the render returns
  // must not leave the comparison showing, hence the token.
  const holdWithout = async (step: Step) => {
    const token = ++compareToken.current;
    setCompareStep(step);
    setView("compare");
    try {
      if (!(await ensureOpen())) return;
      const r = await call<{ preview: string }>("render", {
        backdrop: step === "backdrop" ? null : paramsRef.current,
        eyes: step === "eyes" ? null : eyesRef.current,
        skin: step === "skin" ? null : skinRef.current,
        removals: step !== "removals",
      });
      if (token === compareToken.current) setCompareImg(r.preview);
    } catch (e) {
      handleError(e);
    }
  };
  const releaseCompare = () => {
    compareToken.current++;
    setCompareImg(null);
    setView((v) => (v === "compare" ? "result" : v));
  };

  // Full-resolution detail when zoomed in. The tile must show exactly what the
  // viewer shows (result, original, or a per-step before), so each request uses
  // that view's render arguments, and a tile for anything else is dropped.
  const detailArgs = (): Record<string, unknown> | null => {
    if (view === "mask") return null;
    if (view === "before") return { backdrop: null, eyes: null, removals: false };
    const all = {
      backdrop: paramsRef.current,
      eyes: eyesRef.current,
      skin: skinRef.current,
      removals: true,
    };
    if (view === "compare" && compareStep === "skin") return { ...all, skin: null };
    if (view === "compare" && compareStep === "backdrop") return { ...all, backdrop: null };
    if (view === "compare" && compareStep === "eyes") return { ...all, eyes: null };
    if (view === "compare" && compareStep === "removals") return { ...all, removals: false };
    return all;
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
      handleError(e);
    } finally {
      detailInFlight.current = false;
    }
  }, [handleError]);

  const onDetailNeeded = useCallback(
    (region: [number, number, number, number] | null, scale: number) => {
      wantedRegion.current = region ? { region, scale } : null;
      if (!region) setDetail(null);
      else requestDetail();
    },
    [requestDetail],
  );

  // Whenever the picture changes, the old tile is stale: drop it and re-ask.
  useEffect(() => {
    contentVersion.current++;
    setDetail(null);
    if (wantedRegion.current) requestDetail();
  }, [result, compareImg, view, compareStep, requestDetail]);

  const exportImage = async () => {
    if (!image || !(await ensureOpen())) return;
    const stem = image.path.replace(/\.[^./]+$/, "");
    const path = await save({
      defaultPath: `${stem}_retouched.tif`,
      filters: [{ name: "TIFF", extensions: bothCases(["tif", "tiff"]) }],
    });
    if (!path) return;
    setBusy(true);
    try {
      const r = await call<{ path: string }>("export", {
        path,
        backdrop: paramsRef.current,
        eyes: eyesRef.current,
        skin: skinRef.current,
      });
      setStatus(`Exported ${r.path.split("/").pop()}`);
    } catch (e) {
      handleError(e);
    } finally {
      setBusy(false);
    }
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
      : view === "compare"
        ? (compareImg ?? result)
        : view === "before" || !result
          ? image?.preview
          : result;

  return (
    <div className="app">
      <TitleBar />
      <header className="toolbar">
        {!tutorialDismissed && (
          <div className="video-link">
            <a
              className="video-link__cta"
              href={TUTORIAL_VIDEO_URL ?? "#"}
              onClick={(e) => {
                e.preventDefault();
                if (TUTORIAL_VIDEO_URL) openUrl(TUTORIAL_VIDEO_URL);
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
        <button onClick={exportImage} disabled={!image || busy}>
          <Download size={16} /> Export
        </button>
        <button
          disabled={!image || view === "mask"}
          onPointerDown={() => setView("before")}
          onPointerUp={() => setView((v) => (v === "before" ? "result" : v))}
          onPointerLeave={() => setView((v) => (v === "before" ? "result" : v))}
          title="Hold to see the untouched original"
        >
          <Eye size={16} /> Original
        </button>
        <div className="toolbar__group">
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
          <button disabled={!image} onClick={() => viewerRef.current?.fit()} title="Fit (Ctrl+0)">
            <Maximize size={16} /> Fit
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
        <div className="toolbar__spacer" />
        {image && (
          <span className="toolbar__info">
            {image.path.split("/").pop()} · {image.width}×{image.height} ·{" "}
            {image.bit_depth ? `${image.bit_depth}-bit` : "preview"}
          </span>
        )}
      </header>

      <main className="viewer" onDragStart={(e) => e.preventDefault()}>
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
          empty={<div className="viewer__empty">Open a RAW file, 16-bit TIFF or JPEG to start</div>}
        >
          {({ panning }) => (
            <>
              {image && view === "result" && (
                <PaintOverlay
                  image={imgEl}
                  active={brushOn && !panning && !zoomTool}
                  radius={brushRadius}
                  pending={removing}
                  onStroke={onStroke}
                  colour={removeMode === "reflection" ? "rgba(64, 200, 255, 0.45)" : undefined}
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
            </>
          )}
        </Viewer>
        {view === "before" && <div className="viewer__badge">Original</div>}
        {view === "compare" && compareImg && (
          <div className="viewer__badge">Before · {compareStep}</div>
        )}
        {view === "mask" && (
          <div className="viewer__badge">
            Subject mask · red is backdrop · painting {maskMode === "add" ? "subject back in" : "backdrop"}
          </div>
        )}
      </main>

      <aside className="sidebar">
        <section className="panel">
          <h2>Remove</h2>
          <div className="panel__buttons">
            <button
              disabled={!image}
              onClick={() => pickRemoveBrush("fill")}
              className={brushOn && removeMode === "fill" ? "active" : ""}
              title="Paint over anything to remove it (B)"
            >
              <Paintbrush size={15} /> Brush
            </button>
            <button
              disabled={!image}
              onClick={() => pickRemoveBrush("reflection")}
              className={brushOn && removeMode === "reflection" ? "active" : ""}
              title="Glasses reflection (alpha): paint over a reflection on a lens to remove it"
            >
              <Glasses size={15} /> Glasses (alpha)
            </button>
            <button
              disabled={!image || removals === 0 || removing}
              onClick={undoRemove}
              title="Undo the last removal (Ctrl+Z)"
            >
              <Undo2 size={15} /> Undo
            </button>
            <button
              disabled={!image || removals === 0 || removing}
              onClick={clearRemovals}
              title="Undo every removal"
            >
              <Trash2 size={15} /> Clear
            </button>
          </div>
          <div className="panel__buttons">
            <button
              disabled={!image || removals === 0 || removing || view === "mask"}
              onPointerDown={() => holdWithout("removals")}
              onPointerUp={releaseCompare}
              onPointerLeave={releaseCompare}
              title="Hold to see the photo without your removals"
            >
              <Eye size={15} /> Hold for before
            </button>
          </div>
          <Slider
            label="Brush size"
            hint="Brush size ([ and ] keys)"
            min={BRUSH.min}
            max={BRUSH.max}
            step={BRUSH.step}
            value={brushRadius}
            defaultValue={BRUSH.default}
            format={(v) => (v * 100).toFixed(1)}
            disabled={!image}
            onChange={setBrushRadius}
          />
          {removeMode === "reflection" && (
            <Slider
              label="Reflection strength"
              hint="How much of the reflection each new stroke removes"
              min={0}
              max={1}
              step={0.05}
              value={reflectionStrength}
              defaultValue={1}
              disabled={!image}
              onChange={setReflectionStrength}
            />
          )}
          <p className="panel__model">
            {removeMode === "reflection"
              ? "Glasses reflection (alpha): paint just the reflection, a little past its edges, not the whole lens. A faint trace may remain; paint over it again to take more."
              : removals > 0
                ? `${removals} removal${removals === 1 ? "" : "s"} · LaMa · Apache-2.0`
                : "Paint over a distraction; it's filled when you let go. Fill: LaMa · Apache-2.0"}
          </p>
        </section>

        <section className="panel">
          <h2>Backdrop</h2>
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
              format={s.format}
              disabled={!image}
              onChange={(v) => updateParam(s.key, v)}
            />
          ))}
          <div className="panel__buttons">
            <button
              disabled={!image || view === "mask"}
              onPointerDown={() => holdWithout("backdrop")}
              onPointerUp={releaseCompare}
              onPointerLeave={releaseCompare}
              title="Hold to see the photo without backdrop smoothing"
            >
              <Eye size={15} /> Hold for before
            </button>
            <button disabled={!image} onClick={showMask} className={view === "mask" ? "active" : ""}>
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
          {maskModel && (
            <p className="panel__model">
              Subject mask: {maskModel.name} · {maskModel.licence}
            </p>
          )}
        </section>

        <section className="panel">
          <h2>Skin</h2>
          <div className="tabs">
            {SKIN_TABS.map((t) => (
              <button
                key={t.key}
                className={skinTab === t.key ? "active" : ""}
                disabled={!t.ready}
                onClick={() => setSkinTab(t.key)}
                title={t.ready ? undefined : "Coming next"}
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
              format={
                s.key === "shine"
                  ? (v) => (v < -0.005 ? `Matte ${(-v).toFixed(2)}` : v > 0.005 ? `Gloss ${v.toFixed(2)}` : "Natural")
                  : undefined
              }
              disabled={!image}
              onChange={(v) => updateSkin(skinTab, s.key, v)}
            />
          ))}
          <div className="panel__buttons">
            <button
              disabled={
                !image ||
                view === "mask" ||
                Object.values(skinParams).every((r) => Object.values(r).every((v) => v === 0))
              }
              onPointerDown={() => holdWithout("skin")}
              onPointerUp={releaseCompare}
              onPointerLeave={releaseCompare}
              title="Hold to see the photo without the skin retouching"
            >
              <Eye size={15} /> Hold for before
            </button>
          </div>
          <p className="panel__model">
            Facial hair, eyes, brows and lips are left alone automatically. Neck and Body are coming
            next.
          </p>
        </section>

        <section className="panel">
          <h2>Eyes</h2>
          {EYE_SLIDERS.map((s) => (
            <Slider
              key={s.key}
              label={s.label}
              hint={s.hint}
              min={0}
              max={1}
              step={0.01}
              value={eyesParams[s.key]}
              defaultValue={EYE_DEFAULTS[s.key]}
              disabled={!image}
              onChange={(v) => updateEyes(s.key, v)}
            />
          ))}
          <div className="panel__buttons">
            <button
              disabled={
                !image || view === "mask" || Object.values(eyesParams).every((v) => v <= 0)
              }
              onPointerDown={() => holdWithout("eyes")}
              onPointerUp={releaseCompare}
              onPointerLeave={releaseCompare}
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
        </section>
      </aside>


      {strip.length > 0 && (
        <FilmStrip
          items={strip.map((i) =>
            // The photo being edited: its dot follows the sliders live.
            i.path === image?.path
              ? {
                  ...i,
                  edited: !sameSettings(
                    { backdrop: params, eyes: eyesParams, skin: skinParams },
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
          Feed Chris' coffee addiction <Smile size={13} color="#ffcc33" />{" "}
          <Coffee size={13} color="#ffcc33" />
        </a>
      </footer>

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

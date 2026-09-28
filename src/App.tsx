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
} from "lucide-react";
import {
  BackdropParams,
  EngineError,
  EyesParams,
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

const COFFEE_URL = "https://buymeacoffee.com/chriscorkphotography";
// No Retouch tutorial yet: the banner says so until this is set.
const TUTORIAL_VIDEO_URL: string | null = null;
const TUTORIAL_DISMISSED_KEY = "retouch.tutorialDismissed";

const PREVIEW_EDGE = 2048; // keep in sync with the engine's server.PREVIEW_EDGE
import "./App.css";

const DEFAULTS: BackdropParams = {
  strength: 1,
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
type Step = "removals" | "backdrop" | "eyes";

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
  const [faces, setFaces] = useState<number | null>(null);
  const [maskModel, setMaskModel] = useState<ModelInfo | null>(null);
  const [licencePrompt, setLicencePrompt] = useState<EngineError | null>(null);
  const [busy, setBusy] = useState(false);
  const [brushOn, setBrushOn] = useState(false);
  const [brushRadius, setBrushRadius] = useState(BRUSH.default);
  const [removals, setRemovals] = useState(0);
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

  const render = useCallback(async () => {
    if (inFlight.current) {
      dirty.current = true;
      return;
    }
    inFlight.current = true;
    setBusy(true);
    try {
      do {
        dirty.current = false;
        const r = await call<{ preview: string; faces: number | null }>("render", {
          backdrop: paramsRef.current,
          eyes: eyesRef.current,
        });
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
  }, [handleError]);

  const updateParam = (key: keyof BackdropParams, value: number) => {
    const next = { ...paramsRef.current, [key]: value };
    paramsRef.current = next;
    setParams(next);
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
        const r = await call<{ preview: string; removals: number }>(method, {
          ...extra,
          backdrop: paramsRef.current,
          eyes: eyesRef.current,
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
    [handleError],
  );

  const onStroke = (points: [number, number][], radius: number) =>
    removalCall("remove", { points, radius });
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
      } else if (e.key === "[") {
        setBrushRadius((r) => clampBrush(r / 1.2));
      } else if (e.key === "]") {
        setBrushRadius((r) => clampBrush(r * 1.2));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const openImage = async () => {
    // Start in the last image's folder. Otherwise GTK may open on its "Recent"
    // view, which lists files flat and makes them look as if they've moved.
    const lastDir = image?.path.replace(/\/[^/]*$/, "") ?? readLastDir();
    const path = await open({
      multiple: false,
      defaultPath: lastDir ?? undefined,
      filters: [
        { name: "Images and RAW", extensions: bothCases([...IMAGE_EXTENSIONS, ...RAW_EXTENSIONS]) },
        { name: "RAW", extensions: bothCases(RAW_EXTENSIONS) },
        { name: "Images", extensions: bothCases(IMAGE_EXTENSIONS) },
      ],
    });
    if (!path) return;
    setError(null);
    setResult(null);
    setMask(null);
    setView("result");
    setRemovals(0);
    setMaskEdits(0);
    setFaces(null);
    try {
      const r = await call<OpenResult>("open", { path });
      setImage({ ...r, path });
      saveLastDir(path.replace(/\/[^/]*$/, ""));
      render();
    } catch (e) {
      handleError(e);
    }
  };

  // Mask view: the photo with the backdrop tinted red, where the brush corrects
  // the subject mask. Fetched fresh each time, since removals change the photo.
  const maskCall = useCallback(
    async (method: string, extra: Record<string, unknown> = {}) => {
      setMaskPending(true);
      try {
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
    [handleError],
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
      const r = await call<{ preview: string }>("render", {
        backdrop: step === "backdrop" ? null : paramsRef.current,
        eyes: step === "eyes" ? null : eyesRef.current,
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
    const all = { backdrop: paramsRef.current, eyes: eyesRef.current, removals: true };
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
        if (!want || !args) break;
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
    if (!image) return;
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
                else setStatus("Tutorial video coming soon");
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
            {image.path.split("/").pop()} · {image.width}×{image.height} · {image.bit_depth}-bit
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
                />
              )}
              {image && view === "mask" && (
                <PaintOverlay
                  image={imgEl}
                  active={!panning && !zoomTool}
                  radius={brushRadius}
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
              onClick={() => {
                setBrushOn((on) => !on);
                setZoomTool(false);
              }}
              className={brushOn ? "active" : ""}
              title="Paint over anything to remove it (B)"
            >
              <Paintbrush size={15} /> Brush
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
          <p className="panel__model">
            {removals > 0
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
              <p className="panel__model">
                Paint on the photo to correct the mask. Brush size is shared with the Remove brush.
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

      <footer className="statusbar">
        {error ? (
          <span className="statusbar__error" onClick={() => setError(null)}>
            {error}
          </span>
        ) : (
          <span>
            {busy && <span className="spinner" />} {status}
          </span>
        )}
      </footer>

      <footer className="app-footer">
        <span className="oss-note">
          Retouch is free and open-source software. Developed and maintained by Chris Cork
          Photography.
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

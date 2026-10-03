import { useEffect, useRef } from "react";

/** Crop and straighten, as the engine applies it at export: the photo is
 * turned by ``angle`` degrees (positive clockwise) about its centre on a
 * canvas the same size as the photo, then x0..x1, y0..y1 (fractions of that
 * canvas) are kept. */
export interface Crop {
  angle: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}
export const NO_CROP: Crop = { angle: 0, x0: 0, y0: 0, x1: 1, y1: 1 };
export const isNoCrop = (c: Crop) =>
  c.angle === 0 && c.x0 === 0 && c.y0 === 0 && c.x1 === 1 && c.y1 === 1;

type Rect = Pick<Crop, "x0" | "y0" | "x1" | "y1">;
type Handle = "move" | "rotate" | "n" | "s" | "e" | "w" | "nw" | "ne" | "sw" | "se";

const HANDLE_PX = 12; // how close the pointer must be to grab an edge or corner
const MIN_FRACTION = 0.03; // smallest crop, as a fraction of the photo
const DIM = "rgba(0, 0, 0, 0.6)";
const MAX_ANGLE = 45;
// A curved double arrow, for turning the photo from outside the frame.
const ROTATE_CURSOR = `url("data:image/svg+xml,${encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke-linecap="round" stroke-linejoin="round">' +
    '<g stroke="black" stroke-width="4"><path d="M5 15a8 8 0 0 0 14 0"/><path d="M2 12l3 3 3-3"/><path d="M16 12l3 3 3-3"/></g>' +
    '<g stroke="white" stroke-width="2"><path d="M5 15a8 8 0 0 0 14 0"/><path d="M2 12l3 3 3-3"/><path d="M16 12l3 3 3-3"/></g></svg>',
)}") 12 12, crosshair`;

/** Where a point of the turned canvas (fractions) came from in the photo
 * (fractions), for a photo of w x h pixels. */
function source(x: number, y: number, angle: number, w: number, h: number): [number, number] {
  const t = (-angle * Math.PI) / 180;
  const dx = (x - 0.5) * w;
  const dy = (y - 0.5) * h;
  return [
    0.5 + (dx * Math.cos(t) - dy * Math.sin(t)) / w,
    0.5 + (dx * Math.sin(t) + dy * Math.cos(t)) / h,
  ];
}

const EPS = 1e-6;
/** True if every corner of the rectangle lies on the photo once it is turned. */
function inside(r: Rect, angle: number, w: number, h: number): boolean {
  for (const [x, y] of [
    [r.x0, r.y0],
    [r.x1, r.y0],
    [r.x0, r.y1],
    [r.x1, r.y1],
  ]) {
    const [sx, sy] = source(x, y, angle, w, h);
    if (sx < -EPS || sx > 1 + EPS || sy < -EPS || sy > 1 + EPS) return false;
  }
  return true;
}

const lerp = (a: Rect, b: Rect, t: number): Rect => ({
  x0: a.x0 + (b.x0 - a.x0) * t,
  y0: a.y0 + (b.y0 - a.y0) * t,
  x1: a.x1 + (b.x1 - a.x1) * t,
  y1: a.y1 + (b.y1 - a.y1) * t,
});

/** The furthest point from ``from`` toward ``to`` that stays on the photo
 * (``from`` must be on it). */
function towards(from: Rect, to: Rect, angle: number, w: number, h: number): Rect {
  if (inside(to, angle, w, h)) return to;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 20; i++) {
    const mid = (lo + hi) / 2;
    if (inside(lerp(from, to, mid), angle, w, h)) lo = mid;
    else hi = mid;
  }
  return lerp(from, to, lo);
}

/** Shrink ``r`` toward the photo's centre, keeping its shape, until it fits
 * the turned photo. */
export function fitCrop(c: Crop, w: number, h: number): Crop {
  const r: Rect = { x0: c.x0, y0: c.y0, x1: c.x1, y1: c.y1 };
  if (inside(r, c.angle, w, h)) return c;
  const hw = (r.x1 - r.x0) * 0.001;
  const hh = (r.y1 - r.y0) * 0.001;
  const tiny: Rect = { x0: 0.5 - hw, y0: 0.5 - hh, x1: 0.5 + hw, y1: 0.5 + hh };
  return { angle: c.angle, ...towards(tiny, r, c.angle, w, h) };
}

/** The largest crop of ``ratio`` (width / height, in pixels) centred on the
 * current one that fits the turned photo; ``null`` ratio is free. */
export function cropToRatio(c: Crop, ratio: number, w: number, h: number): Crop {
  const cx = (c.x0 + c.x1) / 2;
  const cy = (c.y0 + c.y1) / 2;
  // Start from the whole canvas at that shape, centred on the crop's centre.
  let pw = w;
  let ph = w / ratio;
  if (ph > h) {
    ph = h;
    pw = h * ratio;
  }
  const fw = pw / w / 2;
  const fh = ph / h / 2;
  return fitCrop({ angle: c.angle, x0: cx - fw, y0: cy - fh, x1: cx + fw, y1: cy + fh }, w, h);
}

interface Props {
  /** The displayed photo (its on-screen rectangle and pixels). */
  image: HTMLImageElement | null;
  /** Pixel size of the photo, for its shape. */
  width: number;
  height: number;
  crop: Crop;
  /** Editing: the photo is shown turned with a frame to drag. Otherwise the
   * photo is left as it is, with what the crop leaves out dimmed. */
  editing: boolean;
  /** Width / height to keep while dragging a corner, or null for free. */
  ratio: number | null;
  onChange: (c: Crop) => void;
}

/**
 * Crop & straighten overlay. Like the other overlays it covers the whole
 * viewer and reads the photo's on-screen rectangle at draw and event time, so
 * it follows zoom and pan.
 */
export default function CropOverlay({ image, width, height, crop, editing, ratio, onChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ handle: Handle; start: [number, number]; from: Crop; startTurn: number } | null>(null);
  // While dragging, the frame is redrawn here on every move and only handed
  // to the app (which re-renders all its panels) on release, so it keeps up
  // with the pointer.
  const cropRef = useRef(crop);
  if (!drag.current) cropRef.current = crop;
  const live = (c: Crop) => {
    cropRef.current = c;
    redraw();
  };

  const redraw = () => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    // A new preview (a slider moved) is still loading: keep the last frame
    // rather than clearing to an empty one, which flickered. Its load event
    // redraws.
    if (editing && (!image.complete || image.naturalWidth === 0)) return;
    const c = canvas.getBoundingClientRect();
    const img = image.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(c.width * dpr);
    const bh = Math.round(c.height * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, c.width, c.height);
    const ox = img.left - c.left;
    const oy = img.top - c.top;
    const iw = img.width;
    const ih = img.height;
    const { angle, x0, y0, x1, y1 } = cropRef.current;

    if (!editing) {
      // The photo as it is, with the turned crop's outline; outside it dimmed.
      const pts = [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
      ].map(([x, y]) => source(x, y, angle, width, height));
      ctx.beginPath();
      ctx.rect(ox, oy, iw, ih);
      pts.forEach(([x, y], i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, ox + x * iw, oy + y * ih));
      ctx.closePath();
      ctx.fillStyle = DIM;
      ctx.fill("evenodd");
      return;
    }

    // Editing: cover the viewer and draw the photo turned, under a straight frame.
    ctx.fillStyle = "#121318";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.save();
    ctx.translate(ox + iw / 2, oy + ih / 2);
    ctx.rotate((angle * Math.PI) / 180);
    ctx.drawImage(image, -iw / 2, -ih / 2, iw, ih);
    ctx.restore();
    const L = ox + x0 * iw;
    const T = oy + y0 * ih;
    const R = ox + x1 * iw;
    const B = oy + y1 * ih;
    ctx.beginPath();
    ctx.rect(0, 0, c.width, c.height);
    ctx.rect(L, T, R - L, B - T);
    ctx.fillStyle = DIM;
    ctx.fill("evenodd");
    // Rule of thirds.
    ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const f of [1 / 3, 2 / 3]) {
      ctx.moveTo(L + (R - L) * f, T);
      ctx.lineTo(L + (R - L) * f, B);
      ctx.moveTo(L, T + (B - T) * f);
      ctx.lineTo(R, T + (B - T) * f);
    }
    ctx.stroke();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(L, T, R - L, B - T);
    // Corner and edge handles.
    ctx.fillStyle = "#ffcc33";
    const mx = (L + R) / 2;
    const my = (T + B) / 2;
    const marks: [number, number][] = [
      [L, T],
      [R, T],
      [L, B],
      [R, B],
    ];
    if (ratio === null) marks.push([mx, T], [mx, B], [L, my], [R, my]);
    for (const [x, y] of marks) ctx.fillRect(x - 4, y - 4, 8, 8);
    if (drag.current?.handle === "rotate") {
      const text = `${angle > 0 ? "+" : ""}${angle.toFixed(1)}°`;
      ctx.font = "600 13px system-ui, sans-serif";
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = "rgba(20, 21, 26, 0.85)";
      ctx.fillRect(mx - tw / 2 - 8, my - 13, tw + 16, 26);
      ctx.fillStyle = "#ffcc33";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, mx, my);
    }
    if (drag.current?.handle === "rotate") {
      // A finer grid while straightening, to line things up against.
      ctx.strokeStyle = "rgba(255, 255, 255, 0.18)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let f = 1; f < 9; f++) {
        ctx.moveTo(L + ((R - L) * f) / 9, T);
        ctx.lineTo(L + ((R - L) * f) / 9, B);
        ctx.moveTo(L, T + ((B - T) * f) / 9);
        ctx.lineTo(R, T + ((B - T) * f) / 9);
      }
      ctx.stroke();
    }
  };

  useEffect(() => {
    redraw();
  });
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => redraw());
    ro.observe(canvas);
    // Zoom and pan move the photo without re-rendering this overlay.
    const mo = image ? new MutationObserver(() => redraw()) : null;
    if (image?.parentElement) mo?.observe(image.parentElement, { attributes: true, attributeFilter: ["style"] });
    image?.addEventListener("load", redraw);
    return () => {
      ro.disconnect();
      mo?.disconnect();
      image?.removeEventListener("load", redraw);
    };
  });

  /** Pointer position as fractions of the photo's on-screen rectangle, and
   * the handle under it. */
  const locate = (e: React.PointerEvent): { at: [number, number]; handle: Handle } => {
    const img = image!.getBoundingClientRect();
    const px = e.clientX - img.left;
    const py = e.clientY - img.top;
    const { x0, y0, x1, y1 } = cropRef.current;
    const L = x0 * img.width;
    const R = x1 * img.width;
    const T = y0 * img.height;
    const B = y1 * img.height;
    const near = (a: number, b: number) => Math.abs(a - b) <= HANDLE_PX;
    const withinX = px > L - HANDLE_PX && px < R + HANDLE_PX;
    const withinY = py > T - HANDLE_PX && py < B + HANDLE_PX;
    let v = withinX && near(py, T) ? "n" : withinX && near(py, B) ? "s" : "";
    let hz = withinY && near(px, L) ? "w" : withinY && near(px, R) ? "e" : "";
    if (ratio !== null && !(v && hz)) {
      v = "";
      hz = "";
    }
    const handle = (v + hz || (px > L && px < R && py > T && py < B ? "move" : "rotate")) as Handle;
    return { at: [px / img.width, py / img.height], handle };
  };

  const CURSORS: Record<Handle, string> = {
    move: "move",
    rotate: ROTATE_CURSOR,
    n: "ns-resize",
    s: "ns-resize",
    e: "ew-resize",
    w: "ew-resize",
    nw: "nwse-resize",
    se: "nwse-resize",
    ne: "nesw-resize",
    sw: "nesw-resize",
  };

  /** The pointer's direction from the frame's centre, in degrees on screen. */
  const turnAt = (e: React.PointerEvent, c: Crop) => {
    const img = image!.getBoundingClientRect();
    const cx = img.left + ((c.x0 + c.x1) / 2) * img.width;
    const cy = img.top + ((c.y0 + c.y1) / 2) * img.height;
    return (Math.atan2(e.clientY - cy, e.clientX - cx) * 180) / Math.PI;
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!editing || !image || e.button !== 0) return;
    const { at, handle } = locate(e);
    e.currentTarget.setPointerCapture(e.pointerId);
    const from = cropRef.current;
    drag.current = { handle, start: at, from, startTurn: turnAt(e, from) };
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!editing || !image) return;
    const d = drag.current;
    if (!d) {
      e.currentTarget.style.cursor = CURSORS[locate(e).handle];
      return;
    }
    const { at } = locate(e);
    const { from } = d;
    if (d.handle === "rotate") {
      let turn = turnAt(e, from) - d.startTurn;
      if (turn > 180) turn -= 360;
      if (turn < -180) turn += 360;
      const angle = Math.max(-MAX_ANGLE, Math.min(MAX_ANGLE, from.angle + turn));
      // Fitted from the frame the drag started with, so it grows back as
      // the photo is turned back.
      const turned = fitCrop({ ...from, angle: Math.round(angle * 10) / 10 }, width, height);
      live(ratio === null ? turned : cropToRatio(turned, ratio, width, height));
      return;
    }
    const dx = at[0] - d.start[0];
    const dy = at[1] - d.start[1];
    const r: Rect = { x0: from.x0, y0: from.y0, x1: from.x1, y1: from.y1 };
    const want = { ...r };
    if (d.handle === "move") {
      want.x0 += dx;
      want.x1 += dx;
      want.y0 += dy;
      want.y1 += dy;
    } else {
      if (d.handle.includes("w")) want.x0 = Math.min(r.x0 + dx, r.x1 - MIN_FRACTION);
      if (d.handle.includes("e")) want.x1 = Math.max(r.x1 + dx, r.x0 + MIN_FRACTION);
      if (d.handle.includes("n")) want.y0 = Math.min(r.y0 + dy, r.y1 - MIN_FRACTION);
      if (d.handle.includes("s")) want.y1 = Math.max(r.y1 + dy, r.y0 + MIN_FRACTION);
    }
    if (ratio !== null && d.handle !== "move") {
      // Keep the shape: the height follows the width, from the fixed corner.
      const fw = want.x1 - want.x0;
      const fh = (fw * width) / ratio / height;
      if (d.handle.includes("n")) want.y0 = want.y1 - fh;
      else want.y1 = want.y0 + fh;
      live({ angle: from.angle, ...towards(r, want, from.angle, width, height) });
      return;
    }
    // Each axis on its own, so at the photo's edge the frame slides along it
    // rather than sticking.
    const xOnly = towards(r, { ...r, x0: want.x0, x1: want.x1 }, from.angle, width, height);
    const both = towards(xOnly, { ...xOnly, y0: want.y0, y1: want.y1 }, from.angle, width, height);
    live({ angle: from.angle, ...both });
  };

  const onPointerUp = () => {
    if (!drag.current) return;
    drag.current = null;
    onChange(cropRef.current);
    redraw();
  };

  return (
    <canvas
      ref={canvasRef}
      className="paint-overlay"
      style={{ pointerEvents: editing ? "auto" : "none" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    />
  );
}

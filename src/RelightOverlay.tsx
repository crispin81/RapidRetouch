import { useEffect, useRef } from "react";
import { drawOnPhoto, photoGeometry, toPhoto } from "./photoGeometry";

/** The Relight ellipse: centre as fractions of the photo's width and height,
 * half-axes as fractions of its long edge, turned clockwise by ``angle``
 * degrees (as the engine's tools/relight.py). */
export interface RelightShape {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  angle: number;
}

interface Props {
  image: HTMLImageElement | null;
  /** The photo's own size in pixels (the ellipse is drawn in its units). */
  width: number;
  height: number;
  shape: RelightShape;
  /** Share of the radius the effect fades over: drawn as a dashed inner ring. */
  feather: number;
  editing: boolean;
  onChange: (change: RelightShape & { feather?: number }) => void;
}

const HANDLE = 5; // css px: handle radius
const HIT = 11; // css px: how near the pointer must be to grab a handle
const ROTATE_REACH = 26; // css px: the turn handle's distance past the top
const MIN_RADIUS = 0.01; // of the long edge

type Grab = "move" | "x" | "y" | "rotate" | "feather";
const MIN_FEATHER = 0.02;
const FEATHER_AT = Math.PI / 4; // the feather handle's place on the dashed ring (lower right)
const cursorFor = (g: Grab | null) =>
  g === "move" ? "move" : g === "rotate" ? "grab" : g === "feather" ? "ew-resize" : g ? "pointer" : "default";

/**
 * Canvas covering the viewer that shows the Relight ellipse and, while
 * editing, lets it be moved (drag inside), stretched (the four edge handles)
 * and turned (the handle above it). Positions come from photoGeometry, so it
 * follows zoom, pan, crops and straightening like the brushes.
 */
export default function RelightOverlay({ image, width, height, shape, feather, editing, onChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const grab = useRef<{ kind: Grab; start: [number, number]; shape: RelightShape } | null>(null);
  const inner = Math.max(MIN_FEATHER, Math.min(1, 1 - feather)); // the dashed ring, as a share of the radius
  const hover = useRef<Grab | null>(null);
  const long = Math.max(width, height);

  /** The ellipse in photo pixels: centre, half-axes, and its axes' directions. */
  const geometry = (s: RelightShape) => {
    const a = (s.angle * Math.PI) / 180;
    return {
      c: [s.cx * width, s.cy * height] as [number, number],
      rx: s.rx * long,
      ry: s.ry * long,
      ax: [Math.cos(a), Math.sin(a)] as [number, number], // along rx
      ay: [-Math.sin(a), Math.cos(a)] as [number, number], // along ry (down when unturned)
    };
  };
  /** Screen css px per photo px. */
  const scale = () => (image ? photoGeometry(image).w / width : 1);

  const handles = (s: RelightShape) => {
    const { c, rx, ry, ax, ay } = geometry(s);
    const at = (u: number, v: number): [number, number] => [c[0] + ax[0] * u + ay[0] * v, c[1] + ax[1] * u + ay[1] * v];
    const reach = ROTATE_REACH / scale();
    return {
      x: [at(rx, 0), at(-rx, 0)],
      y: [at(0, ry), at(0, -ry)],
      rotate: at(0, -ry - reach),
      top: at(0, -ry),
      feather: at(rx * inner * Math.cos(FEATHER_AT), ry * inner * Math.sin(FEATHER_AT)),
    };
  };

  const redraw = () => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    const box = canvas.getBoundingClientRect();
    const g = photoGeometry(image);
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(box.width * dpr);
    const bh = Math.round(box.height * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, box.width, box.height);
    drawOnPhoto(ctx, g, box.left, box.top);
    const k = g.w / width; // photo px -> css px
    ctx.scale(k, k); // draw in photo pixels from here
    const px = 1 / k; // one css px, in photo pixels
    const { c, rx, ry } = geometry(shape);
    const a = (shape.angle * Math.PI) / 180;
    const ring = (r: number, dash: number[], colour: string, w: number) => {
      ctx.setLineDash(dash.map((d) => d * px));
      ctx.lineWidth = w * px;
      ctx.strokeStyle = colour;
      ctx.beginPath();
      ctx.ellipse(c[0], c[1], Math.max(rx * r, px), Math.max(ry * r, px), a, 0, Math.PI * 2);
      ctx.stroke();
    };
    for (const [colour, w] of [
      ["rgba(0, 0, 0, 0.55)", 3],
      ["rgba(255, 255, 255, 0.95)", 1.25],
    ] as const) {
      ring(1, [], colour, w);
      ring(inner, [5, 4], colour, w);
    }
    ctx.setLineDash([]);
    // Centre pin
    ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
    ctx.strokeStyle = "rgba(0, 0, 0, 0.6)";
    ctx.lineWidth = 1.5 * px;
    ctx.beginPath();
    ctx.arc(c[0], c[1], 3.5 * px, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    if (!editing) return;
    const h = handles(shape);
    ctx.beginPath();
    ctx.moveTo(...h.top);
    ctx.lineTo(...h.rotate);
    ctx.stroke();
    for (const [p, kind] of [
      ...h.x.map((p) => [p, "x"] as const),
      ...h.y.map((p) => [p, "y"] as const),
      [h.rotate, "rotate"] as const,
      [h.feather, "feather"] as const,
    ]) {
      ctx.fillStyle = hover.current === kind ? "#ffd84a" : "rgba(255, 255, 255, 0.95)";
      ctx.beginPath();
      ctx.arc(p[0], p[1], HANDLE * px, 0, Math.PI * 2);
      ctx.fill();
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
    return () => ro.disconnect();
  });

  /** The pointer in photo pixels. */
  const at = (e: React.PointerEvent): [number, number] | null => {
    if (!image) return null;
    const [fx, fy] = toPhoto(photoGeometry(image), e.clientX, e.clientY);
    return [fx * width, fy * height];
  };

  const hit = (p: [number, number]): Grab | null => {
    const reach = HIT / scale();
    const near = (q: [number, number]) => Math.hypot(p[0] - q[0], p[1] - q[1]) <= reach;
    const h = handles(shape);
    if (near(h.rotate)) return "rotate";
    if (h.x.some(near)) return "x";
    if (h.y.some(near)) return "y";
    if (near(h.feather)) return "feather";
    const { c, rx, ry, ax, ay } = geometry(shape);
    const d = [p[0] - c[0], p[1] - c[1]];
    const u = (d[0] * ax[0] + d[1] * ax[1]) / rx;
    const v = (d[0] * ay[0] + d[1] * ay[1]) / ry;
    const dn = Math.hypot(u, v); // 1 on the outer ring
    // Anywhere along the dashed ring grabs the feather too.
    if (Math.abs(dn - inner) * Math.min(rx, ry) <= (HIT * 0.6) / scale()) return "feather";
    return dn <= 1 ? "move" : null;
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (!editing || e.button !== 0) return;
    const p = at(e);
    const kind = p && hit(p);
    if (!p || !kind) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    grab.current = { kind, start: p, shape };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const p = at(e);
    if (!p) return;
    const g = grab.current;
    if (!g) {
      const over = editing ? hit(p) : null;
      if (over !== hover.current) {
        hover.current = over;
        (e.currentTarget as HTMLCanvasElement).style.cursor = cursorFor(over);
        redraw();
      }
      return;
    }
    const s = g.shape;
    const { c, ax, ay } = geometry(s);
    const d = [p[0] - c[0], p[1] - c[1]];
    if (g.kind === "move") {
      onChange({
        ...s,
        cx: s.cx + (p[0] - g.start[0]) / width,
        cy: s.cy + (p[1] - g.start[1]) / height,
      });
    } else if (g.kind === "x") {
      onChange({ ...s, rx: Math.max(MIN_RADIUS, Math.abs(d[0] * ax[0] + d[1] * ax[1]) / long) });
    } else if (g.kind === "y") {
      onChange({ ...s, ry: Math.max(MIN_RADIUS, Math.abs(d[0] * ay[0] + d[1] * ay[1]) / long) });
    } else if (g.kind === "feather") {
      const u = (d[0] * ax[0] + d[1] * ax[1]) / (s.rx * long);
      const v = (d[0] * ay[0] + d[1] * ay[1]) / (s.ry * long);
      const ring = Math.min(1, Math.max(0, Math.hypot(u, v)));
      onChange({ ...s, feather: Math.max(MIN_FEATHER, 1 - ring) });
    } else {
      // The handle sits above the centre (−ry): pointing straight up is no turn.
      const angle = (Math.atan2(d[1], d[0]) * 180) / Math.PI + 90;
      onChange({ ...s, angle: ((angle + 540) % 360) - 180 });
    }
  };

  const onPointerUp = () => {
    grab.current = null;
  };

  return (
    <canvas
      ref={canvasRef}
      className="paint-overlay"
      style={{ pointerEvents: editing ? "auto" : "none", cursor: editing ? cursorFor(hover.current) : "default" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onDragStart={(e) => e.preventDefault()}
    />
  );
}

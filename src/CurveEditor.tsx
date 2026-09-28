import { useRef } from "react";

export type Point = [number, number];

export const IDENTITY: Point[] = [
  [0, 0],
  [1, 1],
];
const COLOUR = "#e8e8ea";

const SIZE = 240; // drawing units; the SVG scales to the panel width
const HIT = 9; // grab radius, drawing units
const MIN_GAP = 0.01; // points keep at least this far apart in x

/** Monotone cubic (Fritsch-Carlson) through the points, sampled for drawing:
 * the same curve the engine applies, so what's drawn is what's done. */
function sample(pts: Point[], n = 96): Point[] {
  const x = pts.map((p) => p[0]);
  const y = pts.map((p) => p[1]);
  const k = x.length;
  const h = x.slice(1).map((v, i) => v - x[i]);
  const d = h.map((hi, i) => (y[i + 1] - y[i]) / hi);
  const m = new Array(k).fill(0);
  m[0] = d[0];
  m[k - 1] = d[k - 2];
  for (let i = 1; i < k - 1; i++) {
    if (d[i - 1] * d[i] <= 0) m[i] = 0;
    else {
      const w1 = 2 * h[i] + h[i - 1];
      const w2 = h[i] + 2 * h[i - 1];
      m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]);
    }
  }
  const out: Point[] = [];
  for (let s = 0; s <= n; s++) {
    const t0 = s / n;
    let v: number;
    if (t0 <= x[0]) v = y[0];
    else if (t0 >= x[k - 1]) v = y[k - 1];
    else {
      let i = 0;
      while (i < k - 2 && t0 > x[i + 1]) i++;
      const hh = h[i];
      const t = (t0 - x[i]) / hh;
      v =
        (2 * t ** 3 - 3 * t ** 2 + 1) * y[i] +
        (t ** 3 - 2 * t ** 2 + t) * hh * m[i] +
        (-2 * t ** 3 + 3 * t ** 2) * y[i + 1] +
        (t ** 3 - t ** 2) * hh * m[i + 1];
    }
    out.push([t0, Math.min(1, Math.max(0, v))]);
  }
  return out;
}

interface Props {
  points: Point[];
  disabled?: boolean;
  onChange: (points: Point[]) => void;
}

/**
 * Luminosity point curve: reshapes brightness only, so contrast doesn't shift
 * colours. Click to add a point, drag to move it, double-click to remove it.
 */
export default function CurveEditor({ points: pts, disabled, onChange }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const dragging = useRef<number | null>(null);
  const colour = COLOUR;

  const toCurve = (e: React.PointerEvent): Point => {
    const r = svgRef.current!.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width;
    const y = 1 - (e.clientY - r.top) / r.height;
    return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
  };
  const nearest = (p: Point) => {
    let best = -1;
    let bestD = Infinity;
    pts.forEach(([x, y], i) => {
      const d = Math.hypot((x - p[0]) * SIZE, (y - p[1]) * SIZE);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    return bestD <= HIT * 1.5 ? best : -1;
  };
  const set = onChange;

  const onPointerDown = (e: React.PointerEvent) => {
    if (disabled || e.button !== 0) return;
    const p = toCurve(e);
    let i = nearest(p);
    if (i < 0) {
      // A new point, on the curve's own height so the curve doesn't jump.
      const onCurve = sample(pts, 400).reduce((a, b) => (Math.abs(b[0] - p[0]) < Math.abs(a[0] - p[0]) ? b : a));
      const next = [...pts, [p[0], onCurve[1]] as Point].sort((a, b) => a[0] - b[0]);
      i = next.findIndex((q) => q[0] === p[0]);
      set(next);
    }
    dragging.current = i;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const i = dragging.current;
    if (i === null) return;
    const [x, y] = toCurve(e);
    const lo = i > 0 ? pts[i - 1][0] + MIN_GAP : 0;
    const hi = i < pts.length - 1 ? pts[i + 1][0] - MIN_GAP : 1;
    const next = pts.map((q, j) => (j === i ? ([Math.min(hi, Math.max(lo, x)), y] as Point) : q));
    set(next);
  };
  const onPointerUp = () => {
    dragging.current = null;
  };
  const onDoubleClick = (e: React.MouseEvent) => {
    if (disabled) return;
    const r = svgRef.current!.getBoundingClientRect();
    const p: Point = [(e.clientX - r.left) / r.width, 1 - (e.clientY - r.top) / r.height];
    const i = nearest(p);
    // The end points stay: a curve needs two.
    if (i > 0 && i < pts.length - 1) set(pts.filter((_, j) => j !== i));
  };

  const path = sample(pts)
    .map(([x, y], i) => `${i ? "L" : "M"}${(x * SIZE).toFixed(1)},${((1 - y) * SIZE).toFixed(1)}`)
    .join(" ");

  return (
    <div className={`curves${disabled ? " curves--disabled" : ""}`}>
      <svg
        ref={svgRef}
        className="curves__graph"
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={onDoubleClick}
      >
        {[0.25, 0.5, 0.75].map((g) => (
          <g key={g} className="curves__grid">
            <line x1={g * SIZE} y1={0} x2={g * SIZE} y2={SIZE} />
            <line x1={0} y1={g * SIZE} x2={SIZE} y2={g * SIZE} />
          </g>
        ))}
        <line className="curves__diagonal" x1={0} y1={SIZE} x2={SIZE} y2={0} />
        <path d={path} fill="none" stroke={colour} strokeWidth={2} />
        {pts.map(([x, y], i) => (
          <circle
            key={i}
            cx={x * SIZE}
            cy={(1 - y) * SIZE}
            r={5}
            fill="#1c1d24"
            stroke={colour}
            strokeWidth={2}
          />
        ))}
      </svg>
      <p className="curves__hint">
        Brightness only, so colours stay put. For more contrast, raise the upper part and lower the shadows (an
        S-curve). Double-click a point to remove it.
      </p>
    </div>
  );
}

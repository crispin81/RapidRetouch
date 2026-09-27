import { useEffect, useRef } from "react";

interface Props {
  /** The displayed photo strokes are mapped onto. */
  image: HTMLImageElement | null;
  active: boolean;
  /** Brush radius as a fraction of the image's long edge. */
  radius: number;
  /** True while the engine is filling the last stroke; its paint stays visible. */
  pending: boolean;
  onStroke: (points: [number, number][], radius: number) => void;
  /** Stroke paint colour; defaults to the red used for removals. */
  colour?: string;
}

const STROKE_COLOUR = "rgba(255, 64, 64, 0.45)";

/**
 * Canvas covering the whole viewer for painting removal strokes.
 *
 * It covers the viewer rather than just the photo so it always receives the
 * mouse (a canvas sized from a stale photo measurement let clicks fall through
 * to the <img>, which WebKit then dragged). The photo's on-screen rectangle is
 * read at event and draw time, never cached. Points are reported as fractions
 * of the image's width/height, so they land in the same place at any size.
 */
export default function PaintOverlay({
  image,
  active,
  radius,
  pending,
  onStroke,
  colour = STROKE_COLOUR,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const points = useRef<[number, number][]>([]);
  const drawing = useRef(false);
  const cursor = useRef<[number, number] | null>(null);

  const redraw = () => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
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

    // Image-fraction coordinates -> canvas pixels.
    const ox = img.left - c.left;
    const oy = img.top - c.top;
    const px = (x: number) => ox + x * img.width;
    const py = (y: number) => oy + y * img.height;
    const radiusPx = radius * Math.max(img.width, img.height);

    const pts = points.current;
    if (pts.length) {
      ctx.strokeStyle = ctx.fillStyle = colour;
      ctx.lineWidth = radiusPx * 2;
      ctx.lineCap = ctx.lineJoin = "round";
      if (pts.length === 1) {
        ctx.beginPath();
        ctx.arc(px(pts[0][0]), py(pts[0][1]), radiusPx, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.beginPath();
        pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(px(x), py(y)) : ctx.lineTo(px(x), py(y))));
        ctx.stroke();
      }
    }

    if (active && cursor.current && !pending) {
      const [x, y] = cursor.current;
      for (const [colour, extra] of [
        ["rgba(0, 0, 0, 0.6)", 1.5],
        ["rgba(255, 255, 255, 0.9)", 0],
      ] as const) {
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = colour;
        ctx.beginPath();
        ctx.arc(px(x), py(y), radiusPx + extra, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  };

  // Redraw on any prop change and when the viewer is resized.
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

  // Once the engine has filled the stroke, the red paint is no longer needed.
  useEffect(() => {
    if (!pending) {
      points.current = [];
      redraw();
    }
  }, [pending]);

  /** Pointer position as a fraction of the photo (may fall outside 0..1). */
  const toImage = (e: React.PointerEvent): [number, number] | null => {
    if (!image) return null;
    const r = image.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
  };
  const onPhoto = ([x, y]: [number, number]) => x >= 0 && x <= 1 && y >= 0 && y <= 1;
  const clamp = ([x, y]: [number, number]): [number, number] => [
    Math.min(1, Math.max(0, x)),
    Math.min(1, Math.max(0, y)),
  ];

  const onPointerDown = (e: React.PointerEvent) => {
    if (!active || pending || e.button !== 0) return;
    const p = toImage(e);
    if (!p || !onPhoto(p)) return; // strokes start on the photo
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    drawing.current = true;
    points.current = [p];
    redraw();
  };

  const onPointerMove = (e: React.PointerEvent) => {
    cursor.current = toImage(e);
    if (drawing.current && cursor.current) points.current.push(clamp(cursor.current));
    redraw();
  };

  const onPointerUp = () => {
    if (!drawing.current) return;
    drawing.current = false;
    if (points.current.length) onStroke(points.current, radius);
  };

  const onPointerLeave = () => {
    cursor.current = null;
    redraw();
  };

  return (
    <canvas
      ref={canvasRef}
      className="paint-overlay"
      style={{
        pointerEvents: active ? "auto" : "none",
        cursor: active ? "none" : "default",
      }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onPointerLeave={onPointerLeave}
      onDragStart={(e) => e.preventDefault()}
    />
  );
}

import { useEffect, useRef } from "react";
import { drawOnPhoto, photoGeometry, toPhoto } from "./photoGeometry";

type Pt = [number, number];

interface Props {
  /** The displayed photo the selection is mapped onto. */
  image: HTMLImageElement | null;
  active: boolean;
  /** True while the engine is patching; the selection stays visible. */
  pending: boolean;
  /** The lasso outline and the source offset, both as fractions of w/h. */
  onPatch: (outline: Pt[], offset: Pt) => void;
}

const MIN_POINTS = 3;

/**
 * Photoshop-style patch: draw a lasso around the area to fix, then drag the
 * selection onto clean skin or backdrop. While dragging, the selection shows
 * the texture it will take (from under the dragged outline); letting go asks
 * the engine to blend it in. Escape drops the selection.
 *
 * Mount it only while the tool is on: turning the tool off unmounts it, which
 * drops an unused selection, while ``active`` going false (e.g. space-panning)
 * keeps it.
 *
 * Like PaintOverlay it covers the whole viewer and reads the photo's on-screen
 * rectangle at event and draw time, so it follows zoom and pan.
 */
export default function PatchOverlay({ image, active, pending, onPatch }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const outline = useRef<Pt[]>([]);
  // "lasso": drawing the outline; "moving": dragging it to the source.
  const mode = useRef<"idle" | "lasso" | "selected" | "moving">("idle");
  const dragStart = useRef<Pt>([0, 0]);
  const offset = useRef<Pt>([0, 0]);
  const hover = useRef(false);

  const redraw = () => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    const c = canvas.getBoundingClientRect();
    const g = photoGeometry(image);
    const img = { width: g.w, height: g.h };
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
    const pts = outline.current;
    if (pts.length < 2) return;

    // Photo-fraction coordinates -> canvas pixels, turned with the photo.
    drawOnPhoto(ctx, g, c.left, c.top);
    const ox = 0;
    const oy = 0;
    const path = (dx: number, dy: number) => {
      ctx.beginPath();
      pts.forEach(([x, y], i) => {
        const X = ox + (x + dx) * img.width;
        const Y = oy + (y + dy) * img.height;
        if (i === 0) ctx.moveTo(X, Y);
        else ctx.lineTo(X, Y);
      });
      if (mode.current !== "lasso") ctx.closePath();
    };
    const outlineAt = (dx: number, dy: number, dashed: boolean) => {
      for (const [colour, dash] of [
        ["rgba(0, 0, 0, 0.8)", []],
        ["rgba(255, 255, 255, 0.95)", dashed ? [5, 4] : []],
      ] as const) {
        path(dx, dy);
        ctx.setLineDash(dash as number[]);
        ctx.lineWidth = colour.startsWith("rgba(0") ? 2.5 : 1.2;
        ctx.strokeStyle = colour;
        ctx.stroke();
      }
      ctx.setLineDash([]);
    };

    const [dx, dy] = offset.current;
    if (mode.current === "moving" || (pending && (dx || dy))) {
      // Preview: the source's pixels shown inside the selection. The engine
      // matches the tone to the surroundings, so the final result blends better.
      ctx.save();
      path(0, 0);
      ctx.clip();
      ctx.drawImage(image, ox - dx * img.width, oy - dy * img.height, img.width, img.height);
      ctx.restore();
      outlineAt(dx, dy, true);
    }
    outlineAt(0, 0, mode.current !== "lasso");
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

  // The patch is done (or failed): drop the selection.
  useEffect(() => {
    if (!pending && mode.current === "idle") {
      outline.current = [];
      offset.current = [0, 0];
      redraw();
    }
  }, [pending]);

  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && mode.current !== "idle" && !pending) {
        mode.current = "idle";
        outline.current = [];
        offset.current = [0, 0];
        redraw();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const toImage = (e: React.PointerEvent): Pt | null => {
    if (!image) return null;
    return toPhoto(photoGeometry(image), e.clientX, e.clientY);
  };
  const clamp = ([x, y]: Pt): Pt => [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];

  /** Even-odd point-in-polygon test, in on-screen proportions. */
  const inside = ([x, y]: Pt) => {
    const pts = outline.current;
    let hit = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i];
      const [xj, yj] = pts[j];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  };

  /** Keep the dragged outline on the photo: the engine can't take texture
   * from outside it. */
  const clampOffset = ([dx, dy]: Pt): Pt => {
    const xs = outline.current.map((p) => p[0]);
    const ys = outline.current.map((p) => p[1]);
    return [
      Math.min(1 - Math.max(...xs), Math.max(-Math.min(...xs), dx)),
      Math.min(1 - Math.max(...ys), Math.max(-Math.min(...ys), dy)),
    ];
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (!active || pending || e.button !== 0) return;
    const p = toImage(e);
    if (!p) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    if (mode.current === "selected" && inside(p)) {
      mode.current = "moving";
      dragStart.current = p;
      offset.current = [0, 0];
    } else if (p[0] >= 0 && p[0] <= 1 && p[1] >= 0 && p[1] <= 1) {
      mode.current = "lasso";
      outline.current = [p];
      offset.current = [0, 0];
    }
    redraw();
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const p = toImage(e);
    if (!p) return;
    if (mode.current === "lasso") {
      outline.current.push(clamp(p));
    } else if (mode.current === "moving") {
      offset.current = clampOffset([p[0] - dragStart.current[0], p[1] - dragStart.current[1]]);
    }
    const over = mode.current === "moving" || (mode.current === "selected" && inside(p));
    if (over !== hover.current) {
      hover.current = over;
      (e.currentTarget as HTMLCanvasElement).style.cursor = over ? "move" : "crosshair";
    }
    redraw();
  };

  const onPointerUp = () => {
    if (mode.current === "lasso") {
      mode.current = outline.current.length >= MIN_POINTS ? "selected" : "idle";
      if (mode.current === "idle") outline.current = [];
    } else if (mode.current === "moving") {
      const [dx, dy] = offset.current;
      // A click inside the selection without dragging keeps it selected.
      if (Math.hypot(dx, dy) < 0.002) {
        mode.current = "selected";
        offset.current = [0, 0];
      } else {
        mode.current = "idle"; // cleared when the engine is done (pending falls)
        onPatch(outline.current, [dx, dy]);
      }
    }
    redraw();
  };

  return (
    <canvas
      ref={canvasRef}
      className="paint-overlay"
      style={{
        pointerEvents: active ? "auto" : "none",
        cursor: active ? "crosshair" : "default",
      }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onDragStart={(e) => e.preventDefault()}
    />
  );
}

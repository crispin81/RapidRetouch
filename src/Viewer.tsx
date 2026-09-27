import {
  forwardRef,
  ReactNode,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

/** A full-resolution crop laid over the preview when zoomed in. */
export interface Detail {
  image: string; // base64 JPEG
  region: [number, number, number, number]; // x0, y0, x1, y1 in full-res px
}

export interface ViewerHandle {
  fit: () => void;
  actualPixels: () => void;
  zoomBy: (factor: number) => void;
}

interface Props {
  src: string | null; // preview data URL
  size: { width: number; height: number } | null; // full-resolution size
  previewEdge: number; // long edge of the preview in px
  zoomTool: boolean;
  detail: Detail | null;
  imgRef: (el: HTMLImageElement | null) => void;
  /** Visible region (with margin) and device px per full-res px, once zoomed in
   * past the preview's own detail; null when the preview is sharp enough. */
  onDetailNeeded: (region: [number, number, number, number] | null, scale: number) => void;
  onZoomChange: (label: string) => void;
  empty: ReactNode;
  children: (state: { panning: boolean }) => ReactNode;
}

const MARGIN = 16; // around the fitted image
const MAX_ZOOM = 8; // 800%, in device pixels per image pixel
const WHEEL_RATE = 0.0015;
const DETAIL_DELAY = 200; // ms after the view settles before asking for detail
const DETAIL_PAD = 0.25; // extra area around the view, so small pans stay sharp

/**
 * Zoomable, pannable photo view. Mouse wheel zooms around the cursor; the zoom
 * tool clicks in (Alt+click out); holding Space (or the middle button) drags to
 * pan, as in Photoshop. Scale is in CSS px per full-resolution px, so "100%"
 * means one image pixel per device pixel.
 */
const Viewer = forwardRef<ViewerHandle, Props>(function Viewer(
  { src, size, previewEdge, zoomTool, detail, imgRef, onDetailNeeded, onZoomChange, empty, children },
  ref,
) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });
  const fitMode = useRef(true);
  const [space, setSpace] = useState(false);
  const [alt, setAlt] = useState(false);
  const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const dpr = window.devicePixelRatio || 1;

  const fitScale = useCallback(() => {
    if (!size || !box.w) return 1;
    return Math.min((box.w - 2 * MARGIN) / size.width, (box.h - 2 * MARGIN) / size.height);
  }, [size, box]);

  /** Keep the image on screen: centred when smaller than the view, otherwise
   * its edges may come no further in than the middle of the view. */
  const clamp = useCallback(
    (v: { scale: number; x: number; y: number }) => {
      if (!size) return v;
      const sw = size.width * v.scale;
      const sh = size.height * v.scale;
      const x = sw <= box.w ? (box.w - sw) / 2 : Math.min(box.w / 2, Math.max(box.w / 2 - sw, v.x));
      const y = sh <= box.h ? (box.h - sh) / 2 : Math.min(box.h / 2, Math.max(box.h / 2 - sh, v.y));
      return { scale: v.scale, x, y };
    },
    [size, box],
  );

  const fit = useCallback(() => {
    fitMode.current = true;
    setView(clamp({ scale: fitScale(), x: 0, y: 0 }));
  }, [clamp, fitScale]);

  /** Zoom by ``factor`` keeping the point (cx, cy) in the view fixed. */
  const zoomAt = useCallback(
    (factor: number, cx: number, cy: number) => {
      setView((v) => {
        const min = Math.min(fitScale(), 0.02);
        const scale = Math.min(MAX_ZOOM / dpr, Math.max(min, v.scale * factor));
        const k = scale / v.scale;
        fitMode.current = false;
        return clamp({ scale, x: cx - (cx - v.x) * k, y: cy - (cy - v.y) * k });
      });
    },
    [clamp, fitScale, dpr],
  );

  useImperativeHandle(ref, () => ({
    fit,
    actualPixels: () => zoomAt(1 / dpr / view.scale, box.w / 2, box.h / 2),
    zoomBy: (f) => zoomAt(f, box.w / 2, box.h / 2),
  }));

  // Track the viewer's size; stay fitted while in fit mode.
  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setBox({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    if (fitMode.current) fit();
    else setView((v) => clamp(v));
  }, [box, size, fit, clamp]);

  // A new image starts fitted.
  useEffect(() => {
    fitMode.current = true;
    fit();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size?.width, size?.height]);

  // Zoom readout.
  useEffect(() => {
    onZoomChange(fitMode.current ? "Fit" : `${Math.round(view.scale * dpr * 100)}%`);
  }, [view, dpr, onZoomChange]);

  // Wheel zoom. Registered natively: React's wheel listeners are passive, so
  // they can't stop the webview's own scroll/zoom handling.
  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!size) return;
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * WHEEL_RATE), e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [size, zoomAt]);

  // Space = temporary hand tool; Alt flips the zoom tool to zoom out.
  useEffect(() => {
    const typing = (e: KeyboardEvent) => e.target instanceof HTMLInputElement;
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !typing(e)) {
        e.preventDefault(); // don't "click" a focused button
        setSpace(true);
      }
      if (e.key === "Alt") setAlt(true);
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space" && !typing(e)) {
        e.preventDefault(); // buttons "click" on Space release, not press
        setSpace(false);
      }
      if (e.key === "Alt") setAlt(false);
    };
    const blur = () => {
      setSpace(false);
      setAlt(false);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
    };
  }, []);

  const panning = space || dragging;

  const onPointerDown = (e: React.PointerEvent) => {
    if (!size) return;
    const r = boxRef.current!.getBoundingClientRect();
    if (space || e.button === 1) {
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
      setDragging(true);
    } else if (zoomTool && e.button === 0) {
      zoomAt(e.altKey ? 0.5 : 2, e.clientX - r.left, e.clientY - r.top);
    }
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    fitMode.current = false;
    setView(clamp({ scale: view.scale, x: d.vx + e.clientX - d.x, y: d.vy + e.clientY - d.y }));
  };
  const onPointerUp = () => {
    drag.current = null;
    setDragging(false);
  };

  // Ask for full-resolution detail once zoomed past the preview's own detail,
  // after the view has settled.
  useEffect(() => {
    if (!size) return;
    const previewPerFull = Math.min(1, previewEdge / Math.max(size.width, size.height));
    const devicePerFull = view.scale * dpr;
    if (devicePerFull <= previewPerFull * 1.05) {
      onDetailNeeded(null, 0);
      return;
    }
    const t = setTimeout(() => {
      const vx0 = -view.x / view.scale;
      const vy0 = -view.y / view.scale;
      const vw = box.w / view.scale;
      const vh = box.h / view.scale;
      const region: [number, number, number, number] = [
        Math.max(0, vx0 - vw * DETAIL_PAD),
        Math.max(0, vy0 - vh * DETAIL_PAD),
        Math.min(size.width, vx0 + vw * (1 + DETAIL_PAD)),
        Math.min(size.height, vy0 + vh * (1 + DETAIL_PAD)),
      ];
      onDetailNeeded(region, Math.min(1, devicePerFull));
    }, DETAIL_DELAY);
    return () => clearTimeout(t);
  }, [view, box, size, previewEdge, dpr, onDetailNeeded]);

  const cursor = panning
    ? dragging
      ? "grabbing"
      : "grab"
    : zoomTool
      ? alt
        ? "zoom-out"
        : "zoom-in"
      : undefined;

  return (
    <div
      ref={boxRef}
      className="viewer__box"
      style={{ cursor }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onAuxClick={(e) => e.preventDefault()}
    >
      {src && size ? (
        <div
          className="viewer__stage"
          style={{
            width: size.width * view.scale,
            height: size.height * view.scale,
            transform: `translate(${view.x}px, ${view.y}px)`,
          }}
        >
          <img ref={imgRef} src={src} alt="" className="viewer__image" draggable={false} />
          {detail && (
            <img
              src={`data:image/jpeg;base64,${detail.image}`}
              alt=""
              className="viewer__detail"
              draggable={false}
              style={{
                // Past 200%, show crisp pixels as Photoshop does, not a blurry upscale.
                imageRendering: view.scale * dpr >= 2 ? "pixelated" : "auto",
                left: detail.region[0] * view.scale,
                top: detail.region[1] * view.scale,
                width: (detail.region[2] - detail.region[0]) * view.scale,
                height: (detail.region[3] - detail.region[1]) * view.scale,
              }}
            />
          )}
        </div>
      ) : (
        empty
      )}
      {children({ panning })}
    </div>
  );
});

export default Viewer;

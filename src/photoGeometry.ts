/** Where the displayed photo is on screen, allowing for the turn the viewer
 * gives it (a straightened crop: the viewer turns its stage about the centre).
 * The overlays (brushes, patch, crop) map the pointer and draw through this:
 * a turned element's bounding box is bigger than the photo and isn't it. */
export interface PhotoGeometry {
  cx: number; // centre, client px
  cy: number;
  w: number; // the photo's own size on screen (unturned)
  h: number;
  angle: number; // radians, clockwise
}

export function photoGeometry(image: HTMLImageElement): PhotoGeometry {
  const r = image.getBoundingClientRect(); // a turn about the centre keeps the centre
  const degrees = Number(image.dataset.rotation || 0);
  return {
    cx: r.left + r.width / 2,
    cy: r.top + r.height / 2,
    w: image.offsetWidth || r.width,
    h: image.offsetHeight || r.height,
    angle: (degrees * Math.PI) / 180,
  };
}

/** A pointer position (client px) as fractions of the photo's width and height. */
export function toPhoto(g: PhotoGeometry, clientX: number, clientY: number): [number, number] {
  const dx = clientX - g.cx;
  const dy = clientY - g.cy;
  const c = Math.cos(-g.angle);
  const s = Math.sin(-g.angle);
  return [(dx * c - dy * s) / g.w + 0.5, (dx * s + dy * c) / g.h + 0.5];
}

/** Set a 2D context (whose canvas starts at ``origin`` client px) so that
 * drawing at (x * w, y * h) lands on photo fraction (x, y), turned with it. */
export function drawOnPhoto(ctx: CanvasRenderingContext2D, g: PhotoGeometry, originX: number, originY: number) {
  ctx.translate(g.cx - originX, g.cy - originY);
  ctx.rotate(g.angle);
  ctx.translate(-g.w / 2, -g.h / 2);
}

"""Fill masked areas with an inpainting model, one context crop per area.

Models like LaMa work at roughly 512-1024 px, so running them on a whole 24-60 MP
frame would either blow up memory or lose all detail. Instead each masked area
(nearby ones merged) gets a crop with generous surrounding context, filled at
native resolution when it fits, and only the masked pixels are blended back.
"""

from __future__ import annotations

import cv2
import numpy as np

MAX_SIDE = 1024  # larger crops are downscaled for the model
CONTEXT = 1.5  # context on each side, as a multiple of the area's size
MIN_CONTEXT = 96  # px


def stroke_mask(
    shape: tuple[int, int], points: list[list[float]], radius: float, grow: float = 0.15
) -> np.ndarray:
    """Rasterise a brush stroke at any resolution.

    ``points`` are (x, y) as fractions of the image width/height and ``radius`` a
    fraction of the long edge, so the same stroke lands in the same place on the
    preview and the full-resolution export. By default the mask is grown slightly:
    LaMa fills better with a little clean margin around what's being removed.
    """
    h, w = shape
    r = max(1.0, radius * max(h, w))
    grow = max(2, round(r * grow)) if grow > 0 else 0
    mask = np.zeros((h, w), np.uint8)
    pts = [(round(x * w), round(y * h)) for x, y in points]
    thickness = max(1, round(2 * r + 2 * grow))
    for a, b in zip(pts, pts[1:]):
        cv2.line(mask, a, b, 1, thickness)
    for p in pts:
        cv2.circle(mask, p, round(r + grow), 1, -1)
    return mask.astype(bool)


def _boxes(mask: np.ndarray) -> list[list[int]]:
    """Context boxes (x0, y0, x1, y1) around each masked area, overlaps merged."""
    h, w = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for x, y, bw, bh, _ in stats[1:n]:
        mx = max(MIN_CONTEXT, int(bw * CONTEXT))
        my = max(MIN_CONTEXT, int(bh * CONTEXT))
        boxes.append([max(0, x - mx), max(0, y - my), min(w, x + bw + mx), min(h, y + bh + my)])
    merged = True
    while merged:
        merged = False
        out = []
        for b in boxes:
            for o in out:
                if b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3]:
                    o[:] = [min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])]
                    merged = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def fill(rgb: np.ndarray, mask: np.ndarray, model, status=lambda msg: None) -> np.ndarray:
    """rgb float32 HxWx3 0..1, mask bool HxW (True = remove). Returns a new rgb."""
    out = rgb.copy()
    boxes = _boxes(mask)
    for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
        if len(boxes) > 1:
            status(f"Filling {i} of {len(boxes)}")
        crop, cmask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        ch, cw = cmask.shape
        scale = min(1.0, MAX_SIDE / max(ch, cw))
        if scale < 1:
            size = (max(8, round(cw * scale)), max(8, round(ch * scale)))
            small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
            smask = cv2.resize(cmask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
            filled = cv2.resize(model.predict(small, smask), (cw, ch), interpolation=cv2.INTER_CUBIC)
        else:
            filled = model.predict(crop, cmask)
        # Masked pixels are fully replaced; a 2 px feather outside hides the seam.
        soft = cv2.dilate(cmask.astype(np.float32), np.ones((3, 3), np.uint8), iterations=2)
        soft = np.maximum(cv2.GaussianBlur(soft, (0, 0), 1.0), cmask)[..., None]
        out[y0:y1, x0:x1] = soft * filled + (1 - soft) * crop
    return out

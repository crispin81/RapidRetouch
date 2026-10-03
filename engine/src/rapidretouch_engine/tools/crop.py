"""Crop and straighten, applied last, at export.

A crop is {"angle": degrees, "x0", "y0", "x1", "y1"}: the photo is turned by
``angle`` about its centre (positive turns it clockwise, as on screen), on a
canvas the same size as the photo, and the crop rectangle is taken from that
canvas as fractions of its width and height. The app keeps the rectangle
inside the turned photo, so no empty corners are exported.
"""

from __future__ import annotations

import cv2
import numpy as np

IDENTITY = {"angle": 0.0, "x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}


def is_noop(crop: dict | None) -> bool:
    if not crop:
        return True
    c = {**IDENTITY, **crop}
    return c["angle"] == 0 and (c["x0"], c["y0"], c["x1"], c["y1"]) == (0, 0, 1, 1)


def apply(rgb: np.ndarray, crop: dict | None) -> np.ndarray:
    """Turn and crop ``rgb`` (float32 HxWx3, any resolution)."""
    if is_noop(crop):
        return rgb
    c = {**IDENTITY, **crop}
    h, w = rgb.shape[:2]
    out = rgb
    if c["angle"]:
        # Positive angles turn clockwise on screen; OpenCV's turn the other way.
        m = cv2.getRotationMatrix2D((w / 2, h / 2), -float(c["angle"]), 1.0)
        out = cv2.warpAffine(rgb, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    x0, x1 = sorted((float(np.clip(c["x0"], 0, 1)), float(np.clip(c["x1"], 0, 1))))
    y0, y1 = sorted((float(np.clip(c["y0"], 0, 1)), float(np.clip(c["y1"], 0, 1))))
    px0, px1 = round(x0 * w), max(round(x0 * w) + 1, round(x1 * w))
    py0, py1 = round(y0 * h), max(round(y0 * h) + 1, round(y1 * h))
    return np.clip(out[py0:py1, px0:px1], 0, 1).astype(np.float32)

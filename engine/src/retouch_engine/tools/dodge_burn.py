"""Dodge & burn brush strokes: paint light and shade, in linear light.

Each stroke multiplies the light under it by up to 2**(strength * MAX_STOPS)
(dodge) or divides by it (burn), with a soft edge. Working on linear light keeps
skin's colour natural, as a real change of lighting would, instead of greying
highlights or oversaturating shadows. Overlapping strokes build up, like
repeated passes of a low-flow brush.
"""

from __future__ import annotations

import cv2
import numpy as np

from .colour import linear_to_srgb, srgb_to_linear
from .inpaint import stroke_mask

MAX_STOPS = 0.5  # a full-strength pass: half a stop
MODES = ("dodge", "burn")


def stops_map(shape: tuple[int, int], strokes: list[dict]) -> np.ndarray:
    """Exposure change in stops per pixel, for strokes given as {"mode", "points"
    (fractions of w/h), "radius" (fraction of the long edge), "strength" 0..1,
    "softness" 0..1}."""
    h, w = shape
    stops = np.zeros((h, w), np.float32)
    for s in strokes:
        r_px = s["radius"] * max(h, w)
        soft = float(np.clip(s.get("softness", 0.5), 0, 1))
        # Softness shrinks the hard core and widens the fall-off, so the painted
        # circle stays roughly the reach of the effect.
        sigma = max(0.5, (0.1 + 0.5 * soft) * r_px)
        pts = np.asarray(s["points"], np.float64) * [w, h]
        pad = r_px + 3 * sigma
        x0, y0 = np.maximum(np.floor(pts.min(0) - pad).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(pts.max(0) + pad).astype(int), [w, h])
        if x1 <= x0 or y1 <= y0:
            continue
        # Only the stroke's own box is blurred: a whole-frame blur per stroke adds
        # up fast at full resolution.
        local = [[(x - x0) / (x1 - x0), (y - y0) / (y1 - y0)] for x, y in pts]
        radius = s["radius"] * (1 - 0.6 * soft) * max(h, w) / max(x1 - x0, y1 - y0)
        core = stroke_mask((y1 - y0, x1 - x0), local, radius, grow=0).astype(np.float32)
        m = np.clip(cv2.GaussianBlur(core, (0, 0), sigma), 0, 1)
        sign = 1.0 if s["mode"] == "dodge" else -1.0
        stops[y0:y1, x0:x1] += sign * float(s.get("strength", 0.5)) * MAX_STOPS * m
    return stops


def apply_stops(rgb: np.ndarray, stops: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(np.clip(rgb, 0, 1)) * np.exp2(stops)[..., None]
    return linear_to_srgb(np.clip(lin, 0, 1)).astype(np.float32)


def apply(rgb: np.ndarray, strokes: list[dict]) -> np.ndarray:
    if not strokes:
        return rgb.copy()
    return apply_stops(rgb, stops_map(rgb.shape[:2], strokes))

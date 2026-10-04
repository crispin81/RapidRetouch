"""Hand corrections to the areas the tools find (face, neck and body skin,
clothes), as soft brush strokes.

A stroke is stored as the user painted it: points as fractions of the image's
width/height and a radius as a fraction of its long side, so the same edits
land in the same place on the preview, a zoomed-in crop and the export. The
tools find their area as usual, then the edits are replayed on top: "add"
takes the brushed skin in at full strength, "remove" takes it out.
"""

from __future__ import annotations

import cv2
import numpy as np

REGIONS = ("face", "neck", "body", "clothes", "dodge_burn")
SOFTNESS = 0.25  # edge blur as a fraction of the brush radius (as the subject mask brush)


def apply(weight: np.ndarray, edits: list[dict], full_shape: tuple[int, int], box=None) -> np.ndarray:
    """Replay ``edits`` on ``weight`` (0..1), which covers ``box`` (x0, y0, x1,
    y1 in the full image's pixels; the whole image if None) at any size.
    ``full_shape`` is the full image's (h, w). Returns a new map."""
    if not edits:
        return weight
    H, W = full_shape
    x0, y0, x1, y1 = box if box is not None else (0, 0, W, H)
    h, w = weight.shape[:2]
    sx, sy = w / max(x1 - x0, 1), h / max(y1 - y0, 1)
    out = weight.astype(np.float32).copy()
    for e in edits:
        r = max(0.5, e["radius"] * max(H, W) * sx)
        pts = [(round((x * W - x0) * sx), round((y * H - y0) * sy)) for x, y in e["points"]]
        hard = np.zeros((h, w), np.uint8)
        thickness = max(1, round(2 * r))
        for a, b in zip(pts, pts[1:]):
            cv2.line(hard, a, b, 1, thickness)
        for p in pts:
            cv2.circle(hard, p, max(1, round(r)), 1, -1)
        stroke = np.clip(cv2.GaussianBlur(hard.astype(np.float32), (0, 0), max(0.5, SOFTNESS * r)), 0, 1)
        out = np.maximum(out, stroke) if e["mode"] == "add" else out * (1.0 - stroke)
    return out


def overlay(rgb: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Photo with the area tinted blue: what the tools work on."""
    blue = np.array([0.15, 0.45, 1.0], np.float32)
    a = np.clip(area, 0, 1)[..., None] * 0.55
    return rgb * (1.0 - a) + blue * a

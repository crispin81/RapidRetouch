"""Hand corrections to the AI subject mask, as soft brush strokes.

The model's mask is never modified; edits are replayed on top of it, so undo is
exact and the export can redo them at full resolution.
"""

from __future__ import annotations

import cv2
import numpy as np

from .inpaint import stroke_mask

SOFTNESS = 0.25  # edge blur as a fraction of the brush radius


def soft_stroke(shape: tuple[int, int], points: list[list[float]], radius: float) -> np.ndarray:
    """Brush stroke with a soft edge; the 50% contour sits on the brush circle."""
    hard = stroke_mask(shape, points, radius, grow=0).astype(np.float32)
    sigma = max(0.5, SOFTNESS * radius * max(shape))
    return np.clip(cv2.GaussianBlur(hard, (0, 0), sigma), 0, 1)


def apply(alpha: np.ndarray, edits: list[dict]) -> np.ndarray:
    """Replay edits on a mask (float32 HxW, 1 = subject)."""
    out = alpha.copy()
    for e in edits:
        s = soft_stroke(out.shape, e["points"], e["radius"])
        if e["mode"] == "add":
            out = np.maximum(out, s)
        else:
            out = out * (1.0 - s)
    return out


def overlay(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Photo with the backdrop tinted red (like a quick mask); subject left clear."""
    red = np.array([0.9, 0.12, 0.12], np.float32)
    back = (1.0 - alpha)[..., None] * 0.55
    return rgb * (1.0 - back) + red * back

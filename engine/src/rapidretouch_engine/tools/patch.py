"""Patch (like Photoshop's Patch tool): fix an area with texture from another.

The source area's *texture* is copied into the selected area, but its broad
tone and colour are taken from around the selection, so the patch picks up the
lighting where it lands (a healing blend, done by frequency separation in
linear light rather than Poisson solving, so it works on 16-bit float images).
"""

from __future__ import annotations

import cv2
import numpy as np

from .colour import linear_to_srgb, srgb_to_linear
from .filters import masked_blur

FEATHER = 0.08  # edge softness, as a fraction of the selection's size
TONE_SCALE = 0.25  # broad tone is everything coarser than this share of its size


def selection_mask(shape: tuple[int, int], polygon: list[list[float]]) -> np.ndarray:
    """Filled selection from a lasso polygon given as fractions of width/height."""
    h, w = shape
    pts = np.array([[x * w, y * h] for x, y in polygon], np.float32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(pts * 16).astype(np.int32)], 1, lineType=cv2.LINE_AA, shift=4)
    return mask.astype(np.float32)


def apply(rgb: np.ndarray, polygon: list[list[float]], offset: list[float]) -> np.ndarray:
    """Patch the area inside ``polygon`` with texture from the area ``offset``
    away (fractions of width/height). Returns a new rgb."""
    h, w = rgb.shape[:2]
    mask = selection_mask((h, w), polygon)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return rgb.copy()
    dx, dy = round(offset[0] * w), round(offset[1] * h)
    size = max(ys.max() - ys.min(), xs.max() - xs.min(), 4)
    pad = round(size * 0.6) + 4
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    # The source window, clamped to the image (patching from off-image does nothing).
    sy0, sy1, sx0, sx1 = y0 + dy, y1 + dy, x0 + dx, x1 + dx
    if sy0 < 0 or sx0 < 0 or sy1 > h or sx1 > w:
        return rgb.copy()

    target = srgb_to_linear(np.clip(rgb[y0:y1, x0:x1], 0, 1)).astype(np.float32)
    source = srgb_to_linear(np.clip(rgb[sy0:sy1, sx0:sx1], 0, 1)).astype(np.float32)
    sel = mask[y0:y1, x0:x1]
    sigma = max(1.0, TONE_SCALE * size)

    # Broad tone where the patch lands, taken from *around* the selection so the
    # flaw itself doesn't leak back in; the source's own broad tone, removed.
    ring = cv2.dilate(sel, np.ones((3, 3), np.uint8), iterations=max(1, round(0.15 * size))) - sel
    ring = np.clip(ring, 0, 1)
    target_tone = masked_blur(target, np.clip(ring + (1 - cv2.dilate(sel, np.ones((3, 3), np.uint8), iterations=max(1, round(0.3 * size)))), 0, 1), sigma)
    source_tone = cv2.GaussianBlur(source, (0, 0), sigma)
    # Texture as a ratio in linear light, so it scales naturally with the tone.
    texture = source / np.maximum(source_tone, 1e-4)
    healed = np.clip(target_tone * texture, 0, 1)

    soft = cv2.GaussianBlur(sel, (0, 0), max(0.7, FEATHER * size))
    soft = np.maximum(soft, sel * 0.0)[..., None]
    out_lin = soft * healed + (1 - soft) * target
    out = rgb.copy()
    out[y0:y1, x0:x1] = linear_to_srgb(np.clip(out_lin, 0, 1)).astype(np.float32)
    return out

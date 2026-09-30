"""Blurs shared by the tools."""

from __future__ import annotations

import cv2
import numpy as np


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur whose cost doesn't grow with sigma.

    Wide blurs are done on a shrunken copy and scaled back up — the result is a
    smooth field, so nothing visible is lost, and it's orders of magnitude faster
    than a direct kernel hundreds of pixels wide.
    """
    if sigma <= 8:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape[:2]
    factor = sigma / 4
    small = cv2.resize(
        img,
        (max(1, round(w / factor)), max(1, round(h / factor))),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.GaussianBlur(small, (0, 0), 4)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def masked_blur(img: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution: blur only the weighted pixels, filling gaps from them.

    Where no weighted pixel is within reach of ``sigma``, a much wider pass fills
    in so the field is defined everywhere.
    """
    single = img.ndim == 2
    if single:
        img = img[..., None]
    out = np.zeros_like(img)
    remaining = np.ones(img.shape[:2] + (1,), np.float32)  # share not yet filled
    for i, s in enumerate((sigma, sigma * 4, sigma * 16, sigma * 64)):
        num = blur(img * weight[..., None], s)
        if num.ndim == 2:  # OpenCV drops a single channel's axis
            num = num[..., None]
        den = blur(weight, s)[..., None]
        est = num / np.maximum(den, 1e-6)
        conf = np.clip(den / 0.05, 0, 1) if i < 3 else np.ones_like(den)
        out += remaining * conf * est
        remaining *= 1 - conf
        if remaining.max() <= 0:
            break
    return out[..., 0] if single else out


# Large round (elliptical) kernels make OpenCV's dilate slow: its cost grows
# with the kernel's area. These give the same growth in time that doesn't
# depend on the radius.
def grow_mask(mask: np.ndarray, k: int) -> np.ndarray:
    """A 0..1 mask grown as by a round kernel k px wide: exact, via the
    distance to the mask (anti-aliased edges count from their midpoint)."""
    r = (k - 1) / 2
    if r < 3:
        return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    outside = (mask < 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.maximum(mask, (dist <= r).astype(mask.dtype))


def max_filter(img: np.ndarray, k: int) -> np.ndarray:
    """Local maximum over a round neighbourhood k px wide, approximated by a
    square of the same area (separable, so fast). For smooth fields that are
    blurred afterwards, where the corners don't show."""
    if k <= 7:
        return cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    side = 2 * round(0.886 * (k - 1) / 2) + 1
    return cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (side, side)))

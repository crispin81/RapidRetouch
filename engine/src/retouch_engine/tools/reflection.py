"""Removing reflections on glasses: subtract the light the lens added.

A reflection on a lens is an extra layer of light on top of the face (I = T + R).
Where it isn't blown out, the eye and skin are still there underneath, so rather
than inventing detail the reflection is estimated and subtracted, in linear light.

Estimation, inside a painted stroke (paint just the reflection, a little past
its edges: the band outside the stroke is the reference for the unveiled face):
- Tint: coating reflections are coloured (often green or magenta). Every
  plausible tint is tried, keeping the one that best explains the painted
  pixels as face plus reflection.
- Amount, per pixel: each pixel's colour is split into "face" (the
  surroundings' chromaticity) plus "reflection" (the tint) by least squares.
  That keeps the reflection's own structure (a softbox's stripes) and leaves
  the skin's texture alone.
- Over the eye, which isn't skin-coloured, the amount is carried in from the
  surrounding skin and fitted to the reflection's edges with a guided filter.
"""

from __future__ import annotations

import cv2
import numpy as np

from .colour import linear_to_srgb, srgb_to_linear
from .filters import masked_blur

WINDOW = 0.25  # local-minimum window, as a fraction of the brush radius
RING = 0.8  # width of the reference band outside the stroke, fraction of radius
REFERENCE_SIGMA = 0.7  # how far the reference darkness is interpolated inward
GUIDE_EPS = 1e-4
MIN_STRENGTH = 0.004  # veil (linear) below which a pixel is left alone
MAX_REMOVED = 0.85  # never remove more than this share of a pixel's light
TINT_STEP = 0.02  # chromaticity grid for the tint search
TINT_SAMPLES = 3000


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al.'s guided filter (grey guide): smooths ``src`` while following
    the edges of ``guide``."""
    k = (2 * radius + 1, 2 * radius + 1)
    mean = lambda x: cv2.boxFilter(x, -1, k)  # noqa: E731
    mg, ms = mean(guide), mean(src)
    cov = mean(guide * src) - mg * ms
    var = mean(guide * guide) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return mean(a) * guide + mean(b)


def _find_tint(lin: np.ndarray, skin: np.ndarray, where: np.ndarray) -> np.ndarray | None:
    """The reflection's colour: the tint that best explains the painted pixels
    as face (their surroundings' chromaticity) plus some amount of that tint."""
    idx = np.flatnonzero(where)
    if idx.size < 20:
        return None
    rng = np.random.default_rng(0)
    idx = rng.choice(idx, min(idx.size, TINT_SAMPLES), replace=False)
    I = lin.reshape(-1, 3)[idx]
    S = skin.reshape(-1, 3)[idx]
    grid = np.arange(0.05, 0.91, TINT_STEP)
    best, best_err = None, np.inf
    for x in grid:
        for y in grid:
            z = 1 - x - y
            if z < 0.05:
                continue
            t = np.array([x, y, z], np.float32)
            s_ss = (S * S).sum(1)
            s_sr = S @ t
            s_rr = float(t @ t)
            det = s_ss * s_rr - s_sr * s_sr
            if np.median(det) < 1e-4:  # tint too close to skin colour to separate
                continue
            b_s = (S * I).sum(1)
            b_r = I @ t
            v = np.clip((s_ss * b_r - s_sr * b_s) / np.maximum(det, 1e-8), 0, None)
            k = np.clip((s_rr * b_s - s_sr * b_r) / np.maximum(det, 1e-8), 0, None)
            err = float(np.square(I - k[:, None] * S - v[:, None] * t).sum())
            if err < best_err:
                best, best_err = t, err
    return best


def remove(
    rgb: np.ndarray,
    stroke: np.ndarray,
    radius_px: float,
    strength: float,
    eye_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Subtract a lens reflection inside ``stroke`` (bool HxW). ``eye_mask``
    (float HxW, optional) marks the eye openings, which aren't skin-coloured.
    Returns a new rgb (float32 sRGB 0..1)."""
    h, w = stroke.shape
    r = max(2.0, radius_px)
    ring_px = max(3, round(RING * r))
    ys, xs = np.nonzero(stroke)
    if ys.size == 0:
        return rgb.copy()
    pad = ring_px + round(2 * r)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    crop = rgb[y0:y1, x0:x1]
    inside = stroke[y0:y1, x0:x1].astype(np.float32)
    lin = srgb_to_linear(np.clip(crop, 0, 1)).astype(np.float32)

    kr = 2 * ring_px + 1
    ring = cv2.dilate(inside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kr, kr))) - inside
    around = masked_blur(lin, ring, max(1.0, REFERENCE_SIGMA * r))  # the face, unveiled
    eye = eye_mask[y0:y1, x0:x1] if eye_mask is not None else np.zeros_like(inside)

    skin = around / np.maximum(around.sum(axis=2, keepdims=True), 1e-6)
    tint = _find_tint(lin, skin, (inside > 0) & (eye < 0.5))
    if tint is None:
        return rgb.copy()

    # Per pixel, split the colour into "face" (the surroundings' chromaticity)
    # plus "reflection" (the tint): I = t*skin + v*tint, least squares for t, v.
    # Solving per pixel keeps the reflection's own structure (a softbox's
    # stripes) and leaves the skin's texture in t.
    s_ss = (skin * skin).sum(axis=2)
    s_sr = (skin * tint).sum(axis=2)
    s_rr = float((tint * tint).sum())
    b_s = (skin * lin).sum(axis=2)
    b_r = (lin * tint).sum(axis=2)
    det = s_ss * s_rr - s_sr * s_sr
    veil = np.where(det > 1e-8, (s_ss * b_r - s_sr * b_s) / np.maximum(det, 1e-8), 0.0)
    veil = cv2.GaussianBlur(np.clip(veil, 0, None).astype(np.float32), (0, 0), 0.8)
    # A near-white veil can equally be read as "brighter skin", so some of it
    # stays after one stroke; a second stroke works on the improved image and
    # takes more. (Correcting that from the surroundings' brightness was tried
    # and rejected: at lens edges the surroundings are frame, brow or hair, and
    # it over-removed into magenta blotches.)

    # The eye isn't skin-coloured (a white can look like "reflection"), so there
    # the veil is carried in from the surrounding skin instead, and kept to the
    # reflection's own shape with a guided filter.
    if eye.max() > 0:
        carried = masked_blur(veil, inside * (1 - eye), max(1.0, 0.3 * r))
        guide = lin.mean(axis=2)
        carried = np.clip(guided_filter(guide, carried, max(1, round(WINDOW * r)), GUIDE_EPS), 0, None)
        veil = veil * (1 - eye) + carried * eye
    veil *= inside

    # Subtract, fading out at the stroke's edge so there's no seam.
    soft = cv2.GaussianBlur(inside, (0, 0), max(1.0, 0.15 * r))
    added = (strength * veil * soft)[..., None] * tint
    added = np.minimum(added, MAX_REMOVED * lin)
    out = rgb.copy()
    out[y0:y1, x0:x1] = linear_to_srgb(np.clip(lin - added, 0, 1)).astype(np.float32)
    return out

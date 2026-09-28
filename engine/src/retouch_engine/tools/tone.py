"""Tone: exposure (EV) and curves, the photo's final grade.

Exposure is a gain in linear light (what the camera's exposure would have
done). The curve is a luminosity curve: it reshapes lightness only (Lab L),
leaving colour alone, so a contrast S-curve doesn't also saturate skin the way
an RGB curve does. It's a monotone cubic through the points, so it never
overshoots between them (no bumps or tone reversals a plain spline adds).
"""

from __future__ import annotations

import cv2
import numpy as np

from .colour import linear_to_srgb, srgb_to_linear

LUT_SIZE = 4096


def curve_lut(points: list[list[float]] | None, size: int = LUT_SIZE) -> np.ndarray | None:
    """Lookup table for a curve through ``points`` ((x, y) pairs in 0..1), or
    None for the identity. Flat beyond the first and last points."""
    if not points or len(points) < 2:
        return None
    pts = np.array(sorted((float(x), float(y)) for x, y in points), np.float64)
    pts = np.clip(pts, 0, 1)
    x, y = pts[:, 0], pts[:, 1]
    keep = np.concatenate([[True], np.diff(x) > 1e-6])  # drop duplicate x
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return None
    if len(x) == 2 and np.allclose(x, [0, 1]) and np.allclose(y, [0, 1]):
        return None
    # Fritsch-Carlson monotone cubic (PCHIP) slopes.
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.zeros_like(x)
    m[0], m[-1] = delta[0], delta[-1]
    for i in range(1, len(x) - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    t_all = np.linspace(0, 1, size)
    out = np.empty(size)
    idx = np.clip(np.searchsorted(x, t_all) - 1, 0, len(x) - 2)
    x0, x1, y0, y1 = x[idx], x[idx + 1], y[idx], y[idx + 1]
    m0, m1, hh = m[idx], m[idx + 1], x1 - x0
    t = np.clip((t_all - x0) / hh, 0, 1)
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    out = h00 * y0 + h10 * hh * m0 + h01 * y1 + h11 * hh * m1
    out = np.where(t_all < x[0], y[0], np.where(t_all > x[-1], y[-1], out))
    return np.clip(out, 0, 1).astype(np.float32)


def _apply_lut(values: np.ndarray, lut: np.ndarray) -> np.ndarray:
    i = np.clip(values, 0, 1) * (len(lut) - 1)
    lo = np.floor(i).astype(np.int32)
    hi = np.minimum(lo + 1, len(lut) - 1)
    f = (i - lo).astype(np.float32)
    return lut[lo] * (1 - f) + lut[hi] * f


def is_noop(params: dict | None) -> bool:
    if not params:
        return True
    return float(params.get("ev", 0)) == 0 and curve_lut(params.get("curve")) is None


def apply(rgb: np.ndarray, params: dict | None) -> np.ndarray:
    """``params``: {"ev": stops, "curve": [[x, y], ...] (lightness 0..1)}."""
    if is_noop(params):
        return rgb
    out = np.clip(rgb, 0, 1).astype(np.float32)
    ev = float(params.get("ev", 0))
    if ev:
        out = linear_to_srgb(np.clip(srgb_to_linear(out) * 2.0**ev, 0, 1)).astype(np.float32)
    lut = curve_lut(params.get("curve"))
    if lut is not None:
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2Lab)
        lab[..., 0] = _apply_lut(lab[..., 0] / 100.0, lut) * 100.0
        out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
    return out

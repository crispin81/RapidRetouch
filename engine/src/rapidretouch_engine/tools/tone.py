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

from ..parallel import by_rows
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


# Vibrance, like Lightroom's: skin tones (orange-red hues of moderate
# colour) are protected, and dull colours gain more than already vivid ones.
SKIN_HUE = (25.0, 70.0)  # Lab hue angle (degrees) of skin, fully protected inside
SKIN_HUE_FADE = 15.0  # protection fades out over this either side
SKIN_CHROMA = (4.0, 45.0)  # skin is neither grey nor neon
SKIN_PROTECT = 0.7  # share of the change skin tones are spared
SAT_MAX = 0.69  # colour x (1 + this) at full, for dull colours
VIVID = 60.0  # Lab chroma: colours this vivid gain least


# White balance, as a camera does it: a gain on each channel of linear light.
TEMPERATURE_RANGE = 0.25  # log gain on red (and the opposite on blue) at either end
TINT_RANGE = 0.15  # log gain off green at either end (+ magenta, - green)


def white_balance(rgb: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    """Temperature (-1 bluer .. +1 warmer, amber) and tint (-1 greener ..
    +1 more magenta), as channel gains in linear light, with the overall
    brightness kept."""
    t = TEMPERATURE_RANGE * float(np.clip(temperature, -1, 1))
    m = TINT_RANGE * float(np.clip(tint, -1, 1))
    gains = np.exp(np.array([t, -m, -t], np.float32))
    gains /= float(gains @ np.array([0.2126, 0.7152, 0.0722], np.float32))  # brightness kept
    return linear_to_srgb(np.clip(srgb_to_linear(rgb) * gains, 0, 1)).astype(np.float32)


def vibrance(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Vibrance (-1..1): colour across the photo, skin tones protected (see above)."""
    lab = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    lo, hi = SKIN_HUE
    in_hue = np.clip(1 - np.maximum(lo - hue, hue - hi) / SKIN_HUE_FADE, 0, 1)
    in_chroma = np.clip((chroma - SKIN_CHROMA[0]) / 2, 0, 1) * np.clip((SKIN_CHROMA[1] + 10 - chroma) / 10, 0, 1)
    protect = SKIN_PROTECT * in_hue * in_chroma
    amount = float(np.clip(amount, -1, 1))
    if amount > 0:
        gain = 1 + amount * SAT_MAX * (1 - protect) * (1 - 0.6 * np.clip(chroma / VIVID, 0, 1))
    else:
        gain = 1 + amount * (1 - protect)
    lab[..., 1:] *= gain[..., None]
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def is_noop(params: dict | None) -> bool:
    if not params:
        return True
    return (
        float(params.get("ev", 0)) == 0
        and float(params.get("temperature", 0)) == 0
        and float(params.get("tint", 0)) == 0
        and float(params.get("vibrance", 0)) == 0
        and curve_lut(params.get("curve")) is None
    )


def apply(rgb: np.ndarray, params: dict | None) -> np.ndarray:
    """``params``: {"temperature": -1..1, "tint": -1..1, "ev": stops,
    "vibrance": -1..1, "curve": [[x, y], ...] (lightness 0..1)}, applied in
    that order. Every step is per pixel, so it runs on all cores."""
    if is_noop(params):
        return rgb
    return by_rows(lambda part: _apply(part, params), rgb)


def _apply(rgb: np.ndarray, params: dict) -> np.ndarray:
    out = np.clip(rgb, 0, 1).astype(np.float32)
    temperature, tint = float(params.get("temperature", 0)), float(params.get("tint", 0))
    if temperature or tint:
        out = white_balance(out, temperature, tint)
    ev = float(params.get("ev", 0))
    if ev:
        out = linear_to_srgb(np.clip(srgb_to_linear(out) * 2.0**ev, 0, 1)).astype(np.float32)
    vib = float(params.get("vibrance", 0))
    if vib:
        out = vibrance(out, vib)
    lut = curve_lut(params.get("curve"))
    if lut is not None:
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2Lab)
        lab[..., 0] = _apply_lut(lab[..., 0] / 100.0, lut) * 100.0
        out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
    return out

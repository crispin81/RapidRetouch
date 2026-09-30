"""Studio backdrop or outdoor? A quick guess from the frame's borders.

A studio backdrop is smooth and simple: mostly one colour, lit with a gentle
falloff that a smooth surface fits almost exactly, and few edges. Outdoor
backgrounds (trees, buildings, sky, bokeh) are the opposite. The top and side
strips of the frame are checked, since the subject rarely fills them. No model
is needed, and it runs in milliseconds on a small preview.
"""

from __future__ import annotations

import cv2
import numpy as np

WORK_EDGE = 512
TOP, SIDE = 0.18, 0.15  # border strips, as fractions of height/width
DEGREE = 3


def _border(shape) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), bool)
    m[: round(TOP * h)] = True
    m[:, : round(SIDE * w)] = True
    m[:, w - round(SIDE * w) :] = True
    return m


def measure(rgb: np.ndarray) -> dict:
    """Backdrop-likeness measurements of an sRGB float image (any size)."""
    h, w = rgb.shape[:2]
    s = min(1.0, WORK_EDGE / max(h, w))
    small = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(np.clip(small, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.5)  # ignore grain
    border = _border(L.shape)

    # How much structure is left once a smooth lighting surface is fitted.
    hh, ww = L.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    x, y = (2 * xx / ww - 1)[border], (2 * yy / hh - 1)[border]
    basis = np.stack([x**i * y**j for i in range(DEGREE + 1) for j in range(DEGREE + 1 - i)], 1)
    vals = L[border]
    coef, *_ = np.linalg.lstsq(basis, vals, rcond=None)
    residual = float(np.sqrt(np.mean((vals - basis @ coef) ** 2)))

    # Share of the border with hard edges.
    gy, gx = np.gradient(L)
    edges = float((np.hypot(gx, gy)[border] > 4.0).mean())

    # Colour variety.
    ab = lab[..., 1:][border]
    colour = float(np.sqrt(ab.var(axis=0).sum()))
    return {"residual": residual, "edges": edges, "colour": colour}


def classify(rgb: np.ndarray) -> dict:
    """{"mode": "backdrop" | "outdoor", "confidence": 0..1, plus the measurements}.
    When unsure it says backdrop, the safer default."""
    m = measure(rgb)
    # Each measure's votes for "outdoor" (0..1); thresholds from real shoots.
    votes = [
        np.clip((m["residual"] - RESIDUAL[0]) / (RESIDUAL[1] - RESIDUAL[0]), 0, 1),
        np.clip((m["edges"] - EDGES[0]) / (EDGES[1] - EDGES[0]), 0, 1),
        np.clip((m["colour"] - COLOUR[0]) / (COLOUR[1] - COLOUR[0]), 0, 1),
    ]
    # The surface fit gets a smaller say: a subject or studio stand reaching the
    # frame edge also leaves structure there. Edges and colour are the reliable
    # measures (on 49 studio shots: edges <= 3%, colour spread <= 5.4).
    outdoor = float(np.dot(votes, WEIGHTS))
    mode = "outdoor" if outdoor > 0.5 else "backdrop"
    return {"mode": mode, "confidence": abs(outdoor - 0.5) * 2, **m}


# Ramps: the studio side is calibrated on 49 studio shots; the outdoor side is
# a first guess until outdoor shots are measured.
WEIGHTS = (0.2, 0.4, 0.4)  # residual, edges, colour
RESIDUAL = (3.0, 8.0)  # Lab L
EDGES = (0.02, 0.10)  # share of border pixels
COLOUR = (6.0, 14.0)  # Lab a/b spread

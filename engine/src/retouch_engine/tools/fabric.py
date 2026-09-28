"""Clothes crease removal: flattens fine creases in the clothes, found automatically.

Fine creases are ripples of light and shade between the fabric's weave and the
garment's own shape. That band of brightness is flattened (as a change of
light, in linear RGB, so the colour is untouched), while the weave, seams,
stitching and buttons (finer), prints and dark gaps (far more contrasty) and
the garment's shape and thick folds (broader) are kept. Thick folds are the
same size as the body under the fabric, so flattening them would flatten the
figure too.

Scales are set from the face width, since creases scale with the person in the
frame; without a face, from the image size.
"""

from __future__ import annotations

import cv2
import numpy as np

from .colour import linear_to_srgb, srgb_to_linear
from .filters import masked_blur

CREASE_FINE = 0.02  # face widths: finer than this (weave, seams, stitching) is kept
LIGHTING = 0.2  # face widths: broader than this is the garment's shape and light, kept
PRINT = (0.8, 1.4)  # log-brightness off the fabric's shading: crease -> print or gap
MAX_FLATTEN = 1.0
MAX_STOPS = 1.0
SKIN_SPREAD = 4.0  # the skin colour test's tolerance here (the face tools use 3)
SKIN_MARGIN = 0.006  # share of the long edge left clear around skin
SKIN_AREA = 0.004  # share of the long edge skin is voted over
MASK_EDGE = 1024  # px: the clothes mask is found on a copy this size
NO_FACE_SCALE = 0.12  # stand-in face width, as a share of the long edge


def clothes_mask(rgb: np.ndarray, subject: np.ndarray, faces_px: list[np.ndarray]) -> np.ndarray | None:
    """Where the clothes are, 0..1: the subject, minus skin (by the person's
    own skin colour, so neck, hands and arms are left out) and minus the head.
    Worked out on a small copy; None without a face to take skin colour from."""
    from .skin import FACE_OVAL, SkinModel, face_width

    if not faces_px:
        return None
    h, w = rgb.shape[:2]
    s = min(1.0, MASK_EDGE / max(h, w))
    small = cv2.resize(np.clip(rgb, 0, 1), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sub = cv2.resize(subject, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small.astype(np.float32), cv2.COLOR_RGB2Lab)
    yy = np.mgrid[0 : small.shape[0], 0 : small.shape[1]][0].astype(np.float32)
    skin = np.zeros(small.shape[:2], np.float32)
    below = np.zeros(small.shape[:2], np.float32)
    for lm in faces_px:
        lm = lm * s
        fw = face_width(lm)
        skin = np.maximum(skin, SkinModel(lab, lm).colour_probability(lab, SKIN_SPREAD))
        # Clothes start around the chin; above it is head and hair.
        chin = lm[152][1]
        below = np.maximum(below, np.clip((yy - chin) / max(1.0, 0.15 * fw), 0, 1))
        head = np.zeros(small.shape[:2], np.uint8)
        cv2.fillPoly(head, [np.round(lm[FACE_OVAL]).astype(np.int32)], 1)
        below *= 1 - cv2.dilate(head, np.ones((3, 3), np.uint8), iterations=max(1, round(0.05 * fw)))
    # Skin is a solid area (arm, hand, neck), then grown a little: smoothing a
    # bit less clothing is harmless, flattening an arm's shading is not.
    skin = _vote(skin)
    # Body skin is often lit differently from the face, so learn its colour
    # again from the body skin found so far (forearms, hands, neck) and look
    # once more.
    from .skin import _chromaticity

    confident = (skin > 0.8) & (below > 0.5) & (sub > 0.5)
    if confident.sum() > 200:
        rg = _chromaticity(lab)
        mean = rg[confident].mean(0)
        inv = np.linalg.inv(np.cov(rg[confident].T) + np.eye(2) * 1e-4)
        d = rg - mean
        m2 = np.einsum("...i,ij,...j->...", d, inv, d)
        body = np.exp(-0.5 * m2 / SKIN_SPREAD**2 * 4).astype(np.float32)
        skin = np.maximum(skin, _vote(body))
    skin = cv2.dilate(skin, np.ones((3, 3), np.uint8), iterations=max(2, round(SKIN_MARGIN * MASK_EDGE)))
    skin = cv2.GaussianBlur(skin, (0, 0), 1.5)
    clothes = sub * (1 - skin) * below
    clothes = cv2.GaussianBlur(clothes, (0, 0), 1.0)
    return cv2.resize(clothes, (w, h), interpolation=cv2.INTER_LINEAR)


def _vote(prob: np.ndarray) -> np.ndarray:
    """Skin as a solid area: specks that happen to match skin colour (dark
    fabric's noisy colour) are outvoted by their neighbourhood."""
    area = cv2.GaussianBlur(prob.astype(np.float32), (0, 0), SKIN_AREA * MASK_EDGE)
    return np.clip((area - 0.3) / 0.3, 0, 1)


def gain_map(rgb: np.ndarray, amount: np.ndarray, face_px: float | None = None) -> np.ndarray | None:
    """Per-pixel light gain (linear RGB) that flattens creases by ``amount``
    (0..1 per pixel: the slider times the clothes mask), or None if it changes
    nothing.

    Measured once from the photo and then applied with ``apply_gain``: the
    later tools only change backdrop, skin and eyes, so the same gain still
    fits the fabric after them."""
    h, w = rgb.shape[:2]
    ys, xs = np.nonzero(amount > 1e-3)
    if ys.size == 0:
        return None
    fw = face_px or NO_FACE_SCALE * max(h, w)
    pad = int(2 * LIGHTING * fw)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)

    lin = srgb_to_linear(np.clip(rgb[y0:y1, x0:x1], 0, 1)).astype(np.float32)
    a = amount[y0:y1, x0:x1]
    Y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lg = np.log(np.maximum(Y, 1e-4))

    # Creases are shading ripples between CREASE_FINE and LIGHTING across.
    # Anything finer (weave, stitching, seams, button edges) is never touched,
    # because the change itself is smooth at CREASE_FINE.
    fs = max(0.7, CREASE_FINE * fw)
    bs = max(1.0, LIGHTING * fw)
    # Blown highlights carry no shading to measure: they don't inform the
    # bands, and fabric among many of them is left alone (a smooth share, as
    # bright white fabric has scattered clipped pixels all over).
    clipped = (lin.max(axis=2) > 0.98).astype(np.float32)
    blown = np.clip(4 * cv2.GaussianBlur(clipped, (0, 0), fs), 0, 1)
    inform = (a > 0.05) * (1 - clipped)
    crease = masked_blur(lg, inform, fs) - masked_blur(lg, inform, bs)
    # A print, a dark gap or a pocket's shadow is far more contrasty than any
    # shading ripple: those are kept, and left out of the measurement so they
    # can't leave a halo around themselves.
    strong = np.clip((np.abs(crease) - PRINT[0]) / (PRINT[1] - PRINT[0]), 0, 1)
    strong = cv2.dilate(strong, np.ones((3, 3), np.uint8), iterations=max(1, round(fs)))
    inform = inform * (1 - strong)
    crease = masked_blur(lg, inform, fs) - masked_blur(lg, inform, bs)
    keep = np.clip(1.5 * cv2.GaussianBlur(strong, (0, 0), fs), 0, 1)
    weight = a * (1 - keep) * (1 - blown)
    change = -MAX_FLATTEN * weight * crease
    # Flattening must not brighten or darken the garment overall, only even it
    # out: take away the change's own local average (over the same area the
    # creases are measured against).
    avg = masked_blur(change, (weight > 0.05).astype(np.float32), max(1.0, LIGHTING * fw))
    change = change - weight * avg
    # No crease needs more than a stop.
    gain = np.exp(np.clip(change, -MAX_STOPS * np.log(2), MAX_STOPS * np.log(2)))
    # Lifting near-white fabric must not clip it, or one channel before the
    # others (a colour cast): cap the gain where the brightest channel would.
    gain = np.minimum(gain, np.maximum(1.0, 1.0 / np.maximum(lin.max(axis=2), 1e-4)))
    full = np.ones((h, w), np.float32)
    full[y0:y1, x0:x1] = gain
    return full


def apply_gain(rgb: np.ndarray, gain: np.ndarray | None) -> np.ndarray:
    if gain is None:
        return rgb
    ys, xs = np.nonzero(gain != 1)
    if ys.size == 0:
        return rgb
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    out = rgb.copy()
    lin = srgb_to_linear(np.clip(rgb[y0:y1, x0:x1], 0, 1)) * gain[y0:y1, x0:x1, None]
    out[y0:y1, x0:x1] = linear_to_srgb(np.clip(lin, 0, 1)).astype(np.float32)
    return out


def apply(rgb, amount, face_px=None) -> np.ndarray:
    return apply_gain(rgb, gain_map(rgb, amount, face_px))

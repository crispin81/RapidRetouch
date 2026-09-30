"""Flyaway hairs: stray hairs crossing the background near the head, removed.

A stray hair is a thin line, dark or bright (backlit), over background. They're
found as thin lines outside the solid mass of hair (the subject mask with thin
protrusions opened away, plus a margin so the hairline itself is never
touched), within reach of the head, and only where the background around them
is smooth: a studio backdrop or an out-of-focus background. Over a detailed
background (branches, brickwork, a woven hat) a twig looks just like a hair,
so those areas are left alone.

The strays are then repainted by an inpainting model (LaMa) from the
background around them, which is clean: exactly what the model is good at.
The model runs once for every stray it can find; the slider then takes in
fainter ones as it goes up, so moving it doesn't rerun the model.

Sizes are in face widths, so the tool behaves the same on the preview and the
full-resolution export.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .colour import srgb_to_linear
from .skin import FACE_OVAL, face_width

LINE_SCALES = (0.0012, 0.0025)  # hair widths looked for (face widths)
LINE_STRENGTH = (0.015, 0.08)  # scale-normalised line strength (log brightness): faint .. clear
WEAK, SEED = 0.08, 0.5  # a strand is everything above WEAK joined to a part above SEED
SOLID_OPEN = 0.04  # face widths: thinner than this sticking out of the subject mask is strays
HAIRLINE_MARGIN = 0.03  # face widths kept clear outside the solid hair (its own wisps)
REACH = 0.6  # face widths outside the solid subject that strays are looked for
HEAD_REACH = 1.7  # face widths from the middle of the face: the area looked at
SMOOTH_AREA = 0.03  # face widths the background's smoothness is judged over
SMOOTH_MAX = (0.03, 0.06)  # its detail (std of log brightness, strays taken out): smooth .. detailed
MIN_LENGTH = 0.015  # face widths: shorter marks (noise, dust) aren't hairs
MAX_THICKNESS = 0.004  # face widths: a strand thicker than this on average is a clump, kept
HOLE_GROW = 0.004  # face widths the repainted strands are grown by (the hair's soft halo)


def _lines(lg: np.ndarray, fw: float) -> np.ndarray:
    """How much each pixel looks like a thin line, dark or bright (scale-
    normalised, log brightness), at its best scale."""
    best = np.zeros_like(lg)
    for scale in LINE_SCALES:
        sigma = max(0.7, scale * fw)
        g = cv2.GaussianBlur(lg, (0, 0), sigma)
        gy, gx = np.gradient(g)
        gxy, gxx = np.gradient(gx)
        gyy, _ = np.gradient(gy)
        half = (gxx + gyy) / 2
        root = np.sqrt(((gxx - gyy) / 2) ** 2 + gxy**2)
        l1, l2 = half + root, half - root
        # One direction strongly curved (across the hair), the other not.
        valley = np.where(l1 > 0, l1 - np.abs(l2), 0)
        ridge = np.where(l2 < 0, -l2 - np.abs(l1), 0)
        best = np.maximum(best, sigma**2 * np.clip(np.maximum(valley, ridge), 0, None))
    return best


def strays(rgb: np.ndarray, subject: np.ndarray, faces: list[np.ndarray]) -> np.ndarray:
    """How strongly each pixel is a stray hair (0..1, in the image's pixels).
    ``subject``: the subject mask (0..1) at the same size; ``faces``: landmarks
    as fractions of width/height."""
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), np.float32)
    for face in faces:
        lm = face[:, :2] * np.array([w, h], np.float32)
        fw = face_width(lm)
        if fw < 40:
            continue
        c = lm[FACE_OVAL].mean(0)
        r = HEAD_REACH * fw
        x0, y0 = int(max(c[0] - r, 0)), int(max(c[1] - r, 0))
        x1, y1 = int(min(c[0] + r, w)), int(min(c[1] + r, h))
        crop = np.clip(rgb[y0:y1, x0:x1], 0, 1)
        sub = subject[y0:y1, x0:x1].astype(np.float32)
        Y = srgb_to_linear(crop) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        lg = np.log(np.maximum(Y, 1e-3)).astype(np.float32)

        # The solid subject: its mask with thin protrusions (strays the mask
        # took in) opened away, grown by a margin so the hairline is kept.
        k = 2 * max(1, round(SOLID_OPEN * fw / 2)) + 1
        solid = cv2.morphologyEx((sub > 0.5).astype(np.uint8), cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        km = 2 * max(1, round(HAIRLINE_MARGIN * fw)) + 1
        keep = cv2.dilate(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (km, km)))
        dist = cv2.distanceTransform((1 - solid).astype(np.uint8), cv2.DIST_L2, 5) / fw
        zone = (keep == 0) & (dist < REACH)

        line = _lines(lg, fw)
        strength = np.clip((line - LINE_STRENGTH[0]) / (LINE_STRENGTH[1] - LINE_STRENGTH[0]), 0, 1)
        # Background smoothness, judged with the lines taken out (a median
        # removes anything as thin as a hair).
        mk = 2 * max(1, round(3 * LINE_SCALES[-1] * fw)) + 1
        bg = cv2.medianBlur(lg, min(mk, 5)) if mk <= 5 else cv2.medianBlur(
            np.clip((lg + 8) * 16, 0, 255).astype(np.uint8), mk).astype(np.float32) / 16 - 8
        # Only fine detail counts: a smooth gradient (window light falling
        # across a backdrop, a blurred background) is smooth however large its
        # swing in brightness.
        area = max(1.0, SMOOTH_AREA * fw)
        fine_bg = bg - cv2.GaussianBlur(bg, (0, 0), max(1.0, 0.3 * area))
        detail = np.sqrt(cv2.GaussianBlur(fine_bg * fine_bg, (0, 0), area))
        smooth = 1 - np.clip((detail - SMOOTH_MAX[0]) / (SMOOTH_MAX[1] - SMOOTH_MAX[0]), 0, 1)
        stray = strength * smooth * zone

        # Whole strands: faint stretches count where they join a clearly seen
        # part (hysteresis), and short marks (noise, dust) aren't hairs. Each
        # strand is scored by its clearest part, so the slider takes or leaves
        # it whole rather than in dashes.
        n, labels, stats, _ = cv2.connectedComponentsWithStats((stray > WEAK).astype(np.uint8), connectivity=8)
        score = np.zeros(n, np.float32)
        np.maximum.at(score, labels.ravel(), stray.ravel())
        score[0] = 0
        length = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]).astype(np.float32)
        long_ = length >= MIN_LENGTH * fw
        # A single hair is thin all along; a clump of wisps (part of the
        # hairstyle) is thick for its length.
        thin = stats[:, cv2.CC_STAT_AREA] / np.maximum(length, 1) <= MAX_THICKNESS * fw
        score = np.where(long_ & thin & (score >= SEED), score, 0).astype(np.float32)
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], score[labels])
    return out


_PAINTED: dict[bytes, np.ndarray] = {}
PAINTED_KEEP = 4


def remove(rgb: np.ndarray, subject: np.ndarray, faces: list[np.ndarray], amount: float, inpaint) -> np.ndarray:
    """Remove stray hairs by ``amount`` (0..1): higher takes in fainter ones.
    ``inpaint(rgb, hole)`` is the inpainting model. Returns a new rgb."""
    if amount <= 0 or not faces:
        return rgb
    h, w = rgb.shape[:2]
    stray = strays(rgb, subject, faces)
    fw = max(face_width(f[:, :2] * np.array([w, h], np.float32)) for f in faces)
    k = 2 * max(1, round(HOLE_GROW * fw)) + 1
    hole = cv2.dilate((stray > 0.05).astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    if not hole.any():
        return rgb
    key = hashlib.blake2b(rgb.tobytes() + np.packbits(hole).tobytes(), digest_size=16).digest()
    painted = _PAINTED.pop(key, None)
    if painted is None:
        painted = np.clip(inpaint(rgb, hole), 0, 1).astype(np.float32)
        while len(_PAINTED) >= PAINTED_KEEP:
            _PAINTED.pop(next(iter(_PAINTED)))
    _PAINTED[key] = painted
    # The slider takes in fainter strands as it goes up (each strand's score
    # is its clearest part); each is replaced whole, softly at the edges.
    take = (stray >= SEED + (1 - amount) * (1 - SEED)).astype(np.float32)
    take = cv2.dilate(take, np.ones((k, k), np.uint8))
    take = np.minimum(cv2.GaussianBlur(take, (0, 0), max(0.5, 0.5 * HOLE_GROW * fw)), hole.astype(np.float32))
    return rgb + take[..., None] * (painted - rgb)

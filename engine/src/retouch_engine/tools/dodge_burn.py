"""Dodge & burn: sculpt the face with light, from one slider (like Evoto's
"Sculpt"). The high points (forehead centre, nose bridge, cheekbone tops,
chin) are brightened and the contours (under the cheekbones, the sides of the
nose, the face's edge along the temples, hairline and jaw) are deepened.

Placed from the face landmarks, limited to skin (not eyes, brows, lips or
hair), and applied as a soft change of light in linear RGB, so the skin's
texture and colour are kept, as with a hand-painted dodge & burn layer.
"""

from __future__ import annotations

import cv2
import numpy as np

from .colour import linear_to_srgb, srgb_to_linear
from .filters import blur
from .skin import FACE_OVAL, face_skin, face_width

DODGE_STOPS = 0.5  # at full strength, at the centre of a highlight
BURN_STOPS = 0.6
SOFTNESS = 0.05  # face widths: how gradually each area fades out
EDGE_BAND = 0.08  # face widths inside the outline that are contoured
HIGHLIGHT_FADE = (0.35, 0.75)  # linear brightness: dodging fades out toward white
WORK_FW = 400  # px: the light map is smooth, so it's worked out at this size

# Landmark paths (MediaPipe face mesh), each with a width in face widths.
DODGE_LINES = [
    ([168, 6, 197, 195, 5], 0.035),  # nose bridge
]
BURN_LINES = [
    ([93, 147, 187], 0.08),  # under the cheekbones, ear to mouth
    ([323, 376, 411], 0.08),
    ([122, 129], 0.02),  # sides of the nose
    ([351, 358], 0.02),
]


def _blob(shape, centre, radius) -> np.ndarray:
    m = np.zeros(shape, np.float32)
    cv2.circle(m, tuple(int(round(v)) for v in centre), max(1, int(round(radius))), 1.0, -1)
    return m


def _line(shape, pts, width) -> np.ndarray:
    m = np.zeros(shape, np.float32)
    pts = [tuple(int(round(v)) for v in p) for p in pts]
    t = max(1, int(round(width)))
    for a, b in zip(pts, pts[1:]):
        cv2.line(m, a, b, 1.0, t)
    return m


def zones(shape: tuple[int, int], lm: np.ndarray, fw: float) -> tuple[np.ndarray, np.ndarray]:
    """Soft (dodge, burn) weights 0..1 for one face at landmark scale."""
    dodge = np.zeros(shape, np.float32)
    burn = np.zeros(shape, np.float32)
    # Forehead centre: above the brows, toward the top of the face.
    dodge = np.maximum(dodge, _blob(shape, lm[151] * 0.6 + lm[9] * 0.4, 0.1 * fw))
    for side in ((117, 118, 50), (346, 347, 280)):  # cheekbone tops
        dodge = np.maximum(dodge, _blob(shape, lm[list(side)].mean(0), 0.055 * fw))
    chin = lm[175] + (lm[17] - lm[175]) * 0.35  # just above the chin's point
    dodge = np.maximum(dodge, _blob(shape, chin, 0.06 * fw))
    for path, width in DODGE_LINES:
        dodge = np.maximum(dodge, _line(shape, lm[path], width * fw))
    for path, width in BURN_LINES:
        burn = np.maximum(burn, _line(shape, lm[path], width * fw))
    # The face's edge: a band just inside the outline.
    oval = np.zeros(shape, np.uint8)
    cv2.fillPoly(oval, [np.round(lm[FACE_OVAL]).astype(np.int32)], 1)
    inside = cv2.distanceTransform(oval, cv2.DIST_L2, 5)
    edge = np.clip(1 - inside / (EDGE_BAND * fw), 0, 1) * oval
    burn = np.maximum(burn, edge)
    sigma = max(0.7, SOFTNESS * fw)
    return blur(dodge, sigma), blur(burn, sigma)


def apply(rgb: np.ndarray, faces: list[np.ndarray], amount: float) -> np.ndarray:
    """Sculpt every face by ``amount`` (0..1). Returns a new rgb."""
    out = rgb.copy()
    if amount <= 0:
        return out
    h, w = rgb.shape[:2]
    for face in faces:
        lm = face[:, :2] * np.array([w, h], np.float32)
        fw = face_width(lm)
        if fw < 40:
            continue
        pts = lm[FACE_OVAL]
        x0, y0 = np.maximum(np.floor(pts.min(0) - 0.2 * fw), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(pts.max(0) + 0.2 * fw), [w, h]).astype(int)
        crop = np.clip(out[y0:y1, x0:x1], 0, 1)
        # The light map is smooth, so it's worked out on a small copy.
        s = min(1.0, WORK_FW / fw)
        small = crop if s == 1 else cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        slm = (lm - [x0, y0]) * [small.shape[1] / crop.shape[1], small.shape[0] / crop.shape[0]]
        sfw = fw * small.shape[1] / crop.shape[1]
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)
        skin_w, hair, _ = face_skin(lab, slm)
        W = blur(skin_w * (1 - hair), max(0.7, 0.02 * sfw))
        dodge, burn = zones(small.shape[:2], slm, sfw)
        stops = amount * W * (DODGE_STOPS * dodge - BURN_STOPS * burn)
        if s != 1:
            stops = cv2.resize(stops, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
        lin = srgb_to_linear(crop)
        # Dodging fabric-white highlights only blows them out.
        bright = lin.max(axis=2)
        fade = 1 - np.clip((bright - HIGHLIGHT_FADE[0]) / (HIGHLIGHT_FADE[1] - HIGHLIGHT_FADE[0]), 0, 1)
        stops = np.where(stops > 0, stops * fade, stops)
        out[y0:y1, x0:x1] = linear_to_srgb(np.clip(lin * np.exp2(stops)[..., None], 0, 1))
    return out

"""Mouth: lip colour, lip smoothing and teeth whitening, from face landmarks.

Lips are the band between MediaPipe's outer and inner lip contours. Teeth are
found inside the inner contour by colour: bright and not red, which leaves out
the gums, tongue and the dark inside of the mouth, so a closed mouth or a
slightly misplaced contour just finds no teeth instead of whitening lips.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .eyes import _inner_feather, _poly_mask
from .filters import grow_mask, masked_blur
from .skin import LIPS

LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
MOUTH_CORNERS = (61, 291)

LIP_FEATHER = 0.018  # mouth widths: lips fade in from their outline
# The landmarks' outer lip outline sits a little inside the visible lip edge:
# grown by this (mouth widths) so the edges get the full change too.
LIP_GROW = 0.012
SATURATION_RANGE = 0.5  # full slider right scales lip chroma by 1 + this
MUTE_RANGE = 0.7  # full slider left moves lip colour this far toward the skin's
HUE_RANGE = 25.0  # degrees the lip colour turns at either end of Lip hue
LIP_DETAIL = 0.03  # mouth widths: lip lines and flakes are finer than this
LIP_SHEEN = (6.0, 14.0)  # Lab L above the lip's surround: sheen, kept
LINES_KEEP = 0.1  # share of dark lip lines left at full smoothing

TEETH_GROW = 0.03  # mouth widths: look this far past the inner contour for teeth
TEETH_DARK = (30.0, 12.0)  # Lab L below the brightest teeth: not teeth -> teeth
TEETH_RED = (22.0, 12.0)  # Lab a: gums/tongue -> teeth
TEETH_YELLOW_KEEP = 2.0  # Lab b left at full whitening: a hint of warmth
TEETH_LIFT = 6.0  # Lab L added at full whitening, less near white


@dataclass
class Params:
    lip_saturation: float = 0.0  # -1 muted .. 0 unchanged .. +1 rich
    lip_hue: float = 0.0  # -1 cooler, pinker .. 0 unchanged .. +1 warmer, more coral
    lip_smooth: float = 0.0
    teeth_whiten: float = 0.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "Params":
        d = d or {}
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_noop(self) -> bool:
        return self.lip_saturation == 0 and self.lip_hue == 0 and self.lip_smooth <= 0 and self.teeth_whiten <= 0


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def teeth_mask(lab: np.ndarray, inner: np.ndarray, mw: float) -> np.ndarray:
    """Soft teeth mask within (a little past) the inner lip contour."""
    grow = max(1, round(TEETH_GROW * mw))
    area = cv2.dilate(inner, np.ones((3, 3), np.uint8), iterations=grow)
    if area.sum() < 20:
        return np.zeros_like(inner)
    L, a = lab[..., 0], lab[..., 1]
    bright = np.percentile(L[area > 0.5], 95)
    m = _smoothstep(L, bright - TEETH_DARK[0], bright - TEETH_DARK[1])
    m *= _smoothstep(a, TEETH_RED[0], TEETH_RED[1])
    # Too few bright, pale pixels: a closed mouth or lips only.
    if (m * area).sum() < 0.02 * area.sum():
        return np.zeros_like(inner)
    return cv2.GaussianBlur(m * area, (0, 0), max(0.5, 0.004 * mw))


def _mouth(lab: np.ndarray, lm: np.ndarray, p: Params) -> None:
    """Retouch one mouth in ``lab`` (a crop around it), in place."""
    shape = lab.shape[:2]
    mw = float(np.linalg.norm(lm[MOUTH_CORNERS[0]] - lm[MOUTH_CORNERS[1]]))
    outer = _poly_mask(shape, lm[LIPS])
    inner = _poly_mask(shape, lm[LIPS_INNER])
    reach = grow_mask(outer, 2 * max(1, round(LIP_GROW * mw)) + 1)
    lips = _inner_feather(np.clip(reach - inner, 0, 1), LIP_FEATHER * mw)

    if p.lip_smooth > 0:
        L = lab[..., 0]
        base = masked_blur(L, lips, max(0.7, LIP_DETAIL * mw))
        detail = L - base
        # Dark lines and flakes flattened; the sheen (bright, well above its
        # surround) left alone, so lips don't turn matte.
        sheen = _smoothstep(detail, *LIP_SHEEN)
        keep = np.where(detail < 0, LINES_KEEP, 1 - 0.7 * (1 - sheen))
        amount = p.lip_smooth * lips
        lab[..., 0] = L - amount * (1 - keep) * detail
        # Flakes are also pale: pull their colour toward the lip's own.
        ab = lab[..., 1:]
        ab_base = masked_blur(ab, lips, max(0.7, LIP_DETAIL * mw))
        lab[..., 1:] = ab + (0.6 * amount)[..., None] * (ab_base - ab)

    if p.lip_saturation > 0:
        scale = 1 + SATURATION_RANGE * min(p.lip_saturation, 1.0)
        lab[..., 1:] *= (1 + lips * (scale - 1))[..., None]
    elif p.lip_saturation < 0:
        # Muting toward grey leaves lips lifeless and bluish; muting toward the
        # skin around them looks like a natural, paler lip.
        ring = np.clip(cv2.dilate(outer, np.ones((3, 3), np.uint8), iterations=max(1, round(0.08 * mw))) - outer, 0, 1)
        skin_ab = masked_blur(lab[..., 1:], ring, max(0.7, 0.15 * mw))
        t = (MUTE_RANGE * min(-p.lip_saturation, 1.0) * lips)[..., None]
        lab[..., 1:] += t * (skin_ab - lab[..., 1:])

    if p.lip_hue != 0:
        # Turn the lips' colour around the colour wheel, keeping its strength
        # and the lips' brightness: toward +b (yellow) is warmer, coral;
        # toward -b (blue) is cooler, pink to berry.
        angle = np.deg2rad(HUE_RANGE * float(np.clip(p.lip_hue, -1, 1))) * lips
        a, b = lab[..., 1].copy(), lab[..., 2].copy()
        c, s = np.cos(angle), np.sin(angle)
        lab[..., 1] = a * c - b * s
        lab[..., 2] = a * s + b * c

    if p.teeth_whiten > 0:
        t = teeth_mask(lab, inner, mw) * p.teeth_whiten
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        lab[..., 2] = b - t * np.maximum(b - TEETH_YELLOW_KEEP, 0) * 0.8
        lab[..., 1] = a - t * np.maximum(a, 0) * 0.5
        lab[..., 0] = L + t * TEETH_LIFT * np.clip((100 - L) / 30, 0, 1)


def apply(rgb: np.ndarray, faces: list[np.ndarray], params: dict | Params | None) -> np.ndarray:
    """Retouch every face's mouth. ``faces`` are landmarks as fractions of w/h."""
    p = params if isinstance(params, Params) else Params.from_dict(params)
    out = rgb.copy()
    if p.is_noop():
        return out
    h, w = rgb.shape[:2]
    for face in faces:
        lm = face[:, :2] * [w, h]
        mouth = lm[LIPS]
        mw = float(np.linalg.norm(lm[MOUTH_CORNERS[0]] - lm[MOUTH_CORNERS[1]]))
        if mw < 12:
            continue
        pad = 0.5 * mw
        x0, y0 = np.maximum(np.floor(mouth.min(0) - pad).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(mouth.max(0) + pad).astype(int), [w, h])
        crop = np.clip(out[y0:y1, x0:x1], 0, 1).astype(np.float32)
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab)
        _mouth(lab, lm - [x0, y0], p)
        out[y0:y1, x0:x1] = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
    return out

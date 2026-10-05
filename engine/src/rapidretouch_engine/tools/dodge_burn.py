"""Dodge & burn: shape the face with light, from three sliders.

- Contour: classic retouching shape. The high points (forehead centre, nose
  bridge, cheekbone tops, chin) are brightened and the contours (under the
  cheekbones, the sides of the nose, the face's edge along the temples and
  jaw) deepened. Each zone is a soft field with no edge of its own, and it
  follows the photo's light: dodging fades on the shadow side and burning
  stays out of the highlights, so nothing is painted against the light.
- Highlights: the light already on the face made brighter.
- Shadows: the shade already on the face made deeper.

Highlights and Shadows work on the face's own light and shade (its broad
shape, pores and fine detail taken out), through a curve that rolls off, so a
strong highlight or a deep shadow moves least and nothing clips.

Everything is limited to skin up to the hairline, kept off the eyes and lips,
and faded to nothing at every edge, so no change shows as a line. It's applied
as a change of light in linear RGB, so the skin's texture and colour are kept,
as with a hand-painted dodge & burn layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import region_edit
from .colour import linear_to_srgb, srgb_to_linear
from .eyes import EYES, _inner_feather, _poly_mask
from .filters import blur, grow_mask, masked_blur
from .skin import FACE_OVAL, HAIR_SURE, LIPS, _map_into, _smoothstep, _strands, face_skin, face_width, profile_outline

# The shape of the light, in face widths: finer than DETAIL is texture and
# pores; broader than FORM is the fall-off of the light across the whole face,
# which is left as shot.
DETAIL = 0.05
FORM = 0.15
# Highlights / Shadows at full: the light's own shape strengthened by GAIN,
# rolling off beyond KNEE stops, never more than MAX stops.
HIGHLIGHT_GAIN, HIGHLIGHT_KNEE, HIGHLIGHT_MAX = 1.5, 0.4, 0.6
SHADOW_GAIN, SHADOW_KNEE, SHADOW_MAX = 1.8, 0.4, 0.7
# Contour at full, in stops, at the centre of each zone.
CONTOUR_DODGE = 0.45
CONTOUR_BURN = 0.55
CONTOUR_SOFT = 0.06  # face widths: how gradually each zone fades out
EDGE_BAND = 0.1  # face widths inside the outline that are deepened
# Contour follows the light (stops of the face's own light and shade):
# dodging fades out in shade, burning fades out in highlights.
DODGE_IN_SHADE = (-0.35, 0.05)
BURN_IN_LIGHT = (0.15, 0.45)
EDGE_FADE = 0.08  # face widths over which everything fades in from every edge
KEEP_OFF_MARGIN = 0.02  # face widths around the eyes and lips kept clear...
KEEP_OFF_FADE = 0.03  # ...and faded in over this
HIGHLIGHT_FADE = (0.35, 0.75)  # linear brightness: dodging fades out toward white
WORK_FW = 400  # px: the light map is smooth, so it's worked out at this size
# Hair over the face (a fringe, loose strands) is left as shot. The
# person-parts model's hair map covers the hair itself, softened over this
# (face widths) so it leaves no line; single strands are traced at full
# detail, but only this close (face widths) to that hair, so forehead lines,
# which look much the same, are still dodged.
HAIR_SOFT = 0.01
BEARD_KEEP = 0.85  # share of the change kept off facial hair
BEARD_SOFT = 0.03  # face widths: how gradually that fades in from the cheeks
STRANDS_NEAR_HAIR = 0.08
STRANDS_PAD = 0.01  # face widths of context around where strands are traced

# Contour zones (MediaPipe face mesh landmarks), widths in face widths.
DODGE_LINES = [([168, 6, 197, 195, 5], 0.035)]  # nose bridge
BURN_LINES = [
    ([93, 147, 187], 0.08),  # under the cheekbones, ear to mouth
    ([323, 376, 411], 0.08),
    ([122, 129], 0.025),  # sides of the nose
    ([351, 358], 0.025),
]


@dataclass
class Params:
    contour: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "Params":
        d = d or {}
        if "amount" in d and not ({"contour", "highlights", "shadows"} & set(d)):
            # Saved before the split: the old single Sculpt slider.
            a = float(d["amount"])
            return cls(highlights=a, shadows=a)
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_noop(self) -> bool:
        return self.contour <= 0 and self.highlights <= 0 and self.shadows <= 0


def light_shape(Y: np.ndarray, W: np.ndarray, fw: float) -> np.ndarray:
    """The face's light and shade in stops, relative to the overall light
    falling on it: + where it's lit, - where it's in shade. Measured on skin
    only (``W``), so hair and background don't bleed in."""
    lg = np.log2(np.maximum(Y, 1e-4))
    form = masked_blur(lg, W, max(1.0, DETAIL * fw))
    overall = masked_blur(lg, W, max(1.0, FORM * fw))
    return form - overall


def _roll(x: np.ndarray, gain: float, knee: float, most: float) -> np.ndarray:
    return np.clip(gain * x / (1 + np.abs(x) / knee), -most, most)


def tone_curve(shape: np.ndarray, p: Params) -> np.ndarray:
    """Stops to add for Highlights and Shadows: the light's own shape
    strengthened, lit parts by ``highlights`` and shaded parts by
    ``shadows``, rolling off so the strongest move least."""
    lit = np.clip(shape, 0, None)
    shade = np.clip(shape, None, 0)
    return p.highlights * _roll(lit, HIGHLIGHT_GAIN, HIGHLIGHT_KNEE, HIGHLIGHT_MAX) + p.shadows * _roll(
        shade, SHADOW_GAIN, SHADOW_KNEE, SHADOW_MAX
    )


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


def contour_zones(shape: tuple[int, int], lm: np.ndarray, fw: float) -> tuple[np.ndarray, np.ndarray]:
    """Soft (dodge, burn) fields 0..1 for one face, with no edges of their
    own: each zone is drawn, then blurred well past its size."""
    dodge = np.zeros(shape, np.float32)
    burn = np.zeros(shape, np.float32)
    dodge = np.maximum(dodge, _blob(shape, lm[151] * 0.6 + lm[9] * 0.4, 0.09 * fw))  # forehead centre
    for side in ((117, 118, 50), (346, 347, 280)):  # cheekbone tops
        dodge = np.maximum(dodge, _blob(shape, lm[list(side)].mean(0), 0.05 * fw))
    dodge = np.maximum(dodge, _blob(shape, lm[175] + (lm[17] - lm[175]) * 0.35, 0.05 * fw))  # chin
    for path, width in DODGE_LINES:
        dodge = np.maximum(dodge, _line(shape, lm[path], width * fw))
    for path, width in BURN_LINES:
        burn = np.maximum(burn, _line(shape, lm[path], width * fw))
    # The face's edge along the temples and jaw: a band inside the outline.
    oval = np.zeros(shape, np.uint8)
    cv2.fillPoly(oval, [np.round(lm[FACE_OVAL]).astype(np.int32)], 1)
    inside = cv2.distanceTransform(oval, cv2.DIST_L2, 5)
    burn = np.maximum(burn, np.clip(1 - inside / (EDGE_BAND * fw), 0, 1) * oval)
    sigma = max(0.7, CONTOUR_SOFT * fw)
    soft = lambda m: np.clip(blur(m, sigma) * 1.6, 0, 1)  # noqa: E731
    return soft(dodge), soft(burn)


def apply(
    rgb: np.ndarray,
    faces: list[np.ndarray],
    p: Params,
    head_hair: np.ndarray | None = None,
    edits: list[dict] | None = None,
) -> np.ndarray:
    """Dodge & burn every face. Returns a new rgb. ``head_hair``: the
    person-parts model's hair map for the whole photo (any size), kept clear.
    ``edits``: the user's corrections to where it works (Refine area)."""
    if p.is_noop():
        return rgb.copy()
    return _dodge_burn(rgb, faces, p, head_hair, edits)


def area(
    rgb: np.ndarray, faces: list[np.ndarray], head_hair: np.ndarray | None = None, edits: list[dict] | None = None
) -> np.ndarray:
    """Where dodge & burn works (0..1, the image's size), with the user's
    corrections: shown blue by the Refine area brush."""
    found = np.zeros(rgb.shape[:2], np.float32)
    _dodge_burn(rgb, faces, Params(), head_hair, edits, found)
    return found


def _dodge_burn(rgb, faces, p: Params, head_hair, edits, area_out: np.ndarray | None = None) -> np.ndarray:
    """apply's work; with ``area_out``, only where it works is found (into it)."""
    out = rgb if area_out is not None else rgb.copy()
    h, w = rgb.shape[:2]
    for face in faces:
        lm = face[:, :2] * np.array([w, h], np.float32)
        fw = face_width(lm)
        if fw < 40:
            continue
        outline = profile_outline(lm)  # W below is colour-tested
        x0, y0 = np.maximum(np.floor(outline.min(0) - 0.2 * fw), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(outline.max(0) + 0.2 * fw), [w, h]).astype(int)
        crop = np.clip(out[y0:y1, x0:x1], 0, 1)
        # The light map is smooth, so it's worked out on a small copy.
        s = min(1.0, WORK_FW / fw)
        small = crop if s == 1 else cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        scale = np.array([small.shape[1] / crop.shape[1], small.shape[0] / crop.shape[0]], np.float32)
        slm = (lm - [x0, y0]) * scale
        sfw = fw * scale[0]
        sh = small.shape[:2]
        region = _inner_feather(
            cv2.fillPoly(np.zeros(sh, np.float32), [np.round((outline - [x0, y0]) * scale).astype(np.int32)], 1.0),
            EDGE_FADE * sfw,
        )
        skin_w, hair, _ = face_skin(cv2.cvtColor(small, cv2.COLOR_RGB2Lab), slm, outline=(outline - [x0, y0]) * scale)
        W = skin_w * (1 - hair)
        on_hair = None
        if head_hair is not None:
            on_hair = _smoothstep(_map_into(head_hair, (0, 0, 1, 1), (x0, y0, x1, y1), (h, w), sh), *HAIR_SURE)
            on_hair = np.clip(blur(on_hair, max(0.6, HAIR_SOFT * sfw)) * 1.5, 0, 1)
            W = W * (1 - on_hair)
        Y = srgb_to_linear(small) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        shape = light_shape(Y, W, sfw)
        # The change is kept off the eyes and lips, fading smoothly; hair,
        # stubble and nostrils are only left out of the measurement
        # (following the skin mask's edges drew them into the result as lines).
        keep_off = np.zeros(sh, np.float32)
        for spec in EYES.values():
            keep_off = np.maximum(keep_off, _poly_mask(sh, slm[spec["contour"]]))
        keep_off = np.maximum(keep_off, _poly_mask(sh, slm[LIPS]))
        keep_off = np.clip(blur(grow_mask(keep_off, 2 * round(KEEP_OFF_MARGIN * sfw) + 1), KEEP_OFF_FADE * sfw) * 1.5, 0, 1)
        weight = region * (1 - keep_off)
        # Facial hair is left as it is: burning darkened and dodging greyed a
        # beard, and Highlights/Shadows strengthened its own light and shade
        # ~2.4x as much as skin's. Faded over a wide, soft edge: following the
        # stubble map's patchy edge drew it into the result as lines.
        beard = np.clip(blur(hair, max(0.7, BEARD_SOFT * sfw)) * 1.3, 0, 1)
        weight = weight * (1 - BEARD_KEEP * beard)
        if on_hair is not None:
            weight = weight * (1 - on_hair)
        weight = region_edit.apply(weight, edits or [], (h, w), (x0, y0, x1, y1))
        if area_out is not None:
            shown = cv2.resize(weight, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) if s != 1 else weight
            area_out[y0:y1, x0:x1] = np.maximum(area_out[y0:y1, x0:x1], shown)
            continue

        stops = tone_curve(shape, p)
        if p.contour > 0:
            dodge, burn = contour_zones(sh, slm, sfw)
            t = np.clip((shape - DODGE_IN_SHADE[0]) / (DODGE_IN_SHADE[1] - DODGE_IN_SHADE[0]), 0, 1)
            u = np.clip((shape - BURN_IN_LIGHT[0]) / (BURN_IN_LIGHT[1] - BURN_IN_LIGHT[0]), 0, 1)
            stops = stops + p.contour * (CONTOUR_DODGE * dodge * t - CONTOUR_BURN * burn * (1 - u))
        stops = stops * weight
        if s != 1:
            stops = cv2.resize(stops, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
        if on_hair is not None:
            near = cv2.resize(blur(on_hair, STRANDS_NEAR_HAIR * sfw), (crop.shape[1], crop.shape[0]))
            near = np.clip(near * 4, 0, 1)
            ys, xs = np.nonzero((near > 0.01) & (stops != 0))
            if len(ys):
                # Traced only where it matters: near the hair, where there's a change.
                pad = 2 * round(STRANDS_PAD * fw) + 1
                ya, yb = max(0, ys.min() - pad), min(near.shape[0], ys.max() + 1 + pad)
                xa, xb = max(0, xs.min() - pad), min(near.shape[1], xs.max() + 1 + pad)
                L = cv2.cvtColor(crop[ya:yb, xa:xb].astype(np.float32), cv2.COLOR_RGB2Lab)[..., 0]
                stops[ya:yb, xa:xb] *= 1 - _strands(L, fw) * near[ya:yb, xa:xb]
        lin = srgb_to_linear(crop)
        # Dodging near-white skin only blows it out. Judged over an area, not
        # per pixel: per pixel, bright and dark specks were dodged differently,
        # which changed the texture.
        bright = blur(lin.max(axis=2), max(0.7, 0.01 * fw))
        fade = 1 - np.clip((bright - HIGHLIGHT_FADE[0]) / (HIGHLIGHT_FADE[1] - HIGHLIGHT_FADE[0]), 0, 1)
        stops = np.where(stops > 0, stops * fade, stops)
        out[y0:y1, x0:x1] = linear_to_srgb(np.clip(lin * np.exp2(stops)[..., None], 0, 1))
    return out

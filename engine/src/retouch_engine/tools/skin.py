"""Skin retouching: blemishes, smoothing, tone evening and shine, per region.

Regions (face now; neck and body to follow) are found without face-parsing
models (those are trained on non-commercial data): the face outline comes from
the landmarks, and "is this skin?" from a colour model sampled from each
person's own cheeks and forehead, so it adapts to every skin tone and to the
lighting. Eyes, brows and lips are excluded by landmarks; hair, beard, nostrils
and clothing by the colour model; stubble by its texture (see ``stubble``).

Distances are in face widths (landmark 234 to 454) so everything scales with
the person and matches between preview and export.
"""

from __future__ import annotations

import cv2
import numpy as np

from dataclasses import dataclass

from .colour import linear_to_srgb, srgb_to_linear
from .eyes import EYES, FACE_OVAL, _poly_mask, _relight
from .filters import masked_blur
from .reflection import guided_filter

LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
CHEEK_SAMPLES = (50, 280)  # landmarks at the centre of each cheek
FOREHEAD_SAMPLE = 151
CHEEK_RADIUS = 0.08  # face widths
FOREHEAD_RADIUS = 0.06
EYE_CLEARANCE = 0.25  # eye widths around the eye opening (lashes)
BROW_CLEARANCE = 0.1
LIP_CLEARANCE = 0.02  # face widths
COLOUR_SPREAD = 3.0  # Mahalanobis distance (chromaticity) still counted as skin
DARK_MARGIN = 35.0  # Lab L below the sampled skin's darker tones: hair, nostrils
HIGHLIGHT = (6.0, 18.0)  # Lab L above typical skin: shine, which is still skin
STUBBLE_FINE = 0.004  # face widths: the scale of stubble's texture
STUBBLE_AREA = 0.012
STUBBLE_TEXTURE = (1.8, 3.5)  # texture energy vs the cheeks'
BEARD_LINE_FADE = 0.08  # face widths below the cheekbone line to fade stubble in


def face_width(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[234] - lm[454]))


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def _disk(shape, centre, radius) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, (round(centre[0]), round(centre[1])), max(1, round(radius)), 1, -1)
    return mask.astype(bool)


def _dilated(mask: np.ndarray, px: float) -> np.ndarray:
    k = 2 * max(1, round(px)) + 1
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _chromaticity(lab: np.ndarray) -> np.ndarray:
    """(r, g) share of linear light: the same for a patch of skin whether it's
    lit or in shadow, unlike Lab a/b, which shrink as skin darkens."""
    lin = srgb_to_linear(np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1))
    total = np.maximum(lin.sum(axis=-1, keepdims=True), 1e-4)
    return (lin / total)[..., :2].astype(np.float32)


class SkinModel:
    """A person's skin colour, sampled from cheeks and forehead.

    Skin is recognised by chromaticity (lighting-independent), with a lenient
    darkness cut-off for hair and nostrils; highlights brighter than typical
    skin are skin too (shine is nearly colourless, so fails the colour test)."""

    def __init__(self, lab: np.ndarray, lm: np.ndarray):
        fw = face_width(lm)
        shape = lab.shape[:2]
        sample = np.zeros(shape, bool)
        for i in CHEEK_SAMPLES:
            sample |= _disk(shape, lm[i], CHEEK_RADIUS * fw)
        sample |= _disk(shape, lm[FOREHEAD_SAMPLE], FOREHEAD_RADIUS * fw)
        px = lab[sample]
        chroma = _chromaticity(lab)[sample]
        # Drop specular highlights and deep shadow before modelling the colour.
        lo, hi = np.percentile(px[:, 0], [15, 85])
        keep = (px[:, 0] >= lo) & (px[:, 0] <= hi)
        px, chroma = px[keep], chroma[keep]
        self.rg_mean = chroma.mean(0)
        cov = np.cov(chroma.T) + np.eye(2) * 1e-4  # never narrower than ~0.01
        self.rg_inv = np.linalg.inv(cov)
        self.L_low = float(np.percentile(px[:, 0], 5))
        self.L_typical = float(np.median(px[:, 0]))
        self.tone = px.mean(0)  # the person's typical skin, for tone evening

    def probability(self, lab: np.ndarray) -> np.ndarray:
        d = _chromaticity(lab) - self.rg_mean
        m2 = np.einsum("...i,ij,...j->...", d, self.rg_inv, d)
        colour = np.exp(-0.5 * m2 / COLOUR_SPREAD**2 * 4)
        L = lab[..., 0]
        lit = _smoothstep(L, self.L_low - DARK_MARGIN, self.L_low - DARK_MARGIN / 3)
        shine = _smoothstep(L, self.L_typical + HIGHLIGHT[0], self.L_typical + HIGHLIGHT[1])
        return np.maximum(colour * lit, shine).astype(np.float32)


def stubble(lab: np.ndarray, face: np.ndarray, lm: np.ndarray, model: SkinModel) -> np.ndarray:
    """Where facial hair (stubble, beard shadow) is, 0..1.

    Stubble is dense fine texture over slightly darker, bluer skin. Texture
    energy at stubble scale is compared with the cheeks' own, so pores don't
    count; it's then gated by being darker or bluer than the person's skin."""
    fw = face_width(lm)
    L = lab[..., 0]
    fine = L - cv2.GaussianBlur(L, (0, 0), max(0.5, STUBBLE_FINE * fw))
    energy = cv2.GaussianBlur(fine * fine, (0, 0), max(0.7, STUBBLE_AREA * fw))
    ref_px = np.zeros(L.shape, bool)
    for i in CHEEK_SAMPLES:
        ref_px |= _disk(L.shape, lm[i], CHEEK_RADIUS * fw)
    ref = float(np.median(energy[ref_px])) + 1e-6
    textured = _smoothstep(energy / ref, *STUBBLE_TEXTURE)
    darker = _smoothstep(model.tone[0] - L, 2.0, 10.0)
    bluer = _smoothstep(model.tone[2] - lab[..., 2], 1.5, 6.0)
    # Facial hair only grows below the cheekbones: shadow noise on the forehead
    # or upper cheeks can look like texture, so don't look for it there.
    a, b = lm[234], lm[454]
    normal = np.array([-(b - a)[1], (b - a)[0]], np.float32)
    normal /= max(float(np.linalg.norm(normal)), 1e-6)
    if np.dot(lm[152] - a, normal) < 0:  # point it toward the chin
        normal = -normal
    yy, xx = np.mgrid[0 : L.shape[0], 0 : L.shape[1]].astype(np.float32)
    below = ((xx - a[0]) * normal[0] + (yy - a[1]) * normal[1]) / fw
    beard_zone = _smoothstep(below, 0.0, BEARD_LINE_FADE)
    hair = textured * np.maximum(darker, bluer) * face * beard_zone
    hair = cv2.GaussianBlur(hair, (0, 0), max(0.7, STUBBLE_AREA * fw))
    return np.clip(hair * 1.5, 0, 1).astype(np.float32)


def face_skin(lab: np.ndarray, lm: np.ndarray, alpha: np.ndarray | None = None):
    """Face skin weight (0..1), the stubble map, and the person's SkinModel,
    for a Lab image and one face's landmarks in its pixel coordinates."""
    shape = lab.shape[:2]
    fw = face_width(lm)
    face = _poly_mask(shape, lm[FACE_OVAL])
    if alpha is not None:
        face *= alpha
    exclude = np.zeros(shape, np.float32)
    for spec in EYES.values():
        ew = float(np.linalg.norm(lm[spec["corners"][0]] - lm[spec["corners"][1]]))
        exclude = np.maximum(exclude, _dilated(_poly_mask(shape, lm[spec["contour"]]), EYE_CLEARANCE * ew))
        brow = _poly_mask(shape, cv2.convexHull(lm[spec["brow"]].astype(np.float32))[:, 0])
        exclude = np.maximum(exclude, _dilated(brow, BROW_CLEARANCE * ew))
    exclude = np.maximum(exclude, _dilated(_poly_mask(shape, lm[LIPS]), LIP_CLEARANCE * fw))
    model = SkinModel(lab, lm)
    skin = face * (1 - exclude) * model.probability(lab)
    hair = stubble(lab, face * (1 - exclude), lm, model)
    return skin.astype(np.float32), hair, model


# --- the Face tools -----------------------------------------------------------

BLEMISH_SCALES = (0.004, 0.007, 0.012, 0.02)  # spot sizes, face widths (pores are smaller)
BLEMISH_ROUND = 0.35  # smaller/larger curvature ratio: round spots, not lines
# Calibrated on real skin spots, which score ~0.5-1 (ear folds, stubble edges
# and hair score far higher, but the skin mask excludes them anyway).
BLEMISH_DARK = (0.5, 1.2)  # spot strength ramp, Lab L
BLEMISH_RED = (0.35, 0.9)  # spot strength ramp, Lab a
TEXTURE_KEEP = 0.003  # face widths: finer than this is texture, always kept
SMOOTH_BAND = (0.006, 0.04)  # blotchy light and shade flattened by Smooth
SMOOTH_MAX = 0.75
TONE_SIGMA = 0.03  # broad colour evened by Even tone
TONE_MAX_SHIFT = 12.0
SHINE_SIGMA = 0.05
SHINE_EXCESS = (3.0, 12.0)  # Lab L above the surrounding skin
# "The surrounding skin" is estimated edge-aware (guided filter): low-contrast
# blotches are smoothed, but a strong edge such as the shadow line of split
# lighting is kept, so no tool mistakes the lighting for a skin flaw.
EDGE_EPS = 36.0  # (Lab L)^2: contrasts well above ~6 L count as edges
# Lab L change across a spot's scale that marks a sharp shading edge (a
# shadow line). A cheek's gentle shading is well below this.
EDGE_GATE = (10.0, 18.0)


@dataclass
class RegionParams:
    blemishes: float = 0.0
    smooth: float = 0.0
    even: float = 0.0
    shine: float = 0.0  # -1 matte .. 0 natural .. +1 gloss

    @classmethod
    def from_dict(cls, d: dict | None) -> "RegionParams":
        d = d or {}
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_noop(self) -> bool:
        return self.blemishes <= 0 and self.smooth <= 0 and self.even <= 0 and self.shine == 0


def _blobs(channel: np.ndarray, fw: float) -> np.ndarray:
    """Round-spot strength (units of ``channel``) where it dips: both principal
    curvatures positive and similar, so lines and wrinkles don't count."""
    out = np.zeros_like(channel)
    for scale in BLEMISH_SCALES:
        sigma = max(0.7, scale * fw)
        g = cv2.GaussianBlur(channel, (0, 0), sigma)
        gy, gx = np.gradient(g)
        gxy, gxx = np.gradient(gx)
        gyy, _ = np.gradient(gy)
        half_tr = (gxx + gyy) / 2
        root = np.sqrt(((gxx - gyy) / 2) ** 2 + gxy**2)
        l1, l2 = half_tr + root, half_tr - root
        round_ = np.clip(l2 / np.maximum(l1, 1e-6), 0, 1)
        strength = np.where(l2 > 0, sigma**2 * l2, 0) * _smoothstep(round_, BLEMISH_ROUND, 0.7)
        out = np.maximum(out, strength)
    return out


def _skin_base(channel: np.ndarray, W: np.ndarray, radius_px: float) -> np.ndarray:
    """Edge-aware "surrounding skin": non-skin is filled in from skin first so
    it can't bleed in, then a guided filter smooths blotches but keeps edges."""
    filled = W * channel + (1 - W) * masked_blur(channel, W, max(1.0, radius_px))
    guide = np.ascontiguousarray(filled if channel.ndim == 2 else filled[..., 0])
    if channel.ndim == 2:
        return guided_filter(guide, filled, max(1, round(radius_px)), EDGE_EPS)
    return np.stack(
        [guided_filter(guide, np.ascontiguousarray(filled[..., i]), max(1, round(radius_px)), EDGE_EPS)
         for i in range(channel.shape[2])],
        axis=-1,
    )


def _heal_blemishes(lab, W, fw, amount):
    L = np.ascontiguousarray(lab[..., 0])
    a = np.ascontiguousarray(lab[..., 1])
    dark = _smoothstep(_blobs(L, fw), *[t * (1.4 - 0.6 * amount) for t in BLEMISH_DARK])
    red = _smoothstep(_blobs(-a, fw), *[t * (1.4 - 0.6 * amount) for t in BLEMISH_RED])
    # A spot sitting on a strong shading edge (e.g. a shadow line) is lighting.
    broad = cv2.GaussianBlur(L, (0, 0), max(0.7, BLEMISH_SCALES[-1] * fw))
    gy, gx = np.gradient(broad)
    edge = _smoothstep(np.hypot(gx, gy) * 2 * BLEMISH_SCALES[-1] * fw, *EDGE_GATE)
    spot = np.maximum(dark, red) * W * (1 - edge)
    grow = max(1.0, BLEMISH_SCALES[-1] * fw)
    spot = cv2.GaussianBlur(cv2.dilate(spot, np.ones((3, 3), np.uint8), iterations=max(1, round(grow / 2))), (0, 0), grow / 2)
    spot = np.clip(spot * 1.5, 0, 1)
    # The skin around each spot, then only its broad colour/tone is moved: the
    # finest texture stays, so a healed spot isn't a smooth patch.
    around = masked_blur(lab, W * (1 - spot) ** 2, max(1.0, 1.5 * BLEMISH_SCALES[-1] * fw))
    keep = max(0.5, TEXTURE_KEEP * fw)
    delta = cv2.GaussianBlur(around, (0, 0), keep) - cv2.GaussianBlur(lab, (0, 0), keep)
    lab += (amount * spot)[..., None] * delta


def _smooth(lab, W, fw, amount, eye_w):
    L = np.ascontiguousarray(lab[..., 0])
    fine = cv2.GaussianBlur(L, (0, 0), max(0.5, SMOOTH_BAND[0] * fw))
    coarse = _skin_base(L, W, SMOOTH_BAND[1] * fw)
    _relight(lab, -amount * SMOOTH_MAX * W * (fine - coarse), eye_w)
    ab = np.ascontiguousarray(lab[..., 1:])
    fine_ab = cv2.GaussianBlur(ab, (0, 0), max(0.5, SMOOTH_BAND[0] * fw))
    coarse_ab = masked_blur(ab, W, max(1.0, SMOOTH_BAND[1] * fw))
    lab[..., 1:] -= (0.5 * amount * W)[..., None] * (fine_ab - coarse_ab)


def _even_tone(lab, W, fw, amount, model):
    """Move broad colour toward the person's skin, in lighting-independent
    colour proportions (linear light) and keeping each pixel's brightness.
    Shifting Lab a/b toward the cheeks instead over-saturates shadowed skin,
    which is naturally less saturated in Lab terms, into orange."""
    lin = srgb_to_linear(np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)).astype(np.float32)
    total = np.maximum(lin.sum(axis=-1, keepdims=True), 1e-4)
    chroma = lin / total
    broad = masked_blur(chroma, W, max(1.0, TONE_SIGMA * fw))
    target = np.array([*model.rg_mean, 1 - model.rg_mean.sum()], np.float32)
    gain = np.clip(target / np.maximum(broad, 1e-4), 0.8, 1.25)
    gain = 1 + (amount * W)[..., None] * (gain - 1)
    out = lin * gain
    # Keep brightness: rescale so luminance is unchanged.
    Y = lambda x: x @ np.array([0.2126, 0.7152, 0.0722], np.float32)  # noqa: E731
    out *= (Y(lin) / np.maximum(Y(out), 1e-6))[..., None]
    lab[:] = cv2.cvtColor(linear_to_srgb(np.clip(out, 0, 1)).astype(np.float32), cv2.COLOR_RGB2Lab)


def _shine(lab, W, fw, amount, eye_w):
    L = np.ascontiguousarray(lab[..., 0])
    around = _skin_base(L, W, SHINE_SIGMA * fw)
    excess = np.clip(L - around, 0, None)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    around_c = masked_blur(chroma, W, max(1.0, SHINE_SIGMA * fw))
    washed = _smoothstep(around_c - chroma, 0.0, 8.0)  # shine is paler than skin
    shine = _smoothstep(excess, *SHINE_EXCESS) * np.maximum(washed, 0.3) * W
    if amount < 0:  # matte: tone the hotspot down and give it back its colour
        _relight(lab, amount * 0.85 * shine * excess, eye_w)
        ab_around = masked_blur(np.ascontiguousarray(lab[..., 1:]), W * (1 - shine) ** 2, max(1.0, SHINE_SIGMA * fw))
        lab[..., 1:] += (-amount * shine)[..., None] * (ab_around - lab[..., 1:])
    else:  # gloss: strengthen the highlight that's already there
        _relight(lab, amount * 0.7 * shine * excess, eye_w)
        lab[..., 1:] *= (1 - 0.3 * amount * shine)[..., None]


def _apply_region(lab, W, fw, p: RegionParams, model):
    eye_w = 0.2 * fw  # the eye tools' colour-matching scale, in this face's units
    if p.blemishes > 0:
        _heal_blemishes(lab, W, fw, p.blemishes)
    if p.even > 0:
        _even_tone(lab, W, fw, p.even, model)
    if p.smooth > 0:
        _smooth(lab, W, fw, p.smooth, eye_w)
    if p.shine != 0:
        _shine(lab, W, fw, p.shine, eye_w)


def apply(rgb: np.ndarray, faces: list[np.ndarray], face_params: dict | None) -> np.ndarray:
    """Face skin tools on every face. rgb float32 HxWx3 0..1; faces are
    landmark arrays as fractions of width/height. Returns a new rgb."""
    p = RegionParams.from_dict(face_params)
    out = rgb.copy()
    if p.is_noop():
        return out
    h, w = rgb.shape[:2]
    for face in faces:
        lm = face * np.array([w, h], np.float32)
        fw = face_width(lm)
        if fw < 40:  # too small to retouch meaningfully
            continue
        pts = lm[FACE_OVAL]
        x0, y0 = np.maximum(np.floor(pts.min(0) - 0.15 * fw), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(pts.max(0) + 0.15 * fw), [w, h]).astype(int)
        crop = np.clip(out[y0:y1, x0:x1], 0, 1)
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab)
        skin_w, hair, model = face_skin(lab, lm - [x0, y0])
        W = cv2.GaussianBlur(skin_w * (1 - hair), (0, 0), max(0.7, 0.004 * fw))
        _apply_region(lab, W, fw, p, model)
        out[y0:y1, x0:x1] = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
    return out

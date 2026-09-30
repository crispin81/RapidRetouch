"""Skin retouching: blemishes, smoothing, tone evening and shine, per region.

Regions (face, neck, body) are found without face-parsing models (those are
trained on non-commercial data): the face outline comes from the landmarks, and
"is this skin?" from a colour model sampled from each person's own cheeks and
forehead, so it adapts to every skin tone and to the lighting. Eyes, brows and
lips are excluded by landmarks; hair, beard, nostrils and clothing by the
colour model; stubble by its texture (see ``stubble``). The neck is the skin
below the jaw, a set distance down and either side of the chin; the body is
the rest of the person's skin (see ``body_regions``).

Distances are in face widths (landmark 234 to 454) so everything scales with
the person and matches between preview and export.
"""

from __future__ import annotations

import copy

import cv2
import numpy as np

from dataclasses import dataclass

from .colour import linear_to_srgb, srgb_to_linear
from .eyes import EYES, FACE_OVAL, LINE_HIGH, LINE_LOW, _inner_feather, _poly_mask, _relight, _wrinkle_lift
from .filters import blur, grow_mask, masked_blur, max_filter
from . import region_edit
from .reflection import guided_filter

LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
CHEEK_SAMPLES = (50, 280)  # landmarks at the centre of each cheek
FOREHEAD_SAMPLE = 151
CHEEK_RADIUS = 0.08  # face widths
FOREHEAD_RADIUS = 0.06
EYE_CLEARANCE = 0.25  # eye widths around the eye opening (lashes)
BROW_CLEARANCE = 0.1
LIP_CLEARANCE = 0.02  # face widths
WORK_FW = 600  # px: larger faces are worked out at this width and scaled up
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
    return grow_mask(mask, 2 * max(1, round(px)) + 1)


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

    def colour_probability(self, lab: np.ndarray, spread: float = COLOUR_SPREAD) -> np.ndarray:
        """Skin by colour alone (no shine term): away from the face, bright
        near-white things are clothes far more often than shiny skin. A wider
        ``spread`` also takes in skin lit differently from the face."""
        d = _chromaticity(lab) - self.rg_mean
        m2 = np.einsum("...i,ij,...j->...", d, self.rg_inv, d)
        colour = np.exp(-0.5 * m2 / spread**2 * 4)
        lit = _smoothstep(lab[..., 0], self.L_low - DARK_MARGIN, self.L_low - DARK_MARGIN / 3)
        return (colour * lit).astype(np.float32)

    def probability(self, lab: np.ndarray) -> np.ndarray:
        d = _chromaticity(lab) - self.rg_mean
        m2 = np.einsum("...i,ij,...j->...", d, self.rg_inv, d)
        colour = np.exp(-0.5 * m2 / COLOUR_SPREAD**2 * 4)
        L = lab[..., 0]
        lit = _smoothstep(L, self.L_low - DARK_MARGIN, self.L_low - DARK_MARGIN / 3)
        shine = _smoothstep(L, self.L_typical + HIGHLIGHT[0], self.L_typical + HIGHLIGHT[1])
        return np.maximum(colour * lit, shine).astype(np.float32)


def stubble(
    lab: np.ndarray,
    face: np.ndarray,
    lm: np.ndarray,
    model: SkinModel,
    relative: bool = False,
    texture: tuple[float, float] = STUBBLE_TEXTURE,
) -> np.ndarray:
    """Where facial hair (stubble, beard shadow) is, 0..1.

    Stubble is dense fine texture over slightly darker, bluer skin. Texture
    energy at stubble scale is compared with the cheeks' own, so pores don't
    count; it's then gated by being darker or bluer than the person's skin.
    ``relative`` measures texture as a share of the local brightness: under
    the jaw the neck is in shadow, where the same stubble has far less
    contrast than on the lit cheeks it's compared with."""
    fw = face_width(lm)
    L = lab[..., 0]
    fine = L - blur(L, max(0.5, STUBBLE_FINE * fw))
    if relative:
        fine = fine / np.maximum(blur(L, max(0.7, STUBBLE_AREA * fw)), 10.0) * model.L_typical
    energy = blur(fine * fine, max(0.7, STUBBLE_AREA * fw))
    ref_px = np.zeros(L.shape, bool)
    for i in CHEEK_SAMPLES:
        ref_px |= _disk(L.shape, lm[i], CHEEK_RADIUS * fw)
    ref = float(np.median(energy[ref_px])) + 1e-6
    textured = _smoothstep(energy / ref, *texture)
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
    hair = blur(hair, max(0.7, STUBBLE_AREA * fw))
    return np.clip(hair * 1.5, 0, 1).astype(np.float32)


def forehead_outline(lm: np.ndarray) -> np.ndarray:
    """The face outline with its top raised to about the hairline (the
    landmarks' outline stops partway up the forehead)."""
    up = -_face_down(lm) * HAIRLINE_REACH * face_width(lm)
    top = set(FOREHEAD_ARC)
    return np.array([lm[i] + up if i in top else lm[i] for i in FACE_OVAL], np.float32)


def face_skin(lab: np.ndarray, lm: np.ndarray, alpha: np.ndarray | None = None, outline: np.ndarray | None = None):
    """Face skin weight (0..1), the stubble map, and the person's SkinModel,
    for a Lab image and one face's landmarks in its pixel coordinates.
    ``outline`` replaces the landmarks' face outline (see ``forehead_outline``)."""
    region = face_region(lab.shape[:2], lm, alpha, outline)
    model = SkinModel(lab, lm)
    skin = region * model.probability(lab)
    hair = stubble(lab, region, lm, model)
    return skin.astype(np.float32), hair, model


def face_region(shape, lm: np.ndarray, alpha: np.ndarray | None = None, outline: np.ndarray | None = None) -> np.ndarray:
    """The face outline minus eyes, brows and lips (0..1): where skin can be,
    before the colour test."""
    fw = face_width(lm)
    face = _poly_mask(shape, lm[FACE_OVAL] if outline is None else outline)
    if alpha is not None:
        face *= alpha
    exclude = np.zeros(shape, np.float32)
    for spec in EYES.values():
        ew = float(np.linalg.norm(lm[spec["corners"][0]] - lm[spec["corners"][1]]))
        exclude = np.maximum(exclude, _dilated(_poly_mask(shape, lm[spec["contour"]]), EYE_CLEARANCE * ew))
        brow = _poly_mask(shape, cv2.convexHull(lm[spec["brow"]].astype(np.float32))[:, 0])
        exclude = np.maximum(exclude, _dilated(brow, BROW_CLEARANCE * ew))
    exclude = np.maximum(exclude, _dilated(_poly_mask(shape, lm[LIPS]), LIP_CLEARANCE * fw))
    return (face * (1 - exclude)).astype(np.float32)


# --- wrinkle zones (Face tab) -------------------------------------------------

BROW_TOPS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]  # image-left to right
FOREHEAD_ARC = [21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251]  # upper face outline
BROW_LIFT = 0.03  # forehead zone starts this far above the brows (face widths)
HAIRLINE_REACH = 0.12  # the outline landmarks stop short of the hairline
FROWN_HEIGHT = 0.1
NOSE_WINGS = (129, 358)  # beside each nostril
MOUTH_CORNERS = (61, 291)
SMILE_EXTEND = 0.35  # smile-line zone runs this far past the mouth corner
SMILE_WIDTH = 0.11  # face widths
SMILE_OUTWARD = 0.03  # the fold sits on the cheek side of the nose-mouth line
CHEEK_INNER = 0.05  # face widths out from the nose wing / mouth corner: the cheek zone's inner edge
CHEEK_OUTER = 0.22  # ...and its outer edge
CHEEK_TOP = 0.05  # face widths above the nose wing (below the cheekbone)
CHEEK_BOTTOM = 0.14  # face widths below the mouth corner
CHIN_DEPTH = 0.22
CHIN_SPREAD = 0.06


def _face_down(lm: np.ndarray) -> np.ndarray:
    across = lm[454] - lm[234]
    down = np.array([-across[1], across[0]], np.float32)
    down /= max(float(np.linalg.norm(down)), 1e-6)
    return down if np.dot(lm[152] - lm[234], down) > 0 else -down


def wrinkle_zones(shape: tuple[int, int], lm: np.ndarray) -> dict[str, np.ndarray]:
    """Hard masks for each wrinkle slider's area, in the image's pixels."""
    fw = face_width(lm)
    down = _face_down(lm)
    across = (lm[454] - lm[234]) / max(float(np.linalg.norm(lm[454] - lm[234])), 1e-6)
    up = -down

    frown = np.array(
        [lm[107] + up * FROWN_HEIGHT * fw, lm[336] + up * FROWN_HEIGHT * fw, lm[336], lm[168],
         lm[107]],
        np.float32,
    )
    brows = lm[BROW_TOPS] + up * BROW_LIFT * fw
    forehead = np.concatenate([lm[FOREHEAD_ARC] + up * HAIRLINE_REACH * fw, brows[::-1]])

    smile = np.zeros(shape, np.float32)
    for wing, corner, side in ((NOSE_WINGS[0], MOUTH_CORNERS[0], -1), (NOSE_WINGS[1], MOUTH_CORNERS[1], 1)):
        start, end = lm[wing], lm[corner] + (lm[corner] - lm[wing]) * SMILE_EXTEND
        shift = across * side * SMILE_OUTWARD * fw
        band = np.zeros(shape, np.uint8)
        cv2.line(band, tuple(np.round(start + shift).astype(int)), tuple(np.round(end + shift).astype(int)),
                 1, max(1, round(SMILE_WIDTH * fw)))
        smile = np.maximum(smile, band.astype(np.float32))

    l, r = lm[MOUTH_CORNERS[0]], lm[MOUTH_CORNERS[1]]
    chin = np.array(
        [l - across * CHIN_SPREAD * fw, r + across * CHIN_SPREAD * fw,
         r + across * CHIN_SPREAD * fw + down * CHIN_DEPTH * fw,
         l - across * CHIN_SPREAD * fw + down * CHIN_DEPTH * fw],
        np.float32,
    )
    # The cheek beside each smile line, where finer lines fan out across the
    # cheek (more so with age): from just outside the fold, below the
    # cheekbone, down past the mouth corner.
    cheeks = np.zeros(shape, np.float32)
    for wing, corner, side in ((NOSE_WINGS[0], MOUTH_CORNERS[0], -1), (NOSE_WINGS[1], MOUTH_CORNERS[1], 1)):
        out = across * side * fw
        cheek = np.array(
            [lm[wing] + out * CHEEK_INNER + up * CHEEK_TOP * fw,
             lm[wing] + out * CHEEK_OUTER + up * CHEEK_TOP * fw,
             lm[corner] + out * (CHEEK_OUTER + 0.04) + down * CHEEK_BOTTOM * fw,
             lm[corner] + out * CHEEK_INNER + down * CHEEK_BOTTOM * fw],
            np.float32,
        )
        cheeks = np.maximum(cheeks, _poly_mask(shape, cheek))

    lips = _dilated(_poly_mask(shape, lm[LIPS]), LIP_CLEARANCE * fw)
    frown_mask = _poly_mask(shape, frown)
    return {
        "forehead_lines": _poly_mask(shape, forehead) * (1 - frown_mask),
        "frown_lines": frown_mask,
        "smile_lines": smile * (1 - lips),
        # Overlapping the fold band: each zone fades in from its own edges,
        # so zones that only met left a strip along the fold untouched. The
        # cheek pass runs after the fold's, so it only fills what's left.
        "cheek_lines": np.maximum(cheeks, smile) * (1 - lips),
        "chin_lines": _poly_mask(shape, chin) * (1 - lips) * (1 - smile),
    }


# --- the Face tools -----------------------------------------------------------

BLEMISH_SCALES = (0.004, 0.007, 0.012, 0.02)  # spot sizes, face widths (pores are smaller)
BLEMISH_ROUND = 0.35  # smaller/larger curvature ratio: round spots, not lines
# Calibrated on real skin spots, which score ~0.5-1 (ear folds, stubble edges
# and hair score far higher, but the skin mask excludes them anyway).
BLEMISH_DARK = (0.5, 1.2)  # spot strength ramp, Lab L
BLEMISH_RED = (0.35, 0.9)  # spot strength ramp, Lab a
TEXTURE_KEEP = 0.003  # face widths: finer than this is texture, always kept
# Moles are kept (part of someone's identity; the Remove brush takes one out):
# clearly darker than the skin around them, but not mainly redder, which is
# what marks a blemish.
MOLE_DARK = (2.0, 3.5)  # round-spot strength (Lab L): well above a typical blemish
MOLE_BROWN = (0.35, 0.7)  # redness per unit of darkness above this is a blemish
TEXTURE_BAND = (0.0015, 0.015)  # face widths: pores; finer is grain, coarser is Smooth's
TEXTURE_KNEE = 1.5  # Lab L: pores deeper than this are reduced the most
TEXTURE_MAX = 1.0
TEXTURE_BRIGHT_SHARE = 0.3  # bright specks (light catching the skin) lose this share as much
TEXTURE_HIGHLIGHT = (2.0, 6.0)  # Lab L above the surrounding skin: in a highlight, specks are kept
SMOOTH_BAND = (0.006, 0.04)  # blotchy light and shade flattened by Smooth
SMOOTH_MAX = 0.75
TONE_SIGMA = 0.03  # broad colour evened by Even tone
TONE_MAX_SHIFT = 12.0
TONE_GAIN = (0.7, 1.4)  # how far Even tone may shift each colour share (a red nose needs ~0.75)
SHINE_SIGMA = 0.05
SHINE_EXCESS = (3.0, 12.0)  # Lab L above the surrounding skin
# "The surrounding skin" is estimated edge-aware (guided filter): low-contrast
# blotches are smoothed, but a strong edge such as the shadow line of split
# lighting is kept, so no tool mistakes the lighting for a skin flaw.
EDGE_EPS = 36.0  # (Lab L)^2: contrasts well above ~6 L count as edges
# Lab L change across a spot's scale that marks a sharp shading edge (a
# shadow line). A cheek's gentle shading is well below this.
EDGE_GATE = (10.0, 18.0)


# Wrinkle sliders (Face tab): line scales in eye widths (the eye tool's line
# filler is reused, and a face is ~5 eye widths across), and how much of a
# deep fold is kept even at full strength.
WRINKLE_AREAS = {
    # area: (slider, line scales, share of a deep fold kept)
    "forehead_lines": ("forehead_lines", (0.012, 0.022, 0.035), 0.2),
    "frown_lines": ("frown_lines", (0.012, 0.02, 0.03), 0.25),
    "smile_lines": ("smile_lines", (0.02, 0.035, 0.05), 0.5),  # never erased: looks unnatural
    # The finer lines across the cheek beside it, on the same slider.
    "cheek_lines": ("smile_lines", (0.012, 0.022, 0.035), 0.25),
    "chin_lines": ("chin_lines", (0.015, 0.025, 0.04), 0.35),
}
WRINKLE_SLIDERS = ("forehead_lines", "frown_lines", "smile_lines", "chin_lines")
# Softer creases need a gentler "is this a line?" ramp than the default.
WRINKLE_LINE_RAMP = {"cheek_lines": (0.15, 0.5)}
# Areas placed precisely from the landmarks, around the mouth: they use the
# face region (facial hair and anything too dark still left out) instead of
# the skin colour test. The inside of a smile crease is shadowed and redder
# than the cheek, and failed the colour test for being a crease: only a
# quarter to a half of it was being filled.
PLACED_AREAS = ("smile_lines", "cheek_lines", "chin_lines")


@dataclass
class RegionParams:
    blemishes: float = 0.0
    smooth: float = 0.0
    even: float = 0.0
    shine: float = 0.0  # -1 matte .. 0 natural .. +1 gloss
    texture: float = 0.0  # softens pores; the finest grain is kept
    forehead_lines: float = 0.0  # the wrinkle sliders exist on the Face tab only
    frown_lines: float = 0.0
    smile_lines: float = 0.0
    chin_lines: float = 0.0
    neck_lines: float = 0.0  # Neck tab only

    @classmethod
    def from_dict(cls, d: dict | None) -> "RegionParams":
        d = d or {}
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_noop(self) -> bool:
        return self.shine == 0 and all(
            getattr(self, k) <= 0
            for k in ("blemishes", "smooth", "even", "texture", "neck_lines", *WRINKLE_SLIDERS)
        )


def _blobs(channel: np.ndarray, fw: float) -> np.ndarray:
    """Round-spot strength (units of ``channel``) where it dips: both principal
    curvatures positive and similar, so lines and wrinkles don't count."""
    out = np.zeros_like(channel)
    for scale in BLEMISH_SCALES:
        sigma = max(0.7, scale * fw)
        g = blur(channel, sigma)
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


def _moles(L: np.ndarray, a: np.ndarray, fw: float) -> np.ndarray:
    """Where moles are (0..1), grown a little so their edges are kept too."""
    # Compact and round (shadows along the nose or eye sockets are long, so the
    # round-spot detector ignores them), strongly darker, and brown not red.
    round_dark = _blobs(L, fw)
    small = max(0.7, BLEMISH_SCALES[1] * fw)
    large = max(1.0, 3 * BLEMISH_SCALES[-1] * fw)
    darker = blur(L, large) - blur(L, small)
    redder = blur(a, small) - blur(a, large)
    ratio = redder / np.maximum(darker, 1.0)
    mole = _smoothstep(round_dark, *MOLE_DARK) * (1 - _smoothstep(ratio, *MOLE_BROWN))
    grow = 2 * max(1, round(BLEMISH_SCALES[-1] * fw)) + 1
    return np.clip(max_filter(mole, grow), 0, 1)


def _spots(lab, W, fw, amount):
    """Where the blemishes are (0..1), as found at slider ``amount``: a higher
    setting takes in fainter spots. Moles and spots on a shading edge are left
    out."""
    L = np.ascontiguousarray(lab[..., 0])
    a = np.ascontiguousarray(lab[..., 1])
    dark = _smoothstep(_blobs(L, fw), *[t * (1.4 - 0.6 * amount) for t in BLEMISH_DARK])
    red = _smoothstep(_blobs(-a, fw), *[t * (1.4 - 0.6 * amount) for t in BLEMISH_RED])
    # A spot sitting on a strong shading edge (e.g. a shadow line) is lighting.
    broad = blur(L, max(0.7, BLEMISH_SCALES[-1] * fw))
    gy, gx = np.gradient(broad)
    edge = _smoothstep(np.hypot(gx, gy) * 2 * BLEMISH_SCALES[-1] * fw, *EDGE_GATE)
    spot = np.maximum(dark, red) * W * (1 - edge) * (1 - _moles(L, a, fw))
    grow = max(1.0, BLEMISH_SCALES[-1] * fw)
    spot = blur(cv2.dilate(spot, np.ones((3, 3), np.uint8), iterations=max(1, round(grow / 2))), grow / 2)
    return np.clip(spot * 1.5, 0, 1)


def _heal_blemishes(lab, W, fw, amount):
    """Heal spots, in place: each takes the colour and tone of the skin around
    it, and only its broad colour and tone change, so the finest texture stays
    and a healed spot isn't a smooth patch.

    (Repainting spots with LaMa was tried and removed: on fake spots it left
    35% against this heal's 39%, but it painted new freckles into freckled
    skin, left more redness on real faces, and took 1-2 s per slider move.)"""
    spot = _spots(lab, W, fw, amount)
    keep = max(0.5, TEXTURE_KEEP * fw)
    around = masked_blur(lab, W * (1 - spot) ** 2, max(1.0, 1.5 * BLEMISH_SCALES[-1] * fw))
    delta = blur(around, keep) - blur(lab, keep)
    lab += (amount * spot)[..., None] * delta


def _smooth(lab, W, fw, amount, eye_w):
    L = np.ascontiguousarray(lab[..., 0])
    fine = blur(L, max(0.5, SMOOTH_BAND[0] * fw))
    coarse = _skin_base(L, W, SMOOTH_BAND[1] * fw)
    _relight(lab, -amount * SMOOTH_MAX * W * (fine - coarse), eye_w)
    ab = np.ascontiguousarray(lab[..., 1:])
    fine_ab = blur(ab, max(0.5, SMOOTH_BAND[0] * fw))
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
    gain = np.clip(target / np.maximum(broad, 1e-4), *TONE_GAIN)
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


def _smooth_lines(lab, W, lm, fw, p: RegionParams, eye_w, W_region=None):
    """The Wrinkles group: fill creases in each area, pores kept, folds softened.
    ``W_region``: the face-region skin weight without the colour test, used
    where the zone itself is placed precisely from the landmarks (see
    PLACED_AREAS)."""
    zones = wrinkle_zones(lab.shape[:2], lm)
    for name, (slider, scales, fold_keep) in WRINKLE_AREAS.items():
        amount = getattr(p, slider)
        if amount <= 0:
            continue
        weight = W_region if (W_region is not None and name in PLACED_AREAS) else W
        zone = _inner_feather(zones[name], 0.03 * fw) * weight
        L = np.ascontiguousarray(lab[..., 0])
        ramp = WRINKLE_LINE_RAMP.get(name, (LINE_LOW, LINE_HIGH))
        _relight(lab, amount * _wrinkle_lift(L, zone, eye_w, scales, fold_keep, ramp), eye_w)


def _texture(lab: np.ndarray, W: np.ndarray, fw: float, amount: float) -> None:
    """Soften pores without the plastic look, in place.

    Plastic skin comes from wiping out all fine detail. Pores sit in a narrow
    band of sizes: finer than it is the skin's grain, which is what reads as
    real skin, and is kept as shot. Within the band, the most visible pores (the
    strongest deviations) are reduced most, so the texture evens out instead of
    vanishing.

    Pores are dark pits; the bright specks at the same size are mostly light
    catching the skin (the sparkle in a highlight), which is what makes skin
    look alive. So bright detail is only lightly softened, and not at all
    inside a highlight."""
    L = np.ascontiguousarray(lab[..., 0])
    grain = blur(L, max(0.5, TEXTURE_BAND[0] * fw))
    base = blur(L, max(0.7, TEXTURE_BAND[1] * fw))
    pores = grain - base
    # Stronger pores lose more: a soft knee at TEXTURE_KNEE (Lab L).
    strength = np.abs(pores) / (np.abs(pores) + TEXTURE_KNEE)
    reduce = amount * TEXTURE_MAX * (0.5 + 0.5 * strength)
    highlight = _smoothstep(base - blur(L, max(1.0, SHINE_SIGMA * fw)), *TEXTURE_HIGHLIGHT)
    reduce = np.where(pores > 0, reduce * TEXTURE_BRIGHT_SHARE * (1 - highlight), reduce)
    lab[..., 0] = L - W * reduce * pores


def _apply_region(lab, W, fw, p: RegionParams, model, lm=None, W_even=None):
    """``W_even``: where Even tone works, if not ``W`` (see ``apply``)."""
    eye_w = 0.2 * fw  # the eye tools' colour-matching scale, in this face's units
    if lm is not None:  # wrinkles first, while the fine detail is as shot
        _smooth_lines(lab, W, lm, fw, p, eye_w, W_even)
    if p.blemishes > 0:
        _heal_blemishes(lab, W, fw, p.blemishes)
    if p.even > 0:
        _even_tone(lab, W if W_even is None else W_even, fw, p.even, model)
    if p.smooth > 0:
        _smooth(lab, W, fw, p.smooth, eye_w)
    if p.shine != 0:
        _shine(lab, W, fw, p.shine, eye_w)


def _retouch(out: np.ndarray, box, fw: float, p: RegionParams, weights, work_fw: float = WORK_FW) -> None:
    """Run one region's tools on ``out[box]``, in place.

    ``weights(work, scale)`` is given the Lab crop the tools will work on and
    its size relative to the image, and returns the skin weight, the person's
    SkinModel and their landmarks in the crop's pixels (or None: no wrinkle
    zones), plus an optional function for anything else to do on the crop."""
    x0, y0, x1, y1 = box
    crop = np.clip(out[y0:y1, x0:x1], 0, 1)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab)
    # Large regions (zoomed-in detail, export) are worked out on a smaller copy:
    # every correction here is smooth at ``work_fw``, so the change is measured
    # there, scaled up and added to the full-resolution pixels, whose pores
    # and fine texture stay exactly as shot. ~10x faster at full resolution.
    s = min(1.0, work_fw / fw, WORK_EDGE / max(lab.shape[:2]))
    work = lab if s == 1 else cv2.resize(lab, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    before = work.copy()
    scale = np.array([work.shape[1] / lab.shape[1], work.shape[0] / lab.shape[0]], np.float32)
    wfw = fw * scale[0]
    W, model, wlm, extra, W_even = weights(work, scale)
    _apply_region(work, W, wfw, p, model, wlm, W_even)
    if extra is not None:
        extra(work, W, wfw)
    if s == 1:
        lab = work
    else:
        delta = cv2.resize(work - before, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_CUBIC)
        lab += delta
    if p.texture > 0:
        # Pores are too small to work out on the smaller copy: this one runs
        # at the image's own resolution.
        W_full = W if s == 1 else cv2.resize(W, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_LINEAR)
        _texture(lab, W_full, fw, p.texture)
    out[y0:y1, x0:x1] = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def apply(rgb: np.ndarray, faces: list[np.ndarray], face_params: dict | None, edits: list[dict] | None = None) -> np.ndarray:
    """Face skin tools on every face. rgb float32 HxWx3 0..1; faces are
    landmark arrays as fractions of width/height. ``edits``: the user's
    corrections to the face skin area (see region_edit). Returns a new rgb."""
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

        def weights(work, scale, lm=lm, origin=(x0, y0), box=(x0, y0, x1, y1)):
            wlm = (lm - origin) * scale
            skin_w, hair, model = face_skin(work, wlm)
            soft = max(0.7, 0.004 * face_width(wlm))
            W = blur(skin_w * (1 - hair), soft)
            # Even tone works on the whole face (not eyes, brows, lips, facial
            # hair or anything too dark to be skin) without the colour test: a
            # red nose fails that test for being red, and was left pink.
            W_even = np.maximum(W, blur(face_region(work.shape[:2], wlm) * (1 - hair) * _lit(work, model.L_low), soft))
            W = region_edit.apply(W, edits or [], (h, w), box)
            W_even = region_edit.apply(W_even, edits or [], (h, w), box)
            return W, model, wlm, None, W_even

        _retouch(out, (x0, y0, x1, y1), fw, p, weights)
    return out


def face_area(rgb: np.ndarray, faces: list[np.ndarray], edits: list[dict] | None = None) -> np.ndarray:
    """Where the Face tools work (0..1, the image's size): each face's skin,
    as the tools see it, with the user's corrections applied."""
    h, w = rgb.shape[:2]
    area = np.zeros((h, w), np.float32)
    for face in faces:
        lm = face * np.array([w, h], np.float32)
        fw = face_width(lm)
        if fw < 40:
            continue
        pts = lm[FACE_OVAL]
        x0, y0 = np.maximum(np.floor(pts.min(0) - 0.15 * fw), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(pts.max(0) + 0.15 * fw), [w, h]).astype(int)
        lab = cv2.cvtColor(np.clip(rgb[y0:y1, x0:x1], 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
        skin_w, hair, _ = face_skin(lab, lm - [x0, y0])
        area[y0:y1, x0:x1] = np.maximum(area[y0:y1, x0:x1], skin_w * (1 - hair))
    return region_edit.apply(area, edits or [], (h, w))


# --- neck and body ------------------------------------------------------------

WORK_EDGE = 1600  # px: the longest side a region is worked out at
BODY_WORK_FW = 400  # px: neck and body are worked out at this face width (their flaws are broader)
REGION_EDGE = 1024  # px: the neck and body are found on a copy this size
BODY_SPREAD = 4.0  # the colour test's tolerance away from the face (the face uses 3)
SKIN_AREA = 0.004  # share of the long edge skin is voted over
NECK_TOP = (-0.3, -0.2)  # face widths below the chin (negative: above it) where
# the neck starts beside the jaw: higher up are the ears
NECK_DEPTH = (0.45, 0.65)  # ... and where it ends, straight down the photo from
# the chin (a tilted head doesn't tilt the neck): about the collarbones
NECK_HALF_WIDTH = (0.55, 0.75)  # either side of the chin
HEAD_REACH = 1.3  # face widths from the middle of the face, above the jaw: hair,
# ears and scalp, which belong to neither the neck nor the body
HAIR_TEXTURE = (2.0, 3.5)  # fine-detail energy vs the person's own body skin: hair
HAIR_FINE = 1.0  # px on the small copy: the scale hair strands show at
MIN_REGION_FW = 12  # px on the small copy: smaller faces give no colour sample
CLOTHES_SURE = (0.4, 0.7)  # person-parts model's clothes probability: unsure .. sure
DARK_FACE_L = 20.0  # Lab L: a face darker than this is too noisy to take skin colour from
NECK_LINES = ((0.02, 0.035, 0.055), 0.4)  # scales in eye widths, fold kept
# Someone with a beard or stubble on the face almost always has some under the
# jaw too, where it's sparser and in shadow: for them the neck's stubble test
# is far more sensitive. Anyone else keeps the strict one, so necks aren't
# mistaken for stubble and left unretouched.
BEARDED = 0.25  # share of the lower face that's facial hair
NECK_STUBBLE_BEARDED = (0.25, 0.6)


@dataclass
class Person:
    """One person's neck and body skin, on the small copy they were found on."""

    lm: np.ndarray  # landmarks as fractions of width/height
    model: SkinModel
    neck: np.ndarray
    body: np.ndarray


def _colour_match(rg: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """How closely each pixel's chromaticity matches the ``sample`` pixels'."""
    mean = rg[sample].mean(0)
    inv = np.linalg.inv(np.cov(rg[sample].T) + np.eye(2) * 1e-4)
    d = rg - mean
    m2 = np.einsum("...i,ij,...j->...", d, inv, d)
    return np.exp(-0.5 * m2 / BODY_SPREAD**2 * 4).astype(np.float32)


def _lit(lab: np.ndarray, L_low: float) -> np.ndarray:
    """Bright enough to be skin: well below it is hair, nostrils or black cloth."""
    return _smoothstep(lab[..., 0], L_low - DARK_MARGIN, L_low - DARK_MARGIN / 3)


def _vote(prob: np.ndarray, sigma: float) -> np.ndarray:
    """Skin as a solid area: specks that happen to match skin colour (dark
    fabric's noisy colour) are outvoted by their neighbourhood."""
    return np.clip((blur(prob, sigma) - 0.3) / 0.3, 0, 1)


def body_regions(
    rgb: np.ndarray, faces: list[np.ndarray], subject: np.ndarray | None, clothes: np.ndarray | None = None
) -> list[Person]:
    """Each person's neck and body skin (0..1), found on a small copy of the
    photo after the removals.

    Skin away from the face is recognised by the person's own colour, inside
    the subject mask (so a wooden floor or a warm backdrop can't count), as a
    solid area. It's often lit differently from the face, so its colour is
    learnt once more from the skin found so far. Highlights are nearly
    colourless and fail the colour test; they count where skin surrounds them.
    With several people, skin belongs to whoever's chin is nearest.
    ``clothes``: the person-parts model's clothes probability (any size), to
    leave out clothing the colour tests can't tell from skin."""
    h, w = rgb.shape[:2]
    s = min(1.0, REGION_EDGE / max(h, w))
    small = cv2.resize(np.clip(rgb, 0, 1), None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
    sh, sw = small.shape[:2]
    sub = np.ones((sh, sw), np.float32) if subject is None else cv2.resize(
        subject.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA
    )
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)
    rg = _chromaticity(lab)
    # Where the person-parts model is sure it's clothing: a dress close to skin
    # colour passes every colour test, but not this.
    not_clothes = None
    if clothes is not None:
        c = cv2.resize(clothes.astype(np.float32), (sw, sh), interpolation=cv2.INTER_LINEAR)
        not_clothes = 1 - _smoothstep(blur(c, 2.0), *CLOTHES_SURE)
    fine = lab[..., 0] - blur(lab[..., 0], HAIR_FINE)
    texture = np.sqrt(blur(fine * fine, 3 * HAIR_FINE))
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    area = SKIN_AREA * REGION_EDGE

    found = []
    for face in faces:
        lm = face[:, :2] * np.array([sw, sh], np.float32)
        fw = face_width(lm)
        if fw < MIN_REGION_FW:
            continue
        model = SkinModel(lab, lm)
        down = _face_down(lm)
        chin = lm[152]
        v = ((xx - chin[0]) * down[0] + (yy - chin[1]) * down[1]) / fw
        oval = grow_mask(_poly_mask((sh, sw), lm[FACE_OVAL]), 3)
        centre = lm[FACE_OVAL].mean(0)
        near_head = 1 - _smoothstep(np.hypot(xx - centre[0], yy - centre[1]) / fw, HEAD_REACH, HEAD_REACH + 0.15)
        head = np.maximum(oval, near_head * (1 - _smoothstep(v, *NECK_TOP)))
        # Below the jaw the neck follows the body, which is upright far more
        # often than the head is.
        neck_zone = (
            _smoothstep(v, *NECK_TOP)
            * (1 - _smoothstep((yy - chin[1]) / fw, *NECK_DEPTH))
            * (1 - _smoothstep(np.abs(xx - chin[0]) / fw, *NECK_HALF_WIDTH))
            * (1 - oval)
        )
        L_low = model.L_low
        seed = (neck_zone > 0.5) & (sub > 0.5) & (lab[..., 0] > DARK_FACE_L) & (lab[..., 0] < 95)
        if model.L_typical < DARK_FACE_L and seed.sum() > 100:
            # The face is too dark to read its colour (in deep shade, under a
            # hat brim): at that level it's mostly sensor noise. The colour is
            # learnt instead from the lit skin just below the chin.
            L_low = float(np.percentile(lab[..., 0][seed], 5))
            skin = _vote(_colour_match(rg, seed) * _lit(lab, L_low), area)
        else:
            skin = _vote(model.colour_probability(lab, BODY_SPREAD), area)
        found.append((face[:, :2], model, skin, head, neck_zone, np.hypot(xx - chin[0], yy - chin[1]) / fw, L_low))
    if not found:
        return []

    nearest = np.argmin(np.stack([f[5] for f in found]), axis=0)
    all_heads = np.max(np.stack([f[3] for f in found]), axis=0)
    people = []
    for i, (lm_frac, model, skin, _head, neck_zone, _d, L_low) in enumerate(found):
        mine = blur((nearest == i).astype(np.float32), 2.0) if len(found) > 1 else 1.0
        confident = (skin > 0.8) & (all_heads < 0.1) & (sub > 0.5) & (nearest == i)
        if confident.sum() > 200:
            skin = np.maximum(skin, _vote(_colour_match(rg, confident) * _lit(lab, L_low), area))
        # Hair falling past the jaw can be skin-coloured, but it's full of fine
        # strands where skin is smooth at this size: judged against this
        # person's own body skin, over an area, so an edge or a necklace line
        # alone doesn't count.
        sure = (skin > 0.8) & (all_heads < 0.1) & (sub > 0.5) & (nearest == i)
        if sure.sum() > 200:
            hair = _smoothstep(texture / (float(np.median(texture[sure])) + 1e-3), *HAIR_TEXTURE)
            skin = skin * (1 - _vote(hair, area))
        bright = _smoothstep(lab[..., 0], model.L_typical + HIGHLIGHT[0], model.L_typical + HIGHLIGHT[1])
        skin = np.maximum(skin, bright * _smoothstep(blur(skin, 2 * area), 0.6, 0.8))
        skin = skin * sub * (1 - all_heads) * mine
        if not_clothes is not None:
            skin = skin * not_clothes
        people.append(Person(lm_frac, model, (skin * neck_zone).astype(np.float32),
                             (skin * (1 - neck_zone)).astype(np.float32)))
    return people


def _bearded(lab: np.ndarray, lm: np.ndarray, model: SkinModel) -> bool:
    """Whether the person has facial hair along the jaw and chin."""
    lower = _poly_mask(lab.shape[:2], lm[FACE_OVAL]) * _smoothstep(
        (np.mgrid[0 : lab.shape[0], 0 : lab.shape[1]][0] - lm[1][1]) / face_width(lm), 0.0, 0.05
    )  # below the nose tip
    if lower.sum() < 1:
        return False
    return float((stubble(lab, lower, lm, model) * lower).sum() / lower.sum()) > BEARDED


def _neck_lines(amount: float):
    def fill(lab, W, wfw):
        eye_w = 0.2 * wfw
        zone = _inner_feather((W > 0.5).astype(np.float32), 0.02 * wfw) * W
        L = np.ascontiguousarray(lab[..., 0])
        _relight(lab, amount * _wrinkle_lift(L, zone, eye_w, *NECK_LINES), eye_w)

    return fill


def _added_box(edits: list[dict], shape: tuple[int, int]):
    """Bounding box (x0, y0, x1, y1, full-image pixels) of the "add" strokes
    in ``edits``, or None."""
    h, w = shape
    boxes = []
    for e in edits:
        if e["mode"] != "add":
            continue
        pts = np.array(e["points"], np.float32) * [w, h]
        r = e["radius"] * max(h, w) * 1.5
        boxes.append((*(pts.min(0) - r), *(pts.max(0) + r)))
    if not boxes:
        return None
    b = np.array(boxes)
    return np.array([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()])


def apply_body(
    rgb: np.ndarray,
    people: list[Person],
    neck_params: dict | None,
    body_params: dict | None,
    edits: dict[str, list[dict]] | None = None,
) -> np.ndarray:
    """Neck and body skin tools for the people found by ``body_regions``, at
    whatever resolution ``rgb`` is. ``edits``: the user's corrections to the
    neck and body areas, by region (see region_edit)."""
    out = rgb.copy()
    h, w = rgb.shape[:2]
    for name, p in (("neck", RegionParams.from_dict(neck_params)), ("body", RegionParams.from_dict(body_params))):
        if p.is_noop():
            continue
        for person in people:
            small_w = getattr(person, name)
            region_edits = (edits or {}).get(name, [])
            lm = person.lm * np.array([w, h], np.float32)
            fw = face_width(lm)
            ys, xs = np.nonzero(small_w > 0.02)
            added = _added_box(region_edits, (h, w))
            if fw < 40 or (ys.size == 0 and added is None):
                continue
            sy, sx = h / small_w.shape[0], w / small_w.shape[1]
            pad = 0.15 * fw
            if ys.size:
                lo = np.array([xs.min() * sx, ys.min() * sy]) - pad
                hi = np.array([(xs.max() + 1) * sx, (ys.max() + 1) * sy]) + pad
            else:
                lo, hi = np.array(added[:2], float), np.array(added[2:], float)
            if added is not None:  # skin the user painted in, outside what was found
                lo, hi = np.minimum(lo, added[:2]), np.maximum(hi, added[2:])
            if name == "neck":  # the stubble test compares with the cheeks
                cheeks = lm[list(CHEEK_SAMPLES)]
                lo = np.minimum(lo, cheeks.min(0) - CHEEK_RADIUS * fw)
                hi = np.maximum(hi, cheeks.max(0) + CHEEK_RADIUS * fw)
            x0, y0 = np.maximum(np.floor(lo), 0).astype(int)
            x1, y1 = np.minimum(np.ceil(hi), [w, h]).astype(int)

            def weights(work, scale, small_w=small_w, lm=lm, origin=(x0, y0), person=person, name=name, p=p,
                        box=(x0, y0, x1, y1), region_edits=region_edits):
                # The region's weight, cut from the small copy to this crop.
                gx = (origin[0] + (np.arange(work.shape[1], dtype=np.float32) + 0.5) / scale[0]) / sx - 0.5
                gy = (origin[1] + (np.arange(work.shape[0], dtype=np.float32) + 0.5) / scale[1]) / sy - 0.5
                mx, my = (m.astype(np.float32) for m in np.meshgrid(gx, gy))
                W = cv2.remap(small_w, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                wlm = (lm - origin) * scale
                if name == "neck":  # a beard runs on under the jaw
                    W = W * (1 - stubble(work, W, wlm, person.model, relative=True,
                                         texture=NECK_STUBBLE_BEARDED if _bearded(work, wlm, person.model)
                                         else STUBBLE_TEXTURE))
                W = blur(W, max(0.7, 0.004 * face_width(wlm)))
                W = region_edit.apply(W, region_edits, (h, w), box)
                # Even tone evens this region toward its own average colour, not
                # the face's: a face is often warmer (make-up, more sun), and
                # pulling a chest toward it turns the whole chest orange.
                model = copy.copy(person.model)
                if W.sum() > 1:
                    model.rg_mean = (_chromaticity(work) * W[..., None]).sum((0, 1)) / W.sum()
                extra = _neck_lines(p.neck_lines) if name == "neck" and p.neck_lines > 0 else None
                return W, model, None, extra, None

            _retouch(out, (x0, y0, x1, y1), fw, p, weights, BODY_WORK_FW)
    return out

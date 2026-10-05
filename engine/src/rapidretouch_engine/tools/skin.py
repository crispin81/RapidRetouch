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
from .. import interrupt
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


def forehead_outline(lm: np.ndarray, reach: float | None = None) -> np.ndarray:
    """The face outline with its top raised by ``reach`` face widths
    (HAIRLINE_REACH if None), toward the hairline (the landmarks' outline
    stops partway up the forehead)."""
    up = -_face_down(lm) * (HAIRLINE_REACH if reach is None else reach) * face_width(lm)
    top = set(FOREHEAD_ARC)
    return np.array([lm[i] + up if i in top else lm[i] for i in FACE_OVAL], np.float32)


def profile_outline(lm: np.ndarray, reach: float | None = None) -> np.ndarray:
    """``forehead_outline`` widened to take in every landmark. On a side
    profile the outline's far side runs down the cheek, inside the face, and
    left out the nose, upper lip and front of the chin (P1024004); the hull
    reaches round them. It also takes in background (in front of the lips and
    chin), so only colour-tested maps may use it. Frontal faces: the same."""
    return cv2.convexHull(np.vstack([forehead_outline(lm, reach), lm[:, :2]]).astype(np.float32))[:, 0]


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
# The skin map reaches further, to the hairline on a high forehead: the colour
# test and the hair map stop it at the hair (0.12 left the top of the forehead
# out on P1167822 and P1256049; 0.3 picked up specks along a hat brim).
SKIN_FOREHEAD_REACH = 0.22
FROWN_HEIGHT = 0.1
NOSE_WINGS = (129, 358)  # beside each nostril
# The nose (MediaPipe face mesh): bridge, tip, base and the sides down to the
# wings, as a convex area, taken in by the skin map (see _face_weights).
NOSE_AREA = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 129, 358, 49, 279, 64, 294,
             102, 331, 114, 343, 122, 351, 188, 412, 217, 437, 198, 420]
NOSE_FEATHER = 0.02  # face widths: the nose area fades in over this
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
# Blemishes (see _spots): calibrated on real skin spots, which score ~0.5-1
# (ear folds, stubble edges and hair score far higher, but the skin mask
# excludes them anyway). Large pores score in this range too, which is what
# Blemishes is for alongside Acne.
BLEMISH_DARK = (0.5, 1.2)  # spot strength ramp, Lab L
BLEMISH_RED = (0.35, 0.9)  # spot strength ramp, Lab a
# Acne (see acne_spots and _heal_acne). Sizes in face widths; scores in units
# of the skin's own normal variation around each place.
ACNE_AROUND = 0.06  # the "skin around" a spot is taken over this
ACNE_FINE = 0.0015  # finer than this (grain) doesn't count toward a spot
ACNE_VARIATION = 0.08  # the skin's normal variation is judged over this
ACNE_CLEAREST, ACNE_FAINTEST = 7.0, 3.0  # slider 0 .. 1: how far a spot must stand out
ACNE_SIZE = (0.004, 0.04)  # spot diameters taken
ACNE_EDGE = 15.0  # Lab L across 0.04 face widths of the spot-free skin: a shading edge
# (ordinary facial shading ~6, nose / mouth-corner shadow lines 20+)
ACNE_GROW = 0.004  # healed area reaches this far past each spot
ACNE_TEXTURE = 0.003  # finer than this is texture, kept on top of the rebuilt tone
ACNE_RING = 0.01  # face widths around a candidate that must be skin...
ACNE_SURROUNDED = 0.8  # ...this share of it at least
HAIR_SURE = (0.4, 0.7)  # person-parts model's hair probability: unsure .. sure
# Shadowed skin (under the cheekbones and chin) fails the colour test: it's lit
# by bounce or fill light of another colour, and noisier. Where the person-parts
# model is sure it's face skin, that stands in for the colour test (DSC_2409's
# chin), down to this much darker than the sampled skin (nostrils stay out).
AI_SKIN_SURE = (0.5, 0.8)
AI_SKIN_DARK = 2.0  # x DARK_MARGIN
AI_SKIN_BEARD = 0.04  # face widths: how far from found stubble the colour test still decides
ACNE_RIM = 0.5  # share of a spot's own crisp rim taken out of its texture
TEXTURE_KEEP = 0.003  # face widths: finer than this is texture, always kept
# Moles are kept (part of someone's identity; the Remove brush takes one out):
# clearly darker than the skin around them, but not mainly redder, which is
# what marks a blemish.
MOLE_DARK = (2.0, 3.5)  # round-spot strength (Lab L): well above a typical blemish
MOLE_BROWN = (0.35, 0.7)  # redness per unit of darkness above this is a blemish
TEXTURE_BAND = (0.0015, 0.015)  # face widths: pores; finer is grain, coarser is Smooth's
TEXTURE_KNEE = 1.5  # Lab L: pores deeper than this are reduced the most
TEXTURE_MAX = 1.0
# Hair strands over the skin, kept by Texture: widths (face widths) looked at,
# how line-like they must be (along / across curvature below STRAND_ROUND), and
# how clear (scale-normalised Lab L): a faint strand .. a clear one.
STRAND_SCALES = (0.0012, 0.0025)
STRAND_ROUND = 0.35
STRAND_STRENGTH = (1.0, 2.5)
# Pores: the pore band taken out, up to PORES_MAX of it, dark pits and bright
# bumps alike (Texture spares bright detail, which left raised pores in a
# highlight untouched). Only the sub-pixel grain is kept (on DSC_2376-2 at
# 1385 px face width, the pores and the bumpy "orange peel" between them run
# from ~1 to ~20 px); wider than this flattens fine lines and looks airbrushed.
PORES_BAND = (0.0007, 0.015)  # face widths
PORES_MAX = 0.9
PORES_EDGE = (10.0, 20.0)  # Lab L: detail standing out this much is an edge, kept
PORES_SMOOTH = 0.002  # face widths: the median's rounding smoothed away
STRANDS_NEAR_HAIR = 0.08  # face widths from the hair where Pores keeps single strands
TEXTURE_CHIN_BOOST = 0.25  # Texture this much stronger on the chin
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
    acne: float = 0.0  # spots and acne healed (see acne_spots)
    blemishes: float = 0.0  # large pores and small marks evened out (see _spots)
    smooth: float = 0.0
    even: float = 0.0
    shine: float = 0.0  # -1 matte .. 0 natural .. +1 gloss
    texture: float = 0.0  # softens pores; the finest grain is kept
    pores: float = 0.0  # takes pores out: pits and raised bumps alike, highlights too
    forehead_lines: float = 0.0  # the wrinkle sliders exist on the Face tab only
    frown_lines: float = 0.0
    smile_lines: float = 0.0
    chin_lines: float = 0.0
    neck_lines: float = 0.0  # Neck tab only

    @classmethod
    def from_dict(cls, d: dict | None) -> "RegionParams":
        d = dict(d or {})
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_noop(self) -> bool:
        return self.shine == 0 and all(
            getattr(self, k) <= 0
            for k in ("acne", "blemishes", "smooth", "even", "texture", "pores", "neck_lines", *WRINKLE_SLIDERS)
        )


def _blobs(channel: np.ndarray, fw: float, scales=None) -> np.ndarray:
    """Round-spot strength (units of ``channel``) where it dips: both principal
    curvatures positive and similar, so lines and wrinkles don't count.
    ``scales``: spot sizes looked for (face widths), BLEMISH_SCALES if None."""
    out = np.zeros_like(channel)
    for scale in BLEMISH_SCALES if scales is None else scales:
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


def _local_median(x: np.ndarray, k: int) -> np.ndarray:
    """Median over a k x k window (k odd): the skin as it would be without
    small spots, which a mean would be pulled toward. Worked out on a
    quarter-size copy for large windows (medians of big windows are slow)."""
    h, w = x.shape
    lo, hi = float(np.percentile(x, 0.5)), float(np.percentile(x, 99.5))
    u8 = lambda v: np.clip((v - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)  # noqa: E731
    back = lambda m: m.astype(np.float32) / 255 * (hi - lo) + lo  # noqa: E731
    if k <= 9:
        return back(cv2.medianBlur(u8(x), k))
    small = cv2.resize(x, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    ks = max(3, (k // 4) | 1)
    return cv2.resize(back(cv2.medianBlur(u8(small), ks)), (w, h), interpolation=cv2.INTER_LINEAR)


def acne_spots(lab: np.ndarray, W: np.ndarray, fw: float, amount: float, region: np.ndarray | None = None) -> np.ndarray:
    """Where the spots are (0..1, each whole spot grown a little), as taken at
    slider ``amount``: low takes only the clearest, high fainter ones too.

    A spot is skin darker or redder than the skin around it (a local median,
    which spots don't pull), by more than that skin's own normal variation, so
    one setting suits smooth and textured skin, light and shade. Each
    candidate must then be a compact, round, isolated blob of spot size, on
    skin, off any shading edge, and not a mole (those are kept)."""
    L = np.ascontiguousarray(lab[..., 0])
    a = np.ascontiguousarray(lab[..., 1])
    k = 2 * max(2, round(ACNE_AROUND * fw / 2)) + 1
    fine = max(0.6, ACNE_FINE * fw)
    L_around = _local_median(L, k)
    darker = L_around - blur(L, fine)
    redder = blur(a, fine) - _local_median(a, k)
    # The skin's own normal variation, at the same scale, around each place.
    area = max(1.0, ACNE_VARIATION * fw)
    # Skin with its small holes filled: a red spot fails the skin colour test
    # itself, and must still count as on the skin. Holes up to the largest
    # spot size are filled; the lips, eyes and hair are far bigger and stay out.
    kf = 2 * max(1, round(ACNE_SIZE[1] * fw / 2)) + 1
    skin_u8 = cv2.morphologyEx((W > 0.5).astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kf, kf)))
    skin = skin_u8.astype(np.float32)
    spread = lambda d: np.sqrt(masked_blur(d * d, skin, area)) + 1e-3  # noqa: E731
    score = np.maximum(np.clip(darker, 0, None) / spread(darker), np.clip(redder, 0, None) / spread(redder))
    score = score * skin
    # A spot sitting on a strong shading edge (a shadow line) is lighting.
    # Judged from the skin around, without the spots: a spot's own darkness
    # would otherwise make every spot look like an edge.
    broad = blur(L_around, max(0.7, 0.02 * fw))
    gy, gx = np.gradient(broad)
    on_edge = np.hypot(gx, gy) * 0.04 * fw > ACNE_EDGE
    moles = _moles(L, a, fw) > 0.5

    # Candidates at the most sensitive setting, then judged one by one.
    n, labels, stats, _ = cv2.connectedComponentsWithStats((score > ACNE_FAINTEST).astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(L.shape, np.float32)
    peak = np.zeros(n, np.float32)
    np.maximum.at(peak, labels.ravel(), score.ravel())
    px = max(1.0, fw)
    area_px = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    d = np.sqrt(area_px * 4 / np.pi) / px  # equivalent diameter, face widths
    fill = area_px / np.maximum(bw * bh, 1)  # a round blob fills ~0.78 of its box
    aspect = np.maximum(bw, bh) / np.maximum(np.minimum(bw, bh), 1)
    # The slider: 0 takes only the clearest spots, 1 everything above ACNE_FAINTEST.
    need = ACNE_CLEAREST + (ACNE_FAINTEST - ACNE_CLEAREST) * float(np.clip(amount, 0, 1))
    ok = (peak >= need) & (d >= ACNE_SIZE[0]) & (d <= ACNE_SIZE[1]) & (fill >= 0.4) & (aspect <= 2.5)
    ok[0] = False
    bad = np.zeros(n, np.float32)
    off_region = np.zeros(L.shape, bool) if region is None else region < 0.5  # e.g. hair over the skin
    np.maximum.at(bad, labels.ravel(), (on_edge | moles | off_region).astype(np.float32).ravel())
    ok &= bad < 0.5
    # A spot is surrounded by skin. Next to an eye, a nostril or hair, the
    # ring around a candidate is partly not skin: a feature's edge, not a spot.
    # Skin by the colour test, plus the reddened skin around an inflamed spot,
    # which fails that test: within the face region (``region``: the face
    # minus eyes, brows, lips, facial hair and anything too dark) and redder
    # than the skin around. Blonde hair over the skin is never redder.
    halo = ((W if region is None else region) > 0.5) & (redder > 0)
    real_skin = ((W > 0.5) | halo) & (_strands(L, fw) < 0.5)
    ring_px = max(2, round(ACNE_RING * fw))
    for i in np.nonzero(ok)[0]:
        x, y, bw, bh = stats[i, :4]
        x0, y0 = max(0, x - ring_px), max(0, y - ring_px)
        x1, y1 = min(L.shape[1], x + bw + ring_px), min(L.shape[0], y + bh + ring_px)
        blob = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = cv2.dilate(blob, np.ones((2 * ring_px + 1,) * 2, np.uint8)) & (1 - blob)
        if ring.sum() and real_skin[y0:y1, x0:x1][ring > 0].mean() < ACNE_SURROUNDED:
            ok[i] = False
    spot = ok[labels].astype(np.uint8)
    grow = 2 * max(1, round(ACNE_GROW * fw)) + 1
    spot = cv2.dilate(spot, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    return np.clip(blur(spot.astype(np.float32), max(0.5, 0.5 * ACNE_GROW * fw)) * 1.3, 0, 1)


def _heal_acne(lab, W, fw, amount, region=None, spot=None):
    """Heal spots, in place, the way a retoucher does with frequency
    separation: the skin is split into its broad tone and its fine texture
    (pores, grain), the tone under each spot is rebuilt from the clean skin
    around it, and the texture is put back on top, so a healed spot is gone
    rather than faded, and isn't a smooth patch.

    (Repainting spots with LaMa was tried and removed: it painted new freckles
    into freckled skin and left more redness, and took 1-2 s per slider move.)"""
    if spot is None:
        spot = acne_spots(lab, W, fw, amount, region)
    if spot.max() <= 0:
        return
    split = max(0.6, ACNE_TEXTURE * fw)
    tone = blur(lab, split)
    texture = lab - tone
    hole = (spot > 0.05).astype(np.uint8)
    rebuilt = np.stack(
        [cv2.inpaint(np.ascontiguousarray(tone[..., i]), hole, max(1, round(ACNE_GROW * fw)), cv2.INPAINT_TELEA)
         for i in range(3)],
        axis=-1,
    )
    # The texture inside a spot also carries the spot's own crisp rim: tamed,
    # not removed, so the pores stay.
    texture = texture * (1 - ACNE_RIM * spot)[..., None]
    lab += spot[..., None] * (rebuilt + texture - lab)


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
    """Blemishes: heal small spots and large pores, in place. Each takes the
    colour and tone of the skin around it, and only its broad colour and tone
    change, so the finest texture stays and a healed spot isn't a smooth patch.
    (The tool Acne replaced; brought back alongside it for skin with strong
    pores and many small marks.)"""
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


def _strands(L: np.ndarray, fw: float) -> np.ndarray:
    """Where single hairs lie over the skin (0..1): thin lines, dark or bright
    (blonde hair against shadow), strongly longer than they're wide, so round
    pores and blemishes don't count."""
    best = np.zeros_like(L)
    for scale in STRAND_SCALES:
        sigma = max(0.6, scale * fw)
        g = cv2.GaussianBlur(L, (0, 0), sigma)
        gy, gx = np.gradient(g)
        gxy, gxx = np.gradient(gx)
        gyy, _ = np.gradient(gy)
        half = (gxx + gyy) / 2
        root = np.sqrt(((gxx - gyy) / 2) ** 2 + gxy**2)
        l1, l2 = half + root, half - root  # most and least curved
        across = np.maximum(np.abs(l1), np.abs(l2))
        along = np.minimum(np.abs(l1), np.abs(l2))
        line = sigma**2 * across * np.clip(1 - along / np.maximum(across, 1e-6) / STRAND_ROUND, 0, 1)
        best = np.maximum(best, line)
    strand = _smoothstep(best, *STRAND_STRENGTH)
    k = 2 * max(1, round(STRAND_SCALES[-1] * fw)) + 1
    return np.clip(blur(max_filter(strand, k), max(0.6, STRAND_SCALES[0] * fw)) * 1.5, 0, 1).astype(np.float32)


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
    # Hair falling over the skin (a fringe on the forehead) is the same size
    # as pores, and the skin mask, worked out on a smaller copy, can't see
    # single strands: they're found as thin lines (pores are round) and kept.
    reduce = reduce * (1 - _strands(L, fw))
    lab[..., 0] = L - W * reduce * pores


def _median_skin(L: np.ndarray, fw: float) -> np.ndarray:
    """The skin around each place without its pores and bumps: a median over
    PORES_BAND[1], which ignores small bumps however contrasty (crepey skin
    beside the nose) yet keeps real edges (the nose's outline, the smile
    crease), where a blur made halos and the edge-aware filter took the
    contrasty bumps for edges. Worked at half size, in 8 bits (OpenCV's large
    medians need them), then smoothed past the rounding."""
    small = cv2.resize(L, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    k = min(255, 2 * max(1, round(PORES_BAND[1] * fw * 0.5)) + 1)
    med = cv2.medianBlur(np.clip(small * 2.55, 0, 255).astype(np.uint8), k).astype(np.float32) / 2.55
    med = cv2.resize(med, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_LINEAR)
    return blur(med, max(1.0, PORES_SMOOTH * fw))


def _pores(lab: np.ndarray, W: np.ndarray, fw: float, amount: float, near_hair: np.ndarray | None = None) -> None:
    """Take pores out, in place: the pore band of lightness flattened toward
    the skin around it, pits and raised bumps alike (the bumps are what catch
    the light on textured skin, so highlights aren't spared as Texture spares
    them). The finest grain is kept, and so are single hairs near the hair
    (``near_hair``; everywhere if None). Elsewhere crepey skin, whose fine
    lines look like hairs, is smoothed too."""
    L = np.ascontiguousarray(lab[..., 0])
    grain = blur(L, max(0.5, PORES_BAND[0] * fw))
    detail = grain - _median_skin(L, fw)
    # Whatever stands out far more than a pore is an edge or a feature.
    keep = _smoothstep(np.abs(detail), *PORES_EDGE)
    strands = _strands(L, fw) if near_hair is None else _strands(L, fw) * near_hair
    reduce = amount * PORES_MAX * (1 - strands) * (1 - keep)
    lab[..., 0] = L - W * reduce * detail


def _apply_region(lab, W, fw, p: RegionParams, model, lm=None, W_even=None, spots=None):
    """``W_even``: where Even tone works, if not ``W`` (see ``apply``).
    ``spots``: Acne's spots found at full detail, if given (see acne_masks)."""
    eye_w = 0.2 * fw  # the eye tools' colour-matching scale, in this face's units
    if lm is not None:  # wrinkles first, while the fine detail is as shot
        _smooth_lines(lab, W, lm, fw, p, eye_w, W_even)
        interrupt.check()
    if p.acne > 0:
        # Over the colour-tested skin, not the whole face region as Even tone
        # is: over the region it caught mouth-corner shadows and hair.
        _heal_acne(lab, W, fw, p.acne, W_even, spots)
        interrupt.check()
    if p.blemishes > 0:
        _heal_blemishes(lab, W, fw, p.blemishes)
        interrupt.check()
    if p.even > 0:
        _even_tone(lab, W if W_even is None else W_even, fw, p.even, model)
        interrupt.check()
    if p.smooth > 0:
        _smooth(lab, W, fw, p.smooth, eye_w)
        interrupt.check()
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
    W, model, wlm, extra, W_even, *more = weights(work, scale)
    interrupt.check()
    maps = more[0] if more else {}
    _apply_region(work, W, wfw, p, model, wlm, W_even, maps.get("spots"))
    if extra is not None:
        extra(work, W, wfw)
    if s == 1:
        lab = work
    else:
        delta = cv2.resize(work - before, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_CUBIC)
        lab += delta
    interrupt.check()
    if p.texture > 0:
        # Pores are too small to work out on the smaller copy: this one runs
        # at the image's own resolution.
        W_tex = W if maps.get("texture_gain") is None else W * maps["texture_gain"]
        W_full = W_tex if s == 1 else cv2.resize(W_tex, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_LINEAR)
        _texture(lab, W_full, fw, p.texture)
        interrupt.check()
    if p.pores > 0:
        # Over the whole face region (as Even tone), not the colour-tested skin:
        # the sides of the nose and the smile crease fail the colour test (red,
        # shiny or shadowed), and their pores were left at half strength or less.
        W_pores = W if W_even is None else W_even
        size = (lab.shape[1], lab.shape[0])
        W_full = W_pores if s == 1 else cv2.resize(W_pores, size, interpolation=cv2.INTER_LINEAR)
        near = maps.get("near_hair")
        if near is not None and s != 1:
            near = cv2.resize(near, size, interpolation=cv2.INTER_LINEAR)
        _pores(lab, W_full, fw, p.pores, near)
    out[y0:y1, x0:x1] = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def _face_boxes(shape: tuple[int, int], faces: list[np.ndarray]):
    """For each face big enough to retouch: landmarks in pixels, face width,
    and the crop box (x0, y0, x1, y1) the face tools work in."""
    h, w = shape
    for face in faces:
        lm = face[:, :2] * np.array([w, h], np.float32)
        fw = face_width(lm)
        if fw < 40:  # too small to retouch meaningfully
            continue
        pts = profile_outline(lm, SKIN_FOREHEAD_REACH)  # the face oval, up to the hairline
        x0, y0 = np.maximum(np.floor(pts.min(0) - 0.15 * fw), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(pts.max(0) + 0.15 * fw), [w, h]).astype(int)
        yield lm, fw, (int(x0), int(y0), int(x1), int(y1))


def _map_into(m: np.ndarray, frac_box, box, full_shape, out_shape) -> np.ndarray:
    """Resample ``m``, which covers ``frac_box`` (fractions of the image) at
    any size, onto ``box`` (pixels of an image of ``full_shape``) at
    ``out_shape``. Outside ``frac_box`` is 0."""
    h, w = full_shape
    fx0, fy0, fx1, fy1 = frac_box
    mh, mw = m.shape
    oy, ox = out_shape
    xs = ((box[0] + (np.arange(ox, dtype=np.float32) + 0.5) * (box[2] - box[0]) / ox) / w - fx0) / (fx1 - fx0) * mw - 0.5
    ys = ((box[1] + (np.arange(oy, dtype=np.float32) + 0.5) * (box[3] - box[1]) / oy) / h - fy0) / (fy1 - fy0) * mh - 0.5
    mx, my = (g.astype(np.float32) for g in np.meshgrid(xs, ys))
    return cv2.remap(m.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _face_weights(work, scale, lm, box, full_shape, edits, head_hair, ai_skin=None):
    """The face tools' skin weights on a work crop: W (colour-tested skin, no
    facial hair), W_even (the face region, for Even tone, placed wrinkle
    areas and Acne's surroundings), the person's SkinModel and the landmarks
    in the crop's pixels."""
    h, w = full_shape
    wlm = (lm - box[:2]) * scale
    outline = forehead_outline(wlm, SKIN_FOREHEAD_REACH)
    skin_w, hair, model = face_skin(work, wlm, outline=profile_outline(wlm, SKIN_FOREHEAD_REACH))
    soft = max(0.7, 0.004 * face_width(wlm))
    if ai_skin is not None:
        sure = _smoothstep(_map_into(ai_skin, (0, 0, 1, 1), box, full_shape, work.shape[:2]), *AI_SKIN_SURE)
        dark = DARK_MARGIN * AI_SKIN_DARK
        shaded = _smoothstep(work[..., 0], model.L_low - dark, model.L_low - dark / 3)
        region = face_region(work.shape[:2], wlm, outline=profile_outline(wlm, SKIN_FOREHEAD_REACH))
        # The model calls a beard face skin, and the stubble test is weaker
        # in shadow: near the stubble it finds, the colour test still decides
        # (beard on the shadow side of P1167822's face came in otherwise).
        near_beard = np.clip(blur(hair, max(0.7, AI_SKIN_BEARD * face_width(wlm))) * 3, 0, 1)
        skin_w = np.maximum(skin_w, region * sure * shaded * (1 - near_beard))
    W = blur(skin_w * (1 - hair), soft)
    # Even tone works on the whole face (not eyes, brows, lips, facial
    # hair or anything too dark to be skin) without the colour test: a
    # red nose fails that test for being red, and was left pink.
    W_even = np.maximum(W, blur(face_region(work.shape[:2], wlm, outline=outline) * (1 - hair) * _lit(work, model.L_low), soft))
    if head_hair is not None:
        # Hair over the skin, by the person-parts model: blonde strands
        # pass the colour test and are too soft to read as strands.
        # Brown or blonde hair passes the colour test too: with the outline
        # raised to the hairline, the skin map needs this as much as W_even.
        on_hair = _smoothstep(_map_into(head_hair, (0, 0, 1, 1), box, full_shape, work.shape[:2]), *HAIR_SURE)
        W = W * (1 - on_hair)
        W_even = W_even * (1 - on_hair)
    # The nose is skin, but often fails the colour test for being redder,
    # shinier or shaded down its sides (half strength there on DSC_2376-2):
    # within the nose the tools use the face region instead. Dark nostrils
    # and hair are still out, and each tool keeps its own edge protection.
    nose = _inner_feather(
        _poly_mask(work.shape[:2], cv2.convexHull(wlm[NOSE_AREA].astype(np.float32))[:, 0]),
        NOSE_FEATHER * face_width(wlm),
    )
    # (Its own map, not W_even's: on a side profile the nose is past the outline.)
    nose = nose * _lit(work, model.L_low) * (1 - hair)
    if head_hair is not None:
        nose = nose * (1 - on_hair)
    W = np.maximum(W, nose)
    W_even = np.maximum(W_even, nose)
    W = region_edit.apply(W, edits or [], (h, w), box)
    W_even = region_edit.apply(W_even, edits or [], (h, w), box)
    return W, W_even, model, wlm


def acne_masks(rgb, faces, amount, edits=None, head_hair=None, ai_skin=None) -> list[tuple]:
    """Each face's Acne spot map, found at full detail: (frac_box, map),
    ``frac_box`` the face's crop as fractions of the image. The preview is too
    small to see small spots (a face can be 350 px wide there), so the preview
    heals with these, found once from the full-resolution photo; otherwise the
    preview and the sharp zoomed-in view disagree."""
    h, w = rgb.shape[:2]
    out = []
    for lm, fw, box in _face_boxes((h, w), faces):
        x0, y0, x1, y1 = box
        lab = cv2.cvtColor(np.clip(rgb[y0:y1, x0:x1], 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
        s = min(1.0, WORK_FW / fw, WORK_EDGE / max(lab.shape[:2]))
        work = lab if s == 1 else cv2.resize(lab, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        scale = np.array([work.shape[1] / lab.shape[1], work.shape[0] / lab.shape[0]], np.float32)
        W, W_even, _model, _wlm = _face_weights(work, scale, lm, box, (h, w), edits, head_hair, ai_skin)
        spots = acne_spots(work, W, fw * scale[0], amount, W_even)
        out.append(((x0 / w, y0 / h, x1 / w, y1 / h), spots))
    return out


def apply(
    rgb: np.ndarray,
    faces: list[np.ndarray],
    face_params: dict | None,
    edits: list[dict] | None = None,
    head_hair: np.ndarray | None = None,
    acne: list[tuple] | None = None,
    ai_skin: np.ndarray | None = None,
) -> np.ndarray:
    """Face skin tools on every face. rgb float32 HxWx3 0..1; faces are
    landmark arrays as fractions of width/height. ``edits``: the user's
    corrections to the face skin area (see region_edit). ``head_hair``: the
    person-parts model's hair probability (any size, the photo's shape), to
    keep Acne and Even tone off hair over the skin (a fringe). ``acne``: spot
    maps found at full detail (see acne_masks), used instead of finding
    spots here. ``ai_skin``: the person-parts model's face skin (as
    ``head_hair``), taken as skin where the colour test misses shadows.
    Returns a new rgb."""
    p = RegionParams.from_dict(face_params)
    out = rgb.copy()
    if p.is_noop():
        return out
    h, w = rgb.shape[:2]
    for lm, fw, box in _face_boxes((h, w), faces):

        def weights(work, scale, lm=lm, box=box):
            W, W_even, model, wlm = _face_weights(work, scale, lm, box, (h, w), edits, head_hair, ai_skin)
            spots = None
            if acne and p.acne > 0:
                spots = np.zeros(work.shape[:2], np.float32)
                for frac_box, m in acne:
                    spots = np.maximum(spots, _map_into(m, frac_box, box, (h, w), work.shape[:2]))
            # Texture is a little stronger on the chin, where pores and
            # bumps are most noticeable.
            chin = _inner_feather(wrinkle_zones(work.shape[:2], wlm)["chin_lines"], 0.03 * face_width(wlm))
            # Pores keeps single hairs only near the hair (see _pores).
            near_hair = None
            if head_hair is not None:
                on_hair = _smoothstep(_map_into(head_hair, (0, 0, 1, 1), box, (h, w), work.shape[:2]), *HAIR_SURE)
                near_hair = np.clip(blur(on_hair, max(0.7, STRANDS_NEAR_HAIR * face_width(wlm))) * 4, 0, 1)
            return W, model, wlm, None, W_even, {
                "spots": spots,
                "texture_gain": 1 + TEXTURE_CHIN_BOOST * chin,
                "near_hair": near_hair,
            }

        _retouch(out, box, fw, p, weights)
    return out


def face_area(
    rgb: np.ndarray,
    faces: list[np.ndarray],
    edits: list[dict] | None = None,
    head_hair: np.ndarray | None = None,
    ai_skin: np.ndarray | None = None,
) -> np.ndarray:
    """Where the Face tools work (0..1, the image's size): each face's skin as
    the tools see it (``_face_weights``, hair kept out), with the user's
    corrections applied. Shown blue by the Refine area brush."""
    h, w = rgb.shape[:2]
    area = np.zeros((h, w), np.float32)
    for lm, _fw, box in _face_boxes((h, w), faces):
        x0, y0, x1, y1 = box
        lab = cv2.cvtColor(np.clip(rgb[y0:y1, x0:x1], 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
        W, _W_even, _model, _wlm = _face_weights(lab, 1.0, lm, box, (h, w), edits, head_hair, ai_skin)
        area[y0:y1, x0:x1] = np.maximum(area[y0:y1, x0:x1], W)
    return area


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

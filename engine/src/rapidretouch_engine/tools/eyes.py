"""Under-eye retouching: dark circles and eye bags.

Both work per eye, located by face landmarks, with every distance measured in
eye widths so a headshot and a full-length group behave the same, and so do the
preview and the full-resolution export.

Dark circles: the under-eye skin's broad tone is moved toward what the skin
around it (cheek below, the sides) says it should be, interpolated across the
zone. Interpolating from the surroundings rather than sampling one cheek patch
keeps split-lit faces right: each eye is corrected toward its own lighting.
Brightness is only ever raised; colour moves toward the surrounding skin (the
purple/blue cast). Only the low-frequency tone changes, so texture is kept.

Wrinkles: fine lines under the eye and crow's feet. Lines are about the size of
pores, so smoothing that detail band would erase skin texture and look plastic.
Instead a Hessian (curvature) filter picks out long, thin dark valleys — lines
respond strongly, round pores barely at all — and only those are lifted toward
the skin around them (a morphological closing: the skin as if the line weren't
there). Anything much darker than a wrinkle (lashes, brow tails, stray hairs)
is left alone, and the zones are clipped to the face outline.

The eyes themselves (whites, iris, catchlights) use MediaPipe's iris landmarks
(centre + 4 edge points): the eye opening comes from the lid contour, the iris is
a circle clipped by the lids, the pupil its dark core, the white the rest.

Eye bags: frequency-separated dodge and burn. Light and shade at "bag" scale
(bigger than pores and fine lines, smaller than the face's lighting) is
flattened — the crease shadow lifted more than the bulge's highlight is toned
down — so fine texture is untouched. The crease line itself (tear trough /
lid-cheek junction) is narrower than that band, so the same slider also lifts
it with the wrinkle line filler, tuned wider and allowed to go further.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .colour import linear_to_srgb, srgb_to_linear
from .filters import blur, grow_mask, masked_blur, max_filter

# MediaPipe Face Mesh indices, each curve listed outer corner -> inner corner.
EYES = {
    "right": {
        "corners": (33, 133),
        "lid": [33, 7, 163, 144, 145, 153, 154, 155, 133],
        "contour": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        "brow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        "iris": 468,  # centre; 469-472 are points on its edge
    },
    "left": {
        "corners": (263, 362),
        "lid": [263, 249, 390, 373, 374, 380, 381, 382, 362],
        "contour": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
        "brow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
        "iris": 473,
    },
}

# MediaPipe face outline; wrinkle zones are clipped to it (no hair, ears, backdrop).
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]

CURVE_POINTS = 24
MIN_EYE_PX = 12  # smaller eyes than this aren't worth touching
FORESHORTENED = 0.35  # skip an eye narrower than this fraction of the other

# Shapes and scales, in eye widths.
LASH_CLEARANCE = 0.08  # zones start this far below the lash line
# Zone depth below the lash line, straight down the face. The cheek landmarks sit
# too high on some faces to bound the zones: the bag's crease shadow (and the
# tear trough running down toward the nose) can lie well below them, and a zone
# ending on the crease lifts everything above it into a pale pad with a dark rim.
# Generous zones are safe: skin is only lifted where it's darker than its
# surroundings predict, so already-even skin inside the zone is left alone.
DARK_CIRCLE_DEPTH = 0.7
BAG_DEPTH = 0.8
INNER_EXTEND = 0.2  # zones continue this far past the inner corner (tear trough)
OUTER_EXTEND = 0.1
FEATHER = 0.15
SURROUND = 0.5  # how far around the zone to look for reference skin
TONE_SIGMA = 0.12  # scale of the tone being corrected
REFERENCE_SIGMA = 0.35
BAG_FINE = 0.035  # below this scale is texture, left alone
BAG_COARSE = 0.22  # above this scale is facial lighting, left alone

# Wrinkles, in eye widths unless noted.
WRINKLE_SCALES = (0.010, 0.018, 0.028)  # line half-widths the detector looks for
WRINKLE_CLEARANCE = 0.04  # closer to the lashes than the other zones
WRINKLE_DEPTH = 0.55
WRINKLE_OUTER_EXTEND = 0.3  # under-eye zone runs out to meet the crow's feet
LINE_LOW, LINE_HIGH = 0.4, 1.1  # line strength ramp, Lab L units
# Deep expression folds are softened as a whole, by the same fraction across
# their profile, keeping their shape. Excluding their darkest pixels instead
# hollows the fold out, leaving a thin drawn-on line in a flattened patch.
FOLD_LOW, FOLD_HIGH = 8.0, 18.0  # Lab L depth over which a line counts as a fold
FOLD_KEEP = 0.6  # share of a deep fold's depth that's kept
TOO_DARK_LOW, TOO_DARK_HIGH = 22.0, 30.0  # far deeper than any wrinkle: stray hairs
BROW_CLEARANCE = 0.12  # eyebrows are protected, with this margin
WRINKLE_MAX = 1.0

# The line under the bag, lifted by the Eye bags slider.
CREASE_SCALES = (0.02, 0.035, 0.05)  # wider than wrinkles
# Starts below the lashes but high enough to take in the lower-eyelid crease,
# which sits about 0.2-0.3 eye widths down; a short feather at the top so it
# gets the full lift (lashes are also protected by the eye exclusion and the
# too-dark guard).
CREASE_TOP = 0.12
CREASE_FEATHER = 0.06
CREASE_FOLD_KEEP = 0.15  # removing this line is the point, so little is kept

# Eye whites, iris and catchlights. Distances in eye widths, lightness in Lab L.
LID_MARGIN = 0.03  # stay this far inside the lids (lashes, lid shadow)
CARUNCLE_CLEAR = (0.06, 0.14)  # fade out toward the pink inner corner
WHITES_DESATURATE = 1.0  # share of the whites' colour cast removed at full
WHITES_SKIN_SHARE = 0.3  # whites are aimed at this share of the nearby skin's colour:
# a white lit by the same light, where plain neutral reads blue-grey beside skin
WHITES_TARGET_L = 95.0
WHITES_LIFT = 0.55  # share of the way to WHITES_TARGET_L lifted at full
WHITES_MAX_LIFT = 25.0
WHITES_STRENGTH = 0.625  # overall scale of the slider (halved at the user's request, then +25%)
WHITES_FEATHER = 0.04  # eye widths: the whites edit fades out toward lids, iris and corners
WHITES_LID_FADE = 0.085  # eye widths: fades in from the lids, where the white is shaded
# A real white is brightest beside the iris and falls off into the corners; an
# even lift right out to them looks painted on. Distance from the iris centre,
# in iris radii: full strength up to the first, WHITES_CORNER_KEEP by the second.
# (Both fades 15% shorter than first tried, at the user's request.)
WHITES_TAPER = (1.4, 2.4)
WHITES_CORNER_KEEP = 0.1
WHITES_SHADING = 2.0  # lift scales with (brightness / the eye's white)^this: shading kept
WHITES_NOT_LASH = (0.3, 0.7)  # share of this eye's bright white: darker is lashes, not white
VEIN_SCALES = (0.006, 0.012)  # vein half-widths the detector looks for
# Calibrated on real veins (P1246006, P1256101): they score ~0.3-0.4 as lines
# and are ~1.5-2.5 Lab a redder than the white around them.
VEIN_LOW, VEIN_HIGH = 0.12, 0.5  # redness-line strength ramp (Lab a units)
VEIN_MIN_RED = (0.6, 2.2)  # how much redder than the white around it
VEIN_LID_MARGIN = 0.012  # eye widths inside the lids (the lid rim is red too)
VEIN_CARUNCLE_CLEAR = (0.05, 0.1)  # nearer the inner corner than the whites edit goes
VEIN_FULL = 2.5  # the vein map is scaled by this: half-detected veins are fully replaced
REDNESS_CLEAN_PCT = 20  # the white's cleanest colour: this percentile of its redness (Lab a)
REDNESS_SHARE = 0.85  # share of the white's pinkness above that taken out at full
VEIN_REFERENCE = 0.05  # neighbourhood the clean white is taken from
IRIS_EDGE_KEEP = 0.8  # outer part of the iris (the dark limbal ring) is kept
IRIS_DETAIL = (0.012, 0.08)  # fibre band: finer is noise, coarser is shading
IRIS_DETAIL_GAIN = 0.8
IRIS_DETAIL_CLIP = 6.0  # Lab L
IRIS_SATURATION = 0.6
IRIS_SAT_RANGE = 0.8  # Iris saturation at either end: colour x (1 +/- this)
IRIS_HUE_RANGE = 75.0  # degrees the iris colour turns at either end of Iris hue
# Brown irises carry little colour, so turning it alone barely shows: at either
# end of Iris hue the iris's colour strength (Lab chroma) is brought up to this.
IRIS_HUE_CHROMA = 20.0
IRIS_MAX_LIFT = 8.0
LASH_REACH = 0.14  # eye widths: lashes reach this far out from the lid line
LASH_INSIDE = 0.03  # and start this far inside it (the lash roots on the lid rim)
LASH_CONTEXT = 0.05  # the lid skin a lash is darker than
LASH_DARK = (3.0, 12.0)  # Lab L darker than the skin around: lid texture -> lash
LASH_DETAIL = 0.02  # eye widths: clarity works on detail finer than this
LASH_CLARITY = 1.2
LASH_DEEPEN = 0.4
CATCH_SIGMA = 0.08  # surroundings a catchlight stands out from
CATCH_EXCESS = (6.0, 18.0)  # how much brighter than surroundings to count
CATCH_BOOST = 1.6
CATCH_MIN_L = (45.0, 70.0)  # Lab L: dimmer spots aren't catchlights

MAX_LIFT = 20.0  # Lab L units
MAX_COLOUR_SHIFT = 15.0  # Lab a/b units
BAG_MAX = 0.85  # full slider still leaves a little shape; 100% looks unnatural
BURN_RATIO = 0.6  # highlights toned down less than creases are lifted
COLOUR_MATCH_L = 8.0  # lifts this big (Lab L) take on the surrounding skin's colour
COLOUR_MATCH_SIGMA = 0.08  # neighbourhood for that colour, in eye widths


@dataclass
class Params:
    dark_circles: float = 0.5
    eye_bags: float = 0.4
    wrinkles: float = 0.0
    whites: float = 0.0
    iris: float = 0.0
    iris_saturation: float = 0.0  # -1 muted .. 0 unchanged .. +1 richer iris colour
    iris_hue: float = 0.0  # turns the iris colour around the colour wheel, -1 .. +1
    catchlight: float = 0.0
    veins: float = 0.0
    lashes: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_noop(self) -> bool:
        return self.iris_saturation == 0 and self.iris_hue == 0 and all(
            v <= 0
            for v in (
                self.dark_circles,
                self.eye_bags,
                self.wrinkles,
                self.whites,
                self.iris,
                self.catchlight,
                self.veins,
                self.lashes,
            )
        )


def _resample(pts: np.ndarray, n: int = CURVE_POINTS) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, t[-1], n)
    return np.stack([np.interp(u, t, pts[:, 0]), np.interp(u, t, pts[:, 1])], axis=1)


def _poly_mask(shape, poly: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(poly * 16).astype(np.int32)], 1, lineType=cv2.LINE_AA, shift=4)
    return mask.astype(np.float32)


def _inner_feather(hard: np.ndarray, sigma: float) -> np.ndarray:
    """Soft mask that is exactly zero at the polygon edge and ramps up inside it."""
    soft = blur(hard, max(0.5, sigma))
    return np.clip((soft - 0.5) * 2, 0, 1) * hard


def _face_down(lm: np.ndarray) -> np.ndarray:
    """Unit vector down the face: perpendicular to the line between the outer eye
    corners, pointing toward the mouth. Robust to head tilt."""
    across = lm[263] - lm[33]
    down = np.array([-across[1], across[0]], np.float32)
    down /= max(float(np.linalg.norm(down)), 1e-6)
    if np.dot(lm[13] - (lm[33] + lm[263]) / 2, down) < 0:  # 13 = upper lip
        down = -down
    return down


def _extend(curve: np.ndarray, before: float, after: float) -> np.ndarray:
    """Continue a curve past both ends along its end directions."""
    d0 = curve[0] - curve[1]
    d1 = curve[-1] - curve[-2]
    d0 /= max(float(np.linalg.norm(d0)), 1e-6)
    d1 /= max(float(np.linalg.norm(d1)), 1e-6)
    return np.concatenate([[curve[0] + d0 * before], curve, [curve[-1] + d1 * after]])


def _eye_geometry(lm: np.ndarray, spec: dict):
    outer, inner = lm[spec["corners"][0]], lm[spec["corners"][1]]
    eye_w = float(np.linalg.norm(outer - inner))
    down = _face_down(lm)
    lid = _resample(lm[spec["lid"]])  # outer corner -> inner corner
    top = _extend(lid, OUTER_EXTEND * eye_w, INNER_EXTEND * eye_w) + LASH_CLEARANCE * eye_w * down
    wtop = (
        _extend(lid, WRINKLE_OUTER_EXTEND * eye_w, INNER_EXTEND * eye_w)
        + WRINKLE_CLEARANCE * eye_w * down
    )
    out = (outer - inner) / max(eye_w, 1e-6)  # away from the nose
    up = -down
    # Crow's feet: a fan beyond the outer corner, not reaching up to the brow.
    crows = outer + eye_w * np.array(
        [0.05 * out + 0.25 * up, 0.9 * out + 0.4 * up, 1.05 * out, 0.9 * out + 0.55 * down,
         0.1 * out + 0.35 * down],
        np.float32,
    )
    return {
        "eye_w": eye_w,
        "lid": lid,
        "dark_circle": np.concatenate([top, (top + DARK_CIRCLE_DEPTH * eye_w * down)[::-1]]),
        "bag": np.concatenate([top, (top + BAG_DEPTH * eye_w * down)[::-1]]),
        "under_eye_lines": np.concatenate([wtop, (wtop + WRINKLE_DEPTH * eye_w * down)[::-1]]),
        "bag_crease": np.concatenate(
            [
                top + (CREASE_TOP - LASH_CLEARANCE) * eye_w * down,
                (top + (BAG_DEPTH + 0.1) * eye_w * down)[::-1],
            ]
        ),
        "crows_feet": crows,
        "face": lm[FACE_OVAL],
        "contour": lm[spec["contour"]],
        "brow": lm[spec["brow"]],
        "iris_centre": lm[spec["iris"]],
        "iris_radius": float(
            np.linalg.norm(lm[spec["iris"] + 1 : spec["iris"] + 5] - lm[spec["iris"]], axis=1).mean()
        ),
        "inner_corner": inner,
    }


def _below_lid(shape, lid: np.ndarray, eye_w: float) -> np.ndarray:
    """Everything under the lower lid line (extended sideways), so reference
    skin is never taken from the eye, upper lid or brow."""
    ext = 0.6 * eye_w
    left = lid[0] + (lid[0] - lid[1]) / np.linalg.norm(lid[0] - lid[1]) * ext
    right = lid[-1] + (lid[-1] - lid[-2]) / np.linalg.norm(lid[-1] - lid[-2]) * ext
    far = 10 * eye_w
    poly = np.concatenate([[left], lid, [right], [right + [0, far]], [left + [0, far]]])
    return _poly_mask(shape, poly)


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def _line_strength(
    L: np.ndarray, eye_w: float, scales: tuple[float, ...] = WRINKLE_SCALES
) -> np.ndarray:
    """How strongly each pixel sits in a thin dark line (Lab L units, >= 0).

    The largest Hessian eigenvalue is large and positive across a dark valley;
    along a line the other is near zero, while on a round pore both are similar.
    The (1 - |l2|/l1) factor keeps lines and suppresses pores."""
    strength = np.zeros_like(L)
    for scale in scales:
        sigma = max(0.7, scale * eye_w)
        g = blur(L, sigma)
        gy, gx = np.gradient(g)
        gxy, gxx = np.gradient(gx)
        gyy, _ = np.gradient(gy)
        half_tr = (gxx + gyy) / 2
        root = np.sqrt(((gxx - gyy) / 2) ** 2 + gxy**2)
        l1, l2 = half_tr + root, half_tr - root  # l1 >= l2
        line = sigma**2 * l1 * (1 - np.clip(np.abs(l2) / np.maximum(l1, 1e-6), 0, 1))
        strength = np.maximum(strength, np.where(l1 > 0, line, 0))
    return strength


def _wrinkle_lift(
    L: np.ndarray,
    zone: np.ndarray,
    eye_w: float,
    scales: tuple[float, ...] = WRINKLE_SCALES,
    fold_keep: float = FOLD_KEEP,
    line_ramp: tuple[float, float] = (LINE_LOW, LINE_HIGH),
) -> np.ndarray:
    """Lab L increment that fills the lines in ``zone`` at full strength.
    ``line_ramp``: how line-like a crease must be to be filled (softer
    creases, like those across a smiling cheek, need a gentler ramp)."""
    s_max = max(0.7, scales[-1] * eye_w)
    k = 2 * round(2.5 * s_max) + 1
    # Closing removes dark features narrower than the kernel: the skin as it
    # would be without the line. Pores are filled too, but only lines are used.
    filled = cv2.morphologyEx(L, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    filled = blur(filled, max(0.5, 0.5 * s_max))
    # The lift is kept smooth at pore scale, so pores inside a line still show
    # through it instead of being filled in along with the line.
    depth = np.clip(filled - L, 0, None)
    depth = blur(depth, max(0.5, 0.6 * scales[0] * eye_w))
    weight = _smoothstep(_line_strength(L, eye_w, scales), *line_ramp)
    kd = 2 * round(1.5 * s_max) + 1
    weight = max_filter(weight, kd)
    weight = blur(weight, s_max)
    # How deep the line is around each pixel, so a fold is judged as a whole.
    kf = 2 * round(3 * s_max) + 1
    near = max_filter(depth, kf)
    near = blur(near, s_max)
    fold = 1 - fold_keep * _smoothstep(near, FOLD_LOW, FOLD_HIGH)
    not_hair = 1 - _smoothstep(depth, TOO_DARK_LOW, TOO_DARK_HIGH)
    return depth * weight * fold * not_hair * zone


def _lab_l_to_y(L: np.ndarray) -> np.ndarray:
    """CIE L* -> relative luminance Y (0..1)."""
    f = (L + 16) / 116
    return np.where(L > 8, f**3, L / 903.3)


def _relight(lab: np.ndarray, dL: np.ndarray, eye_w: float) -> None:
    """Raise Lab lightness by ``dL`` as a change of *light*, in place.

    Lifting L alone leaves chroma behind, so skin lifted out of shadow turns
    flat and grey; scaling a/b along with it blows up the noisy colour of the
    darkest pixels into red. What a real change of light does is multiply
    linear RGB by a gain, keeping each pixel's colour proportions — so that's
    what this does, with the gain set to hit the requested lightness.

    Skin inside a crease is also genuinely redder (light scatters under the
    skin there), so a lifted crease would keep that tint as a pink line. The
    more a pixel is lifted, the more its colour is taken from the unlifted skin
    around it."""
    old_ab = lab[..., 1:].copy()
    old = lab[..., 0]
    gain = _lab_l_to_y(np.clip(old + dL, 0, 100)) / np.maximum(_lab_l_to_y(old), 1e-4)
    gain = np.clip(gain, 0.25, 4.0)[..., None]
    lin = srgb_to_linear(np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)) * gain
    lab[:] = cv2.cvtColor(
        linear_to_srgb(np.clip(lin, 0, 1)).astype(np.float32), cv2.COLOR_RGB2Lab
    )
    t = _smoothstep(dL, 0.0, COLOUR_MATCH_L)
    if t.max() <= 0:
        return
    around = masked_blur(old_ab, (1 - t) ** 2, max(0.5, COLOUR_MATCH_SIGMA * eye_w))
    lab[..., 1:] += t[..., None] * (around - lab[..., 1:])


def _pupil_radius(L: np.ndarray, d: np.ndarray, r: float, opening: np.ndarray) -> float:
    """Pupil radius from the brightness profile outward from the iris centre.

    Thresholding "dark pixels" fails on dark eyes, where most of the iris is as
    dark as the pupil: the pupil swallows the iris and only an outer ring gets
    enhanced. Instead find where the profile climbs halfway from the pupil to
    the iris; if they're too alike to tell apart, assume a typical size."""
    rings = []
    for k in range(17):  # 0 .. 0.8 r in steps of 0.05 r
        band = (d >= k * 0.05 * r) & (d < (k + 1) * 0.05 * r) & (opening > 0)
        rings.append(float(np.median(L[band])) if band.sum() >= 3 else np.nan)
    prof = np.array(rings)
    if np.isnan(prof[:8]).all() or np.isnan(prof[12:]).all():
        return 0.35 * r
    dark = float(np.nanmin(prof[:8]))
    iris_level = float(np.nanmedian(prof[12:]))
    if iris_level - dark < 6.0:
        return 0.35 * r
    half = dark + 0.5 * (iris_level - dark)
    for k, v in enumerate(prof):
        if not np.isnan(v) and v > half and k * 0.05 * r > 0.1 * r:
            return float(np.clip(k * 0.05 * r, 0.15 * r, 0.6 * r))
    return 0.35 * r


def _remove_veins(lab: np.ndarray, sclera: np.ndarray, eye_w: float, amount: float) -> None:
    """Blood vessels in the white of the eye: thin lines redder than the white
    around them. The line detector runs on redness (Lab a), so the iris edge,
    lash shadows and the white's own shading don't count; each vein pixel then
    takes the colour and brightness of the clean white beside it (measured
    with the veins left out), so it vanishes rather than becoming a pale line.
    Brightness is only ever raised, so no grey patches appear.

    (Repainting veins with LaMa was tried and removed: it filled them with the
    white around them, which near the corners is already pink, and left the
    eye as red as before, while taking 2-3 s.)"""
    a = np.ascontiguousarray(lab[..., 1])
    hard = (sclera > 0.05).astype(np.float32)
    # A red ridge in a is a dark valley in -a: reuse the wrinkle line detector.
    line = _smoothstep(_line_strength(-a, eye_w, VEIN_SCALES), VEIN_LOW, VEIN_HIGH)
    local_a = masked_blur(a, hard, max(0.7, VEIN_REFERENCE * eye_w))
    redder = _smoothstep(a - local_a, *VEIN_MIN_RED)
    vein = line * redder * hard
    k = 2 * round(max(1.0, VEIN_SCALES[-1] * eye_w)) + 1
    vein = max_filter(vein, k)
    vein = np.clip(blur(vein, max(0.5, VEIN_SCALES[0] * eye_w)), 0, 1)
    clean = masked_blur(lab, hard * (1 - vein) ** 2, max(0.7, VEIN_REFERENCE * eye_w))
    target = clean - lab
    target[..., 0] = np.clip(target[..., 0], 0, None)  # only ever brighten
    # A clearly detected vein is fully replaced at full strength: the vein
    # map ramps up gradually, and used as it is it only ever faded veins.
    weight = np.clip(VEIN_FULL * vein, 0, 1)
    lab += (amount * weight * sclera)[..., None] * target
    # Bloodshot whites are pink all over, not just along the veins: pull the
    # white's redness toward its own cleanest part (colour only; brightness
    # and the white's shading stay as they are).
    inside = sclera > 0.5
    if inside.sum() > 20:
        a = lab[..., 1]
        clean_a = float(np.percentile(a[inside], REDNESS_CLEAN_PCT))
        excess = blur(np.clip(a - clean_a, 0, None), max(0.5, VEIN_SCALES[-1] * eye_w))
        lab[..., 1] -= amount * REDNESS_SHARE * sclera * excess


def _eye_itself(lab: np.ndarray, geo: dict, off: np.ndarray, eye_w: float, p: Params) -> None:
    """Whites, iris and catchlights, in place on the eye crop's Lab."""
    shape = lab.shape[:2]
    L = lab[..., 0].copy()
    opening_hard = _poly_mask(shape, geo["contour"] - off)
    opening = _inner_feather(opening_hard, LID_MARGIN * eye_w)

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    cx, cy = geo["iris_centre"] - off
    r = geo["iris_radius"]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    iris_disk = np.clip((r - d) / max(0.5, 0.06 * r) + 0.5, 0, 1)

    # Catchlights first, from the untouched pixels: small spots much brighter
    # than their surroundings. They're shielded from the whites/iris edits.
    around = blur(L, max(0.7, CATCH_SIGMA * eye_w))
    excess = L - around
    catch = (
        _smoothstep(excess, *CATCH_EXCESS)
        * _smoothstep(L, *CATCH_MIN_L)
        * cv2.dilate(opening_hard, np.ones((3, 3), np.uint8))
    )
    catch = np.clip(blur(catch, max(0.5, 0.01 * eye_w)) * 1.5, 0, 1)

    # The white of the eye: the opening minus the iris and the pink inner corner.
    ic = geo["inner_corner"] - off
    from_inner = np.sqrt((xx - ic[0]) ** 2 + (yy - ic[1]) ** 2) / eye_w
    sclera_ring = np.clip((d - 1.08 * r) / max(0.5, 0.1 * r) + 0.5, 0, 1)
    sclera = opening * sclera_ring * _smoothstep(from_inner, *CARUNCLE_CLEAR) * (1 - catch)

    if p.veins > 0:
        # Veins gather toward the corners and lids, where the whites' own mask
        # has already faded: this one reaches closer, stopping short of the red
        # rim of the lids.
        opening_veins = _inner_feather(opening_hard, VEIN_LID_MARGIN * eye_w)
        vein_zone = opening_veins * sclera_ring * _smoothstep(from_inner, *VEIN_CARUNCLE_CLEAR) * (1 - catch)
        _remove_veins(lab, vein_zone, eye_w, p.veins)

    if p.whites > 0:
        taper = 1 - (1 - WHITES_CORNER_KEEP) * _smoothstep(d / max(r, 1.0), *WHITES_TAPER)
        lids = _inner_feather(opening_hard, WHITES_LID_FADE * eye_w)
        s = WHITES_STRENGTH * p.whites * blur(sclera, max(0.5, WHITES_FEATHER * eye_w)) * lids * taper
        Lw = lab[..., 0].copy()
        # The skin around the eye (outside the opening, not lashes or shadow)
        # carries the light's colour.
        around = (opening_hard < 0.5) & (L > 45)
        skin_ab = np.median(lab[around][:, 1:], axis=0) if around.sum() > 50 else np.zeros(2)
        target_ab = (WHITES_SKIN_SHARE * skin_ab).astype(np.float32)
        lab[..., 1:] += (WHITES_DESATURATE * s)[..., None] * (target_ab - lab[..., 1:])
        # Lashes are told apart by being much darker than this eye's own white,
        # not by a fixed level: the shaded side of a white can be darker than a
        # lash on a brighter photo.
        white_L = np.percentile(Lw[sclera > 0.5], 85) if (sclera > 0.5).sum() > 20 else 60.0
        # The lift is set by the eye's bright white and shared out in proportion
        # to brightness, so the white's own shading survives (lifting every
        # pixel toward one level flattens it).
        lift = np.clip((WHITES_TARGET_L - white_L) * WHITES_LIFT, 0, WHITES_MAX_LIFT)
        lift = lift * np.clip(Lw / max(white_L, 1.0), 0, 1) ** WHITES_SHADING
        not_lash = _smoothstep(Lw, WHITES_NOT_LASH[0] * white_L, WHITES_NOT_LASH[1] * white_L)
        lab[..., 0] += s * not_lash * lift

    if p.iris > 0 or p.iris_saturation != 0 or p.iris_hue != 0:
        pupil_r = _pupil_radius(L, d, r, opening_hard)
        pupil = np.clip((pupil_r - d) / max(0.5, 0.08 * r) + 0.5, 0, 1)
        inner_iris = np.clip((IRIS_EDGE_KEEP * r - d) / max(0.5, 0.1 * r) + 0.5, 0, 1)
        W_iris = iris_disk * inner_iris * opening * (1 - pupil) * (1 - catch)

    if p.iris_hue != 0:
        # Turn the iris colour around the colour wheel, keeping its strength
        # and brightness: blue toward violet (+) or teal-green (-), brown
        # toward gold (+) or red-brown (-).
        amount = float(np.clip(p.iris_hue, -1, 1))
        angle = np.deg2rad(IRIS_HUE_RANGE * amount) * W_iris
        a, b = lab[..., 1].copy(), lab[..., 2].copy()
        chroma = np.hypot(a, b)
        # Grey irises (no hue to turn) and near-black pixels (the pupil's
        # edge: colour there reads as red-eye) are left alone.
        floor = IRIS_HUE_CHROMA * abs(amount) * W_iris * _smoothstep(chroma, 1.0, 4.0) * _smoothstep(L, 18.0, 35.0)
        gain = np.maximum(chroma, floor) / np.maximum(chroma, 1e-3)
        a, b = a * gain, b * gain
        c, s_ = np.cos(angle), np.sin(angle)
        lab[..., 1] = a * c - b * s_
        lab[..., 2] = a * s_ + b * c

    if p.iris_saturation != 0:
        # The iris's colour only: pupil, catchlights, the limbal ring and the
        # whites are left as they are.
        sat = IRIS_SAT_RANGE * float(np.clip(p.iris_saturation, -1, 1))
        lab[..., 1:] *= (1 + sat * W_iris)[..., None]

    if p.iris > 0:
        W = W_iris
        s = p.iris * W
        fine = blur(L, max(0.5, IRIS_DETAIL[0] * eye_w))
        coarse = blur(L, max(0.7, IRIS_DETAIL[1] * eye_w))
        # Detail clipped: a strong edge (iris against the white) must never turn
        # into a bright ring, whatever the landmarks say.
        detail = np.clip(fine - coarse, -IRIS_DETAIL_CLIP, IRIS_DETAIL_CLIP)
        # A fixed lift is a huge relative change on a near-black iris in shadow.
        lift = np.clip((70.0 - L) * 0.15, 0, IRIS_MAX_LIFT) * _smoothstep(L, 10.0, 35.0)
        lab[..., 0] += s * (IRIS_DETAIL_GAIN * detail + lift)
        lab[..., 1:] *= (1 + IRIS_SATURATION * s)[..., None]

    if p.catchlight > 0:
        s = p.catchlight * catch
        lab[..., 0] += s * np.minimum(CATCH_BOOST * np.clip(excess, 0, None), 100.0 - lab[..., 0])
        lab[..., 1:] *= (1 - 0.5 * s)[..., None]  # crisp, neutral highlights

    lab[..., 0] = np.clip(lab[..., 0], 0, 100)


def _lashes(lab: np.ndarray, geo: dict, off: np.ndarray, eye_w: float, amount: float) -> None:
    """Deepen and crisp up the lashes, in place on the eye crop's Lab.

    Lashes grow in a band along the lid line, reaching out from it; within the
    band they're told from lid skin by being dark lines against it, so the
    skin's own texture isn't sharpened."""
    shape = lab.shape[:2]
    opening = _poly_mask(shape, geo["contour"] - off)
    outer = grow_mask(opening, 2 * round(LASH_REACH * eye_w) + 1)
    inner = 1 - grow_mask(1 - opening, 2 * round(LASH_INSIDE * eye_w) + 1)  # eroded
    band = blur(np.clip(outer - inner, 0, 1), max(0.5, 0.03 * eye_w))
    L = np.ascontiguousarray(lab[..., 0])
    around = blur(L, max(0.7, LASH_CONTEXT * eye_w))
    dark = np.clip(around - L, 0, None)
    lash = _smoothstep(dark, *LASH_DARK) * band
    lash = np.clip(blur(lash, max(0.5, 0.006 * eye_w)) * 1.5, 0, 1)
    # Clarity: fine local contrast; and the lashes themselves deepened.
    detail = L - blur(L, max(0.5, LASH_DETAIL * eye_w))
    lab[..., 0] = np.clip(
        L + amount * lash * (LASH_CLARITY * detail - LASH_DEEPEN * dark), 0, 100
    )
    # Deepened lashes keep a neutral dark, not a coloured one.
    lab[..., 1:] *= (1 - 0.3 * amount * lash)[..., None]


def _correct_eye(rgb: np.ndarray, geo: dict, p: Params) -> None:
    """Edit one eye's surroundings of ``rgb`` in place."""
    h, w = rgb.shape[:2]
    eye_w = geo["eye_w"]
    pts = np.concatenate([geo["bag"], geo["contour"], geo["crows_feet"]])
    margin = (SURROUND + 0.3) * eye_w
    x0, y0 = np.maximum(np.floor(pts.min(0) - margin), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(pts.max(0) + margin), [w, h]).astype(int)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    off = np.array([x0, y0], np.float32)
    crop = rgb[y0:y1, x0:x1]
    shape = crop.shape[:2]
    lab = cv2.cvtColor(np.clip(crop, 0, 1), cv2.COLOR_RGB2Lab)

    below = _below_lid(shape, geo["lid"] - off, eye_w)
    eye_hole = grow_mask(_poly_mask(shape, geo["contour"] - off), 2 * round(0.12 * eye_w) + 1)
    skin_below = below * (1 - eye_hole)

    # Wrinkles first, while the fine detail is still exactly as shot.
    if p.wrinkles > 0:
        face = _poly_mask(shape, geo["face"] - off)
        kb = 2 * round(BROW_CLEARANCE * eye_w) + 1
        brow = grow_mask(_poly_mask(shape, cv2.convexHull(geo["brow"] - off)[:, 0]), kb)
        zone = np.maximum(
            _poly_mask(shape, geo["under_eye_lines"] - off),
            _poly_mask(shape, geo["crows_feet"] - off),
        ) * face * (1 - eye_hole) * (1 - brow)
        soft = _inner_feather(zone, FEATHER * eye_w)
        L = np.ascontiguousarray(lab[..., 0])
        _relight(lab, p.wrinkles * WRINKLE_MAX * _wrinkle_lift(L, soft, eye_w), eye_w)

    if p.dark_circles > 0:
        zone = _poly_mask(shape, geo["dark_circle"] - off)
        k = 2 * round(SURROUND * eye_w) + 1
        near = grow_mask(zone, k)
        k2 = 2 * round(0.05 * eye_w) + 1
        gap = grow_mask(zone, k2)
        ring = near * (1 - gap) * skin_below
        if ring.sum() > 50:
            expected = masked_blur(lab, ring, REFERENCE_SIGMA * eye_w)
            current = masked_blur(lab, zone * (1 - eye_hole), TONE_SIGMA * eye_w)
            delta = expected - current
            delta[..., 0] = np.clip(delta[..., 0], 0, MAX_LIFT)
            delta[..., 1:] = np.clip(delta[..., 1:], -MAX_COLOUR_SHIFT, MAX_COLOUR_SHIFT)
            delta = blur(delta, max(0.5, 0.1 * eye_w))
            soft = _inner_feather(zone, FEATHER * eye_w)
            lab += p.dark_circles * soft[..., None] * delta

    if p.eye_bags > 0:
        crease = _poly_mask(shape, geo["bag_crease"] - off) * (1 - eye_hole)
        L = np.ascontiguousarray(lab[..., 0])
        _relight(
            lab,
            p.eye_bags
            * _wrinkle_lift(
                L,
                _inner_feather(crease, CREASE_FEATHER * eye_w),
                eye_w,
                CREASE_SCALES,
                CREASE_FOLD_KEEP,
            ),
            eye_w,
        )
        zone = _poly_mask(shape, geo["bag"] - off)
        L = np.ascontiguousarray(lab[..., 0])
        fine = blur(L, max(0.5, BAG_FINE * eye_w))
        coarse = masked_blur(L, skin_below, BAG_COARSE * eye_w)
        mid = fine - coarse
        adjust = np.where(mid < 0, -mid, -BURN_RATIO * mid)
        soft = _inner_feather(zone, FEATHER * eye_w)
        _relight(lab, p.eye_bags * BAG_MAX * soft * adjust, eye_w)

    if p.whites > 0 or p.iris > 0 or p.iris_saturation != 0 or p.iris_hue != 0 or p.catchlight > 0 or p.veins > 0:
        _eye_itself(lab, geo, off, eye_w, p)

    if p.lashes > 0:
        _lashes(lab, geo, off, eye_w, p.lashes)

    rgb[y0:y1, x0:x1] = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def apply(rgb: np.ndarray, faces: list[np.ndarray], p: Params) -> np.ndarray:
    """rgb float32 HxWx3 0..1; faces are (478, 2) landmark arrays as fractions of
    width/height (from the face landmark model). Returns a new rgb."""
    out = rgb.copy()
    if p.is_noop():
        return out
    h, w = rgb.shape[:2]
    for face in faces:
        lm = face * np.array([w, h], np.float32)
        geos = {side: _eye_geometry(lm, spec) for side, spec in EYES.items()}
        widest = max(g["eye_w"] for g in geos.values())
        for geo in geos.values():
            if geo["eye_w"] < MIN_EYE_PX or geo["eye_w"] < FORESHORTENED * widest:
                continue
            _correct_eye(out, geo, p)
    return out


def eye_openings(shape: tuple[int, int], faces: list[np.ndarray]) -> np.ndarray:
    """Soft mask (float32 HxW) of every eye opening, lashes included, for tools
    that need to treat the eyes differently from skin."""
    h, w = shape
    mask = np.zeros((h, w), np.float32)
    for face in faces:
        lm = face * np.array([w, h], np.float32)
        for spec in EYES.values():
            eye_w = float(np.linalg.norm(lm[spec["corners"][0]] - lm[spec["corners"][1]]))
            one = _poly_mask((h, w), lm[spec["contour"]])
            k = 2 * max(1, round(0.05 * eye_w)) + 1
            one = grow_mask(one, k)
            mask = np.maximum(mask, blur(one, max(0.5, 0.02 * eye_w)))
    return mask



IRIS_SCALE_STOPS = (-1.0, -0.5, 0.0, 0.5, 1.0)
IRIS_SCALE_L = 55.0  # Lab L the colour bar is shown at, whatever the eye's own brightness
IRIS_SCALE_CHROMA = 2.2  # the bar's colours a little richer than the eye's, to read at a glance


def iris_scale(rgb: np.ndarray, faces: list[np.ndarray]) -> list[str] | None:
    """The colours Iris hue turns this photo's irises through, from far left to
    far right, as #rrggbb, for the slider's colour bar; None if no iris was
    found. Measured by applying Iris hue and averaging over the iris."""
    base = apply(rgb, faces, Params())
    # Where Iris hue acts: the change at both ends.
    moved = np.abs(apply(rgb, faces, Params(iris_hue=1.0)) - base).max(2) + np.abs(
        apply(rgb, faces, Params(iris_hue=-1.0)) - base
    ).max(2)
    if moved.max() <= 0:
        return None
    weight = moved / moved.sum()
    colours = []
    for amount in IRIS_SCALE_STOPS:
        out = base if amount == 0 else apply(rgb, faces, Params(iris_hue=amount))
        mean = (out * weight[..., None]).sum((0, 1)).astype(np.float32)
        lab = cv2.cvtColor(mean.reshape(1, 1, 3), cv2.COLOR_RGB2Lab)[0, 0]
        lab[0] = IRIS_SCALE_L
        lab[1:] *= IRIS_SCALE_CHROMA
        rgb8 = np.round(np.clip(cv2.cvtColor(lab.reshape(1, 1, 3), cv2.COLOR_Lab2RGB)[0, 0], 0, 1) * 255).astype(int)
        colours.append("#%02x%02x%02x" % tuple(rgb8))
    return colours

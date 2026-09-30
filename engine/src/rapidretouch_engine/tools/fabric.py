"""Clothes crease removal: flattens creases in the clothes, found automatically.

Creases are ripples of light and shade between the fabric's weave and the
garment's own shape, plus the thin dark line and highlight along each one.
The ripples are flattened and the lines filled in (as a change of light, in
linear RGB, so the colour is untouched), while the weave, seams, stitching and
buttons (finer), prints and dark gaps (contrasty with sharp edges), the
fabric's pattern (stripes: lines all running one way) and the garment's shape
and thick folds (broader) are kept. Thick folds are the same size as the body
under the fabric, so flattening them would flatten the figure too.

The skin that's left out is found by the skin tools (``skin.body_regions``).

Scales are set from the face width, since creases scale with the person in the
frame; without a face, from the image size.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import region_edit
from .colour import linear_to_srgb, srgb_to_linear
from .filters import masked_blur

CREASE_FINE = 0.02  # face widths: finer than this (weave, seams, stitching) is kept
LIGHTING = 0.2  # face widths: broader than this is the garment's shape and light, kept
PRINT = (0.8, 1.4)  # log-brightness off the fabric's shading: crease -> print or gap
# ...but only with a sharp edge: log-brightness change across EDGE_SPAN face
# widths (a fold's rolling shading changes far less over that distance).
EDGE_FINE = 0.003
EDGE_SPAN = 0.01
SHARP = (0.3, 0.6)
MAX_FLATTEN = 1.0
MAX_STOPS = 1.5
# Light fabric is flattened up to its lit level instead of its average. The
# lit level weights brighter parts more (exp(LIT_WEIGHT x log brightness above
# the average)); fabric counts as light when that level (linear) is in LIGHT_FABRIC.
LIT_WEIGHT = 4.0
LIGHT_FABRIC = (0.2, 0.45)
SKIN_MARGIN = 0.006  # share of the long edge left clear around skin
NO_FACE_SCALE = 0.12  # stand-in face width, as a share of the long edge

# Crease lines, removed like wrinkles (see ``crease_lines``): dark features
# narrower than 2 x FILL_REACH are filled; finer than WEAVE shows through.
FILL_REACH = 0.03  # face widths
WEAVE = 0.004
LINE_MAX_DEPTH = 0.9  # log brightness: deeper than this is a gap or a seam, kept
LINE_SCALES = (0.005, 0.009, 0.015)  # line widths the pattern detector looks at
# Stripes, checks and ribbing are lines too, but regular ones: many, running
# the same way. Lines that follow the fabric's dominant direction where that
# direction is strongly shared are treated as its pattern and kept.
PATTERN_AREA = 0.08  # face widths the fabric's direction is judged over
PATTERN_COHERENCE = (0.35, 0.6)  # how strongly one direction dominates: none .. a clear pattern
# ...and how much of the area is covered by lines (scale-normalised strength
# over PATTERN_LINE). Measured: striped shirt 67-91%, creased plain shirt 28-69%.
PATTERN_LINE = 0.06
PATTERN_COVER = (0.65, 0.8)


def clothes_mask(
    rgb: np.ndarray,
    subject: np.ndarray,
    faces: list[np.ndarray],
    clothes: np.ndarray | None = None,
    edits: list[dict] | None = None,
) -> np.ndarray | None:
    """Where the clothes are, 0..1: the subject below the chin, minus the head
    and minus skin (the neck and body skin the skin tools find, so neck,
    chest, hands and arms are left out). Worked out on a small copy; None
    without a face (``faces``: landmarks as fractions of width/height)."""
    from .skin import FACE_OVAL, body_regions, face_width

    people = body_regions(rgb, faces, subject, clothes)
    if not people:
        return None
    h, w = rgb.shape[:2]
    sh, sw = people[0].body.shape
    sub = cv2.resize(subject.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    yy = np.mgrid[0:sh, 0:sw][0].astype(np.float32)
    skin = np.zeros((sh, sw), np.float32)
    below = np.zeros((sh, sw), np.float32)
    for person in people:
        skin = np.maximum(skin, np.maximum(person.neck, person.body))
        lm = person.lm * np.array([sw, sh], np.float32)
        fw = face_width(lm)
        # Clothes start around the chin; above it is head and hair.
        below = np.maximum(below, np.clip((yy - lm[152][1]) / max(1.0, 0.15 * fw), 0, 1))
        head = np.zeros((sh, sw), np.uint8)
        cv2.fillPoly(head, [np.round(lm[FACE_OVAL]).astype(np.int32)], 1)
        below *= 1 - cv2.dilate(head, np.ones((3, 3), np.uint8), iterations=max(1, round(0.05 * fw)))
    # Skin grown a little: smoothing a bit less clothing is harmless,
    # flattening an arm's shading is not.
    skin = cv2.dilate(skin, np.ones((3, 3), np.uint8), iterations=max(2, round(SKIN_MARGIN * max(sh, sw))))
    skin = cv2.GaussianBlur(skin, (0, 0), 1.5)
    clothes = sub * (1 - skin) * below
    clothes = cv2.GaussianBlur(clothes, (0, 0), 1.0)
    # The user's corrections to the clothes area (see region_edit).
    return region_edit.apply(cv2.resize(clothes, (w, h), interpolation=cv2.INTER_LINEAR), edits or [], (h, w))


@dataclass
class Correction:
    """The full-strength crease correction for a photo, measured once: the
    slider then only scales it (``gain``), like a layer's opacity."""

    shape: tuple[int, int]
    box: tuple[int, int, int, int]  # y0, y1, x0, x1: where it changes anything
    change: np.ndarray  # log brightness to add, inside the box
    cap: np.ndarray  # the most gain each pixel takes before its brightest channel clips

    def gain(self, amount: float) -> np.ndarray | None:
        """Per-pixel light gain (linear RGB) at ``amount`` (0..1) of the correction."""
        if amount <= 0:
            return None
        y0, y1, x0, x1 = self.box
        # No crease needs more than MAX_STOPS.
        g = np.exp(np.clip(amount * self.change, -MAX_STOPS * np.log(2), MAX_STOPS * np.log(2)))
        full = np.ones(self.shape, np.float32)
        full[y0:y1, x0:x1] = np.minimum(g, self.cap)
        return full


def gain_map(rgb: np.ndarray, amount: np.ndarray, face_px: float | None = None) -> np.ndarray | None:
    """Per-pixel light gain (linear RGB) that flattens creases by ``amount``
    (0..1 per pixel: the slider times the clothes mask), or None if it changes
    nothing."""
    corr = measure(rgb, amount, face_px)
    return corr.gain(1.0) if corr else None


def measure(rgb: np.ndarray, amount: np.ndarray, face_px: float | None = None) -> Correction | None:
    """The crease correction where ``amount`` (0..1 per pixel: the clothes
    mask) says, or None if there's nothing to correct.

    Measured once from the photo and then applied with ``apply_gain``: the
    other tools only change backdrop and skin, so the same gain still fits
    the fabric after them."""
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
    # can't leave a halo around themselves. A deep fold can be as contrasty,
    # but its shading rolls over gradually where a print's edge is abrupt, so
    # only contrast with a sharp edge nearby counts.
    strong = np.clip((np.abs(crease) - PRINT[0]) / (PRINT[1] - PRINT[0]), 0, 1)
    fine = cv2.GaussianBlur(lg, (0, 0), max(0.5, EDGE_FINE * fw))
    gy, gx = np.gradient(fine)
    edge = np.hypot(gx, gy) * EDGE_SPAN * fw
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=max(1, round(fs)))
    strong = strong * np.clip((edge - SHARP[0]) / (SHARP[1] - SHARP[0]), 0, 1)
    strong = cv2.dilate(strong, np.ones((3, 3), np.uint8), iterations=max(1, round(fs)))
    inform = inform * (1 - strong)
    level = masked_blur(lg, inform, fs)
    broad = masked_blur(lg, inform, bs)
    crease = level - broad
    keep = np.clip(1.5 * cv2.GaussianBlur(strong, (0, 0), fs), 0, 1)
    weight = a * (1 - keep) * (1 - blown)
    # Light fabric (a white shirt) is flattened up to its lit level: its
    # creases are shadows, and evening it toward its average pulled the white
    # down as much as the shadows up, turning it grey. The lit level is a
    # brightness-weighted average, near the top of what's there.
    lit = masked_blur(level, inform * np.exp(LIT_WEIGHT * np.clip(crease, -2, 2)), bs)
    light = np.clip((np.exp(lit) - LIGHT_FABRIC[0]) / (LIGHT_FABRIC[1] - LIGHT_FABRIC[0]), 0, 1)
    change = -MAX_FLATTEN * weight * crease
    # Darker fabric is only evened out, never brightened or darkened overall
    # (lifting it to its sheen would wash it out): take away the change's own
    # local average, over the same area the creases are measured against.
    avg = masked_blur(change, (weight > 0.05).astype(np.float32), max(1.0, LIGHTING * fw))
    change = change - weight * (1 - light) * avg + MAX_FLATTEN * weight * light * (lit - broad)
    # Then the crease lines themselves, finer than that band, filled like
    # wrinkles. (Repainting them with LaMa was tried and removed less: LaMa
    # continues the structure around a hole, and a crease's line is the edge
    # of a fold, so it redrew it.)
    change = change + crease_lines(lg + change, a * (1 - blown), fw, keep_highlights=light)
    # Lifting near-white fabric must not clip it, or one channel before the
    # others (a colour cast): cap the gain where the brightest channel would.
    cap = np.maximum(1.0, 1.0 / np.maximum(lin.max(axis=2), 1e-4)).astype(np.float32)
    return Correction((h, w), (y0, y1, x0, x1), change.astype(np.float32), cap)


def _lines(lg: np.ndarray, fw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How line-like each pixel is (scale-normalised depth, log brightness),
    at its best scale in LINE_SCALES, with the line's direction as a
    double-angle vector (cos 2a, sin 2a) of the across-line direction. Dark
    lines (valleys) and thin highlights (ridges) both count."""
    best = np.zeros_like(lg)
    c2 = np.zeros_like(lg)
    s2 = np.zeros_like(lg)
    for scale in LINE_SCALES:
        sigma = max(0.7, scale * fw)
        g = cv2.GaussianBlur(lg, (0, 0), sigma)
        gy, gx = np.gradient(g)
        gxy, gxx = np.gradient(gx)
        gyy, _ = np.gradient(gy)
        half = (gxx + gyy) / 2
        root = np.sqrt(((gxx - gyy) / 2) ** 2 + gxy**2)
        l1, l2 = half + root, half - root  # most and least curved
        # A line bends strongly across and hardly along: round dips don't count.
        valley = np.where(l1 > 0, l1 * np.clip(1 - np.abs(l2) / np.maximum(l1, 1e-6), 0, 1), 0)
        ridge = np.where(l2 < 0, -l2 * np.clip(1 - np.abs(l1) / np.maximum(-l2, 1e-6), 0, 1), 0)
        strength = sigma**2 * np.maximum(valley, ridge)
        phi = 0.5 * np.arctan2(2 * gxy, gxx - gyy)  # across a valley; a ridge's is at right angles
        sign = np.where(ridge > valley, -1.0, 1.0)
        better = strength > best
        best = np.where(better, strength, best)
        c2 = np.where(better, sign * np.cos(2 * phi), c2)
        s2 = np.where(better, sign * np.sin(2 * phi), s2)
    return best.astype(np.float32), c2.astype(np.float32), s2.astype(np.float32)


def crease_lines(lg: np.ndarray, amount: np.ndarray, fw: float, keep_highlights=0.0) -> np.ndarray:
    """Change of log brightness that removes crease lines, by ``amount`` (0..1
    per pixel), the way the face's wrinkle tools remove wrinkles: the fabric
    as it would be with every narrow dark feature filled in (a closing) is
    worked out, and the fabric is moved toward it. That takes the sharp edge
    where a fold turns away from the light as well as thin creases.

    Light fabric (``keep_highlights``, 0..1 per pixel) is lifted to that
    filled level: its creases are shadows. Darker fabric is moved to halfway
    between it and the level with narrow highlights taken down (an opening),
    so creases even out without the fabric being washed out to its sheen.
    The fabric's pattern (stripes, checks, ribbing: lines sharing one
    direction) and gaps or seams far deeper than a crease are kept, and the
    weave shows through the change."""
    fill, pattern = _line_fill(lg, fw, keep_highlights)
    return (amount * (1 - pattern) * fill).astype(np.float32)


def _line_fill(lg: np.ndarray, fw: float, keep_highlights) -> tuple[np.ndarray, np.ndarray]:
    """The log-brightness fill that takes out the crease lines (see
    ``crease_lines``), and where the fabric's pattern is."""
    pattern = _pattern(lg, fw)
    level = cv2.GaussianBlur(lg.astype(np.float32), (0, 0), max(0.5, WEAVE * fw))
    k = 2 * round(FILL_REACH * fw) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(level, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(level, cv2.MORPH_OPEN, kernel)
    target = keep_highlights * closed + (1 - keep_highlights) * (closed + opened) / 2
    fill = target - level
    deep = np.clip((np.abs(fill) - 0.6 * LINE_MAX_DEPTH) / (0.4 * LINE_MAX_DEPTH), 0, 1)
    fill = cv2.GaussianBlur(fill * (1 - deep), (0, 0), max(0.5, WEAVE * fw))
    return fill, pattern


def _pattern(lg: np.ndarray, fw: float) -> np.ndarray:
    """Where the fabric has a pattern of lines (stripes, checks, ribbing),
    0..1: many lines around each pixel sharing one direction, and this
    pixel's own line running that way.

    Looked for above the weave (a twill's fine diagonals all run one way
    too), and only where lines cover most of the area: stripes are lines
    everywhere, while creases, however strong, leave plain fabric between."""
    strength, c2, s2 = _lines(cv2.GaussianBlur(lg.astype(np.float32), (0, 0), max(0.5, WEAVE * fw)), fw)
    covered = cv2.GaussianBlur((strength > PATTERN_LINE).astype(np.float32), (0, 0), max(1.0, PATTERN_AREA * fw))
    dense = np.clip((covered - PATTERN_COVER[0]) / (PATTERN_COVER[1] - PATTERN_COVER[0]), 0, 1)
    # The fabric's dominant direction around each pixel, and how strongly it
    # dominates (1: every line runs the same way, as in stripes).
    area = max(1.0, PATTERN_AREA * fw)
    vx = cv2.GaussianBlur(strength * c2, (0, 0), area)
    vy = cv2.GaussianBlur(strength * s2, (0, 0), area)
    total = cv2.GaussianBlur(strength, (0, 0), area) + 1e-6
    v = np.hypot(vx, vy)
    coherence = v / total
    align = np.clip((c2 * vx + s2 * vy) / np.maximum(v, 1e-6), 0, 1)
    t = np.clip((coherence - PATTERN_COHERENCE[0]) / (PATTERN_COHERENCE[1] - PATTERN_COHERENCE[0]), 0, 1)
    pattern = t * t * (3 - 2 * t) * align**2 * dense
    # The pattern's lines are a few pixels apart: cover the gaps between them.
    return np.clip(cv2.GaussianBlur(pattern, (0, 0), max(1.0, 2 * LINE_SCALES[-1] * fw)) * 1.5, 0, 1)


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

"""Relight: a radial gradient (an ellipse the user places, usually over the
face) that brightens or darkens and warms or cools what's inside it, fading
out toward its edge, like Lightroom's radial filter. Or what's outside it,
with ``invert``: darken the surroundings to lead the eye to the face.

It's per-pixel, so like the tone it comes after the retouching and moving its
sliders or its shape never reruns anything else. Exposure and warmth are gains
in linear light, as a light would add.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .colour import linear_to_srgb, srgb_to_linear

TEMPERATURE_RANGE = 0.25  # log gain on red (opposite on blue) at either end, as Tone's


@dataclass
class Params:
    cx: float = 0.5  # centre, fractions of the photo's width and height
    cy: float = 0.5
    rx: float = 0.2  # half-axes, fractions of the photo's long edge
    ry: float = 0.25
    angle: float = 0.0  # degrees, clockwise
    feather: float = 0.6  # share of the radius the effect fades over, inward from the edge
    exposure: float = 0.0  # EV
    warmth: float = 0.0  # -1 cooler .. +1 warmer
    invert: bool = False  # change outside the ellipse instead

    @classmethod
    def from_dict(cls, d: dict | None) -> "Params":
        d = d or {}
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def is_noop(self) -> bool:
        return self.exposure == 0 and self.warmth == 0


def weight(shape: tuple[int, int], p: Params, box, full_shape) -> np.ndarray:
    """How much of the effect each pixel of an image of ``shape`` gets (0..1),
    the image covering ``box`` (x0, y0, x1, y1, pixels) of a photo of
    ``full_shape``: the whole photo for the preview, a part when zoomed in."""
    h, w = full_shape
    oh, ow = shape
    x0, y0, x1, y1 = box
    xs = x0 + (np.arange(ow, dtype=np.float32) + 0.5) * ((x1 - x0) / ow) - p.cx * w
    ys = y0 + (np.arange(oh, dtype=np.float32) + 0.5) * ((y1 - y0) / oh) - p.cy * h
    long_edge = max(h, w)
    a = np.radians(p.angle)
    c, s = np.float32(np.cos(a)), np.float32(np.sin(a))
    rx = max(p.rx * long_edge, 1.0)
    ry = max(p.ry * long_edge, 1.0)
    # Into the ellipse's own axes (it's turned clockwise by ``angle``).
    u = (xs[None, :] * c + ys[:, None] * s) / rx
    v = (-xs[None, :] * s + ys[:, None] * c) / ry
    d = np.sqrt(u * u + v * v)
    feather = float(np.clip(p.feather, 0.01, 1.0))
    t = np.clip((1 - d) / feather, 0, 1)
    inside = t * t * (3 - 2 * t)  # smoothstep: no visible edge where the fade starts
    return (1 - inside if p.invert else inside).astype(np.float32)


def apply(rgb: np.ndarray, params: dict | None, box=None, full_shape=None) -> np.ndarray:
    """``rgb`` relit (float32 HxWx3, 0..1); ``box`` and ``full_shape`` as for
    ``weight`` (default: ``rgb`` is the whole photo). Returns ``rgb`` itself
    when there's nothing to do."""
    p = Params.from_dict(params)
    if params is None or p.is_noop():
        return rgb
    h, w = rgb.shape[:2]
    if full_shape is None:
        full_shape = (h, w)
    if box is None:
        box = (0, 0, full_shape[1], full_shape[0])
    wt = weight((h, w), p, box, full_shape)[..., None]
    t = TEMPERATURE_RANGE * float(np.clip(p.warmth, -1, 1))
    log_gain = np.array([t, 0.0, -t], np.float32)
    log_gain += np.float32(np.log(2.0) * p.exposure)
    # Brightness kept for the warmth, as Tone's Temperature: only Exposure brightens.
    log_gain -= np.float32(np.log(np.exp(np.array([t, 0, -t])) @ np.array([0.2126, 0.7152, 0.0722])))
    gain = np.exp(wt * log_gain)
    return linear_to_srgb(np.clip(srgb_to_linear(np.clip(rgb, 0, 1)) * gain, 0, 1)).astype(np.float32)

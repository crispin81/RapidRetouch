"""Backdrop smoothing.

The backdrop is split into a low-frequency layer (the lighting gradient and falloff,
kept) and everything finer (creases, seams, dust, grain — discarded), then matched
synthetic grain is added back so the result doesn't look plasticky.

Hair is the hard part. Blending a smoothed backdrop over the image by the mask would
dilute semi-transparent strands and leave halos. Instead, near the subject we only
apply the backdrop's own low-frequency correction, extrapolated in from clean
backdrop and scaled by (1 - alpha) — the matting equation I = aF + (1-a)B with B
swapped for the smoothed version — so strand detail and original grain survive.

All spatial parameters are fractions of the image's long edge, so a downscaled
preview and the full-resolution export look the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .colour import linear_to_srgb, srgb_to_linear
from .filters import blur as _blur
from .filters import masked_blur as _masked_blur

LF_WORKING_EDGE = 1024  # lighting estimate is low-frequency; no need for full res
EDGE_WORKING_EDGE = 2048


@dataclass
class Params:
    strength: float = 1.0  # 0..1, blend of the whole effect
    smoothness: float = 1.5  # crease scale removed, % of long edge
    evenness: float = 0.0  # 0 = keep broad shading as shot, 1 = even out blotchy shadows
    grain: float = 1.0  # multiplier on the measured grain
    edge_protect: float = 0.4  # width of the hair transition zone, % of long edge
    exposure: float = 0.0  # backdrop brightness in stops (EV); subject untouched
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _resize_to_edge(img: np.ndarray, edge: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, edge / max(h, w))
    if scale == 1.0:
        return img, 1.0
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), scale


def _upsize(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if img.shape[:2] == shape:
        return img
    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def _robust_lighting(img: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Fit the smooth lighting surface, ignoring creases, seams and dust.

    A plain blur only spreads a crease out; here pixels that stand out from the
    current estimate are progressively down-weighted (Tukey biweight, IRLS), so
    they drop out of the fit instead of smearing into it.
    """
    denoised = cv2.GaussianBlur(img, (0, 0), 1.5)
    est = _masked_blur(img, weight, sigma)
    sel = weight > 0.5
    if sel.sum() < 100:
        return est
    robust_w = np.ones_like(weight)
    for _ in range(4):
        resid = np.abs(denoised - est).max(axis=2)
        scale = 1.4826 * np.median(resid[sel]) + 1e-4
        u = resid / (4.0 * scale)
        robust_w = np.where(u < 1, (1 - u**2) ** 2, 0).astype(np.float32)
        est = _masked_blur(img, weight * robust_w, sigma)
    return est


GLOBAL_FIT_EDGE = 256
GLOBAL_FIT_DEGREE = 4
GLOBAL_FIT_ITERS = 15
GLOBAL_FIT_BELOW_WEIGHT = 0.01  # asymmetric least squares weight for darker pixels


def _poly_basis(h: int, w: int) -> np.ndarray:
    """2D polynomial terms, coordinates scaled by the long edge (resolution independent)."""
    long_edge = max(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (2 * xx - (w - 1)) / long_edge
    y = (2 * yy - (h - 1)) / long_edge
    terms = [
        x**i * y**j
        for i in range(GLOBAL_FIT_DEGREE + 1)
        for j in range(GLOBAL_FIT_DEGREE + 1 - i)
    ]
    return np.stack([t.ravel() for t in terms], axis=1)


def global_lighting(img: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """The intended lighting across the whole backdrop: a smooth polynomial surface.

    Fitted with asymmetric least squares, so uneven shadows, blotches and creases
    don't pull it — it describes the key light's gradient and falloff only. Evaluated
    behind the subject too, so it's also the reference for spot detection.

    The fit is deliberately lopsided: a shadow is always darker than the light
    that should be there, so pixels below the surface get a tiny weight and the
    fit climbs out of shadows toward the lit backdrop. A symmetric fit treats
    broad shadows as dimmer lighting and settles about halfway into them; a
    rejection threshold gets stuck wherever the first fit landed.

    ``img`` should already be crease- and grain-free (the output of
    ``_robust_lighting``): with an asymmetric fit, any noise left above the
    surface would bias it upward.
    """
    h, w = img.shape[:2]
    small, _ = _resize_to_edge(img, GLOBAL_FIT_EDGE)
    sh, sw = small.shape[:2]
    w_small = cv2.resize(weight, (sw, sh), interpolation=cv2.INTER_AREA).ravel()
    basis = _poly_basis(sh, sw)
    values = small.reshape(-1, 3)
    sel = w_small > 0.5
    if sel.sum() < basis.shape[1] * 10:
        return _masked_blur(img, weight, max(h, w) / 8)
    robust = np.ones(sel.sum(), np.float32)
    for _ in range(GLOBAL_FIT_ITERS):
        sw_ = np.sqrt(w_small[sel] * robust)[:, None]
        coef, *_ = np.linalg.lstsq(basis[sel] * sw_, values[sel] * sw_, rcond=None)
        fit = (basis[sel] @ coef).mean(axis=1)
        below = values[sel].mean(axis=1) < fit
        robust = np.where(below, GLOBAL_FIT_BELOW_WEIGHT, 1.0).astype(np.float32)
    return (_poly_basis(h, w) @ coef).reshape(h, w, 3).astype(np.float32)


def backdrop_weight(alpha: np.ndarray, edge_protect_px: float) -> np.ndarray:
    """1 where we're confident it's clean backdrop, 0 on and near the subject."""
    w = np.clip((1.0 - alpha - 0.05) / 0.9, 0, 1).astype(np.float32)
    r = max(1, round(edge_protect_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(w, kernel)


def _fine_band(img: np.ndarray) -> np.ndarray:
    return img - cv2.GaussianBlur(img, (0, 0), 1.5)


def _grain_model(img: np.ndarray, weight: np.ndarray):
    """Measure the backdrop's grain: channel covariance and spatial correlation."""
    band = _fine_band(img)
    sel = weight > 0.95
    if sel.sum() < 1000:
        return None
    samples = band[sel]
    # Robust to dust specks and residual crease edges: drop the outer 1% tails.
    lo, hi = np.percentile(samples, [1, 99], axis=0)
    keep = np.all((samples >= lo) & (samples <= hi), axis=1)
    cov = np.cov(samples[keep].T) + np.eye(3) * 1e-10
    # Lag-1 horizontal autocorrelation tells us how "clumpy" the grain is.
    a, b = band[:, :-1], band[:, 1:]
    s = sel[:, :-1] & sel[:, 1:]
    rho = float(np.corrcoef(a[s].mean(axis=1), b[s].mean(axis=1))[0, 1])
    return np.linalg.cholesky(cov), rho


def _grain_sigma_for_rho(rho: float, rng: np.random.Generator) -> float:
    """Pick the pre-blur for white noise whose fine band matches the measured rho."""
    probe = rng.standard_normal((256, 256)).astype(np.float32)
    best, best_err = 0.0, np.inf
    for sigma in (0.0, 0.35, 0.5, 0.65, 0.8, 1.0, 1.25):
        n = cv2.GaussianBlur(probe, (0, 0), sigma) if sigma else probe
        f = n - cv2.GaussianBlur(n, (0, 0), 1.5)
        r = np.corrcoef(f[:, :-1].ravel(), f[:, 1:].ravel())[0, 1]
        if abs(r - rho) < best_err:
            best, best_err = sigma, abs(r - rho)
    return best


def _synth_grain(shape, chol, sigma, rng) -> np.ndarray:
    h, w = shape
    z = rng.standard_normal((h, w, 3)).astype(np.float32)
    if sigma:
        z = cv2.GaussianBlur(z, (0, 0), sigma)
    # Normalise so the *fine band* of the noise has unit variance per channel,
    # then colour it with the measured covariance — matching what we measured.
    std = _fine_band(z).reshape(-1, 3).std(axis=0)
    z /= np.maximum(std, 1e-8)
    return z @ chol.T.astype(np.float32)


@dataclass
class Prepared:
    """Everything that depends only on crease size and edge protection.

    The result is linear in evenness and grain (every step in between is a
    weighted sum with fixed weights), so it's stored as separate terms and
    ``compose`` is just arithmetic — fast enough to track a slider.
    """

    rgb: np.ndarray
    base: np.ndarray  # change at evenness 0, no grain
    even: np.ndarray  # extra change per unit of evenness
    grain: np.ndarray | None  # matched grain at multiplier 1
    # For backdrop exposure: subject mask, backdrop confidence, and the smooth
    # backdrop lighting (kept at working size; scaled up only when used).
    alpha: np.ndarray
    w_soft: np.ndarray
    light_small: np.ndarray  # at evenness 0
    to_even_small: np.ndarray


def prepare(rgb: np.ndarray, alpha: np.ndarray, p: Params) -> Prepared:
    """rgb float32 HxWx3 0..1, alpha float32 HxW (1 = subject)."""
    h, w = rgb.shape[:2]
    long_edge = max(h, w)
    rng = np.random.default_rng(p.seed)

    # Low-frequency lighting estimate from backdrop pixels only, plus the
    # smooth "intended lighting" surface that evenness blends toward.
    img_s, scale = _resize_to_edge(rgb, LF_WORKING_EDGE)
    alpha_s = _upsize(alpha, img_s.shape[:2]) if scale != 1 else alpha
    w_s = backdrop_weight(alpha_s, p.edge_protect / 100 * long_edge * scale)
    lf = _robust_lighting(img_s, w_s, p.smoothness / 100 * long_edge * scale)
    to_even = global_lighting(lf, w_s) - lf

    # Low/mid-frequency correction (smoothed minus actual), extrapolated from clean
    # backdrop into the hair zone at a finer scale so it follows local creases.
    img_e, scale_e = _resize_to_edge(rgb, EDGE_WORKING_EDGE)
    alpha_e = _upsize(alpha, img_e.shape[:2]) if scale_e != 1 else alpha
    w_e = backdrop_weight(alpha_e, p.edge_protect / 100 * long_edge * scale_e)
    edge_sigma = max(1.0, 0.3 / 100 * long_edge * scale_e)
    delta_ext = _masked_blur(_upsize(lf, img_e.shape[:2]) - img_e, w_e, edge_sigma)
    delta_even = _masked_blur(_upsize(to_even, img_e.shape[:2]), w_e, edge_sigma)

    # Full-resolution fields.
    edge_px = p.edge_protect / 100 * long_edge
    w_full = backdrop_weight(alpha, edge_px)
    w_soft = cv2.GaussianBlur(w_full, (0, 0), max(1.0, edge_px / 2))
    # A Gaussian never quite reaches zero; its tail times the large subject-to-
    # backdrop difference would tint the subject, so cap by the backdrop share.
    w_soft = np.minimum(w_soft, 1.0 - alpha)[..., None]
    edge_share = (1.0 - w_soft) * (1.0 - alpha)[..., None]

    base = w_soft * (_upsize(lf, (h, w)) - rgb) + edge_share * _upsize(delta_ext, (h, w))
    even = w_soft * _upsize(to_even, (h, w)) + edge_share * _upsize(delta_even, (h, w))

    # Original grain survives with weight (1 - w_soft). Two independent noises
    # blended linearly lose up to ~30% of their strength, which would show as a
    # smooth ring around the subject, so add synthetic grain in quadrature instead.
    grain = None
    model = _grain_model(rgb, w_full)
    if model is not None:
        chol, rho = model
        sigma = _grain_sigma_for_rho(rho, rng)
        # Scaled by (1 - alpha) too: grain only belongs where backdrop shows
        # through, and the sqrt is steep near zero, so the blur's faint tail
        # would otherwise sprinkle grain over the subject.
        grain_weight = np.sqrt(np.clip(1.0 - (1.0 - w_soft) ** 2, 0, 1))
        grain_weight *= (1.0 - alpha)[..., None]
        grain = grain_weight * _synth_grain((h, w), chol, sigma, rng)

    return Prepared(rgb, base, even, grain, alpha, w_soft, lf, to_even)


def compose(prep: Prepared, p: Params) -> np.ndarray:
    change = prep.base + p.evenness * prep.even
    if prep.grain is not None and p.grain > 0:
        change = change + p.grain * prep.grain
    out = np.clip(prep.rgb + p.strength * change, 0, 1).astype(np.float32)
    if p.exposure != 0:
        out = _expose_backdrop(out, prep, p)
    return out


def _expose_backdrop(rgb: np.ndarray, prep: Prepared, p: Params) -> np.ndarray:
    """Scale the backdrop's light by 2**exposure, leaving the subject alone.

    Done in linear light, so it behaves like changing the backdrop's lighting:
    grain and texture scale with it. At hair edges the matting equation
    I = aF + (1-a)B gives I' = I + (1-a)(g-1)B: only the backdrop's share of
    each pixel changes, so darkening a white backdrop leaves no light halo.
    B is the pixel itself where there's no subject in it (keeping its grain),
    and the smooth backdrop lighting estimate (extrapolated behind the subject)
    wherever there's any: a mixed pixel's own value already contains subject."""
    h, w = rgb.shape[:2]
    gain = 2.0**p.exposure
    lin = srgb_to_linear(rgb)
    light = _upsize(prep.light_small + p.evenness * prep.to_even_small, (h, w))
    light = srgb_to_linear(np.clip(light, 0, 1))
    pure = np.clip(1 - prep.alpha / 0.02, 0, 1)[..., None]
    backdrop = pure * lin + (1 - pure) * light
    lin = lin + (1 - prep.alpha)[..., None] * (gain - 1) * backdrop
    out = linear_to_srgb(np.clip(lin, 0, 1)).astype(np.float32)
    # Solid subject passes through untouched (not even the round trip's rounding).
    return np.where((prep.alpha >= 1)[..., None], rgb, out)


def prepare_key(p: Params) -> tuple:
    """Parameters that require a new ``prepare``; the rest only need ``compose``."""
    return (p.smoothness, p.edge_protect, p.seed)


def apply(rgb: np.ndarray, alpha: np.ndarray, p: Params) -> np.ndarray:
    """rgb float32 HxWx3 0..1, alpha float32 HxW (1 = subject). Returns new rgb."""
    return compose(prepare(rgb, alpha, p), p)

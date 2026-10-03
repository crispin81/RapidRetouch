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

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from .. import parallel
from ..parallel import by_bands
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


# Grain is the same all over the backdrop, so it's measured on evenly spaced
# strips of rows (enough for a steady estimate, a fraction of a 100 MP photo)
# and from at most GRAIN_SAMPLES of their pixels.
GRAIN_STRIPS = 24
GRAIN_STRIP_ROWS = 64
GRAIN_SAMPLES = 2_000_000
GRAIN_EDGE_ROWS = 4  # rows at each strip's edge, where the band's blur runs off it


def _grain_strips(h: int) -> list[slice]:
    if h <= 2 * GRAIN_STRIPS * GRAIN_STRIP_ROWS:
        return [slice(0, h)]
    starts = np.linspace(0, h - GRAIN_STRIP_ROWS, GRAIN_STRIPS).astype(int)
    return [slice(int(a), int(a) + GRAIN_STRIP_ROWS) for a in starts]


def _grain_model(img: np.ndarray, weight_small: np.ndarray):
    """Measure the backdrop's grain: channel covariance and spatial correlation.
    Where it's clean backdrop comes from ``weight_small`` (``backdrop_weight``
    at low resolution, scaled up): only well inside it is used, so its
    precision doesn't matter, and eroding at full resolution took seconds."""
    h, w = img.shape[:2]
    bands, sels = [], []
    e = GRAIN_EDGE_ROWS
    for rows in _grain_strips(h):
        weight = _window_resize(weight_small, (0, rows.start, w, rows.stop), (h, w), (rows.stop - rows.start, w))
        band = _fine_band(img[rows])
        sel = weight > 0.95
        if band.shape[0] > 4 * e:
            band, sel = band[e:-e], sel[e:-e]
        bands.append(band)
        sels.append(sel)
    band = np.concatenate(bands)
    sel = np.concatenate(sels)
    if sel.sum() < 1000:
        return None
    samples = band[sel]
    samples = samples[:: max(1, len(samples) // GRAIN_SAMPLES)]
    # Robust to dust specks and residual crease edges: drop the outer 1% tails.
    lo, hi = np.percentile(samples, [1, 99], axis=0)
    keep = np.all((samples >= lo) & (samples <= hi), axis=1)
    cov = np.cov(samples[keep].T) + np.eye(3) * 1e-10
    # Lag-1 horizontal autocorrelation tells us how "clumpy" the grain is.
    a, b = band[:, :-1], band[:, 1:]
    s = sel[:, :-1] & sel[:, 1:]
    pa, pb = a[s].mean(axis=1), b[s].mean(axis=1)
    step = max(1, len(pa) // GRAIN_SAMPLES)
    rho = float(np.corrcoef(pa[::step], pb[::step])[0, 1])
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


# Grain noise comes in fixed blocks, each from its own seed, so any part of
# the photo (a zoomed-in view) gets exactly the grain the whole photo (the
# export) has there.
NOISE_BLOCK = 256
NOISE_THREADS = 16


def _block_noise(box, full_shape, seed: int, sigma: float) -> np.ndarray:
    """Standard normal float32 noise (3 channels) for ``box`` (x0, y0, x1, y1)
    of a ``full_shape`` image, blurred by ``sigma``, normalised so its fine
    band has unit variance per channel. Blocks are made on several threads
    (NumPy releases the GIL while filling): ~10x faster on a 100 MP frame."""
    H, W = full_shape
    x0, y0, x1, y1 = box
    pad = int(np.ceil(4 * sigma)) + 1 if sigma else 0
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    out = np.empty((Y1 - Y0, X1 - X0, 3), np.float32)
    B = NOISE_BLOCK
    blocks = [(by, bx) for by in range(Y0 // B, (Y1 - 1) // B + 1) for bx in range(X0 // B, (X1 - 1) // B + 1)]

    def fill(block) -> None:
        by, bx = block
        noise = np.random.default_rng([seed, by, bx]).standard_normal((B, B, 3), dtype=np.float32)
        ya, yb = max(Y0, by * B), min(Y1, (by + 1) * B)
        xa, xb = max(X0, bx * B), min(X1, (bx + 1) * B)
        out[ya - Y0 : yb - Y0, xa - X0 : xb - X0] = noise[ya - by * B : yb - by * B, xa - bx * B : xb - bx * B]

    if parallel.inside() or len(blocks) < 4:
        for block in blocks:  # already one of several bands in parallel
            fill(block)
    else:
        with ThreadPoolExecutor(min(NOISE_THREADS, os.cpu_count() or 1)) as pool:
            list(pool.map(fill, blocks))
    if sigma:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    out = np.ascontiguousarray(out[y0 - Y0 : y1 - Y0, x0 - X0 : x1 - X0])
    # The fine band's spread depends only on sigma, so it's measured on a probe.
    probe = np.random.default_rng([seed, 2**31]).standard_normal((512, 512, 3), dtype=np.float32)
    if sigma:
        probe = cv2.GaussianBlur(probe, (0, 0), sigma)
    out /= np.maximum(_fine_band(probe).reshape(-1, 3).std(axis=0), 1e-8).astype(np.float32)
    return out


def _window_resize(field: np.ndarray, box, full_shape, out_shape) -> np.ndarray:
    """A low-resolution ``field`` covering the whole image, scaled up (bicubic,
    as ``_upsize``) to ``box`` of a ``full_shape`` image at ``out_shape``."""
    H, W = full_shape
    h, w = out_shape
    if tuple(box) == (0, 0, W, H):
        return _upsize(field, (h, w))
    x0, y0, x1, y1 = box
    fh, fw = field.shape[:2]
    xs = ((x0 + (np.arange(w, dtype=np.float64) + 0.5) * (x1 - x0) / w) * fw / W - 0.5).astype(np.float32)
    ys = ((y0 + (np.arange(h, dtype=np.float64) + 0.5) * (y1 - y0) / h) * fh / H - 0.5).astype(np.float32)
    mx, my = np.meshgrid(xs, ys)
    return cv2.remap(field, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


@dataclass
class Lighting:
    """What smoothing needs from the whole photo: the lighting and the
    creases' correction at low resolution, and the grain. With it, any part of
    the photo can be smoothed on its own (``prepare`` with a ``box``)."""

    full_shape: tuple[int, int]
    lf: np.ndarray  # smooth backdrop lighting, LF_WORKING_EDGE
    to_even: np.ndarray  # change toward the intended lighting, LF_WORKING_EDGE
    delta_ext: np.ndarray  # correction extrapolated into the hair, EDGE_WORKING_EDGE
    delta_even: np.ndarray
    grain: tuple | None  # (cholesky of the colour covariance, pre-blur sigma)
    seed: int


def lighting(rgb: np.ndarray, alpha: np.ndarray, p: Params) -> Lighting:
    """rgb float32 HxWx3 0..1, alpha float32 HxW (1 = subject), the whole photo."""
    h, w = rgb.shape[:2]
    long_edge = max(h, w)

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

    grain = None
    model = _grain_model(rgb, w_s)
    if model is not None:
        chol, rho = model
        grain = (chol, _grain_sigma_for_rho(rho, np.random.default_rng(p.seed)))
    return Lighting((h, w), lf, to_even, delta_ext, delta_even, grain, p.seed)


def margin(full_shape: tuple[int, int], p: Params) -> int:
    """Pixels around a ``box`` that ``prepare`` needs to see for the box itself
    to come out exactly as in the whole photo (the hair zone's erosion and blur)."""
    edge_px = p.edge_protect / 100 * max(full_shape)
    return max(1, round(edge_px)) + int(np.ceil(4 * max(1.0, edge_px / 2))) + 2


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
    box: tuple[int, int, int, int]  # the part of the photo this covers
    full_shape: tuple[int, int]


def prepare(rgb: np.ndarray, alpha: np.ndarray, p: Params, light: Lighting | None = None, box=None) -> Prepared:
    """rgb float32 HxWx3 0..1, alpha float32 HxW (1 = subject): the whole
    photo, or with ``light`` (from the whole photo) just ``box`` (x0, y0, x1,
    y1) of it. Pixels within ``margin`` of the box's edges (other than the
    photo's own) aren't exact: crop them off."""
    if light is None:
        light = lighting(rgb, alpha, p)
    H, W = light.full_shape
    box = tuple(box) if box is not None else (0, 0, W, H)
    h, w = rgb.shape[:2]
    up = lambda f: _window_resize(f, box, (H, W), (h, w))  # noqa: E731

    # Full-resolution fields.
    edge_px = p.edge_protect / 100 * max(H, W)
    w_full = backdrop_weight(alpha, edge_px)
    w_soft = cv2.GaussianBlur(w_full, (0, 0), max(1.0, edge_px / 2))
    # A Gaussian never quite reaches zero; its tail times the large subject-to-
    # backdrop difference would tint the subject, so cap by the backdrop share.
    w_soft = np.minimum(w_soft, 1.0 - alpha)[..., None]
    edge_share = (1.0 - w_soft) * (1.0 - alpha)[..., None]

    base = w_soft * (up(light.lf) - rgb) + edge_share * up(light.delta_ext)
    even = w_soft * up(light.to_even) + edge_share * up(light.delta_even)

    # Original grain survives with weight (1 - w_soft). Two independent noises
    # blended linearly lose up to ~30% of their strength, which would show as a
    # smooth ring around the subject, so add synthetic grain in quadrature instead.
    grain = None
    if light.grain is not None:
        chol, sigma = light.grain
        # Scaled by (1 - alpha) too: grain only belongs where backdrop shows
        # through, and the sqrt is steep near zero, so the blur's faint tail
        # would otherwise sprinkle grain over the subject.
        grain_weight = np.sqrt(np.clip(1.0 - (1.0 - w_soft) ** 2, 0, 1))
        grain_weight *= (1.0 - alpha)[..., None]
        noise = _block_noise(box, (H, W), light.seed, sigma)
        grain = grain_weight * cv2.transform(noise, chol.astype(np.float32))

    return Prepared(rgb, base, even, grain, alpha, w_soft, light.lf, light.to_even, box, (H, W))


def smooth_area(rgb: np.ndarray, alpha: np.ndarray, p: Params, light: Lighting, box) -> np.ndarray:
    """The smoothed backdrop for ``box`` (x0, y0, x1, y1) of the whole photo
    ``rgb``, exactly as when smoothing all of it."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = box
    m = margin((H, W), p)
    X0, Y0, X1, Y1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    prep = prepare(rgb[Y0:Y1, X0:X1], alpha[Y0:Y1, X0:X1], p, light, (X0, Y0, X1, Y1))
    return compose(prep, p)[y0 - Y0 : y1 - Y0, x0 - X0 : x1 - X0]


def compose(prep: Prepared, p: Params) -> np.ndarray:
    """The smoothed photo from ``prepare``'s terms. Per pixel, so it's worked
    out in bands of rows on all cores."""
    h, w = prep.rgb.shape[:2]
    return by_bands(lambda a, b: _compose_rows(prep, p, a, b), h, w)


def _compose_rows(prep: Prepared, p: Params, a: int, b: int) -> np.ndarray:
    change = prep.base[a:b] + p.evenness * prep.even[a:b]
    if prep.grain is not None and p.grain > 0:
        change = change + p.grain * prep.grain[a:b]
    out = np.clip(prep.rgb[a:b] + p.strength * change, 0, 1).astype(np.float32)
    if p.exposure != 0:
        out = _expose_backdrop(out, prep, p, a, b)
    return out


def _expose_backdrop(rgb: np.ndarray, prep: Prepared, p: Params, a: int, b: int) -> np.ndarray:
    """Scale the backdrop's light by 2**exposure, leaving the subject alone,
    for rows a..b of ``prep`` (``rgb``).

    Done in linear light, so it behaves like changing the backdrop's lighting:
    grain and texture scale with it. At hair edges the matting equation
    I = aF + (1-a)B gives I' = I + (1-a)(g-1)B: only the backdrop's share of
    each pixel changes, so darkening a white backdrop leaves no light halo.
    B is the pixel itself where there's no subject in it (keeping its grain),
    and the smooth backdrop lighting estimate (extrapolated behind the subject)
    wherever there's any: a mixed pixel's own value already contains subject."""
    h, w = rgb.shape[:2]
    gain = 2.0**p.exposure
    alpha = prep.alpha[a:b]
    lin = srgb_to_linear(rgb)
    x0, y0, x1, y1 = prep.box
    full_h = prep.rgb.shape[0]
    rows_box = (x0, y0 + round(a * (y1 - y0) / full_h), x1, y0 + round(b * (y1 - y0) / full_h))
    light_small = prep.light_small + p.evenness * prep.to_even_small
    light = _window_resize(light_small, rows_box, prep.full_shape, (h, w))
    light = srgb_to_linear(np.clip(light, 0, 1))
    pure = np.clip(1 - alpha / 0.02, 0, 1)[..., None]
    backdrop = pure * lin + (1 - pure) * light
    lin = lin + (1 - alpha)[..., None] * (gain - 1) * backdrop
    out = linear_to_srgb(np.clip(lin, 0, 1)).astype(np.float32)
    # Solid subject passes through untouched (not even the round trip's rounding).
    return np.where((alpha >= 1)[..., None], rgb, out)


def prepare_key(p: Params) -> tuple:
    """Parameters that require a new ``prepare``; the rest only need ``compose``."""
    return (p.smoothness, p.edge_protect, p.seed)


def apply(rgb: np.ndarray, alpha: np.ndarray, p: Params) -> np.ndarray:
    """rgb float32 HxWx3 0..1, alpha float32 HxW (1 = subject). Returns new rgb."""
    return compose(prepare(rgb, alpha, p), p)

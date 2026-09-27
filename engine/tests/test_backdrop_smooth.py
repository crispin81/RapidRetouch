import cv2
import numpy as np

from retouch_engine.tools import backdrop_smooth as bs

H, W = 1500, 1000


def synthetic(seed=1, shadows=False):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Lighting: off-centre hotspot falling off to the corners.
    light = 0.55 + 0.25 * np.exp(-(((xx - 350) / 700) ** 2 + ((yy - 500) / 900) ** 2))
    base = np.stack([light * 0.95, light * 0.97, light], axis=2)
    # Creases: a few soft folds of ~10-25 px width, +-3%.
    creases = np.zeros((H, W), np.float32)
    for x0, slope, width, amp in [(150, 0.15, 12, 0.03), (820, -0.1, 20, -0.025), (500, 0.4, 8, 0.02)]:
        d = xx - (x0 + slope * yy)
        creases += amp * np.exp(-(d / width) ** 2)
    # Dust specks.
    for _ in range(40):
        cv2.circle(creases, (int(rng.integers(0, W)), int(rng.integers(0, H))), 2, -0.08, -1)
    clean = base
    if shadows:
        # Uneven shadows: distinct soft blotches with lit backdrop between them,
        # -3% to -5%. (Darkening in a corner is indistinguishable from light
        # falloff, so evenness deliberately keeps it; not tested here.)
        blotch = np.zeros((H, W), np.float32)
        for cx, cy, r, amp in [(250, 330, 130, -0.05), (780, 300, 110, -0.04), (180, 1000, 120, -0.04), (820, 1150, 100, -0.03)]:
            blotch += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / r**2))
        base = base + blotch[..., None]
    img = base + creases[..., None]
    # Grain: slightly clumpy, mostly luminance.
    lum = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.6)
    chroma = rng.standard_normal((H, W, 3)).astype(np.float32) * 0.3
    grain = (lum[..., None] + chroma) * 0.012
    img = img + grain
    # Subject: dark ellipse with a soft, noisy "hair" edge.
    d = np.sqrt(((xx - 500) / 260) ** 2 + ((yy - 900) / 420) ** 2)
    wisps = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 3) * 0.15
    alpha = np.clip((1.08 - d + wisps) / 0.12, 0, 1).astype(np.float32)
    subject = np.array([0.35, 0.22, 0.15], np.float32) + grain
    img = alpha[..., None] * subject + (1 - alpha[..., None]) * img
    return np.clip(img, 0, 1).astype(np.float32), alpha, clean, grain


def region_stats(shadows=False, **params):
    img, alpha, clean, grain = synthetic(shadows=shadows)
    out = bs.apply(img, alpha, bs.Params(**params))
    far = cv2.erode((alpha < 0.01).astype(np.uint8), np.ones((81, 81), np.uint8)).astype(bool)
    ring = (alpha < 0.01) & ~far & (cv2.dilate((alpha > 0.5).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
    return img, alpha, clean, grain, out, far, ring


def test_creases_removed_lighting_kept_grain_matched_subject_untouched():
    img, alpha, clean, grain, out, far, ring = region_stats()
    lowpass = lambda x: cv2.GaussianBlur(x, (0, 0), 4)
    err_in, err_out = np.abs(lowpass(img) - clean), np.abs(lowpass(out) - clean)
    crease = (err_in.max(axis=2) > 0.006) & far
    flat = far & ~crease
    print(f"crease error: before {err_in[crease].mean():.4f} after {err_out[crease].mean():.4f}")
    print(f"flat error:   before {err_in[flat].mean():.4f} after {err_out[flat].mean():.4f}")
    assert err_out[crease].mean() < err_in[crease].mean() * 0.3
    # Lighting preserved: the smoothed backdrop stays within 0.2% of the true light.
    assert err_out[flat].mean() < 0.002

    band = lambda x: x - cv2.GaussianBlur(x, (0, 0), 1.5)
    g_in, g_out = band(img)[far].std(), band(out)[far].std()
    g_ring = band(out)[ring].std()
    print(f"grain std: original {g_in:.5f} far {g_out:.5f} ring {g_ring:.5f}")
    assert 0.85 < g_out / g_in < 1.15
    assert 0.85 < g_ring / g_in < 1.15

    # Fully solid subject is untouched; nearly-solid pixels move only by their
    # tiny backdrop share (the matting equation), well under one 8-bit level.
    assert np.abs(out - img)[alpha >= 1.0].max() == 0
    assert np.abs(out - img)[alpha > 0.999].max() < 0.5 / 255


def test_strength_zero_is_identity():
    img, alpha, *_ = synthetic()
    out = bs.apply(img, alpha, bs.Params(strength=0.0))
    assert np.abs(out - img).max() < 1e-6


def test_preview_matches_full_res():
    img, alpha, *_ = synthetic()
    full = bs.apply(img, alpha, bs.Params(grain=0))
    small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    small_a = cv2.resize(alpha, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    prev = bs.apply(small, small_a, bs.Params(grain=0))
    full_down = cv2.resize(full, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    diff = np.abs(cv2.GaussianBlur(full_down - prev, (0, 0), 3)).mean()
    print(f"preview vs full-res mean diff {diff:.5f}")
    assert diff < 0.004


def test_evenness_removes_uneven_shadows_keeps_falloff():
    img, alpha, clean, grain, out, far, ring = region_stats(shadows=True, evenness=1.0)
    unshadowed = synthetic(shadows=False)[0]
    lowpass = lambda x: cv2.GaussianBlur(x, (0, 0), 8)
    depth_in = (lowpass(unshadowed) - lowpass(img)).mean(axis=2)
    shadow = far & (depth_in > 0.005)
    depth_out = (clean - lowpass(out)).mean(axis=2)
    print(f"shadow depth: before {depth_in[shadow].mean():.4f} after {depth_out[shadow].mean():.4f}")
    assert depth_out[shadow].mean() < depth_in[shadow].mean() * 0.25
    # Falloff kept: hotspot-to-corner brightness difference survives.
    hot, corner = out[500, 350].mean(), out[40, 960].mean()
    clean_drop = clean[500, 350].mean() - clean[40, 960].mean()
    assert abs((hot - corner) - clean_drop) < 0.02


def test_evenness_zero_keeps_broad_shading():
    img, alpha, clean, grain, out, far, ring = region_stats(shadows=True, evenness=0.0)
    shaded = synthetic(shadows=True)[0]
    lowpass = lambda x: cv2.GaussianBlur(x, (0, 0), 25)
    # At a broad scale the shadows are still there.
    assert np.abs(lowpass(out) - lowpass(shaded))[far].mean() < 0.003


def test_exposure_darkens_only_backdrop_share_of_each_pixel():
    """Compose a subject with a soft edge over a flat white backdrop in linear
    light; after -1 EV every pixel should be aF + (1-a)(B/2) — backdrop halved,
    subject untouched, no light halo at the soft edge."""
    from retouch_engine.tools.colour import linear_to_srgb, srgb_to_linear

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - 500) / 260) ** 2 + ((yy - 900) / 420) ** 2)
    alpha = np.clip((1.05 - d) / 0.15, 0, 1).astype(np.float32)  # wide soft edge
    B = np.array([0.8, 0.8, 0.78], np.float32)  # linear light, near white
    F = np.array([0.05, 0.03, 0.02], np.float32)
    lin = alpha[..., None] * F + (1 - alpha[..., None]) * B
    img = linear_to_srgb(lin).astype(np.float32)

    out = bs.apply(img, alpha, bs.Params(strength=0, grain=0, exposure=-1))
    expected = alpha[..., None] * F + (1 - alpha[..., None]) * B * 0.5
    err = np.abs(srgb_to_linear(out) - expected)
    edge = (alpha > 0.02) & (alpha < 0.98)
    print(f"exposure error: backdrop {err[alpha == 0].max():.5f} edge {err[edge].max():.5f}")
    assert err[alpha == 0].max() < 1e-4
    assert err[edge].max() < 0.01
    assert np.abs(out - img)[alpha >= 1].max() == 0

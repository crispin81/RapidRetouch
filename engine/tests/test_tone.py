import numpy as np

from rapidretouch_engine.tools import tone
from rapidretouch_engine.tools.colour import srgb_to_linear


def test_identity_curve_and_zero_ev_change_nothing():
    img = np.random.default_rng(0).uniform(0, 1, (20, 20, 3)).astype(np.float32)
    assert tone.is_noop({"ev": 0, "curve": [[0, 0], [1, 1]]})
    assert tone.apply(img, {"ev": 0, "curve": [[0, 0], [1, 1]]}) is img


def test_one_stop_doubles_linear_light():
    img = np.full((4, 4, 3), 0.3, np.float32)
    out = tone.apply(img, {"ev": 1})
    np.testing.assert_allclose(srgb_to_linear(out), 2 * srgb_to_linear(img), rtol=1e-3)


def test_curve_passes_through_points_and_never_overshoots():
    pts = [[0, 0], [0.25, 0.4], [0.5, 0.5], [0.75, 0.6], [1, 1]]
    lut = tone.curve_lut(pts)
    xs = np.linspace(0, 1, len(lut))
    for x, y in pts:
        assert abs(np.interp(x, xs, lut) - y) < 1e-3
    assert np.all(np.diff(lut) >= -1e-6)  # monotone points give a monotone curve


def test_contrast_curve_leaves_colour_alone():
    import cv2

    skin = np.full((4, 4, 3), (0.8, 0.55, 0.45), np.float32)
    out = tone.apply(skin, {"curve": [[0, 0], [0.25, 0.18], [0.75, 0.85], [1, 1]]})
    before = cv2.cvtColor(skin, cv2.COLOR_RGB2Lab)[0, 0]
    after = cv2.cvtColor(out, cv2.COLOR_RGB2Lab)[0, 0]
    assert after[0] > before[0] + 1  # brighter (a light tone on an S-curve)
    np.testing.assert_allclose(after[1:], before[1:], atol=0.5)  # same colour

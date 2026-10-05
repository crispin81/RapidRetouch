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


def test_vibrance_protects_skin_and_boosts_dull_colours_most():
    import cv2

    def lab_patch(L, a, b):
        return cv2.cvtColor(np.full((4, 4, 3), (L, a, b), np.float32), cv2.COLOR_Lab2RGB)

    skin_rgb, blue_rgb, grey_blue = lab_patch(65, 15, 18), lab_patch(50, 10, -45), lab_patch(55, 4, -12)
    chroma = lambda rgb: float(np.hypot(*cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[0, 0, 1:]))
    gain = lambda rgb: chroma(tone.apply(rgb, {"vibrance": 1.0})) / chroma(rgb)
    assert gain(skin_rgb) < gain(blue_rgb)  # skin protected
    assert gain(grey_blue) > gain(blue_rgb)  # dull colours gain more (vibrance)
    assert chroma(tone.apply(blue_rgb, {"vibrance": -1.0})) < 0.1 * chroma(blue_rgb)  # -1: near grey
    assert not tone.is_noop({"vibrance": 0.2})


def test_white_balance_warms_cools_and_tints_without_changing_brightness():
    grey = np.full((4, 4, 3), 0.5, np.float32)
    warm = tone.apply(grey, {"temperature": 1.0})[0, 0]
    cool = tone.apply(grey, {"temperature": -1.0})[0, 0]
    magenta = tone.apply(grey, {"tint": 1.0})[0, 0]
    assert warm[0] > warm[2] and cool[2] > cool[0]
    assert magenta[1] < magenta[0] and magenta[1] < magenta[2]
    Y = lambda rgb: float(srgb_to_linear(rgb) @ np.array([0.2126, 0.7152, 0.0722]))
    assert abs(Y(warm) - Y(grey[0, 0])) < 0.01 and abs(Y(magenta) - Y(grey[0, 0])) < 0.01



def test_dehaze_takes_off_the_photos_own_veil_and_keeps_white():
    from rapidretouch_engine.tools import tone

    hazy = np.linspace(0.25, 1.0, 300, dtype=np.float32)[None, :, None].repeat(3, axis=2).repeat(10, axis=0)
    veil = tone.haze_veil(hazy)
    out = tone.apply(hazy, {"dehaze": 1.0, "veil": veil})
    assert out[0, 0, 0] < 0.05  # the lifted black is black again
    assert abs(out[0, -1, 0] - 1.0) < 1e-3  # white stays white
    misty = tone.apply(hazy, {"dehaze": -1.0})
    assert misty[0, 0, 0] > hazy[0, 0, 0]  # below zero adds haze

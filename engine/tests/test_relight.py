import numpy as np

from rapidretouch_engine.tools import relight

P = {"cx": 0.5, "cy": 0.5, "rx": 0.2, "ry": 0.2, "angle": 0, "feather": 0.5, "exposure": 1.0, "warmth": 0}


def test_brightens_inside_by_the_stops_and_leaves_outside():
    rgb = np.full((200, 300, 3), 0.4, np.float32)
    out = relight.apply(rgb, P)
    lin = lambda v: ((v + 0.055) / 1.055) ** 2.4  # noqa: E731
    assert abs(lin(out[100, 150, 0]) / lin(0.4) - 2) < 0.01  # +1 EV at the centre
    assert np.allclose(out[:, 280:], 0.4)  # past the edge: untouched


def test_invert_changes_only_the_outside():
    rgb = np.full((200, 300, 3), 0.4, np.float32)
    out = relight.apply(rgb, {**P, "invert": True, "exposure": -1})
    assert np.allclose(out[100, 150], 0.4) and out[100, 290, 0] < 0.3


def test_a_zoomed_in_crop_matches_the_whole_photo():
    full = relight.apply(np.full((2000, 3000, 3), 0.4, np.float32), {**P, "angle": 30, "rx": 0.1})
    crop = relight.apply(np.full((100, 100, 3), 0.4, np.float32), {**P, "angle": 30, "rx": 0.1}, (1400, 900, 1600, 1100), (2000, 3000))
    assert np.abs(crop - full[900:1100:2, 1400:1600:2]).max() < 1e-4


def test_nothing_to_do_returns_the_image():
    rgb = np.zeros((4, 4, 3), np.float32)
    assert relight.apply(rgb, None) is rgb and relight.apply(rgb, {**P, "exposure": 0}) is rgb

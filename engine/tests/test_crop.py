import numpy as np

from rapidretouch_engine.tools import crop


def test_crop_takes_the_rectangle():
    rgb = np.zeros((100, 200, 3), np.float32)
    rgb[:, 100:] = 1  # right half white
    out = crop.apply(rgb, {"x0": 0.5, "y0": 0.25, "x1": 1.0, "y1": 0.75})
    assert out.shape == (50, 100, 3) and out.min() == 1


def test_positive_angle_turns_clockwise():
    rgb = np.zeros((101, 101, 3), np.float32)
    rgb[45:56, 50:] = 1  # a bar pointing right (3 o'clock)
    out = crop.apply(rgb, {"angle": 90})
    # Turned clockwise by 90 degrees it points down (6 o'clock).
    assert out[80, 50].mean() > 0.9 and out[50, 80].mean() < 0.1


def test_no_crop_changes_nothing():
    rgb = np.random.default_rng(0).random((10, 10, 3)).astype(np.float32)
    assert crop.apply(rgb, None) is rgb and crop.is_noop(crop.IDENTITY)

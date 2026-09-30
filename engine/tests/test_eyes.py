import cv2
import numpy as np

from rapidretouch_engine.tools import eyes


def test_line_detector_prefers_wrinkles_over_pores():
    """A thin dark line should score far higher than dark dots of the same width,
    so wrinkles are lifted while pore texture is left alone."""
    eye_w = 200.0  # detector scales are in eye widths: ~2-3.6 px here
    L = np.full((200, 400), 60.0, np.float32)
    cv2.line(L, (20, 60), (380, 70), 54.0, 3)  # a wrinkle: long, 3 px wide, 6 L deep
    for x in range(30, 380, 25):  # isolated pores: same width and depth
        for y in (130, 165):
            cv2.circle(L, (x, y), 1, 54.0, -1)
    L = cv2.GaussianBlur(L, (0, 0), 0.8)
    s = eyes._line_strength(L, eye_w)
    line = s[55:75, 40:360].max(axis=0).mean()
    pores = s[115:185, :].max()
    print(f"line strength {line:.2f}, strongest pore {pores:.2f}")
    assert line > eyes.LINE_HIGH  # fully lifted
    assert pores < eyes.LINE_LOW  # not lifted at all


def test_wrinkles_off_is_identity():
    rgb = np.random.default_rng(1).random((300, 300, 3)).astype(np.float32)
    face = np.random.default_rng(2).random((478, 2)).astype(np.float32)
    out = eyes.apply(rgb, [face], eyes.Params(dark_circles=0, eye_bags=0, wrinkles=0))
    assert np.array_equal(out, rgb)

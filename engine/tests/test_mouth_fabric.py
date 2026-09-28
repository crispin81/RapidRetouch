import cv2
import numpy as np

from retouch_engine.tools import fabric, mouth


def test_teeth_mask_finds_pale_teeth_but_not_red_gums():
    lab = np.zeros((100, 100, 3), np.float32)
    lab[..., 0], lab[..., 1], lab[..., 2] = 45, 30, 15  # gum / tongue red
    lab[40:60, 30:70] = (85, 2, 18)  # yellowish teeth
    inner = np.zeros((100, 100), np.float32)
    inner[35:65, 20:80] = 1
    m = mouth.teeth_mask(lab, inner, mw=100)
    assert m[50, 50] > 0.9 and m[38, 25] < 0.1


def test_closed_mouth_has_no_teeth():
    lab = np.zeros((100, 100, 3), np.float32)
    lab[..., 0], lab[..., 1], lab[..., 2] = 50, 25, 12  # all lip
    inner = np.zeros((100, 100), np.float32)
    inner[48:52, 20:80] = 1
    assert mouth.teeth_mask(lab, inner, mw=100).max() == 0


def test_creases_flatten_but_backdrop_and_weave_stay():
    rng = np.random.default_rng(0)
    h, w, fw = 400, 400, 200.0
    x = np.arange(w, dtype=np.float32)
    crease = 1 + 0.3 * np.sin(2 * np.pi * x / (0.25 * fw))  # soft ripples
    weave = 1 + 0.04 * rng.standard_normal((h, w)).astype(np.float32)
    lum = 0.25 * crease[None, :] * weave
    rgb = np.repeat(lum[..., None] ** (1 / 2.2), 3, axis=2).astype(np.float32)
    amount = np.ones((h, w), np.float32)
    amount[:, :60] = 0  # not clothes (backdrop) at the left
    out = fabric.apply(rgb, amount, fw)
    ripple = lambda im: cv2.GaussianBlur(im[150:250, 150:250, 1], (0, 0), 4).std()  # noqa: E731
    assert ripple(out) < 0.35 * ripple(rgb)
    fine = lambda im: (im - cv2.GaussianBlur(im, (0, 0), 2))[150:250, 150:250, 1].std()  # noqa: E731
    assert fine(out) > 0.7 * fine(rgb)
    assert np.array_equal(out[:, :40], rgb[:, :40])

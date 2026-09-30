import cv2
import numpy as np

from rapidretouch_engine.tools import skin
from rapidretouch_engine.tools.filters import blur


def _person(h, w, body):
    """A stand-in person: only the face width (landmarks 234 to 454) matters
    to the body tools."""
    lm = np.full((478, 2), 0.5, np.float32)
    lm[234], lm[454] = (0.35, 0.2), (0.65, 0.2)
    lab = np.zeros((8, 8, 3), np.float32)
    lab[...] = (60, 15, 18)
    model = skin.SkinModel.__new__(skin.SkinModel)
    model.rg_mean = np.array([0.45, 0.33], np.float32)
    model.rg_inv = np.eye(2, dtype=np.float32) * 1e4
    model.L_low, model.L_typical, model.tone = 50.0, 60.0, lab[0, 0]
    return skin.Person(lm, model, np.zeros_like(body), body)


def test_body_even_tone_and_smooth_keep_the_colour_and_leave_the_rest():
    rng = np.random.default_rng(1)
    h, w = 600, 800
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (62, 14, 17)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(25):  # soft red blotches, as on a chest
        cy, cx = rng.integers(150, 450), rng.integers(200, 600)
        lab[..., 1] += 6 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 18.0**2))
    lab[..., 0] += rng.normal(0, 0.8, (h, w)).astype(np.float32)  # grain
    rgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
    body = np.zeros((h // 2, w // 2), np.float32)
    body[60:240, 80:320] = 1  # the small copy's body skin: the middle of the photo
    person = _person(h, w, body)
    p = {"even": 1.0, "smooth": 1.0}
    out = skin.apply_body(rgb, [person], None, p)
    before = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[160:440, 200:600]
    after = cv2.cvtColor(out, cv2.COLOR_RGB2Lab)[160:440, 200:600]
    assert blur(after[..., 1], 20).std() < 0.6 * blur(before[..., 1], 20).std()
    assert abs(after[..., 1].mean() - before[..., 1].mean()) < 1.0  # evened, not recoloured
    grain = lambda l: (l[..., 0] - blur(l[..., 0], 2)).std()  # noqa: E731
    assert grain(after) > 0.8 * grain(before)
    assert np.array_equal(out[:60], rgb[:60])  # outside the body skin


def test_no_body_settings_change_nothing():
    rgb = np.random.default_rng(0).random((100, 100, 3)).astype(np.float32)
    person = _person(100, 100, np.ones((50, 50), np.float32))
    assert np.array_equal(skin.apply_body(rgb, [person], {"neck_lines": 0}, {}), rgb)
    assert skin.RegionParams.from_dict({"neck_lines": 0.5}).is_noop() is False

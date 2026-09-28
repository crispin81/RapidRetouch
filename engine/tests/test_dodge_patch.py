import numpy as np

from retouch_engine.tools import dodge_burn, patch
from retouch_engine.tools.colour import srgb_to_linear


def _grey(h=200, w=300, v=0.4):
    return np.full((h, w, 3), v, np.float32)


def test_dodge_brightens_by_its_stops_and_burn_undoes_it():
    img = _grey()
    stroke = {"points": [[0.5, 0.5]], "radius": 0.1, "strength": 1.0, "softness": 0.0}
    dodged = dodge_burn.apply(img, [{**stroke, "mode": "dodge"}])
    gain = srgb_to_linear(dodged[100, 150]) / srgb_to_linear(img[100, 150])
    np.testing.assert_allclose(gain, 2**dodge_burn.MAX_STOPS, rtol=0.02)
    assert np.allclose(dodged[5, 5], img[5, 5])  # untouched away from the stroke
    both = dodge_burn.apply(img, [{**stroke, "mode": "dodge"}, {**stroke, "mode": "burn"}])
    np.testing.assert_allclose(both, img, atol=1e-4)


def test_patch_copies_texture_but_keeps_the_target_tone():
    rng = np.random.default_rng(0)
    h, w = 200, 300
    img = np.empty((h, w, 3), np.float32)
    img[:] = np.linspace(0.3, 0.6, w, dtype=np.float32)[None, :, None]  # lighting ramp
    img[:, :150] *= 1 + 0.05 * rng.standard_normal((h, 150, 1)).astype(np.float32)  # textured left
    img[90:110, 215:235] = 0.1  # a dark blemish on the smooth right
    poly = [[200 / w, 80 / h], [250 / w, 80 / h], [250 / w, 120 / h], [200 / w, 120 / h]]
    out = patch.apply(img, poly, [-120 / w, 0])
    inside = out[92:108, 217:233]
    # Blemish gone, the tone is the ramp's where it lands, texture came along.
    assert abs(inside.mean() - img[80:120, 190:260][img[80:120, 190:260] > 0.2].mean()) < 0.03
    assert inside.std() > 0.005
    assert np.allclose(out[:, :100], img[:, :100])  # source untouched

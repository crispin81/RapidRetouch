import numpy as np

from retouch_engine.tools import dodge_burn, patch


def _grey(h=200, w=300, v=0.4):
    return np.full((h, w, 3), v, np.float32)


def test_dodge_burn_zones_brighten_high_points_and_deepen_the_edge():
    # A synthetic face: landmarks laid out on a unit square scaled to 400 px.
    rng = np.random.default_rng(0)
    lm = rng.uniform(150, 250, (478, 2)).astype(np.float32)
    from retouch_engine.tools.skin import FACE_OVAL

    t = np.linspace(0, 2 * np.pi, len(FACE_OVAL), endpoint=False)
    lm[FACE_OVAL] = np.stack([200 + 150 * np.sin(t), 200 - 180 * np.cos(t)], 1)
    lm[151], lm[9] = (200, 90), (200, 140)
    dodge, burn = dodge_burn.zones((400, 400), lm, fw=300)
    assert dodge[110, 200] > 0.5  # forehead centre
    assert burn[200, 55] > 0.5 and dodge[200, 55] < 0.1  # the face's edge


def test_dodge_burn_off_changes_nothing():
    img = _grey()
    assert np.array_equal(dodge_burn.apply(img, [], 0.0), img)


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

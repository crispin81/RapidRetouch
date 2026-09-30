import numpy as np

from rapidretouch_engine.tools import dodge_burn, patch


def _grey(h=200, w=300, v=0.4):
    return np.full((h, w, 3), v, np.float32)


def test_highlights_and_shadows_strengthen_the_existing_light_and_shade():
    # A face's contours: a lit ridge (a cheekbone) and a shaded hollow (under
    # it), each about a tenth of a face wide, plus pore-scale texture, which
    # must not count as light.
    rng = np.random.default_rng(0)
    h, w, fw = 300, 300, 300.0
    x = np.arange(w, dtype=np.float32)
    bumps = 0.5 * np.exp(-((x - 90) ** 2) / (2 * 15.0**2)) - 0.5 * np.exp(-((x - 210) ** 2) / (2 * 15.0**2))
    Y = 0.25 * np.exp2(bumps)[None, :].repeat(h, 0)
    Y *= 1 + 0.05 * rng.standard_normal((h, w)).astype(np.float32)
    shape = dodge_burn.light_shape(Y, np.ones((h, w), np.float32), fw)
    both = dodge_burn.tone_curve(shape, dodge_burn.Params(highlights=1, shadows=1))
    lit, shade = both[150, 90], both[150, 210]
    assert lit > 0.15 and shade < -0.15  # clearly stronger than the old Sculpt (~0.1)
    assert np.abs(both[:, 140:160]).max() < abs(lit)  # the plain skin between moves least
    assert np.abs(np.diff(both[150])).max() < 0.03  # pores don't become dodge & burn
    # Each slider only touches its own side.
    only_h = dodge_burn.tone_curve(shape, dodge_burn.Params(highlights=1))
    only_s = dodge_burn.tone_curve(shape, dodge_burn.Params(shadows=1))
    assert only_h.min() >= 0 and only_s.max() <= 0


def test_tone_curve_rolls_off_and_never_clips():
    shape = np.linspace(-3, 3, 601, dtype=np.float32)
    added = dodge_burn.tone_curve(shape, dodge_burn.Params(highlights=1, shadows=1))
    assert np.all(np.diff(added) >= 0)  # never reverses the light
    assert added.max() <= dodge_burn.HIGHLIGHT_MAX and added.min() >= -dodge_burn.SHADOW_MAX


def test_old_sculpt_setting_loads_as_highlights_and_shadows():
    p = dodge_burn.Params.from_dict({"amount": 0.4})
    assert (p.contour, p.highlights, p.shadows) == (0, 0.4, 0.4)


def test_dodge_burn_off_changes_nothing():
    img = _grey()
    assert np.array_equal(dodge_burn.apply(img, [], dodge_burn.Params()), img)


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

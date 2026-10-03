import cv2
import numpy as np

from rapidretouch_engine.tools import skin


def _skin_with_a_spot(h=300, w=300, fw=600.0):
    rng = np.random.default_rng(0)
    lab = np.zeros((h, w, 3), np.float32)
    lab[...] = (65, 14, 16)  # plain skin
    lab[..., 0] += rng.normal(0, 0.8, (h, w)).astype(np.float32)  # pores / grain
    lab[..., 1] += rng.normal(0, 0.8, (h, w)).astype(np.float32)  # skin's natural colour variation
    yy, xx = np.mgrid[0:h, 0:w]
    spot = np.exp(-((xx - 150) ** 2 + (yy - 150) ** 2) / (2 * (0.006 * fw) ** 2)).astype(np.float32)
    lab[..., 1] += 10 * spot  # an inflamed red spot
    lab[..., 0] -= 3 * spot
    return lab, spot


def test_acne_heals_a_red_spot_and_leaves_plain_skin():
    lab, spot = _skin_with_a_spot()
    W = np.ones(lab.shape[:2], np.float32)
    before = lab.copy()
    skin._heal_acne(lab, W, 600.0, 1.0)
    core = spot > 0.5
    assert (lab[..., 1][core] - 14).mean() < 0.4 * (before[..., 1][core] - 14).mean()  # redness mostly gone
    far = np.zeros_like(core); far[:60, :60] = True
    assert np.abs(lab[far] - before[far]).max() < 1e-4  # plain skin untouched
    # Grain is kept inside the healed spot: no smooth patch.
    assert lab[..., 0][core].std() > 0.4


def test_acne_slider_takes_clearer_spots_first():
    lab, spot = _skin_with_a_spot()
    lab[..., 1] -= 8.5 * spot  # a faint spot now: slightly redder only
    lab[..., 0] += 3 * spot
    W = np.ones(lab.shape[:2], np.float32)
    assert skin.acne_spots(lab, W, 600.0, 0.0).max() == 0  # low: only the clearest
    assert skin.acne_spots(lab, W, 600.0, 1.0).max() > 0.5  # high: fainter ones too


def test_old_blemishes_setting_loads_as_acne():
    assert skin.RegionParams.from_dict({"blemishes": 0.6}).acne == 0.6

import numpy as np

from rapidretouch_engine.tools import region_edit


def test_add_and_remove_strokes():
    area = np.zeros((100, 200), np.float32)
    area[:, 100:] = 1  # the tool found the right half
    edits = [
        {"mode": "add", "points": [[0.2, 0.5]], "radius": 0.05},  # take in a spot on the left
        {"mode": "remove", "points": [[0.8, 0.5]], "radius": 0.05},  # take out a spot on the right
    ]
    out = region_edit.apply(area, edits, (100, 200))
    assert out[50, 40] > 0.9 and out[50, 160] < 0.1
    assert out[5, 5] == 0 and out[5, 195] == 1  # untouched elsewhere
    assert np.array_equal(region_edit.apply(area, [], (100, 200)), area)


def test_edits_land_in_the_same_place_on_a_crop_at_any_size():
    # The full image is 1000 x 2000; the tool's map covers the box x 500..1500,
    # y 250..750 at half size (250 x 500).
    edits = [{"mode": "add", "points": [[0.5, 0.5]], "radius": 0.01}]  # the image centre
    crop = region_edit.apply(np.zeros((250, 500), np.float32), edits, (1000, 2000), box=(500, 250, 1500, 750))
    ys, xs = np.nonzero(crop > 0.5)
    assert abs(ys.mean() - 125) < 2 and abs(xs.mean() - 250) < 2  # the crop's centre

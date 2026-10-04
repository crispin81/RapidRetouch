import numpy as np

from rapidretouch_engine.server import Engine, _blend


def test_staged_opacity_blends_and_reuses_the_unblended_result():
    e = Engine(lambda m: None)
    calls = []

    def run(img, steps):
        calls.append(1)
        return img + 1.0

    base = np.zeros((4, 4, 3), np.float32)
    full = e._staged(lambda: base, "k", [("a", {"x": 1, "opacity": 1.0}, run)])
    half = e._staged(lambda: base, "k", [("a", {"x": 1, "opacity": 0.5}, run)])
    quarter = e._staged(lambda: base, "k", [("a", {"x": 1, "opacity": 0.25}, run)])
    assert np.allclose(full, 1) and np.allclose(half, 0.5) and np.allclose(quarter, 0.25)
    assert len(calls) == 1  # the tool ran once; opacity changes only re-blended


def test_blend_at_full_is_the_result_itself():
    a, b = np.zeros(3, np.float32), np.ones(3, np.float32)
    assert _blend(a, b, 1.0) is b and np.allclose(_blend(a, b, 0.3), 0.3)

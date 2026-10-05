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


def test_detail_render_gives_way_and_resumes_from_its_finished_steps():
    from rapidretouch_engine.server import Superseded

    e = Engine(lambda m: None)
    calls = []

    def run(img, steps):
        calls.append(1)
        return img + 1.0

    base = np.zeros((4, 4, 3), np.float32)
    stages = [("a", {"x": 1}, run), ("b", {"x": 2}, run)]
    asked = iter([False, True])  # a newer request arrives after the first step
    e.newer_waiting = lambda: next(asked)
    e._yielding = True
    try:
        e._staged(lambda: base, "k", stages)
        raise AssertionError("expected it to give way")
    except Superseded:
        pass
    e.newer_waiting = lambda: False
    assert np.allclose(e._staged(lambda: base, "k", stages), 2)
    assert len(calls) == 2  # step "a" wasn't run again
    e._yielding = False
    e.newer_waiting = lambda: True
    assert np.allclose(e._staged(lambda: base, "k", stages), 2)  # previews and exports never give way

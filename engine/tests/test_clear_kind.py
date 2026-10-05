import numpy as np

from rapidretouch_engine.server import Engine


def test_clear_one_kind_keeps_the_others_and_redoes_only_those_after_it():
    e = Engine(lambda m: None)
    e.status = lambda *a, **k: None
    e.image = type("I", (), {"rgb": np.zeros((8, 8, 3), np.float32)})()
    e.preview = np.zeros((8, 8, 3), np.float32)
    made = []

    def stroke(rgb, st):  # each removal adds its own number, so the order shows
        made.append(st["n"])
        return rgb + st["n"]

    e._apply_stroke = stroke
    e._preview = lambda look: ""
    for n, kind in [(1, "fill"), (2, "reflection"), (4, "fill"), (8, "patch")]:
        st = {"kind": kind, "n": n}
        e.filled_previews.append(stroke(e._edited_preview(), st))
        e.strokes.append(st)
    made.clear()
    r = e.clear_removals(kind="reflection")
    assert r["removals"] == 3 and r["removal_kinds"] == {"fill": 2, "patch": 1}
    assert made == [4, 8]  # the first brush stroke was kept as it was
    assert np.allclose(e._edited_preview(), 1 + 4 + 8)
    made.clear()
    e.clear_removals(kind="reflection")  # nothing of that kind left: nothing redone
    assert made == []
    assert e.clear_removals()["removals"] == 0

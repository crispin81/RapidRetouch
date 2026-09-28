from retouch_engine import presets


def test_save_list_replace_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert presets.list_all() == []
    presets.save("Headshot / Men", {"eyes": {"whites": 0.3}})
    presets.save("Beauty", {"skin": {"face": {"smooth": 0.5}}})
    assert [p["name"] for p in presets.list_all()] == ["Beauty", "Headshot / Men"]
    presets.save("headshot / men", {"eyes": {"whites": 0.6}})  # same file: replaced
    got = {p["name"]: p["settings"] for p in presets.list_all()}
    assert len(got) == 2 and got["headshot / men"] == {"eyes": {"whites": 0.6}}
    presets.delete("Beauty")
    assert [p["name"] for p in presets.list_all()] == ["headshot / men"]

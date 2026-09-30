"""Saved presets: named sets of slider settings, one JSON file each.

Stored in the user's config folder (XDG), so they survive reinstalls and can
be backed up or shared by copying the files.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


def folder() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "rapidretouch" / "presets"


def _file(name: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-") or "preset"
    return folder() / f"{slug}.json"


def list_all() -> list[dict]:
    """[{"name", "settings"}], sorted by name. Unreadable files are skipped."""
    out = []
    if folder().is_dir():
        for f in folder().glob("*.json"):
            try:
                data = json.loads(f.read_text())
                out.append({"name": str(data["name"]), "settings": data["settings"]})
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return sorted(out, key=lambda p: p["name"].lower())


def save(name: str, settings: dict) -> None:
    name = name.strip()
    if not name:
        raise ValueError("a preset needs a name")
    f = _file(name)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"name": name, "settings": settings}, indent=2))
    tmp.replace(f)  # never leave a half-written preset


def delete(name: str) -> None:
    _file(name).unlink(missing_ok=True)

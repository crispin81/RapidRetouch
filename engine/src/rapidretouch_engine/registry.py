"""Model registry: each backend is a YAML manifest plus a thin adapter.

Manifests live in ``manifests/``. Adding a model means dropping in a manifest and,
if its architecture is new, an adapter module in ``adapters/`` — core code never
changes. Weights are never shipped; they download on first use from the official
source named in the manifest.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MANIFEST_DIR = Path(__file__).parent / "manifests"


def data_dir() -> Path:
    """Where downloaded models live: the platform's per-user app data folder
    (%LOCALAPPDATA%\\RapidRetouch on Windows, ~/Library/Application
    Support/RapidRetouch on a Mac, $XDG_DATA_HOME/rapidretouch on Linux)."""
    if sys.platform == "win32":
        path = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "RapidRetouch"
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / "RapidRetouch"
    else:
        path = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "rapidretouch"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


class LicenceNotAccepted(Exception):
    def __init__(self, manifest: "Manifest"):
        super().__init__(
            f"{manifest.name} is licensed {manifest.licence} and must be accepted before download"
        )
        self.manifest = manifest


@dataclass
class Manifest:
    id: str
    name: str
    task: str
    licence: str
    licence_url: str
    commercial_use: bool
    bundled_default: bool
    source: dict[str, Any]
    adapter: str
    quality_note: str = ""
    min_vram_gb: float = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        raw = yaml.safe_load(path.read_text())
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        return cls(
            **{k: v for k, v in raw.items() if k in known},
            extra={k: v for k, v in raw.items() if k not in known},
        )

    def badges(self) -> list[str]:
        badges = []
        if self.quality_note:
            badges.append(self.quality_note)
        if not self.commercial_use:
            badges.append("Non-commercial licence")
        return badges

    def provenance(self) -> dict[str, Any]:
        """What gets recorded in each export's settings."""
        return {
            "id": self.id,
            "licence": self.licence,
            "source": self.source,
        }


def fetch_url(manifest: "Manifest", status=lambda msg: None) -> Path:
    """Download a ``kind: url`` model once, verified against its sha256."""
    src = manifest.source
    dest = models_dir() / manifest.id / src["filename"]
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(src["url"]) as resp, open(partial, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done, last_pct = 0, -1
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            pct = done * 100 // total if total else -1
            if pct != last_pct and pct % 5 == 0:
                status(f"Downloading {manifest.name}: {pct}%")
                last_pct = pct
    if digest.hexdigest() != src["sha256"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"{manifest.name} download failed its checksum; refusing to use it"
        )
    partial.rename(dest)
    return dest


def prefetch(manifest: "Manifest", status=lambda msg: None) -> None:
    """Download a model's files now (first-launch setup) rather than the first
    time a tool needs it, into the same models folder they're loaded from."""
    kind = manifest.source.get("kind")
    if kind == "url":
        fetch_url(manifest, status)
    elif kind == "huggingface":
        from huggingface_hub import snapshot_download

        snapshot_download(
            manifest.source["repo"],
            revision=manifest.source["revision"],
            cache_dir=models_dir(),  # where the adapter loads it from
            allow_patterns=["*.py", "*.json", "*.safetensors", "*.txt"],
        )


def best_device() -> str:
    """Where models run: an NVIDIA GPU ("cuda"), else a Mac's GPU ("mps"),
    else the CPU. RAPIDRETOUCH_DEVICE overrides it (for testing)."""
    forced = os.environ.get("RAPIDRETOUCH_DEVICE")
    if forced:
        return forced
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Registry:
    def __init__(self, manifest_dir: Path = MANIFEST_DIR):
        self.manifests = {
            m.id: m for m in (Manifest.load(p) for p in sorted(manifest_dir.glob("*.yaml")))
        }
        self._loaded: dict[str, Any] = {}
        self.status = lambda msg: None  # set by the server to report downloads
        self._acceptances_path = data_dir() / "accepted_licences.json"

    def for_task(self, task: str) -> list[Manifest]:
        return [m for m in self.manifests.values() if m.task == task]

    def default_for(self, task: str) -> Manifest:
        for m in self.for_task(task):
            if m.bundled_default:
                return m
        raise KeyError(f"no default model for task {task!r}")

    def is_accepted(self, manifest: Manifest) -> bool:
        if manifest.commercial_use:
            return True
        return manifest.id in self._read_acceptances()

    def accept_licence(self, model_id: str) -> None:
        accepted = self._read_acceptances()
        accepted.add(model_id)
        self._acceptances_path.write_text(json.dumps(sorted(accepted)))

    def _read_acceptances(self) -> set[str]:
        try:
            return set(json.loads(self._acceptances_path.read_text()))
        except FileNotFoundError:
            return set()

    def get(self, model_id: str, device: str | None = None):
        """Return a loaded adapter instance, downloading weights on first use."""
        if model_id in self._loaded:
            return self._loaded[model_id]
        manifest = self.manifests[model_id]
        if not self.is_accepted(manifest):
            raise LicenceNotAccepted(manifest)
        module = importlib.import_module(f"rapidretouch_engine.adapters.{manifest.adapter}")
        if manifest.source.get("kind") == "url":
            fetch_url(manifest, self.status)
        instance = module.Adapter(manifest, models_dir(), device or best_device())
        self._loaded[model_id] = instance
        return instance

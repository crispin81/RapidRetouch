"""JSON-lines protocol spoken to the Tauri app over stdin/stdout.

Request:  {"id": 1, "method": "open", "params": {...}}
Response: {"id": 1, "result": {...}}  or  {"id": 1, "error": {"message": ..., ...}}
Event:    {"event": "status", "message": "..."}  (unsolicited, e.g. download progress)

Anything a library prints goes to stderr so it can't corrupt the protocol stream.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import imageio, presets, system
from .registry import LicenceNotAccepted, Registry
from .tools import backdrop_smooth, dodge_burn, fabric, inpaint, mask_edit, patch, reflection, region_edit, scene, tone
from .tools import crop as crop_tool
from .tools import eyes as eye_tool
from .tools import mouth as mouth_tool
from .tools import skin as skin_tool

PREVIEW_EDGE = 2048
EXPORT_BIT_DEPTH = 16  # TIFF exports are 16-bit (JPEGs are 8-bit by nature)
# The settings record (<export>.retouch.json: every step, its settings, the
# edits and the AI models used) is off for now: it cluttered delivery folders
# (Chris, 2026-10-04). Planned instead: the same record in the image's own
# metadata (XMP in JPEGs, the description tag in TIFFs).
WRITE_SETTINGS_FILE = False
# Zooming in (see render_region): how much more than the view is rendered on
# each side (as a share of its size), how many such areas are kept for panning
# back, and the face tools' crop at full resolution and how many of its
# stages are kept (each ~200 MB at 100 MP: the four face tools' results for
# two looks, so going back and forth between sliders finds them).
ZOOM_SPARE = 0.25
FACE_WINDOW_MARGIN = 0.5  # face widths around the face outline: every face tool works within it
FACE_WINDOW_CACHE_SIZE = 10
STAGE_CACHE_SIZE = 12  # preview stages kept: the current look's, plus a before view's
# Those caches are also held to a share of the computer's memory, so a 16 GB
# laptop with a 100 MP photo doesn't swap (at least two entries are always
# kept, for going back and forth), and fewer zoomed areas are kept there.
MEMORY = system.total_memory()
STAGE_CACHE_BUDGET = int(0.04 * MEMORY)
FACE_WINDOW_CACHE_BUDGET = int(0.06 * MEMORY)
ZOOM_TILES = 4 if MEMORY >= 32 << 30 else 2
METHODS = {
    "ping",
    "models",
    "accept_licence",
    "open",
    "thumbnail",
    "forget",
    "mask",
    "mask_paint",
    "undo_mask_edit",
    "clear_mask_edits",
    "region_view",
    "region_paint",
    "undo_region_edit",
    "clear_region_edits",
    "render",
    "iris_scale",
    "remove",
    "undo_remove",
    "clear_removals",
    "render_region",
    "presets",
    "save_preset",
    "delete_preset",
    "export",
}


def _proxy(img: np.ndarray, edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, edge / max(h, w))
    if scale == 1.0:
        return img
    size = (round(w * scale), round(h * scale))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _jpeg_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(
        buf, "JPEG", quality=90
    )
    return base64.b64encode(buf.getvalue()).decode()


class Engine:
    def __init__(self, emit):
        self.emit = emit
        self.registry = Registry()
        self.registry.status = self.status
        self.path: Path | None = None
        self.image: imageio.LoadedImage | None = None
        self.preview: np.ndarray | None = None
        self.alpha: np.ndarray | None = None  # full-res subject mask, as the model made it
        self.alpha_preview_raw: np.ndarray | None = None
        self.alpha_preview: np.ndarray | None = None  # with the hand edits applied
        self.mask_edits: list[dict] = []
        # The user's corrections to each area the tools find (see region_edit).
        self.region_edits: dict[str, list[dict]] = {r: [] for r in region_edit.REGIONS}
        self.faces: list[np.ndarray] | None = None  # landmarks, as fractions of w/h
        self.mask_model: str | None = None
        # Prepared backdrop for the preview, keyed by what it depends on. Two slots,
        # so holding "before" on removals (backdrop measured from the unedited
        # photo) and releasing doesn't cost a full recompute each way.
        self.backdrop_cache: dict[tuple, backdrop_smooth.Prepared] = {}
        # Paint-to-remove strokes, applied before backdrop smoothing. The preview
        # after each stroke is kept so undo is instant.
        self.strokes: list[dict] = []
        self.filled_previews: list[np.ndarray] = []
        # Bumped on every removal or mask edit. Undo followed by a new stroke leaves
        # the stroke count unchanged, so the count alone can't key the backdrop cache.
        self.edit_version = 0
        # Bumped on mask edits only: the "without removals" backdrop depends on
        # the mask but not on removals, so it survives new removal strokes.
        self.mask_version = 0
        # Full-resolution pipeline stages, each (key, value) keyed by exactly what
        # it depends on, shared by zoomed-in detail views and export.
        self._full: dict[str, tuple] = {}
        self._eye_masks: dict[tuple, np.ndarray] = {}  # eye openings, per image size
        # Edits of photos in the film strip that aren't open right now, by path.
        # Only the edit lists are kept (cheap); pixels are recomputed on return.
        self.sessions: dict[str, dict] = {}
        # The preview's clothes mask and crease gain, each (key, value), so
        # other sliders never recompute them.
        self._preview_cache: dict[str, tuple] = {}
        self._full_found: dict[str, tuple] = {}
        # The preview before the tone step, by look: tone changes reuse it. Two
        # slots, so a hold-for-before render doesn't evict the current look.
        self._pretone: dict[str, np.ndarray] = {}
        # The preview after each subject tool, by everything it depends on (see
        # ``_staged``), least recently used first. Full precision (~34 MB
        # each): in half precision, tools that make threshold decisions turned
        # the rounding into visible differences depending on slider history.
        self._stage_cache: dict[str, np.ndarray] = {}
        self._parts: dict[str, np.ndarray] | None = None  # person-parts probabilities (see _part)
        self._acne: tuple | None = None  # (key, spot maps) found at full detail (see _acne_masks)
        self._zoom_tiles: list[tuple] = []  # (key, area, untoned pixels), see render_region
        self._face_window_cache: dict[str, np.ndarray] = {}  # face tools at full resolution, see _zoom_area
        self._iris_scale: tuple | None = None  # (colours,) for the open photo, see iris_scale
        self._exporting: float | None = None
        # Export files written in the background (see export), one at a time.
        self._writer = ThreadPoolExecutor(max_workers=1)
        self._writing = None  # progress of the export under way, 0..1

    def status(self, message: str) -> None:
        self.emit({"event": "status", "message": message})
        if self._exporting is not None:
            # During an export, each step also moves its progress bar on.
            if message.startswith("Writing"):
                self._exporting = EXPORT_WRITING
            else:
                self._exporting += (EXPORT_WRITING - self._exporting) * EXPORT_STEP
            self.emit({"event": "progress", "fraction": round(self._exporting, 3)})

    # --- methods -----------------------------------------------------------

    def ping(self) -> dict:
        import torch

        return {
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "Apple GPU"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
            else "CPU",
        }

    def models(self) -> list[dict]:
        return [
            {
                "id": m.id,
                "name": m.name,
                "task": m.task,
                "licence": m.licence,
                "licence_url": m.licence_url,
                "commercial_use": m.commercial_use,
                "default": m.bundled_default,
                "accepted": self.registry.is_accepted(m),
                "badges": m.badges(),
                "note": m.quality_note,
            }
            for m in self.registry.manifests.values()
        ]

    def accept_licence(self, model_id: str) -> dict:
        self.registry.accept_licence(model_id)
        return {"accepted": model_id}

    def open(self, path: str) -> dict:
        # Keep the current photo's edits so switching back restores them.
        if self.path is not None:
            self.sessions[str(self.path)] = {
                "strokes": self.strokes,
                "mask_edits": self.mask_edits,
                "region_edits": self.region_edits,
                "faces": self.faces,
            }
        self.status(f"Loading {Path(path).name}")
        self.path = Path(path)
        self.image = imageio.load(path)
        self.preview = _proxy(self.image.rgb, PREVIEW_EDGE)
        self.alpha = self.alpha_preview_raw = self.alpha_preview = None
        self.mask_edits = []
        self.region_edits = {r: [] for r in region_edit.REGIONS}
        self.faces = None
        self._full = {}
        self._eye_masks = {}
        self.backdrop_cache = {}
        self.strokes, self.filled_previews = [], []
        self._preview_cache, self._full_found = {}, {}
        self._pretone, self._stage_cache = {}, {}
        self._parts = None
        self._acne = None
        self._iris_scale = None
        self._zoom_tiles, self._face_window_cache = [], {}
        self.edit_version += 1
        saved = self.sessions.pop(str(self.path), None)
        if saved:
            self.mask_edits = saved["mask_edits"]
            self.region_edits = saved.get("region_edits") or {r: [] for r in region_edit.REGIONS}
            self.faces = saved["faces"]
            base = self.preview
            for i, stroke in enumerate(saved["strokes"], 1):
                self.status(f"Restoring edits ({i} of {len(saved['strokes'])})")
                base = self._apply_stroke(base, stroke)
                self.filled_previews.append(base)
            self.strokes = list(saved["strokes"])
        h, w = self.image.rgb.shape[:2]
        return {
            "width": w,
            "height": h,
            "bit_depth": self.image.bit_depth,
            "preview": _jpeg_b64(self.preview),
            "removals": len(self.strokes),
            "mask_edits": len(self.mask_edits),
            "region_edits": {r: len(v) for r, v in self.region_edits.items()},
        }

    def thumbnail(self, path: str, edge: int = 240) -> dict:
        """Quick preview (base64 JPEG) for the film strip or for scanning through
        photos, with the photo's full size so the viewer can lay it out, and a
        guess at studio backdrop vs outdoor (measured at scene.WORK_EDGE)."""
        im, (w, h) = imageio.thumbnail(path, max(edge, scene.WORK_EDGE))
        guess = scene.classify(np.asarray(im, np.float32) / 255)
        im.thumbnail((edge, edge))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85 if edge <= 400 else 90)
        return {
            "image": base64.b64encode(buf.getvalue()).decode(),
            "width": w,
            "height": h,
            "scene": guess["mode"],
        }

    def presets(self) -> list[dict]:
        return presets.list_all()

    def save_preset(self, name: str, settings: dict) -> list[dict]:
        presets.save(name, settings)
        return presets.list_all()

    def delete_preset(self, name: str) -> list[dict]:
        presets.delete(name)
        return presets.list_all()

    def forget(self, path: str) -> dict:
        """Drop the stored edits of a photo taken out of the film strip."""
        self.sessions.pop(str(path), None)
        return {"forgotten": path}

    def _ensure_mask(self, model_id: str | None = None) -> None:
        manifest = (
            self.registry.manifests[model_id]
            if model_id
            else self.registry.default_for("subject_mask")
        )
        if self.alpha is not None and self.mask_model == manifest.id:
            return
        self.status(f"Loading {manifest.name}…")
        model = self.registry.get(manifest.id)
        on_cpu = getattr(model, "device", "") == "cpu"
        self.status("Finding the subject (slower without a graphics card)…" if on_cpu else "Finding the subject…")
        self.alpha = model.predict(self.image.rgb)
        self.alpha_preview_raw = _proxy(self.alpha, PREVIEW_EDGE)
        self.alpha_preview = mask_edit.apply(self.alpha_preview_raw, self.mask_edits)
        self.mask_model = manifest.id
        self.backdrop_cache = {}

    def mask(self, model: str | None = None) -> dict:
        """The subject mask as a red overlay on the photo (red = backdrop)."""
        self._require_image()
        self._ensure_mask(model)
        return self._mask_result()

    def mask_paint(self, mode: str, points: list[list[float]], radius: float) -> dict:
        """Hand-correct the mask: mode "add" marks subject, "subtract" marks backdrop."""
        self._require_image()
        if mode not in ("add", "subtract"):
            raise ValueError(f"unknown mask mode {mode!r}")
        if not points:
            raise ValueError("empty stroke")
        self._ensure_mask()
        edit = {"mode": mode, "points": points, "radius": radius}
        self.mask_edits.append(edit)
        # Only the new stroke needs applying; undo replays from the raw mask.
        self.alpha_preview = mask_edit.apply(self.alpha_preview, [edit])
        self.edit_version += 1
        self.mask_version += 1
        return self._mask_result()

    def undo_mask_edit(self) -> dict:
        self._require_image()
        self._ensure_mask()
        if self.mask_edits:
            self.mask_edits.pop()
            self.alpha_preview = mask_edit.apply(self.alpha_preview_raw, self.mask_edits)
            self.edit_version += 1
            self.mask_version += 1
        return self._mask_result()

    def clear_mask_edits(self) -> dict:
        self._require_image()
        self._ensure_mask()
        self.mask_edits = []
        self.alpha_preview = self.alpha_preview_raw
        self.edit_version += 1
        self.mask_version += 1
        return self._mask_result()

    def _area(self, region: str) -> np.ndarray:
        """The area a Skin tab or the Clothes panel works on (0..1), at preview
        size, with the user's corrections applied."""
        rgb = self._edited_preview()
        h, w = rgb.shape[:2]
        found_key = (self.edit_version, self.mask_version)
        if region == "face":
            return skin_tool.face_area(rgb, self._ensure_faces(), self.region_edits["face"], self._part("hair"))
        self._ensure_mask()
        if region == "clothes":
            self._creases(rgb, self.alpha_preview, 0.0, self._preview_cache, found_key)
            area = self._preview_cache["clothes"][1]
            return np.zeros((h, w), np.float32) if area is None else area
        people = self._body_people(rgb, self.alpha_preview, self._preview_cache, found_key)
        area = np.zeros((h, w), np.float32)
        for person in people:
            area = np.maximum(area, cv2.resize(getattr(person, region), (w, h), interpolation=cv2.INTER_LINEAR))
        return region_edit.apply(area, self.region_edits[region], (h, w))

    def _region(self, region: str) -> str:
        if region not in region_edit.REGIONS:
            raise ValueError(f"unknown area {region!r}")
        return region

    def region_view(self, region: str) -> dict:
        """The area (face, neck, body or clothes) shown in blue over the photo."""
        self._require_image()
        region = self._region(region)
        shown = region_edit.overlay(self._edited_preview(), self._area(region))
        return {"preview": _jpeg_b64(shown), "edits": len(self.region_edits[region])}

    def region_paint(self, region: str, mode: str, points: list[list[float]], radius: float) -> dict:
        """Correct an area: mode "add" takes the brushed part in, "remove" takes it out."""
        self._require_image()
        region = self._region(region)
        if mode not in ("add", "remove"):
            raise ValueError(f"unknown mode {mode!r}")
        if not points:
            raise ValueError("empty stroke")
        self.region_edits[region] = self.region_edits[region] + [{"mode": mode, "points": points, "radius": radius}]
        return self.region_view(region)

    def undo_region_edit(self, region: str) -> dict:
        self._require_image()
        region = self._region(region)
        self.region_edits[region] = self.region_edits[region][:-1]
        return self.region_view(region)

    def clear_region_edits(self, region: str) -> dict:
        self._require_image()
        region = self._region(region)
        self.region_edits[region] = []
        return self.region_view(region)

    def _mask_result(self) -> dict:
        shown = mask_edit.overlay(self._edited_preview(), self.alpha_preview)
        return {"preview": _jpeg_b64(shown), "mask_edits": len(self.mask_edits)}

    # Every call that returns a preview takes the photo's "look": each tool's
    # settings (None leaves a tool out) and whether to include the removals.
    # Leaving one out is how the per-step before views are made.
    LOOK = {
        "backdrop": None,
        "eyes": None,
        "skin": None,
        "mouth": None,
        "model": None,
        "removals": True,
        "clothes": None,  # {"creases": 0..1}
        "dodge_burn": None,  # {"contour", "highlights", "shadows": 0..1}
        "tone": None,  # {"ev": stops, "curves": {...}}: the final grade
        # Each face panel's overall amount (0..1): its result blended back
        # toward the photo before it, like a layer's opacity.
        "opacity": None,  # {"skin", "dodge_burn", "eyes", "mouth": 0..1}
    }

    def _look(self, look: dict) -> dict:
        unknown = set(look) - set(self.LOOK)
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
        return {**self.LOOK, **look}

    def _preview(self, look: dict) -> str:
        return _jpeg_b64(self._render(self._look(look)))

    def render(self, **look) -> dict:
        """Preview of the whole pipeline with the given look."""
        self._require_image()
        return {
            "preview": self._preview(look),
            "faces": len(self.faces) if self.faces is not None else None,
        }

    def iris_scale(self) -> dict:
        """The colours Iris hue moves this photo's irises through, for the
        slider's colour bar (see eyes.iris_scale); measured once per photo."""
        self._require_image()
        if self._iris_scale is None:
            self._iris_scale = (eye_tool.iris_scale(self.preview, self._ensure_faces()),)
        return {"colours": self._iris_scale[0]}

    def remove(
        self,
        points: list[list[float]],
        radius: float,
        kind: str = "fill",
        strength: float = 1.0,
        offset: list[float] | None = None,
        **look,
    ) -> dict:
        """Apply one Remove-panel edit, then re-render the preview. ``kind`` is
        "fill" (LaMa removal), "reflection" (glasses reflection, alpha) or
        "patch" (``points`` is the lasso outline, ``offset`` where the texture
        comes from, both as fractions of width/height)."""
        self._require_image()
        self._look(look)
        if not points:
            raise ValueError("empty stroke")
        if kind not in ("fill", "reflection", "patch"):
            raise ValueError(f"unknown stroke kind {kind!r}")
        stroke = {"kind": kind, "points": points, "radius": radius}
        if kind == "reflection":
            stroke["strength"] = float(strength)
        if kind == "patch":
            if len(points) < 3 or offset is None:
                raise ValueError("a patch needs an outline and an offset")
            stroke = {"kind": kind, "points": points, "offset": [float(v) for v in offset]}
        base = self._edited_preview()
        self.status({"reflection": "Removing reflection", "patch": "Patching"}.get(kind, "Removing"))
        self.filled_previews.append(self._apply_stroke(base, stroke))
        self.strokes.append(stroke)
        self.edit_version += 1
        return {"preview": self._preview(look), "removals": len(self.strokes)}

    def undo_remove(self, **look) -> dict:
        self._require_image()
        if self.strokes:
            self.strokes.pop()
            self.filled_previews.pop()
            self.edit_version += 1
        return {"preview": self._preview(look), "removals": len(self.strokes)}

    def clear_removals(self, **look) -> dict:
        self._require_image()
        self.strokes, self.filled_previews = [], []
        self.edit_version += 1
        return {"preview": self._preview(look), "removals": 0}

    def _crease_amount(self, look: dict) -> float:
        return float((look["clothes"] or {}).get("creases", 0))

    def _creases(self, rgb: np.ndarray, alpha: np.ndarray, amount: float, cache: dict, key) -> np.ndarray | None:
        """Crease-flattening gain for ``rgb`` (the photo after removals): the
        clothes are found automatically (subject minus skin and head). The mask
        and the gain are cached in ``cache`` under ``key``."""
        faces = self._ensure_faces()
        h, w = rgb.shape[:2]
        faces_px = [f[:, :2] * [w, h] for f in faces]
        key = (key, json.dumps(self.region_edits["clothes"]))
        hit = cache.get("clothes")
        if not hit or hit[0] != key:
            self.status("Finding the clothes")
            cache["clothes"] = (
                key,
                fabric.clothes_mask(rgb, alpha, faces, self._clothes(), self.region_edits["clothes"]),
            )
        mask = cache["clothes"][1]
        if mask is None:
            return None
        hit = cache.get("creases")
        if not hit or hit[0] != key:
            self.status("Smoothing creases")
            fw = max((skin_tool.face_width(f) for f in faces_px), default=None)
            cache["creases"] = (key, fabric.measure(rgb, mask, fw))
        corr = cache["creases"][1]
        return corr.gain(amount) if corr else None

    def _part(self, name: str) -> np.ndarray:
        """Where the person-parts model sees ``name`` ("clothes", "hair", ...),
        0..1 on a copy up to 1024 px, found once per photo from the original."""
        if self._parts is None:
            manifest = self.registry.default_for("person_parts")
            self.status(f"Loading {manifest.name}…")
            model = self.registry.get(manifest.id)
            self.status("Telling skin from hair and clothes")
            self._parts = model.predict(self.image.rgb)
        return self._parts[name]

    def _acne_masks(self, amount: float, edits: list[dict]) -> list:
        """Acne's spot maps found on the full-resolution photo, remembered for
        the last setting (see skin.acne_masks)."""
        key = json.dumps([amount, edits])
        if self._acne is None or self._acne[0] != key:
            self.status("Finding spots")
            self._acne = (key, skin_tool.acne_masks(self.image.rgb, self._ensure_faces(), amount, edits, self._part("hair")))
        return self._acne[1]

    def _clothes(self) -> np.ndarray:
        return self._part("clothes")

    def _body_people(self, rgb: np.ndarray, alpha: np.ndarray, cache: dict, key) -> list:
        """Each person's neck and body skin, found in ``rgb`` (the photo after
        removals) inside the subject mask; cached in ``cache`` under ``key``."""
        hit = cache.get("people")
        if not hit or hit[0] != key:
            self.status("Finding neck and body skin")
            cache["people"] = (key, skin_tool.body_regions(rgb, self._ensure_faces(), alpha, self._clothes()))
        return cache["people"][1]

    def _ensure_faces(self) -> list[np.ndarray]:
        if self.faces is None:
            manifest = self.registry.default_for("face_landmarks")
            self.status(f"Loading {manifest.name}…")
            model = self.registry.get(manifest.id)
            self.status("Finding faces")
            self.faces = model.predict(self.image.rgb)
        return self.faces

    def _apply_stroke(self, rgb: np.ndarray, stroke: dict) -> np.ndarray:
        """One Remove-panel stroke, at whatever resolution ``rgb`` is."""
        h, w = rgb.shape[:2]
        if stroke.get("kind") == "patch":
            return patch.apply(rgb, stroke["points"], stroke["offset"])
        if stroke.get("kind", "fill") == "reflection":
            mask = inpaint.stroke_mask((h, w), stroke["points"], stroke["radius"], grow=0)
            return reflection.remove(
                rgb,
                mask,
                stroke["radius"] * max(h, w),
                stroke.get("strength", 1.0),
                self._eye_openings((h, w)),
            )
        mask = inpaint.stroke_mask((h, w), stroke["points"], stroke["radius"])
        return inpaint.fill(rgb, mask, self._inpaint_model(), self.status)

    def _eye_openings(self, shape: tuple[int, int]) -> np.ndarray:
        if shape not in self._eye_masks:
            self._eye_masks[shape] = eye_tool.eye_openings(shape, self._ensure_faces())
        return self._eye_masks[shape]

    def _inpaint_model(self):
        manifest = self.registry.default_for("inpaint")
        self.status(f"Loading {manifest.name}…")
        return self.registry.get(manifest.id)

    def _edited_preview(self) -> np.ndarray:
        return self.filled_previews[-1] if self.filled_previews else self.preview

    def _face_stages(self, look: dict, people, frame=None) -> list[tuple[str, dict, object]]:
        """Skin, neck & body, dodge & burn, eyes and mouth, in that order: the
        ones this look uses, each (name, settings, run). ``run(rgb, steps)``
        works at whatever resolution ``rgb`` is and, given ``steps``, records
        what it did there (for the export). ``people()`` gives the neck and
        body skin (see ``_body_people``); it's only called when a Neck or Body
        slider is set.

        ``frame``: (box, full_shape) to run the face tools on just ``box`` of
        the full-resolution photo (zooming in; see ``_face_window``) instead of
        the whole of it. Faces, the hair map and area edits are moved into the
        box's coordinates; Neck & Body aren't available."""
        landmarks = lambda: [self.registry.default_for("face_landmarks").provenance()]  # noqa: E731
        opacities = look.get("opacity") or {}
        opacity = lambda group: float(np.clip(opacities.get(group, 1.0), 0, 1))  # noqa: E731
        if frame is None:
            faces = self._ensure_faces
            hair = lambda: self._part("hair")  # noqa: E731
            framed_edits = lambda edits: edits  # noqa: E731
        else:
            box, full_shape = frame
            faces = lambda: _frame_faces(self._ensure_faces(), box, full_shape)  # noqa: E731
            hair = lambda: _frame_map(self._part("hair"), box, full_shape)  # noqa: E731
            framed_edits = lambda edits: _frame_edits(edits, box, full_shape)  # noqa: E731
        stages = []
        skin = look["skin"] or {}
        face_p = skin_tool.RegionParams.from_dict(skin.get("face"))
        if not face_p.is_noop() and opacity("skin") > 0:
            face_edits = list(self.region_edits["face"])

            def run(rgb, steps):
                # Smaller than the photo (the preview): Acne's spots are found
                # at full detail, so the preview and the zoomed-in view agree.
                acne = None
                if face_p.acne > 0 and frame is None and rgb.shape[1] < self.image.rgb.shape[1]:
                    acne = self._acne_masks(face_p.acne, face_edits)
                rgb = skin_tool.apply(rgb, faces(), skin.get("face"), framed_edits(face_edits), hair(), acne)
                if steps is not None:
                    steps.append({"tool": "skin", "face": asdict(face_p), "area_edits": face_edits,
                                  "faces": len(self.faces), "models": landmarks()})
                return rgb
            stages.append(("skin", {"params": asdict(face_p), "edits": face_edits, "opacity": opacity("skin")}, run))
        neck_p = skin_tool.RegionParams.from_dict(skin.get("neck"))
        body_p = skin_tool.RegionParams.from_dict(skin.get("body"))
        if frame is None and not (neck_p.is_noop() and body_p.is_noop()) and opacity("skin") > 0:
            body_edits = {r: list(self.region_edits[r]) for r in ("neck", "body")}

            def run(rgb, steps):
                rgb = skin_tool.apply_body(rgb, people(), skin.get("neck"), skin.get("body"), body_edits)
                if steps is not None:
                    steps.append({
                        "tool": "skin_body",
                        "neck": asdict(neck_p),
                        "body": asdict(body_p),
                        "area_edits": body_edits,
                        "faces": len(self.faces),
                        "models": [self.registry.manifests[self.mask_model].provenance(), *landmarks()],
                    })
                return rgb
            stages.append(("skin_body", {"neck": asdict(neck_p), "body": asdict(body_p), "edits": body_edits,
                                         "opacity": opacity("skin")}, run))
        db_p = dodge_burn.Params.from_dict(look["dodge_burn"])
        if not db_p.is_noop() and opacity("dodge_burn") > 0:
            def run(rgb, steps):
                rgb = dodge_burn.apply(rgb, faces(), db_p, hair())
                if steps is not None:
                    steps.append({"tool": "dodge_burn", "params": asdict(db_p), "faces": len(self.faces), "models": landmarks()})
                return rgb
            stages.append(("dodge_burn", {**asdict(db_p), "opacity": opacity("dodge_burn")}, run))
        eyes_p = eye_tool.Params.from_dict(look["eyes"] or {})
        if look["eyes"] is not None and not eyes_p.is_noop() and opacity("eyes") > 0:
            def run(rgb, steps):
                rgb = eye_tool.apply(rgb, faces(), eyes_p)
                if steps is not None:
                    steps.append({"tool": "eyes", "params": asdict(eyes_p), "faces": len(self.faces), "models": landmarks()})
                return rgb
            stages.append(("eyes", {**asdict(eyes_p), "opacity": opacity("eyes")}, run))
        mouth_p = mouth_tool.Params.from_dict(look["mouth"])
        if look["mouth"] is not None and not mouth_p.is_noop() and opacity("mouth") > 0:
            def run(rgb, steps):
                rgb = mouth_tool.apply(rgb, faces(), mouth_p)
                if steps is not None:
                    steps.append({"tool": "mouth", "params": asdict(mouth_p), "faces": len(self.faces), "models": landmarks()})
                return rgb
            stages.append(("mouth", {**asdict(mouth_p), "opacity": opacity("mouth")}, run))
        return stages

    def _face_tools(self, rgb: np.ndarray, look: dict, people, steps: list[dict] | None = None) -> np.ndarray:
        """Every face and body tool this look uses, in order, uncached (the
        full-resolution pipeline). Without Neck & Body, the face tools run on
        a crop around the faces (as zooming in does): at 100 MP every tool
        otherwise copied the whole photo. Returns a new image."""
        body = any(
            not skin_tool.RegionParams.from_dict((look["skin"] or {}).get(r)).is_noop() for r in ("neck", "body")
        )
        face = None if body else self._face_window()
        if face is None:
            for _name, settings, run in self._face_stages(look, people):
                op = settings.get("opacity", 1.0)
                out = run(rgb, steps)
                if op < 1 and steps:
                    steps[-1]["opacity"] = op
                rgb = _blend(rgb, out, op)
            return rgb
        x0, y0, x1, y1 = face
        stages = self._face_stages(look, None, frame=(face, rgb.shape[:2]))
        if not stages:
            return rgb
        crop = np.array(rgb[y0:y1, x0:x1], np.float32)
        for _name, settings, run in stages:
            op = settings.get("opacity", 1.0)
            out = run(crop, steps)
            if op < 1 and steps:
                steps[-1]["opacity"] = op
            crop = _blend(crop, out, op)
        rgb = rgb.copy()
        rgb[y0:y1, x0:x1] = crop
        return rgb

    def _staged(
        self, rgb, base_key, stages: list[tuple[str, dict, object]], cache=None, size=STAGE_CACHE_SIZE, budget=None
    ) -> np.ndarray:
        """Run ``stages`` on ``rgb`` (a callable, only called if needed),
        reusing the output of every stage whose settings, and whose earlier
        stages' settings, are unchanged. Moving one tool's slider then reruns
        only that tool and the ones after it."""
        if cache is None:
            cache = self._stage_cache
        # A stage's opacity only blends its own result with its input, so its
        # unblended result is kept under a key without it: moving an Opacity
        # slider only re-blends.
        entries, key = [], base_key
        for name, settings, _run in stages:
            op = float(settings.get("opacity", 1.0))
            plain = {k: v for k, v in settings.items() if k != "opacity"}
            raw_key = json.dumps([key, name, plain], sort_keys=True)
            key = raw_key if op >= 1 else json.dumps([raw_key, op])
            entries.append((raw_key, key, op))

        def take(k):
            hit = cache.pop(k, None)
            if hit is not None:
                cache[k] = hit  # most recently used
            return hit

        def keep(k, value):
            cache[k] = value
            # Oldest first, down to ``size`` entries and the memory budget.
            while len(cache) > size or (
                len(cache) > 2 and sum(v.nbytes for v in cache.values()) > (budget or STAGE_CACHE_BUDGET)
            ):
                cache.pop(next(iter(cache)))

        start, img = 0, None
        for i in range(len(stages) - 1, -1, -1):
            hit = take(entries[i][1])
            if hit is not None:
                start, img = i + 1, hit
                break
        if img is None:
            img = rgb()
        for i in range(start, len(stages)):
            raw_key, key, op = entries[i]
            raw = take(raw_key) if op < 1 else None
            if raw is None:
                raw = stages[i][2](img, None)
                keep(raw_key, raw)
            img = _blend(img, raw, op)
            if op < 1:
                keep(key, img)
        return img

    def _render(self, look: dict) -> np.ndarray:
        """Preview of the full pipeline: removals, backdrop smoothing, skin,
        neck & body, dodge & burn, eyes, mouth, clothes creases, then the tone
        (EV, curves).

        The subject tools come after the backdrop: they only touch the subject,
        where the backdrop step changes nothing, so their sliders never force a
        backdrop recompute. Dodge & burn follows skin smoothing, so smoothing
        never flattens the shaping. Each subject tool's output is kept (see
        ``_staged``), so a slider reruns only its own tool and those after it.

        Tone is last and per-pixel, so the image before it is kept: moving the
        exposure or the curve only re-applies the tone.
        """
        key = json.dumps(
            [{**look, "tone": None}, self.edit_version, self.mask_version, self.region_edits], sort_keys=True
        )
        img = self._pretone.pop(key, None)  # re-inserted below: LRU order
        if img is None:
            img = self._render_retouch(look)
            if len(self._pretone) >= 2:
                self._pretone.pop(next(iter(self._pretone)))
        self._pretone[key] = img
        return tone.apply(img, look["tone"])

    def _render_retouch(self, look: dict) -> np.ndarray:
        """The preview up to (not including) the tone."""
        found_key = (self.edit_version if look["removals"] else -1, self.mask_version)

        def base():
            return self._edited_preview() if look["removals"] else self.preview

        def backdrop():
            return self._render_backdrop(look["backdrop"], look["model"], look["removals"])

        def people():
            self._ensure_mask()
            return self._body_people(base(), self.alpha_preview, self._preview_cache, found_key)

        stages = self._face_stages(look, people)
        # Creases last: they only touch the clothes and the other tools only
        # the skin, so the order doesn't show, and the Creases slider then
        # never reruns them.
        amount = self._crease_amount(look)
        if amount > 0:
            def creases(rgb, _steps):
                self._ensure_mask()
                gain = self._creases(base(), self.alpha_preview, amount, self._preview_cache, found_key)
                return fabric.apply_gain(rgb, gain)
            stages.append(("creases", {"amount": amount}, creases))
        # Everything the stages' input depends on: the backdrop step's settings
        # and the edits (removals, mask).
        base_key = json.dumps(
            [look["backdrop"], look["model"], look["removals"], self.edit_version, self.mask_version], sort_keys=True
        )
        return self._staged(backdrop, base_key, stages)

    def _render_backdrop(
        self, backdrop: dict | None, model: str | None, removals: bool = True
    ) -> np.ndarray:
        base = self._edited_preview() if removals else self.preview
        if backdrop is None:
            return base
        self._ensure_mask(model)
        p = backdrop_smooth.Params.from_dict(backdrop)
        # Removals change the pixels the backdrop is measured from, so they're
        # part of the key; the slider-only params still just recompose.
        version = self.edit_version if removals else self.mask_version
        key = (backdrop_smooth.prepare_key(p), version, removals)
        prep = self.backdrop_cache.pop(key, None)  # re-inserted below: LRU order
        if prep is None:
            self.status("Smoothing backdrop")
            prep = backdrop_smooth.prepare(base, self.alpha_preview, p)
            if len(self.backdrop_cache) >= 2:
                self.backdrop_cache.pop(next(iter(self.backdrop_cache)))
        self.backdrop_cache[key] = prep
        return backdrop_smooth.compose(prep, p)

    def render_region(self, region: list[float], scale: float, **look) -> dict:
        """A crop of the full-resolution result, for zoomed-in viewing.

        ``region`` is (x0, y0, x1, y1) in full-resolution pixels; ``scale`` is the
        output size relative to full resolution (<= 1: never send more pixels than
        the screen shows). Only the area in view (and a little around it) is
        rendered, and the last few areas are kept, so panning back only crops
        and encodes (see ``_zoom_area``)."""
        self._require_image()
        look = self._look(look)
        h, w = self.image.rgb.shape[:2]
        x0, y0 = max(0, int(region[0])), max(0, int(region[1]))
        x1, y1 = min(w, int(np.ceil(region[2]))), min(h, int(np.ceil(region[3])))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("empty region")
        # Cached before the tone: the tone is per-pixel, so it's applied to just
        # the crop, and moving the exposure or curve never re-renders. A little
        # more than the view is rendered, so small pans are only a crop.
        untoned = {**look, "tone": None}
        key = json.dumps(
            [untoned, self.strokes, self.mask_edits, self.region_edits, self.mask_model], sort_keys=True
        )
        hit = next(
            (t for t in self._zoom_tiles if t[0] == key and t[1][0] <= x0 and t[1][1] <= y0 and t[1][2] >= x1 and t[1][3] >= y1),
            None,
        )
        if hit is None:
            self.status("Rendering full-resolution detail")
            px, py = round((x1 - x0) * ZOOM_SPARE), round((y1 - y0) * ZOOM_SPARE)
            area = (max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py))
            hit = (key, area, self._zoom_area(untoned, area))
            self._zoom_tiles = [t for t in self._zoom_tiles if t[0] == key][-(ZOOM_TILES - 1):] + [hit]
        _key, (ax0, ay0, _ax1, _ay1), pixels = hit
        crop = pixels[y0 - ay0 : y1 - ay0, x0 - ax0 : x1 - ax0]
        scale = min(1.0, max(0.01, scale))
        if scale < 1:
            size = (max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale)))
            crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        crop = tone.apply(crop, look["tone"])
        return {"image": _jpeg_b64(crop), "region": [x0, y0, x1, y1]}

    def export(
        self, path: str, crop: dict | None = None, long_edge: int | None = None, background: bool = False, **look
    ) -> dict:
        """Export the retouched photo at full resolution, colour profile
        embedded: a 16-bit TIFF (lossless, the highest quality for further
        editing) or a maximum-quality JPEG, by the file's extension. Any other
        name gets ".tif". ``long_edge``: scale down (after the crop) so the
        longer side is this many pixels, the other in proportion; never up.

        ``background``: write the file on a thread of its own and return as
        soon as the image is ready, so a batch retouches the next photo while
        this one is written (1-2 s at 100 MP); a "written" event follows with
        the path (and an error, if writing failed). One file is written at a
        time, which bounds the memory held."""
        self._require_image()
        out = Path(path)
        if out.suffix.lower() not in (".tif", ".tiff", ".jpg", ".jpeg"):
            out = out.with_name(out.name + ".tif")
        if out.resolve() == self.path.resolve():
            raise ValueError("refusing to overwrite the original image")
        look = {**self._look(look), "removals": True}
        self._exporting = 0.0
        try:
            rgb, steps = self._full_pipeline(look)
            if not crop_tool.is_noop(crop):
                rgb = crop_tool.apply(rgb, crop)
                steps.append({"tool": "crop", "params": {**crop_tool.IDENTITY, **crop}})
            if long_edge:
                h, w = rgb.shape[:2]
                scale = int(long_edge) / max(h, w)
                if scale < 1:
                    size = (max(1, round(w * scale)), max(1, round(h * scale)))
                    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
                    steps.append({"tool": "resize", "long_edge": int(long_edge), "size": list(size)})
            self.status(f"Writing {out.name}")
            # Untagged originals are treated as sRGB throughout, so say so in the file.
            icc = self.image.icc or imageio._srgb_icc()
            if background:
                if self._writing is not None:
                    self._writing.result()  # the previous file first: one at a time

                def write(out=out, rgb=rgb, icc=icc):
                    try:
                        imageio.save(out, rgb, EXPORT_BIT_DEPTH, icc)
                        self.emit({"event": "written", "path": str(out)})
                    except Exception as e:  # reported to the app, not lost on this thread
                        traceback.print_exc()
                        self.emit({"event": "written", "path": str(out), "error": str(e)})

                self._writing = self._writer.submit(write)
            else:
                imageio.save(out, rgb, EXPORT_BIT_DEPTH, icc)
        finally:
            self._exporting = None
        if not WRITE_SETTINGS_FILE:
            return {"path": str(out), "settings": None, "pending": background}
        settings = {"source": str(self.path), "steps": steps}
        sidecar = out.with_name(out.name + ".retouch.json")
        sidecar.write_text(json.dumps(settings, indent=2))
        return {"path": str(out), "settings": str(sidecar), "pending": background}

    def _cached(self, stage: str, key, compute):
        hit = self._full.get(stage)
        if hit and hit[0] == key:
            return hit[1]
        value = compute()
        self._full[stage] = (key, value)
        return value

    def _edited_full(self, look: dict) -> tuple[np.ndarray, dict | None, str | None]:
        """The full-resolution photo with the Remove/Patch strokes applied (if
        the look includes them), the step to record, and a key for it."""
        if not (look["removals"] and self.strokes):
            return self.image.rgb, None, None
        # Strokes only ever get appended (or undone), so if the cached fills
        # are a prefix of the current strokes, only the new ones need filling.
        done, out = [], self.image.rgb
        hit = self._full.get("edited")
        if hit and hit[0] == self.strokes[: len(hit[0])]:
            done, out = hit[0], hit[1]
        for i, stroke in enumerate(self.strokes[len(done):], len(done) + 1):
            self.status(f"Removing at full resolution ({i} of {len(self.strokes)})")
            out = self._apply_stroke(out, stroke)
        self._full["edited"] = (list(self.strokes), out)
        kinds = {s.get("kind", "fill") for s in self.strokes}
        models = []
        if "fill" in kinds:
            models.append(self.registry.default_for("inpaint").provenance())
        if "reflection" in kinds:  # eye openings come from the face landmarks
            models.append(self.registry.default_for("face_landmarks").provenance())
        return out, {"tool": "remove", "strokes": self.strokes, "models": models}, json.dumps(self.strokes)

    def _alpha_full(self) -> np.ndarray:
        """The subject mask at full resolution, with the hand edits (cached)."""
        def compute():
            if not self.mask_edits:
                return self.alpha
            self.status("Applying mask edits at full resolution")
            return mask_edit.apply(self.alpha, self.mask_edits)
        return self._cached("alpha", json.dumps([self.mask_model, self.mask_edits]), compute)

    def _lighting(self, p, edited, alpha, removal_key, mask_key) -> "backdrop_smooth.Lighting":
        """The backdrop's whole-photo lighting and grain at full resolution
        (see backdrop_smooth.lighting), shared by zooming in and export."""
        def compute():
            self.status("Measuring the backdrop at full resolution")
            return backdrop_smooth.lighting(edited, alpha, p)
        return self._cached("lighting", (backdrop_smooth.prepare_key(p), removal_key, mask_key), compute)

    def _face_window(self) -> tuple[int, int, int, int] | None:
        """The part of the full-resolution photo every face tool works within
        (all faces), or None if there's no face to retouch."""
        h, w = self.image.rgb.shape[:2]
        lo, hi = [], []
        for face in self._ensure_faces():
            lm = face[:, :2] * np.array([w, h], np.float32)
            fw = skin_tool.face_width(lm)
            if fw < 40:
                continue
            pts = lm[skin_tool.FACE_OVAL]
            lo.append(pts.min(0) - FACE_WINDOW_MARGIN * fw)
            hi.append(pts.max(0) + FACE_WINDOW_MARGIN * fw)
        if not lo:
            return None
        x0, y0 = np.maximum(np.floor(np.min(lo, 0)), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(np.max(hi, 0)), [w, h]).astype(int)
        return int(x0), int(y0), int(x1), int(y1)

    def _zoom_area(self, look: dict, box: tuple[int, int, int, int]) -> np.ndarray:
        """``box`` of the full-resolution result, before the tone, working on
        as little of the photo as possible: the backdrop for just that area,
        and the face tools only if a face is in it, on a crop around the faces
        (each tool's result kept, so moving one slider reruns only that tool
        and the ones after it). Neck & Body and Creases work on the whole
        person, so with those the whole photo is rendered (and kept)."""
        x0, y0, x1, y1 = box
        if self._crease_amount(look) > 0 or any(
            not skin_tool.RegionParams.from_dict((look["skin"] or {}).get(r)).is_noop() for r in ("neck", "body")
        ):
            key = json.dumps([look, self.strokes, self.mask_edits, self.region_edits, self.mask_model], sort_keys=True)
            full = self._cached("result", key, lambda: self._full_pipeline(look)[0])
            return full[y0:y1, x0:x1]
        edited, _step, removal_key = self._edited_full(look)
        mask_key = json.dumps([self.mask_model, self.mask_edits])
        backdrop = look["backdrop"]

        def smoothed(area):
            ax0, ay0, ax1, ay1 = area
            if backdrop is None:
                return edited[ay0:ay1, ax0:ax1]
            self._ensure_mask(look["model"])
            p = backdrop_smooth.Params.from_dict(backdrop)
            alpha = self._alpha_full()
            light = self._lighting(p, edited, alpha, removal_key, mask_key)
            return backdrop_smooth.smooth_area(edited, alpha, p, light, area)

        out = np.array(smoothed(box), np.float32)
        face = self._face_window() if any(look[k] for k in ("skin", "dodge_burn", "eyes", "mouth")) else None
        if face is not None:
            ix0, iy0 = max(x0, face[0]), max(y0, face[1])
            ix1, iy1 = min(x1, face[2]), min(y1, face[3])
            stages = self._face_stages(look, None, frame=(face, edited.shape[:2]))
            if stages and ix1 > ix0 and iy1 > iy0:
                self.status("Retouching the face at full resolution")
                base_key = json.dumps([backdrop, removal_key, mask_key, face], sort_keys=True)
                faced = self._staged(lambda: np.array(smoothed(face), np.float32), base_key, stages,
                                     self._face_window_cache, FACE_WINDOW_CACHE_SIZE, FACE_WINDOW_CACHE_BUDGET)
                out[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = faced[iy0 - face[1] : iy1 - face[1], ix0 - face[0] : ix1 - face[0]]
        return out

    def _full_pipeline(self, look: dict) -> tuple[np.ndarray, list[dict]]:
        """The whole pipeline at full resolution, in the preview's order.
        Returns the image and the steps to record in the export's settings."""
        steps: list[dict] = []
        backdrop, model = look["backdrop"], look["model"]
        rgb, removal_step, removal_key = self._edited_full(look)
        if removal_step:
            steps.append(removal_step)
        edited = rgb
        mask_key = json.dumps([self.mask_model, self.mask_edits])

        def edited_alpha():
            return self._alpha_full()

        if backdrop is not None:
            self._ensure_mask(model)
            p = backdrop_smooth.Params.from_dict(backdrop)
            alpha = self._cached("alpha", mask_key, edited_alpha)

            light = self._lighting(p, edited, alpha, removal_key, mask_key)
            self.status("Smoothing backdrop at full resolution")
            # Not kept: at 100 MP it's several GB, and zooming in has its own path.
            rgb = backdrop_smooth.compose(backdrop_smooth.prepare(rgb, alpha, p, light), p)
            steps.append(
                {
                    "tool": "backdrop_smooth",
                    "params": asdict(p),
                    "mask_edits": self.mask_edits,
                    "models": [self.registry.manifests[self.mask_model].provenance()],
                }
            )

        def people():
            self._ensure_mask(model)
            key = json.dumps([self.mask_model, self.mask_edits])  # the mask may only now be loaded
            alpha = self._cached("alpha", key, edited_alpha)
            return self._body_people(edited, alpha, self._full_found, (removal_key, key))

        if any(look[k] for k in ("skin", "dodge_burn", "eyes", "mouth")):
            self.status("Retouching the face at full resolution")
        rgb = self._face_tools(rgb, look, people, steps)

        # Creases last, as in the preview (see ``_render_retouch``).
        amount = self._crease_amount(look)
        if amount > 0:
            self._ensure_mask(model)
            key = json.dumps([self.mask_model, self.mask_edits])
            alpha = self._cached("alpha", key, edited_alpha)
            gain = self._creases(edited, alpha, amount, self._full_found, (removal_key, key))
            rgb = fabric.apply_gain(rgb, gain)
            steps.append(
                {
                    "tool": "clothes_creases",
                    "amount": amount,
                    "area_edits": self.region_edits["clothes"],
                    "models": [
                        self.registry.manifests[self.mask_model].provenance(),
                        self.registry.default_for("face_landmarks").provenance(),
                    ],
                }
            )

        if not tone.is_noop(look["tone"]):
            rgb = tone.apply(rgb, look["tone"])
            steps.append({"tool": "tone", "params": look["tone"]})

        return rgb, steps

    def _require_image(self) -> None:
        if self.image is None:
            raise ValueError("no image open")


def serve() -> None:
    # Keep a private handle on the real stdout for protocol messages, then point
    # fd 1 at stderr so stray prints from libraries can't corrupt the stream.
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    lock = threading.Lock()

    def emit(msg: dict) -> None:
        with lock:
            proto.write(json.dumps(msg) + "\n")
            proto.flush()

    engine = Engine(emit)
    # Thumbnails only read the file, so they're made on a thread of their own:
    # loading a whole shoot into the film strip then never holds up the photo
    # being edited (each is ~0.3 s of RAW decoding, much of it outside the GIL).
    # One thread: a second made thumbnails faster but slowed renders by ~0.1 s.
    background = ThreadPoolExecutor(max_workers=1)
    emit({"event": "ready"})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            emit({"id": None, "error": {"message": str(e)}})
            continue
        if req.get("method") in BACKGROUND:
            background.submit(_handle, engine, req, emit)
        else:
            _handle(engine, req, emit)


# Export progress: which steps run depends on the photo and its settings, so
# each step moves the bar this share of the way to EXPORT_WRITING, and writing
# the file takes it the rest of the way to the end.
EXPORT_STEP = 0.3
EXPORT_WRITING = 0.9

BACKGROUND = {"thumbnail"}  # methods that touch no engine state


def _blend(before: np.ndarray, after: np.ndarray, opacity: float) -> np.ndarray:
    """``after`` at ``opacity`` over ``before``, as a layer's opacity."""
    if opacity >= 1:
        return after
    return before + np.float32(opacity) * (after - before)


def _frame_faces(faces: list[np.ndarray], box, full_shape) -> list[np.ndarray]:
    """Face landmarks (fractions of the whole photo) as fractions of ``box``."""
    h, w = full_shape
    x0, y0, x1, y1 = box
    out = []
    for f in faces:
        g = np.array(f, np.float64)
        g[:, 0] = (g[:, 0] * w - x0) / (x1 - x0)
        g[:, 1] = (g[:, 1] * h - y0) / (y1 - y0)
        out.append(g.astype(np.float32))
    return out


def _frame_map(m: np.ndarray, box, full_shape) -> np.ndarray:
    """A map covering the whole photo (any size), cropped to ``box`` at the same detail."""
    h, w = full_shape
    x0, y0, x1, y1 = box
    size = (max(1, round((y1 - y0) * m.shape[0] / h)), max(1, round((x1 - x0) * m.shape[1] / w)))
    return skin_tool._map_into(m, (0, 0, 1, 1), box, full_shape, size)


def _frame_edits(edits: list[dict], box, full_shape) -> list[dict]:
    """Area edits (fractions of the whole photo) as fractions of ``box``."""
    h, w = full_shape
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    return [
        {
            **e,
            "points": [((x * w - x0) / bw, (y * h - y0) / bh) for x, y in e["points"]],
            "radius": e["radius"] * max(h, w) / max(bw, bh),
        }
        for e in edits
    ]


def _handle(engine: Engine, req: dict, emit) -> None:
    req_id = req.get("id")
    try:
        method = req["method"]
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        result = getattr(engine, method)(**req.get("params", {}))
        emit({"id": req_id, "result": result})
    except LicenceNotAccepted as e:
        emit(
            {
                "id": req_id,
                "error": {
                    "message": str(e),
                    "kind": "licence",
                    "model": e.manifest.id,
                    "licence": e.manifest.licence,
                    "licence_url": e.manifest.licence_url,
                },
            }
        )
    except Exception as e:
        traceback.print_exc()
        emit({"id": req_id, "error": {"message": str(e)}})

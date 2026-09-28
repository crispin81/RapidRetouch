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
import traceback
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import imageio, presets
from .registry import LicenceNotAccepted, Registry
from .tools import backdrop_smooth, dodge_burn, fabric, inpaint, mask_edit, patch, reflection, scene, tone
from .tools import eyes as eye_tool
from .tools import mouth as mouth_tool
from .tools import skin as skin_tool

PREVIEW_EDGE = 2048
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
    "render",
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
        self._full_crease: dict[str, tuple] = {}
        # The preview before the tone step, by look: tone changes reuse it. Two
        # slots, so a hold-for-before render doesn't evict the current look.
        self._pretone: dict[str, np.ndarray] = {}

    def status(self, message: str) -> None:
        self.emit({"event": "status", "message": message})

    # --- methods -----------------------------------------------------------

    def ping(self) -> dict:
        import torch

        return {
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
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
                "faces": self.faces,
            }
        self.status(f"Loading {Path(path).name}")
        self.path = Path(path)
        self.image = imageio.load(path)
        self.preview = _proxy(self.image.rgb, PREVIEW_EDGE)
        self.alpha = self.alpha_preview_raw = self.alpha_preview = None
        self.mask_edits = []
        self.faces = None
        self._full = {}
        self._eye_masks = {}
        self.backdrop_cache = {}
        self.strokes, self.filled_previews = [], []
        self._preview_cache, self._full_crease = {}, {}
        self._pretone = {}
        self.edit_version += 1
        saved = self.sessions.pop(str(self.path), None)
        if saved:
            self.mask_edits = saved["mask_edits"]
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
        self.status(f"Loading {manifest.name} (first run downloads it)")
        model = self.registry.get(manifest.id)
        self.status("Finding the subject")
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
        "dodge_burn": None,  # {"amount": 0..1}
        "tone": None,  # {"ev": stops, "curves": {...}}: the final grade
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
        hit = cache.get("clothes")
        if not hit or hit[0] != key:
            self.status("Finding the clothes")
            cache["clothes"] = (key, fabric.clothes_mask(rgb, alpha, faces_px))
        mask = cache["clothes"][1]
        if mask is None:
            return None
        hit = cache.get("creases")
        if not hit or hit[0] != (key, amount):
            self.status("Smoothing creases")
            fw = max((skin_tool.face_width(f) for f in faces_px), default=None)
            cache["creases"] = ((key, amount), fabric.gain_map(rgb, amount * mask, fw))
        return cache["creases"][1]

    def _ensure_faces(self) -> list[np.ndarray]:
        if self.faces is None:
            manifest = self.registry.default_for("face_landmarks")
            self.status(f"Loading {manifest.name} (first run downloads it)")
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
        self.status(f"Loading {manifest.name} (first run downloads it)")
        return self.registry.get(manifest.id)

    def _edited_preview(self) -> np.ndarray:
        return self.filled_previews[-1] if self.filled_previews else self.preview

    def _face_tools(self, rgb: np.ndarray, look: dict, steps: list[dict] | None = None) -> np.ndarray:
        """Skin, dodge & burn, eyes and mouth, in that order, at whatever resolution ``rgb`` is.
        With ``steps``, what was done is recorded there (for the export)."""
        landmarks = lambda: [self.registry.default_for("face_landmarks").provenance()]  # noqa: E731
        face_p = skin_tool.RegionParams.from_dict((look["skin"] or {}).get("face"))
        if look["skin"] and not face_p.is_noop():
            rgb = skin_tool.apply(rgb, self._ensure_faces(), look["skin"].get("face"))
            if steps is not None:
                steps.append({"tool": "skin", "face": asdict(face_p), "faces": len(self.faces), "models": landmarks()})
        amount = float((look["dodge_burn"] or {}).get("amount", 0))
        if amount > 0:
            rgb = dodge_burn.apply(rgb, self._ensure_faces(), amount)
            if steps is not None:
                steps.append({"tool": "dodge_burn", "amount": amount, "faces": len(self.faces), "models": landmarks()})
        eyes_p = eye_tool.Params.from_dict(look["eyes"] or {})
        if look["eyes"] is not None and not eyes_p.is_noop():
            rgb = eye_tool.apply(rgb, self._ensure_faces(), eyes_p)
            if steps is not None:
                steps.append({"tool": "eyes", "params": asdict(eyes_p), "faces": len(self.faces), "models": landmarks()})
        mouth_p = mouth_tool.Params.from_dict(look["mouth"])
        if look["mouth"] is not None and not mouth_p.is_noop():
            rgb = mouth_tool.apply(rgb, self._ensure_faces(), mouth_p)
            if steps is not None:
                steps.append({"tool": "mouth", "params": asdict(mouth_p), "faces": len(self.faces), "models": landmarks()})
        return rgb

    def _render(self, look: dict) -> np.ndarray:
        """Preview of the full pipeline: removals, backdrop smoothing, clothes
        creases, skin, dodge & burn, eyes, mouth, then the tone (EV, curves).

        The subject tools come after the backdrop: they only touch the subject,
        where the backdrop step changes nothing, so their sliders never force a
        backdrop recompute. Dodge & burn follows skin smoothing, so smoothing
        never flattens the shaping.

        Tone is last and per-pixel, so the image before it is kept: moving the
        exposure or the curve only re-applies the tone.
        """
        key = json.dumps(
            [{**look, "tone": None}, self.edit_version, self.mask_version], sort_keys=True
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
        out = self._render_backdrop(look["backdrop"], look["model"], look["removals"])
        amount = self._crease_amount(look)
        if amount > 0:
            self._ensure_mask()
            key = (self.edit_version if look["removals"] else -1, self.mask_version)
            base = self._edited_preview() if look["removals"] else self.preview
            gain = self._creases(base, self.alpha_preview, amount, self._preview_cache, key)
            out = fabric.apply_gain(out, gain)
        return self._face_tools(out, look)

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
        the screen shows). The full-resolution result is cached, so panning around
        only crops and encodes."""
        self._require_image()
        look = self._look(look)
        h, w = self.image.rgb.shape[:2]
        x0, y0 = max(0, int(region[0])), max(0, int(region[1]))
        x1, y1 = min(w, int(np.ceil(region[2]))), min(h, int(np.ceil(region[3])))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("empty region")
        # Cached before the tone: the tone is per-pixel, so it's applied to just
        # the crop, and moving the exposure or curve never re-renders the photo.
        untoned = {**look, "tone": None}
        key = json.dumps(
            [untoned, self.strokes, self.mask_edits, self.mask_model], sort_keys=True
        )
        cached = self._full.get("result")
        if cached and cached[0] == key:
            full = cached[1]
        else:
            self.status("Rendering full-resolution detail")
            full, _ = self._full_pipeline(untoned)
            self._full["result"] = (key, full)
        crop = full[y0:y1, x0:x1]
        scale = min(1.0, max(0.01, scale))
        if scale < 1:
            size = (max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale)))
            crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        crop = tone.apply(crop, look["tone"])
        return {"image": _jpeg_b64(crop), "region": [x0, y0, x1, y1]}

    def export(self, path: str, **look) -> dict:
        self._require_image()
        out = Path(path)
        if out.resolve() == self.path.resolve():
            raise ValueError("refusing to overwrite the original image")
        look = {**self._look(look), "removals": True}
        rgb, steps = self._full_pipeline(look)
        self.status(f"Writing {out.name}")
        imageio.save(out, rgb, self.image.bit_depth, self.image.icc)
        settings = {"source": str(self.path), "steps": steps}
        sidecar = out.with_name(out.name + ".retouch.json")
        sidecar.write_text(json.dumps(settings, indent=2))
        return {"path": str(out), "settings": str(sidecar)}

    def _cached(self, stage: str, key, compute):
        hit = self._full.get(stage)
        if hit and hit[0] == key:
            return hit[1]
        value = compute()
        self._full[stage] = (key, value)
        return value

    def _full_pipeline(self, look: dict) -> tuple[np.ndarray, list[dict]]:
        """The whole pipeline at full resolution, in the preview's order.
        Returns the image and the steps to record in the export's settings."""
        rgb = self.image.rgb
        steps: list[dict] = []
        removals, backdrop, model = look["removals"], look["backdrop"], look["model"]

        if removals and self.strokes:
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
            rgb = out
            kinds = {s.get("kind", "fill") for s in self.strokes}
            models = []
            if "fill" in kinds:
                models.append(self.registry.default_for("inpaint").provenance())
            if "reflection" in kinds:  # eye openings come from the face landmarks
                models.append(self.registry.default_for("face_landmarks").provenance())
            steps.append({"tool": "remove", "strokes": self.strokes, "models": models})
        edited = rgb
        removal_key = json.dumps(self.strokes) if removals else None
        mask_key = json.dumps([self.mask_model, self.mask_edits])

        def edited_alpha():
            if not self.mask_edits:
                return self.alpha
            self.status("Applying mask edits at full resolution")
            return mask_edit.apply(self.alpha, self.mask_edits)

        if backdrop is not None:
            self._ensure_mask(model)
            p = backdrop_smooth.Params.from_dict(backdrop)
            alpha = self._cached("alpha", mask_key, edited_alpha)

            def prepare():
                self.status("Smoothing backdrop at full resolution")
                return backdrop_smooth.prepare(rgb, alpha, p)

            prep = self._cached(
                "backdrop", (backdrop_smooth.prepare_key(p), removal_key, mask_key), prepare
            )
            rgb = backdrop_smooth.compose(prep, p)
            steps.append(
                {
                    "tool": "backdrop_smooth",
                    "params": asdict(p),
                    "mask_edits": self.mask_edits,
                    "models": [self.registry.manifests[self.mask_model].provenance()],
                }
            )

        amount = self._crease_amount(look)
        if amount > 0:
            self._ensure_mask(model)
            alpha = self._cached("alpha", json.dumps([self.mask_model, self.mask_edits]), edited_alpha)
            gain = self._creases(edited, alpha, amount, self._full_crease, (removal_key, mask_key))
            rgb = fabric.apply_gain(rgb, gain)
            steps.append(
                {
                    "tool": "clothes_creases",
                    "amount": amount,
                    "models": [
                        self.registry.manifests[self.mask_model].provenance(),
                        self.registry.default_for("face_landmarks").provenance(),
                    ],
                }
            )

        if any(look[k] for k in ("skin", "dodge_burn", "eyes", "mouth")):
            self.status("Retouching the face at full resolution")
        rgb = self._face_tools(rgb, look, steps)
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

    def emit(msg: dict) -> None:
        proto.write(json.dumps(msg) + "\n")
        proto.flush()

    engine = Engine(emit)
    emit({"event": "ready"})
    for line in sys.stdin:
        if not line.strip():
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
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

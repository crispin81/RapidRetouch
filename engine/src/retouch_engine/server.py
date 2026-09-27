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

from . import imageio
from .registry import LicenceNotAccepted, Registry
from .tools import backdrop_smooth, inpaint, mask_edit
from .tools import eyes as eye_tool

PREVIEW_EDGE = 2048
METHODS = {
    "ping",
    "models",
    "accept_licence",
    "open",
    "mask",
    "mask_paint",
    "undo_mask_edit",
    "clear_mask_edits",
    "render",
    "remove",
    "undo_remove",
    "clear_removals",
    "render_region",
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
        self.status(f"Loading {Path(path).name}")
        self.path = Path(path)
        self.image = imageio.load(path)
        self.preview = _proxy(self.image.rgb, PREVIEW_EDGE)
        self.alpha = self.alpha_preview_raw = self.alpha_preview = None
        self.mask_edits = []
        self.faces = None
        self._full = {}
        self.backdrop_cache = {}
        self.strokes, self.filled_previews = [], []
        self.edit_version += 1
        h, w = self.image.rgb.shape[:2]
        return {
            "width": w,
            "height": h,
            "bit_depth": self.image.bit_depth,
            "preview": _jpeg_b64(self.preview),
        }

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

    def render(
        self,
        backdrop: dict | None = None,
        eyes: dict | None = None,
        removals: bool = True,
        model: str | None = None,
    ) -> dict:
        """Preview of the whole pipeline with the given tool settings. A tool is
        left out by passing None (or removals=False), for per-step before views."""
        self._require_image()
        return {
            "preview": _jpeg_b64(self._render(backdrop, eyes, model, removals)),
            "faces": len(self.faces) if self.faces is not None else None,
        }

    def remove(
        self,
        points: list[list[float]],
        radius: float,
        backdrop: dict | None = None,
        eyes: dict | None = None,
    ) -> dict:
        """Fill one painted stroke with LaMa, then re-render the preview."""
        self._require_image()
        if not points:
            raise ValueError("empty stroke")
        stroke = {"points": points, "radius": radius}
        base = self._edited_preview()
        mask = inpaint.stroke_mask(base.shape[:2], points, radius)
        lama = self._inpaint_model()
        self.status("Removing")
        self.filled_previews.append(inpaint.fill(base, mask, lama, self.status))
        self.strokes.append(stroke)
        self.edit_version += 1
        return {"preview": _jpeg_b64(self._render(backdrop, eyes)), "removals": len(self.strokes)}

    def undo_remove(self, backdrop: dict | None = None, eyes: dict | None = None) -> dict:
        self._require_image()
        if self.strokes:
            self.strokes.pop()
            self.filled_previews.pop()
            self.edit_version += 1
        return {"preview": _jpeg_b64(self._render(backdrop, eyes)), "removals": len(self.strokes)}

    def clear_removals(self, backdrop: dict | None = None, eyes: dict | None = None) -> dict:
        self._require_image()
        self.strokes, self.filled_previews = [], []
        self.edit_version += 1
        return {"preview": _jpeg_b64(self._render(backdrop, eyes)), "removals": 0}

    def _ensure_faces(self) -> list[np.ndarray]:
        if self.faces is None:
            manifest = self.registry.default_for("face_landmarks")
            self.status(f"Loading {manifest.name} (first run downloads it)")
            model = self.registry.get(manifest.id)
            self.status("Finding faces")
            self.faces = model.predict(self.image.rgb)
        return self.faces

    def _inpaint_model(self):
        manifest = self.registry.default_for("inpaint")
        self.status(f"Loading {manifest.name} (first run downloads it)")
        return self.registry.get(manifest.id)

    def _edited_preview(self) -> np.ndarray:
        return self.filled_previews[-1] if self.filled_previews else self.preview

    def _render(
        self,
        backdrop: dict | None,
        eyes_params: dict | None = None,
        model: str | None = None,
        removals: bool = True,
    ) -> np.ndarray:
        """Preview of the full pipeline: removals, backdrop smoothing, then eyes.

        Eyes go last: they only touch skin inside the subject, where the backdrop
        step changes nothing, so eye sliders never force a backdrop recompute.
        """
        out = self._render_backdrop(backdrop, model, removals)
        if eyes_params is not None:
            p = eye_tool.Params.from_dict(eyes_params)
            if not p.is_noop():
                out = eye_tool.apply(out, self._ensure_faces(), p)
        return out

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

    def render_region(
        self,
        region: list[float],
        scale: float,
        backdrop: dict | None = None,
        eyes: dict | None = None,
        removals: bool = True,
        model: str | None = None,
    ) -> dict:
        """A crop of the full-resolution result, for zoomed-in viewing.

        ``region`` is (x0, y0, x1, y1) in full-resolution pixels; ``scale`` is the
        output size relative to full resolution (<= 1: never send more pixels than
        the screen shows). The full-resolution result is cached, so panning around
        only crops and encodes."""
        self._require_image()
        h, w = self.image.rgb.shape[:2]
        x0, y0 = max(0, int(region[0])), max(0, int(region[1]))
        x1, y1 = min(w, int(np.ceil(region[2]))), min(h, int(np.ceil(region[3])))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("empty region")
        key = json.dumps(
            [backdrop, eyes, removals, self.strokes, self.mask_edits, self.mask_model],
            sort_keys=True,
        )
        cached = self._full.get("result")
        if cached and cached[0] == key:
            full = cached[1]
        else:
            self.status("Rendering full-resolution detail")
            full, _ = self._full_pipeline(backdrop, eyes, removals, model)
            self._full["result"] = (key, full)
        crop = full[y0:y1, x0:x1]
        scale = min(1.0, max(0.01, scale))
        if scale < 1:
            size = (max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale)))
            crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        return {"image": _jpeg_b64(crop), "region": [x0, y0, x1, y1]}

    def export(
        self,
        path: str,
        backdrop: dict | None = None,
        eyes: dict | None = None,
        model: str | None = None,
    ) -> dict:
        self._require_image()
        out = Path(path)
        if out.resolve() == self.path.resolve():
            raise ValueError("refusing to overwrite the original image")
        rgb, steps = self._full_pipeline(backdrop, eyes, True, model)
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

    def _full_pipeline(
        self,
        backdrop: dict | None,
        eyes_params: dict | None,
        removals: bool = True,
        model: str | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """The whole pipeline at full resolution: removals, backdrop, eyes.
        Returns the image and the steps to record in the export's settings."""
        rgb = self.image.rgb
        steps: list[dict] = []
        h, w = rgb.shape[:2]

        if removals and self.strokes:
            lama = self._inpaint_model()

            # Strokes only ever get appended (or undone), so if the cached fills
            # are a prefix of the current strokes, only the new ones need filling.
            done, out = [], self.image.rgb
            hit = self._full.get("edited")
            if hit and hit[0] == self.strokes[: len(hit[0])]:
                done, out = hit[0], hit[1]
            for i, stroke in enumerate(self.strokes[len(done):], len(done) + 1):
                self.status(f"Removing at full resolution ({i} of {len(self.strokes)})")
                mask = inpaint.stroke_mask((h, w), stroke["points"], stroke["radius"])
                out = inpaint.fill(out, mask, lama)
            self._full["edited"] = (list(self.strokes), out)
            rgb = out
            steps.append(
                {"tool": "remove", "strokes": self.strokes, "models": [lama.manifest.provenance()]}
            )

        if backdrop is not None:
            self._ensure_mask(model)
            p = backdrop_smooth.Params.from_dict(backdrop)
            mask_key = json.dumps([self.mask_model, self.mask_edits])

            def edited_alpha():
                if not self.mask_edits:
                    return self.alpha
                self.status("Applying mask edits at full resolution")
                return mask_edit.apply(self.alpha, self.mask_edits)

            alpha = self._cached("alpha", mask_key, edited_alpha)
            removal_key = json.dumps(self.strokes) if removals else None

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

        if eyes_params is not None:
            p = eye_tool.Params.from_dict(eyes_params)
            if not p.is_noop():
                faces = self._ensure_faces()
                self.status("Retouching eyes at full resolution")
                rgb = eye_tool.apply(rgb, faces, p)
                steps.append(
                    {
                        "tool": "eyes",
                        "params": asdict(p),
                        "faces": len(faces),
                        "models": [self.registry.default_for("face_landmarks").provenance()],
                    }
                )
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

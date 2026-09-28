"""MediaPipe Face Landmarker (Apache-2.0): 478 landmarks per face, on the CPU."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MAX_FACES = 20
REFINE_EDGE = 1024  # each face is re-run at this size for precise landmarks
REFINE_MARGIN = 0.35  # context around the face box for the refinement pass


class Adapter:
    def __init__(self, manifest, models_dir: Path, device: str):
        from mediapipe.tasks.python import BaseOptions, vision

        self.manifest = manifest
        self.edge = int(manifest.extra.get("input_edge", 2048))
        path = models_dir / manifest.id / manifest.source["filename"]
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path)),
                num_faces=MAX_FACES,
                running_mode=vision.RunningMode.IMAGE,
            )
        )

    def _detect(self, rgb: np.ndarray, edge: int) -> list[np.ndarray]:
        """Landmarks as fractions of ``rgb``'s width/height, run at ``edge`` px."""
        import mediapipe as mp

        h, w = rgb.shape[:2]
        scale = min(1.0, edge / max(h, w))
        if scale < 1:
            rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        u8 = np.ascontiguousarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=u8))
        return [
            np.array([[p.x, p.y] for p in face], np.float32) for face in result.face_landmarks
        ]

    def predict(self, rgb: np.ndarray) -> list[np.ndarray]:
        """rgb float32 HxWx3 0..1. Returns one (478, 2) array per face, as fractions
        of the image width/height, so they apply at any resolution.

        Faces are found on a downscaled copy, then each is re-run on a close crop
        of the full-resolution image. On the small copy an eye can be only ~45 px
        wide, and a few pixels of error put the iris and lid lines on the eyelid
        skin; the close-up pass is several times more precise."""
        h, w = rgb.shape[:2]
        size = np.array([w, h], np.float32)
        refined = []
        for face in self._detect(rgb, self.edge):
            pts = face * size
            lo, hi = pts.min(0), pts.max(0)
            pad = (hi - lo) * REFINE_MARGIN
            x0, y0 = np.maximum(np.floor(lo - pad), 0).astype(int)
            x1, y1 = np.minimum(np.ceil(hi + pad), size).astype(int)
            crop = rgb[y0:y1, x0:x1]
            candidates = self._detect(crop, REFINE_EDGE) if crop.size else []
            if not candidates:
                refined.append(face)
                continue
            # The crop can catch part of a neighbour: keep the matching face.
            span = np.array([x1 - x0, y1 - y0], np.float32)
            back = [c * span + [x0, y0] for c in candidates]
            best = min(back, key=lambda b: float(np.abs(b - pts).mean()))
            refined.append((best / size).astype(np.float32))
        return refined

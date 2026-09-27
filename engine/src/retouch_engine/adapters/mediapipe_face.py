"""MediaPipe Face Landmarker (Apache-2.0): 478 landmarks per face, on the CPU."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MAX_FACES = 20


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

    def predict(self, rgb: np.ndarray) -> list[np.ndarray]:
        """rgb float32 HxWx3 0..1. Returns one (478, 2) array per face, as fractions
        of the image width/height, so they apply at any resolution."""
        import mediapipe as mp

        h, w = rgb.shape[:2]
        scale = min(1.0, self.edge / max(h, w))
        if scale < 1:
            rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        u8 = np.ascontiguousarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=u8))
        return [
            np.array([[p.x, p.y] for p in face], np.float32) for face in result.face_landmarks
        ]

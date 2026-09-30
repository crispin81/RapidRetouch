"""MediaPipe Multi-class Selfie Segmentation (Apache-2.0), on the CPU.

Labels each pixel as background, hair, body skin, face skin, clothes or
accessories. The model itself works at 256 x 256, so its map is coarse: the
tools use it to decide what a region is (skin or a skin-coloured dress), while
their own colour tests keep the edges precise.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CLASSES = ("background", "hair", "body_skin", "face_skin", "clothes", "others")


class Adapter:
    def __init__(self, manifest, models_dir: Path, device: str):
        from mediapipe.tasks.python import BaseOptions, vision

        self.manifest = manifest
        self.edge = int(manifest.extra.get("input_edge", 1024))
        path = models_dir / manifest.id / manifest.source["filename"]
        self.segmenter = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.IMAGE,
                output_confidence_masks=True,
                output_category_mask=False,
            )
        )

    def predict(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        """rgb float32 HxWx3 0..1. Returns each class's probability (0..1) as
        a float32 map on a copy at most ``input_edge`` px on its long side;
        callers resize it to whatever they work at."""
        import mediapipe as mp

        h, w = rgb.shape[:2]
        scale = min(1.0, self.edge / max(h, w))
        if scale < 1:
            rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        u8 = np.ascontiguousarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        result = self.segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=u8))
        return {
            name: np.asarray(m.numpy_view(), np.float32).reshape(rgb.shape[:2]).copy()
            for name, m in zip(CLASSES, result.confidence_masks)
        }

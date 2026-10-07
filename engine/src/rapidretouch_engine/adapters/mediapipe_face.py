"""MediaPipe Face Landmarker (Apache-2.0): 478 landmarks per face, on the CPU."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MAX_FACES = 20
REFINE_EDGE = 1024  # each face is re-run at this size for precise landmarks
REFINE_MARGIN = 0.35  # context around the face box for the refinement pass
# MediaPipe's face finder looks at a ~128 px copy of what it is given, so on a
# full-length shot the face is a few pixels and missed. Overlapping square
# tiles of these fractions of the short side give small faces the same
# close-up search as a headshot (~0.1-0.3 s for 24 MP).
TILE_FRACTIONS = (1 / 2, 1 / 4)
TILE_EDGE = 512
LIFT_GAMMA = 0.6  # midtone lift for the brightened retry on dark photos
EDGE_GAP = 0.01  # a tile's face this close to its edge is cut off by it


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

    def _find(self, rgb: np.ndarray) -> list[tuple[np.ndarray, bool]]:
        """Every face, small or large, from the whole frame plus overlapping
        tiles, each with whether only a tile found it.

        Tiles overlap by half, so any face under half a tile is whole in at
        least one. A tile only adds faces that sit wholly inside it: one
        touching an inner tile edge is a piece of a bigger face (the whole
        frame or a larger tile finds it). Of the detections of one face, the
        widest is kept, and anything centred inside it is part of it."""
        h, w = rgb.shape[:2]
        size = np.array([w, h], np.float32)
        found = [(face, False) for face in self._detect(rgb, self.edge)]
        for fraction in TILE_FRACTIONS:
            win = int(min(h, w) * fraction)
            if win < 64:
                continue
            step = win // 2
            for y in range(0, h - win + step, step):
                for x in range(0, w - win + step, step):
                    y0, x0 = min(y, h - win), min(x, w - win)
                    tile = rgb[y0 : y0 + win, x0 : x0 + win]
                    for face in self._detect(tile, TILE_EDGE):
                        lo, hi = face.min(0), face.max(0)
                        cut = ((lo < EDGE_GAP) & (np.array([x0, y0]) > 0)).any() or (
                            (hi > 1 - EDGE_GAP) & (np.array([x0 + win, y0 + win]) < size)
                        ).any()
                        if not cut:
                            found.append(((face * win + [x0, y0]) / size, True))
        kept: list[tuple[np.ndarray, bool]] = []
        for face, tiled in sorted(found, key=lambda f: -float(np.ptp(f[0][:, 0] * w))):
            centre = face.mean(0)
            if not any(  # kept faces are the wider: is this one inside one?
                ((centre > k.min(0)) & (centre < k.max(0))).all() for k, _ in kept
            ):
                kept.append((face.astype(np.float32), tiled))
        return kept[:MAX_FACES]

    def predict(self, rgb: np.ndarray) -> list[np.ndarray]:
        """rgb float32 HxWx3 0..1. Returns one (478, 2) array per face, as fractions
        of the image width/height, so they apply at any resolution.

        If none is found, the search runs again on a brightened copy: in a
        low-key shot the face can be too dark for the detector to see."""
        faces = self._landmarks(rgb)
        if not faces:
            peak = float(np.percentile(rgb[::8, ::8].max(2), 99.5))
            faces = self._landmarks(np.clip(rgb / max(peak, 1e-3), 0, 1) ** LIFT_GAMMA)
        return faces

    def _landmarks(self, rgb: np.ndarray) -> list[np.ndarray]:
        """The faces in ``rgb``, each landmarked precisely.

        Faces are found on a downscaled copy, then each is re-run on a close crop
        of the full-resolution image. On the small copy an eye can be only ~45 px
        wide, and a few pixels of error put the iris and lid lines on the eyelid
        skin; the close-up pass is several times more precise."""
        h, w = rgb.shape[:2]
        size = np.array([w, h], np.float32)
        refined = []
        for face, tiled in self._find(rgb):
            pts = face * size
            lo, hi = pts.min(0), pts.max(0)
            pad = (hi - lo) * REFINE_MARGIN
            x0, y0 = np.maximum(np.floor(lo - pad), 0).astype(int)
            x1, y1 = np.minimum(np.ceil(hi + pad), size).astype(int)
            crop = rgb[y0:y1, x0:x1]
            candidates = self._detect(crop, REFINE_EDGE) if crop.size else []
            if not candidates:
                # A tile sees so little that leaves, a shoulder or grass can
                # pass for a face; a real one is found again close up.
                if not tiled:
                    refined.append(face)
                continue
            # The crop can catch part of a neighbour: keep the matching face.
            span = np.array([x1 - x0, y1 - y0], np.float32)
            back = [c * span + [x0, y0] for c in candidates]
            best = min(back, key=lambda b: float(np.abs(b - pts).mean()))
            refined.append((best / size).astype(np.float32))
        return refined

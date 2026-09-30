"""BiRefNet subject segmentation (MIT)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Adapter:
    def __init__(self, manifest, models_dir: Path, device: str):
        from transformers import AutoModelForImageSegmentation

        self.manifest = manifest
        self.device = device
        self.size = int(manifest.extra.get("input_size", 1024))
        self.model = AutoModelForImageSegmentation.from_pretrained(
            manifest.source["repo"],
            revision=manifest.source["revision"],
            trust_remote_code=True,
            cache_dir=models_dir,
        )
        self.model.to(device).eval().half()

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: float32 HxWx3 in 0..1. Returns a soft alpha, float32 HxW, 1 = subject.

        The GPU is shared with other apps (e.g. RapidRAW's AI server), so working
        memory is handed back after every run, and running out falls back to a
        retry and then the CPU instead of failing.
        """
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small).permute(2, 0, 1)[None]
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        try:
            pred = self._run(x)
        except torch.OutOfMemoryError:
            _free_gpu()
            try:
                pred = self._run(x)
            except torch.OutOfMemoryError:
                _free_gpu()
                print("birefnet: GPU memory full, running on CPU", file=sys.stderr)
                pred = self._run_cpu(x)
        finally:
            _free_gpu()
        return np.clip(cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)

    @torch.inference_mode()
    def _run(self, x: torch.Tensor) -> np.ndarray:
        out = self.model(x.to(self.device).half())[-1]
        return out.sigmoid()[0, 0].float().cpu().numpy()

    @torch.inference_mode()
    def _run_cpu(self, x: torch.Tensor) -> np.ndarray:
        # Half precision is poorly supported on CPU; borrow a float32 copy.
        self.model.to("cpu").float()
        try:
            return self.model(x.float())[-1].sigmoid()[0, 0].numpy()
        finally:
            self.model.to(self.device).half()


def _free_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

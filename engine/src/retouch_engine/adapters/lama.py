"""LaMa inpainting (Apache-2.0), TorchScript build of the official big-lama weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Adapter:
    def __init__(self, manifest, models_dir: Path, device: str):
        self.manifest = manifest
        self.device = device
        path = models_dir / manifest.id / manifest.source["filename"]
        self.model = torch.jit.load(str(path), map_location=device).eval()

    def predict(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """rgb float32 HxWx3 0..1, mask bool HxW (True = fill). Returns filled rgb.

        Sizes are padded to a multiple of 8 (reflect), which LaMa's downsampling
        needs, and cropped back afterwards.
        """
        h, w = mask.shape
        ph, pw = -h % 8, -w % 8
        img = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        msk = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
        x = torch.from_numpy(img).permute(2, 0, 1)[None].to(self.device)
        m = torch.from_numpy(msk.astype(np.float32))[None, None].to(self.device)
        try:
            with torch.inference_mode():
                out = self.model(x, m)[0].permute(1, 2, 0)[:h, :w].float().cpu().numpy()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.clip(out, 0, 1)

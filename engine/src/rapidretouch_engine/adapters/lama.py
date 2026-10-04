"""LaMa inpainting (Apache-2.0), TorchScript build of the official big-lama weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Adapter:
    def __init__(self, manifest, models_dir: Path, device: str):
        self.manifest = manifest
        self.device = device
        self.path = models_dir / manifest.id / manifest.source["filename"]
        try:
            self.model = torch.jit.load(str(self.path), map_location=device).eval()
        except Exception:  # noqa: BLE001  a GPU that can't take it: the CPU
            self.device = "cpu"
            self.model = torch.jit.load(str(self.path), map_location="cpu").eval()

    def predict(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """rgb float32 HxWx3 0..1, mask bool HxW (True = fill). Returns filled rgb.

        Sizes are padded to a multiple of 8 (reflect), which LaMa's downsampling
        needs, and cropped back afterwards.
        """
        h, w = mask.shape
        ph, pw = -h % 8, -w % 8
        img = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        msk = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
        try:
            try:
                out = self._run(img, msk, h, w)
            except (RuntimeError, NotImplementedError):
                if self.device == "cpu":
                    raise
                # A GPU without some op (a Mac's, for the FFTs): the CPU from now on.
                self.device = "cpu"
                self.model = torch.jit.load(str(self.path), map_location="cpu").eval()
                out = self._run(img, msk, h, w)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.clip(out, 0, 1)

    def _run(self, img: np.ndarray, msk: np.ndarray, h: int, w: int) -> np.ndarray:
        x = torch.from_numpy(img).permute(2, 0, 1)[None].to(self.device)
        m = torch.from_numpy(msk.astype(np.float32))[None, None].to(self.device)
        with torch.inference_mode():
            return self.model(x, m)[0].permute(1, 2, 0)[:h, :w].float().cpu().numpy()

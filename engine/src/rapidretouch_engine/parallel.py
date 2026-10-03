"""Per-pixel work split across CPU cores.

NumPy and OpenCV release Python's lock while they compute, so a pixel-by-
pixel step run on bands of rows in threads uses every core, with exactly the
same result as one pass over the whole image.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

THREADS = min(16, os.cpu_count() or 1)
MIN_PIXELS = 500_000  # below this, splitting costs more than it saves
_pool = ThreadPoolExecutor(THREADS) if THREADS > 1 else None
_inside = threading.local()  # set on the pool's threads: work there isn't split again


def inside() -> bool:
    """True on one of the threads already splitting work (don't split again)."""
    return getattr(_inside, "on", False)


def by_rows(fn: Callable[[np.ndarray], np.ndarray], img: np.ndarray) -> np.ndarray:
    """``fn(img)`` for a per-pixel ``fn`` (each output pixel depends only on
    the same input pixel), computed on bands of rows in parallel."""
    h = img.shape[0]
    pixels = h * (img.shape[1] if img.ndim > 1 else 1)
    if _pool is None or pixels < MIN_PIXELS or h < 2 * THREADS or getattr(_inside, "on", False):
        return fn(img)
    bounds = np.linspace(0, h, THREADS + 1).astype(int)

    def band(i: int) -> np.ndarray:
        # A step inside ``fn`` that splits too would wait on this same pool.
        _inside.on = True
        try:
            return fn(img[bounds[i] : bounds[i + 1]])
        finally:
            _inside.on = False

    parts = list(_pool.map(band, range(THREADS)))
    return np.concatenate(parts, axis=0)


def by_bands(fn: Callable[[int, int], np.ndarray], h: int, w: int) -> np.ndarray:
    """``fn(a, b)`` gives rows a..b of an h x w (x ...) result, worked out from
    those rows alone; the bands are computed in parallel and stacked."""
    if _pool is None or h * w < MIN_PIXELS or h < 2 * THREADS or getattr(_inside, "on", False):
        return fn(0, h)
    bounds = np.linspace(0, h, THREADS + 1).astype(int)

    def band(i: int) -> np.ndarray:
        _inside.on = True
        try:
            return fn(int(bounds[i]), int(bounds[i + 1]))
        finally:
            _inside.on = False

    return np.concatenate(list(_pool.map(band, range(THREADS))), axis=0)

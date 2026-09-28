"""Load/save images as float32 RGB 0..1, keeping bit depth and ICC profile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ICC_TAG = 34675

RAW_EXTENSIONS = {
    ".rw2", ".nef", ".nrw", ".cr2", ".cr3", ".crw", ".arw", ".srf", ".sr2",
    ".raf", ".orf", ".dng", ".pef", ".srw", ".rwl", ".3fr", ".fff", ".iiq",
    ".erf", ".mef", ".mos", ".x3f", ".kdc", ".dcr",
}


@dataclass
class LoadedImage:
    rgb: np.ndarray  # float32 HxWx3, 0..1, in the file's own (gamma-encoded) space
    bit_depth: int
    icc: bytes | None


def _srgb_icc() -> bytes:
    from PIL import ImageCms

    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _load_raw(path: Path) -> LoadedImage:
    """Develop a RAW file with LibRaw to 16-bit sRGB.

    Neutral development: camera white balance, sRGB tone curve, no auto-brighten
    (LibRaw's auto-brighten clips the brightest 1% — on a white backdrop, that's
    the backdrop). It won't match an edit made in a raw editor.
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        data = raw.postprocess(
            output_bps=16,
            use_camera_wb=True,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.sRGB,
            gamma=(2.4, 12.92),
        )
    return LoadedImage(data.astype(np.float32) / 65535.0, 16, _srgb_icc())


def load(path: str | Path) -> LoadedImage:
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        return _load_raw(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            data = page.asarray()
            icc_tag = page.tags.get(ICC_TAG)
            icc = bytes(icc_tag.value) if icc_tag else None
    else:
        with Image.open(path) as im:
            icc = im.info.get("icc_profile")
            data = np.asarray(im.convert("RGB"))
    if data.ndim == 2:
        data = np.repeat(data[..., None], 3, axis=2)
    data = data[..., :3]
    if data.dtype == np.uint16:
        return LoadedImage(data.astype(np.float32) / 65535.0, 16, icc)
    if data.dtype == np.uint8:
        return LoadedImage(data.astype(np.float32) / 255.0, 8, icc)
    if np.issubdtype(data.dtype, np.floating):
        return LoadedImage(np.clip(data.astype(np.float32), 0, 1), 32, icc)
    raise ValueError(f"unsupported pixel type {data.dtype} in {path}")


def save(path: str | Path, rgb: np.ndarray, bit_depth: int, icc: bytes | None) -> None:
    path = Path(path)
    rgb = np.clip(rgb, 0, 1)
    if path.suffix.lower() in (".tif", ".tiff"):
        if bit_depth == 8:
            data = np.round(rgb * 255).astype(np.uint8)
        else:
            data = np.round(rgb * 65535).astype(np.uint16)
        extratags = [(ICC_TAG, "B", len(icc), icc, True)] if icc else []
        tifffile.imwrite(
            path, data, photometric="rgb", compression="zlib", extratags=extratags
        )
    else:
        im = Image.fromarray(np.round(rgb * 255).astype(np.uint8))
        im.save(path, quality=95, icc_profile=icc) if icc else im.save(path, quality=95)


def thumbnail(path: str | Path, edge: int = 240) -> tuple[Image.Image, tuple[int, int]]:
    """A small, correctly rotated preview, fast, plus the photo's full size
    (width, height) as it will be once opened.

    RAW files use the camera's embedded JPEG (a few ms) rotated by LibRaw's
    orientation flag, falling back to a quick half-size decode; other files are
    read and shrunk."""
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        import io

        import rawpy

        with rawpy.imread(str(path)) as raw:
            flip = raw.sizes.flip
            full = (raw.sizes.width, raw.sizes.height)
            if flip in (5, 6):
                full = full[::-1]
            try:
                th = raw.extract_thumb()
                if th.format == rawpy.ThumbFormat.JPEG:
                    im = Image.open(io.BytesIO(th.data)).convert("RGB")
                else:
                    im = Image.fromarray(th.data).convert("RGB")
            except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                im = Image.fromarray(raw.postprocess(half_size=True, use_camera_wb=True))
                flip = 0  # postprocess already applies the orientation
        # LibRaw flip: 3 = 180 degrees, 5 = 90 anticlockwise, 6 = 90 clockwise.
        rotate = {3: Image.Transpose.ROTATE_180, 5: Image.Transpose.ROTATE_90,
                  6: Image.Transpose.ROTATE_270}.get(flip)
        if rotate is not None:
            im = im.transpose(rotate)
    elif path.suffix.lower() in (".tif", ".tiff"):
        rgb = load(path).rgb  # 16-bit TIFFs aren't reliably readable by PIL
        im = Image.fromarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        full = im.size
    else:
        from PIL import ImageOps

        with Image.open(path) as src:
            im = ImageOps.exif_transpose(src).convert("RGB")
        full = im.size
    im.thumbnail((edge, edge))
    return im, full


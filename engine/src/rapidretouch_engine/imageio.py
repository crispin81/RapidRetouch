"""Load/save images as float32 RGB 0..1, keeping bit depth and ICC profile."""

from __future__ import annotations

import functools
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

# Pillow refuses images over ~179 MP as a guard against hostile web images;
# these are the user's own photos (a 100 MP GFX frame is already over half).
Image.MAX_IMAGE_PIXELS = None

ICC_TAG = 34675
ORIENTATION_TAG = 274

RAW_EXTENSIONS = {
    ".rw2", ".nef", ".nrw", ".cr2", ".cr3", ".crw", ".arw", ".srf", ".sr2",
    ".raf", ".orf", ".dng", ".pef", ".srw", ".rwl", ".3fr", ".fff", ".iiq",
    ".erf", ".mef", ".mos", ".x3f", ".kdc", ".dcr",
}

SUPPORTED = RAW_EXTENSIONS | {".tif", ".tiff", ".jpg", ".jpeg"}


def _check_type(path: Path) -> None:
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"{path.name}: RapidRetouch opens RAW, TIFF and JPEG files")


@dataclass
class LoadedImage:
    rgb: np.ndarray  # float32 HxWx3, 0..1, in the file's own (gamma-encoded) space
    bit_depth: int
    icc: bytes | None


@functools.cache
def _srgb_icc() -> bytes:
    from PIL import ImageCms

    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _develop_params() -> dict:
    """LibRaw development settings, shared by the full open and the quick
    previews so a preview's colours match the opened photo exactly."""
    import rawpy

    return dict(
        use_camera_wb=True,
        no_auto_bright=True,
        output_color=rawpy.ColorSpace.sRGB,
        gamma=(2.4, 12.92),
    )


_DISPLAY_TRANSFORMS: dict[bytes, object] = {}


def to_display(im: Image.Image, icc: bytes | None) -> Image.Image:
    """An 8-bit RGB image in the photo's own colour space (``icc``), converted
    to sRGB for the screen: the app shows previews as sRGB, so an iPhone's
    Display P3 or a camera's Adobe RGB JPEG otherwise looks washed out.
    Exports keep the photo's own space (the profile is embedded)."""
    if not icc or icc == _srgb_icc():
        return im  # sRGB already (untagged is sRGB by convention; RAWs develop to it)
    if icc not in _DISPLAY_TRANSFORMS:
        from PIL import ImageCms

        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            _DISPLAY_TRANSFORMS[icc] = ImageCms.buildTransform(
                src, ImageCms.createProfile("sRGB"), "RGB", "RGB",
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            )
        except Exception:  # a broken or non-RGB profile: show the numbers as they are
            _DISPLAY_TRANSFORMS[icc] = None
    transform = _DISPLAY_TRANSFORMS[icc]
    if transform is None:
        return im
    from PIL import ImageCms

    from . import parallel

    # Per pixel, so on bands of rows across cores (LittleCMS takes ~37 ms for
    # a 2048 px preview on one, which a slider drag would feel).
    def convert(a: np.ndarray) -> np.ndarray:
        return np.asarray(ImageCms.applyTransform(Image.fromarray(a), transform))

    return Image.fromarray(parallel.by_rows(convert, np.asarray(im.convert("RGB"))))


def _load_raw(path: Path) -> LoadedImage:
    """Develop a RAW file with LibRaw to 16-bit sRGB.

    Neutral development: camera white balance, sRGB tone curve, no auto-brighten
    (LibRaw's auto-brighten clips the brightest 1% — on a white backdrop, that's
    the backdrop). It won't match an edit made in a raw editor.
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        data = raw.postprocess(output_bps=16, **_develop_params())
    return LoadedImage(data.astype(np.float32) / 65535.0, 16, _srgb_icc())


def _orient(data: np.ndarray, orientation: int) -> np.ndarray:
    """Turn pixels upright from an EXIF/TIFF orientation value (1-8)."""
    if orientation in (2, 4, 5, 7):
        data = data[:, ::-1]  # mirror first, then rotate
    turns = {3: 2, 4: 2, 5: 1, 6: 3, 7: 3, 8: 1}.get(orientation, 0)
    return np.ascontiguousarray(np.rot90(data, turns))


def load(path: str | Path) -> LoadedImage:
    path = Path(path)
    _check_type(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        return _load_raw(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            data = page.asarray()
            icc_tag = page.tags.get(ICC_TAG)
            icc = bytes(icc_tag.value) if icc_tag else None
            orient_tag = page.tags.get(ORIENTATION_TAG)
            data = _orient(data, int(orient_tag.value) if orient_tag else 1)
    else:
        # Phones and many cameras store every JPEG landscape and record how to
        # turn it in the EXIF orientation tag; exif_transpose applies it, as
        # the thumbnails already do.
        from PIL import ImageOps

        with Image.open(path) as im:
            icc = im.info.get("icc_profile")
            data = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
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


def save(
    path: str | Path,
    rgb: np.ndarray,
    bit_depth: int,
    icc: bytes | None,
    meta: "metadata.Metadata | None" = None,
    software: str = "RapidRetouch",
) -> None:
    """``meta``: the original's camera information, to carry over (see
    metadata.py)."""
    from . import metadata

    path = Path(path)
    rgb = np.clip(rgb, 0, 1)
    if path.suffix.lower() in (".tif", ".tiff"):
        if bit_depth == 8:
            data = np.round(rgb * 255).astype(np.uint8)
        else:
            data = np.round(rgb * 65535).astype(np.uint16)
        extratags = [(ICC_TAG, "B", len(icc), icc, True)] if icc else []
        when = None
        if meta:
            extratags += metadata.tiff_tags(meta, software)
            try:
                when = datetime.strptime(str(meta.basic.get(metadata.DATETIME)), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
        tifffile.imwrite(
            path, data, photometric="rgb", compression="zlib", extratags=extratags,
            software=software, datetime=when,
        )
    else:
        # JPEG at its highest quality: quality 100 and full-resolution colour
        # (4:4:4, no chroma subsampling), so fine colour edges such as hair
        # against a backdrop stay clean.
        im = Image.fromarray(np.round(rgb * 255).astype(np.uint8))
        extra = {"icc_profile": icc} if icc else {}
        if meta:
            extra["exif"] = metadata.exif_bytes(meta, im.size, software)
        im.save(path, "JPEG", quality=100, subsampling=0, optimize=True, **extra)


def thumbnail(path: str | Path, edge: int = 240) -> tuple[Image.Image, tuple[int, int]]:
    """A small, correctly rotated preview, fast, plus the photo's full size
    (width, height) as it will be once opened.

    RAW files are developed exactly as when opened, but at half size (which
    skips full demosaicing, ~0.17 s for 24 MP). The camera's embedded JPEG would
    be ten times faster but carries the camera's own picture style, so the
    colours visibly jumped when the photo then opened."""
    path = Path(path)
    _check_type(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            # Read the size first: a half-size develop updates it to half.
            full = (raw.sizes.width, raw.sizes.height)
            if raw.sizes.flip in (5, 6):
                full = full[::-1]
            data = raw.postprocess(half_size=True, output_bps=8, **_develop_params())
        im = Image.fromarray(data)  # postprocess applies the orientation itself
    elif path.suffix.lower() in (".tif", ".tiff"):
        loaded = load(path)  # 16-bit TIFFs aren't reliably readable by PIL
        im = Image.fromarray(np.round(np.clip(loaded.rgb, 0, 1) * 255).astype(np.uint8))
        im = to_display(im, loaded.icc)
        full = im.size
    else:
        from PIL import ImageOps

        with Image.open(path) as src:
            icc = src.info.get("icc_profile")
            im = to_display(ImageOps.exif_transpose(src).convert("RGB"), icc)
        full = im.size
    im.thumbnail((edge, edge))
    return im, full


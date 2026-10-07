"""Photos open upright whatever their EXIF/TIFF orientation tag says."""

import numpy as np
import tifffile
from PIL import Image, ImageOps

from rapidretouch_engine import imageio

# An asymmetric test pattern, so every turn and mirror is distinguishable.
PATTERN = np.zeros((40, 60, 3), np.uint8)
PATTERN[:10, :20] = (255, 0, 0)
PATTERN[30:, 50:] = (0, 0, 255)


def _expected(orientation):
    im = Image.fromarray(PATTERN)
    exif = im.getexif()
    exif[imageio.ORIENTATION_TAG] = orientation
    im.info["exif"] = exif.tobytes()
    return np.asarray(ImageOps.exif_transpose(im))


def test_jpeg_orientation(tmp_path):
    for o in range(1, 9):
        path = tmp_path / f"o{o}.jpg"
        exif = Image.Exif()
        exif[imageio.ORIENTATION_TAG] = o
        Image.fromarray(PATTERN).save(path, exif=exif.tobytes(), quality=100)
        with Image.open(path) as im:  # JPEG is lossy: compare with its own pixels
            want = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        got = np.round(imageio.load(path).rgb * 255).astype(np.uint8)
        assert np.array_equal(got, want), o


def test_other_types_refused(tmp_path):
    path = tmp_path / "photo.png"
    Image.fromarray(PATTERN).save(path)
    for read in (imageio.load, imageio.thumbnail):
        try:
            read(path)
        except ValueError as e:
            assert "RAW, TIFF and JPEG" in str(e)
        else:
            raise AssertionError("a PNG was opened")


def test_tiff_orientation(tmp_path):
    for o in range(1, 9):
        path = tmp_path / f"o{o}.tif"
        tifffile.imwrite(path, PATTERN, photometric="rgb",
                         extratags=[(imageio.ORIENTATION_TAG, "H", 1, o, True)])
        got = np.round(imageio.load(path).rgb * 255).astype(np.uint8)
        assert np.array_equal(got, _expected(o)), o


def test_jpeg_upright(tmp_path):
    path = tmp_path / "side.jpg"
    exif = Image.Exif()
    exif[imageio.ORIENTATION_TAG] = 6  # stored landscape, shown portrait
    Image.fromarray(PATTERN).save(path, exif=exif.tobytes(), quality=100)
    assert imageio.load(path).rgb.shape[:2] == (60, 40)
    assert imageio.thumbnail(path)[1] == (40, 60)


WIDE_ICC = "/usr/share/ghostscript/iccprofiles/rommrgb.icc"  # ProPhoto RGB


def test_wide_gamut_previews_show_in_srgb(tmp_path):
    """A JPEG in a wide colour space previews as the colours it means, not
    washed out; the loaded pixels (and so exports) stay in its own space."""
    import os

    import pytest
    from PIL import ImageCms

    if not os.path.exists(WIDE_ICC):
        pytest.skip("no ProPhoto profile on this machine")
    srgb = np.zeros((32, 96, 3), np.uint8)
    srgb[:, :32] = (200, 40, 40)
    srgb[:, 32:64] = (40, 160, 60)
    srgb[:, 64:] = (60, 90, 210)
    wide = ImageCms.ImageCmsProfile(WIDE_ICC)
    to_wide = ImageCms.buildTransform(ImageCms.createProfile("sRGB"), wide, "RGB", "RGB")
    im = ImageCms.applyTransform(Image.fromarray(srgb), to_wide)
    path = tmp_path / "wide.jpg"
    im.save(path, quality=100, subsampling=0, icc_profile=wide.tobytes())

    loaded = imageio.load(path)
    stored = np.round(loaded.rgb * 255)
    assert np.abs(stored - srgb).mean() > 15  # the numbers differ: wide space
    assert loaded.icc == wide.tobytes()
    shown = np.asarray(imageio.to_display(Image.fromarray(stored.astype(np.uint8)), loaded.icc), np.float32)
    assert np.abs(shown - srgb).mean() < 3
    thumb = np.asarray(imageio.thumbnail(path, 96)[0], np.float32)
    assert np.abs(thumb - srgb).mean() < 3

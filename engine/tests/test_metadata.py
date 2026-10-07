"""Exports carry the original's camera information, but nothing about its
old pixels (orientation, size, thumbnail)."""

import re

import numpy as np
import tifffile
from PIL import Image

from rapidretouch_engine import imageio, metadata


def _original(tmp_path):
    exif = Image.Exif()
    exif[metadata.MAKE] = "FUJIFILM"
    exif[metadata.MODEL] = "GFX100S\x00\x00"  # cameras pad with zero bytes
    exif[metadata.ARTIST] = "    "  # and with spaces when unset
    exif[274] = 6  # orientation: stored sideways
    sub = exif.get_ifd(metadata.EXIF_IFD)
    sub[metadata.DATE_TAKEN] = "2025:05:01 17:52:25"
    sub[metadata.EXPOSURE_TIME] = 0.005
    sub[metadata.F_NUMBER] = 2.5
    sub[metadata.ISO] = 320
    sub[metadata.LENS_MODEL] = "GF80mmF1.7 R WR"
    sub[0x927C] = b"maker notes with offsets"
    path = tmp_path / "original.jpg"
    Image.new("RGB", (60, 40), (120, 100, 90)).save(path, exif=exif.tobytes())
    return path


def test_read_cleans(tmp_path):
    meta = metadata.read(_original(tmp_path))
    assert meta.basic[metadata.MODEL] == "GFX100S"
    assert metadata.ARTIST not in meta.basic
    assert 274 not in meta.basic
    assert 0x927C not in meta.camera
    assert meta.camera[metadata.LENS_MODEL] == "GF80mmF1.7 R WR"


def test_jpeg_export(tmp_path):
    src = _original(tmp_path)
    loaded = imageio.load(src)
    out = tmp_path / "out.jpg"
    imageio.save(out, loaded.rgb, 8, None, metadata.read(src), "RapidRetouch test")
    with Image.open(out) as im:
        assert im.size == (40, 60)  # upright
        exif = im.getexif()
    assert 274 not in exif  # pixels are upright now: no turn to apply
    assert exif[metadata.MAKE] == "FUJIFILM"
    assert exif[metadata.SOFTWARE] == "RapidRetouch test"
    sub = exif.get_ifd(metadata.EXIF_IFD)
    assert sub[metadata.DATE_TAKEN] == "2025:05:01 17:52:25"
    assert sub[metadata.LENS_MODEL] == "GF80mmF1.7 R WR"
    assert (sub[0xA002], sub[0xA003]) == (40, 60)
    assert 0x927C not in sub


def test_tiff_export(tmp_path):
    src = _original(tmp_path)
    out = tmp_path / "out.tif"
    imageio.save(out, imageio.load(src).rgb, 16, None, metadata.read(src), "RapidRetouch test")
    with tifffile.TiffFile(out) as tif:
        tags = tif.pages[0].tags
        assert tags[metadata.MAKE].value == "FUJIFILM"
        assert tags[metadata.SOFTWARE].value == "RapidRetouch test"
        xmp = tags[700].value.decode()
    assert "<exif:DateTimeOriginal>2025-05-01T17:52:25<" in xmp
    assert "<exif:ExposureTime>1/200<" in xmp
    assert "<exif:FNumber>5/2<" in xmp
    assert re.search(r"<exif:ISOSpeedRatings>.*320.*</exif:ISOSpeedRatings>", xmp)
    assert np.asarray(imageio.load(out).rgb).shape == (60, 40, 3)


def test_no_metadata(tmp_path):
    path = tmp_path / "plain.jpg"
    Image.new("RGB", (8, 8)).save(path)
    assert metadata.read(path) is None
    imageio.save(tmp_path / "out.jpg", imageio.load(path).rgb, 8, None, None)
    imageio.save(tmp_path / "out.tif", imageio.load(path).rgb, 16, None, None)

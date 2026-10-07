"""The original photo's camera information (EXIF), carried into exports.

Descriptive fields only: camera, lens, capture date and exposure, author and
copyright, and GPS if the original had it. Nothing that describes the stored
pixels (size, orientation, strips, the camera's thumbnail, its maker notes,
whose internal offsets break when moved), since the export's pixels are new.

JPEG exports get real EXIF. 16-bit TIFF exports (written by tifffile, which
can't write EXIF's sub-directory) get the basic TIFF tags plus the rest as an
XMP packet, which Lightroom, Capture One and Bridge read.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

EXIF_IFD, GPS_IFD = 0x8769, 0x8825
MAKE, MODEL, SOFTWARE, DATETIME, ARTIST, COPYRIGHT, DESCRIPTION = 271, 272, 305, 306, 315, 33432, 270
BASIC = (DESCRIPTION, MAKE, MODEL, DATETIME, ARTIST, COPYRIGHT)
DATE_TAKEN, DATE_DIGITISED = 36867, 36868
EXPOSURE_TIME, F_NUMBER, ISO, FOCAL_LENGTH = 33434, 33437, 34855, 37386
LENS_MAKE, LENS_MODEL = 42035, 42036
CAMERA_FIELDS = (
    DATE_TAKEN, DATE_DIGITISED,
    36880, 36881, 36882,  # time-zone offsets
    37520, 37521, 37522,  # sub-second times
    EXPOSURE_TIME, F_NUMBER, 34850, ISO, 34864,  # program, ISO, its kind
    37380, 37381, 37383, 37385,  # exposure bias, max aperture, metering, flash
    FOCAL_LENGTH, 41989, 41986, 41987,  # 35 mm focal length, exposure mode, white balance
    42032, 42033, 42034, LENS_MAKE, LENS_MODEL, 42037,  # owner, serials, lens
)


@dataclass
class Metadata:
    basic: dict[int, object] = field(default_factory=dict)
    camera: dict[int, object] = field(default_factory=dict)
    gps: dict[int, object] = field(default_factory=dict)


def _clean(value):
    """Cameras pad unused text fields with spaces or zero bytes."""
    if isinstance(value, str):
        value = value.replace("\x00", "").strip()
        return value or None
    if isinstance(value, bytes):
        return None  # undecoded binary: not worth carrying
    return value


def _from_exif(exif: Image.Exif) -> Metadata:
    def pick(source, keys):
        out = {}
        for k in keys:
            if k in source and (v := _clean(source[k])) is not None:
                out[k] = v
        return out

    gps = exif.get_ifd(GPS_IFD)
    return Metadata(
        basic=pick(exif, BASIC),
        camera=pick(exif.get_ifd(EXIF_IFD), CAMERA_FIELDS),
        gps={k: v for k, v in gps.items() if not isinstance(v, bytes) or k == 0},  # 0: GPS version
    )


def read(path: str | Path) -> Metadata | None:
    """The photo's camera information, or None if it has none we can read.

    TIFF-based RAWs (NEF, ARW, DNG...) and JPEG/TIFF are read directly; the
    others (RW2, RAF, CR3...) from the preview JPEG the camera embedded,
    which carries the same EXIF. Never raises: an export without its
    metadata is better than no export."""
    path = Path(path)
    found = None
    try:
        with Image.open(path) as im:
            found = _from_exif(im.getexif())
    except Exception:
        pass
    if found is None or DATE_TAKEN not in found.camera:
        from .imageio import RAW_EXTENSIONS

        if path.suffix.lower() in RAW_EXTENSIONS:
            try:
                import rawpy

                with rawpy.imread(str(path)) as raw:
                    thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    with Image.open(io.BytesIO(thumb.data)) as im:
                        embedded = _from_exif(im.getexif())
                    if found is None:
                        found = embedded
                    else:  # fill what the direct read lacked
                        found.basic = {**embedded.basic, **found.basic}
                        found.camera = {**embedded.camera, **found.camera}
                        found.gps = found.gps or embedded.gps
            except Exception:
                pass
    if found is None or not (found.basic or found.camera or found.gps):
        return None
    return found


def exif_bytes(meta: Metadata, size: tuple[int, int], software: str) -> bytes:
    """EXIF for a JPEG export of ``size`` (width, height)."""
    exif = Image.Exif()
    for k, v in meta.basic.items():
        exif[k] = v
    exif[SOFTWARE] = software
    sub = exif.get_ifd(EXIF_IFD)
    sub.update(meta.camera)
    sub[0xA002], sub[0xA003] = size  # pixel width, height
    if meta.gps:
        exif.get_ifd(GPS_IFD).update(meta.gps)
    return exif.tobytes()


def tiff_tags(meta: Metadata, software: str) -> list[tuple]:
    """tifffile ``extratags`` for a TIFF export: the basic tags, and an XMP
    packet with the camera fields. (Software and DateTime are tifffile's own
    arguments; see imageio.save.)"""
    tags = [(k, "s", 0, str(v), True) for k, v in meta.basic.items() if k != DATETIME]
    xmp = _xmp(meta, software).encode("utf-8")
    tags.append((700, "B", len(xmp), xmp, True))
    return tags


def _iso_date(value) -> str | None:
    try:
        return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return None


def _ratio(value) -> str:
    """An EXIF rational as XMP wants it, e.g. "1/200"."""
    from fractions import Fraction

    try:
        f = Fraction(int(value.numerator), int(value.denominator))  # 10/2000 -> 1/200
    except (AttributeError, ZeroDivisionError, TypeError, ValueError):
        f = Fraction(float(value)).limit_denominator(100000)
    return f"{f.numerator}/{f.denominator}"


def _gps_coord(dms, ref) -> str | None:
    """EXIF degrees/minutes/seconds + N/S/E/W as XMP's "51,30.1234N"."""
    try:
        d, m, s = (float(x) for x in dms)
        return f"{int(d)},{m + s / 60:.6f}{ref}"
    except (TypeError, ValueError):
        return None


def _xmp(meta: Metadata, software: str) -> str:
    props: list[str] = []

    def add(name: str, value) -> None:
        if value is not None:
            props.append(f"   <{name}>{escape(str(value))}</{name}>")

    b, c, g = meta.basic, meta.camera, meta.gps
    add("tiff:Make", b.get(MAKE))
    add("tiff:Model", b.get(MODEL))
    add("xmp:CreatorTool", software)
    taken = _iso_date(c.get(DATE_TAKEN)) if DATE_TAKEN in c else None
    add("exif:DateTimeOriginal", taken)
    add("xmp:CreateDate", taken)
    if EXPOSURE_TIME in c:
        add("exif:ExposureTime", _ratio(c[EXPOSURE_TIME]))
    if F_NUMBER in c:
        add("exif:FNumber", _ratio(c[F_NUMBER]))
    if FOCAL_LENGTH in c:
        add("exif:FocalLength", _ratio(c[FOCAL_LENGTH]))
    add("exifEX:LensMake", c.get(LENS_MAKE))
    add("exifEX:LensModel", c.get(LENS_MODEL))
    add("aux:Lens", c.get(LENS_MODEL))
    if 2 in g and 4 in g:
        add("exif:GPSLatitude", _gps_coord(g[2], g.get(1, "N")))
        add("exif:GPSLongitude", _gps_coord(g[4], g.get(3, "E")))
    if ISO in c:
        iso = c[ISO][0] if isinstance(c[ISO], tuple) else c[ISO]
        props.append(f"   <exif:ISOSpeedRatings><rdf:Seq><rdf:li>{int(iso)}</rdf:li></rdf:Seq></exif:ISOSpeedRatings>")
    if ARTIST in b:
        props.append(f"   <dc:creator><rdf:Seq><rdf:li>{escape(str(b[ARTIST]))}</rdf:li></rdf:Seq></dc:creator>")
    if COPYRIGHT in b:
        props.append(
            '   <dc:rights><rdf:Alt><rdf:li xml:lang="x-default">'
            f"{escape(str(b[COPYRIGHT]))}</rdf:li></rdf:Alt></dc:rights>"
        )
    body = "\n".join(props)
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:tiff="http://ns.adobe.com/tiff/1.0/"\n'
        '    xmlns:exif="http://ns.adobe.com/exif/1.0/"\n'
        '    xmlns:exifEX="http://cipa.jp/exif/1.0/"\n'
        '    xmlns:aux="http://ns.adobe.com/exif/1.0/aux/"\n'
        '    xmlns:xmp="http://ns.adobe.com/xap/1.0/"\n'
        '    xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        f"{body}\n"
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )

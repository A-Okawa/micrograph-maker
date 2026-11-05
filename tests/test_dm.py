"""DigitalMicrograph .dm3 / .dm4: files written here tag by tag, then read back.

    python tests/test_dm.py
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edsfig import aztec, dm  # noqa: E402

_ARRAY_TYPE = {np.dtype("int16"): 2, np.dtype("int32"): 3, np.dtype("uint16"): 4, np.dtype("uint32"): 5,
               np.dtype("float32"): 6, np.dtype("float64"): 7, np.dtype("int8"): 9, np.dtype("uint8"): 10}
_IMAGE_TYPE = {np.dtype("int16"): 1, np.dtype("float32"): 2, np.dtype("uint8"): 6, np.dtype("int32"): 7,
               np.dtype("int8"): 9, np.dtype("uint16"): 10, np.dtype("uint32"): 11, np.dtype("float64"): 12}


class _Writer:
    def __init__(self, version: int, little: bool = True):
        self.v, self.e = version, "<" if little else ">"

    def count(self, n: int) -> bytes:
        return struct.pack(">Q" if self.v == 4 else ">I", n)

    def group(self, entries: list) -> bytes:
        out = bytearray(b"\x00\x00") + self.count(len(entries))
        for label, val in entries:
            lab = label.encode("latin-1")
            body = self.group(val) if isinstance(val, list) else self.data(val)
            out += bytes([20 if isinstance(val, list) else 21]) + struct.pack(">H", len(lab)) + lab
            if self.v == 4:
                out += struct.pack(">Q", len(body))
            out += body
        return bytes(out)

    def data(self, val) -> bytes:
        if isinstance(val, str):  # DM keeps text as an array of UTF-16 code units
            val = np.array([ord(c) for c in val], np.uint16)
        if isinstance(val, np.ndarray):
            arr = np.ascontiguousarray(val.ravel())
            info = [20, _ARRAY_TYPE[arr.dtype], arr.size]
            payload = arr.astype(arr.dtype.newbyteorder(self.e)).tobytes()
        elif isinstance(val, float):
            info, payload = [7], struct.pack(self.e + "d", val)
        else:
            info, payload = [3], struct.pack(self.e + "i", int(val))
        return b"%%%%" + self.count(len(info)) + b"".join(self.count(i) for i in info) + payload

    def file(self, root: list) -> bytes:
        body = self.group(root)
        size = struct.pack(">Q" if self.v == 4 else ">I", len(body) + 8)
        return struct.pack(">I", self.v) + size + struct.pack(">I", 1 if self.e == "<" else 0) + body


def _image(data: np.ndarray, scale: float, unit: str, name: str, frames: int = 1, dtype_code: int | None = None) -> list:
    h, w = data.shape[-2:] if data.ndim >= 2 and dtype_code != 23 else data.shape[:2]
    dims = [("", w), ("", h)] + ([("", frames)] if frames > 1 else [])
    cal = [("", [("Origin", 0.0), ("Scale", float(scale)), ("Units", unit)]) for _ in range(2)]
    return [("ImageData", [("Calibrations", [("Dimension", cal)]), ("Data", data),
                           ("DataType", dtype_code or _IMAGE_TYPE[data.dtype]), ("Dimensions", dims)]),
            ("ImageTags", []), ("Name", name)]


def _file(version: int, main: list, little: bool = True, thumb_big: bool = True) -> bytes:
    rng = np.random.default_rng(0)
    th = rng.integers(0, 255, (80, 90) if thumb_big else (8, 9), dtype=np.uint8)  # a thumbnail, flagged as such
    root = [("ImageList", [("", _image(th, 1.0, "", "thumb")), ("", main)]),
            ("Thumbnails", [("", [("ImageIndex", 0)])])]
    return _Writer(version, little).file(root)


def test_versions_byte_orders_and_thumbnail():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 60000, (48, 64), dtype=np.uint16)
    for version in (3, 4):
        for little in (True, False):
            data = _file(version, _image(img, 2.5, "nm", "HAADF"), little)
            assert dm.is_dm(data)
            im = dm.read_dm(data)
            assert np.array_equal(im.data, img), (version, little)  # not the (larger) thumbnail
            assert im.scale == (2.5, 2.5) and im.units == ("nm", "nm") and im.name == "HAADF"
            assert abs(im.um_per_px - 0.0025) < 1e-12 and im.frames == 1


def test_types_stack_and_rgb():
    rng = np.random.default_rng(2)
    for dt in ("int16", "float32", "int32", "uint8", "float64"):
        img = (rng.normal(0, 100, (30, 40))).astype(dt)
        im = dm.read_dm(_file(4, _image(img, 0.1, "µm", "x")))
        assert im.data.dtype == np.dtype(dt) and np.array_equal(im.data, img), dt
    stack = rng.integers(0, 1000, (3, 20, 25), dtype=np.uint16)
    im = dm.read_dm(_file(3, _image(stack, 1.0, "nm", "s", frames=3)))
    assert im.frames == 3 and np.array_equal(im.data, stack[0])
    rgb = rng.integers(0, 255, (12, 16, 3), dtype=np.uint8)
    words = (rgb[..., 2].astype(np.uint32) | rgb[..., 1].astype(np.uint32) << 8 |
             rgb[..., 0].astype(np.uint32) << 16 | np.uint32(255) << 24)  # stored as B, G, R, A bytes
    im = dm.read_dm(_file(4, _image(words, 1.0, "nm", "rgb", dtype_code=23)))
    assert im.data.shape == (12, 16, 3) and np.array_equal(im.data, rgb)


def test_display_and_calibration_in_the_app_model():
    rng = np.random.default_rng(3)
    img = rng.normal(1000, 50, (40, 50)).astype(np.float32)
    img[0, 0], img[0, 1] = -1e6, 1e6  # outliers do not wash out the display
    src = aztec.load_sources(_file(4, _image(img, 0.5, "nm", "BF")), "bf.dm4")[0]
    assert src.rgba.shape == (40, 50, 4) and (src.rgba[..., 3] == 255).all()
    g = src.rgba[..., 0].astype(int)
    inner = g[1:]
    assert inner.min() <= 3 and inner.max() >= 252  # 0.1-99.9 % mapped to the full range
    order = np.argsort(img[1:].ravel())
    assert np.all(np.diff(inner.ravel()[order]) >= 0)  # linear: grey order kept
    assert abs(src.meta_um_per_px - 0.0005) < 1e-12 and not src.reciprocal and not src.note
    diff = aztec.load_sources(_file(3, _image(img, 0.02, "1/nm", "DP")), "dp.dm3")[0]
    assert diff.meta_um_per_px is None and diff.reciprocal and "1/nm" in diff.note
    p = aztec.read_images([(_file(4, _image(img, 0.5, "nm", "BF")), "bf.dm4")]).panels[0]
    assert p.scale_source == "metadata" and p.band is None and p.size == (50, 40)


if __name__ == "__main__":
    test_versions_byte_orders_and_thumbnail()
    test_types_stack_and_rgb()
    test_display_and_calibration_in_the_app_model()
    print("ok")

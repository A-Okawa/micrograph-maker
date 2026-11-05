"""Gatan DigitalMicrograph images (.dm3 / .dm4), read with NumPy only.

A DM file is a tree of tagged values.  The pictures are under ImageList; each has
ImageData/Data (the pixels), ImageData/Dimensions and ImageData/Calibrations/Dimension (pixel
size and unit per axis).  The tag structure is big-endian; the values use the byte order given in
the header.  The first image of a .dm3 is usually the thumbnail, so the largest one is used.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

_SIMPLE = {  # encoded type -> (struct/numpy code, size)
    2: ("h", 2), 3: ("i", 4), 4: ("H", 2), 5: ("I", 4), 6: ("f", 4), 7: ("d", 8),
    8: ("?", 1), 9: ("b", 1), 10: ("B", 1), 11: ("q", 8), 12: ("Q", 8),
}
_UNIT_UM = {"nm": 1e-3, "µm": 1.0, "um": 1.0, "μm": 1.0, "mm": 1e3, "m": 1e6, "Å": 1e-4, "A": 1e-4, "pm": 1e-6}


def is_dm(data: bytes) -> bool:
    return len(data) > 24 and data[:4] in (b"\x00\x00\x00\x03", b"\x00\x00\x00\x04")


@dataclass
class DMImage:
    data: np.ndarray  # 2-D (first frame of a stack) or H x W x 3 for RGB
    scale: tuple[float, float]  # per pixel, x and y
    units: tuple[str, str]
    name: str = ""
    frames: int = 1

    @property
    def um_per_px(self) -> float | None:
        """Real-space pixel size in µm, or None (uncalibrated, reciprocal space, ...)."""
        u = self.units[0].strip()
        if u in _UNIT_UM and self.scale[0] > 0:
            return self.scale[0] * _UNIT_UM[u]
        return None


class _Parser:
    def __init__(self, data: bytes):
        self.d = data
        self.v = struct.unpack(">I", data[:4])[0]
        if self.v not in (3, 4):
            raise ValueError("DigitalMicrograph 3/4 形式ではありません")
        self.p = 4 + (4 if self.v == 3 else 8)  # version, file size
        self.e = "<" if struct.unpack(">I", data[self.p: self.p + 4])[0] == 1 else ">"
        self.p += 4
        self.root = self._group()

    def _be(self, code: str, n: int):
        v = struct.unpack(">" + code, self.d[self.p: self.p + n])[0]
        self.p += n
        return v

    def _count(self) -> int:
        return self._be("Q", 8) if self.v == 4 else self._be("I", 4)

    def _group(self) -> list:
        self.p += 2  # sorted, open
        out = []
        for _ in range(self._count()):
            kind = self.d[self.p]
            self.p += 1
            n = self._be("H", 2)
            label = self.d[self.p: self.p + n].decode("latin-1")
            self.p += n
            if self.v == 4:
                self.p += 8  # size of the entry
            if kind == 20:
                out.append((label, self._group()))
            elif kind == 21:
                out.append((label, self._data()))
            else:
                raise ValueError(f"壊れたタグ（種類 {kind}）")
        return out

    def _data(self):
        if self.d[self.p: self.p + 4] != b"%%%%":
            raise ValueError("壊れたタグデータ")
        self.p += 4
        n = self._count()
        info = [self._count() for _ in range(n)]
        return self._value(info)

    def _value(self, info: list[int]):
        t = info[0]
        if t in _SIMPLE:
            code, size = _SIMPLE[t]
            v = struct.unpack(self.e + code, self.d[self.p: self.p + size])[0]
            self.p += size
            return v
        if t == 18:  # string
            n = info[1]
            v = self.d[self.p: self.p + n].decode("latin-1", "ignore")
            self.p += n
            return v
        if t == 15:  # struct: 15, name length, field count, (name length, type) * fields
            types = [info[4 + 2 * i] for i in range(info[2])]
            return tuple(self._value([ft]) for ft in types)
        if t == 20:  # array
            et = info[1]
            if et == 15:  # of structs: 20, 15, name length, field count, (name length, type)*, count
                nf = info[3]
                types = [info[5 + 2 * i] for i in range(nf)]
                count = info[4 + 2 * nf]
                dt = np.dtype([(f"f{i}", self.e + _SIMPLE[ft][0]) for i, ft in enumerate(types)])
                arr = np.frombuffer(self.d, dtype=dt, count=count, offset=self.p)
                self.p += dt.itemsize * count
                return arr
            if et in _SIMPLE:
                code, size = _SIMPLE[et]
                count = info[2]
                arr = np.frombuffer(self.d, dtype=np.dtype(self.e + code), count=count, offset=self.p)
                self.p += size * count
                return arr
        raise ValueError(f"未対応のデータ型 {info[:3]}")


def _get(group: list, *path):
    node = group
    for key in path:
        if not isinstance(node, list):
            return None
        hit = [v for k, v in node if k == key]
        if not hit:
            return None
        node = hit[0]
    return node


def _text(v) -> str:
    if isinstance(v, np.ndarray):
        return "".join(chr(int(c)) for c in v if int(c))
    return str(v or "")


def _image(entry: list) -> DMImage | None:
    data = _get(entry, "ImageData", "Data")
    dims = _get(entry, "ImageData", "Dimensions")
    dtype = _get(entry, "ImageData", "DataType")
    if not isinstance(data, np.ndarray) or not isinstance(dims, list) or len(dims) < 2:
        return None
    shape = [int(v) for _, v in dims]  # x, y[, frames]
    w, h = shape[0], shape[1]
    frames = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    if dtype in (3, 5, 13, 27, 28) or data.dtype.names:  # complex / packed complex (FFTs): not a picture here
        return None
    if dtype in (8, 23):  # RGB(A) stored as 32-bit words
        rgba = data.view(np.uint8).reshape(-1, 4)[: w * h * frames]
        img = rgba.reshape(frames, h, w, 4)[0][..., [2, 1, 0]]  # stored B, G, R, A
    else:
        if data.size < w * h:
            return None
        img = data[: w * h * frames].reshape(frames, h, w)[0]
    scales, units = [1.0, 1.0], ["", ""]
    cal = _get(entry, "ImageData", "Calibrations", "Dimension")
    if isinstance(cal, list):
        for i, (_, dim) in enumerate(cal[:2]):
            if isinstance(dim, list):
                sc = _get(dim, "Scale")
                scales[i] = float(sc) if isinstance(sc, (int, float)) else 1.0
                units[i] = _text(_get(dim, "Units"))
    name = _text(_get(entry, "Name"))
    return DMImage(np.array(img), (scales[0], scales[1]), (units[0], units[1]), name, frames)


def read_dm(data: bytes) -> DMImage:
    """The main picture of a .dm3/.dm4 file (thumbnails skipped; the largest 2-D image; first frame
    of a stack)."""
    root = _Parser(data).root
    thumbs = set()  # the file lists which images are thumbnails
    for _, t in (_get(root, "Thumbnails") or []):
        idx = _get(t, "ImageIndex") if isinstance(t, list) else None
        if isinstance(idx, (int, float)):
            thumbs.add(int(idx))
    images = []
    lst = _get(root, "ImageList")
    for i, (_, entry) in enumerate(lst or []):
        if i in thumbs:
            continue
        if isinstance(entry, list):
            im = _image(entry)
            if im is not None and im.data.ndim >= 2 and min(im.data.shape[:2]) > 1:
                images.append(im)
    if not images:
        raise ValueError("画像が入っていません（スペクトルや複素数データのみのファイル）")
    return max(images, key=lambda im: im.data.shape[0] * im.data.shape[1])


def to_8bit(img: np.ndarray, lo_pct: float = 0.1, hi_pct: float = 99.9) -> np.ndarray:
    """Linear display mapping of a TEM image: the 0.1-99.9 % range of the data -> 0..255."""
    if img.ndim == 3:  # already RGB
        return np.clip(img, 0, 255).astype(np.uint8)
    x = np.asarray(img, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(x.shape, np.uint8)
    lo, hi = np.percentile(finite, (lo_pct, hi_pct))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(x.shape, np.uint8)
    y = (np.nan_to_num(x, nan=lo) - lo) * 255.0 / (hi - lo)
    return np.clip(np.round(y), 0, 255).astype(np.uint8)

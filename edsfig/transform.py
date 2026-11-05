"""Turning and mirroring a picture.

* Quarter turns (90, 180, 270 degrees) and mirroring only rearrange the pixels: nothing is
  resampled, every value stays as it was.
* Any other angle needs the pixels to be made anew on the turned grid (interpolation): bicubic
  by default, or nearest neighbour (keeps the original grey values, jagged edges).  The pixel
  pitch is kept, so a length in µm per pixel is unchanged and a scale bar stays correct.
  The angle is split into quarter turns (exact) plus a remainder of at most 45 degrees, so only
  that remainder is interpolated.  The empty corners of a turned picture are cut off (the largest
  upright rectangle inside it) or filled with white / black.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

import numpy as np
from PIL import Image

_CACHE: OrderedDict = OrderedDict()  # turned pictures, reused between redraws
_CACHE_MAX = 24


def is_identity(angle_cw: float, flip_h: bool = False, flip_v: bool = False) -> bool:
    return abs(_norm(angle_cw)) < 1e-9 and not flip_h and not flip_v


def _norm(a: float) -> float:
    """Angle in (-180, 180]."""
    a = math.fmod(float(a), 360.0)
    if a <= -180.0:
        a += 360.0
    elif a > 180.0:
        a -= 360.0
    return a


def split_angle(angle_cw: float) -> tuple[int, float]:
    """Clockwise angle -> (clockwise quarter turns 0..3, remaining clockwise angle in [-45, 45))."""
    a = _norm(angle_cw)
    q = int(math.floor((a + 45.0) / 90.0))
    rest = a - 90.0 * q
    if abs(rest) < 1e-9:
        rest = 0.0
    return q % 4, rest


def largest_upright_rect(w: float, h: float, angle_deg: float) -> tuple[float, float]:
    """Width and height of the largest axis-aligned rectangle inside a w x h rectangle turned by
    angle_deg (centred)."""
    if w <= 0 or h <= 0:
        return 0.0, 0.0
    a = math.radians(angle_deg)
    sin_a, cos_a = abs(math.sin(a)), abs(math.cos(a))
    long_side, short_side = (w, h) if w >= h else (h, w)
    if short_side <= 2.0 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * short_side  # touches the long sides only
        wr, hr = (x / sin_a, x / cos_a) if w >= h else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def rotate(img: np.ndarray, angle_cw: float = 0.0, flip_h: bool = False, flip_v: bool = False,
           fill: str = "crop", resample: str = "bicubic") -> np.ndarray:
    """The picture turned clockwise by angle_cw degrees, then mirrored (left-right / up-down).
    fill: "crop" (cut the empty corners away), "white" or "black" (keep the whole picture)."""
    if is_identity(angle_cw, flip_h, flip_v):
        return img
    key = (hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest(), img.shape,
           round(_norm(angle_cw), 6), bool(flip_h), bool(flip_v), fill, resample)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit
    q, rest = split_angle(angle_cw)
    out = np.rot90(img, k=-q) if q else img  # np.rot90 turns counter-clockwise for k > 0
    if rest:
        h, w = out.shape[:2]
        mode = Image.NEAREST if resample == "nearest" else Image.BICUBIC
        colour = (0, 0, 0) if fill == "black" else (255, 255, 255)
        im = Image.fromarray(np.ascontiguousarray(out)).convert("RGB")
        big = np.asarray(im.rotate(-rest, resample=mode, expand=True, fillcolor=colour))  # PIL turns counter-clockwise
        if fill == "crop":
            wr, hr = largest_upright_rect(w, h, rest)
            cw, ch = max(1, int(math.floor(wr)) - 2), max(1, int(math.floor(hr)) - 2)  # 1 px off the blended edge
            H, W = big.shape[:2]
            x0, y0 = (W - cw) // 2, (H - ch) // 2
            big = big[y0: y0 + ch, x0: x0 + cw]
        out = big
    if flip_h:
        out = out[:, ::-1]
    if flip_v:
        out = out[::-1]
    out = np.ascontiguousarray(out)
    _CACHE[key] = out
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return out


def describe(angle_cw: float, flip_h: bool = False, flip_v: bool = False) -> str:
    """Short Japanese description, e.g. '時計回り 90°・左右反転'."""
    parts = []
    a = _norm(angle_cw)
    if abs(a) > 1e-9:
        parts.append(f"{'時計回り' if a > 0 else '反時計回り'} {abs(a):g}°")
    if flip_h:
        parts.append("左右反転")
    if flip_v:
        parts.append("上下反転")
    return "・".join(parts) or "なし"

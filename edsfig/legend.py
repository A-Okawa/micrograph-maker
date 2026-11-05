"""Colour key of an AZtec layered image ("EDS Layered Image").

AZtec draws the key into a corner of the picture: a row of small boxes, one per layer, each
filled with the layer's colour and labelled with its element symbol ("O", "Si"), plus a white
box for the electron image ("Electron").  The boxes are found by geometry alone: inside a box
every row has one colour (AZtec shades the boxes from top to bottom, never across), so the fill
forms a flat patch around the letters, while the picture around it is grainy.  A few boxes of
the same height on one line, in a corner, make a key.

The element symbols are read like map titles (textread), the colours are taken from the fill.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import textread


@dataclass
class Chip:
    x0: int  # the box's flat interior, inside the picture (inclusive)
    y0: int
    x1: int
    y1: int
    color: tuple[int, int, int]  # fill colour at full brightness
    electron: bool  # grey or white box: the electron image layer, not an element
    ink: np.ndarray | None = field(default=None, repr=False)  # the letters (0..1), cropped
    text: str = ""
    score: float = 0.0
    margin: float = 0.0


@dataclass
class Legend:
    side: str  # "bottom" | "top": the edge the key sits at
    rows: int  # rows from that edge that hold the key (cut off with it)
    chips: list[Chip]

    @property
    def elements(self) -> list[Chip]:
        return [c for c in self.chips if not c.electron]


def _flat(x: np.ndarray, tol: int) -> np.ndarray:
    """Pixels equal (within tol in every channel) to both horizontal neighbours."""
    d = np.abs(np.diff(x, axis=1)).max(axis=2)
    out = np.zeros(x.shape[:2], bool)
    out[:, 1:-1] = (d[:, :-1] <= tol) & (d[:, 1:] <= tol)
    return out


def _runs(flat_row: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate(([0], flat_row.astype(np.int8), [0])))
    return [(int(a), int(b) - 1) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_len]


def _chips_in(x: np.ndarray, y_off: int, H: int, W: int, side: str, tol: int) -> list[Chip]:
    """Boxes of a key: rows of one flat colour with the same left and right ends, repeated over
    the height of a box (the rows through the letters break up and are skipped)."""
    h_min, h_max = max(6, int(0.018 * H)), max(10, int(0.12 * H))
    max_gap = int(0.07 * H) + 2  # the rows of letters inside a box
    flat = _flat(x, tol)
    groups: list[dict] = []  # same columns (±2 px), close rows, a colour that changes smoothly
    active: list[dict] = []
    for y in range(x.shape[0]):
        active = [g for g in active if y - g["y1"] <= max_gap]
        for a, b in _runs(flat[y], 5):
            if np.abs(x[y, a] - x[y, b]).max() > 2 * tol:  # a slow drift along the row is not a fill
                continue
            col = x[y, a: b + 1].mean(axis=0)
            match = [g for g in active if abs(g["a"] - a) <= 2 and abs(g["b"] - b) <= 2
                     and np.abs(g["col"] - col).max() <= 64]
            if match:
                g = max(match, key=lambda g: g["y1"])
                g["y1"], g["n"], g["col"] = y, g["n"] + 1, col
            else:
                g = {"a": a, "b": b, "y0": y, "y1": y, "n": 1, "col": col}
                groups.append(g)
                active.append(g)
    found = []
    for g in groups:
        h, w = g["y1"] - g["y0"] + 1, g["b"] - g["a"] + 1
        if not (h_min <= h <= h_max and w >= max(4, 0.3 * h) and w <= 12 * h and g["n"] >= max(4, 0.3 * h)):
            continue
        gy0, gy1 = g["y0"] + y_off, g["y1"] + y_off
        if (side == "bottom" and H - 1 - gy1 > 0.06 * H) or (side == "top" and gy0 > 0.06 * H):
            continue  # a key sits right at the edge
        found.append((g["a"], gy0, g["b"], gy1))
    if not found:
        return []
    # boxes of one key: the same top and bottom, side by side, the first one near the left edge
    best: list[tuple[int, int, int, int]] = []
    for ref in found:
        row = sorted((b for b in found if abs(b[1] - ref[1]) <= 3 and abs(b[3] - ref[3]) <= 3), key=lambda b: b[0])
        chain = [row[0]] if row else []
        for b in row[1:]:
            h = b[3] - b[1] + 1
            if b[0] > chain[-1][2] and b[0] - chain[-1][2] - 1 <= 1.5 * h:
                chain.append(b)
        if chain and chain[0][0] <= 0.08 * W and len(chain) > len(best):
            best = chain
    return [Chip(*b, color=(0, 0, 0), electron=False) for b in best]


def _row_fill(box: np.ndarray, tol: int) -> np.ndarray:
    """Fill colour of every row of a box: from its flat pixels, and across the rows of letters
    (few flat pixels there) interpolated from the rows above and below."""
    flat = _flat(box, tol)
    n = flat.sum(axis=1)
    known = np.flatnonzero(n >= max(3, 0.25 * box.shape[1]))
    if known.size == 0:
        return np.median(box, axis=1)
    fill = np.zeros((box.shape[0], 3))
    for y in known:
        fill[y] = np.median(box[y][flat[y]], axis=0)
    for ch in range(3):
        fill[:, ch] = np.interp(np.arange(box.shape[0]), known, fill[known, ch])
    return fill


def find_legend(rgb: np.ndarray, tol: int = 8) -> Legend | None:
    """The colour key in the bottom-left (or top-left) corner of a picture, or None."""
    H, W = rgb.shape[:2]
    if H < 60 or W < 60:
        return None
    x = rgb[..., :3].astype(np.int16)
    span = max(12, int(0.2 * H))
    for side in ("bottom", "top"):
        y_off = H - span if side == "bottom" else 0
        region = x[y_off: y_off + span, : max(12, int(0.75 * W))]
        chips = _chips_in(region, y_off, H, W, side, tol)
        if len(chips) < 2:  # an element and the electron image at least
            continue
        good = []
        for c in chips:
            box = x[c.y0: c.y1 + 1, c.x0: c.x1 + 1]
            fill = _row_fill(box, tol)
            diff = np.abs(box - fill[:, None, :]).max(axis=2).astype(np.float64)
            ink = np.clip((diff - 30) / 120.0, 0, 1)  # the letters (the box's outline is outside the flat fill)
            if (ink > 0.5).mean() < 0.02:  # a box without letters is not a key
                continue
            n = len(fill)  # colour: the lower middle of the box (AZtec shades the top lighter, the last row too)
            col = np.median(fill[int(0.55 * n): max(int(0.55 * n) + 1, int(0.85 * n))], axis=0)
            c.electron = float(col.max() - col.min()) < 40
            c.color = tuple(int(round(v)) for v in 255 * col / max(1.0, float(col.max())))
            c.ink = textread.crop_ink(ink)
            good.append(c)
        if len(good) < 2 or not any(not c.electron for c in good):
            continue
        border = max(2, int(round(0.08 * (good[0].y1 - good[0].y0 + 1))))  # the box outline outside the flat fill
        if side == "bottom":
            rows = H - (min(c.y0 for c in good) - border)
        else:
            rows = max(c.y1 for c in good) + border + 1
        return Legend(side, int(min(rows, H // 3)), good)
    return None


def read_legend(legend: Legend, reader: textread.Reader) -> None:
    """Element symbol of every coloured box (the electron box is left as it is)."""
    for c in legend.elements:
        if c.ink is None:
            continue
        best = None
        for fonts in ("aztec", "title"):
            res = reader.read_element(c.ink, fonts=fonts)
            if res.text and (best is None or res.score > best.score):
                best = res
        if best is not None:
            c.text, c.score, c.margin = best.text, best.score, best.margin

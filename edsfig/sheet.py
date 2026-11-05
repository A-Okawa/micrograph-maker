"""JEOL EPMA map sheets: the map window of JEOL's EPMA software saved as one picture.

The element maps (and electron images: SED, BED-C ...) sit on teal tiles in a grid.  Below each
map, a white box holds its name, a thin scale bar and the bar's length ("O ——— 20 µm"); the
colour scales and the measurement conditions are listed at the side.

Every tile becomes one picture: the map is cropped exactly (no resampling), its name is read like
an AZtec map title and its scale bar gives the pixel size.  All tiles come from one stage scan, so
they share one scale: the bar is measured on every tile and the scale value read from all of them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import textread

# names of electron images in JEOL EPMA software (secondary / backscattered electrons ...)
TEAL = (0, 128, 128)  # the tiles' background
DETECTORS = ["SED", "SEI", "BED-C", "BED-T", "BEC", "BET", "BEI", "COMPO", "TOPO", "BSE", "CL"]


@dataclass
class Tile:
    box: tuple[int, int, int, int]  # the map, x0, y0, x1, y1 (inclusive) in the picture
    label_box: tuple[int, int, int, int] | None = None  # the white box below the map
    bar_px: float = 0.0  # length of the scale bar (pixels)
    bar_x: tuple[int, int] = (0, 0)  # its first and last column in the picture
    bar_y: int = 0
    name_ink: np.ndarray | None = None  # ink (0..1) of the name left of the bar
    scale_ink: np.ndarray | None = None  # ink of the length right of the bar ("20 µm")
    preview: np.ndarray | None = None  # the label box, grey (dark text on white)


def _runs(flags: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """(first, last) of every run of True at least min_len long."""
    d = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    return [(int(a), int(b) - 1) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_len]


def teal_mask(rgb: np.ndarray) -> np.ndarray:
    """The tiles' background: RGB (0, 128, 128), allowing for JPEG, and flat (a map's own teal-like
    pixels differ from their neighbours)."""
    x = rgb[..., :3].astype(np.int16)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    colour = (r <= 40) & (np.abs(g - b) <= 22) & (g >= 100) & (g <= 156)
    d = np.zeros(x.shape[:2], np.int16)
    for dy, dx in ((0, 1), (1, 0)):  # largest difference to the right and lower neighbours
        diff = np.abs(x[dy:, dx:] - x[: x.shape[0] - dy, : x.shape[1] - dx]).max(axis=2)
        d[: x.shape[0] - dy, : x.shape[1] - dx] = np.maximum(d[: x.shape[0] - dy, : x.shape[1] - dx], diff)
    return colour & (d <= 24)


def _best_window(strength: np.ndarray, n: int) -> int:
    """Start of the n consecutive entries with the most content (JPEG blends teal into the edges)."""
    if strength.size <= n:
        return 0
    c = np.concatenate(([0.0], np.cumsum(strength)))
    return int(np.argmax(c[n:] - c[:-n]))


def find_tiles(rgb: np.ndarray) -> list[Tile]:
    """Every map on a teal tile, with its label box below it (empty for any other picture).

    In a column of tiles, every row of the teal background runs across the whole tile; the maps
    are the tall gaps between such rows, and a label box sits on the teal below each map."""
    H, W = rgb.shape[:2]
    if H < 64 or W < 64:
        return []
    sub = rgb[::4, ::4, :3].astype(np.int16)  # quick look first: hardly any teal, no sheet
    if ((sub[..., 0] <= 40) & (np.abs(sub[..., 1] - sub[..., 2]) <= 22) & (sub[..., 1] >= 100)
            & (sub[..., 1] <= 156)).mean() < 0.02:
        return []
    T = teal_mask(rgb)
    if T.mean() < 0.02:
        return []
    light = (rgb[..., :3] >= 215).all(axis=2)
    found = []  # (tile column, map rows, label strip rows)
    for cx0, cx1 in _runs(T.mean(axis=0) >= 0.08, 32):  # columns of tiles
        f = T[:, cx0: cx1 + 1].mean(axis=1)  # teal share of each row
        bands = _runs(f >= 0.8, 2)
        for (_, a1), (b0, _) in zip(bands, bands[1:]):
            g0, g1 = a1 + 1, b0 - 1
            if g1 - g0 + 1 < 32 or np.mean(f[g0: g1 + 1] <= 0.35) < 0.8:
                continue  # not a map (a label strip, a separator)
            nxt = [r for r in bands if r[0] > g1]  # the teal under the map, then the label strip
            s0 = nxt[0][1] + 1 if nxt else H
            s1 = (nxt[1][0] - 1) if len(nxt) > 1 else min(H - 1, s0 + 47)
            found.append(((cx0, cx1), (g0, g1), (s0, min(s1, s0 + 47))))
    if not found:
        return []
    # exact edges: content is far from teal; the background, and the rows JPEG blends with it, are near
    dist = np.abs(rgb[..., :3].astype(np.int16) - np.array(TEAL, np.int16)).sum(axis=2).astype(np.float32)
    boxes = []
    for (cx0, cx1), (g0, g1), _ in found:
        top = max(0, g0 - 6)
        tealish = (dist[top: max(top + 1, g0 - 1), :] <= 90).mean(axis=0) >= 0.5  # the tile's width, above the map
        mid = (cx0 + cx1) // 2
        span = [r for r in _runs(tealish, 16) if r[0] <= mid <= r[1]]
        tx0, tx1 = span[0] if span else (cx0, cx1)
        y_lo, y_hi = max(0, g0 - 1), min(H - 1, g1 + 1)
        dr = dist[y_lo: y_hi + 1, tx0: tx1 + 1].mean(axis=1)
        rr = _runs(dr >= 0.5 * np.median(dr), 16)
        if not rr:
            boxes.append(None)
            continue
        a, b = max(rr, key=lambda r: r[1] - r[0])
        y0, y1 = y_lo + a, y_lo + b
        dc = dist[y0 + 2: max(y0 + 3, y1 - 1), tx0: tx1 + 1].mean(axis=0)
        cr = _runs(dc >= 0.5 * np.median(dc), 16)
        if not cr:
            boxes.append(None)
            continue
        a, b = max(cr, key=lambda r: r[1] - r[0])
        boxes.append((tx0 + a, y0, tx0 + b, y1))
    # one stage scan: every map has the same size; a tile off by a row (JPEG) takes its best window
    sizes = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for b in boxes if b]
    if not sizes:
        return []
    w, h = max(set(sizes), key=lambda v: (sizes.count(v), -v[0] * v[1]))
    # tiles in one row of the grid share their rows, tiles in one column their columns: a map whose
    # own edge looks like the background (smooth blue-green signal) takes the place of the others
    good = [b for b in boxes if b and (b[2] - b[0] + 1, b[3] - b[1] + 1) == (w, h)]
    for k, b in enumerate(boxes):
        if b is None:
            continue
        same_row = [g[1] for g in good if abs(g[1] - b[1]) <= h // 2]
        same_col = [g[0] for g in good if abs(g[0] - b[0]) <= w // 2]
        if same_row and same_col:
            y0, x0 = int(np.median(same_row)), int(np.median(same_col))
            boxes[k] = (x0, y0, x0 + w - 1, y0 + h - 1)
    tiles = []
    for ((cx0, cx1), _, (s0, s1)), bx in zip(found, boxes):
        if bx is None:
            continue
        x0, y0, x1, y1 = bx
        if abs((x1 - x0 + 1) - w) <= 2 and abs((y1 - y0 + 1) - h) <= 2 and (y1 - y0 + 1) != h:
            lo_y = max(0, y0 - 2)  # rows: teal above and below, so the most distant rows are the map
            y0 = lo_y + _best_window(dist[lo_y: y1 + 3, x0: x1 + 1].mean(axis=1), h)
            y1 = y0 + h - 1
        cx0, cx1 = min(cx0, x0), max(cx1, x1)
        tile = Tile(box=(x0, y0, x1, y1))
        if s0 <= s1:
            lf = light[s0: s1 + 1, cx0: cx1 + 1].mean(axis=1)
            rows = _runs((lf >= 0.15) & (lf < 0.9), 6)  # partly white: the box on the teal
            if rows:
                ly0, ly1 = s0 + rows[0][0], s0 + rows[0][1]
                cols = _runs(T[ly0: ly1 + 1, cx0: cx1 + 1].mean(axis=0) <= 0.2, 12)  # the box: not teal
                cols = [c for c in cols if c[0] <= 0.25 * (cx1 - cx0 + 1)]  # at the left of the tile
                if cols:
                    bx0, bx1 = cols[0]
                    _read_box(rgb, tile, (cx0 + bx0, ly0, cx0 + bx1, ly1))
        tiles.append(tile)
    return tiles


def _read_box(rgb: np.ndarray, tile: Tile, box: tuple[int, int, int, int]) -> None:
    """Scale bar, name and length in a tile's label box (pixels of the box; nothing when no bar)."""
    x0, y0, x1, y1 = box
    px = rgb[y0: y1 + 1, x0: x1 + 1, :3].astype(np.float32)
    grey = px.mean(axis=2)
    chroma = px.max(axis=2) - px.min(axis=2)
    ink = np.clip((215.0 - grey) / 160.0, 0.0, 1.0) * (chroma < 60)  # black text on white, not teal
    strong = ink > 0.5
    best = (0, 0, 0, 0)  # run length, row, first, last
    for r in range(strong.shape[0]):
        for a, b in _runs(strong[r]):
            if b - a + 1 > best[0]:
                best = (b - a + 1, r, a, b)
    n, r, a, b = best
    if n < 12:
        return
    rows = [rr for rr in range(max(0, r - 3), min(strong.shape[0], r + 4))
            if strong[rr, a + n // 4: b - n // 4 + 1].mean() > 0.9]
    if not rows or rows[-1] - rows[0] > 3:  # a thin line, not a block of text
        return
    r0, r1 = rows[0], rows[-1]
    # its length with partial pixels at the ends (JPEG softens them)
    col = ink[max(0, r0 - 1): r1 + 2].sum(axis=0)
    mid = float(np.median(col[a + 2: b - 1])) if b - a > 4 else float(col[a: b + 1].max())
    lo, hi = max(0, a - 2), min(ink.shape[1], b + 3)
    tile.bar_px = float(np.minimum(1.0, col[lo:hi] / max(mid, 1e-6)).sum())
    tile.bar_x, tile.bar_y = (x0 + a, x0 + b), y0 + r0
    rest = ink.copy()
    rest[max(0, r0 - 1): r1 + 2, lo:hi] = 0.0
    tile.name_ink = textread.crop_ink(rest[:, :lo])
    tile.scale_ink = textread.crop_ink(rest[:, hi:])
    tile.label_box = box
    tile.preview = np.clip(grey, 0, 255).astype(np.uint8)


def sheet_tiles(rgb: np.ndarray) -> list[Tile]:
    """The tiles of a JEOL EPMA map sheet: tiles with a scale bar whose maps share one size (one
    stage scan).  Empty when the picture is not such a sheet."""
    tiles = [t for t in find_tiles(rgb) if t.bar_px > 0 and t.name_ink is not None]
    if not tiles:
        return []
    sizes = [(t.box[2] - t.box[0] + 1, t.box[3] - t.box[1] + 1) for t in tiles]
    w, h = max(set(sizes), key=sizes.count)
    tiles = [t for t, (tw, th) in zip(tiles, sizes) if abs(tw - w) <= 2 and abs(th - h) <= 2]
    return sorted(tiles, key=lambda t: (t.box[1] // max(1, h // 2), t.box[0]))  # rows of the grid, left to right


def is_sheet(rgb: np.ndarray) -> bool:
    return bool(sheet_tiles(rgb))

"""Plain SEM / TEM images: strip the instrument's info band (and a map's colour bar), read its scale.

Most microscopes save the picture with a band of text along one edge: magnification, voltage,
date and a scale bar.  JEOL draws it white-on-black, Hitachi tabletop SEMs black-on-white,
Thermo Fisher (FEI) a grey table, TEM-EDS software a white strip with the element name.  The band
is found as the block of rows at the edge made of a flat background plus text and lines,
unlike the continuous grey levels of a micrograph.

The scale bar is a straight stroke in the band: with short end ticks (Hitachi ruler, FEI, the
outlined bar of TEM-EDS maps), split in two by its label (FEI), or a solid block (JEOL).  Its
label is the nearby text that reads as "number + unit".

Some field-emission SEMs (JEOL) draw the text and the bar straight onto the picture instead.
That overlay is found as a thin, straight, pure white (or black) line next to a row of text of the
same colour near the edge; the rows from the topmost mark to the edge are cut off.

Pixel size from metadata (FEI/Thermo and ZEISS TIFF tags, the .txt that JEOL and Hitachi save
next to the image, or a DigitalMicrograph file) takes precedence over the bar when available.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import textread


@dataclass
class Band:
    side: str  # "bottom" | "top"
    start: int  # first row of the band
    height: int
    bg: float  # background grey level of the band (overlays: nan)
    overlay: bool = False  # text drawn onto the picture itself (no band of its own): rows are cut off
    bright: bool = True  # overlays: white text (False: black)
    marks: np.ndarray | None = field(default=None, repr=False, compare=False)  # overlays: where the text is
    legend: bool = False  # the colour key of an AZtec layered image drawn onto the picture: rows are cut off


def dilate(m: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation with a (2r+1) x (2r+1) square."""
    out = m.copy()
    h, w = m.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy or dx:
                out[max(0, dy): h + min(0, dy), max(0, dx): w + min(0, dx)] |= \
                    m[max(0, -dy): h + min(0, -dy), max(0, -dx): w + min(0, -dx)]
    return out


def gray(rgba: np.ndarray) -> np.ndarray:
    return rgba[..., :3].astype(np.float32).mean(axis=2)


def _row_mode(r: np.ndarray) -> tuple[float, float]:
    """Most frequent grey level of a row (±3) and the share of the row within it."""
    hist = np.bincount(np.clip(np.round(r), 0, 255).astype(np.int64), minlength=256)
    win = np.convolve(hist, np.ones(7), mode="same")
    m = int(np.argmax(win))
    return float(m), float(win[m] / r.size)


def _bar_row(r: np.ndarray, bg: float, near_bg: float) -> bool:
    """A row through a long scale bar: background plus one flat level drawn as one to three long
    strokes (a bar over more than half the width leaves less than half the row as background)."""
    ink = np.abs(r - bg) > 4
    if ink.mean() < 0.1 or near_bg < 0.05:  # a bar never fills the whole row
        return False
    m, share = _row_mode(r[ink])
    if abs(m - bg) < 60 or near_bg + share * ink.mean() < 0.97:  # two clearly different levels only
        return False
    on = np.abs(r - m) <= 3
    d = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
    lengths = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    return 0 < len(lengths) <= 3 and 0.1 * r.size <= lengths.max() < 0.9 * r.size


def find_band(g: np.ndarray, rgb: np.ndarray | None = None) -> Band | None:
    """Info band along the bottom (or top) edge, or None.

    A band row is either mostly flat background with some text (>= 45 % within ±4 of the band's
    background), a row through a long ruler (background plus one flat level in a few long strokes,
    e.g. Hitachi tabletop SEMs at low magnification) or a drawn line (>= 85 % of one flat level).
    Lines of another colour may run for at most 3 rows (table borders); a longer run is picture
    content (e.g. an empty map region).
    The marks in a band are text and lines (grey, glyph-sized); scattered or coloured dots on a
    flat background are a sparse map or a dark-field picture, not a band."""
    H, W = g.shape
    found = []
    for side in ("bottom", "top"):
        rows = range(H - 1, -1, -1) if side == "bottom" else range(H)
        modes: list[float] = []
        n = other = 0
        bg = None
        for y in rows:
            if n > 0.35 * H:
                break
            r = g[y]
            m, flat = _row_mode(r)
            if flat >= 0.85:
                modes.append(m)
                bg = float(np.median(modes))
            if bg is None:
                break
            near_bg = float((np.abs(r - bg) <= 4).mean())
            if near_bg >= 0.45 or _bar_row(r, bg, near_bg):
                other = 0
            elif flat >= 0.85:  # a line of another level (table border, ruler)
                other += 1
                if other > 3:
                    n -= other - 1
                    break
            else:
                # Lines just before the picture: the border of a grey databar (FEI) belongs to the
                # band; next to a white/black band they may be picture rows (an empty map row).
                if not (25 < bg < 230 and other <= 2):
                    n -= other
                break
            n += 1
        if bg is None or n < 6 or n > 0.3 * H:
            continue
        band = g[H - n:] if side == "bottom" else g[:n]
        if (np.abs(band - bg) >= 60).mean() < 0.003:  # no text or graphics: just a border
            continue
        if (np.abs(band - bg) <= 4).mean() < 0.55:  # a drawn band is mostly flat background
            continue
        ink = np.abs(band - bg) >= 60
        if rgb is not None:
            brgb = (rgb[H - n:] if side == "bottom" else rgb[:n])[..., :3].astype(np.int16)
            chroma = brgb.max(axis=2) - brgb.min(axis=2)
            if (chroma[ink] > 40).mean() > 0.5:  # coloured marks: an element map's signal
                continue
        comps = components(ink)
        marks = sum(c[4] for c in comps if c[3] - c[1] + 1 >= 5 or c[2] - c[0] + 1 >= 8)
        if marks < 0.5 * ink.sum():  # mostly isolated dots, not glyphs or lines
            continue
        exact = drawn_share(band, bg)
        if exact < 0.6:  # the noise of a micrograph, not a drawn background
            continue
        found.append((Band(side, H - n if side == "bottom" else 0, n, bg), exact))
    if not found:
        return None
    # a band drawn by the instrument has one exact background level; a flat part of the picture
    # (dark sky above a sample, say) only looks flat: prefer the drawn one, then the taller
    return max(found, key=lambda be: (be[1] >= 0.8, be[0].height))[0]


def drawn_share(band: np.ndarray, bg: float) -> float:
    """Of the band's background-like pixels (within 4 of bg), the share exactly at its most common
    level (±1): about 1 for a drawn band (even as JPEG: 0.73 at the worst), 0.5-0.7 for a flat
    stretch of micrograph with its noise."""
    v = np.clip(np.round(band), 0, 255).astype(np.int64)
    m = int(np.bincount(v.ravel(), minlength=256).argmax())
    near = np.abs(band - bg) <= 4
    return float(((np.abs(v - m) <= 1) & near).sum() / max(1, near.sum()))


def image_box(shape: tuple[int, int], band: Band | None, band_px: int | None = None) -> tuple[int, int, int, int]:
    """Box (x0, y0, x1, y1, inclusive) of the micrograph once `band_px` rows are removed."""
    H, W = shape
    side = band.side if band else "bottom"
    n = band.height if (band and band_px is None) else max(0, int(band_px or 0))
    n = min(n, H - 1)
    return (0, n, W - 1, H - 1) if side == "top" else (0, 0, W - 1, H - 1 - n)


def find_colorbar(rgb: np.ndarray) -> tuple[str, int] | None:
    """Intensity colour bar along the left (or right) edge of a map: identical columns holding a
    smooth, monotonic gradient over most of the height.  Returns (side, width) or None."""
    H, W = rgb.shape[:2]
    if W < 40 or H < 40:
        return None
    x = rgb.astype(np.float32)
    rows = np.arange(H)
    for side in ("left", "right"):
        ref = x[:, 0] if side == "left" else x[:, W - 1]
        w = 0
        for k in range(min(W // 8, 60)):
            col = x[:, k] if side == "left" else x[:, W - 1 - k]
            if np.abs(col - ref).mean() > 8:
                break
            w += 1
        if w < 4 or w >= W // 8:
            continue
        # the edge between the strip and the picture: the strongest column-to-column change just
        # after the identical columns (JPEG blurs it over a couple of columns)
        best_k, best_d = None, 0.0
        for k in range(w, min(w + 4, W // 8)):
            a = x[:, k] if side == "left" else x[:, W - 1 - k]
            b = x[:, k - 1] if side == "left" else x[:, W - k]
            d = float(np.abs(a - b).mean())
            if d > best_d:
                best_k, best_d = k, d
        if best_k is None or best_d < 15:  # no clear edge
            continue
        w = best_k
        v = ref.mean(axis=1)
        if v.max() - v.min() < 60:
            continue
        r = np.corrcoef(rows, v)[0, 1]
        if abs(r) < 0.95:  # a gradient from top to bottom (either direction)
            continue
        return side, w
    return None


# --------------------------------------------------------------------------- text drawn on the picture
def components(on: np.ndarray, diag: bool = True) -> list[tuple[int, int, int, int, int]]:
    """8-connected (diag=False: 4-connected) components of a mask as (x0, y0, x1, y1, pixel count),
    from horizontal runs."""
    reach = 1 if diag else 0  # runs touching only at a corner join only with 8-connectivity
    parent: list[int] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    runs: list[tuple[int, int, int]] = []
    prev: list[int] = []
    for y in range(on.shape[0]):
        d = np.diff(np.concatenate(([0], on[y].astype(np.int8), [0])))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
        cur, k = [], 0
        for a, b in zip(starts.tolist(), ends.tolist()):
            i = len(runs)
            runs.append((y, a, b))
            parent.append(i)
            cur.append(i)
            while k < len(prev) and runs[prev[k]][2] < a - reach:
                k += 1
            j = k
            while j < len(prev) and runs[prev[j]][1] <= b + reach:
                ra, rb = find(i), find(prev[j])
                if ra != rb:
                    parent[ra] = rb
                j += 1
        prev = cur
    comps: dict[int, list[int]] = {}
    for i, (y, a, b) in enumerate(runs):
        c = comps.setdefault(find(i), [a, y, b, y, 0])
        c[0], c[1], c[2], c[3] = min(c[0], a), min(c[1], y), max(c[2], b), max(c[3], y)
        c[4] += b - a + 1
    return [tuple(c) for c in comps.values()]


def _text_line(glyphs: list, x0: int, x1: int, y0: int, y1: int):
    """A line of text next to rows y0..y1 and near columns x0..x1: three or more marks of similar
    height on one baseline, some of them as close together as letters of a word.
    Returns (top, bottom, glyph height, members) or None."""
    best = None
    for b in sorted({c[3] for c in glyphs}):
        grp = [c for c in glyphs if abs(c[3] - b) <= 2]
        if len(grp) < 3:
            continue
        h = float(np.median([c[3] - c[1] + 1 for c in grp]))
        grp = [c for c in grp if 0.6 * h <= c[3] - c[1] + 1 <= 1.5 * h]
        if len(grp) < 3:
            continue
        top, bot = min(c[1] for c in grp), max(c[3] for c in grp)
        if max(0, top - y1, y0 - bot) > 2 * h:
            continue
        if not any(c[2] >= x0 - 3 * h and c[0] <= x1 + 3 * h for c in grp):
            continue
        xs = sorted(grp, key=lambda c: c[0])
        if sum(1 for a, c in zip(xs, xs[1:]) if c[0] - a[2] <= h) < 2:  # no word: scattered marks
            continue
        if best is None or len(grp) > len(best[3]):
            best = (top, bot, int(round(h)), grp)
    return best


def find_overlay(g: np.ndarray) -> tuple[Band, "Bar"] | None:
    """Scale bar and text drawn directly onto the picture near its bottom (or top) edge, as JEOL
    field-emission SEMs do ("5.0kV  x22,000  ____ 1µm  7.9 10 LED" in white over the image).

    Found as a thin, straight, uniformly white (or black) horizontal line next to a row of
    text-sized marks of the same colour.  Returns the rows to cut off (from the topmost overlay
    mark to the edge) and the bar, with coordinates relative to those rows."""
    H, W = g.shape
    if H < 60 or W < 60:
        return None
    zone = max(30, int(0.2 * H))
    best = None
    for side in ("bottom", "top"):
        r0 = H - zone if side == "bottom" else 0
        reg = g[r0: r0 + zone]
        for bright in (True, False):
            on = reg >= 225 if bright else reg <= 30
            if on.mean() > 0.5:  # mostly saturated: not text on a picture
                continue
            if on.mean() > 0.1:  # a busy region (many saturated pixels): no reliable overlay
                continue
            bars = []
            for x0, x1, y0, y1 in _blocks(on, max(30, int(0.06 * W))):
                L, th = x1 - x0 + 1, y1 - y0 + 1
                if L >= 0.85 * W or th > max(10, int(0.012 * H)):
                    continue
                outside = [y for y in (y0 - 2, y0 - 3, y1 + 2, y1 + 3) if 0 <= y < zone]
                if len(outside) < 2 or any(on[y, x0: x1 + 1].mean() > 0.2 for y in outside):
                    continue  # the edge of a bright region, not a drawn line
                bars.append((x0, x1, y0, y1))
            if not bars:
                continue
            comps = components(on)
            for x0, x1, y0, y1 in bars:
                glyphs = []
                for c in comps:
                    cx0, cy0, cx1, cy1, n = c
                    w, h = cx1 - cx0 + 1, cy1 - cy0 + 1
                    if cy0 <= y1 and cy1 >= y0 and cx0 <= x1 and cx1 >= x0:
                        continue  # the bar itself
                    if 5 <= h <= max(14, int(0.1 * H)) and w <= 3 * h + 4 and 0.08 <= n / (w * h) <= 0.95:
                        glyphs.append(c)
                line = _text_line(glyphs, x0, x1, y0, y1)
                if line is None:
                    continue
                ltop, lbot, hgt, members = line
                top, bot = min(y0, ltop), max(y1, lbot)
                # glyphs hanging below the line (descenders, a comma) and a second line of text
                for c in glyphs:
                    if c[1] <= bot + hgt and c[3] >= top - hgt and c[3] - c[1] + 1 <= 2 * hgt:
                        top, bot = min(top, c[1]), max(bot, c[3])
                second = _text_line([c for c in glyphs if c[3] < top or c[1] > bot], 0, W - 1, top, bot)
                if second is not None:
                    top, bot = min(top, second[0]), max(bot, second[1])
                pad = 3  # anti-aliasing and the shadow drawn under the text
                if side == "bottom":
                    if H - (r0 + bot) > 0.1 * H:  # must reach the edge region
                        continue
                    start = max(0, r0 + top - pad)
                    band = Band("bottom", start, H - start, float("nan"), overlay=True, bright=bright)
                else:
                    if r0 + top > 0.1 * H:
                        continue
                    end = min(H, r0 + bot + pad + 1)
                    band = Band("top", 0, end, float("nan"), overlay=True, bright=bright)
                off = r0 - band.start
                # the drawn marks: saturated strokes in the text rows, widened over their anti-aliasing
                rows = g[band.start: band.start + band.height]
                ron = rows >= 225 if bright else rows <= 30
                lo, hi = max(0, top + off - pad), min(band.height, bot + off + pad + 1)
                keep = np.zeros_like(ron)
                keep[lo:hi] = ron[lo:hi]
                band.marks = dilate(keep, 2)
                ink = (np.clip((reg - 205.0) / 40.0, 0, 1) if bright else np.clip((50.0 - reg) / 40.0, 0, 1))
                mid = ink[(y0 + y1) // 2, max(0, x0 - 2): x1 + 3]
                bar = Bar(x0, x1, y0 + off, y1 + off, float(mid.sum()), False, bright)
                score = len(members)
                if best is None or score > best[0]:
                    best = (score, band, bar)
    return (best[1], best[2]) if best else None


# --------------------------------------------------------------------------- scale bar in the band
@dataclass
class Bar:
    x0: int
    x1: int
    y0: int
    y1: int
    length_px: float
    ticks: bool
    bright: bool = False  # drawn lighter than the band background
    gap: tuple[int, int] | None = None  # columns of the label when the bar is split by it
    marks: list[float] | None = None  # a ruler drawn as a row of ticks (Hitachi FE-SEM): their centres


def inks(b: np.ndarray, bg: float) -> tuple[np.ndarray, np.ndarray]:
    """Ink strength 0..1 of graphics lighter / darker than the band background."""
    up = np.clip((b - bg) / max(255.0 - bg, 40.0), 0, 1)
    down = np.clip((bg - b) / max(bg, 40.0), 0, 1)
    return up, down


def _box_mean(x: np.ndarray, w: np.ndarray, r: int) -> np.ndarray:
    """Mean of x over the pixels where w is 1, in a (2r+1) x (2r+1) box (integral images)."""
    def box(a):
        c = np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge").cumsum(0).cumsum(1)
        return c[2 * r + 1:, 2 * r + 1:] - c[:-2 * r - 1, 2 * r + 1:] - c[2 * r + 1:, :-2 * r - 1] + c[:-2 * r - 1, :-2 * r - 1]
    num, den = box(x * w), box(w)
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def overlay_ink(rows: np.ndarray, bright: bool) -> np.ndarray:
    """Ink 0..1 of white (or black) text drawn over a picture: how far a pixel stands out from
    the local picture around it, so that thin, anti-aliased strokes count fully."""
    x = rows.astype(np.float64) if bright else 255.0 - rows.astype(np.float64)
    pic = (x < 200).astype(np.float64)  # text is near white; everything else is picture (or shadow)
    bg = _box_mean(x, pic, 10)
    bg = np.where(np.isnan(bg), np.median(x[pic > 0]) if pic.any() else 128.0, bg)
    return np.clip((x - bg - 25.0) / 35.0, 0, 1) * (x >= 150)


def band_inks(g: np.ndarray, band: Band) -> tuple[np.ndarray, np.ndarray]:
    """(lighter, darker) ink of the band's rows.  An overlay has no flat background: its ink is
    the white or black of the drawn text against the picture around it."""
    rows = g[band.start: band.start + band.height]
    if not band.overlay:
        return inks(rows, band.bg)
    zero = np.zeros_like(rows)
    ink = overlay_ink(rows, band.bright).astype(rows.dtype)
    if band.marks is not None and band.marks.shape == ink.shape:
        ink = ink * band.marks  # only the drawn marks: no specks of the picture between them
    return (ink, zero) if band.bright else (zero, ink)


def _blocks(on: np.ndarray, min_len: int) -> list[tuple[int, int, int, int]]:
    """Horizontal strokes: runs of >= min_len pixels, stacked over consecutive rows with the
    same extent (±1).  Returns (x0, x1, y0, y1), inclusive."""
    h, w = on.shape
    runs = []
    for y in range(h):
        d = np.diff(np.concatenate(([0], on[y].astype(np.int8), [0])))
        s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        runs.append([(int(a), int(b) - 1) for a, b in zip(s, e) if b - a >= min_len])
    out, used = [], set()
    for y in range(h):
        for (a, b) in runs[y]:
            if (y, a, b) in used:
                continue
            y1 = y
            while y1 + 1 < h:
                nxt = [(c, d) for (c, d) in runs[y1 + 1] if abs(c - a) <= 1 and abs(d - b) <= 1]
                if not nxt:
                    break
                y1 += 1
                used.add((y1, *nxt[0]))
            out.append((a, b, y, y1))
    return out


def _tick(on: np.ndarray, xc: int, y0: int, y1: int) -> float | None:
    """Centre of a short (2-10 px), thin stroke sticking out above or below a stroke at column ~xc."""
    h, w = on.shape
    lo, hi = max(0, xc - 2), min(w, xc + 3)
    best = None
    for ys in (range(y0 - 1, max(-1, y0 - 12), -1), range(y1 + 1, min(h, y1 + 12))):
        n = 0
        for y in ys:
            if on[y, lo:hi].any():
                n += 1
            else:
                break
        if 2 <= n <= 10:
            win = on[list(ys)[:n], lo:hi]
            if win.any(axis=0).sum() <= 4:
                colsum = win.sum(axis=0)  # the stroke is the column with the most ink
                c = np.flatnonzero(colsum == colsum.max()) + lo
                best = float(c.mean())
    return best


def _tick_row(ink: np.ndarray, on: np.ndarray, width_px: int, bright: bool) -> Bar | None:
    """A ruler drawn as a row of short ticks or dots (Hitachi FE-SEM: 11 marks; the label gives the
    length from the first to the last): five or more marks of one height on one line, evenly spaced.
    Returns a Bar from the first to the last mark's centre (ink-weighted, sub-pixel)."""
    h_max = max(4, int(0.6 * on.shape[0]))
    marks = []
    for x0, y0, x1, y1, n in components(on):
        h, w = y1 - y0 + 1, x1 - x0 + 1
        if 3 <= h <= h_max and w <= max(6, 0.5 * h):
            win = ink[y0: y1 + 1, x0: x1 + 1] * on[y0: y1 + 1, x0: x1 + 1]
            cx = float((win.sum(axis=0) * np.arange(x0, x1 + 1)).sum() / max(win.sum(), 1e-9))
            marks.append((cx, x0, y0, x1, y1))
    best = None
    for ref in marks:
        row = sorted(m for m in marks if abs(m[2] - ref[2]) <= 1 and abs(m[4] - ref[4]) <= 1)
        if len(row) < 5:
            continue
        xs = np.array([m[0] for m in row])
        gaps = np.diff(xs)
        # the longest run of even steps (a letter of the same height next to the ruler is not a step);
        # whole-pixel ticks of a fractional step alternate by a pixel
        med = float(np.median(gaps))
        even = np.abs(gaps - med) <= max(1.5, 0.1 * med)
        run, start, best_run = 0, 0, (0, 0)
        for k, ok in enumerate(even):
            run, start = (run + 1, start) if ok else (0, k + 1)
            if run > best_run[0]:
                best_run = (run, start)
        n_gaps, k0 = best_run
        if n_gaps < 4:  # five marks at least
            continue
        sel = row[k0: k0 + n_gaps + 1]
        step = float(np.median(np.diff([m[0] for m in sel])))
        if step < 3 * max(m[3] - m[1] + 1 for m in sel):  # marks apart, not the strokes of a word
            continue
        L = sel[-1][0] - sel[0][0]
        if L < 0.04 * width_px:
            continue
        if best is None or len(sel) > len(best.marks):
            y0, y1 = min(m[2] for m in sel), max(m[4] for m in sel)
            best = Bar(int(round(sel[0][0])), int(round(sel[-1][0])), y0, y1, float(L), True, bright,
                       marks=[m[0] for m in sel])
    return best


def find_bar(up: np.ndarray, down: np.ndarray, width_px: int) -> Bar | None:
    cands = []  # (preference, length, Bar)
    # strokes of letters are shorter than half the band height; a scale bar is longer
    min_len = max(10, int(0.012 * width_px), int(0.5 * up.shape[0]))
    for bright, ink in ((True, up), (False, down)):
        on = ink > 0.5
        # a line across (nearly) the whole band is a border, not a bar; a ruler with end ticks may
        # still be long
        blocks = [b for b in _blocks(on, min_len) if b[1] - b[0] + 1 < 0.97 * width_px]
        info = []
        for (x0, x1, y0, y1) in blocks:
            tl, tr = _tick(on, x0, y0, y1), _tick(on, x1, y0, y1)
            info.append((x0, x1, y0, y1, tl, tr))
            if tl is not None and tr is not None:
                cands.append((3, tr - tl, Bar(x0, x1, y0, y1, tr - tl, True, bright)))
            elif x1 - x0 + 1 >= 0.85 * width_px:
                continue
            elif y1 - y0 + 1 >= 4 and tl is None and tr is None:  # solid bar (JEOL)
                mid = ink[(y0 + y1) // 2, max(0, x0 - 2): x1 + 3]
                mid = np.where(mid < 0.05, 0.0, mid)  # the background's own faint level is not bar
                cands.append((1, float(mid.sum()), Bar(x0, x1, y0, y1, float(mid.sum()), False, bright)))
        # a bar split by its label: two strokes on the same rows, ticks only at the outer ends
        for a in info:
            for b in info:
                if b[0] <= a[1] or a[4] is None or b[5] is None or a[5] is not None or b[4] is not None:
                    continue
                if abs(a[2] - b[2]) > 1 or abs(a[3] - b[3]) > 1:
                    continue
                gap = (a[1] + 1, b[0] - 1)
                if gap[1] - gap[0] < 6 or not on[:, gap[0]: gap[1] + 1].any():
                    continue
                L = b[5] - a[4]
                cands.append((3, L, Bar(a[0], b[1], a[2], a[3], L, True, bright, gap)))
        tr = _tick_row(ink, on, width_px, bright)  # a ruler of ticks (Hitachi FE-SEM)
        if tr is not None:
            cands.append((2, tr.length_px, tr))
    if not cands:
        return None
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
    return cands[0][2]


def _remove_component(on: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    """Copy of `on` without the 8-connected component(s) touching the seeds (the bar and its ticks)."""
    out = on.copy()
    h, w = on.shape
    q = deque((y, x) for y, x in seeds if 0 <= y < h and 0 <= x < w and out[y, x])
    for y, x in q:
        out[y, x] = False
    while q:
        y, x = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and out[yy, xx]:
                    out[yy, xx] = False
                    q.append((yy, xx))
    return out


def _strip_borders(on: np.ndarray) -> np.ndarray:
    """Without the lines of a table (FEI databar): vertical strokes over most of the band height
    and horizontal strokes much longer than any word."""
    h, w = on.shape
    out = on.copy()
    if h >= 12:
        for x in np.flatnonzero(on.sum(axis=0) >= 0.7 * h):
            d = np.diff(np.concatenate(([0], on[:, x].astype(np.int8), [0])))
            for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                if b - a >= 0.7 * h:  # with its anti-aliased neighbours
                    out[a:b, max(0, x - 1): x + 2] = False
    long = max(60, int(0.15 * w), 8 * h)
    for y in np.flatnonzero(on.sum(axis=1) >= long):
        d = np.diff(np.concatenate(([0], on[y].astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a >= long:
                out[y, a:b] = False
    return out


def _split_lines(on: np.ndarray, lines: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Row ranges taller than one line of text (two lines bridged by a logo spanning both, as in the
    two-line JEOL band) split at the emptiest row between them.  A single line, with its ascenders
    and descenders, is never split."""
    hs = [c[3] - c[1] + 1 for c in components(on) if c[3] - c[1] + 1 >= 5]
    if not hs:
        return lines
    h = float(np.median(hs))  # glyph height
    counts = on.sum(axis=1).astype(float)
    out = []
    for s, e in lines:
        if e - s <= 1.7 * h:
            out.append((s, e))
            continue
        seg = counts[s:e]
        dense = seg > 0.25 * np.percentile(seg[seg > 0], 75)  # rows through the body of a line of text
        d = np.diff(np.concatenate(([0], dense.astype(np.int8), [0])))
        bodies = [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))
                  if b - a >= 0.5 * h]
        if len(bodies) < 2:  # one line with tall marks (ascenders, slashes, descenders)
            out.append((s, e))
            continue
        cuts = [s + (b0 + a1) // 2 for (_, b0), (a1, _) in zip(bodies, bodies[1:])]  # midway between lines
        edges = [s] + cuts + [e]
        out += list(zip(edges, edges[1:]))
    return sorted((s, e) for s, e in out if e - s >= 5)


def text_phrases(ink: np.ndarray, bar: Bar | None) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Words and runs of up to three words in the band, as (ink mask, bbox), nearest to the bar first
    (the label inside a split bar comes first of all)."""
    on = ink > 0.5
    if bar is not None:
        cols = [int(round(c)) for c in bar.marks] if bar.marks else [bar.x0, bar.x1]
        seeds = [(y, x + dx) for y in range(bar.y0, bar.y1 + 1) for x in cols for dx in (-1, 0, 1)]
        if bar.gap:
            seeds += [(y, x) for y in range(bar.y0, bar.y1 + 1) for x in (bar.gap[0] - 1, bar.gap[1] + 1)]
        on = _remove_component(on, seeds)
    on = _strip_borders(on)
    rows = on.any(axis=1)
    lines, y = [], 0
    while y < len(rows):
        if rows[y]:
            s = y
            while y < len(rows) and rows[y]:
                y += 1
            if y - s >= 5:
                lines.append((s, y))
        else:
            y += 1
    lines = _split_lines(on, lines)
    out = []  # (words, line top, line bottom)
    for a, b in lines:
        hgt = b - a
        # glyphs closer than ~0.6 line heights form a word ("1" has wide side bearings in Arial)
        segs = textread.segments(on[a:b].astype(float), gap=max(3, int(0.6 * hgt)))
        words = [(s, e) for s, e in segs if e - s >= 1]
        for i in range(len(words)):  # a word, or up to three words separated by ordinary spaces
            out.append((words[i: i + 1], a, b))
            for j in (i + 1, i + 2):
                if j < len(words) and words[j][0] - words[j - 1][1] - 1 <= 1.2 * hgt:
                    out.append((words[i: j + 1], a, b))
                else:
                    break
    res = []
    for gr, a, b in out:
        x0, x1 = gr[0][0], gr[-1][1]
        m = ink[a:b, x0: x1 + 1] * on[a:b, x0: x1 + 1]
        res.append((m, (x0, a, x1, b - 1)))
    if bar is not None:
        def dist(item):
            x0, y0, x1, y1 = item[1]
            if bar.gap and x0 >= bar.gap[0] - 2 and x1 <= bar.gap[1] + 2:
                return -1
            dx = max(0, x0 - bar.x1, bar.x0 - x1)
            dy = max(0, y0 - bar.y1, bar.y0 - y1)
            return dx + 2 * dy
        res.sort(key=dist)
    return res


def _ends_with_m(m: np.ndarray) -> bool:
    """Every scale label ends in 'm' (nm, µm, um, mm): three or more stems in the last glyph."""
    segs = textread.segments(m, thr=0.5)
    if not segs:
        return False
    a, b = segs[-1]
    last = m[:, a: b + 1] > 0.5
    rows = np.flatnonzero(last.any(axis=1))
    if rows.size < 3:
        return False
    stems = 0
    for f in (0.45, 0.6, 0.75):
        r = last[rows[0] + int(f * (rows[-1] - rows[0]))]
        stems = max(stems, int((np.diff(np.concatenate(([0], r.astype(np.int8), [0]))) == 1).sum()))
    return stems >= 3


def read_band_scale(ink: np.ndarray, bar: Bar | None, width_px: int, reader: textread.Reader):
    """Best "number + unit" text near the bar -> (ReadResult, label mask, bbox) or (None, mask, bbox)."""
    best = None
    first = None
    for m, bbox in text_phrases(ink, bar)[:40]:
        if m.shape[0] < 5 or m.shape[1] < 6 or not _ends_with_m(m):
            continue
        if first is None:
            first = (m, bbox)
        r = reader.read_scale(m, fonts="band")
        v, u = textread.parse_length(r.text)
        if v is None or not r.text:
            continue
        if bar is not None:
            fov_um = v * textread.UNIT_TO_UM[u] / bar.length_px * width_px
            if not (0.0005 <= fov_um <= 50000):  # implausible for an SEM/TEM field of view
                continue
        if best is None or r.score > best[0].score + 0.02:
            best = (r, m, bbox)
        if r.score >= 0.97 or (r.score >= 0.85 and r.margin >= 0.05):  # confident: stop looking
            break
    if best is None:
        return (None, *first) if first else (None, None, None)
    return best


def read_band_element(ink: np.ndarray, bar: Bar | None, skip: tuple | None, reader: textread.Reader):
    """Element name printed in the band of a TEM-EDS map ('C K', 'Si K'): the right-most text that
    does not overlap the scale label.  Returns (ReadResult, mask) or (None, None)."""
    cands = [(m, bb) for m, bb in text_phrases(ink, bar)
             if skip is None or bb[2] < skip[0] or bb[0] > skip[2] or bb[3] < skip[1] or bb[1] > skip[3]]
    if bar is not None:  # not the bar's own label either
        cands = [(m, bb) for m, bb in cands if bb[0] > bar.x1 + 4 * (bb[3] - bb[1] + 1) or bb[2] < bar.x0]
    if not cands:
        return None, None
    right = max(bb[2] for _, bb in cands)
    group = [(m, bb) for m, bb in cands if bb[2] == right]  # "K", "C K": the longest ending there
    m, bb = max(group, key=lambda c: c[1][2] - c[1][0])
    return reader.read_element(m, fonts="band"), m


# --------------------------------------------------------------------------- metadata
def pixel_size_from_tags(tags: dict) -> float | None:
    """µm per pixel from FEI/Thermo (34682/34680) or ZEISS (34118) TIFF tags."""
    for t in (34682, 34680):
        v = tags.get(t)
        if v:
            s = v.decode("latin-1", "ignore") if isinstance(v, bytes) else str(v)
            m = re.search(r"\[Scan\][^\[]*?PixelWidth\s*=\s*([0-9.eE+-]+)", s)
            if m and float(m.group(1)) > 0:
                return float(m.group(1)) * 1e6
    v = tags.get(34118)
    if v:
        s = v.decode("latin-1", "ignore") if isinstance(v, bytes) else str(v)
        m = re.search(r"Image Pixel Size\s*=\s*([0-9.]+)\s*(pm|nm|µm|um|mm)", s)
        if m:
            f = {"pm": 1e-6, "nm": 1e-3, "µm": 1.0, "um": 1.0, "mm": 1e3}[m.group(2)]
            return float(m.group(1)) * f
    return None


def pixel_size_from_txt(text: str) -> float | None:
    """µm per pixel from the .txt saved next to the image (Hitachi 'PixelSize', JEOL micron bar)."""
    m = re.search(r"^\s*PixelSize\s*=\s*([0-9.eE+-]+)", text, re.M)  # Hitachi, nm
    if m and float(m.group(1)) > 0:
        return float(m.group(1)) * 1e-3
    bar = re.search(r"\$\$SM_MICRON_BAR\s+([0-9.]+)", text)  # JEOL: bar length in px ...
    mark = re.search(r"\$\$SM_MICRON_MARKER\s+([0-9.]+)\s*(nm|um|µm|mm)", text)  # ... and its value
    if bar and mark and float(bar.group(1)) > 0:
        return float(mark.group(1)) * textread.UNIT_TO_UM[mark.group(2).replace("um", "µm")] / float(bar.group(1))
    return None

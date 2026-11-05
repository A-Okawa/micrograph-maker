"""Lay out cropped panels as a figure and export it without degrading the pixels.

Pictures of different shapes can be lined up by width (cells of one size), by height (rows),
trimmed to one shape (crop), or arranged freely in nested rows and columns written with their
numbers (free, see tree.py); in every case a picture is scaled by one factor for x and y.
Labels and scale bars are sized in points, so they are the same size on every panel.

* Raster (PNG / TIFF): the output is sized so that every panel gets at least
  ``oversample`` output pixels per original pixel; nothing is ever shrunk.  Lossless only.
* Vector (PDF / SVG): each image is embedded with its original pixels (no resampling);
  element labels and scale bars are vector text/shapes you can still edit.
"""
from __future__ import annotations

import io
import json
import math
import threading
import zipfile
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache

import matplotlib
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import patheffects
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from .aztec import Panel

FONT_FAMILY = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
# applied only while drawing/saving, so the user's own matplotlib settings are untouched
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": FONT_FAMILY,
    "pdf.fonttype": 42,  # real (editable) text in PDF
    "svg.fonttype": "none",  # real text in SVG
    "svg.image_inline": True,
}
# matplotlib is not thread-safe and Streamlit may build a download while the preview redraws
_LOCK = threading.RLock()


@dataclass
class Style:
    ncols: int = 3
    electron_span: int = 1  # 2 -> electron image fills a 2x2 block
    # output size.  Pictures are only ever scaled by the same factor in x and y, so asking for a
    # width AND a height never stretches anything: the layout is fitted inside and centred.
    size_mode: str = "width"  # width | height | box
    width_mm: float = 170.0
    height_mm: float = 120.0
    auto_cols: bool = False  # box mode: pick the number of columns that makes the pictures largest
    # pictures of different shapes:  width = same width (cells of one size; a shorter picture sits
    # at `valign` in its cell) | height = same height (rows) | crop = trimmed to one shape (crop_aspect)
    # | free = nested rows and columns written in `spec` (see tree.py)
    align: str = "width"
    spec: str = ""  # free arrangement, e.g. "1 2 (3/4/5)\n6 7 8"; the numbers are the items' `ref`
    valign: str = "center"  # top | center | bottom
    # pictures per row from the top, e.g. (2, 3) for 5 pictures; rows beyond the list hold ncols each
    row_counts: tuple[int, ...] | None = None
    row_align: str = "auto"  # a row with fewer pictures: left | center | right (auto: left, same height: center)
    crop_aspect: float | None = None  # crop: width/height to trim to (None = the shape that trims least)
    gap_mm: float = 1.0
    margin_mm: float = 0.0
    background: str = "white"  # white | black | transparent
    # element labels (top-left of each map)
    label_on: bool = True
    label_pt: float = 10.0
    label_color: str = "white"  # white | black | map | #rrggbb
    label_outline: bool = False  # outline around the label text (off by default)
    label_corner: str = "top-left"  # top-left | top-right | bottom-left | bottom-right
    inset_em: float = 0.5  # distance of labels and scale bars from the picture edge, in units of the larger font size
    # scale bar
    scalebar_on: str = "electron"  # electron | all | none
    scalebar_um: float | None = None  # None -> automatic round length
    scalebar_corner: str = "bottom-right"  # bottom-right | bottom-left
    scalebar_color: str = "white"  # white | black | auto (black on a light background, else white) | #rrggbb
    scalebar_text: bool = True
    scalebar_pt: float = 9.0
    scalebar_thick_pt: float = 2.0  # thickness of the bar in points (the same on every panel)
    scalebar_outline: bool = False
    # pixels
    oversample: int = 2  # raster: output px per original px for the most detailed panel
    dpi: float | None = None  # if set, overrides oversample (warns when panels are shrunk)
    interpolation: str = "nearest"  # nearest (crisp pixels) | lanczos (smooth), raster only


@dataclass
class Adjust:
    """Per-picture brightness / contrast, applied uniformly to the whole picture.

    Grey pictures (SEI / SEM) get the curve directly.  Colour maps get it on their brightness
    (the max of R, G, B) with R, G, B scaled together, so hues do not change.
      auto        grey: stretch the 0.5-99.5 % range to full scale (levels); colour: brightest 0.5 % -> full
      contrast    % change around the picture's mean: -100 = flat, 0 = unchanged, +100 = twice the contrast
      brightness  % of full scale added: +20 adds 51 grey levels
      gamma       mid-tones: >1 brighter, <1 darker
      gain        plain multiplication (kept for the Python API)
    """

    gain: float = 1.0
    gamma: float = 1.0
    auto: bool = False
    brightness: float = 0.0
    contrast: float = 0.0

    @property
    def identity(self) -> bool:
        return (not self.auto and abs(self.gain - 1) < 1e-9 and abs(self.gamma - 1) < 1e-9
                and abs(self.brightness) < 1e-9 and abs(self.contrast) < 1e-9)


@dataclass
class Item:
    panel: Panel
    label: str = ""
    adjust: Adjust = field(default_factory=Adjust)
    crop: tuple[int, int, int, int] | None = None  # trimmed part of the picture: x0, y0, x1, y1 (end excluded)
    bar_color: str | None = None  # this picture's scale bar and its text: white | black | auto (None = Style's)
    ref: int | None = None  # the picture's number (the "順" column); the free arrangement refers to it

    @property
    def image(self) -> np.ndarray:
        """The picture as placed: an exact crop of the panel (no resampling)."""
        img = self.panel.image
        if self.crop is not None:
            x0, y0, x1, y1 = self.crop
            img = img[y0:y1, x0:x1]
        return img

    @property
    def size(self) -> tuple[int, int]:
        if self.crop is not None:
            x0, y0, x1, y1 = self.crop
            return x1 - x0, y1 - y0
        return self.panel.size

    @property
    def field_um(self) -> tuple[float, float] | None:
        u = self.panel.um_per_px
        return None if u is None else (self.size[0] * u, self.size[1] * u)


def crop_box(size: tuple[int, int], aspect: float | None = None, fx: float = 0.5, fy: float = 0.5,
             zoom: float = 1.0) -> tuple[int, int, int, int]:
    """Part of a W x H picture kept when it is trimmed to `aspect` (width/height): the largest such
    rectangle (or 1/zoom of it), placed at fx, fy (0 = left / top, 0.5 = centre, 1 = right / bottom).
    Whole pixels only: trimming cuts, it never stretches or resamples."""
    W, H = int(size[0]), int(size[1])
    z = max(1.0, float(zoom))
    if not aspect or aspect <= 0:
        cw = int(min(W, max(1, round(W / z))))
        ch = int(min(H, max(1, round(H / z))))
    else:
        t = float(aspect)
        h_max = (H if W / H > t else W / t) / z  # wider than wanted: trim left/right, else top/bottom
        # whole pixels: the tallest crop (within 2 % of the largest) whose shape matches `aspect`
        # best, so that pictures trimmed to the same shape line up to a fraction of a pixel
        best = None
        top = max(1, int(math.floor(h_max + 1e-9)))
        for ch in range(top, max(0, int(top * 0.98)) , -1):
            cw = int(round(ch * t))
            if cw < 1 or cw > W or ch > H:
                continue
            err = abs(cw / ch - t) / t
            if best is None or err < best[0] - 1e-12:
                best = (err, cw, ch)
            if err <= 1e-4:
                break
        if best is None:
            best = (0.0, max(1, min(W, int(round(top * t)))), min(H, top))
        cw, ch = best[1], best[2]
    x0 = int(round((W - cw) * min(1.0, max(0.0, float(fx)))))
    y0 = int(round((H - ch) * min(1.0, max(0.0, float(fy)))))
    return x0, y0, x0 + cw, y0 + ch


def least_trim_aspect(sizes: list[tuple[int, int]]) -> float | None:
    """The width/height (one of the pictures' own) that keeps the most of all pictures when each is
    trimmed to it."""
    aspects = [w / h for w, h in sizes if w > 0 and h > 0]
    if not aspects:
        return None
    return max(aspects, key=lambda t: sum(min(a, t) / max(a, t) for a in aspects))


# --------------------------------------------------------------------------- helpers
def is_grey(img: np.ndarray, tol: int = 8) -> bool:
    x = img.astype(np.int16)
    return bool(np.abs(x[..., 0] - x[..., 1]).max() <= tol and np.abs(x[..., 1] - x[..., 2]).max() <= tol)


def adjusted(img: np.ndarray, a: Adjust) -> np.ndarray:
    """The picture with its brightness / contrast settings applied (untouched when they are neutral)."""
    if a is None or a.identity:
        return img
    x = img.astype(np.float64)
    grey = is_grey(img)
    v = x.mean(axis=2, keepdims=True) if grey else x.max(axis=2, keepdims=True)
    t = v.copy()
    if a.auto:
        if grey:  # levels: 0.5 % of the pixels may saturate at each end
            lo, hi = np.percentile(v, (0.5, 99.5))
            t = (t - lo) * 255.0 / max(hi - lo, 1.0)
        else:  # colour map: keep black, brightest 0.5 % -> full
            t = t * 255.0 / max(np.percentile(v, 99.5), 1.0)
    t = t * a.gain
    if abs(a.contrast) > 1e-9:
        m = float(t.mean())
        t = (t - m) * max(0.0, 1.0 + a.contrast / 100.0) + m
    t = np.clip(t + 2.55 * a.brightness, 0, 255)
    if abs(a.gamma - 1) > 1e-9:
        t = 255.0 * (t / 255.0) ** (1.0 / max(a.gamma, 1e-3))
    if grey:
        out = np.repeat(t, 3, axis=2)
    else:
        out = x * np.divide(t, v, out=np.zeros_like(v), where=v > 0)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def nice_length(field_um: float, frac: float = 0.3) -> float:
    """Largest 1-2-5 value not longer than `frac` of the field width."""
    target = field_um * frac
    k = math.floor(math.log10(target))
    best = 10.0**k
    for m in (1, 2, 5, 10):
        if m * 10.0**k <= target:
            best = m * 10.0**k
    return best


def bar_length(it: "Item", st: "Style") -> tuple[float, bool] | None:
    """The scale bar a picture gets: (length in µm, whether a fixed length was too long for this
    picture, so that its own automatic length is used instead), or None (no bar wanted here, or the
    picture's scale is unknown).  A fixed length must stay under 90 % of the picture's width."""
    panel = it.panel
    want = st.scalebar_on == "all" or (st.scalebar_on == "electron" and panel.kind == "electron")
    if not want or not panel.um_per_px:
        return None
    iw = it.size[0]
    if st.scalebar_um and st.scalebar_um / panel.um_per_px < 0.9 * iw:
        return float(st.scalebar_um), False
    return nice_length(iw * panel.um_per_px), bool(st.scalebar_um)


def format_length(um: float) -> str:
    if um >= 1000:
        return f"{um / 1000:g} mm"
    if um < 1:
        return f"{um * 1000:g} nm"
    return f"{um:g} µm"


def _color(c: str, panel: Panel) -> str | tuple:
    if c == "map":
        if panel.color is not None:
            return tuple(v / 255 for v in panel.color)
        return "white"
    return c


def label_words(label: str, panel: Panel, default) -> list[tuple[str, object]]:
    """The label as (text, colour) pieces.  A layered image's element symbols get the colours of
    their layers (from its colour key); other words, and every other picture, the label colour."""
    layers = {t: tuple(v / 255 for v in c) for t, c in (getattr(panel, "layers", None) or [])}
    words = label.split()
    if not layers or not any(w in layers for w in words):
        return [(label, default)]
    return [(w, layers.get(w, default)) for w in words]


def _luminance(c) -> float:
    r, g, b = mcolors.to_rgb(c)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _edge(c) -> str:
    """Outline colour that stands out from the text / bar colour: white around black, else black."""
    return "white" if _luminance(c) < 0.25 else "black"


@lru_cache(maxsize=256)
def text_width_pt(text: str, pt: float) -> float:
    """Width of bold label text in points (the scale bar's number), in the figure's font."""
    try:
        with _LOCK, matplotlib.rc_context(RC):
            path = TextPath((0, 0), text, size=pt, prop=FontProperties(family="sans-serif", weight="bold"))
            return float(path.get_extents().width)
    except Exception:
        return 0.6 * pt * len(text)


def background_color(img: np.ndarray, region: tuple[float, float, float, float]) -> str:
    """'black' when the picture is light where the scale bar and its text go, else 'white'."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = region
    xa, xb = max(0, int(math.floor(x0))), min(w, int(math.ceil(x1)))
    ya, yb = max(0, int(math.floor(y0))), min(h, int(math.ceil(y1)))
    if xb <= xa or yb <= ya:
        return "white"
    a = img[ya:yb, xa:xb, :3].astype(np.float64)
    lum = float((0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).mean())
    return "black" if lum > 128 else "white"


# --------------------------------------------------------------------------- layout
@dataclass
class Placed:
    item: Item
    row: int
    col: int
    span: int


def row_lengths(n: int, ncols: int, counts=None) -> list[int]:
    """Number of pictures in each row: the given counts from the top, then ncols per row."""
    ncols = max(1, int(ncols))
    out, left = [], n
    for c in list(counts or []):
        if left <= 0:
            break
        c = max(1, int(c))
        out.append(min(c, left))
        left -= out[-1]
    while left > 0:
        out.append(min(ncols, left))
        left -= out[-1]
    return out


def place(items: list[Item], ncols: int, electron_span: int, counts=None) -> tuple[list[Placed], int]:
    if counts:  # rows of given lengths (no 2 x 2 blocks)
        out, k = [], 0
        for r, n in enumerate(row_lengths(len(items), ncols, counts)):
            for c in range(n):
                out.append(Placed(items[k], r, c, 1))
                k += 1
        return out, max((p.row + 1 for p in out), default=1)
    ncols = max(1, int(ncols))
    occ: set[tuple[int, int]] = set()
    out: list[Placed] = []
    r = c = 0
    for it in items:
        span = electron_span if (it.panel.kind == "electron" and electron_span > 1 and ncols >= electron_span) else 1
        while True:
            if c + span <= ncols and all((r + i, c + j) not in occ for i in range(span) for j in range(span)):
                break
            c += 1
            if c >= ncols:
                r, c = r + 1, 0
        for i in range(span):
            for j in range(span):
                occ.add((r + i, c + j))
        out.append(Placed(it, r, c, span))
        c += span
        if c >= ncols:
            r, c = r + 1, 0
    nrows = max((p.row + p.span for p in out), default=1)
    return out, nrows


@dataclass
class Geometry:
    cell_w: int  # px
    cell_h: int
    gap: int
    margin: int
    ox: int  # offset of the grid inside the figure (rounding remainder, or the free space in box mode)
    oy: int
    width: int  # figure px
    height: int
    dpi: float
    shrink: float  # smallest (output px / source px) over all pictures; <1 means a picture is shrunk
    ncols: int
    nrows: int
    boxes: list  # per placed picture: (x, y, w, h, scale) in output px; w/h == source width/height exactly
    refs: list = field(default_factory=list)  # per placed picture: its number (Item.ref, or its place in the list)
    capped: bool = False  # free arrangement: the output was limited in size (some pictures get fewer px than asked)

    @property
    def width_mm(self) -> float:
        return self.width / self.dpi * 25.4

    @property
    def height_mm(self) -> float:
        return self.height / self.dpi * 25.4


REGULAR = ("electron", "map", "layered")


def cell_aspect(placed: list[Placed]) -> float:
    """Cell height/width: the tallest of the ordinary pictures, so that they all fit by width
    (columns line up) and differ by at most a pixel or two in height.  Odd ones (spectra) are
    fitted inside the cell instead."""
    pool = [p.item for p in placed if p.item.panel.kind in REGULAR] or [p.item for p in placed]
    return max(it.size[1] / it.size[0] for it in pool)


def resolved(items: list[Item], st: Style) -> list[Item]:
    """The items as they are drawn.  Crop mode: pictures without a crop of their own are trimmed at
    their centre to the common shape (spectra and other odd pictures are left whole)."""
    if st.align != "crop":
        return items
    t = st.crop_aspect or least_trim_aspect([it.panel.size for it in items if it.panel.kind in REGULAR])
    return [replace(it, crop=crop_box(it.panel.size, t)) if (it.crop is None and it.panel.kind in REGULAR and t)
            else it for it in items]


def fit_scale(iw: int, ih: int, bw: float, bh: float) -> float:
    """The one factor used for BOTH directions: the largest that fits the picture in its block."""
    return min(bw / iw, bh / ih)


def _cell_mm(st: Style, ncols: int, nrows: int, A: float) -> float:
    from_w = (st.width_mm - 2 * st.margin_mm - (ncols - 1) * st.gap_mm) / ncols
    from_h = (st.height_mm - 2 * st.margin_mm - (nrows - 1) * st.gap_mm) / (nrows * A)
    if st.size_mode == "height":
        return from_h
    if st.size_mode == "box":
        return min(from_w, from_h)
    return from_w


def best_ncols(items: list[Item], st: Style) -> int:
    """Number of columns that gives the largest pictures inside the requested width x height."""
    if st.row_counts:  # the rows are given
        return max(1, int(st.ncols))
    if st.align == "height":
        best, best_h = max(1, int(st.ncols)), -1.0
        for n in range(1, len(items) + 1):
            h = _row_height_mm(st, _rows(items, n))
            if h > best_h + 1e-9:
                best, best_h = n, h
        return best
    best, best_cw = max(1, int(st.ncols)), -1.0
    for n in range(1, len(items) + 1):
        placed, nrows = place(items, n, st.electron_span)
        used = max(p.col + p.span for p in placed)
        cw = _cell_mm(st, used, nrows, cell_aspect(placed))
        if cw > best_cw + 1e-9:
            best, best_cw = n, cw
    return best


def _rows(items: list[Item], ncols: int, counts=None) -> list[list[Item]]:
    out, k = [], 0
    for n in row_lengths(len(items), ncols, counts):
        out.append(items[k: k + n])
        k += n
    return out


def _row_shift(st: Style, default: str) -> float:
    """Where a shorter row sits: 0 = left, 0.5 = centre, 1 = right."""
    a = st.row_align if st.row_align in ("left", "center", "right") else default
    return {"left": 0.0, "center": 0.5, "right": 1.0}[a]


def _row_height_mm(st: Style, rows: list[list[Item]]) -> float:
    """Same-height mode: the one picture height (mm) that fits the rows into the requested size."""
    Wc = st.width_mm - 2 * st.margin_mm
    Hc = st.height_mm - 2 * st.margin_mm
    from_w = min((Wc - (len(r) - 1) * st.gap_mm) / sum(it.size[0] / it.size[1] for it in r) for r in rows)
    from_h = (Hc - (len(rows) - 1) * st.gap_mm) / len(rows)
    if st.size_mode == "height":
        return from_h
    if st.size_mode == "box":
        return min(from_w, from_h)
    return from_w


def _layout_rows(items: list[Item], st: Style) -> tuple[list[Placed], Geometry]:
    """Same height: every picture is drawn the same height; a row's pictures sit side by side
    (their widths follow their own shapes) and each row is centred."""
    ncols = best_ncols(items, st) if (st.size_mode == "box" and st.auto_cols) else max(1, int(st.ncols))
    rows = _rows(items, ncols, st.row_counts)
    h_mm = _row_height_mm(st, rows)
    if not h_mm > 0:
        raise ValueError("出力サイズに対して、間隔・余白が大きすぎます")
    if st.dpi:
        ppm = st.dpi / 25.4
        hpx = float(max(8, math.floor(h_mm * ppm + 1e-6)))
    else:  # the picture with the most rows gets N output px per source px, the others more
        hpx = float(max(1, int(st.oversample)) * max(it.size[1] for it in items))
        ppm = hpx / h_mm
    gap = int(round(st.gap_mm * ppm))
    margin = int(round(st.margin_mm * ppm))
    row_w = [sum(hpx * it.size[0] / it.size[1] for it in r) + (len(r) - 1) * gap for r in rows]
    content_w = max(row_w)
    grid_w = 2 * margin + content_w
    grid_h = 2 * margin + len(rows) * hpx + (len(rows) - 1) * gap
    want_w = st.width_mm * ppm if st.size_mode in ("width", "box") else 0
    want_h = st.height_mm * ppm if st.size_mode in ("height", "box") else 0
    W = max(int(math.ceil(grid_w - 1e-6)), int(round(want_w)))
    H = max(int(math.ceil(grid_h - 1e-6)), int(round(want_h)))
    ox, oy = math.floor((W - grid_w) / 2 + 1e-9), math.floor((H - grid_h) / 2 + 1e-9)
    placed, boxes, shrink = [], [], math.inf
    fx = _row_shift(st, "center")
    for r, (row, rw) in enumerate(zip(rows, row_w)):
        x = ox + margin + (content_w - rw) * fx
        y = oy + margin + r * (hpx + gap)
        for c, it in enumerate(row):
            iw, ih = it.size
            k = hpx / ih  # same factor k for x and y: the aspect ratio cannot change
            boxes.append((math.floor(x + 1e-9), y, iw * k, ih * k, k))
            placed.append(Placed(it, r, c, 1))
            shrink = min(shrink, k)
            x += iw * k + gap
    widths = sorted(b[2] for b in boxes)
    g = Geometry(int(round(widths[len(widths) // 2])), int(round(hpx)), gap, margin, ox, oy, W, H, ppm * 25.4,
                 shrink, max(len(r) for r in rows), len(rows), boxes, numbers(items))
    return placed, g


# a free arrangement may ask for a huge raster (a 4k picture in a small box): keep it renderable
MAX_SIDE_PX = 30000
MAX_AREA_PX = 300e6


def numbers(items: list[Item]) -> list[int]:
    """Each picture's number: its `ref` (the "順" column), or its place in the list."""
    return [it.ref if it.ref is not None else k + 1 for k, it in enumerate(items)]


def free_parts(items: list[Item], spec: str):
    """(arrangement, {number: item}) for the free arrangement.  Numbers written for pictures that
    are not among the items are left out; items whose number is not written are not placed."""
    from . import tree as T

    node = T.parse(spec)
    by_ref = dict(zip(numbers(items), items))
    node = T.prune(node, set(by_ref))
    if node is None:
        raise ValueError("配置に、図に使う画像の番号がありません")
    return node, by_ref


def _layout_tree(items: list[Item], st: Style) -> tuple[list[Placed], Geometry]:
    """Free arrangement: nested rows (same height) and columns (same width), solved exactly."""
    from . import tree as T

    node, by_ref = free_parts(items, st.spec)
    aspect = {r: it.size[0] / it.size[1] for r, it in by_ref.items()}
    w_req = st.width_mm - 2 * st.margin_mm
    h_req = st.height_mm - 2 * st.margin_mm
    use_w, use_h = st.size_mode in ("width", "box"), st.size_mode in ("height", "box")
    if (use_w and w_req <= 0) or (use_h and h_req <= 0):
        raise ValueError("出力サイズに対して、間隔・余白が大きすぎます")
    wc, hc, rects = T.solve(node, aspect, st.gap_mm, width=w_req if use_w else None, height=h_req if use_h else None)
    fig_w = st.width_mm if use_w else wc + 2 * st.margin_mm
    fig_h = st.height_mm if use_h else hc + 2 * st.margin_mm
    if st.dpi:
        ppm = st.dpi / 25.4
    else:  # every picture gets at least N output px per source px (N = oversample)
        N = max(1, int(st.oversample))
        ppm = max(N * by_ref[r].size[0] / w for r, (x, y, w, h) in rects.items())
    limit = min(MAX_SIDE_PX / (max(fig_w, fig_h) * ppm), math.sqrt(MAX_AREA_PX / (fig_w * fig_h)) / ppm)
    capped = limit < 1
    if capped:
        ppm *= limit
    ox, oy = (fig_w - wc) / 2, (fig_h - hc) / 2  # mm; the margin, or the free space of a box
    placed, boxes = [], []
    for n, (r, (x, y, w, h)) in enumerate(rects.items()):
        it = by_ref[r]
        iw, ih = it.size
        k = w * ppm / iw  # same factor k for x and y: the aspect ratio cannot change
        boxes.append((math.floor((ox + x) * ppm + 1e-9), math.floor((oy + y) * ppm + 1e-9), iw * k, ih * k, k))
        placed.append(Placed(it, 0, n, 1))
    W = max(int(round(fig_w * ppm)), max(int(math.ceil(b[0] + b[2] - 1e-6)) for b in boxes))
    H = max(int(round(fig_h * ppm)), max(int(math.ceil(b[1] + b[3] - 1e-6)) for b in boxes))
    ws, hs = sorted(b[2] for b in boxes), sorted(b[3] for b in boxes)
    top = node.kids if node.kind == "col" else [node]
    ncols = max(len(t.kids) if t.kind == "row" else 1 for t in top)
    g = Geometry(int(round(ws[len(ws) // 2])), int(round(hs[len(hs) // 2])), int(round(st.gap_mm * ppm)),
                 int(round(st.margin_mm * ppm)), int(round(ox * ppm)), int(round(oy * ppm)), W, H, ppm * 25.4,
                 min(b[4] for b in boxes), ncols, len(top), boxes, list(rects), capped)
    return placed, g


def layout(items: list[Item], st: Style) -> tuple[list[Placed], Geometry]:
    items = [it for it in items if it.panel is not None]
    if not items:
        raise ValueError("配置する画像がありません")
    items = resolved(items, st)
    if st.align == "free":
        return _layout_tree(items, st)
    if st.align == "height":
        return _layout_rows(items, st)
    ncols = best_ncols(items, st) if (st.size_mode == "box" and st.auto_cols) else max(1, int(st.ncols))
    placed, nrows = place(items, ncols, st.electron_span, st.row_counts)
    ncols = max(p.col + p.span for p in placed)  # fewer pictures than columns: no empty columns
    A = cell_aspect(placed)
    cw_mm = _cell_mm(st, ncols, nrows, A)
    if not cw_mm > 0:
        raise ValueError("出力サイズに対して、間隔・余白が大きすぎます")
    # resolution: px per mm
    if st.dpi:
        ppm = st.dpi / 25.4
        cell_w = max(8, int(math.floor(cw_mm * ppm + 1e-6)))
    else:
        # every picture gets at least N output px per source px (N = oversample)
        N = max(1, int(st.oversample))
        need = max(N * max(it.size[0], it.size[1] / A) / p.span for p in placed for it in [p.item])
        cell_w = int(math.ceil(need - 1e-9))
        ppm = cell_w / cw_mm
    cell_h = int(math.ceil(cell_w * A - 1e-6))
    gap = int(round(st.gap_mm * ppm))
    margin = int(round(st.margin_mm * ppm))
    grid_w = 2 * margin + ncols * cell_w + (ncols - 1) * gap
    grid_h = 2 * margin + nrows * cell_h + (nrows - 1) * gap
    want_w = st.width_mm * ppm if st.size_mode in ("width", "box") else 0
    want_h = st.height_mm * ppm if st.size_mode in ("height", "box") else 0
    W = max(grid_w, int(round(want_w)))
    H = max(grid_h, int(round(want_h)))
    ox, oy = (W - grid_w) // 2, (H - grid_h) // 2
    boxes, shrink = [], math.inf
    fy = {"top": 0.0, "bottom": 1.0}.get(st.valign, 0.5)
    # a row with fewer pictures than columns sits left / centred / right (not with 2 x 2 blocks)
    shift = {}
    if all(p.span == 1 for p in placed):
        fx = _row_shift(st, "left")
        for r in {p.row for p in placed}:
            used = max(p.col + 1 for p in placed if p.row == r)
            shift[r] = (ncols - used) * fx
    for p in placed:
        iw, ih = p.item.size
        bx = ox + margin + math.floor((p.col + shift.get(p.row, 0.0)) * (cell_w + gap) + 1e-9)
        by = oy + margin + p.row * (cell_h + gap)
        bw = p.span * cell_w + (p.span - 1) * gap
        bh = p.span * cell_h + (p.span - 1) * gap
        k = fit_scale(iw, ih, bw, bh)
        dw, dh = iw * k, ih * k  # same factor k for x and y: the aspect ratio cannot change
        x = bx + math.floor((bw - dw) / 2 + 1e-9)
        y = by + math.floor((bh - dh) * fy + 1e-9)
        boxes.append((x, y, dw, dh, k))
        shrink = min(shrink, k)
    g = Geometry(cell_w, cell_h, gap, margin, ox, oy, W, H, ppm * 25.4, shrink, ncols, nrows, boxes,
                 numbers(items))
    return placed, g


def geometry(placed: list[Placed], nrows: int, st: Style) -> Geometry:  # backwards compatible helper
    return layout([p.item for p in placed], st)[1]


def check_aspect(fig: Figure, placed: list[Placed], g: Geometry) -> None:
    """Hard guarantee: every picture is drawn with its source width/height ratio."""
    for ax, p in zip(fig.axes, placed):
        ax.apply_aspect()
        pos = ax.get_position()
        w, h = pos.width * g.width, pos.height * g.height
        iw, ih = p.item.size
        if abs((w / h) / (iw / ih) - 1) > 1e-6:
            raise RuntimeError(f"縦横比が変わる配置になりました（{p.item.panel.src.name}）。このエラーは不具合です。")


# --------------------------------------------------------------------------- drawing
def render(items: list[Item], st: Style, vector: bool = False):
    """Return (matplotlib Figure, Geometry)."""
    placed, g = layout(items, st)
    fig = Figure(figsize=(g.width / g.dpi, g.height / g.dpi), dpi=g.dpi)  # no pyplot: no global state
    FigureCanvasAgg(fig)
    transparent = st.background == "transparent"
    fig.patch.set_facecolor("none" if transparent else st.background)
    fig.patch.set_alpha(0 if transparent else 1)
    interp = "none" if vector else ("nearest" if st.interpolation == "nearest" else "lanczos")
    pt2px = g.dpi / 72.0
    # distances from the edge and the bar thickness are set in points: the same on every panel,
    # however large each picture is drawn
    fonts = ([st.label_pt] if st.label_on else []) + ([st.scalebar_pt] if st.scalebar_on != "none" else [])
    inset_out = st.inset_em * max(fonts or [st.label_pt]) * pt2px
    thick_out = max(st.scalebar_thick_pt * pt2px, 1.6)
    for p, (x, y, w, h, k) in zip(placed, g.boxes):
        panel = p.item.panel
        img = adjusted(p.item.image, p.item.adjust)
        ih, iw = img.shape[:2]
        ax = fig.add_axes([x / g.width, 1 - (y + h) / g.height, w / g.width, h / g.height])
        ax.set_axis_off()
        # aspect="equal": one source pixel is square on output, enforced by matplotlib itself
        ax.imshow(img, interpolation=interp, extent=(0, iw, ih, 0), aspect="equal", resample=False)
        ax.set_xlim(0, iw)
        ax.set_ylim(ih, 0)
        s = 1.0 / k  # source px per output px
        inset = inset_out * s
        # element label (a layered image: each element symbol in its layer's colour, side by side)
        if st.label_on and p.item.label:
            col = _color(st.label_color, panel)
            right = st.label_corner.endswith("right")
            bottom = st.label_corner.startswith("bottom")
            ty = ih - inset if bottom else inset
            words = label_words(p.item.label, panel, col)
            if len(words) == 1:
                word, c = words[0]
                outline = [patheffects.withStroke(linewidth=0.16 * st.label_pt, foreground=_edge(c))]
                ax.text(iw - inset if right else inset, ty, word, color=c, fontsize=st.label_pt, fontweight="bold",
                        ha="right" if right else "left", va="bottom" if bottom else "top",
                        path_effects=outline if st.label_outline else None)
            else:
                gap = 0.3 * st.label_pt * pt2px * s  # a word space, in source px
                widths = [text_width_pt(w, st.label_pt) * pt2px * s for w, _ in words]
                tx = (iw - inset - sum(widths) - gap * (len(words) - 1)) if right else inset
                for (word, c), wpx in zip(words, widths):
                    outline = [patheffects.withStroke(linewidth=0.16 * st.label_pt, foreground=_edge(c))]
                    ax.text(tx, ty, word, color=c, fontsize=st.label_pt, fontweight="bold", ha="left",
                            va="bottom" if bottom else "top", path_effects=outline if st.label_outline else None)
                    tx += wpx + gap
        # scale bar (a fixed length too long for this picture: its own automatic length, see bar_length)
        plan = bar_length(p.item, st)
        if plan is not None:
            L = plan[0]
            Lpx = L / panel.um_per_px
            th = thick_out * s
            left = st.scalebar_corner == "bottom-left"
            by = ih - inset - th
            ty = by - 0.35 * st.scalebar_pt * pt2px * s  # baseline area of the text above the bar
            text = format_length(L)
            tw = text_width_pt(text, st.scalebar_pt) * pt2px * s if st.scalebar_text else 0.0
            # the bar and its number share one centre; the wider of the two sets the distance
            # from the edge (a number wider than a short bar moves the bar inwards with it)
            half = max(Lpx, tw) / 2
            tx = inset + half if left else iw - inset - half
            bx = tx - Lpx / 2
            col = p.item.bar_color or st.scalebar_color or "white"
            if col == "auto":  # judged on the picture where the bar and its text are drawn
                top = ty - (1.25 * st.scalebar_pt * pt2px * s if st.scalebar_text else 0)
                col = background_color(img, (min(bx, tx - tw / 2) - 0.05 * Lpx, top,
                                             max(bx + Lpx, tx + tw / 2) + 0.05 * Lpx, by + th))
            edge = _edge(col)
            rect = Rectangle((bx, by), Lpx, th, facecolor=col, edgecolor=edge if st.scalebar_outline else "none",
                             linewidth=0.6 if st.scalebar_outline else 0)
            ax.add_patch(rect)
            if st.scalebar_text:
                bar_outline = [patheffects.withStroke(linewidth=0.16 * st.scalebar_pt, foreground=edge)]
                ax.text(tx, ty, text, color=col,
                        fontsize=st.scalebar_pt, fontweight="bold", ha="center", va="bottom",
                        path_effects=bar_outline if st.scalebar_outline else None)
    check_aspect(fig, placed, g)
    return fig, g


# --------------------------------------------------------------------------- export
def export(items: list[Item], st: Style, fmt: str) -> bytes:
    fmt = fmt.lower()
    vector = fmt in ("pdf", "svg")
    buf = io.BytesIO()
    transparent = st.background == "transparent"
    with _LOCK, matplotlib.rc_context(RC):
        fig, g = render(items, st, vector=vector)
        if fmt in ("tif", "tiff"):
            fig.canvas.draw()
            im = Image.fromarray(np.array(fig.canvas.buffer_rgba()))
            if not transparent:
                im = im.convert("RGB")
            im.save(buf, format="TIFF", compression="tiff_lzw", dpi=(g.dpi, g.dpi))
        else:
            meta = {"png": {"Software": "edsfig"}, "pdf": {"Creator": "edsfig"}, "svg": {"Creator": "edsfig"}}.get(fmt)
            fig.savefig(buf, format=fmt, dpi=g.dpi, transparent=transparent,
                        facecolor=fig.get_facecolor(), metadata=meta)
    return buf.getvalue()


def preview_png(items: list[Item], st: Style, max_px: int = 1600) -> tuple[bytes, Geometry]:
    """Fast preview: same layout, rendered at a modest resolution."""
    buf = io.BytesIO()
    with _LOCK, matplotlib.rc_context(RC):
        fig, g = render(items, st, vector=False)
        scale = min(1.0, max_px / max(g.width, g.height))
        fig.savefig(buf, format="png", dpi=g.dpi * scale, facecolor=fig.get_facecolor(),
                    transparent=st.background == "transparent")
    return buf.getvalue(), g


def info(items: list[Item], st: Style, report_name: str = "") -> dict:
    placed, g = layout(items, st)
    rows = []
    ppm = g.dpi / 25.4
    for pl, (x, y, w, h, k), ref in zip(placed, g.boxes, g.refs or [None] * len(placed)):
        it = pl.item
        p = it.panel
        sb = p.scalebar
        x0, y0, x1, y1 = it.crop or (0, 0, *p.size)
        turned = p.turned
        rows.append({
            "number": ref,
            "source": p.src.name,
            "kind": p.kind,
            "label": it.label,
            "pixels": list(it.size),
            # the placed picture inside the source file: x0, y0, x1, y1 (end excluded); for a turned
            # picture: the image body in the file, and the trimmed part of the turned picture
            "region_in_source": ([p.box[0], p.box[1], p.box[2] + 1, p.box[3] + 1] if turned else
                                 [p.box[0] + x0, p.box[1] + y0, p.box[0] + x1, p.box[1] + y1]),
            "turned": ({"rotation_cw_deg": p.rotation, "flip_left_right": p.flip_h, "flip_up_down": p.flip_v,
                        "corners": p.rotate_fill, "interpolation": p.rotate_resample,
                        "region_in_turned": [x0, y0, x1, y1]} if turned else None),
            "trimmed": it.crop is not None and tuple(it.size) != tuple(p.size),
            "drawn_px": [round(w, 3), round(h, 3)],
            "box_mm": [round(x / ppm, 3), round(y / ppm, 3), round(w / ppm, 3), round(h / ppm, 3)],
            "scale_out_px_per_src_px": round(k, 6),
            "aspect_source": round(it.size[0] / it.size[1], 9),
            "aspect_output": round(w / h, 9),
            "um_per_px": p.um_per_px,
            "field_um": [round(v, 4) for v in it.field_um] if it.field_um else None,
            "aztec_scale_label": sb.text if sb else None,
            "aztec_scale_bar_px": round(sb.length_px, 2) if sb else None,
            "adjust": asdict(it.adjust),
            "scalebar_color": it.bar_color or st.scalebar_color,
            "scalebar_um": (bar_length(it, st) or (None,))[0],  # the bar as drawn (None: no bar)
            "layers": ([[t, "#%02x%02x%02x" % tuple(c)] for t, c in p.layers] if getattr(p, "layers", None) else None),
            "key_rows_removed": p.band_px if getattr(p, "legend", None) is not None else None,
        })
    return {"report": report_name, "style": asdict(st),
            "output_px": [g.width, g.height], "output_mm": [round(g.width_mm, 2), round(g.height_mm, 2)],
            "columns": g.ncols, "dpi": round(g.dpi, 1),
            "min_output_px_per_source_px": round(g.shrink, 3), "panels": rows}


def panels_zip(items: list[Item], st: Style, report_name: str = "") -> bytes:
    """Every panel as a lossless PNG (no annotation) + calibration JSON.  Panels whose brightness or
    contrast was changed are also saved unchanged in original/ (the source pixels)."""
    buf = io.BytesIO()
    items = [pl.item for pl in layout(items, st)[0]]  # as drawn: trimmed, in placing order (free: only those placed)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, it in enumerate(items, 1):
            name = f"{i:02d}_{it.label or it.panel.kind}.png".replace("/", "_")
            for folder, arr in (("", adjusted(it.image, it.adjust)),
                                ("original/", None if it.adjust.identity else it.image)):
                if arr is None:
                    continue
                b = io.BytesIO()
                Image.fromarray(arr).save(b, format="PNG")
                z.writestr(folder + name, b.getvalue())
        z.writestr("calibration.json", json.dumps(info(items, st, report_name), ensure_ascii=False, indent=2))
    return buf.getvalue()

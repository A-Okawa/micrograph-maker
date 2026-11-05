"""Layout guarantees: pictures are never stretched, whatever output size is requested.

    python tests/test_layout.py
"""
from __future__ import annotations

import io
import itertools
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from edsfig import figure as F  # noqa: E402
from test_synthetic import _report  # noqa: E402


def _items(tmp: Path, extra_wide: bool = True):
    rep, _ = _report(tmp)
    items = [F.Item(p, p.label) for p in rep.panels]
    if extra_wide:  # an odd, wide picture (like a spectrum) must be fitted, not stretched
        from edsfig import aztec

        wide = aztec.SourceImage("wide.png", np.full((50, 300, 4), 200, np.uint8), "x", 9)
        items.append(F.Item(aztec.Panel(src=wide, box=(0, 0, 299, 49), kind="spectrum")))
    return items


def test_aspect_never_changes(tmp: Path = Path(".")):
    items = _items(tmp)
    combos = [("width",) + c for c in itertools.product(
        ("width", "height", "box"), (1, 2, 3, 4), (1, 2), (False, True), (None, 300.0), (0.0, 2.5))]
    combos += [(al, mode, ncols, 1, auto, dpi, 1.0) for al, mode, ncols, auto, dpi in itertools.product(
        ("height", "crop"), ("width", "height", "box"), (1, 2, 3, 4), (False, True), (None, 300.0))]
    n = 0
    for align, mode, ncols, span, auto, dpi, margin in combos:
        st = F.Style(size_mode=mode, ncols=ncols, electron_span=span, auto_cols=auto, dpi=dpi, align=align,
                     width_mm=120.0, height_mm=200.0 if mode != "width" else 120.0, margin_mm=margin)
        placed, g = F.layout(items, st)
        for p, (x, y, w, h, k) in zip(placed, g.boxes):
            iw, ih = p.item.size
            assert abs((w / h) / (iw / ih) - 1) < 1e-12, (align, mode, ncols, p.item.panel.src.name, w, h)
            assert abs(w - iw * k) < 1e-9 and abs(h - ih * k) < 1e-9
            assert x >= 0 and y >= 0 and x + w <= g.width + 1e-6 and y + h <= g.height + 1e-6
        if align == "height":  # every picture drawn exactly the same height
            assert max(b[3] for b in g.boxes) - min(b[3] for b in g.boxes) < 1e-9
        if align == "crop":  # the ordinary pictures share one shape and one drawn size
            reg = [b for p, b in zip(placed, g.boxes) if p.item.panel.kind in F.REGULAR]
            assert max(b[2] for b in reg) - min(b[2] for b in reg) < 1.0 and max(b[3] for b in reg) - min(b[3] for b in reg) < 1.0
        fig, _ = F.render(items, st)  # raises if matplotlib drew any picture with another ratio
        n += 1
    print(f"aspect kept in {n} layouts")


def test_same_height_fills_the_width(tmp: Path = Path(".")):
    items = _items(tmp, extra_wide=False)
    for ncols in (2, 3, 7):
        st = F.Style(align="height", ncols=ncols, width_mm=170.0, gap_mm=1.0)
        placed, g = F.layout(items, st)
        rows = {}
        for p, b in zip(placed, g.boxes):
            rows.setdefault(p.row, []).append(b)
        widest = max(sum(b[2] for b in r) + (len(r) - 1) * g.gap for r in rows.values())
        assert abs(widest - g.width) <= 2, (ncols, widest, g.width)  # the widest row spans the figure
        assert abs(g.width_mm - 170.0) <= 3 * 25.4 / g.dpi


def test_valign_in_cells(tmp: Path = Path(".")):
    items = _items(tmp, extra_wide=False)  # electron image and maps differ slightly in shape
    tops = {}
    for va in ("top", "center", "bottom"):
        placed, g = F.layout(items, F.Style(ncols=3, valign=va))
        for p, (x, y, w, h, k) in zip(placed, g.boxes):
            by = g.oy + g.margin + p.row * (g.cell_h + g.gap)
            if va == "top":
                assert y == by
            if va == "bottom":
                assert abs((y + h) - (by + g.cell_h)) < 1.0
        tops[va] = [b[1] for b in g.boxes]
    assert tops["top"] != tops["bottom"] or all(b[3] == g.cell_h for b in g.boxes)


def test_crop_box_and_trimmed_pixels(tmp: Path = Path(".")):
    assert F.crop_box((1280, 960), 4 / 3) == (0, 0, 1280, 960)  # already that shape: nothing cut
    assert F.crop_box((1280, 960), 1.0) == (160, 0, 1120, 960)  # centred square
    assert F.crop_box((1280, 960), 1.0, fx=0) == (0, 0, 960, 960)
    assert F.crop_box((1280, 960), 1.0, fx=1) == (320, 0, 1280, 960)
    assert F.crop_box((1280, 960), None, zoom=2.0, fx=0, fy=1) == (0, 480, 640, 960)
    for size in ((1280, 853), (523, 512), (1024, 884), (640, 480), (37, 101)):
        for t in (1.0, 4 / 3, 16 / 9, 0.75):
            x0, y0, x1, y1 = F.crop_box(size, t)
            w, h = x1 - x0, y1 - y0
            assert 0 <= x0 and 0 <= y0 and x1 <= size[0] and y1 <= size[1]
            assert abs(w / h - t) / t < max(2e-4, 1.0 / h), (size, t, w, h)  # one shape, to a fraction of a pixel
            assert w >= 0.97 * min(size[0], size[1] * t) - 1 and h >= 0.97 * min(size[1], size[0] / t) - 1  # ~largest
    assert abs(F.least_trim_aspect([(640, 480), (1280, 893), (1024, 884)]) - 4 / 3) < 1e-12
    # the trimmed picture is drawn pixel for pixel (2 x 2 per source pixel)
    items = _items(tmp, extra_wide=False)
    st = F.Style(ncols=3, oversample=2, label_on=False, scalebar_on="none", align="crop", crop_aspect=1.0)
    placed, g = F.layout(items, st)
    out = np.asarray(Image.open(io.BytesIO(F.export(items, st, "png"))).convert("RGB"))
    x, y, w, h, k = g.boxes[0]
    it = placed[0].item
    assert it.crop is not None and it.size[0] == it.size[1]
    assert np.array_equal(it.image, it.panel.image[it.crop[1]:it.crop[3], it.crop[0]:it.crop[2]])
    if abs(k - 2.0) < 1e-12:
        region = out[int(y):int(y + h), int(x):int(x + w)]
        assert np.array_equal(region, np.kron(it.image, np.ones((2, 2, 1), np.uint8)))
    info = F.info(items, st)
    assert info["panels"][0]["trimmed"] and info["panels"][0]["pixels"] == list(it.size)


def test_requested_size(tmp: Path = Path(".")):
    items = _items(tmp, extra_wide=False)
    for mode, dpi, align in itertools.product(("width", "height", "box"), (None, 600.0), ("width", "height", "crop")):
        if True:
            st = F.Style(size_mode=mode, width_mm=170.0, height_mm=60.0, ncols=3, dpi=dpi, align=align)
            _, g = F.layout(items, st)
            px_mm = 25.4 / g.dpi
            if mode in ("width", "box"):
                assert abs(g.width_mm - 170.0) <= 3 * px_mm, (mode, align, g.width_mm)
            if mode in ("height", "box"):
                assert abs(g.height_mm - 60.0) <= 3 * px_mm, (mode, align, g.height_mm)
            if mode == "box":  # everything inside the box
                assert g.width_mm <= 170.0 + px_mm and g.height_mm <= 60.0 + px_mm


def test_raster_pixels_exact_and_square(tmp: Path = Path(".")):
    items = _items(tmp, extra_wide=False)
    st = F.Style(ncols=3, oversample=2, label_on=False, scalebar_on="none", size_mode="box",
                 width_mm=170.0, height_mm=150.0)
    png = F.export(items, st, "png")
    out = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    placed, g = F.layout(items, st)
    x, y, w, h, k = g.boxes[0]
    assert abs(k - 2.0) < 1e-12 and float(w).is_integer() and float(h).is_integer()
    sem = items[0].panel.image
    region = out[int(y):int(y + h), int(x):int(x + w)]
    assert np.array_equal(region, np.kron(sem, np.ones((2, 2, 1), np.uint8))), "electron image not an exact 2x copy"
    # every picture occupies a w x h block with the source ratio (checked on the actual pixels)
    for p, (x, y, w, h, k) in zip(placed, g.boxes):
        ink_rows = round(h)
        ink_cols = round(w)
        iw, ih = p.item.panel.size
        assert abs(ink_cols / ink_rows - iw / ih) < 2 / min(ink_rows, ink_cols)


def test_pictures_per_row(tmp: Path = Path(".")):
    from edsfig import aztec

    rng = np.random.default_rng(5)
    panels = []
    for i in range(5):
        src = aztec.SourceImage(f"p{i}.png", np.dstack([rng.integers(0, 255, (300, 400, 3), dtype=np.uint8),
                                                         np.full((300, 400, 1), 255, np.uint8)]), "x", i)
        panels.append(aztec.Panel(src=src, box=(0, 0, 399, 299), kind="electron"))
    items = [F.Item(p) for p in panels]
    assert F.row_lengths(5, 3) == [3, 2] and F.row_lengths(5, 3, (2,)) == [2, 3]
    assert F.row_lengths(5, 2) == [2, 2, 1] and F.row_lengths(5, 3, (2, 3, 4)) == [2, 3]
    for align in ("width", "crop", "height"):
        for ra, want in (("left", 0.0), ("center", 0.5), ("right", 1.0)):
            st = F.Style(ncols=3, row_counts=(2, 3), row_align=ra, align=align)
            placed, g = F.layout(items, st)
            assert [(p.row, p.col) for p in placed] == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)], align
            top, bottom = g.boxes[:2], g.boxes[2:]
            assert abs(top[0][1] - top[1][1]) < 1e-9 and top[0][1] < bottom[0][1]  # two on the top row
            # the shorter top row: left / centred / right against the full bottom row
            left_gap = top[0][0] - bottom[0][0]
            right_gap = (bottom[2][0] + bottom[2][2]) - (top[1][0] + top[1][2])
            free = left_gap + right_gap
            assert free > 10 and abs(left_gap - want * free) <= 2, (align, ra, left_gap, right_gap)
            F.render(items, st)  # aspect check
    # automatic: grid rows to the left, same-height rows centred (as before)
    _, g = F.layout(items, F.Style(ncols=3, row_counts=(2, 3)))
    assert abs(g.boxes[0][0] - g.boxes[2][0]) <= 1
    _, g = F.layout(items, F.Style(ncols=3, row_counts=(2, 3), align="height"))
    assert g.boxes[0][0] > g.boxes[2][0] + 10


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        test_aspect_never_changes(Path(d))
        test_requested_size(Path(d))
        test_raster_pixels_exact_and_square(Path(d))
        test_same_height_fills_the_width(Path(d))
        test_valign_in_cells(Path(d))
        test_crop_box_and_trimmed_pixels(Path(d))
        test_pictures_per_row(Path(d))
    print("ok")

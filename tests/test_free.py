"""Free arrangement (nested rows and columns written with numbers) and per-picture trimming.

    python tests/test_free.py
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edsfig import aztec  # noqa: E402
from edsfig import figure as F  # noqa: E402
from edsfig import tree as T  # noqa: E402

# the layout of a real figure: two SEM images and three maps stacked; three SEM images; two STEM
# images each with a 2 x 2 block of maps, next to a large TEM image
USER = "1 2 (3/4/5)\n6 7 8\n(9 (10 11/12 13) / 14 (15 16/17 18)) 19"
SIZES = {1: (1280, 960), 2: (1024, 704), 3: (256, 192), 4: (256, 192), 5: (256, 192), 6: (1280, 860),
         7: (500, 560), 8: (480, 600), 9: (512, 512), 10: (128, 128), 11: (128, 128), 12: (128, 128),
         13: (128, 128), 14: (512, 512), 15: (128, 128), 16: (128, 128), 17: (128, 128), 18: (128, 128),
         19: (2048, 1448)}
MAPS = {3, 4, 5, 10, 11, 12, 13, 15, 16, 17, 18}


def _panel(n: int, w: int, h: int, kind: str = "electron") -> aztec.Panel:
    rng = np.random.default_rng(n)
    rgba = np.full((h, w, 4), 255, np.uint8)
    if kind == "map":
        rgba[..., :3] = 0
        rgba[..., n % 3] = rng.integers(0, 255, (h, w))
    else:
        rgba[..., :3] = rng.integers(60, 200, (h, w))[..., None]
    p = aztec.Panel(src=aztec.SourceImage(f"p{n}.png", rgba, f"p{n}", n), box=(0, 0, w - 1, h - 1), kind=kind)
    p.um_per_px = 0.05 * n
    p.label = f"E{n}" if kind == "map" else ""
    return p


def _items(sizes=SIZES) -> list[F.Item]:
    return [F.Item(_panel(n, *sizes[n], "map" if n in MAPS else "electron"), "M" if n in MAPS else "", ref=n)
            for n in sorted(sizes)]


def test_parse():
    assert str(T.parse(USER)) == "((1 2 (3/4/5))/(6 7 8)/(((9 ((10 11)/(12 13)))/(14 ((15 16)/(17 18)))) 19))"
    assert str(T.parse("１　２ （３／４）")) == "(1 2 (3/4))"  # typed with a Japanese input method
    assert str(T.parse("1 2\n\n3 4\n")) == "((1 2)/(3 4))"  # blank lines are ignored
    assert str(T.parse("1 2 /\n3")) == "((1 2)/3)"  # a "/" at the end of a line is one break
    assert str(T.parse("1 2; 3 4")) == str(T.parse("1 2 / 3 4")) == "((1 2)/(3 4))"
    assert str(T.parse("(1 2) 3")) == "(1 2 3)" and str(T.parse("((1/2)/3)")) == "(1/2/3)"
    assert str(T.parse("1,2、3")) == "(1 2 3)"
    for bad, word in [("", "空"), ("1 2 x", "使えない文字"), ("1 (2 3", "数が合いません"), ("1 2)", "多く"),
                      ("1 / / 2", "番号がありません"), ("1 1", "2回以上"), ("()", "番号がありません"),
                      ("0 1", "0 番"), ("1 2 /", "番号がありません")]:
        try:
            T.parse(bad)
        except T.SpecError as e:
            assert word in str(e), (bad, str(e))
        else:
            raise AssertionError(f"accepted {bad!r}")
    assert T.refs_of(T.parse(USER)) == list(range(1, 20))
    assert str(T.prune(T.parse(USER), set(range(1, 20)) - {3, 4, 10, 11, 12, 13})) == \
        "((1 2 5)/(6 7 8)/((9/(14 ((15 16)/(17 18)))) 19))"
    assert T.prune(T.parse("1 2"), {5}) is None
    assert T.default_text([1, 2, 3, 4, 5, 6, 7], [3, 3, 1]) == "1 2 3\n4 5 6\n7"


def test_renumber_text():
    """When pictures are renumbered (one left out goes to the bottom), the arrangement follows them."""
    old = "1 7 (2/3/4)\n８ 9 5\n(10 (11/12)) 13"  # a full-width 8 too
    new = T.renumber_text(old, {7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 6: 13})
    assert new == "1 6 (2/3/4)\n7 8 5\n(9 (10/11)) 12", new
    assert str(T.parse(new)) == "((1 6 (2/3/4))/(7 8 5)/(9 (10/11) 12))"
    assert T.renumber_text("1 2 20 3", {1: 2, 2: 1}) == "2 1 20 3"  # an unknown number stays
    assert T.renumber_text("1 2 3", {2: 3, 3: 2}) == "1 3 2"
    assert T.renumber_text("4 5", {4: 5}) == "5 6"  # 5 is taken now: the unknown 5 moves out of the way


def _bbox(n: T.Node, rects) -> tuple[float, float, float, float]:
    rs = [rects[r] for r in T.refs_of(n)]
    return (min(r[0] for r in rs), min(r[1] for r in rs), max(r[0] + r[2] for r in rs), max(r[1] + r[3] for r in rs))


def _check_groups(n: T.Node, rects, gap: float) -> None:
    if n.kind == "pic":
        return
    boxes = [_bbox(k, rects) for k in n.kids]
    for a, b in zip(boxes, boxes[1:]):
        if n.kind == "row":  # side by side: same top and bottom, one gap apart
            assert abs(a[1] - b[1]) < 1e-9 and abs(a[3] - b[3]) < 1e-9, (str(n), a, b)
            assert abs(b[0] - a[2] - gap) < 1e-9, (str(n), a, b)
        else:  # stacked: same left and right edges, one gap apart
            assert abs(a[0] - b[0]) < 1e-9 and abs(a[2] - b[2]) < 1e-9, (str(n), a, b)
            assert abs(b[1] - a[3] - gap) < 1e-9, (str(n), a, b)
    for k in n.kids:
        _check_groups(k, rects, gap)


def test_solve_exact():
    node = T.parse(USER)
    aspect = {n: w / h for n, (w, h) in SIZES.items()}
    for gap in (0.0, 1.0, 2.5):
        W, H, rects = T.solve(node, aspect, gap, width=166.0)
        assert abs(W - 166.0) < 1e-9
        assert list(rects) == list(range(1, 20))  # reading order
        for n, (x, y, w, h) in rects.items():
            assert abs((w / h) / aspect[n] - 1) < 1e-12, n  # every picture keeps its shape exactly
        _check_groups(node, rects, gap)
        box = _bbox(node, rects)
        assert abs(box[0]) < 1e-9 and abs(box[1]) < 1e-9 and abs(box[2] - W) < 1e-9 and abs(box[3] - H) < 1e-9
    # the height given, or a box (the limiting side wins)
    W, H, _ = T.solve(node, aspect, 1.0, height=150.0)
    assert abs(H - 150.0) < 1e-9 and W > 0
    W1, H1, _ = T.solve(node, aspect, 1.0, width=166.0)
    W, H, _ = T.solve(node, aspect, 1.0, width=166.0, height=H1 * 0.5)
    assert abs(H - H1 * 0.5) < 1e-9 and W < 166.0
    W, H, _ = T.solve(node, aspect, 1.0, width=166.0, height=H1 * 2)
    assert abs(W - 166.0) < 1e-9 and abs(H - H1) < 1e-9
    try:
        T.solve(node, aspect, 40.0, width=100.0)
    except ValueError as e:
        assert "間隔" in str(e)
    else:
        raise AssertionError("an impossible gap was accepted")


def test_layout_and_render():
    items = _items()
    for mode, dpi, margin in [("width", None, 0.0), ("width", 600.0, 2.0), ("height", None, 0.0), ("box", None, 1.0)]:
        st = F.Style(align="free", spec=USER, size_mode=mode, width_mm=170.0, height_mm=200.0, dpi=dpi,
                     margin_mm=margin, scalebar_on="all")
        placed, g = F.layout(items, st)
        assert g.refs == list(range(1, 20))
        for p, (x, y, w, h, k) in zip(placed, g.boxes):
            iw, ih = p.item.size
            assert abs((w / h) / (iw / ih) - 1) < 1e-12
            assert x >= 0 and y >= 0 and x + w <= g.width + 1e-6 and y + h <= g.height + 1e-6
        if dpi is None:
            assert g.shrink >= 2 - 1e-9  # every picture gets at least 2 x 2 output px per source px
            assert abs(min(b[4] for b in g.boxes) - 2.0) < 1e-9  # ...and the most detailed one exactly 2
        if mode in ("width", "box"):
            assert abs(g.width_mm - 170.0) < 0.05, g.width_mm
        if mode in ("height", "box"):
            assert g.height_mm <= 200.0 + 0.05
        F.render(items, st)  # raises if any picture were drawn with another ratio


def test_same_text_and_bars_everywhere():
    """One point size for every label and scale-bar number, one thickness for every bar, one
    distance from the edges, whether the picture is drawn large or small."""
    items = _items()
    st = F.Style(align="free", spec=USER, scalebar_on="all", label_pt=9, scalebar_pt=8, scalebar_thick_pt=2.5)
    fig, g = F.render(items, st)
    pt2px = g.dpi / 72
    sizes, thick, insets = set(), [], []
    for ax, (x, y, w, h, k) in zip(fig.axes, g.boxes):
        for t in ax.texts:
            sizes.add((t.get_text()[:1].isalpha() and t.get_text() == "M", t.get_fontsize()))
            if t.get_text() == "M":  # label at the top left: its distance from the edge in output px
                insets.append(t.get_position()[0] * k)
        for r in ax.patches:
            thick.append(r.get_height() * k)  # output px
    assert sizes == {(True, 9.0), (False, 8.0)}, sizes
    assert len(thick) == 19 and max(thick) - min(thick) < 1e-6 and abs(thick[0] - 2.5 * pt2px) < 1e-6, thick[:3]
    assert len(insets) == 11 and max(insets) - min(insets) < 1e-6 and abs(insets[0] - 0.5 * 9 * pt2px) < 1e-6


def test_numbers_and_missing_pictures():
    items = _items({1: (100, 80), 2: (100, 100), 4: (60, 120)})  # picture 3 not in use
    placed, g = F.layout(items, F.Style(align="free", spec="1 2 (3/4)"))
    assert g.refs == [1, 2, 4] and [p.item.ref for p in placed] == [1, 2, 4]
    placed, g = F.layout(items, F.Style(align="free", spec="2 1"))  # 4 not written: not placed
    assert g.refs == [2, 1]
    try:
        F.layout(items, F.Style(align="free", spec="3 7"))
    except ValueError as e:
        assert "番号がありません" in str(e)
    else:
        raise AssertionError
    # without numbers, the place in the list counts
    plain = [F.Item(_panel(k, 100, 80)) for k in range(3)]
    assert F.layout(plain, F.Style(align="free", spec="3 / 1 2"))[1].refs == [3, 1, 2]


def test_trimmed_pictures_keep_their_pixels():
    items = _items({1: (400, 300), 2: (300, 300), 3: (200, 300)})
    items[0].crop = F.crop_box((400, 300), 1.0, 0.25, 0.5, 1.5)  # a square, zoomed, left of centre
    items[2].crop = (20, 30, 180, 270)  # a free rectangle
    st = F.Style(align="free", spec="1 2 3", scalebar_on="all")
    placed, g = F.layout(items, st)
    assert [p.item.size for p in placed] == [(200, 200), (300, 300), (160, 240)]
    pdf = F.export(items, st, "pdf")
    dims = sorted(zip(map(int, re.findall(rb"/Width (\d+)", pdf)), map(int, re.findall(rb"/Height (\d+)", pdf))))
    assert dims == [(160, 240), (200, 200), (300, 300)], dims  # embedded at their own pixels
    png = Image.open(io.BytesIO(F.export(items, st, "png")))
    assert png.size == (g.width, g.height)
    info = F.info(items, st, "t")
    assert [r["number"] for r in info["panels"]] == [1, 2, 3]
    assert [r["trimmed"] for r in info["panels"]] == [True, False, True]
    assert abs(sum(r["box_mm"][2] for r in info["panels"]) + 2 * st.gap_mm - 170.0) < 0.01


def test_huge_raster_is_capped():
    # a 4k picture drawn 10 mm wide would need ~140 000 px across at 2 px per source px
    items = _items({1: (4096, 4096), 2: (1600, 100)})
    placed, g = F.layout(items, F.Style(align="free", spec="1 2", width_mm=170.0))
    assert g.capped and g.shrink < 2
    assert g.width <= F.MAX_SIDE_PX + 1 and g.width * g.height <= F.MAX_AREA_PX * 1.01
    assert not F.layout(items, F.Style(align="free", spec="1 / 2", width_mm=170.0))[1].capped


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")

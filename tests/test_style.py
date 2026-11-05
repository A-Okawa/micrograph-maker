"""Scale bar colour per picture: white / black / automatic (from the picture under the bar).

    python tests/test_style.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edsfig import aztec  # noqa: E402
from edsfig import figure as F  # noqa: E402


def _panel(value: int, name: str, w: int = 400, h: int = 300) -> aztec.Panel:
    rgba = np.full((h, w, 4), 255, np.uint8)
    rgba[..., :3] = value
    p = aztec.Panel(src=aztec.SourceImage(name, rgba, name, 0), box=(0, 0, w - 1, h - 1), kind="electron")
    p.um_per_px = 0.1  # field 40 µm -> automatic bar 10 µm = 100 px
    return p


def _inks(items: list[F.Item], st: F.Style) -> list[tuple[int, int]]:
    """(dark pixels, light pixels) drawn on each uniformly grey panel."""
    out = np.asarray(Image.open(io.BytesIO(F.export(items, st, "png"))).convert("L")).astype(int)
    placed, g = F.layout(items, st)
    res = []
    for p, (x, y, w, h, k) in zip(placed, g.boxes):
        v = int(p.item.panel.image[0, 0, 0])
        r = out[int(y) + 1: int(y + h) - 1, int(x) + 1: int(x + w) - 1]
        res.append((int((r < v - 60).sum()), int((r > v + 60).sum())))
    return res


def test_colour_per_picture():
    st = F.Style(ncols=2, label_on=False, scalebar_on="all", scalebar_color="white")
    items = [F.Item(_panel(128, "a"), bar_color="black"), F.Item(_panel(128, "b"))]  # b follows the style
    (a_dark, a_light), (b_dark, b_light) = _inks(items, st)
    assert a_dark > 500 and a_light == 0, (a_dark, a_light)  # black bar and black number
    assert b_light > 500 and b_dark == 0, (b_dark, b_light)  # white (the common setting)


def test_automatic_colour():
    st = F.Style(ncols=2, label_on=False, scalebar_on="all", scalebar_color="auto")
    items = [F.Item(_panel(225, "light")), F.Item(_panel(40, "dark"))]
    (l_dark, l_light), (d_dark, d_light) = _inks(items, st)
    assert l_dark > 500 and l_light == 0, (l_dark, l_light)  # light picture -> black bar
    assert d_light > 500 and d_dark == 0, (d_dark, d_light)  # dark picture -> white bar
    # judged where the bar is drawn: a picture dark at the bottom right only gets a white bar
    p = _panel(225, "corner")
    p.src.rgba[200:, 200:, :3] = 30
    assert F.background_color(p.image, (250, 220, 390, 290)) == "white"
    assert F.background_color(p.image, (10, 10, 150, 150)) == "black"


def test_outline_contrasts_with_the_bar():
    st = F.Style(ncols=1, label_on=False, scalebar_on="all", scalebar_outline=True)
    (dark, light), = _inks([F.Item(_panel(128, "a"), bar_color="black")], st)
    assert dark > 500 and light > 50, (dark, light)  # black bar with a white rim


def test_number_stays_inside_the_picture():
    """A short bar with a wide number: the number is moved inward, not drawn past the edge."""
    for corner in ("bottom-right", "bottom-left"):
        st = F.Style(ncols=1, label_on=False, scalebar_on="all", scalebar_um=0.5, scalebar_pt=14,
                     background="black", margin_mm=4.0, scalebar_corner=corner)
        items = [F.Item(_panel(40, "a"))]
        out = np.asarray(Image.open(io.BytesIO(F.export(items, st, "png"))).convert("L")).astype(int)
        _, g = F.layout(items, st)
        x, y, w, h, k = g.boxes[0]
        inside = out[int(y): int(y + h), int(x): int(x + w)]
        outside = out.copy()
        outside[int(y): int(y + h), int(x): int(x + w)] = 0
        assert (inside > 200).sum() > 100, corner  # the number is drawn ...
        assert (outside > 100).sum() == 0, (corner, int((outside > 100).sum()))  # ... and only on the picture


def test_number_centred_on_the_bar():
    """The number sits centred over the bar, for a long bar and for a bar shorter than its number."""
    import matplotlib
    for um, wider in ((0.5, "text"), (10.0, "bar")):
        for corner in ("bottom-right", "bottom-left"):
            st = F.Style(ncols=1, label_on=False, scalebar_on="all", scalebar_um=um, scalebar_pt=14,
                         scalebar_corner=corner)
            p = _panel(90, "a")
            p.um_per_px = 0.05
            with matplotlib.rc_context(F.RC):
                fig, _ = F.render([F.Item(p)], st)
                fig.canvas.draw()
                r = fig.canvas.get_renderer()
                ax = fig.axes[0]
                bar = ax.patches[0].get_window_extent(renderer=r)
                txt = ax.texts[0].get_window_extent(renderer=r)
                box = ax.get_window_extent(r)
            assert abs((bar.x0 + bar.x1) / 2 - (txt.x0 + txt.x1) / 2) < 0.5, (um, corner, bar, txt)
            assert (txt.width > bar.width) == (wider == "text"), (um, txt.width, bar.width)
            gap = box.x1 - max(bar.x1, txt.x1) if corner == "bottom-right" else min(bar.x0, txt.x0) - box.x0
            assert 3 < gap < 30, (um, corner, gap)  # the wider of the two keeps the usual distance


def test_fixed_length_too_long_for_a_picture():
    """A fixed bar length longer than a picture's field must not drop the bar: that picture gets its
    own automatic length (and bar_length says so); a fixed length that fits is used as it is."""
    import matplotlib
    wide, narrow = _panel(90, "wide"), _panel(90, "narrow")  # 400 px each
    wide.um_per_px, narrow.um_per_px = 0.5, 0.1  # fields 200 µm and 40 µm
    items = [F.Item(wide), F.Item(narrow)]
    st = F.Style(ncols=2, label_on=False, scalebar_on="all", scalebar_um=50.0)
    assert F.bar_length(items[0], st) == (50.0, False)
    assert F.bar_length(items[1], st) == (10.0, True)  # 50 µm > 90 % of 40 µm: 10 µm instead
    with matplotlib.rc_context(F.RC):
        fig, _ = F.render(items, st)
        widths = [round(ax.patches[0].get_width(), 3) if ax.patches else None for ax in fig.axes]
    assert widths == [100.0, 100.0], widths  # 50 µm / 0.5 and 10 µm / 0.1: both pictures have a bar
    assert F.info(items, st)["panels"][1]["scalebar_um"] == 10.0
    nobar = F.Item(_panel(90, "x"))
    nobar.panel.um_per_px = None
    assert F.bar_length(nobar, st) is None


if __name__ == "__main__":
    test_colour_per_picture()
    test_automatic_colour()
    test_outline_contrasts_with_the_bar()
    test_number_stays_inside_the_picture()
    test_number_centred_on_the_bar()
    test_fixed_length_too_long_for_a_picture()
    print("ok")

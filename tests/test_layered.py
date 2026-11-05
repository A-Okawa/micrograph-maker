"""Layered EDS images ("EDS Layered Image"): the colour key drawn into the picture is found, its
element symbols and colours are read, its rows are cut off, and the label is drawn in the layers'
colours.

    python tests/test_layered.py
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from edsfig import aztec, legend, textread  # noqa: E402
from edsfig import figure as F  # noqa: E402
from test_synthetic import FONTS, _docx, _panel  # noqa: E402

MAGENTA, RED = (255, 0, 255), (255, 0, 0)


def _keyed(rng, key=(("O", MAGENTA), ("Si", RED)), electron=True, W=400, H=300) -> np.ndarray:
    """A layered picture with an AZtec-style colour key in the bottom-left corner: boxes shaded
    lighter at the top, a light outline, white letters (black on the white electron box)."""
    noise = np.zeros((H, W, 3), np.uint8)
    noise[..., 0] = rng.integers(40, 140, (H, W))
    noise[..., 2] = rng.integers(20, 110, (H, W))
    im = Image.fromarray(noise)
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(str(FONTS.parent / "fonts" / "Inter-Regular.otf"), 15)
    h, x, y0 = 24, 4, H - 4 - 24
    boxes = list(key) + ([("Electron", (255, 255, 255))] if electron else [])
    for text, col in boxes:
        w = int(d.textlength(text, font=font)) + 10
        for r in range(h):  # lighter at the top
            t = 0.25 * (1 - r / h)
            d.line([(x, y0 + r), (x + w - 1, y0 + r)], fill=tuple(int(c + (255 - c) * t) for c in col))
        d.rectangle([x, y0, x + w - 1, y0 + h - 1], outline=tuple(int(c + (255 - c) * 0.6) for c in col))
        ink = (40, 40, 40) if text == "Electron" else (255, 255, 255)
        l, t_, r_, b_ = d.textbbox((0, 0), text, font=font)
        d.text((x + (w - (r_ - l)) / 2 - l, y0 + (h - (b_ - t_)) / 2 - t_), text, font=font, fill=ink)
        x += w + 5
    return np.asarray(im)


def _maps(rng, W=400, H=300):
    o = np.zeros((H, W, 3), np.uint8)
    o[..., 0] = o[..., 2] = rng.integers(0, 255, (H, W))
    si = np.zeros((H, W, 3), np.uint8)
    si[..., 0] = rng.integers(0, 255, (H, W))
    return o, si


def _report(tmp: Path, extra=()):
    rng = np.random.default_rng(1)
    sem = np.repeat(rng.integers(30, 200, (300, 400, 1), dtype=np.uint8), 3, axis=2)
    o, si = _maps(rng)
    imgs = [_panel("Electron Image 1", sem, 80, "5µm", size=(420, 360)),
            _panel("O Kα1", o, 80, "5µm", size=(420, 360)),
            _panel("Si Kα1", si, 80, "5µm", size=(420, 360)),
            _panel("EDS Layered Image 2", _keyed(rng), 80, "5µm", size=(420, 360))] + list(extra)
    reader = textread.Reader(store_path=tmp / "ex.json", learn=False)
    return aztec.read_report(_docx(imgs), name="layered.docx", reader=reader)


def test_key_found_read_and_cut(tmp: Path = Path(".")):
    rep = _report(tmp)
    kinds = [p.kind for p in rep.panels]
    assert kinds == ["electron", "map", "map", "layered"], kinds
    lay = rep.panels[3]
    assert lay.legend is not None and [c.electron for c in lay.legend.chips] == [False, False, True]
    assert lay.label == "O Si" and lay.include, (lay.label, lay.include)
    for (t, c), want in zip(lay.layers, (MAGENTA, RED)):
        assert max(abs(a - b) for a, b in zip(c, want)) <= 40, (t, c)
    # the key's rows are cut off the bottom, the rest is the exact embedded picture
    assert 24 <= lay.band_px <= 34, lay.band_px
    assert lay.size == (400, 300 - lay.band_px)
    fx0, fy0, fx1, fy1 = lay.full_box
    assert lay.box == (fx0, fy0, fx1, fy1 - lay.band_px)  # the top of the picture, untouched
    aztec.set_band(lay, 0)
    assert lay.size == (400, 300)  # kept whole
    aztec.set_band(lay, None)
    assert lay.size == (400, 300 - lay.band_px)
    # the maps have no key, and their own labels
    assert [p.label for p in rep.panels[1:3]] == ["O", "Si"] and all(p.legend is None for p in rep.panels[:3])
    # the field of view is the same for the whole set (the width is unchanged by the cut)
    assert not [w for w in rep.warnings if "視野幅" in w], rep.warnings


def test_read_without_maps(tmp: Path = Path(".")):
    """A loose AZtec picture: the letters of the boxes alone."""
    rng = np.random.default_rng(2)
    png = io.BytesIO()
    _panel("EDS Layered Image 4", _keyed(rng), 80, "5µm", size=(420, 360)).save(png, "PNG")
    reader = textread.Reader(store_path=tmp / "ex2.json", learn=False)
    rep = aztec.read_images([(png.getvalue(), "layered.png")], reader=reader)
    p = rep.panels[0]
    assert p.kind == "layered" and p.label == "O Si", (p.kind, p.label, [(c.text, c.score) for c in p.legend.chips])


def test_title_without_key(tmp: Path = Path(".")):
    """A layered image exported without its key: the title "EDS Layered Image" still tells."""
    rng = np.random.default_rng(3)
    mix = np.zeros((300, 400, 3), np.uint8)
    mix[..., 0] = rng.integers(0, 255, (300, 400))
    mix[..., 2] = rng.integers(0, 120, (300, 400))
    rep = _report(tmp, extra=[_panel("EDS Layered Image 3", mix, 80, "5µm", size=(420, 360))])
    p = rep.panels[4]
    assert p.kind == "layered" and p.label == "" and p.legend is None, (p.kind, p.label)


def test_label_in_layer_colours(tmp: Path = Path(".")):
    rep = _report(tmp)
    items = [F.Item(p, p.label) for p in rep.panels]
    st = F.Style(ncols=2, label_pt=16, scalebar_on="none", label_color="white")
    out = np.asarray(Image.open(io.BytesIO(F.export(items, st, "png"))).convert("RGB")).astype(int)
    placed, g = F.layout(items, st)

    def count(box, col):
        x, y, w, h, k = box
        r = out[int(y): int(y + 0.25 * h), int(x): int(x + 0.5 * w)]
        return int((np.abs(r - np.array(col)).max(axis=2) <= 40).sum())

    lay_box, o_box = g.boxes[3], g.boxes[1]
    assert count(lay_box, MAGENTA) > 150 and count(lay_box, RED) > 150, (count(lay_box, MAGENTA), count(lay_box, RED))
    assert count(o_box, (255, 255, 255)) > 150  # the maps keep the label colour of the settings
    fig, _ = F.render(items, st)
    texts = [(t.get_text(), t.get_color()) for t in fig.axes[3].texts]
    assert [t for t, _ in texts] == ["O", "Si"], texts
    info = F.info(items, st, "t")
    assert info["panels"][3]["layers"][0][0] == "O" and info["panels"][3]["key_rows_removed"] == rep.panels[3].band_px
    # an edited label: unknown words keep the label colour
    words = F.label_words("Si TiO2", rep.panels[3], "white")
    assert words[0][0] == "Si" and words[1] == ("TiO2", "white")
    assert F.label_words("Sample A", rep.panels[0], "white") == [("Sample A", "white")]


def test_no_key_in_ordinary_pictures():
    rng = np.random.default_rng(4)
    o, si = _maps(rng)
    for img in (o, si, np.repeat(rng.integers(0, 255, (300, 400, 1), dtype=np.uint8), 3, axis=2),
                np.zeros((300, 400, 3), np.uint8)):
        assert legend.find_legend(img) is None


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        for name, fn in list(globals().items()):
            if name.startswith("test_"):
                fn(Path(d)) if fn.__code__.co_argcount else fn()
    print("ok")

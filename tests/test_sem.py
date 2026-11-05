"""SEM images: the instrument's info band is removed and its scale bar is measured and read.

Synthetic pictures in the two common styles: white band with a tick ruler (Hitachi tabletop)
and black band with a solid bar (JEOL).   python tests/test_sem.py
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

from edsfig import aztec, textread  # noqa: E402
from edsfig import figure as F  # noqa: E402

FONTS = ROOT / "edsfig" / "assets" / "fonts"


def _micrograph(w: int, h: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(120, 25, (h, w)).clip(30, 200)
    return base.astype(np.uint8)


def hitachi_like(bar_px: int = 300, label: str = "50 um") -> bytes:
    """640x480 picture + 40 px white band: tick ruler (ticks at both ends) and label below its end."""
    img = np.full((520, 640), 255, np.uint8)
    img[:480] = _micrograph(640, 480, 1)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    x0, x1 = 630 - bar_px, 630
    d.rectangle([x0, 483, x1, 487], fill=0)
    for x in np.linspace(x0, x1, 11):
        d.line([(round(x), 480), (round(x), 482)], fill=0)
    f = ImageFont.truetype(str(FONTS / "LiberationSans-Regular.ttf"), 17)
    d.text((40, 492), "2026/10/23 07:20 H  SD4.2  x800", font=f, fill=0)
    tw = d.textlength(label, font=f)
    d.text((x1 - tw, 492), label, font=f, fill=0)
    b = io.BytesIO()
    im.save(b, "TIFF")
    return b.getvalue()


def jeol_like(bar_px: int = 40, label: str = "10µm") -> bytes:
    """1280x960 picture + 64 px black band: solid white bar with the label to its right."""
    img = np.zeros((1024, 1280), np.uint8)
    img[:960] = _micrograph(1280, 960, 2)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([1000, 963, 1000 + bar_px - 1, 977], fill=255)
    f = ImageFont.truetype(str(FONTS / "FreeMonoBold.ttf"), 22)
    d.text((1150, 962), label, font=f, fill=255)
    d.text((470, 995), "x330      15.0kV LED      SEM     WD 9.9mm", font=f, fill=255)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def hitachi_fe_like(step: float = 41.45, label: str = "2.00µm") -> bytes:
    """1280x960 picture + 62 px black band (Hitachi FE-SEM): the ruler is a row of 11 white ticks at
    the top of the band, the label (the length from the first tick to the last) below its right end."""
    img = np.zeros((960, 1280), np.uint8)
    img[:898] = _micrograph(1280, 898, 3)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    x_last = 1249.5
    for k in range(11):
        x = x_last - (10 - k) * step
        d.rectangle([round(x) - 2, 898, round(x) + 2, 910], fill=255)
    f = ImageFont.truetype(str(FONTS / "LiberationSans-Regular.ttf"), 34)
    d.text((14, 912), "15.0kV 14.6mm M-x35.0k SE(U) 2026/09/01", font=f, fill=255)
    tw = d.textlength(label, font=f)
    d.text((1234 - tw, 912), label, font=f, fill=255)
    b = io.BytesIO()
    im.save(b, "TIFF")
    return b.getvalue()


def test_tick_ruler(tmp: Path = Path(".")):
    """A ruler drawn as a row of ticks (no line between them): first to last tick."""
    reader = textread.Reader(store_path=tmp / "ex.json", learn=False)
    reader.exemplars = {"element": [], "scale": []}
    for step, label in ((41.45, "2.00µm"), (25.0, "500nm"), (60.0, "5.00µm")):
        p = aztec.read_images([(hitachi_fe_like(step, label), "fe.tif")], reader=reader).panels[0]
        assert p.band_px == 62 and p.size == (1280, 898), (label, p.band_px, p.size)
        assert abs(p.scalebar.length_px - round(10 * step)) <= 1.0, (label, p.scalebar.length_px)
        assert textread.parse_length(p.scalebar.text) == textread.parse_length(label), (label, p.scalebar.text)
    # a band without any ruler: no bar, nothing made up from the letters
    from edsfig import sem

    g = np.asarray(Image.open(io.BytesIO(hitachi_fe_like())).convert("L")).astype(np.float32)
    g[898:912] = 0  # ticks removed
    band = sem.find_band(g)
    up, down = sem.band_inks(g, band)
    assert sem.find_bar(up, down, 1280) is None


def jeol_two_lines(bar_px: int = 100, label: str = "10µm") -> bytes:
    """1280x960 picture + 66 px black band with two lines of Consolas-like text (JEOL EPMA): the bar
    and its label on the first line, the field of view and date on the second, and a logo at the
    right spanning both lines."""
    import matplotlib

    mono = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSansMono.ttf"
    img = np.zeros((1026, 1280), np.uint8)
    img[:960] = _micrograph(1280, 960, 5)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(str(mono), 21)
    d.text((70, 966), "BED-C   25.0kV  WD 10.7mm  x1.00k  Std.P.C.68  STD", font=f, fill=255)
    d.rectangle([980, 972, 980 + bar_px - 1, 978], fill=255)
    d.text((980 + bar_px + 16, 966), label, font=f, fill=255)
    for x, text in ((626, "000"), (752, "FOV:128x96µm"), (1013, "10/16/2026")):  # the date under the label
        d.text((x, 994), text, font=f, fill=255)
    d.ellipse([1196, 978, 1256, 1010], outline=255, width=4)  # the maker's logo, across both lines
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


def test_two_line_band(tmp: Path = Path(".")):
    """A two-line band whose logo bridges the lines: the label on the bar's line is still found."""
    reader = textread.Reader(store_path=tmp / "ex.json", learn=False)
    reader.exemplars = {"element": [], "scale": []}
    for bar_px, label in ((100, "10µm"), (125, "5µm"), (80, "500nm")):
        p = aztec.read_images([(jeol_two_lines(bar_px, label), "jeol.png")], reader=reader).panels[0]
        assert p.band_px == 66 and p.size == (1280, 960), (label, p.band_px, p.size)
        assert abs(p.scalebar.length_px - bar_px) <= 1.0, (label, p.scalebar.length_px)
        assert textread.parse_length(p.scalebar.text) == textread.parse_length(label), (label, p.scalebar.text)


def jeol_flat_top(seed: int = 7) -> bytes:
    """1280x960 picture whose top 69 rows are an almost flat dark stretch with a bright particle (it
    looks like a band), and the real 64 px black band at the bottom: a 33 px bar on the first line
    with "10µm" far to its right, "x300 5.0kV LED SEM WD 9.6mm" on the second."""
    rng = np.random.default_rng(seed)
    img = np.ones((1024, 1280), np.uint8)
    img[:960] = _micrograph(1280, 960, seed)
    img[:69] = rng.normal(99, 2.0, (69, 1280)).clip(0, 255).astype(np.uint8)  # dark, nearly flat
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.ellipse([240, 40, 460, 80], fill=200)  # a particle at the top edge of the stretch
    f = ImageFont.truetype(str(FONTS / "FreeMonoBold.ttf"), 24)
    d.rectangle([1017, 963, 1049, 977], fill=255)  # 33 px = 10 µm
    d.text((1188, 960), "10µm", font=f, fill=255)
    for x, t in ((476, "x300"), (697, "5.0kV"), (816, "LED"), (970, "SEM"), (1121, "WD 9.6mm")):
        d.text((x, 994), t, font=f, fill=255)
    out = io.BytesIO()
    im.save(out, "TIFF")
    return out.getvalue()


def test_drawn_band_beats_flat_picture(tmp: Path = Path(".")):
    """A flat stretch of picture at the top must not be taken for the band: the drawn one at the
    bottom (one exact background level) is cut, and its short bar and label are read."""
    reader = textread.Reader(store_path=tmp / "ex.json", learn=False)
    reader.exemplars = {"element": [], "scale": []}
    p = aztec.read_images([(jeol_flat_top(), "x300.tif")], reader=reader).panels[0]
    assert p.band is not None and p.band.side == "bottom" and p.band_px == 64, (p.band, p.band_px)
    assert p.size == (1280, 960), p.size
    assert abs(p.scalebar.length_px - 33) <= 0.6, p.scalebar.length_px
    assert textread.parse_length(p.scalebar.text) == (10.0, "µm"), p.scalebar.text


def test_bands_bars_and_labels(tmp: Path = Path(".")):
    reader = textread.Reader(store_path=tmp / "ex.json", learn=False)
    reader.exemplars = {"element": [], "scale": []}  # read from the fonts only, nothing learned
    rep = aztec.read_images([(hitachi_like(), "a.tif"), (jeol_like(), "b.png")], reader=reader)
    a, b = rep.panels
    assert rep.kind == "images" and not rep.warnings
    assert a.size == (640, 480) and a.band_px == 40, (a.size, a.band_px)
    assert b.size == (1280, 960) and b.band_px == 64, (b.size, b.band_px)
    assert a.kind == b.kind == "electron"
    assert abs(a.scalebar.length_px - 300) <= 0.5, a.scalebar.length_px  # tick centre to tick centre
    assert abs(b.scalebar.length_px - 40) <= 0.5, b.scalebar.length_px  # solid bar
    assert textread.parse_length(a.scalebar.text) == (50.0, "µm"), a.scalebar.text
    assert textread.parse_length(b.scalebar.text) == (10.0, "µm"), b.scalebar.text
    assert abs(a.um_per_px - 50 / 300) < 1e-3 and abs(b.um_per_px - 10 / 40) < 1e-3
    # the micrograph is cut out bit-exactly (no band, no resampling)
    src = np.asarray(Image.open(io.BytesIO(hitachi_like())).convert("RGB"))[:480]
    assert np.array_equal(a.image, src)
    # the band can be kept (0) or set by hand
    aztec.set_band(a, 0)
    assert a.size == (640, 520)
    aztec.set_band(a, None)
    assert a.size == (640, 480)


def test_long_ruler(tmp: Path = Path(".")):
    """A ruler longer than half the width (Hitachi at x1.8k / x500): its rows are mostly bar, not
    background, and still belong to the band."""
    reader = textread.Reader(store_path=tmp / "ex.json", learn=False)
    reader.exemplars = {"element": [], "scale": []}
    for bar_px, label in ((357, "50 um"), (396, "200 um"), (560, "20 um")):
        p = aztec.read_images([(hitachi_like(bar_px, label), "a.tif")], reader=reader).panels[0]
        assert p.band_px == 40 and p.size == (640, 480), (bar_px, p.band_px)
        assert abs(p.scalebar.length_px - bar_px) <= 0.5, (bar_px, p.scalebar.length_px)
        assert textread.parse_length(p.scalebar.text) == textread.parse_length(label), (label, p.scalebar.text)


def test_long_ruler_jpeg(tmp: Path = Path(".")):
    """The same pictures saved as JPEG: the ruler's black (0) comes out as 0-20 and the white
    around it as 240-255.  The band must still be cut whole (no ruler left on the picture) and its
    scale read."""
    reader = textread.Reader(store_path=tmp / "ex.json", learn=False)
    reader.exemplars = {"element": [], "scale": []}
    for bar_px, label in ((357, "50 um"), (396, "200 um"), (560, "20 um")):
        im = Image.open(io.BytesIO(hitachi_like(bar_px, label)))
        for q in (90, 75, 50):
            b = io.BytesIO()
            im.save(b, "JPEG", quality=q)
            p = aztec.read_images([(b.getvalue(), "a.jpg")], reader=reader).panels[0]
            assert p.band_px == 40 and p.size == (640, 480), (bar_px, q, p.band_px)
            assert p.scalebar is not None and abs(p.scalebar.length_px - bar_px) <= 1, (bar_px, q, p.scalebar)
            assert textread.parse_length(p.scalebar.text) == textread.parse_length(label), (label, q, p.scalebar.text)


def test_metadata_wins(tmp: Path = Path(".")):
    reader = textread.Reader(store_path=tmp / "ex.json", learn=False)
    txt = "$CM_MAG 330\n$$SM_MICRON_BAR 35\n$$SM_MICRON_MARKER 10um\n"
    rep = aztec.read_images([(jeol_like(), "b.png")], reader=reader, txts={"b": txt})
    p = rep.panels[0]
    assert p.scale_source == "metadata" and abs(p.um_per_px - 10 / 35) < 1e-9
    txt2 = "SampleName=x\nPixelSize=312.5\n"
    rep = aztec.read_images([(hitachi_like(), "a.tif")], reader=reader, txts={"a": txt2})
    assert abs(rep.panels[0].um_per_px - 0.3125) < 1e-9


def test_figure_with_sample_names(tmp: Path = Path(".")):
    reader = textread.Reader(store_path=tmp / "ex.json", learn=False)
    rep = aztec.read_images([(hitachi_like(), "a.tif"), (jeol_like(), "b.png")], reader=reader)
    items = [F.Item(p, name) for p, name in zip(rep.panels, ["Sample A", "Sample B"])]
    for mode in ("width", "height", "box"):
        st = F.Style(ncols=2, size_mode=mode, width_mm=170, height_mm=100, scalebar_on="all")
        pdf = F.export(items, st, "pdf")
        assert pdf[:5] == b"%PDF-"
        placed, g = F.layout(items, st)
        for p, (x, y, w, h, k) in zip(placed, g.boxes):
            iw, ih = p.item.panel.size
            assert abs((w / h) / (iw / ih) - 1) < 1e-12


def test_no_band_in_plain_micrographs():
    """Dark or saturated areas at the edge of a micrograph must not be taken for an info band."""
    from edsfig import sem

    rng = np.random.default_rng(0)
    cases = []
    g = rng.normal(120, 25, (480, 640)).clip(0, 255)
    cases.append(g)
    g = g.copy(); g[380:] = rng.normal(20, 8, (100, 640)).clip(0, 255); g[400:410, 100:200] = 230
    cases.append(g)
    g = rng.normal(120, 25, (480, 640)).clip(0, 255); g[400:] = rng.normal(250, 4, (80, 640)).clip(0, 255)
    g[430:440, 300:350] = 30
    cases.append(g)
    g = rng.normal(15, 6, (480, 640)).clip(0, 255); g[200:260, 200:400] = 200
    cases.append(g)
    for g in cases:
        assert sem.find_band(g.astype(np.float32)) is None


if __name__ == "__main__":
    test_no_band_in_plain_micrographs()
    with tempfile.TemporaryDirectory() as d:
        test_bands_bars_and_labels(Path(d))
        test_long_ruler(Path(d))
        test_long_ruler_jpeg(Path(d))
        test_tick_ruler(Path(d))
        test_two_line_band(Path(d))
        test_drawn_band_beats_flat_picture(Path(d))
        test_metadata_wins(Path(d))
        test_figure_with_sample_names(Path(d))
    print("ok")

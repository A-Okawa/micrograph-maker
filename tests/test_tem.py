"""TEM and FE-SEM layouts: FEI/Thermo databar, TEM-EDS map strip, JEOL text drawn on the picture.

Synthetic pictures that imitate each layout.   python tests/test_tem.py
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

from edsfig import aztec, sem, textread  # noqa: E402

FONT = str(ROOT / "edsfig" / "assets" / "fonts" / "LiberationSans-Regular.ttf")


def _micrograph(w: int, h: int, seed: int, lo: float = 60, hi: float = 200) -> np.ndarray:
    """Smooth random texture (no flat areas, no saturated pixels)."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(lo, hi, (h // 16 + 2, w // 16 + 2))
    big = np.asarray(Image.fromarray(small.astype(np.float32), mode="F").resize((w, h), Image.BICUBIC))
    return np.clip(big + rng.normal(0, 6, (h, w)), lo - 20, hi + 20).astype(np.uint8)


def _png(im: Image.Image, fmt: str = "PNG", **kw) -> bytes:
    b = io.BytesIO()
    im.save(b, fmt, **kw)
    return b.getvalue()


def fei_like() -> bytes:
    """1024 x 884 picture + 59 px grey databar: a table of values with white borders, and the
    scale bar split by its label ("10 µm") with ticks at its outer ends.  Another cell holds a
    length too ("HFW 37.2 µm"), which must not be taken for the bar's label."""
    W, H, B = 1024, 884, 59
    img = np.full((H + B, W), 171, np.uint8)
    img[:H] = _micrograph(W, H, 4)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 19)
    y0 = H
    d.line([(0, y0), (W - 1, y0)], fill=255)
    cells = [(0, 54, "", ""), (55, 162, "HV", "10.00 kV"), (163, 248, "mag", "4006 x"),
             (249, 357, "WD", "15.0 mm"), (358, 459, "HFW", "37.2 µm"), (460, 505, "tilt", "7 °")]
    for x0, x1, top, bottom in cells:
        d.line([(x1, y0), (x1, y0 + B - 1)], fill=255)
        for txt, yy in ((top, y0 + 3), (bottom, y0 + 31)):
            if txt:
                tw = d.textlength(txt, font=f)
                d.text(((x0 + x1) / 2 - tw / 2, yy), txt, font=f, fill=255)
    d.line([(506, y0 + 29), (W - 1, y0 + 29)], fill=255)  # the bar's cell is split in two rows
    ym = y0 + 14
    lx0, lx1 = 618, 886  # tick centres: 268 px
    d.line([(lx0, ym), (715, ym)], fill=255, width=2)
    d.line([(790, ym), (lx1, ym)], fill=255, width=2)
    for x in (lx0, lx1):
        d.line([(x, ym - 3), (x, ym + 3)], fill=255, width=1)
    tw = d.textlength("10 µm", font=f)
    d.text((752 - tw / 2, y0 + 3), "10 µm", font=f, fill=255)
    return _png(im, "TIFF")


def teds_like(element: str = "C K") -> bytes:
    """TEM-EDS map: intensity colour bar along the left edge (11 px), red map, and a 19 px white
    strip with an outlined scale bar, its label and the element name at the right."""
    rng = np.random.default_rng(5)
    H, W = 512, 523
    rgb = np.full((H + 19, W, 3), 255, np.uint8)
    t = 1 - np.arange(H) / (H - 1)
    rgb[:H, :11, 0] = (255 * t)[:, None]
    rgb[:H, :11, 1] = (27 * t)[:, None]
    rgb[:H, :11, 2] = 0
    sig = np.clip(rng.gamma(1.5, 40, (H, W - 11)), 0, 255)
    rgb[:H, 11:, 0] = sig
    rgb[:H, 11:, 1] = sig * 0.1
    rgb[:H, 11:, 2] = 0
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    d.rectangle([13, H + 7, 13 + 128, H + 11], outline=(0, 0, 0))
    f = ImageFont.truetype(FONT, 14)
    d.text((148, H + 2), "2.5 µm", font=f, fill=(0, 0, 0))
    tw = d.textlength(element, font=f)
    d.text((W - 4 - tw, H + 2), element, font=f, fill=(0, 0, 0))
    return _png(im)


def overlay_like(w: int = 1280, h: int = 960, bar: tuple = (358, 577), label: str = "1 µm",
                 px: int = 30, quality: int = 90) -> bytes:
    """JEOL FE-SEM: white text with a black shadow drawn directly onto the bottom of the picture
    ("5.0kV  x22,000  ____ 1 µm  7.9 10 LED"), saved as JPEG."""
    img = _micrograph(w, h, 6)
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, px)
    s = w / 1280
    yb = round(896 * h / 960)
    yt = yb + max(6, round(10 * s))
    x0, x1 = bar
    th = max(2, round(4 * s))

    def text(x, y, t):
        d.text((x + 1, y + 1), t, font=f, fill=(0, 0, 0))
        d.text((x, y), t, font=f, fill=(255, 255, 255))

    d.rectangle([x0 + 1, yb + 1, x1 + 1, yb + th], fill=(0, 0, 0))
    d.rectangle([x0, yb, x1, yb + th - 1], fill=(255, 255, 255))
    text(round(65 * s), yt, "5.0kV")
    text(round(245 * s), yt, "x22,000")
    tw = d.textlength(label, font=f)
    text(x1 - tw, yt, label)
    text(round(1085 * s), yt, "7.9 10 LED")
    return _png(im, "JPEG", quality=quality)


def _reader(tmp: Path) -> textread.Reader:
    r = textread.Reader(store_path=tmp / "ex.json", learn=False)
    r.exemplars = {"element": [], "scale": []}  # read from the fonts only, nothing learned
    return r


def test_fei_databar(tmp: Path = Path(".")):
    rep = aztec.read_images([(fei_like(), "fei.tif")], reader=_reader(tmp))
    p = rep.panels[0]
    assert p.band_px == 59 and p.size == (1024, 884), (p.band_px, p.size)
    assert abs(p.scalebar.length_px - 268) <= 1, p.scalebar.length_px
    assert textread.parse_length(p.scalebar.text) == (10.0, "µm"), p.scalebar.text  # not "37.2 µm"
    assert abs(p.um_per_px - 10 / p.scalebar.length_px) < 1e-9


def test_teds_map(tmp: Path = Path(".")):
    rep = aztec.read_images([(teds_like("C K"), "c.png"), (teds_like("O K"), "o.png")], reader=_reader(tmp))
    for p, el in zip(rep.panels, ("C", "O")):
        assert p.crop_lr == (11, 0), p.crop_lr  # the colour bar is cut off
        assert p.band_px == 19 and p.size == (512, 512), (p.band_px, p.size)
        assert p.kind == "map" and p.label == el, (p.kind, p.label, p.label_candidates[:3])
        assert abs(p.scalebar.length_px - 128) <= 1, p.scalebar.length_px
        assert textread.parse_length(p.scalebar.text) == (2.5, "µm"), p.scalebar.text
        # the map is an exact crop: no colour bar column, no strip row
        src = np.asarray(Image.open(io.BytesIO(teds_like("C K"))).convert("RGB"))
        assert np.array_equal(rep.panels[0].image, src[:512, 11:])
    aztec.set_band(rep.panels[0], 0)  # keeping the strip keeps the colour bar removed
    assert rep.panels[0].size == (512, 531)


def test_overlay_text_on_picture(tmp: Path = Path(".")):
    rd = _reader(tmp)
    rep = aztec.read_images([(overlay_like(), "a.jpg"),
                             (overlay_like(640, 480, (300, 459), "500 nm", px=16), "b.jpg")], reader=rd)
    a, b = rep.panels
    assert a.band is not None and a.band.overlay and a.band.side == "bottom", a.band
    assert 896 - 6 <= a.band.start <= 896 - 1, a.band.start  # from just above the bar to the edge
    assert a.size == (1280, a.band.start), a.size
    assert abs(a.scalebar.length_px - 220) <= 1, a.scalebar.length_px
    assert textread.parse_length(a.scalebar.text) == (1.0, "µm"), a.scalebar.text
    assert abs(a.um_per_px * 1000 - 1000 / 220) < 0.05, a.um_per_px
    assert b.band is not None and b.band.overlay, b.band
    assert abs(b.scalebar.length_px - 160) <= 1, b.scalebar.length_px
    assert textread.parse_length(b.scalebar.text) == (500.0, "nm"), b.scalebar.text
    aztec.set_band(a, 0)  # the overlaid rows can be kept
    assert a.size == (1280, 960)


def test_no_overlay_in_plain_pictures():
    """Bright or dark features near the edge of a micrograph are not an overlay."""
    rng = np.random.default_rng(1)
    cases = []
    g = _micrograph(800, 600, 7).astype(np.float32)
    cases.append(g)
    g2 = g.copy(); g2[560:564, 100:400] = 255  # a lone bright line, no text
    cases.append(g2)
    g3 = g.copy()
    for _ in range(40):  # bright particles
        y, x, r = rng.integers(480, 600), rng.integers(0, 800), rng.integers(3, 9)
        yy, xx = np.ogrid[:600, :800]
        g3[(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = 255
    cases.append(g3)
    g4 = g.copy()
    for _ in range(60):  # dark particles (TEM bright field)
        y, x, r = rng.integers(480, 600), rng.integers(0, 800), rng.integers(4, 12)
        yy, xx = np.ogrid[:600, :800]
        g4[(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = 5
    g4[590:592, 200:330] = 5
    cases.append(g4)
    g5 = g.copy(); g5[540:] = 255  # saturated bottom
    cases.append(g5)
    for k, g in enumerate(cases):
        assert sem.find_overlay(g) is None, k


if __name__ == "__main__":
    test_no_overlay_in_plain_pictures()
    with tempfile.TemporaryDirectory() as d:
        test_fei_databar(Path(d))
        test_teds_map(Path(d))
        test_overlay_text_on_picture(Path(d))
    print("ok")

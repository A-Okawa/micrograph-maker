"""Self-contained checks on a synthetic AZtec-like report (no real data needed).

    python -m pytest tests        # or:  python tests/test_synthetic.py
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edsfig import aztec, textread  # noqa: E402
from edsfig import figure as F  # noqa: E402

FONTS = ROOT / "edsfig" / "assets" / "fonts"


def _panel(title: str, rgb: np.ndarray, bar_px: int, label: str, size=(283, 254), box=(8, 25)) -> Image.Image:
    W, H = size
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    fb = ImageFont.truetype(str(FONTS / "Carlito-Bold.ttf"), 16)
    fr = ImageFont.truetype(str(FONTS / "Carlito-Regular.ttf"), 14)
    tw = d.textlength(title, font=fb)
    d.text(((W - tw) / 2, 6), title, font=fb, fill=(0, 0, 0, 255))
    x0, y0 = box
    h, w = rgb.shape[:2]
    im.paste(Image.fromarray(rgb).convert("RGBA"), (x0, y0))
    yb = y0 + h + 3
    xl, xr = x0 + 5, x0 + 5 + bar_px
    d.line([(xl, yb), (xr, yb)], fill=(0, 0, 0, 255), width=2)
    d.line([(xl, yb), (xl, yb + 8)], fill=(0, 0, 0, 255), width=2)
    d.line([(xr, yb), (xr, yb + 8)], fill=(0, 0, 0, 255), width=2)
    lw = d.textlength(label, font=fr)
    d.text(((xl + xr - lw) / 2, yb + 3), label, font=fr, fill=(0, 0, 0, 255))
    return im


def _docx(images: list[Image.Image]) -> bytes:
    rels, blips = [], []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, im in enumerate(images, 1):
            b = io.BytesIO()
            im.save(b, "PNG")
            z.writestr(f"word/media/image{i}.png", b.getvalue())
            rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image{i}.png"/>')
            blips.append(f'<w:p><w:r><w:drawing><a:blip r:embed="rId{i}"/></w:drawing></w:r></w:p>')
        z.writestr("word/_rels/document.xml.rels", '<?xml version="1.0"?><Relationships>' + "".join(rels) + "</Relationships>")
        z.writestr("word/document.xml", '<?xml version="1.0"?><w:document><w:body>' + "".join(blips) + "</w:body></w:document>")
    return buf.getvalue()


def _report(tmp_store: Path):
    rng = np.random.default_rng(0)
    sem = np.repeat(rng.integers(30, 200, (200, 266, 1), dtype=np.uint8), 3, axis=2)
    ti = np.zeros((200, 266, 3), np.uint8)
    ti[..., 1] = ti[..., 2] = rng.integers(0, 255, (200, 266), dtype=np.uint8)
    o = np.zeros((200, 266, 3), np.uint8)
    o[..., 1] = rng.integers(0, 255, (200, 266), dtype=np.uint8)
    imgs = [_panel("Electron Image 1", sem, 76, "5µm"), _panel("Ti Kα1", ti, 76, "5µm"), _panel("O Kα1", o, 76, "5µm")]
    reader = textread.Reader(store_path=tmp_store / "ex.json", learn=False)
    return aztec.read_report(_docx(imgs), name="synthetic.docx", reader=reader), imgs


def test_geometry_and_reading(tmp_path=Path(".")):
    rep, imgs = _report(Path(tmp_path))
    kinds = [p.kind for p in rep.panels]
    assert kinds == ["electron", "map", "map"], kinds
    for p in rep.panels:
        assert p.size == (266, 200), p.size  # exact image body, nothing else
        assert abs(p.scalebar.length_px - 76) <= 1.0, p.scalebar.length_px
    assert [p.label for p in rep.panels[1:]] == ["Ti", "O"]
    assert rep.panels[0].scalebar.text == "5µm"
    # crop is bit-identical to the embedded pixels
    src = np.asarray(imgs[1])[25:225, 8:274, :3]
    assert np.array_equal(rep.panels[1].image, src)


def test_export_keeps_pixels(tmp_path=Path(".")):
    rep, _ = _report(Path(tmp_path))
    items = [F.Item(p, p.label) for p in rep.panels]
    st = F.Style(ncols=3, oversample=2, label_on=False, scalebar_on="none")
    png = F.export(items, st, "png")
    out = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    fig, g = F.render(items, st)
    sem = rep.panels[0].image
    region = out[: g.cell_h, : g.cell_w]
    assert np.array_equal(region, np.kron(sem, np.ones((2, 2, 1), np.uint8)))
    assert g.shrink >= 2.0
    for fmt in ("pdf", "svg", "tiff"):
        assert len(F.export(items, st, fmt)) > 1000


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_geometry_and_reading(Path(d))
        test_export_keeps_pixels(Path(d))
    print("ok")

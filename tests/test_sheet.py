"""JEOL EPMA map windows: maps on teal tiles with their name and scale bar below (synthetic windows
drawn here, lossless and as JPEG), and pictures that must not be taken for one.

    python tests/test_sheet.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edsfig import aztec, sheet, textread  # noqa: E402

FONT = str(Path(aztec.__file__).parent / "assets" / "fonts" / "LiberationSans-Bold.ttf")
TEAL = (0, 128, 128)
NAMES = ["Fe", "Si", "O", "Al", "Mg", "Ca", "SED", "BED-C"]
COLS, ROWS = 4, 2
MAP_W, MAP_H, TOP, GAP_BELOW, LABEL_H, SEP = 256, 192, 40, 20, 19, 4


def _rainbow(v: np.ndarray) -> np.ndarray:
    """Blue - cyan - green - yellow - red, like the maps' colour scale."""
    stops = np.array([[0, 0, 160], [0, 160, 255], [0, 220, 120], [255, 240, 0], [255, 0, 0]], float)
    t = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def window(seed: int = 0) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
    """A map window: tiles on a grid (maps, then two electron images), a panel at the right."""
    rng = np.random.default_rng(seed)
    tile_h = TOP + MAP_H + GAP_BELOW + LABEL_H
    W, H = SEP + COLS * (MAP_W + SEP) + 260, SEP + ROWS * (tile_h + SEP)
    a = np.full((H, W, 3), 240, np.uint8)
    boxes = []
    font = ImageFont.truetype(FONT, 15)
    labels = []
    for k, name in enumerate(NAMES):
        r, c = divmod(k, COLS)
        x0, y0 = SEP + c * (MAP_W + SEP), SEP + r * (tile_h + SEP)
        a[y0: y0 + tile_h, x0: x0 + MAP_W] = TEAL
        my = y0 + TOP
        if name in ("SED", "BED-C"):
            g = np.clip(rng.normal(150, 40, (MAP_H, MAP_W)), 0, 255).astype(np.uint8)
            a[my: my + MAP_H, x0: x0 + MAP_W] = g[..., None]
        else:
            v = rng.random((MAP_H // 2, MAP_W // 2))
            v = np.repeat(np.repeat(v, 2, 0), 2, 1)  # 2x zoom of the measured points, as on screen
            a[my: my + MAP_H, x0: x0 + MAP_W] = _rainbow(0.15 + 0.7 * v)
            if k == 1:  # a smooth blue-green part right at the map's bottom edge, close to the teal
                a[my + MAP_H - 6: my + MAP_H, x0: x0 + 120] = (0, 125, 140)
        boxes.append((x0, my, x0 + MAP_W - 1, my + MAP_H - 1))
        labels.append((x0, my + MAP_H + GAP_BELOW, name))
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    for x0, ly, name in labels:
        bx = x0 + 3 + int(font.getlength(name)) + 12  # the bar after the name, then its length
        bw = bx - x0 + 40 + 10 + int(font.getlength("20 µm")) + 4
        d.rectangle((x0, ly, x0 + bw - 1, ly + LABEL_H - 1), fill=(255, 255, 255))
        d.text((x0 + 3, ly + 1), name, font=font, fill=(0, 0, 0))
        d.line((bx, ly + 9, bx + 39, ly + 9), fill=(0, 0, 0), width=1)  # 40 px = 20 µm
        d.text((bx + 50, ly + 1), "20 µm", font=font, fill=(0, 0, 0))
    px = SEP + COLS * (MAP_W + SEP)  # the panel at the side: colour scales and text
    for k in range(6):
        d.rectangle((px + 10 + 40 * k, 20, px + 22 + 40 * k, 80), fill=tuple(int(v) for v in _rainbow(np.array(k / 5))))
        d.text((px + 6 + 40 * k, 2), "Lv.", font=font, fill=(0, 0, 0))
    return im, boxes


def _jpeg(im: Image.Image, q: int = 85) -> np.ndarray:
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))


def test_tiles_found_exactly():
    im, boxes = window()
    tiles = sheet.sheet_tiles(np.asarray(im))
    assert [t.box for t in tiles] == boxes, [t.box for t in tiles]
    assert all(abs(t.bar_px - 40) < 0.01 for t in tiles), [t.bar_px for t in tiles]


def test_jpeg_keeps_the_boxes():
    im, boxes = window(1)
    tiles = sheet.sheet_tiles(_jpeg(im))
    assert [t.box for t in tiles] == boxes, [t.box for t in tiles]
    assert all(abs(t.bar_px - 40) <= 0.6 for t in tiles), [round(t.bar_px, 2) for t in tiles]


def test_pictures_names_and_scale():
    im, boxes = window(2)
    b = io.BytesIO()
    im.save(b, "PNG")
    R = textread.Reader(store_path=Path(__file__).parent / "_no_store.json", learn=False)
    rep = aztec.read_sheet(b.getvalue(), "window.png", R)
    assert rep.kind == "aztec" and len(rep.panels) == len(NAMES)
    kinds = [p.kind for p in rep.panels]
    assert kinds == ["map"] * 6 + ["electron"] * 2, kinds
    labels = [p.label for p in rep.panels]
    assert labels[:6] == NAMES[:6] and labels[6:] == ["", ""], labels
    assert all(p.scalebar.text in ("20µm", "20um") for p in rep.panels)
    assert all(abs(p.um_per_px - 0.5) < 1e-9 for p in rep.panels), [p.um_per_px for p in rep.panels]
    assert len({id(p.scalebar.label_mask) for p in rep.panels}) == 1  # one scale value for the sheet
    assert len({p.src.uid for p in rep.panels}) == len(NAMES)  # every map its own picture key
    assert all(p.body.shape == (MAP_H, MAP_W, 3) for p in rep.panels)
    # the same file among loose images: the maps are pictures of one set
    rep2 = aztec.read_images([(b.getvalue(), "window.png")], reader=R)
    assert len(rep2.panels) == len(NAMES) and {p.set_id for p in rep2.panels} == {"window.png"}


def test_not_a_sheet():
    rng = np.random.default_rng(3)
    sem = np.clip(rng.normal(120, 30, (480, 640)), 0, 255).astype(np.uint8)
    assert not sheet.is_sheet(np.repeat(sem[..., None], 3, axis=2))
    teal_photo = np.zeros((300, 400, 3), np.uint8)
    teal_photo[:] = TEAL
    teal_photo[100:200, 100:300] = rng.integers(0, 255, (100, 200, 3))  # teal all round, no label box
    assert not sheet.is_sheet(teal_photo)
    im, _ = window(4)
    a = np.asarray(im).copy()
    a[:, :, :] = a[:, :, [1, 0, 2]]  # tiles of another colour
    assert not sheet.is_sheet(a)


if __name__ == "__main__":
    test_tiles_found_exactly()
    test_jpeg_keeps_the_boxes()
    test_pictures_names_and_scale()
    test_not_a_sheet()
    print("ok")

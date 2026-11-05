"""Turning and mirroring: quarter turns are exact, other angles keep the pixel pitch (scale).

    python tests/test_rotate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edsfig import aztec, transform  # noqa: E402
from edsfig import figure as F  # noqa: E402


def _panel(img: np.ndarray, kind: str = "electron", um: float | None = 0.01) -> aztec.Panel:
    h, w = img.shape[:2]
    rgba = np.dstack([img, np.full((h, w, 1), 255, np.uint8)])
    p = aztec.Panel(src=aztec.SourceImage("p.png", rgba, "p", 0, meta_um_per_px=um), box=(0, 0, w - 1, h - 1), kind=kind)
    p.um_per_px = um
    return p


def test_quarter_turns_and_mirrors_are_exact():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)
    assert transform.split_angle(90) == (1, 0.0) and transform.split_angle(-90) == (3, 0.0)
    assert transform.split_angle(135) == (2, -45.0) and transform.split_angle(12.5) == (0, 12.5)
    assert np.array_equal(transform.rotate(img, 90), np.rot90(img, -1))  # clockwise
    assert np.array_equal(transform.rotate(img, -90), np.rot90(img, 1))
    assert np.array_equal(transform.rotate(img, 180), img[::-1, ::-1])
    assert np.array_equal(transform.rotate(img, 450), np.rot90(img, -1))
    assert np.array_equal(transform.rotate(img, 0, flip_h=True), img[:, ::-1])
    assert np.array_equal(transform.rotate(img, 0, flip_v=True), img[::-1])
    assert np.array_equal(transform.rotate(img, 90, flip_h=True), np.rot90(img, -1)[:, ::-1])  # turn, then mirror
    assert transform.rotate(img, 0) is img
    # a marked pixel goes where a clockwise quarter turn puts it: (x, y) -> (H-1-y, x)
    m = np.zeros((30, 40, 3), np.uint8)
    m[5, 33] = 255
    out = transform.rotate(m, 90)
    assert out.shape == (40, 30, 3) and out[33, 30 - 1 - 5, 0] == 255


def test_odd_angles():
    flat = np.full((300, 400, 3), 100, np.uint8)
    for a in (12.5, -30.0, 44.0, 91.0, -170.0):
        out = transform.rotate(flat, a)  # empty corners cut away: only picture is left
        assert out.min() == out.max() == 100, a
        q, rest = transform.split_angle(a)
        w, h = (400, 300) if q % 2 == 0 else (300, 400)
        wr, hr = transform.largest_upright_rect(w, h, rest)
        assert out.shape[1] <= wr and out.shape[0] <= hr and out.shape[1] >= wr - 4 and out.shape[0] >= hr - 4
        whole = transform.rotate(flat, a, fill="white")  # corners kept and filled
        assert (whole == 255).any() and whole.shape[0] * whole.shape[1] > 300 * 400
    # a line rising at 20 degrees becomes level when the picture is turned 20 degrees clockwise
    im = Image.new("RGB", (400, 300), (0, 0, 0))
    d = ImageDraw.Draw(im)
    x0, y0 = 60, 230
    x1, y1 = x0 + 280 * np.cos(np.radians(20)), y0 - 280 * np.sin(np.radians(20))
    d.line([(x0, y0), (x1, y1)], fill=(255, 255, 255), width=3)
    out = transform.rotate(np.asarray(im), 20.0)
    ys, xs = np.nonzero(out[..., 0] > 128)
    slope = np.polyfit(xs, ys, 1)[0]
    assert abs(slope) < 0.01, slope
    # nearest neighbour keeps the original grey levels
    rng = np.random.default_rng(1)
    levels = rng.choice([10, 120, 240], size=(60, 80))
    img = np.repeat(levels[..., None], 3, axis=2).astype(np.uint8)
    out = transform.rotate(img, 17.0, resample="nearest")
    assert set(np.unique(out)) <= {10, 120, 240}


def test_scale_and_layout_after_turning():
    rng = np.random.default_rng(2)
    p = _panel(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8))
    aztec.set_rotation(p, 90)
    assert p.size == (300, 400) and p.body_size == (400, 300) and p.turned
    assert p.field_um == (3.0, 4.0)  # 0.01 µm per pixel either way
    aztec.set_rotation(p, 30)
    w, h = p.size
    assert abs(p.field_um[0] - w * 0.01) < 1e-12
    st = F.Style(ncols=2, scalebar_on="all")
    q = _panel(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8))
    aztec.set_rotation(q, -90, flip_h=True)
    items = [F.Item(p, "a"), F.Item(q, "b")]
    placed, g = F.layout(items, st)
    for pl, (x, y, bw, bh, k) in zip(placed, g.boxes):
        iw, ih = pl.item.size
        assert abs((bw / bh) / (iw / ih) - 1) < 1e-12
    F.render(items, st)  # raises if any picture were drawn with another ratio
    info = F.info(items, st)
    assert info["panels"][0]["turned"]["rotation_cw_deg"] == 30.0 and info["panels"][1]["turned"]["flip_left_right"]
    assert info["panels"][0]["region_in_source"] == [0, 0, 400, 300]
    # trimming works on the turned picture
    st2 = F.Style(ncols=2, align="crop", crop_aspect=1.0)
    for pl in F.layout(items, st2)[0]:
        assert pl.item.size[0] == pl.item.size[1]


def test_field_check_ignores_turning():
    """An EDS set shares one field of view: turning one picture must not look like a mismatch."""
    rng = np.random.default_rng(3)
    a = _panel(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8), "electron")
    b = _panel(rng.integers(0, 256, (150, 200, 3), dtype=np.uint8), "map", um=0.02)
    rep = aztec.Report("x", [a, b], kind="aztec")
    aztec.set_rotation(a, 90)
    aztec.calibrate(rep)
    assert not rep.warnings, rep.warnings
    assert a.um_per_px == 0.01 and b.um_per_px == 0.02


if __name__ == "__main__":
    test_quarter_turns_and_mirrors_are_exact()
    test_odd_angles()
    test_scale_and_layout_after_turning()
    test_field_check_ignores_turning()
    print("ok")

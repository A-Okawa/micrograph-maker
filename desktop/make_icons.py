"""Draws the app icon: a white figure with one grey micrograph (with a scale bar) and three element maps, 2×2.
Writes edsfig/assets/icon.png (1024 px, also used by the app) and desktop/icons/MicrographMaker.png / .ico / .icns
(the PC app).  Run again only when the design changes:  python desktop/make_icons.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parents[1] / "edsfig" / "assets" / "icon.png"
ICONS = Path(__file__).resolve().parent / "icons"
S = 4096  # drawn large, then reduced (smooth edges)
N = 96  # "pixels" of each small map (the blocky look of an element map)


def smooth(rng, n, sigma):
    a = Image.fromarray((rng.random((n, n)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma))
    v = np.asarray(a, np.float32)
    return (v - v.min()) / max(np.ptp(v), 1e-6)


def element_map(rng, ink):
    field = smooth(rng, N, N * 0.07) ** 2.2
    hit = rng.random((N, N)) < 0.03 + 0.95 * field
    level = np.where(hit, 0.45 + 0.55 * rng.random((N, N)), 0.0)
    rgb = level[..., None] * np.array(ink, np.float32)[None, None] / 255.0
    return Image.fromarray((rgb * 255).astype(np.uint8))


def micrograph(rng):
    grains = 0.6 * smooth(rng, N, N * 0.05) + 0.4 * smooth(rng, N, N * 0.015)
    shade = np.gradient(grains, axis=1)
    g = np.clip(0.30 + 0.55 * grains + 3.0 * shade, 0.08, 0.95)
    return Image.fromarray((np.stack([g, g, g], -1) * 255).astype(np.uint8))


def main() -> None:
    rng = np.random.default_rng(11)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    m = int(S * 0.085)  # transparent margin, as Mac icons have
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((m, m, S - m, S - m), radius=int(S * 0.18), fill=(250, 250, 248, 255),
                        outline=(196, 202, 210, 255), width=int(S * 0.006))
    inner = S - 2 * m
    pad, gap = int(inner * 0.12), int(inner * 0.04)
    t = (inner - 2 * pad - gap) // 2
    tiles = [micrograph(rng), element_map(rng, (240, 70, 175)), element_map(rng, (70, 225, 95)),
             element_map(rng, (70, 165, 255))]
    mask = Image.new("L", (t, t), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, t - 1, t - 1), radius=int(t * 0.06), fill=255)
    for i, tile in enumerate(tiles):
        resample = Image.BICUBIC if i == 0 else Image.NEAREST
        x = m + pad + (i % 2) * (t + gap)
        y = m + pad + (i // 2) * (t + gap)
        img.paste(tile.resize((t, t), resample).convert("RGBA"), (x, y), mask)
    x0, y0 = m + pad, m + pad  # the scale bar on the micrograph
    bw, bh = int(t * 0.46), int(t * 0.085)
    x1, y1 = x0 + t - int(t * 0.08), y0 + t - int(t * 0.1)
    d.rectangle((x1 - bw, y1 - bh, x1, y1), fill=(255, 255, 255, 255), outline=(20, 20, 20, 255), width=int(t * 0.012))
    icon = img.resize((1024, 1024), Image.LANCZOS)
    icon.save(OUT)
    ICONS.mkdir(exist_ok=True)
    icon.save(ICONS / "MicrographMaker.png")
    icon.save(ICONS / "MicrographMaker.ico", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (24, 24), (16, 16)])
    icon.save(ICONS / "MicrographMaker.icns")
    print(OUT, *sorted(ICONS.iterdir()), sep="\n")


if __name__ == "__main__":
    main()

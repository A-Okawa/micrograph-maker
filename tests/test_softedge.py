"""AZtec picture whose last image row is not fully opaque (a soft bottom edge): that row lies just under the
detected image box and is as wide as the image.  It must not be measured as the scale bar (it made a "10 µm"
bar as long as the whole image: a field of 10 µm instead of 61.6 µm)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edsfig import aztec  # noqa: E402


def picture(soft_edge: bool) -> np.ndarray:
    a = np.zeros((160, 220), np.uint8)
    a[10:110, 5:205] = 255            # the image: 200 x 100
    if soft_edge:
        a[110, 5:205] = 200           # one more row, partly transparent
    a[130:132, 20:71] = 255           # the bar: ticks at x = 20 and 70 -> 50 px
    a[132:136, 20] = 255
    a[132:136, 70] = 255
    return a


def test_soft_edge_is_not_the_bar():
    for soft in (False, True):
        a = picture(soft)
        box = aztec.find_image_box(a)
        assert box is not None and box[0] == 5 and box[2] == 204, box
        sb = aztec._find_scalebar(a, box)
        assert sb is not None and abs(sb.length_px - 50) < 1.0, (soft, sb and sb.length_px)
        assert sb.y >= 130, (soft, sb.y)


if __name__ == "__main__":
    test_soft_edge_is_not_the_bar()
    print("ok")

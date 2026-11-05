"""Brightness / contrast: exact definitions, neutral settings leave the pixels untouched.

    python tests/test_adjust.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edsfig import figure as F  # noqa: E402


def grey(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.uint8).reshape(1, -1, 1)
    return np.repeat(v, 3, axis=2)


def test_neutral_is_untouched():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
    assert F.adjusted(img, F.Adjust()) is img


def test_brightness_contrast_gamma_on_grey():
    img = grey([50, 100, 150])  # mean 100
    assert F.adjusted(img, F.Adjust(brightness=20))[0, :, 0].tolist() == [101, 151, 201]  # +0.2*255 = +51
    assert F.adjusted(img, F.Adjust(contrast=100))[0, :, 0].tolist() == [0, 100, 200]  # x2 around the mean
    assert F.adjusted(img, F.Adjust(contrast=-100))[0, :, 0].tolist() == [100, 100, 100]  # flat
    out = F.adjusted(grey([64, 128, 192]), F.Adjust(gamma=2.0))[0, :, 0]
    assert out[1] > 128 and out[0] > 64 and out[2] > 192  # mid-tones brighter
    assert (F.adjusted(img, F.Adjust(brightness=100)) == 255).all()


def test_auto_levels():
    v = np.linspace(40, 200, 1000)
    img = grey(np.round(v))
    out = F.adjusted(img, F.Adjust(auto=True))[0, :, 0]
    assert out.min() == 0 and out.max() == 255  # 0.5-99.5 % stretched to full scale
    assert np.all(np.diff(out.astype(int)) >= 0)  # monotonic: order of grey levels kept


def test_colour_hue_kept():
    rng = np.random.default_rng(1)
    t = rng.integers(0, 120, (40, 40)).astype(float)
    img = np.stack([0 * t, t, t], axis=2).astype(np.uint8)  # cyan map
    img[0, 0] = 0
    for a in (F.Adjust(auto=True), F.Adjust(contrast=60, brightness=10), F.Adjust(gamma=1.5)):
        out = F.adjusted(img, a).astype(int)
        assert (out[..., 0] == 0).all(), a  # red stays zero: still cyan
        assert np.abs(out[..., 1] - out[..., 2]).max() <= 1, a
        assert out[0, 0].sum() == 0  # black stays black


if __name__ == "__main__":
    test_neutral_is_untouched()
    test_brightness_contrast_gamma_on_grey()
    test_auto_levels()
    test_colour_hue_kept()
    print("ok")

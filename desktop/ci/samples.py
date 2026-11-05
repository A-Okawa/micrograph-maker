# -*- coding: utf-8 -*-
"""Sample files for the checks of the PC app, made from the generators of the tests (no real data):
a synthetic AZtec Word report (electron image, Ti and O maps, 5 µm bars of 76 px) and a synthetic SEM picture with a
Hitachi-like band (50 µm over 300 px).

usage: python samples.py OUT_DIR      → writes synthetic.docx and hitachi.tif, prints their paths (os.pathsep between)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'tests'))

import test_sem  # noqa: E402
import test_synthetic  # noqa: E402

EXPECT = {'synthetic.docx': {'labels': ['Ti', 'O'], 'nm_per_px': '65.79 nm'},
          'hitachi.tif': {'nm_per_px': '166.67 nm'}}


def write(out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    sem = np.repeat(rng.integers(30, 200, (200, 266, 1), dtype=np.uint8), 3, axis=2)
    ti = np.zeros((200, 266, 3), np.uint8)
    ti[..., 1] = ti[..., 2] = rng.integers(0, 255, (200, 266), dtype=np.uint8)
    o = np.zeros((200, 266, 3), np.uint8)
    o[..., 1] = rng.integers(0, 255, (200, 266), dtype=np.uint8)
    panels = [test_synthetic._panel('Electron Image 1', sem, 76, '5µm'), test_synthetic._panel('Ti Kα1', ti, 76, '5µm'),
              test_synthetic._panel('O Kα1', o, 76, '5µm')]
    paths = [os.path.join(out, 'synthetic.docx'), os.path.join(out, 'hitachi.tif')]
    with open(paths[0], 'wb') as fh:
        fh.write(test_synthetic._docx(panels))
    with open(paths[1], 'wb') as fh:
        fh.write(test_sem.hitachi_like(300, '50 um'))
    return paths


if __name__ == '__main__':
    print(os.pathsep.join(write(sys.argv[1])))

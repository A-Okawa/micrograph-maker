"""Word reports (EDS sets) and loose SEM images with different conditions in one figure.

    python tests/test_mixed.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from edsfig import aztec, textread  # noqa: E402
from edsfig import figure as F  # noqa: E402
from test_sem import hitachi_like, jeol_like  # noqa: E402
from test_synthetic import _report  # noqa: E402


def _parts(tmp: Path):
    eds, _ = _report(tmp)  # one EDS set: 5 µm over 76 px, all three pictures the same field
    reader = textread.Reader(store_path=tmp / "ex.json", learn=False)
    reader.exemplars = {"element": [], "scale": []}
    sem = aztec.read_images([(hitachi_like(), "a.tif"), (jeol_like(), "b.png")], reader=reader)
    return eds, sem


def test_each_set_and_image_keeps_its_own_scale(tmp: Path = Path(".")):
    eds, sem = _parts(tmp)
    eds_um = [p.um_per_px for p in eds.panels]
    sem_um = [p.um_per_px for p in sem.panels]
    rep = aztec.combine([eds, sem], "all")
    assert rep.kind == "mixed" and len(rep.panels) == 5
    assert [p.set_id for p in rep.panels] == ["synthetic.docx"] * 3 + ["", ""]
    assert [p.um_per_px for p in rep.panels] == eds_um + sem_um  # nothing taken from another file
    assert not rep.warnings  # very different fields (EDS 17 µm, SEM 200 / 320 µm) are not a mismatch
    # a wrong reading inside the EDS set is still caught, and named
    aztec.set_scale_text(rep.panels[1], "50 µm")
    aztec.calibrate(rep)
    assert len(rep.warnings) == 1 and "synthetic.docx" in rep.warnings[0], rep.warnings


def test_two_sets_are_checked_separately(tmp: Path = Path(".")):
    a, _ = _report(tmp)
    b, _ = _report(tmp)
    b.name = "other.docx"
    for p in b.panels:  # the second set at another magnification: 20 µm over the same 76 px
        aztec.set_scale_text(p, "20 µm")
    aztec.calibrate(b)
    rep = aztec.combine([a, b])
    assert not rep.warnings, rep.warnings
    assert abs(rep.panels[3].um_per_px / rep.panels[0].um_per_px - 4.0) < 1e-9


def test_figure_from_mixed_parts(tmp: Path = Path(".")):
    eds, sem = _parts(tmp)
    rep = aztec.combine([eds, sem])
    items = [F.Item(p, p.label) for p in rep.panels if p.include]
    for align in ("width", "height", "crop"):
        st = F.Style(ncols=3, align=align, scalebar_on="all")
        assert F.export(items, st, "pdf")[:5] == b"%PDF-"
        placed, g = F.layout(items, st)
        assert len(g.boxes) == len(items)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        test_each_set_and_image_keeps_its_own_scale(Path(d))
        test_two_sets_are_checked_separately(Path(d))
        test_figure_from_mixed_parts(Path(d))
    print("ok")

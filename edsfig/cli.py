"""Command line: AZtec .docx / SEM / TEM images -> figure files (for batch work or calling from other tools).

    python -m edsfig report.docx [more.docx ...] --cols 3 --width 170 --formats pdf,png
    python -m edsfig a.tif b.tif c.tif --labels "Sample A,Sample B,Sample C" --cols 3
    python -m edsfig tem1.dm4 tem2.dm4 --contrast 20 --cols 2
    python -m edsfig a.tif b.jpg c.tif --align height          # same height, widths follow the shapes
    python -m edsfig a.tif b.jpg c.tif --align crop --crop-aspect 4:3   # trimmed to one shape (centre)
    python -m edsfig t1.dm4 t2.dm4 --rotate "90,-12.5" --flip ",h"   # turn / mirror (scale kept)
    python -m edsfig a.tif b.tif c.tif --order "3,1,2"            # placing order
    python -m edsfig a.tif b.tif c.tif d.tif e.tif --rows 2,3 --row-align center   # 2 on top, 3 below
    python -m edsfig r.docx a.tif b.tif --combine --layout "1 6 (2/3/4); 7 5"     # free arrangement
    python -m edsfig a.tif b.tif --layout "1 2" --trim "1=10,0,90,100"            # trim picture 1 (%)

--layout writes nested rows and columns with the pictures' numbers (their placing order):
a space puts pictures side by side (same height), "/" stacks them (same width), ( ) groups,
";" starts the next row.  Labels and scale bars are sized in points, the same on every panel.

Every .docx becomes its own figure; all image files given together (TIFF/PNG/JPEG/BMP and
DigitalMicrograph .dm3/.dm4) become one figure (a .txt saved by the SEM next to an image, with
the same name, is used for its pixel size).  --combine puts everything into one figure.
Labels and scale values are read automatically; override them when needed:

    python -m edsfig report.docx --labels "O,C,Yb,Si" --scale "100 um"

Pictures are only ever scaled by the same factor in x and y.  --width and --height together
fit the figure inside that box (the free space becomes background).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import aztec, textread
from . import figure as F

IMAGE_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".dm3", ".dm4"}


def _aspect(text: str) -> float | None:
    """'4:3' -> 1.333..., '1.5' -> 1.5, 'auto' -> None."""
    t = (text or "").strip().lower()
    if t in ("", "auto"):
        return None
    try:
        if ":" in t:
            w, h = (float(v) for v in t.split(":", 1))
            return w / h if w > 0 and h > 0 else None
        v = float(t)
        return v if v > 0 else None
    except ValueError:
        raise SystemExit(f"--crop-aspect: '{text}' is not a width:height ratio (e.g. 4:3)")


def _rows_arg(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    try:
        vals = tuple(int(v) for v in text.replace(" ", "").split(",") if v)
    except ValueError:
        raise SystemExit(f"--rows: '{text}' is not a list of numbers (e.g. 2,3)")
    if not vals or min(vals) < 1:
        raise SystemExit(f"--rows: '{text}' needs numbers of 1 or more")
    return vals


def _per_picture(text: str | None, n: int) -> list[str]:
    """'90' -> the same for all n pictures; '90,,-12' -> one per picture in order (empty = none)."""
    if not text:
        return [""] * n
    vals = [v.strip() for v in text.split(",")]
    return vals * n if len(vals) == 1 else (vals + [""] * n)[:n]


def _trims(text: str | None) -> dict[int, tuple[float, float, float, float]]:
    """'2=10,0,90,100; 5=0,0,50,50' -> {2: (10, 0, 90, 100), 5: (0, 0, 50, 50)} (percent of the picture)."""
    out = {}
    for part in [p for p in (text or "").split(";") if p.strip()]:
        try:
            n, box = part.split("=", 1)
            l, t, r, b = (float(v) for v in box.split(","))
        except ValueError:
            raise SystemExit(f"--trim: '{part.strip()}' should be N=left,top,right,bottom (percent)")
        if not (0 <= l < r <= 100 and 0 <= t < b <= 100):
            raise SystemExit(f"--trim: '{part.strip()}' needs 0 <= left < right <= 100 and 0 <= top < bottom <= 100")
        out[int(n)] = (l, t, r, b)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="edsfig", description="AZtec EDS report (.docx) / SEM / TEM images -> publication figure")
    ap.add_argument("inputs", nargs="+", help=".docx (AZtec report) and/or image files")
    ap.add_argument("-o", "--out", default=".", help="output folder")
    ap.add_argument("--formats", default="pdf,png", help="comma list of png,tiff,pdf,svg,zip")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--rows", help="pictures per row from the top, e.g. '2,3' (further rows: --cols each)")
    ap.add_argument("--row-align", choices=["auto", "left", "center", "right"], default="auto",
                    help="a row with fewer pictures (auto: left; centre when --align height)")
    ap.add_argument("--auto-cols", action="store_true", help="with --width and --height: columns that make pictures largest")
    ap.add_argument("--width", type=float, default=None, help="figure width in mm (default 170)")
    ap.add_argument("--height", type=float, default=None, help="figure height in mm")
    ap.add_argument("--gap", type=float, default=1.0, help="gap between panels in mm")
    ap.add_argument("--align", choices=["width", "height", "crop"], default="width",
                    help="pictures of different shapes: same width | same height | trimmed to one shape")
    ap.add_argument("--valign", choices=["top", "center", "bottom"], default="center",
                    help="--align width: where a shorter picture sits in its cell")
    ap.add_argument("--crop-aspect", default="auto",
                    help="--align crop: width:height to trim to, e.g. 4:3, 1:1, 1.5 (auto = the shape that trims least)")
    ap.add_argument("--big-electron", action="store_true", help="electron image as a 2x2 block")
    ap.add_argument("--labels", help="comma list of labels in order: element symbols (maps) or sample names (images)")
    ap.add_argument("--order", help="placing order of the pictures as listed, e.g. '3,1,2' (left to right, then the "
                                    "next row; pictures not named follow)")
    ap.add_argument("--scale", help="scale value for every bar, e.g. '5 um' (overrides reading)")
    ap.add_argument("--bar", type=float, default=0.0, help="new scale bar length in µm (0 = automatic)")
    ap.add_argument("--bar-on", choices=["electron", "all", "none"], default="electron")
    ap.add_argument("--bar-color", default="white",
                    help="scale bar and its text: white | black | auto (black on light pictures) | #rrggbb")
    ap.add_argument("--bar-colors", help="per placed picture, in order, e.g. 'white,black,,auto' (empty = --bar-color)")
    ap.add_argument("--label-pt", type=float, default=10.0)
    ap.add_argument("--label-color", default="white", help="white | black | map | #rrggbb")
    ap.add_argument("--label-corner", default="top-left", choices=["top-left", "top-right", "bottom-left", "bottom-right"])
    ap.add_argument("--label-outline", action="store_true", help="outline around the label text (off by default)")
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--dpi", type=float, default=0.0)
    ap.add_argument("--auto-bright", action="store_true", help="stretch each map to full brightness")
    ap.add_argument("--auto-levels", action="store_true", help="automatic levels for every picture (grey: 0.5-99.5 %%)")
    ap.add_argument("--brightness", type=float, default=0.0, help="every picture: -100..100 (%%)")
    ap.add_argument("--contrast", type=float, default=0.0, help="every picture: -100..200 (%%, around the mean)")
    ap.add_argument("--gamma", type=float, default=1.0, help="every picture: mid-tones (>1 brighter)")
    ap.add_argument("--keep-band", action="store_true", help="images: keep the info band / overlaid text rows")
    ap.add_argument("--combine", action="store_true",
                    help="everything in one figure: each .docx stays one EDS set, images stay independent")
    ap.add_argument("--rotate", help="clockwise degrees: one value for every placed picture, or a comma list in order "
                                     "(e.g. '90,,-12.5'); quarter turns are exact, other angles are interpolated")
    ap.add_argument("--flip", help="mirror: one value for all, or a comma list in order: h (left-right), v (up-down), hv")
    ap.add_argument("--rotate-fill", choices=["crop", "white", "black"], default="crop",
                    help="odd angles: cut the empty corners away (crop) or fill them")
    ap.add_argument("--rotate-resample", choices=["bicubic", "nearest"], default="bicubic")
    ap.add_argument("--no-learn", action="store_true", help="do not store labels as exemplars")
    ap.add_argument("--layout", help="free arrangement with the pictures' numbers (placing order), e.g. "
                    "'1 2 (3/4/5); 6 7 8': space = side by side (same height), / = stacked (same width), "
                    "( ) = group, ; = next row")
    ap.add_argument("--trim", help="trim pictures by number, in %% of the picture: 'N=left,top,right,bottom', "
                    "several separated by ';', e.g. '2=10,0,90,100; 5=0,0,50,50'")
    ap.add_argument("--bar-thick", type=float, default=2.0, help="scale bar thickness in points")
    a = ap.parse_args(argv)

    if a.width and a.height:
        mode = "box"
    elif a.height:
        mode = "height"
    else:
        mode = "width"
    reader = textread.Reader(learn=not a.no_learn)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = F.Style(ncols=a.cols, size_mode=mode, width_mm=a.width or 170.0, height_mm=a.height or 120.0,
                 auto_cols=a.auto_cols, gap_mm=a.gap, electron_span=2 if a.big_electron else 1,
                 align=a.align, valign=a.valign, crop_aspect=_aspect(a.crop_aspect),
                 row_counts=_rows_arg(a.rows), row_align=a.row_align,
                 scalebar_um=a.bar or None, scalebar_on=a.bar_on, scalebar_color=a.bar_color,
                 label_pt=a.label_pt, label_color=a.label_color, label_outline=a.label_outline,
                 label_corner=a.label_corner, oversample=a.oversample, dpi=a.dpi or None,
                 scalebar_thick_pt=a.bar_thick)
    if a.layout:
        from . import tree

        try:
            tree.parse(a.layout)
        except tree.SpecError as e:
            raise SystemExit(f"--layout: {e}")
        st.align, st.spec = "free", a.layout

    paths = [Path(p) for p in a.inputs]
    jobs = []  # (report, output stem)
    for p in paths:
        if p.suffix.lower() == ".docx":
            jobs.append((aztec.read_report(p, reader=reader), p.stem))
    images = [p for p in paths if p.suffix.lower() in IMAGE_EXT]
    if images:
        txts = {}
        for p in images:  # the .txt an SEM saves next to the image
            t = p.with_suffix(".txt")
            if t.exists():
                txts[p.stem] = t.read_text(encoding="utf-8", errors="ignore")
        rep = aztec.read_images([(p.read_bytes(), p.name) for p in images], reader=reader, txts=txts,
                                name=f"{len(images)} images")
        n_dm = sum(1 for p in images if p.suffix.lower() in (".dm3", ".dm4"))
        kind = "TEM" if n_dm == len(images) else "SEM_TEM" if n_dm else "SEM"
        jobs.append((rep, images[0].stem if len(images) == 1 else kind))

    if a.combine and len(jobs) > 1:
        jobs = [(aztec.combine([r for r, _ in jobs], "combined"), "combined")]
    status = 0
    for rep, stem in jobs:
        targets = [p for p in rep.panels if p.kind == "map"] if rep.kind == "aztec" else \
            [p for p in rep.panels if p.include]
        if a.labels:
            for p, lab in zip(targets, [s.strip() for s in a.labels.split(",")]):
                p.label = lab
        if a.keep_band and rep.kind == "images":
            for p in rep.panels:
                aztec.set_band(p, 0)
        if a.rotate or a.flip:  # after reading the scale: turning keeps the pixel pitch
            placed_panels = [p for p in rep.panels if p.include]
            angles = _per_picture(a.rotate, len(placed_panels))
            flips = _per_picture(a.flip, len(placed_panels))
            for p, ang, fl in zip(placed_panels, angles, flips):
                fl = (fl or "").lower()
                aztec.set_rotation(p, float(ang) if ang else 0.0, "h" in fl, "v" in fl, a.rotate_fill, a.rotate_resample)
        if a.scale:
            for p in rep.panels:
                aztec.set_scale_text(p, a.scale)
        aztec.calibrate(rep)

        def adjust(p):
            return F.Adjust(auto=a.auto_levels or (a.auto_bright and p.kind != "electron"),
                            brightness=a.brightness, contrast=a.contrast, gamma=a.gamma)

        items = [F.Item(p, p.label, adjust(p)) for p in rep.panels if p.include]
        if a.order:
            from . import order as ordering

            places = ordering.from_list(a.order, len(items))
            items = [items[k] for k in sorted(range(len(items)), key=lambda k: places[k])]
        if a.bar_colors:
            for it, c in zip(items, [c.strip() for c in a.bar_colors.split(",")]):
                it.bar_color = c or None
        for n, box in _trims(a.trim).items():
            if not 1 <= n <= len(items):
                raise SystemExit(f"--trim: there is no picture {n} (1-{len(items)})")
            W, H = items[n - 1].size
            l, t, r, b = box
            x0, y0 = min(W - 1, round(W * l / 100)), min(H - 1, round(H * t / 100))
            items[n - 1].crop = (x0, y0, max(x0 + 1, min(W, round(W * r / 100))), max(y0 + 1, min(H, round(H * b / 100))))
        print(f"{rep.name}:")
        for p in rep.panels:
            sb = p.scalebar
            flag = ""
            if p.kind == "map" and not (p.label_score >= 0.8 and p.label_margin >= 0.03):
                flag = "  <- please check the label"
            elif sb is not None and sb.text and not (sb.score >= 0.8 and sb.margin >= 0.03) and not a.scale:
                flag = "  <- please check the scale"
            field = f"{p.field_um[0]:.4g} µm" if p.field_um else "-"
            band = ""
            if p.band_px:
                band = f"  {'overlaid text rows' if getattr(p.band, 'overlay', False) else 'band'} {p.band_px}px removed"
            if p.src.note:
                band += f"  ({p.src.note})"
            print(f"  {p.src.name:14s} {p.kind:9s} {p.size[0]}x{p.size[1]}px  label={p.label or '-':10s}"
                  f"  scale={(sb.text if sb else '-'):8s} ({p.scale_source or 'none'})  field width={field}{band}{flag}")
        for w in rep.warnings:
            print("  WARNING:", w)
            status = 1
        if not items:
            print("  nothing to place")
            continue
        for fmt in [f.strip().lower() for f in a.formats.split(",") if f.strip()]:
            if fmt == "zip":
                data, fn = F.panels_zip(items, st, rep.name), f"{stem}_panels.zip"
            else:
                data, fn = F.export(items, st, fmt), f"{stem}_figure.{'tif' if fmt == 'tiff' else fmt}"
            (out / fn).write_bytes(data)
            print("  ->", out / fn)
        info = F.info(items, st, rep.name)
        how = "free arrangement" if st.align == "free" else f"{info['columns']} columns"
        print(f"  size {info['output_mm'][0]} x {info['output_mm'][1]} mm, {info['dpi']} dpi, {how}")
        (out / f"{stem}_calibration.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        if not a.no_learn and (a.labels or a.scale):  # only what the user stated explicitly
            if a.labels and rep.kind == "aztec":
                for p in targets:
                    if p.label and p.title_mask is not None:
                        reader.remember("element", p.title_mask, p.label, save=False)
            if a.scale:
                for p in rep.panels:
                    if p.scalebar is not None and p.scalebar.label_mask is not None:
                        reader.remember("scale", p.scalebar.label_mask, a.scale, save=False)
            reader.save()
    return status


if __name__ == "__main__":
    raise SystemExit(main())

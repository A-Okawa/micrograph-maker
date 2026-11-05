"""Read AZtec (Oxford Instruments) report exports and split each picture into its parts.

An AZtec Word report embeds every electron image / element map as its own PNG with a
transparent background.  Inside each PNG the layout is always the same:

    title text            ("Ti Kα1", "Electron Image 4", "電子線像 8")
    opaque image body     (the actual data, a solid rectangle)
    scale bar + label     ("|____|  5µm")

Because the background is transparent, the parts can be separated by geometry alone
(no image recognition): the image body is the large fully-opaque rectangle, the title is
the ink above it and the scale bar is the ink below it.  Only the two short texts (element
symbol and scale value) have to be read; that is done in ``textread``.

Nothing here resamples or re-encodes pixels: the image body is an exact crop of the
embedded PNG.
"""
from __future__ import annotations

import hashlib
import io
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from . import textread

RASTER_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


# --------------------------------------------------------------------------- data classes
@dataclass
class SourceImage:
    """One picture exactly as it was embedded (or as the file on disk)."""

    name: str
    rgba: np.ndarray  # H x W x 4, uint8
    origin: str = ""
    order: int = 0
    meta_um_per_px: float | None = None  # pixel size recorded by the instrument (TIFF tags / .txt / DM)
    note: str = ""  # e.g. why a DigitalMicrograph picture has no real-space scale
    reciprocal: bool = False  # calibrated in 1/length (diffraction pattern, FFT): no scale bar
    uid: str = ""  # the same picture loaded again (alone or with other files) gets the same uid


@dataclass
class ScaleBar:
    length_px: float  # tick-centre to tick-centre, in source pixels
    y: int
    x_left: float
    x_right: float
    label_mask: np.ndarray | None = None  # ink (0..1) of the label text
    text: str = ""  # e.g. "5µm" (best guess, may be corrected by the user)
    value: float | None = None
    unit: str = "µm"
    score: float = 0.0
    margin: float = 0.0

    @property
    def length_um(self) -> float | None:
        if self.value is None:
            return None
        return self.value * textread.UNIT_TO_UM.get(self.unit, 1.0)


@dataclass
class Panel:
    src: SourceImage
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive) of the image body
    kind: str  # "electron" | "map" | "layered" | "spectrum" | "other"
    title_mask: np.ndarray | None = None
    label: str = ""  # text to print at the top-left (element symbol for maps)
    label_score: float = 0.0
    label_margin: float = 0.0
    label_candidates: list[tuple[str, float]] = field(default_factory=list)
    scalebar: ScaleBar | None = None
    color: tuple[int, int, int] | None = None  # dominant colour of a map
    um_per_px: float | None = None  # pixel size of the cropped image body
    include: bool = True
    band: object | None = None  # sem.Band: the instrument's info band that was removed
    band_px: int = 0  # rows removed with it
    crop_lr: tuple[int, int] = (0, 0)  # columns removed at the left / right (a map's colour bar)
    band_preview: np.ndarray | None = None  # grey picture of the band around bar and label
    scale_source: str = ""  # "bar" | "metadata" | "manual" | ""
    # turning / mirroring of the image body (see transform.py); the pixel pitch is unchanged
    rotation: float = 0.0  # degrees, clockwise
    flip_h: bool = False
    flip_v: bool = False
    rotate_fill: str = "crop"  # crop | white | black: the empty corners of a picture turned at an odd angle
    rotate_resample: str = "bicubic"  # bicubic | nearest
    set_id: str = ""  # in a combined figure: the EDS set (Word file) the picture belongs to; "" = a loose image
    # layered image ("EDS Layered Image"): the colour key AZtec draws into a corner (legend.Legend; its rows
    # are cut off like a band) and each coloured layer's element symbol and colour
    legend: object | None = None
    layers: list = field(default_factory=list)  # [(symbol, (r, g, b)), ...]
    full_box: tuple[int, int, int, int] | None = None  # the body before rows were cut off it (AZtec pictures)

    @property
    def body(self) -> np.ndarray:
        """Exact crop of the image body (RGB uint8) as it is in the file; no resampling."""
        x0, y0, x1, y1 = self.box
        return self.src.rgba[y0 : y1 + 1, x0 : x1 + 1, :3]

    @property
    def body_size(self) -> tuple[int, int]:
        """Width and height of the image body as it is in the file (before turning)."""
        x0, y0, x1, y1 = self.box
        return x1 - x0 + 1, y1 - y0 + 1

    @property
    def turned(self) -> bool:
        from . import transform

        return not transform.is_identity(self.rotation, self.flip_h, self.flip_v)

    @property
    def image(self) -> np.ndarray:
        """The image body as placed: turned / mirrored when asked (quarter turns and mirroring keep
        every pixel; other angles are interpolated), otherwise the exact crop."""
        if not self.turned:
            return self.body
        from . import transform

        return transform.rotate(self.body, self.rotation, self.flip_h, self.flip_v, self.rotate_fill,
                                self.rotate_resample)

    @property
    def size(self) -> tuple[int, int]:
        if not self.turned:
            return self.body_size
        h, w = self.image.shape[:2]
        return w, h

    @property
    def field_um(self) -> tuple[float, float] | None:
        if self.um_per_px is None:
            return None
        w, h = self.size
        return w * self.um_per_px, h * self.um_per_px

    def title_preview(self, pad: int = 2) -> np.ndarray | None:
        """Grey image of the original title (for showing next to the guess)."""
        return _ink_preview(self.title_mask, pad)

    def scale_label_preview(self, pad: int = 2) -> np.ndarray | None:
        if self.scalebar is None:
            return None
        return _ink_preview(self.scalebar.label_mask, pad)

    def legend_preview(self, pad: int = 3) -> np.ndarray | None:
        """The colour key of a layered image as it is drawn in the file (RGB)."""
        if self.legend is None or not self.legend.chips:
            return None
        bx0, by0, bx1, by1 = self.full_box or self.box
        cs = self.legend.chips
        x0, x1 = max(0, min(c.x0 for c in cs) - pad), min(bx1 - bx0, max(c.x1 for c in cs) + pad)
        y0, y1 = max(0, min(c.y0 for c in cs) - pad), min(by1 - by0, max(c.y1 for c in cs) + pad)
        return self.src.rgba[by0 + y0: by0 + y1 + 1, bx0 + x0: bx0 + x1 + 1, :3]


@dataclass
class Report:
    name: str
    panels: list[Panel]
    warnings: list[str] = field(default_factory=list)
    kind: str = "aztec"  # "aztec": one field of view (EDS set) | "images": independent pictures (SEM)
    #                      | "mixed": EDS sets (Panel.set_id) and loose images in one figure

    @property
    def electron(self) -> Panel | None:
        for p in self.panels:
            if p.kind == "electron":
                return p
        return None


# --------------------------------------------------------------------------- loading
def load_sources(path_or_bytes, name: str | None = None) -> list[SourceImage]:
    """Load pictures from a .docx (document order) or from a single image file."""
    if isinstance(path_or_bytes, (str, Path)):
        p = Path(path_or_bytes)
        data = p.read_bytes()
        name = name or p.name
    else:
        data = bytes(path_or_bytes)
        name = name or "upload"
    if data[:2] == b"PK":
        return _sources_from_docx(data, name)
    from . import dm, sem

    uid = hashlib.md5(data).hexdigest()[:12]

    if dm.is_dm(data):  # Gatan DigitalMicrograph (.dm3 / .dm4)
        im = dm.read_dm(data)
        img8 = dm.to_8bit(im.data)
        rgb = img8 if img8.ndim == 3 else np.repeat(img8[..., None], 3, axis=2)
        rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=2)
        note = ""
        u = im.units[0].strip()
        reciprocal = u.startswith("1/") or u.endswith("-1") or u.endswith("⁻¹")
        if im.um_per_px is None:
            note = (f"単位が {u} のため（回折像など）実空間のスケールバーは付けません" if u else "較正情報がありません")
        elif abs(im.scale[0] - im.scale[1]) > 1e-6 * max(im.scale):
            note = f"画素が正方形ではありません（x {im.scale[0]:g}, y {im.scale[1]:g} {im.units[0]}）。画素はそのまま並べ、スケールは x を使います"
        if im.frames > 1:
            note = (note + "。" if note else "") + f"{im.frames} 枚の重ね（スタック）のうち1枚目を使います"
        return [SourceImage(name=name, rgba=rgba, origin=name, order=0, meta_um_per_px=im.um_per_px, note=note,
                            reciprocal=reciprocal, uid=uid)]

    im = Image.open(io.BytesIO(data))
    meta = None
    try:
        meta = sem.pixel_size_from_tags(dict(getattr(im, "tag_v2", {}) or {}))
    except Exception:
        meta = None
    return [SourceImage(name=name, rgba=_to_rgba(im), origin=name, order=0, meta_um_per_px=meta, uid=uid)]


def _to_rgba(im: Image.Image) -> np.ndarray:
    if im.mode in ("I;16", "I;16B", "I", "F"):
        a = np.asarray(im, dtype=np.float64)
        a = (255 * (a - a.min()) / max(1e-9, a.max() - a.min())).astype(np.uint8)
        im = Image.fromarray(a)
    return np.asarray(im.convert("RGBA")).copy()


_NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _sources_from_docx(data: bytes, name: str) -> list[SourceImage]:
    zf = zipfile.ZipFile(io.BytesIO(data))
    doc_id = hashlib.md5(data).hexdigest()[:10]
    names = set(zf.namelist())
    doc = zf.read("word/document.xml").decode("utf-8-sig", errors="replace")
    rels_xml = zf.read("word/_rels/document.xml.rels").decode("utf-8-sig", errors="replace")
    rels: dict[str, str] = {}
    for m in re.finditer(r"<Relationship\b([^>]*)/?>", rels_xml):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        if "Id" in attrs and "Target" in attrs and attrs.get("TargetMode") != "External":
            rels[attrs["Id"]] = attrs["Target"]
    # picture references in document order: DrawingML blips and legacy VML imagedata
    refs = re.findall(r'<a:blip\b[^>]*?r:embed="([^"]+)"|<v:imagedata\b[^>]*?r:id="([^"]+)"', doc)
    out: list[SourceImage] = []
    seen: set[str] = set()
    for a, b in refs:
        rid = a or b
        target = rels.get(rid)
        if not target:
            continue
        if target.startswith("/"):
            part = target.lstrip("/")
        else:
            part = posixpath.normpath(posixpath.join("word", target))
        if part not in names or part in seen:
            continue
        if Path(part).suffix.lower() not in RASTER_EXT:
            continue  # EMF/WMF etc. are not supported
        seen.add(part)
        try:
            im = Image.open(io.BytesIO(zf.read(part)))
            rgba = _to_rgba(im)
        except Exception:
            continue
        out.append(SourceImage(name=Path(part).name, rgba=rgba, origin=name, order=len(out), uid=f"{doc_id}/{part}"))
    return out


# --------------------------------------------------------------------------- geometry
def _longest_runs(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape[0], dtype=int)
    for y, row in enumerate(mask):
        if not row.any():
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        s = np.flatnonzero(d == 1)
        e = np.flatnonzero(d == -1)
        out[y] = int((e - s).max())
    return out


def _longest_true_segment(flags: np.ndarray) -> tuple[int, int] | None:
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return None
    blocks = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    b = max(blocks, key=len)
    return int(b[0]), int(b[-1])


def find_image_box(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest fully-opaque rectangle (the image body) in an RGBA picture."""
    H, W = alpha.shape
    if alpha.min() >= 250:  # no transparency at all -> whole picture
        return None
    op = alpha >= 250
    runs = _longest_runs(op)
    if runs.max() < 16:
        return None
    rows = _longest_true_segment(runs >= 0.9 * runs.max())
    if rows is None:
        return None
    y0, y1 = rows
    if y1 - y0 + 1 < 16:
        return None
    cols = _longest_true_segment(op[y0 : y1 + 1].mean(0) >= 0.98)
    if cols is None:
        return None
    x0, x1 = cols
    if x1 - x0 + 1 < 16:
        return None
    return x0, y0, x1, y1


def _find_scalebar(alpha: np.ndarray, box) -> ScaleBar | None:
    x0, y0, x1, y1 = box
    below = alpha[y1 + 1 :].astype(np.float64) / 255.0
    if below.size == 0 or below.max() < 0.3:
        return None
    strong = below > 0.6
    runs = _longest_runs(strong)
    if runs.max() < 12:
        return None
    r = int(np.argmax(runs))  # first row of the horizontal stroke
    row = strong[r]
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1) - 1
    k = int(np.argmax(e - s))
    xa, xb = int(s[k]), int(e[k])
    # stroke thickness
    rr = r
    while rr + 1 < below.shape[0] and (below[rr + 1, xa + 3 : xb - 2] > 0.6).mean() > 0.8:
        rr += 1
    # end ticks: ink just below the stroke near both ends -> alpha-weighted centre
    t0, t1 = rr + 2, min(below.shape[0], rr + 6)

    def tick_centre(xc: int) -> float | None:
        lo, hi = max(0, xc - 3), min(below.shape[1], xc + 4)
        win = below[t0:t1, lo:hi]
        if win.size == 0 or win.sum() < 0.8:
            return None
        w = win.sum(0)
        return float((np.arange(lo, hi) * w).sum() / w.sum())

    cl, cr = tick_centre(xa), tick_centre(xb)
    if cl is None or cr is None:
        # no ticks: use the stroke ends (sub-pixel from partial coverage)
        full = below[r : rr + 1].max(0)
        cl = xa - full[xa - 1] if xa > 0 else float(xa)
        cr = xb + (full[xb + 1] if xb + 1 < full.size else 0.0)
    # label: the ink under the stroke, between the ticks
    lab = below[rr + 2 :, int(np.ceil(cl)) + 3 : int(np.floor(cr)) - 2].copy()
    lab_mask = textread.crop_ink(lab) if lab.size and lab.max() > 0.3 else None
    return ScaleBar(
        length_px=float(cr - cl),
        y=y1 + 1 + r,
        x_left=float(cl),
        x_right=float(cr),
        label_mask=lab_mask,
    )


def _classify(rgb: np.ndarray, has_bar: bool, require_bar: bool = True) -> tuple[str, tuple[int, int, int] | None]:
    """electron | map | layered | spectrum | other.

    In an AZtec report every electron image and map carries a scale bar, so pictures without
    one (spectra, logos) are set aside.  Single image files have no bar: require_bar=False."""
    h, w = rgb.shape[:2]
    if not has_bar and w / max(h, 1) >= 1.8:  # wide picture without a scale bar: the sum spectrum
        return "spectrum", None
    x = rgb.astype(np.int16)
    chroma = x.max(2) - x.min(2)
    if np.percentile(chroma, 99.5) <= 12:
        return ("electron" if (has_bar or not require_bar) else "other"), None
    if require_bar and not has_bar:
        return "other", None
    sat = chroma > 40
    if sat.sum() < 20:
        return "map", None
    px = x[sat].astype(np.float64)
    mx = px.max(1)
    mn = px.min(1)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    d = np.maximum(mx - mn, 1e-9)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    hist, _ = np.histogram(h, bins=36, range=(0, 360))
    k = int(np.argmax(hist))
    near = sum(hist[(k + j) % 36] for j in (-2, -1, 0, 1, 2))
    if near / hist.sum() < 0.75:
        return "layered", None
    # dominant colour: brightest saturated pixels, normalised to full brightness
    lum = px.sum(1)
    top = px[lum >= np.percentile(lum, 95)].mean(0)
    col = tuple(int(round(c)) for c in 255 * top / max(1e-9, top.max()))
    return "map", col


def attach_legend(p: Panel) -> None:
    """A colour key drawn into the picture makes it a layered image; the key's rows are cut off
    (like an instrument band: the table's "帯 (px)" sets how many, 0 keeps them)."""
    from . import legend as L, sem

    x0, y0, x1, y1 = p.box
    lg = L.find_legend(p.src.rgba[y0: y1 + 1, x0: x1 + 1, :3])
    if lg is None:
        return
    p.kind, p.color, p.legend = "layered", None, lg
    p.full_box = p.box
    n = lg.rows
    p.band = sem.Band(lg.side, (y1 + 1 - n) if lg.side == "bottom" else y0, n, float("nan"), overlay=True, legend=True)
    p.band_px = n
    p.box = (x0, y0, x1, y1 - n) if lg.side == "bottom" else (x0, y0 + n, x1, y1)


def name_layers(p: Panel, reader: textread.Reader, maps: list[Panel] | None = None) -> None:
    """Element symbol and colour of every coloured box of a layered image's key.  The symbols are
    read from the boxes; the maps of the same report help: a box has the colour of its element's map."""
    from . import legend as L

    if p.legend is None:
        return
    L.read_legend(p.legend, reader)
    pool = [m for m in (maps or []) if m.kind == "map" and m.color is not None and m.label]
    for c in p.legend.elements:
        near = sorted(pool, key=lambda m: max(abs(a - b) for a, b in zip(c.color, m.color)))
        if not near:
            continue
        d1 = max(abs(a - b) for a, b in zip(c.color, near[0].color))
        d2 = max(abs(a - b) for a, b in zip(c.color, near[1].color)) if len(near) > 1 else 999
        m = near[0]
        if d1 <= 60 and d2 - d1 >= 40 and not (c.score >= 0.8 and c.margin >= 0.03 and c.text != m.label):
            c.text = m.label
            c.score, c.margin = max(c.score, min(0.95, m.label_score)), max(c.margin, m.label_margin)
    named = [c for c in p.legend.elements if c.text]
    p.layers = [(c.text, c.color) for c in named]
    p.label = " ".join(c.text for c in named)
    p.label_score = min((c.score for c in p.legend.elements), default=0.0)
    p.label_margin = min((c.margin for c in p.legend.elements), default=0.0)


def analyse(src: SourceImage) -> Panel:
    a = src.rgba
    alpha = a[..., 3]
    box = find_image_box(alpha)
    if box is None:  # no transparency: a plain image file, use it whole
        H, W = alpha.shape
        kind, col = _classify(a[..., :3], has_bar=True, require_bar=False)
        p = Panel(src=src, box=(0, 0, W - 1, H - 1), kind=kind, color=col, full_box=(0, 0, W - 1, H - 1))
        if kind in ("map", "layered"):
            attach_legend(p)
        return p
    x0, y0, x1, y1 = box
    title = alpha[:y0].astype(np.float64) / 255.0
    title_mask = textread.crop_ink(title) if title.size and title.max() > 0.3 else None
    bar = _find_scalebar(alpha, box)
    rgb = a[y0 : y1 + 1, x0 : x1 + 1, :3]
    kind, col = _classify(rgb, has_bar=bar is not None)
    p = Panel(src=src, box=box, kind=kind, title_mask=title_mask, scalebar=bar, color=col, full_box=box)
    if kind in ("map", "layered"):
        attach_legend(p)
    return p


def _read_titles(p: Panel, reader: textread.Reader) -> None:
    """Map titles -> element symbol; a title starting "EDS" ("EDS Layered Image 6") is a layered image."""
    if p.kind == "map" and p.title_mask is not None and p.legend is None:
        if reader.first_word_is(p.title_mask, "EDS"):
            p.kind, p.color, p.label = "layered", None, ""
        else:
            res = reader.read_element(p.title_mask)
            p.label, p.label_score, p.label_margin, p.label_candidates = res.text, res.score, res.margin, res.candidates
    elif p.kind == "layered" and p.legend is None:
        p.label = ""


# --------------------------------------------------------------------------- whole report
def read_report(path_or_bytes, name: str | None = None, reader: textread.Reader | None = None) -> Report:
    srcs = load_sources(path_or_bytes, name)
    reader = reader or textread.Reader()
    panels = [analyse(s) for s in srcs]
    rep = Report(name=name or (Path(path_or_bytes).name if isinstance(path_or_bytes, (str, Path)) else "upload"), panels=panels)
    for p in panels:
        _read_titles(p, reader)
        if p.scalebar is not None and p.scalebar.label_mask is not None:
            res = reader.read_scale(p.scalebar.label_mask)
            sb = p.scalebar
            sb.text, sb.score, sb.margin = res.text, res.score, res.margin
            sb.value, sb.unit = textread.parse_length(res.text)
        p.include = p.kind in ("electron", "map", "layered")
    for p in panels:  # after the maps: their colours help to name the layers
        name_layers(p, reader, panels)
    calibrate(rep)
    return rep


def analyse_image(src: SourceImage, reader: textread.Reader, txt: str | None = None) -> Panel:
    """A loose image file: an AZtec export (transparent margins) or a plain SEM micrograph."""
    if src.rgba[..., 3].min() < 250 and find_image_box(src.rgba[..., 3]) is not None:
        p = analyse(src)  # AZtec-style picture with transparent margins
        _read_titles(p, reader)
        name_layers(p, reader)
        if p.scalebar is not None and p.scalebar.label_mask is not None:
            res = reader.read_scale(p.scalebar.label_mask)
            sb = p.scalebar
            sb.text, sb.score, sb.margin = res.text, res.score, res.margin
            sb.value, sb.unit = textread.parse_length(res.text)
        p.include = p.kind in ("electron", "map", "layered")
        return p
    from . import sem

    g = sem.gray(src.rgba)
    band = sem.find_band(g, src.rgba)
    overlay_bar = None
    if band is None:  # text and scale bar drawn onto the picture itself (JEOL FE-SEM)
        ov = sem.find_overlay(g)
        if ov is not None:
            band, overlay_bar = ov
    x0, y0, x1, y1 = sem.image_box(g.shape, band)
    cb = sem.find_colorbar(src.rgba[y0: y1 + 1, x0: x1 + 1, :3])  # intensity bar of a TEM-EDS map
    crop = ((cb[1], 0) if cb[0] == "left" else (0, cb[1])) if cb else (0, 0)
    box = (x0 + crop[0], y0, x1 - crop[1], y1)
    kind, col = _classify(src.rgba[y0: y1 + 1, box[0]: box[2] + 1, :3], has_bar=True, require_bar=False)
    p = Panel(src=src, box=box, kind=kind, color=col, band=band, band_px=band.height if band else 0, crop_lr=crop)
    if kind in ("map", "layered") and band is None:  # an exported layered image with its colour key
        attach_legend(p)
        name_layers(p, reader)
    if txt:
        try:
            src.meta_um_per_px = src.meta_um_per_px or sem.pixel_size_from_txt(txt)
        except Exception:
            pass
    if band is not None:
        up, down = sem.band_inks(g, band)
        bar = overlay_bar or sem.find_bar(up, down, g.shape[1])
        ink = (up if bar.bright else down) if bar is not None else (up if up.sum() > down.sum() else down)
        res, mask, bbox = sem.read_band_scale(ink, bar, g.shape[1], reader)
        if bar is not None:
            sb = ScaleBar(length_px=bar.length_px, y=band.start + bar.y0, x_left=float(bar.x0),
                          x_right=float(bar.x1), label_mask=mask)
            if res is not None:
                sb.text, sb.score, sb.margin = res.text, res.score, res.margin
                sb.value, sb.unit = textread.parse_length(res.text)
            p.scalebar = sb
        if kind == "map":  # TEM-EDS: the element is printed in the band ("C K")
            er, em = sem.read_band_element(ink, bar, bbox, reader)
            if er is not None and er.text:
                p.label, p.label_score, p.label_margin, p.label_candidates = er.text, er.score, er.margin, er.candidates
                p.title_mask = em
        # picture of the band around the bar and its label, for checking by eye
        xs = [v for v in ((bar.x0, bar.x1) if bar else ()) + ((bbox[0], bbox[2]) if bbox else ())]
        lo, hi = (max(0, min(xs) - 12), min(g.shape[1], max(xs) + 13)) if xs else (0, g.shape[1])
        p.band_preview = np.clip(255.0 - ink[:, lo:hi] * 255.0, 0, 255).astype(np.uint8)  # dark on white
    p.include = kind in ("electron", "map", "layered")
    return p


def sheet_panels(src: SourceImage, reader: textread.Reader) -> list[Panel] | None:
    """A JEOL EPMA map window saved as one picture: one picture per tile (see sheet.py), the maps
    cropped exactly, named from the box under each map, all with the sheet's scale.  None for any
    other picture."""
    from . import sheet as SH

    if src.rgba[..., 3].min() < 250:  # (AZtec exports have transparent margins)
        return None
    tiles = SH.sheet_tiles(src.rgba[..., :3])
    if not tiles:
        return None
    # one stage scan, one scale: the bar as measured on every tile, the length as read most often
    bar_px = float(np.median([t.bar_px for t in tiles]))
    reads = []
    for t in tiles[:6]:  # the same text on every tile
        if t.scale_ink is not None:
            r = reader.read_scale(t.scale_ink, fonts="epma")
            if r.text:
                reads.append((r, t))
    if reads:
        texts = [textread.pretty_length(r.text) for r, _ in reads]
        top = max(set(texts), key=lambda x: (texts.count(x), sum(r.score for r, _ in reads if textread.pretty_length(r.text) == x)))
        res, ref = max(((r, t) for r, t in reads if textread.pretty_length(r.text) == top), key=lambda rt: rt[0].score)
    else:
        res, ref = textread.ReadResult("", 0.0, 0.0), tiles[0]
    stem, ext = Path(src.name).stem, Path(src.name).suffix
    panels = []
    for k, t in enumerate(tiles):
        nm = reader.read_name(t.name_ink, SH.DETECTORS, "epma")
        x0, y0, x1, y1 = t.box
        body = src.rgba[y0: y1 + 1, x0: x1 + 1, :3].astype(np.int16)
        grey = float(np.percentile(body.max(axis=2) - body.min(axis=2), 99.5)) <= 12
        if nm.text in SH.DETECTORS and nm.score >= 0.6:  # SED, BED-C ...: an electron image
            kind = "electron"
        elif nm.text in textread.ELEMENTS and nm.score >= 0.6:
            kind = "map"
        else:
            kind = "electron" if grey else "map"
        tsrc = SourceImage(name=f"{stem}_{k + 1:02d}{ext}", rgba=src.rgba, origin=src.origin, order=src.order,
                           uid=f"{src.uid}/tile{k + 1:02d}")
        sb = ScaleBar(length_px=bar_px, y=t.bar_y, x_left=float(t.bar_x[0]), x_right=float(t.bar_x[1]),
                      label_mask=ref.scale_ink, text=res.text, score=res.score, margin=res.margin)
        sb.value, sb.unit = textread.parse_length(res.text)
        p = Panel(src=tsrc, box=t.box, kind=kind, title_mask=t.name_ink, label=nm.text if kind == "map" else "",
                  label_score=nm.score, label_margin=nm.margin, label_candidates=nm.candidates, scalebar=sb,
                  full_box=t.box, set_id=src.name)
        p.band_preview = ref.preview  # the box under the map: name, bar and length as drawn
        panels.append(p)
    return panels


def read_sheet(path_or_bytes, name: str | None = None, reader: textread.Reader | None = None) -> Report:
    """A JEOL EPMA map window (one picture, many maps) as one figure, like an AZtec report: the maps
    share one field of view.  A picture that is not such a window is read as a plain image."""
    srcs = load_sources(path_or_bytes, name)
    reader = reader or textread.Reader()
    panels = []
    for src in srcs:
        panels += sheet_panels(src, reader) or [analyse_image(src, reader)]
    rep = Report(name=name or (srcs[0].name if srcs else "upload"), panels=panels, kind="aztec")
    calibrate(rep)
    return rep


def read_images(files: list[tuple[bytes, str]], reader: textread.Reader | None = None,
                txts: dict[str, str] | None = None, name: str | None = None) -> Report:
    """Loose image files (SEM micrographs, exported maps) as one set of independent pictures.  A JEOL
    EPMA map window gives one picture per map (those share one scale: Panel.set_id)."""
    reader = reader or textread.Reader()
    txts = txts or {}
    panels = []
    for i, (data, fname) in enumerate(files):
        for src in load_sources(data, fname):
            src.order = i
            tiles = None if src.meta_um_per_px else sheet_panels(src, reader)
            panels += tiles or [analyse_image(src, reader, txts.get(Path(fname).stem))]
    rep = Report(name=name or f"画像 {len(panels)} 枚", panels=panels, kind="images")
    calibrate(rep)
    return rep


def calibrate(rep: Report) -> None:
    """Pixel size of every panel: instrument metadata, else its own scale bar.

    AZtec set: all pictures share one field of view, so the field widths are cross-checked and a
    panel whose bar could not be read inherits the others' field.  Loose images are independent.
    A combined figure ("mixed") does this for each EDS set on its own.
    """
    rep.warnings = [w for w in rep.warnings if not w.startswith("[scale]")]
    for p in rep.panels:
        p.um_per_px = None
        p.scale_source = ""
        sb = p.scalebar
        if p.src.meta_um_per_px:
            p.um_per_px, p.scale_source = p.src.meta_um_per_px, "metadata"
        elif sb is not None and sb.length_um and sb.length_px > 0:
            p.um_per_px, p.scale_source = sb.length_um / sb.length_px, "bar"
    if rep.kind == "aztec":
        _cross_check(rep.panels, rep.warnings, "")
    elif rep.kind in ("mixed", "images"):  # EDS sets and EPMA map windows; loose images stand alone
        sets: dict[str, list[Panel]] = {}
        for p in rep.panels:
            if p.set_id:
                sets.setdefault(p.set_id, []).append(p)
        for sid, panels in sets.items():
            _cross_check(panels, rep.warnings, sid)


def _cross_check(panels: list[Panel], warnings: list[str], set_name: str) -> None:
    """One EDS set shares one field of view: compare the field widths, fill in missing ones."""
    regular = ("electron", "map", "layered")
    fields = [(p, p.body_size[0] * p.um_per_px) for p in panels  # as in the file: turning does not count
              if p.um_per_px and p.kind in regular]
    if not fields:
        return
    fw = float(np.median([f for _, f in fields]))
    vals = [f for _, f in fields]
    if max(vals) / min(vals) > 1.03:
        names = {"electron": "電子像", "map": "マップ", "layered": "重ね合わせ"}
        seen: dict[str, list[str]] = {}
        for p, f in fields:
            seen.setdefault(f"{f:.4g} µm", []).append(names.get(p.kind, p.kind))
        parts = [f"{'・'.join(sorted(set(k)))} {v}" for v, k in seen.items()]
        where = f"{set_name}：" if set_name else ""
        warnings.append(f"[scale] {where}視野幅が画像間で一致しません（" + "／".join(parts) +
                        "）。同じ視野のはずなので、スケール値の読み取りを確認してください。")
    for p in panels:
        if p.um_per_px is None and p.kind in regular:
            p.um_per_px, p.scale_source = fw / p.body_size[0], "bar"


def combine(reports: list[Report], name: str = "") -> Report:
    """One figure from several reports: the pictures of each Word report stay one EDS set (their
    fields are checked against each other), loose images stay independent."""
    panels = []
    for r in reports:
        for p in r.panels:
            if r.kind == "aztec":
                p.set_id = r.name
            panels.append(p)
    rep = Report(name=name or "まとめた図", panels=panels, kind="mixed")
    calibrate(rep)
    return rep


def set_scale_text(panel: Panel, text: str) -> None:
    """Apply a user-corrected scale label such as '5 µm'."""
    if panel.scalebar is None:
        return
    v, u = textread.parse_length(text)
    panel.scalebar.text = text
    panel.scalebar.value, panel.scalebar.unit = v, u


def set_band(panel: Panel, band_px: int | None) -> None:
    """Remove `band_px` rows of instrument band (None = as detected, 0 = keep the whole picture)."""
    if panel.band is None and not band_px:
        return
    if panel.full_box is not None:  # an AZtec picture (or a picture with a colour key): rows off its body
        x0, y0, x1, y1 = panel.full_box
        side = panel.band.side if panel.band is not None else "bottom"
        n = panel.band.height if (band_px is None and panel.band is not None) else \
            max(0, min(int(band_px or 0), (y1 - y0 + 1) // 2))
        panel.box = (x0, y0, x1, y1 - n) if side == "bottom" else (x0, y0 + n, x1, y1)
        panel.band_px = n
        return
    from . import sem

    H = panel.src.rgba.shape[0]
    n = panel.band.height if (band_px is None and panel.band is not None) else max(0, min(int(band_px or 0), H // 2))
    x0, y0, x1, y1 = sem.image_box(panel.src.rgba.shape[:2], panel.band, n)
    panel.box = (x0 + panel.crop_lr[0], y0, x1 - panel.crop_lr[1], y1)  # the colour bar stays removed
    panel.band_px = n


def set_rotation(panel: Panel, angle_cw: float = 0.0, flip_h: bool = False, flip_v: bool = False,
                 fill: str = "crop", resample: str = "bicubic") -> None:
    """Turn the image body clockwise by angle_cw degrees, then mirror it.  The scale (µm per pixel)
    read from the original picture stays valid: turning keeps the pixel pitch."""
    panel.rotation = float(angle_cw or 0.0)
    panel.flip_h, panel.flip_v = bool(flip_h), bool(flip_v)
    panel.rotate_fill = fill if fill in ("crop", "white", "black") else "crop"
    panel.rotate_resample = "nearest" if resample == "nearest" else "bicubic"


def _ink_preview(mask: np.ndarray | None, pad: int) -> np.ndarray | None:
    if mask is None:
        return None
    m = np.pad(mask, pad)
    return (255 - np.clip(m, 0, 1) * 255).astype(np.uint8)


def iter_reports(paths: Iterable[str | Path], reader: textread.Reader | None = None):
    reader = reader or textread.Reader()
    for p in paths:
        yield read_report(p, reader=reader)

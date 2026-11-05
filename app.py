"""Micrograph Maker — SEM・TEM・EPMA の像と EDS・WDS マップ（AZtec の Word 出力など）から論文用の図を作る Streamlit アプリ.

起動:  streamlit run app.py   （ネット接続は不要。ブラウザで http://localhost:8501 が開きます）
Streamlit 1.30 以降で動きます（Anaconda 同梱の古い版でも可。版によって使える機能だけを使います）。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import inspect
import io
import json
import os
import re
import threading
import zipfile
from dataclasses import asdict, replace
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from edsfig import __version__ as APP_VERSION
from edsfig import aztec, textread, transform
from edsfig import order as ordering
from appinfo import APP_NAME, APP_TAGLINE
from edsfig import figure as F
from edsfig import project as P
from edsfig import sheet as SH
from edsfig import tree as T

st.set_page_config(page_title=APP_NAME, page_icon="🔬", layout="wide")

# Shared on the campus network (started with serve.py, or bound to an address other PCs reach): students
# open it from their own PCs.  Their files stay in this PC's memory only; cached results go after some hours.
def _shared() -> bool:
    if os.environ.get("EDSFIG_SERVER", "") == "1":
        return True
    try:
        address = str(st.get_option("server.address") or "").strip().lower()
    except Exception:
        address = ""
    return address not in ("", "localhost", "127.0.0.1", "::1")


SERVER = _shared()
# PC 版（desktop/launcher.py が起動した Mac・Windows のアプリ）。EDSFIG_DESKTOP は window（アプリの窓）か
# browser（ブラウザで開いた）。共有モードでは使わない
DESKTOP = "" if SERVER else os.environ.get("EDSFIG_DESKTOP", "")
TTL = 6 * 3600 if SERVER else None
APP_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def lab_password() -> str:
    """The lab's password for the shared app: EDSFIG_PASSWORD, or password.txt next to app.py ("" = none)."""
    pw = os.environ.get("EDSFIG_PASSWORD", "")
    if not pw:
        try:
            pw = (APP_DIR / "password.txt").read_text(encoding="utf-8-sig")
        except OSError:
            pw = ""
    return pw.strip()


PASSWORD = lab_password() if SERVER else ""
if PASSWORD and not st.session_state.get("_auth"):
    st.title(APP_NAME)
    typed = st.text_input("パスワード", type="password", help="研究室で決めたパスワード（分からなければ管理している人に聞いてください）")
    if typed and hmac.compare_digest(typed.strip().encode("utf-8"), PASSWORD.encode("utf-8")):
        st.session_state["_auth"] = True
        st.rerun()
    if typed:
        st.error("パスワードが違います")
    st.stop()


# --------------------------------------------------------------------------- Streamlit version compatibility
def _version() -> tuple[int, int]:
    nums = [int(x) for x in re.findall(r"\d+", st.__version__)[:2]]
    return (nums + [0, 0])[0], (nums + [0, 0])[1]


ST_VERSION = _version()
# download_button(data=callable) builds the file only when clicked (Streamlit 1.52+)
DEFERRED_DOWNLOAD = ST_VERSION >= (1, 52)


def _params(func) -> dict:
    try:
        return dict(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return {}


def opt(func, **kwargs) -> dict:
    """Only the keyword arguments this Streamlit version accepts."""
    p = _params(func)
    return {k: v for k, v in kwargs.items() if k in p}


def full_width(func) -> dict:
    """'Fill the available width' in the spelling of this Streamlit version."""
    p = _params(func)
    w = p.get("width")
    if w is not None and isinstance(w.default, str):  # newer: width="stretch"
        return {"width": "stretch"}
    for k in ("use_container_width", "use_column_width"):  # older spellings
        if k in p:
            return {k: True}
    return {}


def toggle(container, *args, **kwargs):
    f = getattr(container, "toggle", None) or container.checkbox  # st.toggle is 1.26+
    return f(*args, **kwargs)


if not (hasattr(st, "data_editor") and hasattr(st, "column_config")):
    st.error(f"このアプリには Streamlit 1.23 以降が必要です（いまは {st.__version__}）。"
             "ターミナルで `pip install -U streamlit` を実行してから、もう一度起動してください。")
    st.stop()

FW_IMAGE = full_width(st.image)
FW_EDITOR = full_width(st.data_editor)
FW_BUTTON = full_width(st.button)
FW_DOWNLOAD = full_width(st.download_button)
RESAMPLE = getattr(Image, "Resampling", Image)  # Pillow 9.1+ / older

KIND_JA = {"electron": "電子像", "map": "マップ", "layered": "重ね合わせ", "spectrum": "スペクトル", "other": "その他"}
JA_KIND = {v: k for k, v in KIND_JA.items()}
IMAGE_TYPES = ["png", "tif", "tiff", "jpg", "jpeg", "bmp", "dm3", "dm4"]
UPLOAD_TYPES = ["docx"] + IMAGE_TYPES + ["txt", P.EXT.lstrip(".")]
FORMATS = [  # label, extension, MIME type
    ("PDF", "pdf", "application/pdf"),
    ("PNG", "png", "image/png"),
    ("TIFF", "tiff", "image/tiff"),
    ("SVG", "svg", "image/svg+xml"),
    ("画像ごと（ZIP）", "zip", "application/zip"),
]
FORMAT_HELP = {"zip": "切り出した画像をPNGで保存（ラベル・バーなし）＋較正情報 calibration.json。明るさ等を変えた画像は、変える前の元画素も original/ に入ります"}


# --------------------------------------------------------------------------- cached work
@st.cache_resource
def get_reader() -> textread.Reader:
    return textread.Reader()


@st.cache_data(show_spinner="AZtecの出力を解析しています…", max_entries=64, ttl=TTL)
def parse_docx(data: bytes, name: str) -> aztec.Report:
    return aztec.read_report(data, name=name, reader=get_reader())


@st.cache_data(show_spinner="EPMA のマップ画面を解析しています…", max_entries=32, ttl=TTL)
def parse_sheet(data: bytes, name: str) -> aztec.Report:
    return aztec.read_sheet(data, name=name, reader=get_reader())


@st.cache_data(show_spinner=False, max_entries=256, ttl=TTL)
def is_map_window(data: bytes) -> bool:
    """A JEOL EPMA map window saved as one picture (maps on teal tiles): read as its own figure."""
    try:
        rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:  # (DigitalMicrograph, 16-bit TIFF ...: not a screen picture)
        return False
    return SH.is_sheet(rgb)


@st.cache_data(show_spinner="画像を解析しています…", max_entries=16, ttl=TTL)
def parse_images(blobs: tuple, names: tuple, txts: tuple, title: str) -> aztec.Report:
    return aztec.read_images(list(zip(blobs, names)), reader=get_reader(), txts=dict(txts), name=title)


def data_uri(arr: np.ndarray | None, max_w: int | None = None, zoom: int = 1) -> str | None:
    if arr is None:
        return None
    im = Image.fromarray(arr)
    if max_w and im.width > max_w:  # thumbnail for the table only
        im = im.resize((max_w, max(1, round(im.height * max_w / im.width))), RESAMPLE.LANCZOS)
    if zoom > 1:
        im = im.resize((im.width * zoom, im.height * zoom), RESAMPLE.NEAREST)
    b = io.BytesIO()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def confident(score: float, margin: float) -> bool:
    return score >= 0.80 and margin >= 0.03


BLANK = data_uri(np.full((4, 4), 255, np.uint8))


def scale_status(p: aztec.Panel) -> str:
    if p.src.meta_um_per_px:
        return "スケール：機器の記録"
    if p.src.reciprocal:
        return "スケールなし（回折像など）"
    sb = p.scalebar
    if sb is None:
        return "スケールなし"
    if not sb.text:
        return "スケール要入力"
    return "スケール✓" if confident(sb.score, sb.margin) else "スケール要確認"


@st.cache_data(show_spinner=False, max_entries=64, ttl=TTL)
def default_rows(key: str, _rep: aztec.Report) -> pd.DataFrame:
    rows = []
    images = _rep.kind in ("images", "mixed")  # loose pictures (or some of them)
    bands = images or any(p.legend is not None for p in _rep.panels)  # rows cut off: info band, colour key
    for i, p in enumerate(_rep.panels):
        if p.kind == "map" or (p.kind == "layered" and p.legend is not None):
            status = "✓" if confident(p.label_score, p.label_margin) else "要確認"
        elif images and p.kind == "electron":
            status = scale_status(p)
        else:
            status = ""
        if p.legend is not None:  # a layered image: its colour key, as drawn
            title = data_uri(p.legend_preview())
        elif p.title_mask is not None:
            title = data_uri(p.title_preview(), zoom=2)
        else:
            title = data_uri(p.scale_label_preview(), zoom=2) if images else None
        row = {
            "use": p.include,
            "order": i + 1,
            "thumb": data_uri(p.image, 180),
            "title": title or BLANK,  # older Streamlit prints "None" otherwise
            "label": p.label,
            "status": status,
            "kind": KIND_JA[p.kind],
        }
        if bands:
            row["band"] = int(p.band_px)
        rows.append(row)
    return pd.DataFrame(rows)


def mask_key(m: np.ndarray) -> str:
    return hashlib.md5(np.round(m * 255).astype(np.uint8).tobytes() + str(m.shape).encode()).hexdigest()[:10]


def picture_keys(rep: aztec.Report) -> list[str]:
    """A key for every picture that stays the same when files are added or removed (the content of its
    file and its place in the file), so that its settings follow it into any figure.  The same picture
    loaded twice gets #2, #3."""
    out, seen = [], {}
    for p in rep.panels:
        base = p.src.uid or f"{p.src.origin}|{p.src.name}|{p.src.order}"
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}#{seen[base]}")
    return out


def scale_groups(rep: aztec.Report) -> dict[str, list[int]]:
    """Panels whose scale label is the same ink are confirmed together (metadata wins over bars)."""
    groups: dict[str, list[int]] = {}
    pks = picture_keys(rep)
    for i, p in enumerate(rep.panels):
        if p.kind in ("other", "spectrum") or p.scalebar is None or p.src.meta_um_per_px:
            continue
        m = p.scalebar.label_mask
        groups.setdefault(mask_key(m) if m is not None else f"bar|{pks[i]}", []).append(i)
    return groups


def needs_field(rep: aztec.Report) -> list[int]:
    """Pictures without metadata and without a scale bar (the field width must be typed)."""
    return [i for i, p in enumerate(rep.panels)
            if p.kind not in ("other", "spectrum") and p.scalebar is None and not p.src.meta_um_per_px
            and not p.src.reciprocal]


def pretty_len(text: str) -> str:
    return textread.pretty_length(text)


def cell_text(v) -> str:
    """A table cell as text; a cleared cell (None/NaN) is empty."""
    return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v).strip()


def cell_num(v, default: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(x) or x <= 0 else x


def cell_int(v) -> int | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else max(0, int(round(x)))


ADJ_DEFAULT = {"auto": False, "brightness": 0, "contrast": 0, "gamma": 1.0}
ADJ_PREFIX = {"auto": "aa", "brightness": "ab", "contrast": "ac", "gamma": "ag"}


def adj_key(name: str, pk: str) -> str:
    return f"{ADJ_PREFIX[name]}_{pk}"


def to_adjust(d: dict | None) -> F.Adjust:
    d = {**ADJ_DEFAULT, **(d or {})}
    return F.Adjust(auto=bool(d["auto"]), brightness=float(d["brightness"]), contrast=float(d["contrast"]),
                    gamma=float(d["gamma"]))


def reset_adjust(pk: str) -> None:
    for n, d in ADJ_DEFAULT.items():
        st.session_state[adj_key(n, pk)] = d


def copy_adjust(pk: str, targets: list[str]) -> None:
    vals = {n: st.session_state.get(adj_key(n, pk), d) for n, d in ADJ_DEFAULT.items()}
    for t in targets:
        for n, v in vals.items():
            st.session_state[adj_key(n, t)] = v


CROP_DEFAULT = {"zoom": 1.0, "fx": 50, "fy": 50}
CROP_PREFIX = {"zoom": "cz", "fx": "cx", "fy": "cy"}
ASPECT_PRESETS = {"自動（切り取りが最も少ない形）": None, "1:1（正方形）": 1.0, "4:3（横長）": 4 / 3, "3:2（横長）": 1.5,
                  "16:9（横長）": 16 / 9, "3:4（縦長）": 3 / 4, "2:3（縦長）": 2 / 3}


def crop_key(name: str, pk: str) -> str:
    return f"{CROP_PREFIX[name]}_{pk}"


def aspect_options(rep: aztec.Report) -> list[str]:
    return list(ASPECT_PRESETS) + [f"{i + 1}. {p.src.name} と同じ形" for i, p in enumerate(rep.panels)
                                   if p.kind in F.REGULAR]


def aspect_of(choice: str | None, rep: aztec.Report, idx: list[int]) -> float | None:
    """Width/height to trim to: a preset, the shape of one picture, or the one that trims least."""
    if choice in ASPECT_PRESETS and ASPECT_PRESETS[choice]:
        return ASPECT_PRESETS[choice]
    if choice and choice.endswith(" と同じ形"):
        try:
            w, h = rep.panels[int(choice.split(".", 1)[0]) - 1].size
            return w / h
        except (ValueError, IndexError):
            pass
    return F.least_trim_aspect([rep.panels[i].size for i in idx if rep.panels[i].kind in F.REGULAR])


def crop_boxes(rep: aztec.Report, idx: list[int], choice: str | None, per: dict[int, dict]) -> dict[int, tuple]:
    t = aspect_of(choice, rep, idx)
    out = {}
    for i in idx:
        p = rep.panels[i]
        if t and p.kind in F.REGULAR:
            c = {**CROP_DEFAULT, **(per.get(i) or {})}
            out[i] = F.crop_box(p.size, t, c["fx"] / 100, c["fy"] / 100, c["zoom"])
    return out


def reset_crop(pk: str) -> None:
    for n, d in CROP_DEFAULT.items():
        st.session_state[crop_key(n, pk)] = d


# trimming of one picture (any arrangement except "トリミングして揃える", which trims all to one shape)
TRIM_SHAPES = {"切り取らない": "none", "元の縦横比のまま": "same", "1:1（正方形）": 1.0, "4:3（横長）": 4 / 3,
               "3:2（横長）": 1.5, "16:9（横長）": 16 / 9, "3:4（縦長）": 3 / 4, "2:3（縦長）": 2 / 3,
               "9:16（縦長）": 9 / 16, "範囲を指定": "free"}
TRIM_DEFAULT = {"shape": "切り取らない", "zoom": 1.0, "fx": 50, "fy": 50, "xr": (0.0, 100.0), "yr": (0.0, 100.0)}
TRIM_PREFIX = {"shape": "ts", "zoom": "tz", "fx": "tfx", "fy": "tfy", "xr": "txr", "yr": "tyr"}


def trim_key(name: str, pk: str) -> str:
    return f"{TRIM_PREFIX[name]}_{pk}"


def trim_box(size: tuple[int, int], t: dict) -> tuple[int, int, int, int] | None:
    """The part of a picture kept by its trimming settings (whole pixels), or None for all of it."""
    W, H = int(size[0]), int(size[1])
    t = {**TRIM_DEFAULT, **(t or {})}
    shape = TRIM_SHAPES.get(t["shape"], "none")
    if shape == "none":
        return None
    if shape == "free":
        (xa, xb), (ya, yb) = sorted(t["xr"]), sorted(t["yr"])
        x0, x1 = int(round(W * xa / 100)), int(round(W * xb / 100))
        y0, y1 = int(round(H * ya / 100)), int(round(H * yb / 100))
        x0, y0 = min(max(0, x0), W - 1), min(max(0, y0), H - 1)
        box = (x0, y0, min(W, max(x1, x0 + 1)), min(H, max(y1, y0 + 1)))
    else:
        box = F.crop_box((W, H), W / H if shape == "same" else shape, t["fx"] / 100, t["fy"] / 100, t["zoom"])
    return None if box == (0, 0, W, H) else box


def trim_boxes(rep: aztec.Report, idx, trims: dict | None) -> dict[int, tuple]:
    out = {}
    for i in idx:
        b = trim_box(rep.panels[i].size, (trims or {}).get(i))
        if b is not None:
            out[i] = b
    return out


def reset_trim(pk: str) -> None:
    for n, d in TRIM_DEFAULT.items():
        st.session_state[trim_key(n, pk)] = d


def same_view(rep: aztec.Report, i: int, idx: list[int]) -> list[int]:
    """Pictures that show the same field as picture i, for copying its trimming: the electron image
    and maps of one Word report (maps are often coarser), or pictures of the same pixel size.  The
    trimming settings are fractions of the picture, so they pick the same region at any resolution."""
    p = rep.panels[i]
    a = p.size[0] / p.size[1]
    out = []
    for j in idx:
        q = rep.panels[j]
        if j == i or abs((q.size[0] / q.size[1]) / a - 1) > 0.02:
            continue
        one_set = p.kind in F.REGULAR and q.kind in F.REGULAR and (
            rep.kind == "aztec" or (bool(p.set_id) and q.set_id == p.set_id))
        if one_set or q.size == p.size:
            out.append(j)
    return out


def copy_trim(pk: str, targets: list[str]) -> None:
    vals = {n: st.session_state.get(trim_key(n, pk), d) for n, d in TRIM_DEFAULT.items()}
    for t in targets:
        for n, v in vals.items():
            st.session_state[trim_key(n, t)] = v


def number_overlay(png: bytes, g: F.Geometry) -> bytes:
    """The preview with each picture's number in the middle (on screen only, never exported)."""
    from PIL import ImageDraw, ImageFont

    im = Image.open(io.BytesIO(png)).convert("RGBA")
    s = im.width / g.width
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font_file = Path(F.__file__).parent / "assets" / "fonts" / "Carlito-Bold.ttf"
    for (x, y, w, h, k), ref in zip(g.boxes, g.refs):
        size = int(max(12, min(56, 0.3 * min(w, h) * s)))
        try:
            font = ImageFont.truetype(str(font_file), size)
        except OSError:
            font = ImageFont.load_default()
        text = str(ref)
        l, t, r, b = d.textbbox((0, 0), text, font=font)
        cx, cy = (x + w / 2) * s, (y + h / 2) * s
        pad = 0.35 * size
        rad = max(r - l, b - t) / 2 + pad
        d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=(0, 0, 0, 170), outline=(255, 196, 0, 255),
                  width=max(1, size // 12))
        d.text((cx - (l + r) / 2, cy - (t + b) / 2), text, font=font, fill=(255, 255, 255, 255))
    out = io.BytesIO()
    Image.alpha_composite(im, layer).convert("RGB").save(out, "PNG")
    return out.getvalue()


def crop_picture(img: np.ndarray, box: tuple) -> np.ndarray:
    """The whole picture with the part that is kept framed and the rest dimmed."""
    x0, y0, x1, y1 = box
    out = (img.astype(np.float32) * 0.35).astype(np.uint8)
    out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    t = max(2, round(max(img.shape[:2]) / 160))
    col = np.array([255, 196, 0], np.uint8)
    out[y0: min(y1, y0 + t), x0:x1] = col
    out[max(y0, y1 - t): y1, x0:x1] = col
    out[y0:y1, x0: min(x1, x0 + t)] = col
    out[y0:y1, max(x0, x1 - t): x1] = col
    return out


ROT_DEFAULT = {"angle": 0.0, "fh": False, "fv": False}
ROT_PREFIX = {"angle": "ra", "fh": "rh", "fv": "rv"}


def rot_key(name: str, pk: str) -> str:
    return f"{ROT_PREFIX[name]}_{pk}"


def apply_rotations(rep: aztec.Report, idx, rot: dict | None) -> None:
    """rot = {"per": {index: {angle, fh, fv}}, "fill": crop|white|black, "resample": bicubic|nearest}"""
    rot = rot or {}
    for i in idx:
        r = {**ROT_DEFAULT, **((rot.get("per") or {}).get(i) or {})}
        aztec.set_rotation(rep.panels[i], r["angle"], r["fh"], r["fv"], rot.get("fill", "crop"),
                           rot.get("resample", "bicubic"))


def copy_rotation(pk: str, targets: list[str]) -> None:
    vals = {n: st.session_state.get(rot_key(n, pk), d) for n, d in ROT_DEFAULT.items()}
    for t in targets:
        for n, v in vals.items():
            st.session_state[rot_key(n, t)] = v


def by_index(store: dict, pks: list[str]) -> dict[int, object]:
    """Per-picture settings (kept by picture key) as {panel index: setting} for one figure."""
    return {i: store[pk] for i, pk in enumerate(pks) if pk in store}


def _num(v, default: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(x) else x


def is_used(v) -> bool:
    return cell_text(v) in ("True", "true", "1")


def with_places(table: pd.DataFrame, places: list[int]) -> pd.DataFrame:
    """The table numbered by `places` (by row position) and listed in that order (rows keep their
    picture index, so every per-picture setting stays attached)."""
    out = table.copy()
    out["order"] = [int(v) for v in places]
    return out.sort_values("order", kind="stable")


def renumbered(table: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame | None:
    """The "順" column read as places: a picture given a number moves there and the others shift.
    None when the numbers are already 1..n in that order."""
    old = [_num(v, k + 1) for k, v in enumerate(base["order"])]
    new = [_num(v, old[k]) for k, v in enumerate(table["order"])]
    places = ordering.renumber(old, new)
    if places == [_num(v, -1) for v in table["order"]] and places == sorted(places):
        return None  # numbered 1..n and listed in that order already
    return with_places(table, places)


def used_first(table: pd.DataFrame, switched_on=(), switched_off=()) -> pd.DataFrame:
    """Numbered 1..n with the pictures in use first and those left out at the bottom; a picture just
    switched off goes to the very bottom, one just switched on right after the others in use."""
    rows = list(table.index)
    places = ordering.used_first([_num(v, 1e9) for v in table["order"]], [is_used(v) for v in table["use"]],
                                 {rows.index(i) for i in switched_on}, {rows.index(i) for i in switched_off})
    return with_places(table, places)


def switched(table: pd.DataFrame, base: pd.DataFrame) -> tuple[list, list]:
    """Rows whose "使う" was switched on / off since `base`."""
    on, off = [], []
    for i in table.index:
        if i in base.index:
            was, now = is_used(base.at[i, "use"]), is_used(table.at[i, "use"])
            if now and not was:
                on.append(i)
            elif was and not now:
                off.append(i)
    return on, off


def renumber_spec(rkey: str, old: pd.DataFrame, new: pd.DataFrame) -> None:
    """The figure's free arrangement follows its pictures when their numbers change."""
    mapping = {}
    for i in new.index:
        if i in old.index:
            a, b = cell_int(old.at[i, "order"]), cell_int(new.at[i, "order"])
            if a and b:
                mapping[a] = b
    if all(a == b for a, b in mapping.items()):
        return
    for store in ("specs", "spec_ok"):
        d = st.session_state.setdefault(store, {})
        if d.get(rkey):
            d[rkey] = T.renumber_text(d[rkey], mapping)
    wk = f"spec_{rkey}"
    if wk in st.session_state:
        st.session_state[wk] = T.renumber_text(st.session_state[wk], mapping)


def remember_rows(table: pd.DataFrame, pks: list[str]) -> None:
    """Each picture's 使う・ラベル・種類・帯 as last set, for any figure it appears in."""
    rows = st.session_state.setdefault("picrow", {})
    for i in table.index:
        rec = {"use": is_used(table.at[i, "use"]), "label": cell_text(table.at[i, "label"]),
               "kind": cell_text(table.at[i, "kind"])}
        if "band" in table.columns and cell_int(table.at[i, "band"]) is not None:
            rec["band"] = cell_int(table.at[i, "band"])
        rows[pks[i]] = rec


def best_source(rkey: str, pks: list[str]) -> str | None:
    """The figure seen before that shares the most pictures with this one (the last shown on a tie)."""
    tables, tpks = st.session_state.get("tables", {}), st.session_state.get("tpks", {})
    mine, shown = set(pks), st.session_state.get("_shown")
    best, best_n = None, 0
    for k, m in tpks.items():
        if k == rkey or k not in tables:
            continue
        n = len(mine & set(m.values()))
        if n > best_n or (n == best_n and n > 0 and k == shown):
            best, best_n = k, n
    return best


def table_for(rkey: str, rep: aztec.Report, pks: list[str], adopt: bool = False) -> pd.DataFrame:
    """The figure's table as it stands: its own, or, for a figure seen for the first time, numbered as in
    an opened work file, or (another file was added, say) like the figure that shares the most pictures
    with it, new pictures after those.  Every picture keeps its 使う・ラベル・種類・帯 as last set in any
    figure, and the pictures in use come first.  adopt=True: this figure is being shown (its
    arrangement follows along)."""
    tables, tpks = st.session_state.setdefault("tables", {}), st.session_state.setdefault("tpks", {})
    known = tables.get(rkey)
    src = sv = None
    if known is not None:
        t = known.copy()
    else:
        t = default_rows(rkey, rep).copy()
        sv = saved_figure(rkey, pks, exact=True)  # this very figure in an opened work file
        if sv is None:
            src = best_source(rkey, pks)
            if src is None:  # a work file opened with its files in another order, or with more files
                sv = saved_figure(rkey, pks)
        if sv is not None:  # the order as saved
            rank = sv["order"]
        elif src is not None:  # the order of the figure it grew out of
            st_, sp = tables[src], tpks[src]
            rank = {sp[i]: _num(st_.at[i, "order"], 1e9) for i in st_.index if i in sp}
        else:
            rank = None
        if rank:
            pos = {i: k for k, i in enumerate(t.index)}
            seq = sorted(t.index, key=lambda i: (0, rank[pks[i]], pos[i]) if pks[i] in rank else (1, 0, pos[i]))
            t = with_places(t, [seq.index(i) + 1 for i in t.index])
    before = t.copy()
    rows = st.session_state.get("picrow", {})
    for i in t.index:
        rec = rows.get(pks[i])
        if rec:
            for col, v in rec.items():
                if col in t.columns:
                    t.at[i, col] = v
    if sv is not None:  # numbers and 使う as saved (nothing was just switched)
        t = used_first(t)
    else:
        on, off = switched(t, before)
        t = used_first(t, on, off)
    if sv is not None:  # its free arrangement and trimming shape come along (numbers follow the pictures)
        new_num = {pks[i]: cell_int(t.at[i, "order"]) for i in t.index}
        mapping = {n: new_num[pk] for pk, n in sv["order"].items() if new_num.get(pk)}
        for store, field in (("specs", "spec"), ("spec_ok", "spec_ok")):
            d = st.session_state.setdefault(store, {})
            if sv.get(field) and not d.get(rkey):
                d[rkey] = T.renumber_text(sv[field], mapping)
        ca = st.session_state.setdefault("crop_aspect", {})
        if sv.get("aspect") and rkey not in ca and (sv.get("key") == rkey or sv["aspect"] in ASPECT_PRESETS):
            ca[rkey] = sv["aspect"]
    elif adopt:
        if known is not None:
            renumber_spec(rkey, before, t)
        elif src is not None:  # the free arrangement and the common trimming shape come along
            new_num = {pks[i]: cell_int(t.at[i, "order"]) for i in t.index}
            st_, sp = tables[src], tpks[src]
            mapping = {cell_int(st_.at[i, "order"]): new_num[sp[i]] for i in st_.index if sp.get(i) in new_num}
            for store in ("specs", "spec_ok"):
                d = st.session_state.setdefault(store, {})
                if d.get(src) and not d.get(rkey):
                    d[rkey] = T.renumber_text(d[src], mapping)
            ca = st.session_state.setdefault("crop_aspect", {})
            if ca.get(src) in ASPECT_PRESETS and rkey not in ca:
                ca[rkey] = ca[src]
    return t


def move_picture(rkey: str, i: int, step: int) -> None:
    """◀ / ▶ on a card: one place earlier / later among the pictures in use."""
    tables = st.session_state.setdefault("tables", {})
    t = tables.get(rkey)
    if t is None:
        return
    if i not in t.index:
        return
    places = ordering.move([_num(v, k + 1) for k, v in enumerate(t["order"])], [is_used(v) for v in t["use"]],
                           list(t.index).index(i), step)
    if places is None:
        return
    new = with_places(t, places)
    tables[rkey] = new
    st.session_state.setdefault("bases", {})[rkey] = new  # shown in a fresh table widget
    ver = st.session_state.setdefault("edver", {})
    ver[rkey] = ver.get(rkey, 0) + 1


def picture_name(rep: aztec.Report, p: aztec.Panel) -> str:
    """How a picture is called on screen: the file name of a loose image, the kind of an EDS picture
    (with its Word file in a combined figure)."""
    if rep.kind == "images" or (rep.kind == "mixed" and not p.set_id):
        return p.src.name
    if rep.kind == "mixed":
        return f"{KIND_JA[p.kind]}（{Path(p.set_id).stem[:16]}）"
    return KIND_JA[p.kind]


def parse_rows(text: str) -> tuple[int, ...] | None:
    """'2,3' -> (2, 3); empty or anything else -> None."""
    parts = [t for t in re.split(r"[,、，\s]+", (text or "").strip()) if t]
    if not parts or not all(t.isdigit() and int(t) >= 1 for t in parts):
        return None
    return tuple(int(t) for t in parts)


def apply_bands(rep: aztec.Report, table: pd.DataFrame) -> None:
    if "band" in table.columns:
        for i, row in table.iterrows():
            aztec.set_band(rep.panels[i], cell_int(row["band"]))


BAR_CHOICES = {"共通": None, "白": "white", "黒": "black"}


def build_items(rep: aztec.Report, table: pd.DataFrame, scale_text: dict[str, str],
                fields: dict[int, str] | None = None, adj: dict[int, dict] | None = None,
                crop: dict | None = None, bars: dict[int, str] | None = None,
                rot: dict | None = None, trims: dict | None = None) -> list[F.Item]:
    """Apply the user's table and scale values to a (fresh, cached-copy) report.
    crop = {"aspect": choice, "per": {index: {zoom, fx, fy}}} trims the pictures to one shape;
    trims = {index: {shape, zoom, fx, fy, xr, yr}} trims pictures one by one (when crop is None);
    bars = {index: "共通" | "白" | "黒"} is each picture's scale bar colour; rot turns pictures.
    Every item carries its "順" number (the free arrangement refers to it)."""
    apply_bands(rep, table)
    apply_rotations(rep, range(len(rep.panels)), rot)
    for gk, idx in scale_groups(rep).items():
        if gk in scale_text:
            for i in idx:
                aztec.set_scale_text(rep.panels[i], scale_text[gk])
    for i, row in table.iterrows():
        rep.panels[i].kind = JA_KIND.get(cell_text(row["kind"]), rep.panels[i].kind)
    aztec.calibrate(rep)
    for i, text in (fields or {}).items():  # typed field width for pictures without a bar
        v, u = textread.parse_length(text)
        p = rep.panels[i]
        if v and p.um_per_px is None:
            p.um_per_px, p.scale_source = v * textread.UNIT_TO_UM[u] / p.body_size[0], "manual"  # as in the file
    use = table["use"].map(lambda v: cell_text(v) in ("True", "true", "1"))
    rows = table[use].sort_values("order", kind="stable")
    idx = [int(i) for i in rows.index]
    if crop:
        boxes = crop_boxes(rep, idx, crop.get("aspect"), crop.get("per") or {})
    else:
        boxes = trim_boxes(rep, idx, trims)
    items = []
    for k, (i, row) in enumerate(rows.iterrows()):
        items.append(F.Item(rep.panels[i], cell_text(row["label"]), to_adjust((adj or {}).get(i)), boxes.get(int(i)),
                            BAR_CHOICES.get((bars or {}).get(int(i), "共通")), ref=cell_int(row["order"]) or k + 1))
    return items


def default_spec(items: list[F.Item], ncols: int, counts) -> str:
    """The grid arrangement written out as a free arrangement (rows of numbers)."""
    refs = F.numbers(items)
    return T.default_text(refs, F.row_lengths(len(refs), ncols, counts))


def export_signature(items: list[F.Item], style: F.Style, extra: str) -> str:
    """Changes whenever anything that affects the exported file changes."""
    h = hashlib.md5(extra.encode())
    h.update(repr(asdict(style)).encode())
    for it in items:
        p = it.panel
        h.update(f"{p.src.origin}|{p.src.name}|{p.src.order}|{p.box}|{p.kind}|{it.label}|{asdict(it.adjust)}|"
                 f"{p.um_per_px!r}|{it.crop}|{it.bar_color}|{p.rotation}|{p.flip_h}|{p.flip_v}|{p.rotate_fill}|"
                 f"{p.rotate_resample}|{it.ref}".encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- the sidebar's choices
SIZE_MODES = {"幅を指定": "width", "高さを指定": "height", "幅と高さを指定": "box"}
ARRANGES = {"列数で並べる": "grid", "自由配置（番号で指定）": "free"}
ROW_ALIGNS = {"自動": "auto", "左寄せ": "left", "中央": "center", "右寄せ": "right"}
ALIGNS = {"幅を揃える": "width", "高さを揃える": "height", "トリミングして揃える": "crop"}
VALIGNS = {"中央": "center", "上": "top", "下": "bottom"}
BACKGROUNDS = {"白": "white", "黒": "black", "透明": "transparent"}
LABEL_COLORS = {"白": "white", "黒": "black", "マップの色": "map"}
CORNERS = {"左上": "top-left", "右上": "top-right", "左下": "bottom-left", "右下": "bottom-right"}
BAR_TARGETS = {"電子像・SEM像": "electron", "全パネル": "all", "付けない": "none"}
BAR_CORNERS = {"右下": "bottom-right", "左下": "bottom-left"}
BAR_COLORS = {"白": "white", "黒": "black", "自動（背景で選ぶ）": "auto"}
RES_MODES = ["元画素の整数倍", "dpi を指定"]
ROT_FILLS = {"切り取る": "crop", "白で埋める": "white", "黒で埋める": "black"}
ROT_RESAMPLES = {"バイキュービック": "bicubic", "最近傍": "nearest"}
INTERPS = {"ドットのまま": "nearest", "なめらか": "lanczos"}

# Every setting of the sidebar: key -> (default, what it may be: bool / str / options / (min, max)).
# The widgets take their value from these keys only (never value=), so a work file can set them.
SIDEBAR = {
    "combine": (False, bool),
    "size_mode": ("幅を指定", list(SIZE_MODES)),
    "width_mm": (170.0, (20.0, 1000.0)),
    "height_mm": (120.0, (20.0, 1000.0)),
    "arrange": ("列数で並べる", list(ARRANGES)),
    "auto_cols": (False, bool),
    "ncols": (3, (1, 10)),
    "rows_text": ("", str),
    "row_align": ("自動", list(ROW_ALIGNS)),
    "gap_mm": (1.0, (0.0, 20.0)),
    "margin_mm": (0.0, (0.0, 30.0)),
    "align": ("幅を揃える", list(ALIGNS)),
    "valign": ("中央", list(VALIGNS)),
    "big": (False, bool),
    "bg": ("白", list(BACKGROUNDS)),
    "label_on": (True, bool),
    "label_pt": (10.0, (4.0, 48.0)),
    "label_col": ("白", list(LABEL_COLORS)),
    "lo": (False, bool),
    "label_corner": ("左上", list(CORNERS)),
    "bar_target": ("電子像・SEM像", list(BAR_TARGETS)),
    "bar_len": (0.0, (0.0, 100000.0)),
    "bar_corner": ("右下", list(BAR_CORNERS)),
    "bar_col": ("白", list(BAR_COLORS)),
    "bar_pt": (9.0, (4.0, 48.0)),
    "bar_text": (True, bool),
    "bo": (False, bool),
    "bar_thick_pt": (2.0, (0.5, 8.0)),
    "res_mode": ("元画素の整数倍", RES_MODES),
    "oversample": (2, (1, 8)),
    "dpi": (600, (72, 2400)),
    "rot_fill": ("切り取る", list(ROT_FILLS)),
    "rot_resample": ("バイキュービック", list(ROT_RESAMPLES)),
    "interp": ("ドットのまま", list(INTERPS)),
}


def fits(rule: tuple, v) -> bool:
    """A saved value the widget can show (anything else keeps the current setting)."""
    default, kind = rule
    if kind is bool:
        return isinstance(v, bool)
    if kind is str:
        return isinstance(v, str)
    if kind == "pair":  # a range slider: two numbers within 0..100
        return (isinstance(v, (list, tuple)) and len(v) == 2
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 100 for x in v))
    if isinstance(kind, list):
        return v in kind
    lo, hi = kind
    return isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi


def as_saved(rule: tuple, v):
    """The value in the type its widget was made with (int sliders get ints, float ones floats)."""
    default = rule[0]
    if rule[1] == "pair":
        return tuple(float(x) for x in v)
    return v if isinstance(default, (bool, str)) else type(default)(v)


# per-picture settings in a work file: name in the file -> (store in the session, fields)
PICTURE_STORES = {
    "picrow": ("picrow", {"use": (True, bool), "label": ("", str), "kind": ("電子像", list(KIND_JA.values())),
                          "band": (0, (0, 100000))}),
    "adjust": ("adj_pk", {"auto": (False, bool), "brightness": (0, (-100, 100)), "contrast": (0, (-100, 200)),
                          "gamma": (1.0, (0.3, 3.0))}),
    "rotate": ("rot_pk", {"angle": (0.0, (-360.0, 360.0)), "fh": (False, bool), "fv": (False, bool)}),
    "crop": ("crop_pk", {"zoom": (1.0, (1.0, 5.0)), "fx": (50, (0, 100)), "fy": (50, (0, 100))}),
    "trim": ("trims_pk", {"shape": ("切り取らない", list(TRIM_SHAPES)), "zoom": (1.0, (1.0, 5.0)), "fx": (50, (0, 100)),
                          "fy": (50, (0, 100)), "xr": ((0.0, 100.0), "pair"), "yr": ((0.0, 100.0), "pair")}),
}
# widgets holding one picture's setting (key = prefix + "_" + picture key), or one figure's
PICTURE_WIDGET = re.compile(r"^(aa|ab|ac|ag|cz|cx|cy|ts|tz|tfx|tfy|txr|tyr|ra|rh|rv|bc|fw)_(.+)$")
FIGURE_WIDGET = re.compile(r"^(spec|ca)_(.+)$")


class Blob:
    """A file taken out of a work file (used like an uploaded file)."""

    def __init__(self, name: str, data: bytes, md5: str):
        self.name, self._data, self.md5 = name, data, md5

    def getvalue(self) -> bytes:
        return self._data


def upload_id(f) -> str:
    fid = getattr(f, "file_id", None)
    return str(fid) if fid else f"{f.name}|{hashlib.md5(f.getvalue()).hexdigest()}"


def md5_of(f) -> str:
    """The content key of a loaded file (kept per upload: a file is read through once)."""
    if getattr(f, "md5", None):
        return f.md5
    cache = st.session_state.setdefault("_md5", {})
    fid = getattr(f, "file_id", None)
    if fid and fid in cache:
        return cache[fid]
    h = hashlib.md5(f.getvalue()).hexdigest()
    if fid:
        cache[fid] = h
    return h


def restore(doc: dict) -> None:
    """Put a work file's settings in place (once, when it is loaded).  Pictures and figures it does not
    mention keep theirs; the widgets of those it does start again from the saved values."""
    ss = st.session_state

    def part(d, k, kind=dict):  # one part of the saved settings ({} / [] if missing or damaged)
        v = d.get(k) if isinstance(d, dict) else None
        return v if isinstance(v, kind) else kind()

    s = part(doc, "state")
    for k, v in part(s, "sidebar").items():
        if k in SIDEBAR and fits(SIDEBAR[k], v):
            ss[k] = as_saved(SIDEBAR[k], v)
    pics = part(s, "pictures")
    mine: set[str] = set()
    for name, (store, fields) in PICTURE_STORES.items():
        dst = ss.setdefault(store, {})
        for pk, rec in part(pics, name).items():
            if isinstance(rec, dict):
                dst[pk] = {f: as_saved(fields[f], v) for f, v in rec.items() if f in fields and fits(fields[f], v)}
                mine.add(pk)
    bars = ss.setdefault("barcol_pk", {})
    for pk, v in part(pics, "bar").items():
        if isinstance(v, str) and v in BAR_CHOICES:
            bars[pk] = v
            mine.add(pk)
    figs = [{"key": str(f.get("key")), "spec": str(f.get("spec") or ""), "spec_ok": str(f.get("spec_ok") or ""),
             "aspect": f.get("aspect") if isinstance(f.get("aspect"), str) else None,
             "order": {pk: int(n) for pk, n in f["order"].items() if isinstance(n, (int, float)) and not isinstance(n, bool)}}
            for f in part(s, "figures", list) if isinstance(f, dict) and isinstance(f.get("order"), dict)]
    for f in figs:
        mine |= set(f["order"])
    scales = {k: v for k, v in part(s, "scales").items() if isinstance(k, str) and isinstance(v, str)}
    ss.setdefault("scales_g", {}).update(scales)
    # figures showing these pictures are built again from the work file (order, layout, trimming shape)
    gone = {k for k, m in ss.get("tpks", {}).items() if mine & set(m.values())} | {f.get("key") for f in figs}
    for k in list(ss.keys()):
        m, mf = PICTURE_WIDGET.match(k), FIGURE_WIDGET.match(k)
        if (m and m.group(2) in mine) or (k.startswith("sc_") and k[3:] in scales) or (mf and mf.group(2) in gone):
            del ss[k]
    for store in ("tables", "bases", "tpks", "specs", "spec_ok", "crop_aspect"):
        d = ss.setdefault(store, {})
        for k in gone:
            d.pop(k, None)
    ver = ss.setdefault("edver", {})
    for k in gone:
        if k:
            ver[k] = ver.get(k, 0) + 1  # a fresh table widget
    keys = {f.get("key") for f in figs}  # (figures of a work file opened before stay, after these)
    ss["saved_figs"] = figs + [f for f in ss.get("saved_figs", []) if f.get("key") not in keys]
    shown = next((f for f in figs if f.get("key") == s.get("shown")), None)
    ss["_restore_fig"] = {"key": s.get("shown"), "pks": list(shown["order"]) if shown else []}
    ss["_shown"] = None


def expand_uploads(uploads: list) -> tuple[list, list[tuple[str, str]]]:
    """The files to work on.  A work file (.edsfig) is opened: its files join the others (a file also
    loaded on its own is taken once) and, the first time, its settings are put in place.  Also returns
    notes for the sidebar: (info | warning | error, text)."""
    ss = st.session_state
    works = [f for f in uploads if Path(f.name).suffix.lower() == P.EXT]
    plain = [f for f in uploads if Path(f.name).suffix.lower() != P.EXT]
    ids = [upload_id(f) for f in works]
    opened = ss.setdefault("_works", {})
    for k in [k for k in opened if k not in ids]:  # taken out of the file box
        del opened[k]
    ss["_applied"] = [k for k in ss.get("_applied", []) if k in ids]
    alive = {getattr(f, "file_id", None) for f in plain}
    ss["_md5"] = {k: v for k, v in ss.get("_md5", {}).items() if k in alive}
    notes, out, inside, settings_only = [], [], set(), []
    for f, k in zip(works, ids):
        if k not in opened:
            try:
                opened[k] = P.unpack(f.getvalue())
            except P.ProjectError as e:
                opened[k] = str(e)
        got = opened[k]
        if isinstance(got, str):
            notes.append(("error", f"{f.name}：{got}"))
            continue
        doc, contents = got
        if k not in ss["_applied"]:
            restore(doc)
            ss["_applied"].append(k)
        for name, data, h in contents:
            if h not in inside:
                inside.add(h)
                out.append(Blob(name, data, h))
        if doc.get("with_files"):
            notes.append(("info", f"📂 {f.name}：{doc.get('saved', '')} に保存した作業を開きました（ファイル {len(contents)} 件）"))
        else:
            settings_only.append((f, doc))
    for f in plain:
        if not inside or md5_of(f) not in inside:  # the same file inside a work file is taken once
            out.append(f)
    if settings_only:
        present = {md5_of(f) for f in out}
        for f, doc in settings_only:
            miss = P.missing(doc, present)
            if miss:
                notes.append(("warning", f"{f.name} は設定だけの作業ファイルです。元のファイルも入れてください："
                              + "、".join(miss[:6]) + (f" ほか {len(miss) - 6} 件" if len(miss) > 6 else "")))
            else:
                notes.append(("info", f"📂 {f.name}：{doc.get('saved', '')} に保存した設定を当てはめました"))
    return out, notes


def saved_figure(rkey: str, pks: list[str], exact: bool = False) -> dict | None:
    """The figure of the opened work file that this figure is: the same one, or (when the files came
    in a different order or with more files) the one sharing the most pictures."""
    figs = st.session_state.get("saved_figs") or []
    for f in figs:
        if f.get("key") == rkey:
            return f
    if exact:
        return None
    mine, best, n_best = set(pks), None, 0
    for f in figs:
        n = len(mine & set(f["order"]))
        if n > n_best:
            best, n_best = f, n
    return best


# --------------------------------------------------------------------------- sidebar: settings
sb = st.sidebar
sb.title(APP_NAME)
if SERVER:
    sb.caption("研究室の共有アプリです。入れたファイルはこのアプリを動かしている PC のメモリ上だけで処理し、"
               "保存も外部への送信もしません（作業の保存は自分の PC へのダウンロードです）。")
uploads = sb.file_uploader("AZtecのWord出力（.docx）やSEM・TEM像を入れてください", type=UPLOAD_TYPES,
                           accept_multiple_files=True,
                           help="Word出力はレポートごとに1つの図。SEM・TEM像（TIFF・PNG・JPEG・BMP、DigitalMicrograph の .dm3/.dm4）は"
                                "まとめて1つの図になります。SEMが画像と一緒に保存する .txt を入れると、その画素寸法を使います。"
                                "倍率や装置が違うファイルを一度に入れても、スケールはファイルごとに読み取ります。"
                                "JEOL EPMA のマップ画面を保存した画像は、Word 出力と同じく1枚ごとに1つの図になります。"
                                "保存した作業ファイル（.edsfig）を入れると、保存したときの続きから作業できます。")
files, upload_notes = expand_uploads(list(uploads or []))  # work files opened (their settings put in place once)
for level, text in upload_notes:
    (sb.caption if level == "info" else getattr(sb, level))(text)
save_box = sb.container()  # "作業の保存": filled in at the end of each run, with the settings as they are then
for k, (d, _) in SIDEBAR.items():  # every setting starts from its default (or from a work file)
    if k not in st.session_state:
        st.session_state[k] = d
SCREEN_TYPES = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")  # a saved screen (EPMA map window) is one of these
WINDOWS = {id(f) for f in files if Path(f.name).suffix.lower().lstrip(".") in SCREEN_TYPES and is_map_window(f.getvalue())}
_n_docx = sum(1 for f in files if Path(f.name).suffix.lower() == ".docx") + len(WINDOWS)  # one figure each
_n_img = sum(1 for f in files if Path(f.name).suffix.lower().lstrip(".") in IMAGE_TYPES and id(f) not in WINDOWS)
combine_all = toggle(sb, "全部を1つの図にまとめる", key="combine", disabled=_n_docx + (1 if _n_img else 0) < 2,
                     help="Word出力（EDS）とSEM・TEM像を1枚の図に並べます。Word出力ごとの画像は同じ視野として扱い"
                          "（視野幅の照合もその中だけ）、SEM・TEM像はそれぞれ独立にスケールを読みます。"
                          "オフのときは、Word出力ごとの図と、SEM・TEM像をまとめた図に分かれます。")

sb.subheader("出力サイズ")
size_mode = SIZE_MODES[sb.radio("決め方", list(SIZE_MODES), key="size_mode", horizontal=True,
                                help="画像は縦横同じ倍率でしか拡大縮小しません（縦横比は元画像のまま）。"
                                     "幅と高さの両方を指定すると、その枠に収まる最大の大きさで中央に配置し、余りは背景になります。")]
c1, c2 = sb.columns(2)
width_mm = c1.number_input("幅 (mm)", 20.0, 1000.0, step=5.0, key="width_mm", disabled=size_mode == "height",
                           help="片段 85 mm、両段 170 mm が目安")
height_mm = c2.number_input("高さ (mm)", 20.0, 1000.0, step=5.0, key="height_mm", disabled=size_mode == "width")

sb.subheader("配置")
free = ARRANGES[sb.radio("並べ方", list(ARRANGES), key="arrange",
                         help="自由配置：横並びの中に縦積み、その中に 2×2 など、入れ子の配置を番号で書きます"
                              "（「4. プレビュー」の上の欄）。図全体を1枚として組むので、"
                              "元素記号・スケールバーの文字とバーの太さは、どのパネルでも同じ大きさになります。")] == "free"
c1, c2 = sb.columns(2)
auto_cols = toggle(c2, "列数を自動", key="auto_cols", disabled=size_mode != "box" or free,
                   help="「幅と高さを指定」のとき、画像が一番大きくなる列数を選びます")
ncols = c1.number_input("列数", 1, 10, step=1, key="ncols", disabled=bool(auto_cols and size_mode == "box") or free,
                        help="自由配置では、配置の欄を「列数の並びから作り直す」ときに使います")
rows_text = sb.text_input("行ごとの枚数（空欄＝列数ずつ）", key="rows_text", disabled=free,
                          help="上の行から順に、1行に並べる枚数をカンマで区切って入れます。"
                               "例：5枚を 3列で「2,3」→ 上 2枚・下 3枚。「2」だけなら上 2枚、あとは列数ずつ。",
                          **opt(st.text_input, placeholder="例: 2,3"))
row_counts = parse_rows(rows_text)
if rows_text.strip() and row_counts is None:
    sb.caption("⚠ 1以上の数字をカンマで区切って入れてください（例: 2,3）。今は列数ずつ並べています。")
row_align = ROW_ALIGNS[sb.selectbox("枚数の少ない行の位置", list(ROW_ALIGNS), key="row_align", disabled=free,
                                    help="ほかの行より枚数の少ない行（例：2-3 の上の行）を、左寄せ・中央・右寄せのどこに置くか。"
                                         "自動＝「幅を揃える」「トリミング」では左寄せ、「高さを揃える」では中央")]
c1, c2 = sb.columns(2)
gap_mm = c1.number_input("間隔 (mm)", 0.0, 20.0, step=0.25, key="gap_mm")
margin_mm = c2.number_input("外余白 (mm)", 0.0, 30.0, step=0.5, key="margin_mm")
align = ALIGNS[sb.radio("形の違う画像の揃え方", list(ALIGNS), key="align", disabled=free,
                        help="幅を揃える：同じ幅で並べ、高さの違いは余白になります。"
                             "高さを揃える：同じ高さで横に並べ、幅は画像ごとの形のままです。"
                             "トリミングして揃える：全部の画像を同じ縦横比に切り取り、幅も高さも揃えます"
                             "（切り取る位置と範囲は画像ごとに「2.」で調整）。"
                             "どの方法でも、画像は縦横同じ倍率でしか拡大縮小しません。"
                             "自由配置では使いません（横並びは高さ、縦積みは幅が自動で揃います）。")]
if free:
    align = "free"
valign = VALIGNS[sb.selectbox("縦の位置（幅を揃えるとき）", list(VALIGNS), key="valign", disabled=align != "width",
                              help="背の低い画像を、マスの上・中央・下のどこに置くか")]
big = toggle(sb, "電子像を 2×2 で大きく", key="big", disabled=align in ("height", "free") or bool(row_counts),
             help="「幅を揃える」「トリミングして揃える」のときに使えます（行ごとの枚数を指定したときは使えません）")
bg = sb.radio("背景", list(BACKGROUNDS), key="bg", horizontal=True)

sb.subheader("ラベル（元素記号・サンプル名）")
label_on = toggle(sb, "表示する", key="label_on")
c1, c2 = sb.columns(2)
label_pt = c1.number_input("文字 (pt)", 4.0, 48.0, step=0.5, key="label_pt")
label_col = c2.selectbox("色", list(LABEL_COLORS), key="label_col",
                         help="重ね合わせ図の元素記号は、どの設定でも色見本（凡例）の色で入ります")
c1, c2 = sb.columns(2)
label_outline = toggle(c1, "縁取り", key="lo", help="白い文字は黒で、黒い文字は白で縁取ります")
label_corner = c2.selectbox("位置", list(CORNERS), key="label_corner")

sb.subheader("スケールバー")
bar_target = sb.radio("付ける画像", list(BAR_TARGETS), key="bar_target", horizontal=True)
c1, c2 = sb.columns(2)
bar_len = c1.number_input("長さ (µm)", 0.0, 100000.0, step=1.0, format="%g", key="bar_len",
                          help="0 = 画像ごとに、視野幅の約1/4〜1/3になる切りのよい長さを自動で選びます")
bar_corner = c2.selectbox("位置 ", list(BAR_CORNERS), key="bar_corner")
c1, c2 = sb.columns(2)
bar_col = c1.selectbox("色 ", list(BAR_COLORS), key="bar_col",
                       help="自動＝バーと数値を置く場所が明るい画像には黒、暗い画像には白。"
                            "画像ごとに変えるときは「2. 画像ごとの調整」で選びます")
bar_pt = c2.number_input("文字 (pt) ", 4.0, 48.0, step=0.5, key="bar_pt")
c1, c2 = sb.columns(2)
bar_text = toggle(c1, "数値を表示", key="bar_text")
bar_outline = toggle(c2, "縁取り ", key="bo", help="白いバーは黒で、黒いバーは白で縁取ります")
bar_thick = sb.slider("太さ (pt)", 0.5, 8.0, step=0.25, key="bar_thick_pt",
                      help="どのパネルでも同じ太さになります（画像の大きさによりません）")

sb.subheader("画素")
res_mode = sb.radio("解像度", RES_MODES, key="res_mode", horizontal=True,
                    help="整数倍：最も細かい画像の1画素を N×N 画素で出力。縮小は起きません。")
c1, c2 = sb.columns(2)
oversample = c1.number_input("倍率 N", 1, 8, step=1, key="oversample", disabled=res_mode != RES_MODES[0])
dpi = c2.number_input("dpi", 72, 2400, step=50, key="dpi", disabled=res_mode != RES_MODES[1])
rot_fill = ROT_FILLS[sb.radio("斜めに回転したときの四隅", list(ROT_FILLS), key="rot_fill", horizontal=True,
                              help="90°単位以外の回転でできる四隅の空き。切り取る＝空きの無い最大の長方形を使います")]
rot_resample = ROT_RESAMPLES[sb.radio("回転の補間（90°単位以外）", list(ROT_RESAMPLES), key="rot_resample", horizontal=True,
                                      help="90°単位の回転と反転は画素を並べ替えるだけで、補間はしません。"
                                           "それ以外の角度は画素を作り直します。最近傍は元の画素値をそのまま使います（輪郭はギザギザ）")]
interp = sb.radio("拡大の仕方（PNG/TIFF）", list(INTERPS), key="interp", horizontal=True,
                  help="ドットのまま＝最近傍（EDSマップの画素がそのまま見える）。PDF/SVGは常に元画素をそのまま埋め込みます。")
learn = toggle(sb, "確認した読み取りを記憶する", True,
               help="書き出し時に、確認済みの元素記号・スケール値と元の文字の形を記憶し、次回の読み取りに使います（このPC内のみ）。")
if DESKTOP == "browser":  # the PC app opened in the browser ends from here (in its own window: closing the window)
    sb.divider()
    if st.session_state.get("_quit_ask"):
        sb.warning("終了しますか？保存していない作業は消えます。")
        q1, q2 = sb.columns(2)
        if q1.button("終了する", type="primary"):
            st.info("終了しました。このタブは閉じてかまいません。")
            threading.Timer(0.8, os._exit, args=(0,)).start()  # after this page has been sent
            st.stop()
        if q2.button("やめる"):
            st.session_state["_quit_ask"] = False
            st.rerun()
    elif sb.button("アプリを終了", help="このアプリを閉じます（ブラウザのタブを閉じても、しばらくすると終わります）"):
        st.session_state["_quit_ask"] = True
        st.rerun()

style = F.Style(
    ncols=int(ncols), electron_span=2 if (big and align in ("width", "crop") and not row_counts) else 1,
    size_mode=size_mode,
    width_mm=float(width_mm), row_counts=row_counts, row_align=row_align,
    height_mm=float(height_mm), auto_cols=bool(auto_cols), gap_mm=float(gap_mm), align=align, valign=valign,
    margin_mm=float(margin_mm), background=BACKGROUNDS[bg],
    label_on=label_on, label_pt=float(label_pt), label_color=LABEL_COLORS[label_col],
    label_outline=label_outline, label_corner=CORNERS[label_corner],
    scalebar_on=BAR_TARGETS[bar_target], scalebar_um=float(bar_len) or None, scalebar_corner=BAR_CORNERS[bar_corner],
    scalebar_color=BAR_COLORS[bar_col], scalebar_text=bar_text, scalebar_pt=float(bar_pt),
    scalebar_thick_pt=float(bar_thick), scalebar_outline=bar_outline,
    oversample=int(oversample), dpi=float(dpi) if res_mode == RES_MODES[1] else None,
    interpolation=INTERPS[interp],
)

# --------------------------------------------------------------------------- main
# Edits live here, per report.  The table widget is given the same data for as long as it stays on
# screen (older Streamlit rebuilds the widget, dropping in-progress edits, whenever its data changes).
TABLES = st.session_state.setdefault("tables", {})  # latest edited table
BASES = st.session_state.setdefault("bases", {})  # data handed to the table widget while shown
SCALES = st.session_state.setdefault("scales_g", {})  # scale values by label ink, typed field widths by picture

if not files:
    st.session_state["_shown"] = None
    st.title(APP_NAME)
    st.caption(APP_TAGLINE)
    st.markdown(
        "AZtec の Word 出力（`.docx`）や SEM・TEM 像を左に入れると、画像を並べた論文用の図を作ります。\n\n"
        "- AZtec：タイトル・スケールバー・余白を外して**画像本体だけ**を切り出し、各マップの左上に**元素記号**を入れます\n"
        "- SEM・TEM像：下の**情報帯（倍率・日付・スケール）を外し**、帯のスケールから画素寸法を求めます。左上に**サンプル名**を入れられます。"
        "文字が画像に直接重ねて書かれた形式（JEOL など）は、文字のある下端の行を切り落とします\n"
        "- JEOL EPMA：マップ画面（青緑の地に WDS マップが並んだ画面）を保存した画像から、マップを1枚ずつ切り出し、"
        "下の白い枠の元素名とスケールバーを読みます\n"
        "- TEM：DigitalMicrograph の `.dm3` / `.dm4` をそのまま読み、記録された画素寸法を使います。"
        "TEM-EDS のマップは左のカラーバーを外し、帯の元素名（「C K」など）を読みます\n"
        "- **新しいスケールバー**を付け、列数・出力サイズ・文字サイズを決めて書き出します"
        "（PNG / TIFF は可逆、PDF / SVG は元画素を埋め込み・文字は編集可）\n"
        "- 画像は縦横同じ倍率でしか拡大縮小しません（**縦横比は元画像のまま**）\n"
        "- 左の「作業の保存」で、読み込んだファイルと設定（並び順・ラベル・調整・配置など）を1つの**作業ファイル（.edsfig）**に"
        "保存できます。その作業ファイルを入れると、続きから作業できます\n\n"
        + ("研究室の PC で動いている共有アプリです。入れたファイルはその PC のメモリ上だけで処理され、外部には送られず、"
           "ほかの人からも見えません。図と作業ファイルは自分の PC にダウンロードされます。" if SERVER else
           "ネット接続は使いません。データはこのPCの中だけで処理されます。")
    )
    st.stop()

# reports: every .docx is one figure; loose images together make one figure
docs, imgs, txts, windows = [], [], [], []
for f in files:
    ext = Path(f.name).suffix.lower().lstrip(".")
    (docs if ext == "docx" else txts if ext == "txt" else windows if id(f) in WINDOWS else imgs).append(f)
reports = []  # (display name, kind, key, payload)
for f in docs:
    data = f.getvalue()
    reports.append((f.name, "docx", "d" + hashlib.md5(data).hexdigest()[:10], (data, f.name)))
for f in windows:  # a JEOL EPMA map window: its maps make one figure, like a Word report
    data = f.getvalue()
    reports.append((f.name, "sheet", "s" + hashlib.md5(data).hexdigest()[:10], (data, f.name)))
if imgs:
    blobs = tuple(f.getvalue() for f in imgs)
    inames = tuple(f.name for f in imgs)
    tx = tuple((Path(f.name).stem, f.getvalue().decode("utf-8", "ignore")) for f in txts)
    h = hashlib.md5(b"".join(hashlib.md5(b).digest() for b in blobs) + "|".join(inames).encode())
    n_dm = sum(1 for f in imgs if Path(f.name).suffix.lower() in (".dm3", ".dm4"))
    title = f"{'TEM' if n_dm == len(imgs) else 'SEM・TEM' if n_dm else 'SEM'}像 {len(imgs)} 枚"
    reports.append((title, "images", "i" + h.hexdigest()[:10], (blobs, inames, tx, title)))
if combine_all and len(reports) > 1:  # everything in one figure
    parts = tuple((rk, pl) for _, rk, _, pl in reports)
    hm = hashlib.md5("|".join(k for _, _, k, _ in reports).encode()).hexdigest()[:10]
    parts_t = ([f"Word {len(docs)} 件"] if docs else []) + ([f"EPMA {len(windows)} 件"] if windows else []) \
        + ([f"画像 {len(imgs)} 枚"] if imgs else [])
    mtitle = "まとめた図（" + "・".join(parts_t) + "）"
    reports = [(mtitle, "mixed", "m" + hm, (parts, mtitle))]
if not reports:
    st.session_state["_shown"] = None
    st.info("画像か Word 出力を入れてください（.txt だけでは図になりません）。")
    st.stop()


@st.cache_data(show_spinner="1つの図にまとめています…", max_entries=8, ttl=TTL)
def parse_mixed(parts: tuple, title: str) -> aztec.Report:
    return aztec.combine([parse_docx(*pl) if rk == "docx" else parse_sheet(*pl) if rk == "sheet" else parse_images(*pl)
                          for rk, pl in parts], title)


def load(kind: str, payload) -> aztec.Report:
    if kind == "docx":
        return parse_docx(*payload)
    if kind == "sheet":
        return parse_sheet(*payload)
    if kind == "mixed":
        return parse_mixed(*payload)
    return parse_images(*payload)


names: list[str] = []  # unique display names (two uploads may share a file name)
for n, *_ in reports:
    lab, k = n, 2
    while lab in names:
        lab, k = f"{n} ({k})", k + 1
    names.append(lab)
want = st.session_state.pop("_restore_fig", None)  # an opened work file: the figure shown when it was saved
if len(reports) > 1:
    keys_now = [k for _, _, k, _ in reports]
    if want:
        target = want.get("key") if want.get("key") in keys_now else None
        if target is None and want.get("pks"):  # (its files loaded in another order: the most pictures shared)
            wp, n_best = set(want["pks"]), 0
            for _, rk2, k2, pl2 in reports:
                n_shared = len(wp & set(picture_keys(load(rk2, pl2))))
                if n_shared > n_best:
                    target, n_best = k2, n_shared
        if target is not None:
            st.session_state["fig"] = names[keys_now.index(target)]
    if st.session_state.get("fig") not in names:
        st.session_state.pop("fig", None)
    choice = st.selectbox("図", names, key="fig")
else:
    choice = names[0]
name, rkind, key, payload = reports[names.index(choice)]
rep = load(rkind, payload)
def figure_stem(name: str, kind: str) -> str:
    """File name stem: the Word file's name, "combined", or SEM / TEM / SEM_TEM for images."""
    if kind in ("docx", "sheet"):
        return Path(name).stem
    if kind == "mixed":
        return "combined"
    return name.split("像")[0].replace("・", "_")


stem = figure_stem(name, rkind)


def current_pictures() -> set[str]:
    """Picture keys of everything loaded (what a work file keeps settings for)."""
    sig = tuple(k for _, _, k, _ in reports)
    got = st.session_state.get("_curpics")
    if got and got[0] == sig:
        return got[1]
    pks = set()
    for _, rk2, k2, pl2 in reports:
        pks |= set(picture_keys(rep if k2 == key else load(rk2, pl2)))
    st.session_state["_curpics"] = (sig, pks)
    return pks


def gather_state() -> dict:
    """Every setting, as it is now, for a work file (only those of the pictures loaded now)."""
    ss = st.session_state
    cur_pks = current_pictures()
    pictures = {n: {pk: dict(v) for pk, v in (ss.get(store) or {}).items() if pk in cur_pks}
                for n, (store, _) in PICTURE_STORES.items()}
    pictures["bar"] = {pk: v for pk, v in (ss.get("barcol_pk") or {}).items() if pk in cur_pks}
    tables, tpks = ss.get("tables", {}), ss.get("tpks", {})
    specs, spec_ok, aspect = ss.get("specs", {}), ss.get("spec_ok", {}), ss.get("crop_aspect", {})
    figures = []
    for rkey, t in tables.items():
        tp = tpks.get(rkey)
        if not tp or not set(tp.values()) <= cur_pks:
            continue
        order = {tp[i]: cell_int(t.at[i, "order"]) for i in t.index if i in tp and cell_int(t.at[i, "order"])}
        figures.append({"key": rkey, "order": order, "spec": specs.get(rkey) or "",
                        "spec_ok": spec_ok.get(rkey) or "", "aspect": aspect.get(rkey)})
    scales = {k: v for k, v in (ss.get("scales_g") or {}).items()  # by label ink (any picture), or by picture
              if not k.startswith(("fw|", "bar|")) or k.split("|", 1)[1] in cur_pks}
    return {"sidebar": {k: ss.get(k, d) for k, (d, _) in SIDEBAR.items()}, "pictures": pictures,
            "scales": scales, "figures": figures, "shown": key}


def size_text(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{max(1, round(n / 1e3))} KB"


def save_panel() -> None:
    """作業の保存 (in the sidebar, under the file box): the loaded files as they are, and every setting."""
    state = gather_state()
    st.session_state["_last_state"] = state  # (read by the tests)
    loaded = [(f.name, f.getvalue()) for f in files]
    total = sum(len(b) for _, b in loaded)
    when = datetime.now().strftime("%m%d-%H%M")
    full_name, conf_name = f"{stem}_作業_{when}{P.EXT}", f"{stem}_設定_{when}{P.EXT}"
    mime = "application/octet-stream"  # (a .zip type would make some browsers unpack it)
    with save_box.expander("作業の保存（続きから再開）"):
        st.caption("読み込んだファイル（元のまま・無圧縮）と、表の並び順・使う・ラベル、明るさ・回転・トリミング、配置、"
                   "スケール、左の設定を、1つの作業ファイル（.edsfig）に保存します。"
                   "次回は上の欄にその作業ファイルを入れるだけで、続きから作業できます。")
        if DEFERRED_DOWNLOAD:
            sums = None
            st.download_button(f"作業を保存（{size_text(total)}）", data=partial(P.pack, loaded, state, True, APP_VERSION),
                               file_name=full_name, mime=mime, **opt(st.download_button, type="primary"),
                               **FW_DOWNLOAD)
        else:  # older Streamlit: build the file first, then save it
            sums = [md5_of(f) for f in files]
            sig = hashlib.md5(json.dumps(state, sort_keys=True, ensure_ascii=False, default=str).encode()
                              + "|".join(sums).encode()).hexdigest()
            if st.button(f"作業ファイルを作る（{size_text(total)}）", **opt(st.button, type="primary"), **FW_BUTTON):
                with st.spinner("作業ファイルを作っています…"):
                    st.session_state["work"] = {"sig": sig, "name": full_name,
                                                "data": P.pack(loaded, state, True, APP_VERSION, sums)}
            w = st.session_state.get("work")
            if w and w["sig"] == sig:
                st.download_button(f"保存：{w['name']}", data=w["data"], file_name=w["name"], mime=mime,
                                   **opt(st.download_button, type="primary"), **FW_DOWNLOAD)
            else:
                st.session_state.pop("work", None)  # settings changed: the file made before is let go
                st.caption("「作業ファイルを作る」を押すと保存ボタンが出ます（設定を変えたら作り直してください）。")
        st.caption(f"大きさは元のファイルの合計とほぼ同じ（{size_text(total)}）です。元のファイルを手元に残しておくなら、"
                   "設定だけの小さなファイルでも再開できます（そのときは元のファイルと一緒に入れます）。")
        try:
            limit_mb = int(st.get_option("server.maxUploadSize"))
        except Exception:  # (an option missing in some version)
            limit_mb = 0
        if limit_mb and total > limit_mb * 1024 * 1024:
            st.warning(f"作業ファイルが {size_text(total)} になり、読み込める上限（{limit_mb} MB）を超えます。"
                       "「設定だけ保存」を使うか、アプリのフォルダの .streamlit/config.toml の maxUploadSize を大きくしてください。")
        st.download_button("設定だけ保存", file_name=conf_name, mime=mime, **FW_DOWNLOAD,
                           data=partial(P.pack, loaded, state, False, APP_VERSION) if DEFERRED_DOWNLOAD
                           else P.pack(loaded, state, False, APP_VERSION, sums))


def halt() -> None:
    """Stop here (nothing more to show), keeping the save buttons."""
    save_panel()
    st.stop()


st.header(name)
if not rep.panels:
    st.session_state["_shown"] = None
    st.error("画像が見つかりませんでした。")
    st.stop()

# ---- 1. panels -------------------------------------------------------------
st.subheader("1. 画像とラベルの確認")
if any("/tile" in p.src.uid for p in rep.panels):
    st.caption("JEOL EPMA のマップ画面：タイルごとにマップを切り出し（画素はそのまま）、下の白い枠の名前（元素記号・SED・BED-C など）と"
               "スケールバー（「20 µm」など）を読んでいます。SED・BED などは電子像になります。"
               "右側の色スケール・測定条件は使いません。")
if rep.kind == "mixed":
    st.caption("Word出力（EDS）の画像とSEM・TEM像を1つの図にまとめています。マップのラベルには元素記号、"
               "SEM・TEM像の「ラベル」にはサンプル名などを入れます。Word出力ごとの画像は同じ視野として視野幅を照合します。")
if rep.kind in ("images", "mixed"):
    st.caption("「ラベル」にサンプル名を入れると、画像の左上（サイドバーで位置を変更可）に入ります。"
               "「帯 (px)」は取り除いた情報帯の高さです。文字が画像に重ねて書かれた形式では、文字のある下端の行数です"
               "（0にすると切らずにそのまま使います）。")
else:
    st.caption("「元の表示」を見て、ラベル（左上に入る文字）が正しいか確認してください。電子像の行にサンプル名を入れることもできます。"
               "使わない画像はチェックを外します。スペクトルなど図に使わない画像は、最初からチェックを外してあります。")
if free:
    st.caption("自由配置：図の並びは「4. プレビュー」の上の配置の欄で、この「順」の番号を使って書きます。"
               "「順」の番号を入れ替えると、配置の欄のその番号の場所に入る画像が入れ替わります。"
               "「使う」を外した画像は一番下に移り、配置の欄の番号もそれに合わせて書き換わります。")
else:
    st.caption("並び順：図は「順」の番号の順に、左上から右へ、次の行へと並びます。「順」に番号を入れるとその位置へ移動し、"
               "ほかの画像は1つずつずれます（例：3番目の画像に 1 と入れると先頭へ）。下の「2.」のカードの「◀ 前へ」「後へ ▶」でも動かせます。"
               "「使う」を外した画像は一番下に移り、もう一度チェックすると使う画像の最後に入ります。")
PK = picture_keys(rep)  # each picture's key: its settings follow it when files are added or removed
if st.session_state.get("_shown") != key or key not in BASES:  # (re)appearing on screen
    BASES[key] = TABLES[key] = table_for(key, rep, PK, adopt=True)
st.session_state["_shown"] = key
column_config = {
    "use": st.column_config.CheckboxColumn("使う", width="small"),
    "order": st.column_config.NumberColumn("順", min_value=1, max_value=99, step=1, width="small"),
    "thumb": st.column_config.ImageColumn("画像", width="small"),
    "title": st.column_config.ImageColumn("元の表示", width="medium"),
    "label": st.column_config.TextColumn("ラベル（サンプル名など）" if rep.kind != "aztec" else "ラベル（左上）",
                                         width="medium" if rep.kind != "aztec" else "small"),
    "status": st.column_config.TextColumn("読み取り", width="small",
                                          help="✓ = 確実、要確認 = 候補が拮抗、要入力 = 読めなかった"),
    "kind": st.column_config.SelectboxColumn("種類", options=list(KIND_JA.values()), width="small"),
}
if "band" in BASES[key].columns:  # loose pictures, or a layered image with its colour key
    column_config["band"] = st.column_config.NumberColumn("帯 (px)", min_value=0, max_value=5000, step=1, width="small",
                                                          help="取り除く情報帯（または文字・色見本が重なった下端）の高さ（画素）。0 = 切らない")
EDVER = st.session_state.setdefault("edver", {})
table = st.data_editor(
    BASES[key],
    key=f"ed_{key}_{EDVER.get(key, 0)}",  # a new widget whenever the order is renumbered
    hide_index=True,
    column_config=column_config,
    disabled=["thumb", "title", "status"],
    **opt(st.data_editor, row_height=64),
    **FW_EDITOR,
)
ro = renumbered(table, BASES[key])
if ro is not None:  # a number typed in "順": move the picture there and number all 1..n again
    ro = used_first(ro)
else:
    sw_on, sw_off = switched(table, BASES[key])
    if sw_on or sw_off:  # switched off: to the very bottom; switched on: after the others in use
        ro = used_first(table, sw_on, sw_off)
        renumber_spec(key, table, ro)
if ro is not None:
    TABLES[key] = BASES[key] = ro
    EDVER[key] = EDVER.get(key, 0) + 1
    remember_rows(ro, PK)
    st.rerun()
TABLES[key] = table
remember_rows(table, PK)
st.session_state.setdefault("tpks", {})[key] = {i: PK[i] for i in table.index}
apply_bands(rep, table)  # the band settings of the table apply to everything below

# ---- 2. brightness / contrast (and trimming) --------------------------------
st.subheader("2. 画像ごとの調整")
st.caption("明るさ・コントラスト・スケールバーの色・回転・トリミングを画像ごとに調整できます。"
           "明るさ等の調整は画像全体に一様にかかり、設定は書き出す較正情報（calibration.json）に記録されます。"
           "自動＝レベル補正（白黒の像は上下0.5%を切り詰めて全階調に、カラーのマップは黒を保ったまま明るい0.5%を最大に）。"
           "コントラストは画像の平均の明るさを中心に変わります（+100%で2倍）。カラーのマップは色相を変えずに明るさだけが変わります。")
# per-picture settings are kept by picture key: they follow the picture into any figure
ADJ = st.session_state.setdefault("adj_pk", {})  # brightness / contrast
BARCOL = st.session_state.setdefault("barcol_pk", {})  # scale bar colour
ROTS = st.session_state.setdefault("rot_pk", {})  # turning / mirroring
CROPS = st.session_state.setdefault("crop_pk", {})  # "トリミングして揃える": zoom and place
TRIMS = st.session_state.setdefault("trims_pk", {})  # trimming of one picture
CROP_ASPECT = st.session_state.setdefault("crop_aspect", {})  # "トリミングして揃える": the shape, per figure
in_use = table[table["use"].map(lambda v: cell_text(v) in ("True", "true", "1"))].sort_values("order", kind="stable")
shown = [int(i) for i in in_use.index]
for i in shown:
    for n, d in ROT_DEFAULT.items():  # set once; the widgets then keep what the user chooses
        if rot_key(n, PK[i]) not in st.session_state:
            st.session_state[rot_key(n, PK[i])] = ROTS.get(PK[i], {}).get(n, d)
    ROTS[PK[i]] = {n: st.session_state[rot_key(n, PK[i])] for n in ROT_DEFAULT}
ROTSET = {"per": by_index(ROTS, PK), "fill": rot_fill, "resample": rot_resample}
apply_rotations(rep, shown, ROTSET)
grey_of = {i: F.is_grey(rep.panels[i].image) for i in shown}
CROPSET = {"aspect": CROP_ASPECT.get(key), "per": {}}
boxes: dict[int, tuple] = {}
if align == "crop":
    opts = aspect_options(rep)
    akey = f"ca_{key}"
    if st.session_state.get(akey) not in opts:  # set once; the widget then keeps the choice
        st.session_state[akey] = CROPSET["aspect"] if CROPSET["aspect"] in opts else opts[0]
    choice = st.selectbox("トリミング後の縦横比（全画像で共通）", opts, key=akey,
                          help="自動＝全画像の切り取りが最も少なくなる形（どれか1枚は切り取られません）。"
                               "「○○と同じ形」を選ぶと、その画像はそのままで、ほかの画像をその形に切り取ります。")
    CROP_ASPECT[key] = CROPSET["aspect"] = choice
    st.caption("黄色の枠の中が図に使う部分です。「拡大」で範囲を狭め、「横位置」「縦位置」で切り取る場所を動かせます。"
               "切り取りは元の画素をそのまま使い、拡大縮小や変形はしません。スケールバーは切り取った後の視野に合わせて付きます。")
    for i in shown:
        for n, d in CROP_DEFAULT.items():  # set once; the sliders then keep what the user chooses
            if crop_key(n, PK[i]) not in st.session_state:
                st.session_state[crop_key(n, PK[i])] = CROPS.get(PK[i], {}).get(n, d)
        CROPS[PK[i]] = {n: st.session_state[crop_key(n, PK[i])] for n in CROP_DEFAULT}
    CROPSET["per"] = by_index(CROPS, PK)
    boxes = crop_boxes(rep, shown, choice, CROPSET["per"])
if align != "crop":  # each picture trimmed on its own (the card's "トリミング")
    for i in shown:
        for n, d in TRIM_DEFAULT.items():  # set once; the widgets then keep what the user chooses
            if trim_key(n, PK[i]) not in st.session_state:
                st.session_state[trim_key(n, PK[i])] = TRIMS.get(PK[i], {}).get(n, d)
        TRIMS[PK[i]] = {n: st.session_state[trim_key(n, PK[i])] for n in TRIM_DEFAULT}
    boxes = trim_boxes(rep, shown, by_index(TRIMS, PK))
    trim_note = st.empty()  # always there: the cards below keep their place (an open expander stays open)
    if boxes:
        trim_note.caption("黄色の枠の中が図に使う部分です（各カードの「トリミング」）。切り取りは元の画素をそのまま使い、"
                          "拡大縮小や変形はしません。スケールバーは切り取った後の視野に合わせて付きます。")
PER_ROW = 4
for start in range(0, len(shown), PER_ROW):
    cols = st.columns(PER_ROW)
    for c, i in zip(cols, shown[start: start + PER_ROW]):
        p, pk = rep.panels[i], PK[i]
        for n, d in ADJ_DEFAULT.items():  # set once; the widgets then keep what the user chooses
            if adj_key(n, pk) not in st.session_state:
                st.session_state[adj_key(n, pk)] = ADJ.get(pk, {}).get(n, d)
        a = to_adjust({n: st.session_state[adj_key(n, pk)] for n in ADJ_DEFAULT})
        thumb = F.adjusted(p.image, a)
        if i in boxes:
            thumb = crop_picture(thumb, boxes[i])
        if thumb.shape[1] > 400:
            im = Image.fromarray(thumb)
            thumb = np.asarray(im.resize((400, max(1, round(im.height * 400 / im.width))), RESAMPLE.LANCZOS))
        pos = shown.index(i) + 1
        name_i = f"{cell_int(table.at[i, 'order']) or pos}. " + (cell_text(table.at[i, "label"]) or picture_name(rep, p))
        if p.turned:
            name_i += f"（{transform.describe(p.rotation, p.flip_h, p.flip_v)}）"
        if i in boxes:
            bx = boxes[i]
            name_i += f"（{bx[2] - bx[0]}×{bx[3] - bx[1]} px を使用）"
        c.image(thumb, caption=name_i, **FW_IMAGE)
        m1, m2 = c.columns(2)
        m1.button("◀ 前へ", key=f"ml_{key}_{i}", on_click=move_picture, args=(key, i, -1), disabled=pos == 1, **FW_BUTTON)
        m2.button("後へ ▶", key=f"mr_{key}_{i}", on_click=move_picture, args=(key, i, 1), disabled=pos == len(shown),
                  **FW_BUTTON)
        toggle(c, "自動（レベル補正）", key=adj_key("auto", pk))
        c.slider("明るさ (%)", -100, 100, step=1, key=adj_key("brightness", pk))
        c.slider("コントラスト (%)", -100, 200, step=5, key=adj_key("contrast", pk))
        c.slider("γ（中間調）", 0.3, 3.0, step=0.05, key=adj_key("gamma", pk))
        b1, b2 = c.columns(2)
        b1.button("元に戻す", key=f"rs_{key}_{i}", on_click=reset_adjust, args=(pk,))
        same = [PK[j] for j in shown if grey_of[j] == grey_of[i] and j != i]
        b2.button("同じ種類に適用", key=f"cp_{key}_{i}", on_click=copy_adjust, args=(pk, same),
                  disabled=not same, help="この設定を、白黒の像どうし（またはカラーのマップどうし）すべてに写します")
        ADJ[pk] = {n: st.session_state[adj_key(n, pk)] for n in ADJ_DEFAULT}
        bkey = f"bc_{pk}"
        if bkey not in st.session_state:  # set once; the widget then keeps the choice
            st.session_state[bkey] = BARCOL.get(pk, "共通")
        no_bar = style.scalebar_on == "none" or (style.scalebar_on == "electron" and p.kind != "electron")
        c.radio("スケールバー・数値の色", list(BAR_CHOICES), key=bkey, horizontal=True, disabled=no_bar,
                help="共通 = サイドバー「スケールバー」の色。"
                     + ("この画像にはスケールバーを付けない設定です（サイドバー「付ける画像」）" if no_bar else ""))
        BARCOL[pk] = st.session_state[bkey]
        with c.expander("回転・反転"):
            st.number_input("回転（時計回り・度）", -360.0, 360.0, step=0.5, format="%.1f", key=rot_key("angle", pk),
                            help="90・180・270° の回転と反転は、画素を並べ替えるだけです（劣化なし）。"
                                 "それ以外の角度は補間で画素を作り直します（方法は左の「画素」で選択）。"
                                 "スケールは回転前の画像から読み取った値をそのまま使います（回転しても1画素の長さは同じ）")
            toggle(st, "左右反転", key=rot_key("fh", pk))
            toggle(st, "上下反転", key=rot_key("fv", pk))
            st.button("全画像に同じ回転・反転", key=f"rc_{key}_{i}", on_click=copy_rotation,
                      args=(pk, [PK[j] for j in shown if j != i]), disabled=len(shown) < 2)
        if align != "crop":
            with c.expander("トリミング"):
                shape = TRIM_SHAPES[st.selectbox("切り取る形", list(TRIM_SHAPES), key=trim_key("shape", pk),
                                                 help="形を選んで「拡大」「横位置」「縦位置」で範囲を決めるか、"
                                                      "「範囲を指定」で上下左右の端を直接決めます")]
                if shape == "free":
                    st.slider("横の範囲 (%)", 0.0, 100.0, step=0.5, key=trim_key("xr", pk),
                              help="使う部分の左端と右端（0 = 画像の左端、100 = 右端）")
                    st.slider("縦の範囲 (%)", 0.0, 100.0, step=0.5, key=trim_key("yr", pk),
                              help="使う部分の上端と下端（0 = 画像の上端、100 = 下端）")
                elif shape != "none":
                    bx = boxes.get(i, (0, 0, *p.size))
                    st.slider("拡大（範囲を狭める）", 1.0, 5.0, step=0.1, key=trim_key("zoom", pk),
                              help="1.0 = その形で切り取れる最大の範囲。2.0 で幅・高さとも半分の範囲になります")
                    st.slider("横位置 (%)", 0, 100, step=1, key=trim_key("fx", pk), disabled=bx[2] - bx[0] >= p.size[0],
                              help="0 = 左端、50 = 中央、100 = 右端")
                    st.slider("縦位置 (%)", 0, 100, step=1, key=trim_key("fy", pk), disabled=bx[3] - bx[1] >= p.size[1],
                              help="0 = 上端、50 = 中央、100 = 下端")
                t1, t2 = st.columns(2)
                t1.button("切り取りを戻す", key=f"tr_{key}_{i}", on_click=reset_trim, args=(pk,))
                views = [PK[j] for j in same_view(rep, i, shown)]
                t2.button("同じ視野の画像に適用", key=f"tc_{key}_{i}", on_click=copy_trim, args=(pk, views),
                          disabled=not views,
                          help="同じWord出力の電子像とマップ、または同じ画素数の画像すべてに、同じ切り取りを写します"
                               "（位置と範囲は画像に対する割合なので、画素数の違うマップでも同じ場所になります）")
        if align == "crop":
            if i in boxes:
                x0, y0, x1, y1 = boxes[i]
                W_i, H_i = p.size
                c.slider("拡大（範囲を狭める）", 1.0, 5.0, step=0.1, key=crop_key("zoom", pk),
                         help="1.0 = その形で切り取れる最大の範囲。2.0 で幅・高さとも半分の範囲になります")
                c.slider("横位置 (%)", 0, 100, step=1, key=crop_key("fx", pk), disabled=x1 - x0 >= W_i,
                         help="0 = 左端、50 = 中央、100 = 右端")
                c.slider("縦位置 (%)", 0, 100, step=1, key=crop_key("fy", pk), disabled=y1 - y0 >= H_i,
                         help="0 = 上端、50 = 中央、100 = 下端")
                c.button("トリミングを戻す", key=f"cr_{key}_{i}", on_click=reset_crop, args=(pk,))
            else:
                c.caption("この画像は切り取りません（スペクトルなど）")

# ---- 3. scale ---------------------------------------------------------------
st.subheader("3. スケールの確認")
groups = scale_groups(rep)
scale_text: dict[str, str] = {}
fields: dict[int, str] = {}
if not groups and not needs_field(rep) and not any(p.src.meta_um_per_px for p in rep.panels):
    st.info("スケールの情報が見つかりませんでした。")
for gk, idx in groups.items():
    p0 = rep.panels[idx[0]]
    sb0 = p0.scalebar
    c1, c2, c3 = st.columns([1.6, 1.2, 3])
    if p0.band_preview is not None:
        c1.image(p0.band_preview, caption=f"元の表示（{p0.src.name}）", **FW_IMAGE)  # fits the column
    elif p0.scale_label_preview() is not None:
        c1.image(p0.scale_label_preview(), caption="元の表示", width=140)
    ok = confident(sb0.score, sb0.margin) if sb0.text else False
    wkey = f"sc_{gk}"  # the same label ink: the same value in every figure
    if wkey not in st.session_state:  # set once; the widget then keeps what the user types
        st.session_state[wkey] = SCALES.get(gk, pretty_len(sb0.text))
    state = "✓" if ok else ("要確認" if sb0.text else "読めませんでした。数値を入力")
    val = c2.text_input(f"スケール値（{state}）", key=wkey, help="例: 5 µm / 250 nm / 1 mm（um でも可）")
    scale_text[gk] = val
    SCALES[gk] = val
    v, u = textread.parse_length(val)
    if v is None:
        c3.error("数値と単位を読み取れません（例: 5 µm）")
    else:
        per = {}  # bar length -> pictures (AZtec: one line for the whole set; SEM: one line per picture)
        for i in idx:
            p = rep.panels[i]
            who = cell_text(table.at[i, "label"]) or picture_name(rep, p)
            k = (p.set_id, round(p.scalebar.length_px, 1)) if (rep.kind == "aztec" or p.set_id) else i
            per.setdefault(k, (p.scalebar.length_px, []))[1].append(who)
        lines = []
        for L, whos in per.values():
            tile = any("/tile" in rep.panels[i].src.uid for i in idx)  # EPMA bars are short and whole pixels
            acc = f"（±{100 / L:.1f}%）" if (rep.kind != "aztec" or tile) and L < 100 else ""
            lines.append(f"{'、'.join(whos)} ／ バー {L:.1f} px{acc} → 1画素 = {v * textread.UNIT_TO_UM[u] / L * 1000:.2f} nm")
        c3.caption("  \n".join(lines))
for i, p in enumerate(rep.panels):
    if p.src.meta_um_per_px and p.kind not in ("other", "spectrum"):
        st.caption(f"{p.src.name}：機器の記録から 1画素 = {p.src.meta_um_per_px * 1000:.3f} nm（スケールバーより優先）")
    if p.src.note:
        st.caption(f"{p.src.name}：{p.src.note}")
    if p.band is not None and getattr(p.band, "legend", False) and p.band_px:
        st.caption(f"{picture_name(rep, p)}（{p.src.name}）：重ね合わせ図の色見本（凡例）が画像に描き込まれているため、"
                   f"{'下端' if p.band.side == 'bottom' else '上端'} {p.band_px} px を切り落とし、"
                   f"元素記号を色見本の色で左上に入れます（{'・'.join(t for t, _ in p.layers) or '読めませんでした'}）。"
                   "表の「帯 (px)」を 0 にすると切らずに使います。")
    elif p.band is not None and getattr(p.band, "overlay", False) and p.band_px:
        st.caption(f"{p.src.name}：文字とスケールバーが画像に重ねて書かれているため、"
                   f"{'下端' if p.band.side == 'bottom' else '上端'} {p.band_px} px（文字のある行）を切り落としています。"
                   "表の「帯 (px)」を 0 にすると切らずに使います。")
for i in needs_field(rep):
    p = rep.panels[i]
    fkey = f"fw_{PK[i]}"
    if fkey not in st.session_state:
        st.session_state[fkey] = SCALES.get(f"fw|{PK[i]}", "")
    c1, c2 = st.columns([2, 3])
    val = c1.text_input(f"{p.src.name} の視野幅（横）", key=fkey, help="スケールバーが見つからない画像です。例: 120 µm")
    SCALES[f"fw|{PK[i]}"] = val
    if val.strip():
        fields[i] = val
    c2.caption(f"{p.body_size[0]} × {p.body_size[1]} px（回転前の横幅を入力）。空欄のままだと、この画像にはスケールバーを付けません。")

items = build_items(rep, table, scale_text, fields, by_index(ADJ, PK), CROPSET if align == "crop" else None,
                    by_index(BARCOL, PK), ROTSET, by_index(TRIMS, PK) if align != "crop" else None)
for w in rep.warnings:
    if w.startswith("[scale]"):
        st.warning(w.replace("[scale] ", ""))
if rep.kind == "aztec":
    fw = [it.panel.body_size[0] * it.panel.um_per_px for it in items
          if it.panel.um_per_px and it.panel.kind in F.REGULAR]  # as in the files (before turning)
    if fw:
        st.caption(f"視野幅（横）: {min(fw):.4g}〜{max(fw):.4g} µm（画像間の差 {(max(fw) / min(fw) - 1) * 100:.1f}%）")
if not items:
    st.info("「使う」にチェックした画像がありません。")
    halt()

# ---- 4. preview -------------------------------------------------------------
st.subheader("4. プレビュー")
SPECS = st.session_state.setdefault("specs", {})  # free arrangement as typed, per figure
SPEC_OK = st.session_state.setdefault("spec_ok", {})  # the last one that could be read, per figure


def reset_spec(skey: str, text: str) -> None:
    st.session_state[skey] = text


if free:
    grid_text = default_spec(items, int(ncols), row_counts)
    skey = f"spec_{key}"
    if skey not in st.session_state:  # set once; the widget then keeps what the user types
        st.session_state[skey] = SPECS.get(key) or grid_text
    c1, c2 = st.columns([3, 2])
    c1.text_area("配置（番号で指定）", key=skey, height=130,
                 help="番号は表の「順」（プレビューの丸の中の番号）。入力したら、枠の外をクリックするか "
                      "Ctrl+Enter（Mac は ⌘+Enter）で反映されます。")
    c2.markdown("**書き方**　スペース＝横に並べる（高さが揃う）、`/`＝縦に積む（幅が揃う）、"
                "`( )`＝ひとまとまり、改行＝次の段  \n"
                "例：`1 2 (3/4/5)` … 1・2 の右に 3〜5 を縦に積む  \n"
                "`(9 (10 11/12 13)) 19` … 9 と 2×2 のマップ、その右に 19")
    c2.button("列数の並びから作り直す", on_click=reset_spec, args=(skey, grid_text),
              help=f"サイドバーの列数（{int(ncols)} 列）で並べた配置を書き込みます")
    spec = st.session_state[skey]
    SPECS[key] = spec
    try:
        written = T.refs_of(T.parse(spec))
        SPEC_OK[key] = spec
    except T.SpecError as e:
        written = None
        st.error(f"配置を読めません：{e}")
        spec = SPEC_OK.get(key) or grid_text
        st.caption("プレビューは前回の正しい配置で表示しています。")
    if written is not None:
        numbered = {cell_int(v) for v in table["order"]}
        used = F.numbers(items)
        unknown = [r for r in written if r not in numbered]
        off = [r for r in written if r in numbered and r not in used]
        missing = [r for r in used if r not in written]
        if unknown:
            st.caption("⚠ この番号の画像はありません：" + "、".join(map(str, unknown)) + f"（表の「順」は 1〜{len(numbered)}）")
        if off:
            st.caption("⚠ 「使う」がオフの画像なので省きました：" + "、".join(map(str, off)))
        if missing:
            st.caption("⚠ 配置に書かれていない画像（図に入りません）：" + "、".join(map(str, missing)))
    style = replace(style, spec=spec)
show_num = toggle(st, "プレビューに番号を表示", free, key=f"num_{'free' if free else 'grid'}",
                  help="画像の番号（表の「順」）をプレビューの中央に表示します。書き出すファイルには入りません")
try:
    png, g = F.preview_png(items, style, max_px=1600)
except (ValueError, RuntimeError) as e:
    st.error(str(e))
    halt()
st.image(number_overlay(png, g) if show_num else png, **FW_IMAGE)
msg = (f"仕上がり {g.width_mm:.1f} × {g.height_mm:.1f} mm（{g.width} × {g.height} px、{g.dpi:.0f} dpi、"
       f"{'自由配置' if free else f'{g.ncols} 列'}） ／ 縦横比は元画像のまま")
if g.capped:
    st.warning(msg + f" ― 画素数が大きくなりすぎるため {g.dpi:.0f} dpi に抑えました（一部の画像は元の1画素を "
               f"{g.shrink:.2f} 画素で出力）。PDF/SVG には元の画素がそのまま入ります。")
elif g.shrink < 0.999:
    st.warning(msg + f" ― 一部の画像が元の {g.shrink * 100:.0f}% に縮小されます。dpi を上げるか「元画素の整数倍」にしてください。")
else:
    st.caption(msg + f" ／ どの画像も元の1画素を {g.shrink:.2f} 画素以上で出力（縮小なし）")
# a scale bar must never go missing unnoticed: say which pictures have none, or a length of their own
placed_refs = set(g.refs or [])
no_scale, own_length = [], []
for it in items:
    if placed_refs and it.ref not in placed_refs:
        continue  # (free arrangement: not placed)
    who = f"{it.ref}. {it.label or picture_name(rep, it.panel)}"
    if style.scalebar_on == "all" or (style.scalebar_on == "electron" and it.panel.kind == "electron"):
        plan = F.bar_length(it, style)
        if plan is None:
            no_scale.append(who)
        elif plan[1]:
            own_length.append(f"{who}（{F.format_length(plan[0])}）")
if own_length:
    st.warning(f"長さ {F.format_length(style.scalebar_um)} のスケールバーは視野に入りきらないため、次の画像はその画像に合う長さで"
               "付けています：" + "、".join(own_length) + "。全部を同じ長さにするときは、左の「長さ (µm)」を短くしてください。")
if no_scale:
    st.warning("スケールが分からないため、次の画像にはスケールバーを付けていません：" + "、".join(no_scale)
               + "。「3. スケールの確認」でスケール値（または視野幅）を入れてください。")

# ---- 5. export --------------------------------------------------------------
st.subheader("5. 書き出し")


def remember_labels() -> None:
    if not learn:
        return
    R = get_reader()
    for it in items:
        if it.panel.kind == "map" and it.label and it.panel.title_mask is not None:
            R.remember("element", it.panel.title_mask, it.label, save=False)
        lg = it.panel.legend
        if lg is not None and it.label:  # the letters of each colour box, as confirmed (in the boxes' order)
            words = it.label.split()
            if len(words) == len(lg.elements) and all(w in textread.ELEMENTS for w in words):
                for c, w in zip(lg.elements, words):
                    R.remember("element", c.ink, w, save=False)
    for gk, idx in groups.items():
        t = scale_text.get(gk)
        m = rep.panels[idx[0]].scalebar.label_mask
        if t and m is not None and textread.parse_length(t)[0] is not None:
            R.remember("scale", m, t, save=False)
    R.save()


def file_name(ext: str) -> str:
    return f"{stem}_panels.zip" if ext == "zip" else f"{stem}_figure.{'tif' if ext == 'tiff' else ext}"


def make_file(ext: str) -> bytes:
    return F.panels_zip(items, style, name) if ext == "zip" else F.export(items, style, ext)


if DEFERRED_DOWNLOAD:  # one click: the file is built when the button is pressed
    cols = st.columns(len(FORMATS))
    for c, (lab, ext, mime) in zip(cols, FORMATS):
        c.download_button(lab, data=partial(make_file, ext), file_name=file_name(ext), mime=mime, on_click=remember_labels,
                          **opt(st.download_button, type="primary" if ext == "pdf" else "secondary", help=FORMAT_HELP.get(ext)),
                          **FW_DOWNLOAD)
else:  # older Streamlit: build the file first, then save it
    c1, c2 = st.columns([3, 1])
    fmt_label = c1.radio("形式", [f[0] for f in FORMATS], horizontal=True, key="fmt")
    _, ext, mime = next(f for f in FORMATS if f[0] == fmt_label)
    sig = export_signature(items, style, f"{key}|{ext}")
    if c2.button("ファイルを作る", **opt(st.button, type="primary"), **FW_BUTTON):
        with st.spinner("書き出しています…"):
            st.session_state["export"] = {"sig": sig, "data": make_file(ext), "name": file_name(ext), "mime": mime}
        remember_labels()
    ex = st.session_state.get("export")
    if ex and ex["sig"] == sig:
        st.download_button(f"保存：{ex['name']}", data=ex["data"], file_name=ex["name"], mime=ex["mime"],
                           **opt(st.download_button, type="primary"), **FW_DOWNLOAD)
    else:
        st.caption("形式を選んで「ファイルを作る」を押すと、保存ボタンが出ます（設定を変えたら作り直してください）。")
st.caption("PDF/SVG：画像は元画素のまま埋め込み、ラベルとスケールバーは文字・図形として後から編集できます。"
           "PNG/TIFF：可逆圧縮。" + ("" if DEFERRED_DOWNLOAD else "画像ごと（ZIP）：" + FORMAT_HELP["zip"]))

if len(reports) > 1:
    batch = []
    for n2, k2kind, k2, pl2 in reports:
        r2 = load(k2kind, pl2)
        pk2 = picture_keys(r2)
        t2 = TABLES[key] if k2 == key else table_for(k2, r2, pk2)
        s2 = {gk: SCALES.get(gk, pretty_len(r2.panels[i[0]].scalebar.text)) for gk, i in scale_groups(r2).items()}
        f2 = {i: SCALES.get(f"fw|{pk2[i]}", "") for i in needs_field(r2)}
        it2 = build_items(r2, t2, s2, {i: v for i, v in f2.items() if v.strip()}, by_index(ADJ, pk2),
                          {"aspect": CROP_ASPECT.get(k2), "per": by_index(CROPS, pk2)} if align == "crop" else None,
                          by_index(BARCOL, pk2),
                          {"per": by_index(ROTS, pk2), "fill": rot_fill, "resample": rot_resample},
                          by_index(TRIMS, pk2) if align != "crop" else None)
        if it2:  # free arrangement: each figure's own (rows of the column count where none was written)
            st2 = replace(style, spec=SPEC_OK.get(k2) or default_spec(it2, int(ncols), row_counts)) if free else style
            batch.append((figure_stem(n2, k2kind), it2, st2))

    def batch_zip() -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for s, it2, st2 in batch:
                for e in ("pdf", "png"):
                    z.writestr(f"{s}_figure.{e}", F.export(it2, st2, e))
        return buf.getvalue()

    blabel = f"全 {len(batch)} 図を同じ設定で一括（PDF＋PNG, ZIP）"
    if DEFERRED_DOWNLOAD:
        st.download_button(blabel, data=batch_zip, file_name="figures.zip", mime="application/zip")
    else:
        bsig = export_signature([it for _, its, _ in batch for it in its], style,
                                "batch|" + "|".join(s2.spec for _, _, s2 in batch))
        if st.button(blabel + "を作る"):
            with st.spinner("書き出しています…"):
                st.session_state["batch"] = {"sig": bsig, "data": batch_zip()}
        bx = st.session_state.get("batch")
        if bx and bx["sig"] == bsig:
            st.download_button("保存：figures.zip", data=bx["data"], file_name="figures.zip", mime="application/zip",
                               **opt(st.download_button, type="primary"))
    st.caption("まだ開いていない図は、自動読み取りのラベルとスケールで書き出します。")

with st.expander("仕組みと注意"):
    st.markdown(
        "- 画像は**縦横同じ倍率でしか拡大縮小しません**。出力の幅と高さを両方指定した場合も、画像は変形させず、"
        "枠に収まる最大の大きさで中央に置きます（余りは背景色）。描画のたびに全画像の縦横比を検算しています。\n"
        "- 形の違う画像は「幅を揃える」「高さを揃える」「トリミングして揃える」から並べ方を選べます（左の「配置」）。"
        "「自由配置」では、横並び・縦積み・入れ子の配置を番号で書きます。横並びは高さ、縦積みは幅がぴったり揃うように"
        "各画像の大きさを計算します。トリミングは元の画素をそのまま切り取るだけで、拡大縮小や変形はしません。\n"
        "- 元素記号・スケールバーの文字（pt）とバーの太さ（pt）、文字・バーと画像の端の間隔は、図全体で共通です。"
        "画像を小さく置いても大きく置いても、文字とバーは同じ大きさになります。\n"
        "- 回転・反転：90°単位と反転は画素を並べ替えるだけです。それ以外の角度は補間で画素を作り直します。"
        "スケールは回転前の画像から読み取った1画素の長さをそのまま使い、回転後の画像に水平なスケールバーを付けます。\n"
        "- AZtecのWord出力は、画像ごとに背景が透明なPNGで入っています。不透明な長方形＝画像本体、その上＝タイトル、"
        "下＝スケールバー、と**形だけで**切り分けます。スケールバーの無い横長の画像（Map Sum Spectrum など）は"
        "スペクトルと判定し、最初は図から外します。\n"
        "- SEM・TEM像は、端にある「平坦な地に文字と線だけの帯」（日立は白地、JEOLは黒地、FEI/Thermo は灰色の表など）を"
        "情報帯として取り除きます。帯の中の一番長い線をスケールバーとし（目盛りがあれば両端の目盛りの中心間）、"
        "近くの「数値＋単位」の文字を読みます。TEM-EDS のマップは、端の強度カラーバーも外します。\n"
        "- 文字とスケールバーが画像に直接重ねて書かれた形式（JEOL の FE-SEM など）は、その文字のある下端の行を切り落とします。"
        "文字の下の画像は元データの時点で上書きされているので、復元はできません。"
        "画像の中に描き込まれた測長線なども画像の一部なので消せません。\n"
        "- DigitalMicrograph（.dm3/.dm4）は画素値の 0.1〜99.9% を白黒の 256 階調に割り当てて表示し、記録された画素寸法を使います"
        "（回折像など単位が 1/nm のものにはスケールバーを付けません）。\n"
        "- JEOL EPMA のマップ画面は、青緑の地の帯にはさまれた長方形をマップとして元の画素のまま切り出し、下の白い枠の細い線"
        "（スケールバー）の長さと「20 µm」などの文字から1画素の大きさを求めます。どのマップも同じ測定なので、同じスケールです。\n"
        "- 文字（元素記号とスケール値）は、候補を元の字形に似たフォントで描いて1文字ずつ照合しています。"
        "確認した読み取りは記憶され、同じ装置・テンプレートの画像では次回から確実に読めます。\n"
        "- SEMのスケールバーが短いと（例：36画素）、バーの長さの丸めで±3%程度の誤差が出ます。"
        "SEMが画像と一緒に保存する .txt（日立の PixelSize、JEOLの $$SM_MICRON_BAR）や、"
        "FEI/Thermo・ZEISS の TIFF に記録された画素寸法があれば、そちらを優先して使います。\n"
        "- 作業ファイル（.edsfig）：読み込んだファイルを1バイトも変えずに（圧縮もせずに）入れたZIP形式のファイルに、"
        "設定（edsfig.json）を添えたものです。設定は画像の中身（MD5）に結び付けてあるので、"
        "「設定だけ保存」したファイルも、同じ元ファイルと一緒に入れれば（入れる順番やフォルダが違っても）当てはまります。\n"
        "- 画質の上限は元画像の解像度です。\n"
        f"- 動作環境：Streamlit {st.__version__}"
        + ("" if DEFERRED_DOWNLOAD else "（1.52 以降に更新すると、書き出しがボタン1回で済みます）")
    )

save_panel()  # last: the work file gets every setting as it stands after this run

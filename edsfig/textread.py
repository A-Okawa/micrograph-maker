"""Read the two short texts in an AZtec picture: the element symbol and the scale value.

No OCR engine is used.  The possible answers are known in advance (element symbols, and
"nice" scale-bar values such as 5 µm or 250 nm), so each candidate is drawn with fonts that
look like AZtec's and compared with the observed ink glyph by glyph (soft Dice overlap,
searched over font size and a few pixels of shift).  Comparing glyph by glyph makes the
result insensitive to letter spacing, and every glyph counts equally, so a narrow "1" is
weighed as much as an "m".

Labels the user confirms are kept as exemplars (exact AZtec ink) and tried first next
time, so recognition improves with use.  Everything runs offline with Pillow + NumPy only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent / "assets"

ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se "
    "Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy "
    "Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am"
).split()
# not measurable by EDS or practically never mapped: only used to break near-ties
_UNLIKELY = set("H He Li Ne Kr Xe Rn Tc Pm Po At Fr Ra Ac Pa Np Pu Am Tl Os".split())

UNIT_TO_UM = {"nm": 1e-3, "µm": 1.0, "um": 1.0, "μm": 1.0, "mm": 1e3}


def _nice_numbers() -> list[str]:
    out: list[str] = []
    for k in range(-1, 4):
        for m in (1, 2, 2.5, 3, 4, 5):
            s = f"{m * 10 ** k:g}"
            if s not in out:
                out.append(s)
    return out


def _three_digit(v: float) -> str | None:
    """Hitachi style '1.00', '10.0' (three significant digits) for values 1..99."""
    if 1 <= v < 10:
        return f"{v:.2f}"
    if 10 <= v < 100:
        return f"{v:.1f}"
    return None


def _scale_numbers() -> list[str]:
    out = []
    for n in _nice_numbers():
        out.append(n)
        t = _three_digit(float(n))
        if t and t not in out:
            out.append(t)
    return out


# spaces are not part of the candidates: the glyph-by-glyph comparison ignores spacing
SCALE_CANDIDATES = [n + u for n in _scale_numbers() for u in ("nm", "µm", "um", "mm")]


def parse_length(text: str) -> tuple[float | None, str]:
    """'5µm' / '5 um' / '250 nm' / '1mm' / '1.00um' -> (value, unit)."""
    t = (text or "").strip().replace("μ", "µ").replace(",", ".")
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*(nm|µm|um|mm)?\s*$", t)
    if not m:
        return None, "µm"
    unit = (m.group(2) or "µm").replace("um", "µm")
    return float(m.group(1)), unit


def pretty_length(text: str) -> str:
    """Normalised display form: '100um' -> '100 µm', '1.00um' -> '1 µm'."""
    v, u = parse_length(text)
    return f"{v:g} {u}" if v is not None else (text or "")


# --------------------------------------------------------------------------- fonts
FONT_SETS = {
    # AZtec map titles: Segoe UI Semibold look-alikes
    "title": (["Inter-SemiBold.otf", "Carlito-Bold.ttf"], ["seguisb.ttf", "segoeuib.ttf"]),
    # AZtec scale labels: Segoe UI look-alikes
    "aztec": (["Inter-Regular.otf", "Carlito-Regular.ttf"], ["segoeui.ttf"]),
    # SEM info bands: Arial-like (Hitachi), Courier-like bold (JEOL), plain sans, Consolas-like (JEOL EPMA)
    "band": (["LiberationSans-Regular.ttf", "FreeMonoBold.ttf", "Inter-Regular.otf", "mpl:DejaVuSansMono.ttf"],
             ["arial.ttf", "courbd.ttf", "consola.ttf"]),
    # JEOL EPMA map windows: bold Arial-like names and lengths under each map
    "epma": (["LiberationSans-Bold.ttf", "mpl:DejaVuSans-Bold.ttf"], ["arialbd.ttf"]),
}


def _font_candidates(bold) -> list[str]:
    """Bundled look-alike fonts, plus the real Windows fonts when the OS has them.
    `bold` is a FONT_SETS key ("title", "aztec", "band"); True/False mean "title"/"aztec"."""
    key = bold if isinstance(bold, str) else ("title" if bold else "aztec")
    bundled, system = FONT_SETS[key]
    out = []
    for f in bundled:
        p = _mpl_font(f[4:]) if f.startswith("mpl:") else ASSETS / "fonts" / f  # "mpl:": shipped with matplotlib
        if p is not None and p.exists():
            out.append(str(p))
    for f in system:
        p = _system_font(f)
        if p is not None:
            out.append(str(p))
    if not out:  # always available: matplotlib's DejaVu
        import matplotlib

        d = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        out.append(str(d / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")))
    return out


def _mpl_font(name: str) -> Path | None:
    try:
        import matplotlib
    except ImportError:
        return None
    return Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / name


# fonts also looked for on a Mac (Microsoft Office for Mac brings Consolas along)
MAC_FONTS = {"consola.ttf"}


@lru_cache(maxsize=64)
def _system_font(name: str) -> Path | None:
    """The real font when this PC has it: Windows' Fonts folder; for MAC_FONTS also the Mac font
    folders and the fonts inside Microsoft Office for Mac."""
    dirs = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"]
    if name in MAC_FONTS:
        dirs += [Path("/Library/Fonts"), Path("/Library/Fonts/Microsoft"), Path.home() / "Library" / "Fonts"]
        dirs += [Path(f"/Applications/Microsoft {app}.app/Contents/Resources/DFonts")
                 for app in ("Word", "Excel", "PowerPoint", "Outlook")]
    for d in dirs:
        for cand in (name, name.capitalize()):
            p = d / cand
            try:
                if p.is_file():
                    return p
            except OSError:
                continue
    return None


@lru_cache(maxsize=256)
def _font(font_path: str, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, px)


@lru_cache(maxsize=40000)
def _draw(text: str, font_path: str, px: int) -> np.ndarray:
    """Text drawn on a fixed line frame (same ascender line for every string)."""
    f = _font(font_path, px)
    asc, desc = f.getmetrics()
    w = int(f.getlength(text)) + 2 * px + 8
    im = Image.new("L", (w, asc + desc + 8), 0)
    ImageDraw.Draw(im).text((px, 4), text, font=f, fill=255)
    a = np.asarray(im, dtype=np.float64) / 255.0
    a.setflags(write=False)
    return a


def _rows(a: np.ndarray) -> tuple[int, int] | None:
    r = np.flatnonzero((a > 0.12).any(1))
    return (int(r[0]), int(r[-1])) if r.size else None


def _cols(a: np.ndarray) -> np.ndarray:
    c = np.flatnonzero((a > 0.12).any(0))
    return a[:, c[0] : c[-1] + 1] if c.size else a[:, :1] * 0


@lru_cache(maxsize=40000)
def _frame(text: str, font_path: str, px: int) -> tuple[int, int] | None:
    """Rows the whole string occupies on the line frame."""
    rows = [_rows(_draw(ch, font_path, px)) for ch in text]
    rows = [r for r in rows if r]
    if not rows:
        return None
    return min(r[0] for r in rows), max(r[1] for r in rows)


@lru_cache(maxsize=80000)
def _unit(unit: str, font_path: str, px: int, r0: int, r1: int) -> np.ndarray:
    """One glyph (or a touching pair) cropped to a given row frame."""
    return _cols(_draw(unit, font_path, px)[r0 : r1 + 1])


@lru_cache(maxsize=4096)
def _render(text: str, font_path: str, px: int) -> np.ndarray:
    m = crop_ink(_draw(text, font_path, px))
    return m if m is not None else np.zeros((1, 1))


@lru_cache(maxsize=64)
def _cap_ratio(font_path: str) -> float:
    """cap height (px) per font px."""
    return _render("H", font_path, 100).shape[0] / 100.0


# --------------------------------------------------------------------------- ink helpers
def crop_ink(m: np.ndarray | None, thr: float = 0.12) -> np.ndarray | None:
    if m is None or m.size == 0:
        return None
    ys = np.flatnonzero((m > thr).any(1))
    xs = np.flatnonzero((m > thr).any(0))
    if ys.size == 0 or xs.size == 0:
        return None
    return m[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]


def segments(m: np.ndarray, gap: int = 1, thr: float = 0.12) -> list[tuple[int, int]]:
    """Column ranges of ink separated by at least `gap` empty columns."""
    col = (m > thr).any(0)
    out: list[list[int]] = []
    x, n = 0, col.size
    while x < n:
        if col[x]:
            s = x
            while x < n and col[x]:
                x += 1
            if out and s - out[-1][1] - 1 < gap:
                out[-1][1] = x - 1
            else:
                out.append([s, x - 1])
        else:
            x += 1
    return [(a, b) for a, b in out]


def split_words(m: np.ndarray, gap_min: int) -> list[np.ndarray]:
    return [crop_ink(m[:, a : b + 1]) for a, b in segments(m, gap_min)]


def line_baseline(line: np.ndarray) -> int:
    """Baseline row of a text line: where most inked columns end (descenders are a minority)."""
    cols = [np.flatnonzero(c > 0.4) for c in line.T]
    bottoms = [int(c[-1]) for c in cols if c.size]
    if not bottoms:
        return line.shape[0] - 1
    vals, counts = np.unique(bottoms, return_counts=True)
    return int(vals[counts == counts.max()].max())


@dataclass
class Word:
    """Ink of one word plus where the line's baseline falls inside it."""

    ink: np.ndarray
    baseline: int

    @property
    def cap(self) -> int:
        rows = np.flatnonzero((self.ink > 0.3).any(1))
        return max(1, self.baseline - int(rows[0]) + 1) if rows.size else self.ink.shape[0]

    @property
    def descender(self) -> bool:
        return bool((self.ink[self.baseline + 2 :] > 0.4).any())


def words_of_line(line: np.ndarray, gap_min: int) -> list[Word]:
    base = line_baseline(line)
    out = []
    for a, b in segments(line, gap_min):
        sub = line[:, a : b + 1]
        rows = np.flatnonzero((sub > 0.12).any(1))
        if rows.size == 0:
            continue
        out.append(Word(ink=sub[rows[0] : rows[-1] + 1], baseline=base - int(rows[0])))
    return out


@lru_cache(maxsize=40000)
def _template_descender(text: str, font_path: str, px: int) -> bool:
    ref = _rows(_draw("H", font_path, px))  # bottom of "H" = baseline on the same frame
    base = ref[1] if ref else 0
    d = _draw(text, font_path, px)
    return bool((d[base + 2 :] > 0.4).any())


def arch(piece: np.ndarray) -> float:
    """Ink across the top minus ink across the bottom of a glyph's middle columns: positive for an
    arch (n, m), negative for a bowl (u)."""
    on = piece > 0.3
    rows, cols = np.flatnonzero(on.any(axis=1)), np.flatnonzero(on.any(axis=0))
    if rows.size < 4 or cols.size < 3:
        return 0.0
    y0, y1, x0, x1 = rows[0], rows[-1], cols[0], cols[-1]
    h, w = y1 - y0 + 1, x1 - x0 + 1
    mid = piece[:, x0 + int(0.3 * w): x1 - int(0.3 * w) + 1]
    t = max(1, int(round(0.25 * h)))
    return float(mid[y0: y0 + t].mean() - mid[y1 - t + 1: y1 + 1].mean())


def dice(obs: np.ndarray, tmpl: np.ndarray, shift: int = 2) -> float:
    """Best soft-Dice overlap of tmpl over obs (top-left aligned, ±shift px)."""
    h, w = tmpl.shape
    P = shift + max(h, w)
    pad = np.zeros((obs.shape[0] + 2 * P, obs.shape[1] + 2 * P))
    pad[P : P + obs.shape[0], P : P + obs.shape[1]] = obs
    win = sliding_window_view(pad, (h, w))[P - shift : P + shift + 1, P - shift : P + shift + 1]
    inter = np.minimum(win, tmpl).sum(axis=(2, 3)).max()
    return float(2 * inter / (obs.sum() + tmpl.sum() + 1e-9))


# --------------------------------------------------------------------------- reader
@dataclass
class ReadResult:
    text: str
    score: float
    margin: float
    candidates: list[tuple[str, float]] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        return self.score >= 0.80 and self.margin >= 0.03


def default_store_path() -> Path:
    base = os.environ.get("EDSFIG_HOME")
    if base:
        return Path(base) / "exemplars.json"
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", str(Path.home()))) / "edsfig" / "exemplars.json"
    return Path.home() / ".edsfig" / "exemplars.json"


class Reader:
    """Glyph-template reader with a small persistent memory of confirmed AZtec labels."""

    def __init__(self, store_path: str | Path | None = None, learn: bool = True):
        self.store_path = Path(store_path) if store_path else default_store_path()
        self.learn = learn
        self.exemplars: dict[str, list[dict]] = {"element": [], "scale": []}
        self._cache: dict[str, ReadResult] = {}
        # one reader serves every user of a shared app: learning and saving one at a time
        self._lock = threading.RLock()
        self._load(ASSETS / "exemplars_seed.json")
        self._load(self.store_path)

    # -- persistence --------------------------------------------------------
    def _load(self, path: Path) -> None:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return
        for kind in ("element", "scale"):
            for e in data.get(kind, []):
                try:
                    arr = np.array(e["mask"], dtype=np.float64) / 255.0
                    self._add(kind, e["text"], arr)
                except Exception:
                    continue

    def _add(self, kind: str, text: str, mask: np.ndarray) -> bool:
        with self._lock:
            return self._add_locked(kind, text, mask)

    def _add_locked(self, kind: str, text: str, mask: np.ndarray) -> bool:
        lst = self.exemplars[kind]
        for e in lst:
            if e["mask"].shape == mask.shape and np.abs(e["mask"] - mask).max() < 0.02:
                if e["text"] == text:
                    return False
                e["text"] = text  # the user corrected an earlier label for the same ink
                return True
        lst.append({"text": text, "mask": mask})
        same = [e for e in lst if e["text"] == text]
        if len(same) > 6:  # keep the store small
            lst.remove(same[0])
        return True

    def save(self) -> bool:
        with self._lock:
            data = {k: [{"text": e["text"], "mask": np.round(e["mask"] * 255).astype(int).tolist()} for e in v]
                    for k, v in self.exemplars.items()}
            tmp = self.store_path.with_name(self.store_path.name + f".{os.getpid()}.tmp")
            try:
                self.store_path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(data), encoding="utf-8")
                os.replace(tmp, self.store_path)  # whole or not at all: never half a file
                return True
            except Exception:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                return False

    def remember(self, kind: str, mask: np.ndarray | None, text: str, save: bool = True) -> bool:
        """Store a user-confirmed label (kind = 'element' -> title mask, 'scale' -> label mask)."""
        text = (text or "").strip()
        if mask is None or not text or not self.learn:
            return False
        if kind == "element":
            if text not in ELEMENTS:  # a custom label ("Ti-K", "TiO2 region") must not become a reading
                return False
            words = words_of_line(mask, max(3, int(0.25 * mask.shape[0])))
            if not words:
                return False
            mask = words[0].ink
        else:
            mask = crop_ink(mask)
            if mask is None:
                return False
            text = text.replace(" ", "").replace("um", "µm").replace("μ", "µ")
        with self._lock:
            changed = self._add(kind, text.strip(), mask)
            if changed:
                self._cache.clear()
                if save:
                    self.save()
        return changed

    # -- matching -----------------------------------------------------------
    def _score_exemplars(self, kind: str, obs: np.ndarray) -> dict[str, float]:
        """Exact AZtec ink seen before.  Only near-identical ink counts (same report template),
        so that e.g. a stored '100µm' can never vouch for an observed '400µm'."""
        out: dict[str, float] = {}
        for e in list(self.exemplars[kind]):  # (a copy: another user may be adding one)
            m = e["mask"]
            if abs(m.shape[0] - obs.shape[0]) > 1 or abs(m.shape[1] - obs.shape[1]) > 1:
                continue
            s = dice(obs, m, shift=1)
            if s > out.get(e["text"], 0):
                out[e["text"]] = s
        return out

    def _score_rendered(self, word: Word, candidates: list[str], bold) -> dict[str, float]:
        obs = word.ink
        cap = word.cap
        desc = word.descender
        H, W = obs.shape
        segs = segments(obs, thr=0.3)  # faint anti-aliasing bridges do not join glyphs
        k = len(segs)
        pieces = [obs[:, a : b + 1] for a, b in segs]
        merged = [obs[:, segs[i][0] : segs[i + 1][1] + 1] for i in range(k - 1)]
        grid = []
        for fp in _font_candidates(bold):
            px0 = int(round(cap / _cap_ratio(fp)))
            grid += [(fp, px) for px in range(max(5, px0 - 2), px0 + 3)]
        if not grid or k == 0:  # too small to read
            return {}
        memo: dict[tuple, float] = {}

        def is_dot(a: int, b: int) -> bool:  # a small blob sitting on the baseline
            rows = np.flatnonzero((obs[:, a : b + 1] > 0.3).any(axis=1))
            return bool(rows.size) and rows[-1] - rows[0] + 1 <= 0.35 * cap and b - a + 1 <= 0.45 * cap \
                and abs(int(rows[-1]) - word.baseline) <= max(1, int(0.15 * cap))

        has_dot = any(is_dot(a, b) for a, b in segs)
        # where the decimal point sits: after how many glyphs ("2.00" and "20.0" differ only there)
        dot_at = next((k for k, (a, b) in enumerate(segs) if is_dot(a, b)), None)

        def best(piece_key, piece, unit_text, text):
            """max Dice of an observed piece vs. a glyph unit over all fonts / sizes."""
            v = 0.0
            for fp, px in grid:
                fr = _frame(text, fp, px)
                if fr is None:
                    continue
                key = (piece_key, unit_text, fp, px, fr)
                s = memo.get(key)
                if s is None:
                    t = _unit(unit_text, fp, px, *fr)
                    s = dice(piece, t) if abs(t.shape[0] - piece.shape[0]) <= 3 else 0.0
                    memo[key] = s
                v = max(v, s)
            if piece_key[0] == "p" and unit_text in ("u", "n"):
                # 'u' and 'n' are the same shape upside down: open at the top (u) or at the bottom (n)
                a = arch(piece)
                if (unit_text == "n" and a < -0.2) or (unit_text == "u" and a > 0.2):
                    v -= 0.1
            return v

        out: dict[str, float] = {}
        for c in candidates:
            n = len(c)
            if not (k <= 2 * n and n <= 3 * k):
                continue
            # cheap structural filters at the nominal size: width, height, descender
            fp0, px0 = grid[len(grid) // len(_font_candidates(bold)) // 2]
            t0 = _render(c, fp0, px0)
            if abs(t0.shape[1] - W) > 0.45 * W + 3 or abs(t0.shape[0] - H) > 4:
                continue
            if _template_descender(c, fp0, px0) != desc:
                continue
            # monotonic alignment: a piece holds 1-3 touching glyphs, or a glyph spans 2 pieces
            D = np.full((k + 1, n + 1), -1.0)
            D[0, 0] = 0.0
            for i in range(k):
                for j in range(n):
                    if D[i, j] < 0:
                        continue
                    for L in (1, 2, 3):  # touching glyphs are allowed, but one glyph per piece is likelier
                        if j + L > n:
                            break
                        v = D[i, j] + best(("p", i), pieces[i], c[j : j + L], c) * (1 - 0.02 * (L - 1))
                        D[i + 1, j + L] = max(D[i + 1, j + L], v)
                    if i + 1 < k:
                        v = D[i, j] + 2 * best(("m", i), merged[i], c[j], c) * 0.98
                        D[i + 2, j + 1] = max(D[i + 2, j + 1], v)
            if D[k, n] <= 0:
                continue
            s = D[k, n] / k
            if bold in (True, "title") and c in _UNLIKELY:
                s -= 0.015
            if ("." in c) != has_dot:  # a decimal point is either there or not
                s -= 0.06
            elif dot_at is not None and "." in c and c.index(".") != dot_at and k == len(c.replace(" ", "")):
                s -= 0.08  # one glyph per piece, and the point after another number of digits
            out[c] = s
        return out

    def _decide(self, kind: str, word: Word | None, candidates: list[str], bold) -> ReadResult:
        if word is None or word.ink.size == 0:
            return ReadResult("", 0.0, 0.0)
        obs = word.ink
        key = f"{kind}|{bold}|" + hashlib.sha1(np.round(obs * 255).astype(np.uint8).tobytes() + str(obs.shape).encode()
                                                + "|".join(candidates).encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        ex = self._score_exemplars(kind, obs)
        scores = self._score_rendered(word, candidates, bold)
        for t, v in ex.items():  # exact AZtec ink beats a look-alike font
            if v >= 0.96:
                scores[t] = max(scores.get(t, 0.0), v)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            res = ReadResult("", 0.0, 0.0)
        else:
            s1 = ranked[0][1]
            s2 = ranked[1][1] if len(ranked) > 1 else 0.0
            res = ReadResult(ranked[0][0], float(min(1.0, s1)), float(s1 - s2),
                             [(t, round(float(min(1.0, v)), 3)) for t, v in ranked[:5]])
        self._cache[key] = res
        return res

    def read_element(self, title_mask: np.ndarray, fonts: str = "title") -> ReadResult:
        """Element symbol = first word of a map title ('Ti Kα1' -> 'Ti', 'C K' -> 'C')."""
        words = words_of_line(title_mask, max(3, int(0.25 * title_mask.shape[0])))
        if not words:
            return ReadResult("", 0.0, 0.0)
        return self._decide("element", words[0], ELEMENTS, bold=fonts)

    def first_word_is(self, title_mask: np.ndarray, word: str, fonts: str = "title") -> bool:
        """Whether the title starts with `word` rather than with an element symbol, e.g. "EDS" in
        "EDS Layered Image 6" (the title of a layered image, not of a map)."""
        words = words_of_line(title_mask, max(3, int(0.25 * title_mask.shape[0])))
        if not words:
            return False
        scores = self._score_rendered(words[0], [word] + ELEMENTS, fonts)
        s = scores.get(word, 0.0)
        return s >= 0.7 and s > max((v for t, v in scores.items() if t != word), default=0.0) + 0.02

    def read_name(self, mask: np.ndarray, extra: list[str], fonts: str) -> ReadResult:
        """A short name: an element symbol or one of `extra` (e.g. "SED", "BED-C"), read as one word."""
        m = crop_ink(mask)
        if m is None:
            return ReadResult("", 0.0, 0.0)
        return self._decide("element", Word(ink=m, baseline=line_baseline(m)), ELEMENTS + list(extra), bold=fonts)

    def read_scale(self, label_mask: np.ndarray, fonts: str = "aztec") -> ReadResult:
        m = crop_ink(label_mask)
        if m is None:
            return ReadResult("", 0.0, 0.0)
        return self._decide("scale", Word(ink=m, baseline=line_baseline(m)), SCALE_CANDIDATES, bold=fonts)

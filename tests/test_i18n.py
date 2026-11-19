"""Screen languages: every text of the sources has its English, Vietnamese and Thai row, with the same
placeholders and markup; library messages turn into the language of the screen and survive pickling."""
from __future__ import annotations

import ast
import pickle
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edsfig import i18n, transform, tree  # noqa: E402
from edsfig.i18n_text import TEXT  # noqa: E402

JA = re.compile(r"[ぁ-んァ-ヶ一-龥]")
# choices shown through format_func=ui (their Japanese text is the stored value) and the table's 読み取り
CHOICES = {"KIND_JA", "ASPECT_PRESETS", "TRIM_SHAPES", "BAR_CHOICES", "SIZE_MODES", "ARRANGES", "ROW_ALIGNS", "ALIGNS",
           "VALIGNS", "BACKGROUNDS", "LABEL_COLORS", "CORNERS", "BAR_TARGETS", "BAR_CORNERS", "BAR_COLORS", "RES_MODES",
           "ROT_FILLS", "ROT_RESAMPLES", "INTERPS", "FORMATS", "names", "APP_TAGLINE"}
STATUS_FUNCS = {"scale_status", "default_rows"}
SOURCES = [ROOT / "app.py", ROOT / "appinfo.py"] + sorted((ROOT / "edsfig").glob("*.py"))


def texts_of(path: Path) -> list[str]:
    tree_ = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for n in ast.walk(tree_):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if name in ("ui", "msg", "translate", "join") and n.args and isinstance(n.args[0], ast.Constant) \
                    and isinstance(n.args[0].value, str):
                out.append(n.args[0].value)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in CHOICES for t in n.targets):
            out += [c.value for c in ast.walk(n.value) if isinstance(c, ast.Constant) and isinstance(c.value, str)]
        if isinstance(n, ast.FunctionDef) and n.name in STATUS_FUNCS:
            for c in ast.walk(n):
                if isinstance(c, (ast.Return, ast.Assign)):
                    out += [d.value for d in ast.walk(c) if isinstance(d, ast.Constant) and isinstance(d.value, str)]
    return [t.rstrip(" ") for t in out if JA.search(t)]


def fields(s: str) -> list[str]:
    return sorted(f"{name}:{spec}" for _, name, spec, _ in string.Formatter().parse(s) if name is not None)


def test_every_text_is_translated():
    table = {row[0] for row in TEXT}
    missing = sorted({t for p in SOURCES for t in texts_of(p)} - table)
    assert not missing, missing[:5]
    assert len(table) == len(TEXT), "a Japanese text is listed twice"


def test_rows_keep_placeholders_and_markup():
    for row in TEXT:
        assert len(row) == 4 and all(isinstance(x, str) and x for x in row), row[0]
        ja = row[0]
        for lang_, text in zip(("en", "vi", "th"), row[1:]):
            assert fields(text) == fields(ja), (lang_, ja[:40])
            for mark in ("**", "`", "\n", "$$"):
                assert text.count(mark) == ja.count(mark), (lang_, mark, ja[:40])
            assert not JA.search(text), (lang_, ja[:40])


def test_ui_and_trailing_spaces():
    try:
        i18n.lang_getter = lambda: "en"
        assert i18n.ui("色") == "Colour" and i18n.ui("色 ") == "Colour "  # two widgets told apart by the space
        assert i18n.ui("{0} 列", 3) == "3 columns"
        assert i18n.ui("この文は表にありません") == "この文は表にありません"
        i18n.lang_getter = lambda: "th"
        assert i18n.ui("幅 (mm)") == "กว้าง (mm)"
    finally:
        i18n.lang_getter = lambda: "ja"
    assert i18n.ui("色 ") == "色 " and i18n.ui("{0} 列", 3) == "3 列"


def test_library_messages():
    d = transform.describe(90, True, False)
    assert d == "時計回り 90°・左右反転" and isinstance(d, str)  # the command line reads Japanese, as before
    assert i18n.show(d, "en") == "90° clockwise, Mirror left–right"
    back = pickle.loads(pickle.dumps(d))  # (reports are cached by pickling)
    assert back == d and i18n.show(back, "vi") == i18n.show(d, "vi") != d
    assert i18n.show(transform.describe(0), "en") == "none"
    try:
        tree.parse("1 2 x")
        raise AssertionError("no error")
    except tree.SpecError as e:
        assert "使えない文字「x」" in str(e)
        assert i18n.error_text(e, "en").startswith("The character “x” cannot be used")
    try:
        tree.parse("1 1 2")
    except tree.SpecError as e:
        assert i18n.error_text(e, "en") == "The same number appears more than once: 1 (each image only once)"
    assert i18n.show(i18n.msg("[scale] {0}", i18n.msg("較正情報がありません")), "en") == "[scale] there is no calibration"
    assert i18n.error_text(ValueError("plain"), "en") == "plain"


if __name__ == "__main__":
    test_every_text_is_translated()
    test_rows_keep_placeholders_and_markup()
    test_ui_and_trailing_spaces()
    test_library_messages()
    print("ok")

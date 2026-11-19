"""Screen languages: 日本語, English, Tiếng Việt, ไทย.  Figures keep the text the user types.

Every screen text is written in Japanese as ui("日本語") (app.py) or msg("日本語") (the library);
English, Vietnamese and Thai come from i18n_text.py, looked up by the Japanese text.  A text missing
there is shown in Japanese (tests/test_i18n.py checks that none is missing).  Values are filled in
after the language is chosen: ui("{0} 行", 5).

Choices keep their Japanese text as the stored value (the session and work files are the same in every
language); the widgets show them through format_func=ui.

The library never needs to know the language: msg() returns the Japanese text (a str, so the command
line and the tests read it as before) that also remembers its template and values, and show() gives it
in the language of the screen.

The language of the current screen comes from lang_getter: app.py points it at its session state
(use_streamlit_session), so every user of a shared server sees their own language.
"""
from __future__ import annotations

try:
    from .i18n_text import TEXT as _ROWS
except ImportError:  # (no table: Japanese only)
    _ROWS = []

LANGUAGES = {"日本語": "ja", "English": "en", "Tiếng Việt": "vi", "ไทย": "th"}
CODES = list(LANGUAGES.values())
NAMES = {code: name for name, code in LANGUAGES.items()}
TABLE = {code: {row[0]: row[i + 1] for row in _ROWS} for i, code in enumerate(("en", "vi", "th"))}


def _japanese() -> str:
    return "ja"


lang_getter = _japanese


def use_streamlit_session(key: str = "lang") -> None:
    """Take the screen language from st.session_state[key] (a code: ja, en, vi or th)."""
    import streamlit as st
    global lang_getter

    def from_session() -> str:
        code = st.session_state.get(key)
        return code if code in CODES else "ja"

    lang_getter = from_session


def lang() -> str:
    try:
        code = lang_getter()
    except Exception:  # (outside a Streamlit run)
        return "ja"
    return code if code in CODES else "ja"


def translate(ja: str, code: str) -> str:
    """The text in one language (as written when it is not in the table).  Spaces at the end are kept:
    two widgets with the same label are told apart by them."""
    if code == "ja" or not isinstance(ja, str):
        return ja
    core = ja.rstrip(" ")
    got = TABLE.get(code, {}).get(core)
    return ja if got is None else got + ja[len(core):]


def ui(ja: str, *args) -> str:
    """The text for the current screen language; {0}, {1} ... are filled with args."""
    text = translate(ja, lang())
    return text.format(*[show(a) for a in args]) if args else text


class Msg(str):
    """A message of the library: its Japanese text, with the template and values it was made from."""
    template: str
    args: tuple

    def __new__(cls, template: str, *args):
        self = super().__new__(cls, template.format(*args) if args else template)
        self.template, self.args = template, args
        return self

    def __reduce__(self):
        return (Msg, (self.template, *self.args))

    def show(self, code: str) -> str:
        text = translate(self.template, code)
        return text.format(*[show(a, code) for a in self.args]) if self.args else text


class Joined(str):
    """Messages joined by a separator (the separator is translated too)."""
    sep: str
    items: tuple

    def __new__(cls, sep: str, items):
        items = tuple(items)
        self = super().__new__(cls, sep.join(str(i) for i in items))
        self.sep, self.items = sep, items
        return self

    def __reduce__(self):
        return (Joined, (self.sep, self.items))

    def show(self, code: str) -> str:
        return translate(self.sep, code).join(show(i, code) for i in self.items)


def msg(template: str, *args) -> Msg:
    return Msg(template, *args)


def join(sep: str, items) -> Joined:
    return Joined(sep, items)


def show(x, code: str | None = None):
    """A value for the screen: library messages in the screen's language, anything else as it is."""
    if isinstance(x, (Msg, Joined)):
        return x.show(code or lang())
    return x


def error_text(e: BaseException, code: str | None = None) -> str:
    """The message of an exception raised with msg(...)."""
    a = e.args[0] if e.args else ""
    return show(a, code) if isinstance(a, (Msg, Joined)) else str(e)

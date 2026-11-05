"""Free arrangement: pictures in rows and columns nested to any depth, written with their numbers.

    1 2 (3/4/5)
    6 7 8
    (9 (10 11/12 13) / 14 (15 16/17 18)) 19

* a space (or a comma) puts pictures side by side: they get the same height
* "/" stacks them: they get the same width
* ( ) makes a group; a new line (or ";") starts the next row of the whole figure

Every picture keeps its own shape (trimmed or not), so all sizes follow exactly from the figure
width: for any group the width is an affine function of its height, w = A*h + B (B collects the
gaps).  The coefficients are combined from the inside out and the sizes handed out from the outside
in, so rows end flush, columns line up and no picture is stretched.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field


class SpecError(ValueError):
    """The arrangement text cannot be read (the message says why, in Japanese)."""


@dataclass
class Node:
    kind: str  # "pic" | "row" (side by side) | "col" (stacked)
    ref: int = 0  # pic: the number written
    kids: list["Node"] = field(default_factory=list)

    def __str__(self) -> str:  # compact form, used by the tests
        if self.kind == "pic":
            return str(self.ref)
        sep = " " if self.kind == "row" else "/"
        return "(" + sep.join(str(k) for k in self.kids) + ")"


SEPARATORS = set(" \t\r,、")


def _tokens(text: str) -> list[tuple[str, object, int]]:
    """(kind, value, position): kinds num, open, close, slash.  A new line outside brackets starts a
    new row (a slash); blank lines and line breaks next to a "/" are ignored."""
    s = unicodedata.normalize("NFKC", text or "")
    out: list[tuple[str, object, int]] = []
    depth, k = 0, 0
    while k < len(s):
        c = s[k]
        if c.isdigit():
            j = k
            while j < len(s) and s[j].isdigit():
                j += 1
            out.append(("num", int(s[k:j]), k))
            k = j
            continue
        if c in "([":
            depth += 1
            out.append(("open", c, k))
        elif c in ")]":
            depth -= 1
            if depth < 0:
                raise SpecError("「)」が「(」より多くあります")
            out.append(("close", c, k))
        elif c in "/;":
            out.append(("slash", c, k))
        elif c == "\n":
            if depth == 0:
                out.append(("nl", c, k))
        elif c in SEPARATORS or c.isspace():
            pass
        else:
            raise SpecError(f"使えない文字「{c}」があります。数字・空白・/・( ) だけで書いてください")
        k += 1
    if depth > 0:
        raise SpecError("「(」と「)」の数が合いません（「)」が足りません）")
    # a line break is a row break only between two complete items
    clean: list[tuple[str, object, int]] = []
    for t in out:
        if t[0] == "nl":
            if clean and clean[-1][0] in ("num", "close"):
                clean.append(("slash", "\n", t[2]))
            continue
        if t[0] == "slash" and clean and clean[-1][0] == "slash" and clean[-1][1] == "\n":
            clean[-1] = t  # "1 2\n/ 3 4": one break
            continue
        clean.append(t)
    while clean and clean[-1][0] == "slash" and clean[-1][1] == "\n":
        clean.pop()
    return clean


def _simplify(n: Node) -> Node:
    """Groups of one collapse; a row inside a row (a column inside a column) merges into it."""
    if n.kind == "pic":
        return n
    kids: list[Node] = []
    for k in (_simplify(k) for k in n.kids):
        kids.extend(k.kids if k.kind == n.kind else [k])
    return kids[0] if len(kids) == 1 else Node(n.kind, kids=kids)


def parse(text: str) -> Node:
    toks = _tokens(text)
    if not toks:
        raise SpecError("配置が空です。番号を書いてください（例：1 2 3）")
    s = unicodedata.normalize("NFKC", text or "")
    pos = 0

    def where(k: int) -> str:
        a, b = max(0, k - 6), min(len(s), k + 7)
        return s[a:b].replace("\n", " ").strip()

    def peek():
        return toks[pos] if pos < len(toks) else None

    def stack() -> Node:  # items separated by "/"
        nonlocal pos
        kids = [row()]
        while peek() is not None and peek()[0] == "slash":
            pos += 1
            kids.append(row())
        return Node("col", kids=kids)

    def row() -> Node:  # items side by side
        nonlocal pos
        kids = []
        while peek() is not None and peek()[0] in ("num", "open"):
            kind, val, at = toks[pos]
            pos += 1
            if kind == "num":
                if val < 1:
                    raise SpecError("0 番の画像はありません（番号は 1 から）")
                kids.append(Node("pic", ref=val))
            else:
                inner = stack()
                if peek() is None or peek()[0] != "close":
                    raise SpecError(f"「{where(at)}」の「(」が閉じていません")
                pos += 1
                kids.append(inner)
        if not kids:
            t = peek()
            at = t[2] if t else len(s) - 1
            raise SpecError(f"「{where(at)}」のあたりに番号がありません（/ の前後、( ) の中、行頭・行末の / など）")
        return Node("row", kids=kids)

    tree = stack()
    if pos != len(toks):
        raise SpecError(f"「{where(toks[pos][2])}」の「)」に対応する「(」がありません")
    refs = refs_of(tree)
    dup = sorted({r for r in refs if refs.count(r) > 1})
    if dup:
        raise SpecError("同じ番号が2回以上あります：" + "、".join(str(d) for d in dup) + "（1つの画像は1回だけ）")
    return _simplify(tree)


def refs_of(n: Node) -> list[int]:
    """The picture numbers in reading order (left to right, top to bottom)."""
    if n.kind == "pic":
        return [n.ref]
    out: list[int] = []
    for k in n.kids:
        out.extend(refs_of(k))
    return out


def prune(n: Node, keep: set[int]) -> Node | None:
    """The arrangement with only the pictures in `keep` (empty groups vanish)."""
    if n.kind == "pic":
        return n if n.ref in keep else None
    kids = [k for k in (prune(k, keep) for k in n.kids) if k is not None]
    if not kids:
        return None
    return _simplify(Node(n.kind, kids=kids))


def renumber_text(text: str, mapping: dict[int, int]) -> str:
    """The arrangement with its picture numbers replaced through `mapping` (old -> new), so that it
    keeps showing the same pictures when they are renumbered; spacing and line breaks stay.  Numbers
    it does not know stay as they are, unless another picture now has that number."""
    import re

    taken = set(mapping.values())
    spare = (n for n in range(max(list(taken) + list(mapping) + [0]) + 1, 10 ** 6) if n not in taken)
    other: dict[int, int] = {}

    def swap(m) -> str:
        n = int(m.group(0))
        if n in mapping:
            return str(mapping[n])
        if n not in taken:
            return m.group(0)
        if n not in other:
            other[n] = next(spare)
        return str(other[n])

    return re.sub(r"\d+", swap, text or "")


def default_text(refs: list[int], lengths: list[int]) -> str:
    """Rows as text: refs [1..7] with row lengths [3, 3, 1] -> '1 2 3\\n4 5 6\\n7'."""
    lines, k = [], 0
    for n in lengths:
        lines.append(" ".join(str(r) for r in refs[k: k + n]))
        k += n
    if k < len(refs):
        lines.append(" ".join(str(r) for r in refs[k:]))
    return "\n".join(l for l in lines if l)


def coefficients(n: Node, aspect: dict[int, float], gap: float, memo: dict[int, tuple[float, float]]) -> tuple[float, float]:
    """(A, B) with width = A * height + B for this group (aspect = width / height of each picture)."""
    if n.kind == "pic":
        r = (float(aspect[n.ref]), 0.0)
    else:
        parts = [coefficients(k, aspect, gap, memo) for k in n.kids]
        g = gap * (len(parts) - 1)
        if n.kind == "row":  # same height h: widths add up
            r = (sum(a for a, _ in parts), sum(b for _, b in parts) + g)
        else:  # same width w: heights (w - b) / a add up
            s = sum(1.0 / a for a, _ in parts)
            r = (1.0 / s, (sum(b / a for a, b in parts) - g) / s)
    memo[id(n)] = r
    return r


def place(n: Node, x: float, y: float, w: float, h: float, gap: float, memo: dict[int, tuple[float, float]],
          out: dict[int, tuple[float, float, float, float]]) -> None:
    """Hand out the rectangles (x, y, w, h), in reading order, keyed by picture number."""
    if n.kind == "pic":
        out[n.ref] = (x, y, w, h)
        return
    for k in n.kids:
        a, b = memo[id(k)]
        if n.kind == "row":
            wk = a * h + b
            place(k, x, y, wk, h, gap, memo, out)
            x += wk + gap
        else:
            hk = (w - b) / a
            place(k, x, y, w, hk, gap, memo, out)
            y += hk + gap


def solve(n: Node, aspect: dict[int, float], gap: float, width: float | None = None, height: float | None = None
          ) -> tuple[float, float, dict[int, tuple[float, float, float, float]]]:
    """Content width, height and the rectangle of every picture (same units as gap).  Give the width,
    the height, or both (the arrangement is then fitted inside: the limiting one wins)."""
    memo: dict[int, tuple[float, float]] = {}
    A, B = coefficients(n, aspect, gap, memo)
    if width is not None and height is not None:
        h = (width - B) / A
        W, H = (width, h) if h <= height else (A * height + B, height)
    elif height is not None:
        W, H = A * height + B, height
    else:
        W = float(width)
        H = (W - B) / A
    rects: dict[int, tuple[float, float, float, float]] = {}
    if W > 0 and H > 0:
        place(n, 0.0, 0.0, W, H, gap, memo, rects)
    if not rects or any(r[2] <= 0 or r[3] <= 0 for r in rects.values()):
        raise ValueError("間隔に対して出力サイズが小さすぎて、この配置では画像が入りません"
                         "（間隔を小さくするか、出力を大きくしてください）")
    return W, H, rects

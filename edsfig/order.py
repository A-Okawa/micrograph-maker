"""Order of the pictures in the figure (left to right, then the next row).

The order is kept as numbers 1..n.  A number typed for one picture is read as the place it should
go to: it moves there and the pictures in between shift by one, instead of two pictures sharing a
number.
"""
from __future__ import annotations


def renumber(old: list[float], new: list[float]) -> list[int]:
    """Places 1..n after an edit.  `old` are the numbers before the edit, `new` after it (a cleared
    cell keeps its old number).  A picture moved to an earlier place goes in front of the one that
    had that number; moved to a later place, it goes behind it."""
    n = len(new)

    def key(k: int):
        moved = new[k] - old[k]
        return new[k], (-1 if moved < 0 else 1 if moved > 0 else 0), old[k], k

    places = [0] * n
    for place, k in enumerate(sorted(range(n), key=key), 1):
        places[k] = place
    return places


def move(order: list[float], used: list[bool], i: int, step: int) -> list[int] | None:
    """Places 1..n after moving picture i one place earlier (step -1) or later (+1) among the
    pictures in use (pictures left out keep their relative place).  None when it cannot move."""
    seq = sorted(range(len(order)), key=lambda k: (order[k], k))
    shown = [k for k in seq if used[k]]
    if i not in shown:
        return None
    j = shown.index(i) + (1 if step > 0 else -1)
    if not 0 <= j < len(shown):
        return None
    target = shown[j]
    seq.remove(i)
    seq.insert(seq.index(target) + (1 if step > 0 else 0), i)
    places = [0] * len(order)
    for place, k in enumerate(seq, 1):
        places[k] = place
    return places


def used_first(order: list[float], used: list[bool], switched_on=(), switched_off=()) -> list[int]:
    """Places 1..n with the pictures in use first and those left out at the bottom, each group in its
    present order.  Rows in `switched_off` (just left out) go to the very bottom, rows in
    `switched_on` (just taken back) right after the others in use."""
    seq = sorted(range(len(order)), key=lambda k: (order[k], k))
    on = [k for k in seq if used[k]]
    off = [k for k in seq if not used[k]]
    on = [k for k in on if k not in switched_on] + [k for k in on if k in switched_on]
    off = [k for k in off if k not in switched_off] + [k for k in off if k in switched_off]
    places = [0] * len(order)
    for place, k in enumerate(on + off, 1):
        places[k] = place
    return places


def from_list(text: str, n: int) -> list[int]:
    """'3,1,2' (1-based picture numbers in the wanted order) -> places 1..n; pictures not named
    follow in their current order."""
    want = []
    for part in (text or "").replace(" ", "").split(","):
        if part.isdigit() and 1 <= int(part) <= n and int(part) - 1 not in want:
            want.append(int(part) - 1)
    rest = [k for k in range(n) if k not in want]
    places = [0] * n
    for place, k in enumerate(want + rest, 1):
        places[k] = place
    return places

"""Placing order: a number typed for a picture moves it there; ◀ / ▶ step among pictures in use.

    python tests/test_order.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edsfig import order  # noqa: E402


def test_typed_number_moves_the_picture():
    old = [1, 2, 3, 4, 5]
    assert order.renumber(old, [1, 2, 1, 4, 5]) == [2, 3, 1, 4, 5]  # 3rd to the front
    assert order.renumber(old, [4, 2, 3, 4, 5]) == [4, 1, 2, 3, 5]  # 1st behind the 4th
    assert order.renumber(old, [1, 2, 3, 4, 2]) == [1, 3, 4, 5, 2]  # last to the 2nd place
    assert order.renumber(old, [99, 2, 3, 4, 5]) == [5, 1, 2, 3, 4]  # past the end: last
    assert order.renumber(old, old) == old  # nothing typed: unchanged
    assert order.renumber([1, 1, 2], [1, 1, 2]) == [1, 2, 3]  # old shared numbers: kept in list order


def test_step_among_pictures_in_use():
    o = [1, 2, 3, 4]
    used = [True, False, True, True]  # the 2nd is left out of the figure
    assert order.move(o, used, 2, -1) == [2, 3, 1, 4]  # 3rd jumps in front of the 1st (skips the unused one)
    assert order.move(o, used, 0, +1) == [3, 1, 2, 4]  # 1st goes behind the 3rd
    assert order.move(o, used, 0, -1) is None  # already first
    assert order.move(o, used, 3, +1) is None  # already last
    assert order.move(o, used, 1, +1) is None  # not in use


def test_order_from_list():
    assert order.from_list("3,1,2", 3) == [2, 3, 1]
    assert order.from_list("2", 4) == [2, 1, 3, 4]  # the rest follow
    assert order.from_list("9, 2, 2, x", 3) == [2, 1, 3]  # nonsense ignored


def test_pictures_left_out_go_to_the_bottom():
    o = [1, 2, 3, 4, 5]
    # the 2nd is switched off: it goes to the very bottom, below one already left out (the 5th)
    assert order.used_first(o, [True, False, True, True, False], switched_off={1}) == [1, 5, 2, 3, 4]
    # taken back: right after the others in use, before the ones left out
    assert order.used_first([1, 5, 2, 3, 4], [True, True, True, True, False], switched_on={1}) == [1, 4, 2, 3, 5]
    # no switch: those in use first, each group keeps its order
    assert order.used_first([3, 1, 2], [False, True, True]) == [3, 1, 2]
    assert order.used_first([1, 2, 3], [False, True, True]) == [3, 1, 2]
    # a typed place keeps its meaning among the pictures in use
    assert order.used_first([2, 1, 3, 4], [True, True, False, True]) == [2, 1, 4, 3]


if __name__ == "__main__":
    test_typed_number_moves_the_picture()
    test_step_among_pictures_in_use()
    test_order_from_list()
    test_pictures_left_out_go_to_the_bottom()
    print("ok")

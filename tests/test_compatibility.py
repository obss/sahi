"""Tests for sahi.utils.compatibility helpers."""

from __future__ import annotations

from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list


def test_fix_shift_amount_list_empty() -> None:
    """An empty shift list must not IndexError; None still defaults to [[0, 0]]."""
    assert fix_shift_amount_list([]) == []
    assert fix_shift_amount_list(None) == [[0, 0]]
    assert fix_shift_amount_list([1, 2]) == [[1, 2]]
    assert fix_shift_amount_list([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]


def test_fix_full_shape_list_empty() -> None:
    """An empty full-shape list must not IndexError; None stays None."""
    assert fix_full_shape_list([]) == []
    assert fix_full_shape_list(None) is None
    assert fix_full_shape_list([100, 200]) == [[100, 200]]
    assert fix_full_shape_list([[100, 200], [50, 60]]) == [[100, 200], [50, 60]]

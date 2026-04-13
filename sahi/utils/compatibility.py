"""Compatibility utilities for version handling."""

from __future__ import annotations


def fix_shift_amount_list(shift_amount_list: list | None) -> list[list[int | float]]:
    """Ensure shift_amount_list is in the expected nested list format.

    Compatibility for sahi v0.8.15 and earlier versions.
    """
    if shift_amount_list is None:
        return [[0, 0]]
    if isinstance(shift_amount_list[0], (int, float)):
        shift_amount_list = [shift_amount_list]
    return shift_amount_list


def fix_full_shape_list(full_shape_list: list | None) -> list[list[int | float]] | None:
    """Ensure full_shape_list is in the expected nested list format.

    Compatibility for sahi v0.8.15 and earlier versions.
    """
    if full_shape_list is not None and isinstance(full_shape_list[0], (int, float)):
        full_shape_list = [full_shape_list]
    return full_shape_list

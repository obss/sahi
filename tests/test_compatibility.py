from __future__ import annotations

from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list


def test_fix_lists() -> None:
    assert fix_shift_amount_list(None) == [[0, 0]]
    assert fix_full_shape_list(None) is None
    for fix in (fix_shift_amount_list, fix_full_shape_list):
        assert fix([]) == []
        assert fix([1, 2]) == [[1, 2]]
        assert fix([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]

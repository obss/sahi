from __future__ import annotations

from typing import Any


def create_ascii_table(data: list[list[Any]]) -> str:
    """
    Creates a clean, properly padded ASCII string grid from a list of lists.

    Args:
        data (List[List[Any]]): A list of lists representing headers and rows.

    Returns:
        str: The formatted ASCII table as a string.
    """
    if not data or not data[0]:
        return ""

    # Convert all elements to strings, handling None values as empty strings
    str_data = [[str(item) if item is not None else "" for item in row] for row in data]

    # Calculate column widths
    num_columns = max(len(row) for row in str_data)
    col_widths = [0] * num_columns
    for row in str_data:
        for i, cell in enumerate(row):
            if i < num_columns:
                col_widths[i] = max(col_widths[i], len(cell))

    # Define border part
    border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    lines = [border]
    for i, row in enumerate(str_data):
        padded_row = row + [""] * (num_columns - len(row))
        content_row = "| " + " | ".join(cell.ljust(w) for cell, w in zip(padded_row, col_widths)) + " |"
        lines.append(content_row)
        if i == 0 or i == len(str_data) - 1:
            lines.append(border)

    return "\n".join(lines)

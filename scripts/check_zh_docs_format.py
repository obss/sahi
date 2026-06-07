"""Check Chinese docs formatting against the English source docs."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ZH_DOCS_ROOT = REPO_ROOT / "docs" / "zh"

ROOT_SOURCE_OVERRIDES = {
    "README.md": REPO_ROOT / "README.md",
    "SECURITY.md": REPO_ROOT / "SECURITY.md",
    "changelog.md": REPO_ROOT / "CHANGELOG.md",
    "contributing.md": REPO_ROOT / "CONTRIBUTING.md",
}


@dataclass(frozen=True)
class Token:
    """A Markdown structure token that should stay aligned across translations."""

    kind: str
    line: int
    column: int
    detail: str
    text: str


@dataclass(frozen=True)
class DocPair:
    """English and Chinese markdown files to compare."""

    english: Path
    chinese: Path


@dataclass(frozen=True)
class Mismatch:
    """A structural mismatch between an English and Chinese doc pair."""

    pair: DocPair
    token_kind: str
    index: int
    expected: Token | None
    actual: Token | None
    anchor: Token | None


TOKEN_LABELS = {
    "admonition": "admonition",
    "bold_line": "bold line",
    "directive": "directive",
    "fence": "fenced code block",
    "heading": "heading",
    "structure": "structure token",
    "table_row": "table row",
}


def strip_frontmatter(lines: list[str]) -> tuple[list[str], int]:
    """Strip YAML frontmatter and return the remaining lines and original line offset."""
    if not lines or lines[0].strip() != "---":
        return lines, 0

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return lines[index + 1 :], index + 1

    return lines, 0


def extract_tokens(path: Path) -> list[Token]:
    """Extract comparable Markdown structure tokens from a file."""
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines, line_offset = strip_frontmatter(raw_lines)

    tokens = []

    in_fence = False
    for index, line in enumerate(lines, start=line_offset + 1):
        fence = re.match(r"^(\s*)```(\S*)", line)
        if fence:
            tokens.append(
                Token(
                    "fence",
                    index,
                    len(fence.group(1)) + 1,
                    f"indent={len(fence.group(1))},lang={fence.group(2)}",
                    line,
                )
            )
            in_fence = not in_fence
            continue

        if in_fence:
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            tokens.append(Token("heading", index, 1, f"H{len(heading.group(1))}", line))
            continue

        bold_line = re.match(r"^(\s*)\*\*[^*]+\*\*\s*$", line)
        if bold_line:
            tokens.append(
                Token(
                    "bold_line",
                    index,
                    len(bold_line.group(1)) + 1,
                    f"indent={len(bold_line.group(1))}",
                    line,
                )
            )
            continue

        admonition = re.match(r"^(\s*)(\?\?\?|!!!)\s+", line)
        if admonition:
            tokens.append(
                Token(
                    "admonition",
                    index,
                    len(admonition.group(1)) + 1,
                    f"indent={len(admonition.group(1))},marker={admonition.group(2)}",
                    line,
                )
            )
            continue

        directive = re.match(r"^(\s*):::\s*(.*)$", line)
        if directive:
            tokens.append(
                Token(
                    "directive",
                    index,
                    len(directive.group(1)) + 1,
                    f"indent={len(directive.group(1))},target={directive.group(2)}",
                    line,
                )
            )
            continue

        if re.match(r"^\s*\|.*\|\s*$", line):
            indent = len(line) - len(line.lstrip(" "))
            tokens.append(Token("table_row", index, indent + 1, f"indent={indent}", line))

    return tokens


def format_path(path: Path) -> str:
    """Format a path relative to the repository root when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def discover_pairs() -> list[DocPair]:
    """Discover Chinese docs and their English source docs."""
    pairs = []

    for chinese_path in sorted(ZH_DOCS_ROOT.rglob("*.md")):
        relative_path = chinese_path.relative_to(ZH_DOCS_ROOT)
        override = ROOT_SOURCE_OVERRIDES.get(relative_path.as_posix())
        english_path = override if override is not None else REPO_ROOT / "docs" / relative_path

        if english_path.exists():
            pairs.append(DocPair(english=english_path, chinese=chinese_path))

    return pairs


def token_signature(tokens: list[Token]) -> list[str]:
    """Return token kind and detail, ignoring translated text and line numbers."""
    return [f"{token.kind}:{token.detail}" for token in tokens]


def compare_pair(pair: DocPair) -> list[Mismatch]:
    """Compare one English/Chinese doc pair."""
    english_tokens = extract_tokens(pair.english)
    chinese_tokens = extract_tokens(pair.chinese)
    mismatches = []

    english_signature = token_signature(english_tokens)
    chinese_signature = token_signature(chinese_tokens)

    if english_signature == chinese_signature:
        return mismatches

    for operation, english_index, chinese_index in align_signatures(english_signature, chinese_signature):
        if operation == "equal" or (english_index is None and chinese_index is None):
            continue

        if operation == "replace":
            mismatches.append(
                Mismatch(
                    pair=pair,
                    token_kind=mismatch_token_kind(
                        expected=english_tokens[english_index],
                        actual=chinese_tokens[chinese_index],
                    ),
                    index=mismatch_index(
                        english_tokens=english_tokens,
                        position=english_index,
                        expected=english_tokens[english_index],
                        actual=chinese_tokens[chinese_index],
                    ),
                    expected=english_tokens[english_index],
                    actual=chinese_tokens[chinese_index],
                    anchor=chinese_tokens[chinese_index],
                )
            )
            continue

        if operation == "delete":
            mismatches.append(
                missing_mismatch(
                    pair=pair,
                    english_tokens=english_tokens,
                    chinese_tokens=chinese_tokens,
                    english_index=english_index,
                    chinese_index=chinese_index or 0,
                )
            )
            continue

        if operation == "insert":
            mismatches.append(
                extra_mismatch(
                    pair=pair,
                    chinese_tokens=chinese_tokens,
                    chinese_index=chinese_index,
                )
            )

    return mismatches


def align_signatures(english: list[str], chinese: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Align two token streams using a minimal edit-distance table."""
    english_len = len(english)
    chinese_len = len(chinese)
    costs = [[0] * (chinese_len + 1) for _ in range(english_len + 1)]
    operations = [[""] * (chinese_len + 1) for _ in range(english_len + 1)]

    for english_index in range(1, english_len + 1):
        costs[english_index][0] = english_index
        operations[english_index][0] = "delete"

    for chinese_index in range(1, chinese_len + 1):
        costs[0][chinese_index] = chinese_index
        operations[0][chinese_index] = "insert"

    priorities = {
        "equal": 0,
        "replace": 1,
        "delete": 2,
        "insert": 3,
    }

    for english_index in range(1, english_len + 1):
        for chinese_index in range(1, chinese_len + 1):
            if english[english_index - 1] == chinese[chinese_index - 1]:
                diagonal_operation = "equal"
                diagonal_cost = costs[english_index - 1][chinese_index - 1]
            else:
                diagonal_operation = "replace"
                diagonal_cost = costs[english_index - 1][chinese_index - 1] + 1

            candidates = [
                (diagonal_cost, priorities[diagonal_operation], diagonal_operation),
                (costs[english_index - 1][chinese_index] + 1, priorities["delete"], "delete"),
                (costs[english_index][chinese_index - 1] + 1, priorities["insert"], "insert"),
            ]
            cost, _, operation = min(candidates)
            costs[english_index][chinese_index] = cost
            operations[english_index][chinese_index] = operation

    aligned_operations = []
    english_index = english_len
    chinese_index = chinese_len
    while english_index > 0 or chinese_index > 0:
        operation = operations[english_index][chinese_index]
        if operation in {"equal", "replace"}:
            aligned_operations.append((operation, english_index - 1, chinese_index - 1))
            english_index -= 1
            chinese_index -= 1
        elif operation == "delete":
            aligned_operations.append((operation, english_index - 1, chinese_index))
            english_index -= 1
        elif operation == "insert":
            aligned_operations.append((operation, english_index, chinese_index - 1))
            chinese_index -= 1
        else:
            break

    aligned_operations.reverse()
    return aligned_operations


def mismatch_token_kind(expected: Token, actual: Token) -> str:
    """Return the diagnostic kind for a mismatch."""
    if expected.kind == actual.kind:
        return expected.kind

    return "structure"


def mismatch_index(
    english_tokens: list[Token],
    position: int,
    expected: Token,
    actual: Token,
) -> int:
    """Return the diagnostic index for a mismatch."""
    if expected.kind == actual.kind:
        return count_kind_before(english_tokens, position, expected.kind)

    return position


def missing_mismatch(
    pair: DocPair,
    english_tokens: list[Token],
    chinese_tokens: list[Token],
    english_index: int,
    chinese_index: int,
) -> Mismatch:
    """Build a mismatch for a missing Chinese token."""
    expected = english_tokens[english_index]
    return Mismatch(
        pair=pair,
        token_kind=expected.kind,
        index=count_kind_before(english_tokens, english_index, expected.kind),
        expected=expected,
        actual=None,
        anchor=get_anchor_token(chinese_tokens, chinese_index),
    )


def extra_mismatch(
    pair: DocPair,
    chinese_tokens: list[Token],
    chinese_index: int,
) -> Mismatch:
    """Build a mismatch for an extra Chinese token."""
    actual = chinese_tokens[chinese_index]
    return Mismatch(
        pair=pair,
        token_kind=actual.kind,
        index=count_kind_before(chinese_tokens, chinese_index, actual.kind),
        expected=None,
        actual=actual,
        anchor=actual,
    )


def count_kind_before(tokens: list[Token], position: int, kind: str) -> int:
    """Count tokens with a kind up to and including a position."""
    return sum(1 for token in tokens[: position + 1] if token.kind == kind) - 1


def get_anchor_token(tokens: list[Token], insertion_index: int) -> Token | None:
    """Return the closest Chinese token to point at when a token is missing."""
    if not tokens:
        return None

    if insertion_index > 0:
        return tokens[insertion_index - 1]

    return tokens[0]


def token_label(token_kind: str) -> str:
    """Return a human-readable token label."""
    return TOKEN_LABELS.get(token_kind, token_kind)


def print_mismatches(mismatches: list[Mismatch]) -> None:
    """Print mismatches as clang-style diagnostics."""
    for mismatch in mismatches:
        print_diagnostic(mismatch)
        print()


def print_diagnostic(mismatch: Mismatch) -> None:
    """Print a single clang-style diagnostic."""
    label = token_label(mismatch.token_kind)
    line_number, column, line_text = diagnostic_location(mismatch)
    ordinal = mismatch.index + 1

    if mismatch.expected and mismatch.actual:
        if mismatch.expected.kind == mismatch.actual.kind:
            message = (
                f"{label} #{ordinal} differs from English source: "
                f"expected {mismatch.expected.detail}, found {mismatch.actual.detail}"
            )
        else:
            message = (
                "Markdown structure differs from English source: "
                f"expected {token_label(mismatch.expected.kind)} ({mismatch.expected.detail}), "
                f"found {token_label(mismatch.actual.kind)} ({mismatch.actual.detail})"
            )
    elif mismatch.expected:
        message = f"missing {label} #{ordinal}: expected {mismatch.expected.detail} from English source"
    else:
        message = f"extra {label} #{ordinal}: found {mismatch.actual.detail} without a matching English token"

    print(f"{format_path(mismatch.pair.chinese)}:{line_number}:{column}: error: {message}")
    print_source_line(line_number, column, line_text)

    if mismatch.expected:
        expected_label = token_label(mismatch.expected.kind)
        print(
            f"{format_path(mismatch.pair.english)}:{mismatch.expected.line}:{mismatch.expected.column}: "
            f"note: expected {expected_label}: {mismatch.expected.detail}"
        )
        print_source_line(mismatch.expected.line, mismatch.expected.column, mismatch.expected.text)
    else:
        print(f"{format_path(mismatch.pair.english)}: note: remove this {label} or align it with the English source")


def diagnostic_location(mismatch: Mismatch) -> tuple[int, int, str]:
    """Return the Chinese line, column, and text for a diagnostic."""
    token = mismatch.actual or mismatch.anchor
    if token is None:
        return 1, 1, ""

    return token.line, token.column, token.text


def print_source_line(line_number: int, column: int, text: str) -> None:
    """Print a source line and caret marker."""
    line_width = max(len(str(line_number)), 1)
    caret_column = max(column, 1)
    print(f"  {line_number:>{line_width}} | {text}")
    print(f"  {' ' * line_width} | {' ' * (caret_column - 1)}^")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-pairs",
        action="store_true",
        help="List the discovered English/Chinese doc pairs without checking them.",
    )
    return parser


def main() -> int:
    """Run the Chinese docs format check."""
    args = build_parser().parse_args()
    pairs = discover_pairs()

    if args.list_pairs:
        for pair in pairs:
            print(f"{format_path(pair.english)} -> {format_path(pair.chinese)}")
        return 0

    mismatches = []
    for pair in pairs:
        mismatches.extend(compare_pair(pair))

    if mismatches:
        print("Chinese docs format mismatches found:")
        print_mismatches(mismatches)
        return 1

    print(f"Chinese docs format check passed for {len(pairs)} doc pairs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

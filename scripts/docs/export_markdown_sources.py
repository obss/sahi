"""Publish the markdown source of every docs page next to its built HTML.

Adding `.md` to any page URL then returns the source, so `/guides/sliced-inference/`
and `/guides/sliced-inference.md` are the same page in two formats. That is convenient
for reading a page in a terminal, and it gives crawlers and language models the prose
without the site chrome around it.

Also writes `llms.txt` at the site root, the conventional index that points at those
markdown files.

Run after `zensical build`, from the repository root.
"""

import pathlib
import re
import sys
from pathlib import Path

import tomllib

DOCS = Path("docs")
SITE = Path("site")
SNIPPET = re.compile(r'^(\s*)--8<--\s+"([^"]+)"\s*$', re.M)
FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
TITLE = re.compile(r"^#\s+(.+)$", re.M)


def resolve_snippets(text: str, depth: int = 0) -> str:
    """Inline `--8<-- "file"` includes so the published source is the real content."""
    if depth > 5:
        return text

    def replace(match: re.Match) -> str:
        indent, target = match.group(1), match.group(2)
        for base in (Path("."), DOCS):
            candidate = base / target
            if candidate.is_file():
                body = resolve_snippets(candidate.read_text(encoding="utf-8"), depth + 1)
                return "\n".join(indent + line for line in body.splitlines())
        return match.group(0)

    return SNIPPET.sub(replace, text)


def nav_titles(nav: object, into: dict) -> dict:
    """Collect {path: title} from the nested nav, which names pages a heading may not."""
    if isinstance(nav, dict):
        for title, value in nav.items():
            if isinstance(value, str):
                into[value] = title
            else:
                nav_titles(value, into)
    elif isinstance(nav, list):
        for item in nav:
            nav_titles(item, into)
    return into


def page_title(text: str, relative: str, from_nav: dict) -> str:
    """Prefer the page heading, fall back to the nav label, then to the file name."""
    found = TITLE.search(FRONT_MATTER.sub("", text))
    if found:
        return found.group(1).strip()
    return from_nav.get(relative, pathlib.PurePosixPath(relative).stem)


def main() -> int:
    if not SITE.is_dir():
        print("site/ not found, run zensical build first", file=sys.stderr)
        return 1

    site_url = "/"
    from_nav: dict = {}
    config = Path("zensical.toml")
    if config.is_file():
        project = tomllib.loads(config.read_text())["project"]
        site_url = project.get("site_url", "/").rstrip("/") + "/"
        nav_titles(project.get("nav", []), from_nav)

    written: list[tuple[str, str]] = []
    for source in sorted(DOCS.rglob("*.md")):
        relative = source.relative_to(DOCS)
        target = SITE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = resolve_snippets(source.read_text(encoding="utf-8"))
        target.write_text(text, encoding="utf-8")
        written.append((relative.as_posix(), page_title(text, relative.as_posix(), from_nav)))

    # English pages only: the translated trees have their own roots.
    english = [(rel, title) for rel, title in written if not rel.startswith(("zh/", "tr/"))]
    lines = [
        "# SAHI",
        "",
        "> A vision library for performing sliced inference on large images and small objects.",
        "",
        "Every page below is the markdown source of the corresponding documentation page.",
        "",
        "## Documentation",
        "",
    ]
    lines += [f"- [{title}]({site_url}{rel})" for rel, title in english]
    (SITE / "llms.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"published {len(written)} markdown sources and llms.txt with {len(english)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

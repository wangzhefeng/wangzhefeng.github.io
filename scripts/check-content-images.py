#!/usr/bin/env python3
"""Validate local image references under content/."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


CONTENT_SUFFIXES = {".md", ".rmd", ".html"}
ASSET_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".svg",
    ".avif",
    ".mp4",
}
SKIP_PREFIXES = (
    "http://",
    "https://",
    "/",
    "#",
    "data:",
    "cid:",
    "mailto:",
    "javascript:",
    "{{",
)
RISKY_CHARS = set(" ()[]{}，：！？、（）")

FENCED_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
PRE_CODE_RE = re.compile(r"<pre.*?>.*?</pre>|<code.*?>.*?</code>", re.DOTALL | re.IGNORECASE)
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\n]+)\)")
HTML_IMAGE_RE = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.IGNORECASE)
TITLE_SUFFIX_RE = re.compile(r'\s+"[^"]*"\s*$')
BLOGDOWN_POSTREF_RE = re.compile(r"^\{\{<\s*blogdown/postref\s*>\}\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="content",
        help="Directory to scan (default: content).",
    )
    return parser.parse_args()


def strip_non_content(text: str, suffix: str) -> str:
    if suffix in {".md", ".rmd"}:
        return FENCED_CODE_RE.sub("", text)
    if suffix == ".html":
        return PRE_CODE_RE.sub("", text)
    return text


def split_markdown_target(target: str) -> str:
    cleaned = target.strip()
    cleaned = TITLE_SUFFIX_RE.sub("", cleaned).strip()
    if cleaned.startswith("<") and cleaned.endswith(">"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def should_skip(ref: str) -> bool:
    if not ref:
        return True
    if ref.startswith(SKIP_PREFIXES):
        return True
    if "url_for(" in ref:
        return True
    return False


def normalize_ref(ref: str) -> str:
    ref = BLOGDOWN_POSTREF_RE.sub("", ref.strip())
    return ref


def iter_refs(text: str, suffix: str) -> list[str]:
    refs: list[str] = []
    if suffix in {".md", ".rmd"}:
        refs.extend(split_markdown_target(match.group(1)) for match in MD_IMAGE_RE.finditer(text))
        refs.extend(match.group(1).strip() for match in HTML_IMAGE_RE.finditer(text))
    elif suffix == ".html":
        refs.extend(match.group(1).strip() for match in HTML_IMAGE_RE.finditer(text))
    return refs


def find_missing_assets(root: Path) -> list[tuple[Path, str]]:
    missing: list[tuple[Path, str]] = []
    for file_path in sorted(root.rglob("*")):
        if file_path.suffix.lower() not in CONTENT_SUFFIXES or not file_path.is_file():
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        content = strip_non_content(text, file_path.suffix.lower())
        for raw_ref in iter_refs(content, file_path.suffix.lower()):
            ref = normalize_ref(raw_ref)
            if should_skip(ref):
                continue
            candidate = (file_path.parent / ref).resolve()
            if candidate.exists():
                continue
            missing.append((file_path, ref))
    return missing


def find_risky_filenames(root: Path) -> list[Path]:
    risky: list[Path] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in ASSET_SUFFIXES:
            continue
        name = file_path.name
        if any(ord(char) > 127 for char in name) or any(char in RISKY_CHARS for char in name):
            risky.append(file_path)
    return risky


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"scan root does not exist: {root}", file=sys.stderr)
        return 2

    missing = find_missing_assets(root)
    risky = find_risky_filenames(root)

    if missing:
        print("Missing local image references:")
        for file_path, ref in missing:
            print(f"  {file_path}: {ref}")
    else:
        print("No missing local image references found.")

    if risky:
        print("\nRisky asset filenames (non-ASCII or spaces/punctuation):")
        for file_path in risky:
            print(f"  {file_path}")
    else:
        print("\nNo risky asset filenames found.")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

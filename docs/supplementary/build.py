"""Validate and package the final supplement; optionally build its HTML edition.

Usage: python docs/supplementary/build.py [--latex-only]
HTML building requires Sphinx 8.1.3. This script does not compile a PDF.
"""

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "docs/supplementary/latex"
OUTPUT = ROOT / "docs/build"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_latex():
    included = set()
    texts = []

    def read(path, stack=()):
        require(path.is_file(), f"Missing LaTeX input: {path}")
        require(path not in stack, f"Cyclic LaTeX input: {path}")
        require(PROJECT.resolve() in path.resolve().parents, f"Input outside project: {path}")
        if path in included:
            return
        included.add(path)
        text = path.read_text(encoding="utf-8")
        texts.append(text)
        for name in re.findall(r"\\input\{([^}]+)\}", text):
            read(PROJECT / (name if name.endswith(".tex") else name + ".tex"), (*stack, path))

    read(PROJECT / "main.tex")
    text = "\n".join(texts)
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    bibliography = re.findall(r"\\bibitem\{([^}]+)\}", text)
    citations = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text) for key in group.split(",")}
    references = set(re.findall(r"\\(?:ref|pageref)\{([^}]+)\}", text))
    references.update(re.findall(r"\\hyperref\[([^]]+)\]", text))
    require(len(labels) == len(set(labels)), "Duplicate LaTeX labels")
    require(len(bibliography) == len(set(bibliography)), "Duplicate bibliography keys")
    require(citations <= set(bibliography), f"Missing citations: {citations-set(bibliography)}")
    require(references <= set(labels), f"Missing references: {references-set(labels)}")
    require(text.count(r"\begin{thebibliography}") == 1, "Expected one bibliography")
    for env in ["document", "longtable", "table", "tabularx", "Verbatim", "thebibliography", "itemize"]:
        require(text.count("\\begin{"+env+"}") == text.count("\\end{"+env+"}"), f"Unbalanced {env}")
    images = set()
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
        path = (PROJECT / name).resolve()
        require(PROJECT.resolve() in path.parents and path.is_file(), f"Invalid image: {name}")
        images.add(path)
    require(included == set(PROJECT.rglob("*.tex")), "Unreferenced LaTeX source files")
    print(f"LaTeX structure: {len(included)} sources, {len(labels)} unique labels, {len(citations)} citation keys.")
    return sorted(included | images | {PROJECT / "README.md"})


def archive(path, files, base):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        for source in sorted(files):
            package.write(source, source.relative_to(base))
    with zipfile.ZipFile(path) as package:
        require(package.testzip() is None, f"Corrupt archive: {path}")
    require(path.is_file() and path.stat().st_size > 0, f"Archive was not saved: {path}")
    print(f"Saved {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")


class HtmlLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.references, self.images = set(), set(), []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#"):
            self.references.add(attrs["href"][1:])
        if tag == "img":
            self.images.append(attrs.get("src", ""))


def build_html():
    try:
        from sphinx.cmd.build import main as sphinx_build
    except ImportError as error:
        raise SystemExit("Install Sphinx 8.1.3, or use --latex-only to package the TeX project.") from error
    source = ROOT / "docs/source"
    target = OUTPUT / "supplementary"
    status = sphinx_build(["-b", "singlehtml", "-n", "-W", "-E",
                           "-c", str(ROOT / "docs/supplementary"),
                           "-d", str(OUTPUT / ".sphinx_cache"), str(source), str(target)])
    require(status == 0, "Sphinx build failed")
    document = target / "supplementary_materials.html"
    text = document.read_text(encoding="utf-8")
    links = HtmlLinks()
    links.feed(text)
    require(links.references <= links.ids | {""}, f"Broken HTML anchors: {links.references-links.ids}")
    for source in links.images:
        require((target / source).is_file(), f"Missing HTML image: {source}")
    require('class="problematic"' not in text and 'system-message' not in text, "HTML parsing errors")
    print(f"HTML links: {len(links.references)} anchors verified.")
    assets = [p for p in (target / "_static").rglob("*") if p.is_file()]
    assets.extend(p for p in (target / "_images").rglob("*") if p.is_file())
    archive(OUTPUT / "exception-rules-supplementary.zip", [document, *assets], target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latex-only", action="store_true", help="Package editable LaTeX without Sphinx or PDF compilation")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    if not args.latex_only:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from sync_from_latex import sync
        sync()
    files = validate_latex()
    if not args.latex_only:
        build_html()
    archive(OUTPUT / "exception-rules-supplementary-latex.zip", files, PROJECT)

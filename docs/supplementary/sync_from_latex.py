"""Synchronise Sphinx chapters from the editable supplement's LaTeX sources.

Supports the commands used in this project. Unrecognised commands fail loudly
instead of silently dropping content. No third-party libraries are required.
"""

import argparse
import csv
from io import StringIO
from pathlib import Path
import re
import shutil
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / "docs/supplementary/latex"
DEST = ROOT / "docs/source/supplementary"


def argument(text, start):
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        raise ValueError(f"Expected a braced argument near {text[start:start+60]!r}")
    level, i = 1, start + 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            level += 1
        elif text[i] == "}":
            level -= 1
            if not level:
                return text[start+1:i], i+1
        i += 1
    raise ValueError("Unclosed LaTeX argument")


def substitute_command(text, command, callback):
    pattern = re.compile(r"\\" + command + r"\b")
    position = 0
    while match := pattern.search(text, position):
        value, end = argument(text, match.end())
        replacement = callback(value)
        text = text[:match.start()] + replacement + text[end:]
        position = match.start() + len(replacement)
    return text


def unescape(text):
    return re.sub(r"\\([{}_&%$#])", r"\1", text)


def split_cells(row):
    cells, start, level, i = [], 0, 0, 0
    while i < len(row):
        if row[i] == "\\":
            i += 2
            continue
        if row[i] == "{":
            level += 1
        elif row[i] == "}":
            level -= 1
        elif row[i] == "&" and level == 0:
            cells.append(row[start:i].strip())
            start = i + 1
        i += 1
    return [*cells, row[start:].strip()]


class Converter:
    def __init__(self, cited_keys=None):
        self.blocks = {}
        self.serial = 0
        self.tables = 0
        self.table_rows = 0
        self.code_blocks = 0
        self.math_blocks = 0
        self.cited_keys = cited_keys

    def protect(self, value):
        key = f"\ue000{self.serial}\ue001"
        self.serial += 1
        self.blocks[key] = value
        return key

    def restore(self, text):
        for _ in range(len(self.blocks)+1):
            result = re.sub(r"\ue000\d+\ue001", lambda m: self.blocks[m[0]], text)
            if result == text:
                return result
            text = result
        raise ValueError("Cyclic conversion placeholder")

    def inline(self, text):
        # Protect mathematics before interpreting braces and textual commands.
        def math(match):
            value = " ".join(match[1].strip().splitlines())
            return self.protect(":math:`" + value + "`")
        text = re.sub(r"\\\((.*?)\\\)", math, text, flags=re.S)
        text = re.sub(r"(?<!\\)\$(.*?)(?<!\\)\$", math, text, flags=re.S)
        text = re.sub(r"\\allowbreak\s*", "", text)
        text = re.sub(r"\\newblock\s*", " ", text)
        text = re.sub(r"\{\\(L|l)\}", lambda m: "Ł" if m[1] == "L" else "ł", text)
        text = re.sub(r"\\(L|l)\b(?:\{\})?", lambda m: "Ł" if m[1] == "L" else "ł", text)
        text = re.sub(r"\{\\'([A-Za-z])\}", lambda m: unicodedata.normalize("NFC", m[1] + "\u0301"), text)
        text = re.sub(r"``(.*?)''", r"“\1”", text, flags=re.S)
        # Font-switch groups are used for literal API names in the source.
        pattern = re.compile(r"\{\\ttfamily\b")
        while match := pattern.search(text):
            value, end = argument(text, match.start())
            value = re.sub(r"\\(?:ttfamily|small|footnotesize)\b\s*", "", value)
            replacement = self.protect("``" + unescape(value).strip() + "``")
            text = text[:match.start()] + replacement + text[end:]
        for command in ["texttt", "nolinkurl"]:
            text = substitute_command(text, command, lambda value: self.protect("``" + unescape(value) + "``"))
        text = substitute_command(text, "textbf", lambda value: "**" + self.inline(value).strip() + "**")
        text = substitute_command(text, "emph", lambda value: "*" + self.inline(value).strip() + "*")
        text = substitute_command(text, "cite", lambda value: " ".join("[" + key.strip() + "]_" for key in value.split(",")))
        text = re.sub(r"\\hyperref\[([^]]+)\]\{([^}]+)\}", lambda m: ":ref:`"+m[2]+" <"+m[1]+">`", text)
        text = re.sub(r"\b(Table|Section)~\\ref\{([^}]+)\}", lambda m: ":ref:`the "+m[1].lower()+" <"+m[2]+">`", text)
        text = substitute_command(text, "ref", lambda value: ":ref:`" + value + "`")
        text = substitute_command(text, "pageref", lambda value: ":ref:`" + value + "`")
        # Fail before decoding escaped backslashes or literal braces.
        unknown = re.findall(r"\\([A-Za-z]+)", text)
        if unknown:
            raise ValueError(f"Unsupported inline commands: {sorted(set(unknown))}")
        text = unescape(text).replace("~", " ")
        return self.restore(text)

    def table(self, body):
        title, labels = [], []
        body = substitute_command(body, "caption", lambda value: title.append(self.inline(value)) or "")
        body = substitute_command(body, "label", lambda value: labels.append(value) or "")
        # Skip repeated headers after the first page of a longtable.
        if r"\endfirsthead" in body:
            first, continuation = body.split(r"\endfirsthead", 1)
            if r"\endhead" not in continuation:
                raise ValueError("Longtable has a first header but no continuation boundary")
            body = first + continuation.split(r"\endhead", 1)[1]
        body = re.sub(r"\\(?:toprule|midrule|bottomrule|addlinespace|hline)\b(?:\[[^]]*\])?", "", body)
        body = re.sub(r"\\cmidrule(?:\[[^]]*\])?(?:\([^)]*\))?\{[^}]*\}", "", body)
        body = body.strip()
        rows = [split_cells(row) for row in re.split(r"\\\\(?:\[[^]]*\])?\s*", body) if row.strip()]
        if len(rows) < 2:
            raise ValueError("Table has no header or no data")
        # CSV tables have one header row. Expand grouped LaTeX headings, then
        # combine each group name with its subheading to retain its meaning.
        if any(re.match(r"\\multicolumn\b", cell) for cell in rows[0]):
            groups = []
            for cell in rows[0]:
                match = re.match(r"\\multicolumn\b", cell)
                if match:
                    count, end = argument(cell, match.end())
                    _, end = argument(cell, end)
                    group_title, end = argument(cell, end)
                    if cell[end:].strip() or not count.isdigit() or int(count) < 1:
                        raise ValueError("Unsupported multicolumn heading")
                    groups.extend([group_title] * int(count))
                else:
                    groups.append(cell)
            if len(groups) != len(rows[1]):
                raise ValueError("Grouped header and subheader widths differ")
            rows = [[" ".join(part for part in (group, detail) if part)
                     for group, detail in zip(groups, rows[1])], *rows[2:]]
        rows = [[" ".join(self.inline(cell).split()) for cell in row] for row in rows]
        columns = len(rows[0])
        if any(len(row) != columns for row in rows):
            raise ValueError(f"Inconsistent table width: {title}")
        self.tables += 1
        self.table_rows += len(rows)-1
        headers, data = rows[0], rows[1:]
        widths = [1]*columns
        if "Rule" in headers and columns == 7:
            widths = [14, 6, 44, 9, 9, 9, 9]
        elif columns == 8 and headers[:4] == ["Dataset", "N", "C", "P"]:
            widths = [28, 10, 7, 7, 12, 12, 12, 12]
        elif columns == 7 and headers[0] == "Method":
            widths = [28, 12, 12, 12, 12, 12, 12]
        elif columns == 4 and headers[0] == "No.":
            widths = [5, 19, 38, 38]
        elif columns == 3 and headers[0] == "Dataset":
            widths = [50, 25, 25]
        out = StringIO()
        csv.writer(out, lineterminator="\n").writerows(data)
        header = StringIO()
        csv.writer(header, lineterminator="").writerow(headers)
        result = "\n".join(".. _"+label+":\n" for label in labels)
        result += "\n.. csv-table:: " + (title[0] if title else "") + "\n"
        result += "   :header: " + header.getvalue() + "\n"
        result += "   :widths: " + ", ".join(map(str, widths)) + "\n\n"
        result += "".join("   "+line+"\n" for line in out.getvalue().splitlines())
        return "\n\n" + result + "\n"

    def convert(self, text):
        def code(match):
            self.code_blocks += 1
            content = match[1].strip("\r\n")
            block = "\n\n.. code-block:: text\n\n" + "\n".join("   "+line for line in content.splitlines()) + "\n\n"
            return self.protect(block)
        text = re.sub(r"\\begin\{Verbatim\}(?:\[[^]]*\])?\s*\n(.*?)\\end\{Verbatim\}", code, text, flags=re.S)
        text = re.sub(r"(?<!\\)%[^\n]*", "", text)
        def figure(match):
            body = match[1]
            graphic = re.search(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", body)
            caption = re.search(r"\\caption\b", body)
            if not graphic or not caption:
                raise ValueError("Figure requires an image and a caption")
            title, _ = argument(body, caption.end())
            source = (LATEX / graphic[1]).resolve()
            if LATEX.resolve() not in source.parents or not source.is_file():
                raise ValueError(f"Invalid figure source: {graphic[1]}")
            labels = re.findall(r"\\label\{([^}]+)\}", body)
            block = "\n\n" + "".join(".. _"+label+":\n\n" for label in labels)
            block += ".. figure:: " + graphic[1] + "\n   :width: 95%\n"
            block += "   :alt: Kaplan-Meier curves for the GBSG2 rule triple\n\n"
            block += "   " + self.inline(title) + "\n\n"
            return self.protect(block)
        text = re.sub(r"\\begin\{figure\}(?:\[[^]]*\])?(.*?)\\end\{figure\}", figure, text, flags=re.S)
        def display(match):
            self.math_blocks += 1
            return self.protect("\n\n.. math::\n\n"+"\n".join("   "+line for line in match[1].strip().splitlines())+"\n\n")
        text = re.sub(r"\\\[(.*?)\\\]", display, text, flags=re.S)
        # Read table environments using balanced column-spec arguments.
        for environment in ["longtable", "table"]:
            pattern = re.compile(r"\\begin\{" + environment + r"\}")
            while match := pattern.search(text):
                start = match.end()
                if start < len(text) and text[start] == "[":
                    start = text.index("]", start)+1
                if environment == "longtable":
                    _, start = argument(text, start)
                ending = r"\end{" + environment + "}"
                end = text.index(ending, start)
                body = text[start:end]
                if environment == "table":
                    tabular = re.search(r"\\begin\{tabularx\}", body)
                    if not tabular:
                        raise ValueError("Table wrapper has no tabularx")
                    _, after_width = argument(body, tabular.end())
                    _, after_spec = argument(body, after_width)
                    body = body[:tabular.start()] + body[after_spec:]
                    body = body.replace(r"\end{tabularx}", "")
                body = re.sub(r"\\(?:centering|small|footnotesize)\b\s*", " ", body)
                body = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", body)
                token = self.protect(self.table(body))
                text = text[:match.start()] + token + text[end+len(ending):]
        def items(match):
            chunks = re.split(r"\\item\b\s*", match[1])
            if chunks[0].strip():
                raise ValueError("Unexpected text before first list item")
            rendered = []
            for chunk in chunks[1:]:
                content = self.convert(chunk).strip().splitlines()
                rendered.append("* "+content[0]+"\n"+"\n".join("  "+line if line else "" for line in content[1:]))
            return self.protect("\n\n"+"\n\n".join(rendered)+"\n\n")
        text = re.sub(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", items, text, flags=re.S)
        def bibliography(match):
            entries = re.split(r"\\bibitem\{([^}]+)\}", match[1])
            output = []
            for i in range(1, len(entries), 2):
                key, content = entries[i], " ".join(self.inline(entries[i+1]).split())
                if self.cited_keys is None or key in self.cited_keys:
                    output.append(".. ["+key+"] "+content)
                else:
                    # Preserve uncited bibliography records without creating a
                    # dangling docutils citation (which Sphinx warns about).
                    output.append(".. _bibliography-"+key+":\n\n**["+key+"]** "+content)
            return self.protect("\n\n"+"\n\n".join(output)+"\n\n")
        text = re.sub(r"\\begin\{thebibliography\}\{[^}]+\}(.*?)\\end\{thebibliography\}", bibliography, text, flags=re.S)
        for command, adornment in [("section", "="), ("subsection", "-"), ("subsubsection", "~")]:
            text = substitute_command(text, command, lambda value, a=adornment: self.protect("\n\n"+self.inline(value)+"\n"+a*len(self.inline(value))+"\n\n"))
        text = substitute_command(text, "label", lambda value: self.protect("\n\n.. _"+value+":\n\n"))
        text = substitute_command(text, "input", lambda value: self.protect("\n\n.. include:: "+Path(value).stem+".inc\n\n"))
        text = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", text)
        text = re.sub(r"\\renewcommand\{\\arraystretch\}\{[^}]*\}", "", text)
        text = re.sub(r"\\(?:begingroup|endgroup|centering)\b[ \t]*", "", text)
        text = re.sub(r"\\(?:small|footnotesize|scriptsize)\b[ \t]*", " ", text)
        text = re.sub(r"\\(?:begin|end)\{(?:center|samepage)\}", "", text)
        text = text.replace(r"\noindent\begin{minipage}{\linewidth}", "").replace(r"\end{minipage}", "")
        text = self.inline(text)
        return re.sub(r"\n{3,}", "\n\n", self.restore(text)).strip()+"\n"


def sync(check=False):
    main = (LATEX / "main.tex").read_text(encoding="utf-8")
    inputs = re.findall(r"\\input\{([^}]+)\}", main)
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S)
    bibliography_name = re.search(r"\\renewcommand\{\\refname\}\{([^}]+)\}", main)
    bibliography_title = bibliography_name[1] if bibliography_name else "References"
    all_sources = "\n".join(path.read_text(encoding="utf-8") for path in LATEX.rglob("*.tex"))
    cited_keys = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", all_sources) for key in group.split(",")}
    pages, reports, empty_sections = {}, [], set()
    for path in sorted((LATEX / "sections").glob("*.tex")):
        converter = Converter(cited_keys)
        content = converter.convert(path.read_text(encoding="utf-8"))
        if not content.strip():
            empty_sections.add(path.stem)
            content = ":orphan:\n"
        if path.stem == "references" and not re.match(r"[^\n]+\n=+", content):
            content = ".. _supplement-references:\n\n"+bibliography_title+"\n"+"="*len(bibliography_title)+"\n\n"+content
        suffix = ".inc" if path.stem == "runtime_by_size" else ".rst"
        pages[DEST / (path.stem+suffix)] = ".. Generated from docs/supplementary/latex/sections/"+path.name+"; edit the LaTeX source.\n\n"+content
        reports.append((path.name, converter.tables, converter.table_rows, converter.code_blocks, converter.math_blocks))
    title_command = re.search(r"\\title\b", main)
    if not title_command:
        raise ValueError("The main LaTeX document has no title")
    source_title, _ = argument(main, title_command.end())
    title_lines = re.split(r"\\\\", source_title)
    title = Converter().inline(title_lines[1] if len(title_lines) > 1 else title_lines[0]).strip()
    master = title+"\n"+"="*len(title)+"\n\n"
    if abstract:
        master += Converter().convert(abstract[1])+"\n"
    master += ".. toctree::\n   :maxdepth: 2\n   :numbered:\n\n"
    master += "".join("   supplementary/"+Path(name).stem+"\n" for name in inputs
                      if Path(name).stem not in empty_sections)
    pages[ROOT / "docs/source/supplementary_materials.rst"] = master
    changed = [str(path.relative_to(ROOT)) for path, content in pages.items()
               if not path.is_file() or path.read_text(encoding="utf-8") != content]
    if check and changed:
        raise ValueError("Documentation is out of sync: "+", ".join(changed))
    if not check:
        for path, content in pages.items():
            path.write_text(content, encoding="utf-8")
    for source in (LATEX / "images").glob("*"):
        target = DEST / "images" / source.name
        if check:
            if not target.is_file() or source.read_bytes() != target.read_bytes():
                raise ValueError(f"Documentation image is out of sync: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for name, tables, rows, code, math in reports:
        print(f"{name}: {tables} tables / {rows} data rows, {code} pseudocodes, {math} display equations")
    print(f"LaTeX documentation {'checked' if check else 'synchronised'}: {len(pages)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify synchronisation without writing files")
    sync(parser.parse_args().check)

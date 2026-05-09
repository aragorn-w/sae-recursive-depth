#!/usr/bin/env python3
# Convert pandoc-emitted floats to two-column-safe forms:
#   longtable  -> table* + tabular  (head/foot reordered for tabular)
#   figure     -> figure*           (span both columns)
# Operates in place on the .tex file given as argv[1].

import pathlib
import re
import sys

if len(sys.argv) != 2:
    sys.exit("usage: twocolumn_floats.py <paper.tex>")

path = pathlib.Path(sys.argv[1])
tex = path.read_text()

tex = tex.replace(r"\begin{figure}", r"\begin{figure*}")
tex = tex.replace(r"\end{figure}", r"\end{figure*}")

# Anchor the colspec end at the `}\n\toprule` boundary instead of the first
# `}\n`. Pandoc's column spec contains nested braces (\real{0.3333}, p{...}),
# and `.*?\}\n` would otherwise stop at the first inner `}\n` and dump the
# real colspec into `body`.
longtable_re = re.compile(
    r"\\begin\{longtable\}\[\]\{(?P<colspec>.*?)\}\n(?P<body>\\toprule.*?)\\end\{longtable\}",
    re.DOTALL,
)


def fix_longtable(match: re.Match) -> str:
    colspec = match.group("colspec")
    body = match.group("body")
    body = re.sub(r"^[ \t]*\\endhead[ \t]*\n", "", body, flags=re.MULTILINE)
    body = re.sub(
        r"^[ \t]*\\bottomrule\\noalign\{\}[ \t]*\n[ \t]*\\endlastfoot[ \t]*\n",
        "",
        body,
        flags=re.MULTILINE,
    )
    body = re.sub(r"\n[ \t]*\n", "\n", body).rstrip() + "\n\\bottomrule\\noalign{}\n"
    # Pandoc sizes proportional column widths against \columnwidth (the single
    # column in twocolumn mode, ~3.4 in). Inside a table* the table spans both
    # columns (\textwidth, ~7.16 in), so retarget the math to \textwidth.
    colspec = colspec.replace(r"\columnwidth", r"\textwidth")
    # Rebalance the 3-column equal-thirds table (the experimental-matrix
    # breakdown) so the Notes column gets the bulk of the width. Markdown
    # tables can't carry per-column widths through pandoc; this is the
    # narrowest patch that keeps Notes readable on a single line.
    if colspec.count("0.3333") == 3:
        widths = ["0.18", "0.10", "0.72"]
        for w in widths:
            colspec = colspec.replace("0.3333", w, 1)
    # [!t] overrides LaTeX's float-fraction heuristics so tall floats land at
    # the next page top instead of being deferred to the end of the document.
    return (
        "\\begin{table*}[!t]\n"
        "\\centering\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        f"{body}"
        "\\end{tabular}\n"
        "\\end{table*}"
    )


tex = longtable_re.sub(fix_longtable, tex)

# IEEEtran auto-numbers subsections (A., B., ...). Markdown headings of the
# form `### A. Title` would render as "A. A. Title" — strip the manual prefix.
tex = re.sub(r"\\subsection\{[A-Z]\. ", r"\\subsection{", tex)

path.write_text(tex)

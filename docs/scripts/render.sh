#!/usr/bin/env bash
# Render WRITEUP.md to LaTeX and PDF.
#
# Usage:
#   bash scripts/render.sh            # render PDF + LaTeX (default)
#   bash scripts/render.sh pdf        # PDF only
#   bash scripts/render.sh tex        # LaTeX only
#   bash scripts/render.sh clean      # remove generated artifacts
#   bash scripts/render.sh figures    # regenerate figures from make_figures.py first
#
# Outputs (in docs/):
#   WRITEUP.pdf — rendered PDF (xelatex, IEEEtran.cls conference, biblatex)
#   WRITEUP.tex — pandoc LaTeX source via the IEEEtran template
#
# Run from the repo root or from docs/. Either works.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

WRITEUP="$DOCS_DIR/WRITEUP.md"
BIBFILE="$DOCS_DIR/references.bib"
PDF_OUT="$DOCS_DIR/WRITEUP.pdf"
TEX_OUT="$DOCS_DIR/WRITEUP.tex"

mode="${1:-all}"

cd "$DOCS_DIR"

case "$mode" in
    clean)
        rm -f WRITEUP.pdf WRITEUP.tex WRITEUP.aux WRITEUP.bbl WRITEUP.bcf \
              WRITEUP.blg WRITEUP.log WRITEUP.out WRITEUP.run.xml WRITEUP.toc \
              WRITEUP.fdb_latexmk WRITEUP.fls WRITEUP.synctex.gz \
              paper.pdf paper.tex paper.aux paper.bbl paper.bcf paper.blg \
              paper.log paper.out paper.run.xml paper.toc paper.fdb_latexmk \
              paper.fls paper.synctex.gz
        echo "cleaned generated artifacts in $DOCS_DIR"
        exit 0
        ;;
    figures)
        echo "[render] regenerating figures via make_figures.py"
        python3 "$SCRIPT_DIR/make_figures.py"
        mode=all
        ;;
esac

if [[ ! -f "$WRITEUP" ]]; then
    echo "error: $WRITEUP not found" >&2
    exit 1
fi
if [[ ! -f "$BIBFILE" ]]; then
    echo "error: $BIBFILE not found" >&2
    exit 1
fi

TEMPLATE="$SCRIPT_DIR/ieeetran.template.tex"

render_tex() {
    echo "[render] tex -> $TEX_OUT"
    pandoc "$WRITEUP" \
        --bibliography="$BIBFILE" \
        --biblatex \
        --standalone \
        --template="$TEMPLATE" \
        --shift-heading-level-by=-1 \
        -o "$TEX_OUT"
    # IEEEtran is twocolumn; longtable is illegal there. Convert to table*.
    python3 "$SCRIPT_DIR/twocolumn_floats.py" "$TEX_OUT"
    echo "[render] tex ok"
}

render_pdf() {
    render_tex
    echo "[render] pdf -> $PDF_OUT"
    pushd "$DOCS_DIR" >/dev/null
    xelatex -interaction=nonstopmode -halt-on-error WRITEUP.tex >/dev/null
    biber WRITEUP >/dev/null
    xelatex -interaction=nonstopmode -halt-on-error WRITEUP.tex >/dev/null
    xelatex -interaction=nonstopmode -halt-on-error WRITEUP.tex >/dev/null
    popd >/dev/null
    echo "[render] pdf ok ($(du -h "$PDF_OUT" | cut -f1))"
}

case "$mode" in
    all)
        render_tex
        render_pdf
        ;;
    pdf)
        render_pdf
        ;;
    tex)
        render_tex
        ;;
    *)
        echo "usage: $0 [all|pdf|tex|figures|clean]" >&2
        exit 2
        ;;
esac

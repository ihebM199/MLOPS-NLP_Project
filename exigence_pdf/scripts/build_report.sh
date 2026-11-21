#!/usr/bin/env bash
# Generate the LaTeX tables from real artefacts, then compile the report.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/export_report_tables.py

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found. Install TeX Live (e.g. apt-get install texlive-latex-extra texlive-lang-french)." >&2
  echo "Generated tables are in rapport/generated/." >&2
  exit 1
fi

cd rapport
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cp main.pdf ../rapport_technique.pdf
echo "Built ../rapport_technique.pdf"

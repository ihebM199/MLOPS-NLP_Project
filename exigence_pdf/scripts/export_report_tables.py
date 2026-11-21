"""Generate LaTeX tables from the *real* artefacts (never hard-code metrics).

    python scripts/export_report_tables.py

Reads ``reports/metrics.json`` and ``reports/comparison_table.csv`` (if present)
and writes ``rapport/generated/{metrics_table,comparison_table}.tex``.
When an artefact is missing, a clearly-marked placeholder is written.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fmt(value: object, digits: int = 4) -> str:
    if value is None or value == "":
        return "--"
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return "--"
        if isinstance(value, int):
            return str(value)
        return f"{value:.{digits}f}"
    return str(value).replace("_", r"\_")


def metrics_table(report: dict) -> str:
    test = report.get("test") or {}
    benchmark = report.get("benchmark") or {}
    threshold = (report.get("threshold") or {}).get("selected", test.get("threshold"))
    rows = [
        ("Accuracy", test.get("accuracy")),
        ("Balanced accuracy", test.get("balanced_accuracy")),
        ("Precision (fake)", test.get("precision")),
        ("Recall (fake)", test.get("recall")),
        ("F1 (fake)", test.get("f1")),
        ("AUC", test.get("auc")),
        ("EER", test.get("eer")),
        ("Seuil", threshold),
        ("Paramètres (M)", (benchmark.get("parameters") or 0) / 1e6 if benchmark.get("parameters") else None),
        ("Latence CPU batch 1 (ms)", benchmark.get("latency_ms_batch1")),
        ("MACs (G)", (benchmark.get("macs") or 0) / 1e9 if benchmark.get("macs") else None),
    ]
    body = "\n".join(f"{label} & {_fmt(value)} \\\\" for label, value in rows)
    return (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Métriques du modèle final sur le split de test (générées depuis "
        "\\texttt{reports/metrics.json}).}\n\\label{tab:metrics}\n"
        "\\begin{tabular}{lr}\n\\toprule\nMétrique & Valeur \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def comparison_table(csv_path: Path) -> str:
    if not csv_path.exists():
        return "% comparison_table.csv absent — lancer scripts/run_experiments.py.\n"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return "% comparison_table.csv vide.\n"

    columns = [
        ("run", "Run"),
        ("mode", "Mode"),
        ("test_accuracy", "Acc."),
        ("test_f1", "F1"),
        ("test_auc", "AUC"),
        ("test_eer", "EER"),
        ("latency_ms_batch1", "Latence (ms)"),
        ("parameters_m", "Params (M)"),
    ]
    header = " & ".join(label for _, label in columns) + r" \\"
    body_lines = []
    for row in rows:
        cells = [_fmt(row.get(key)) for key, _ in columns]
        body_lines.append(" & ".join(cells) + r" \\")
    return (
        "\\begin{table}[H]\n\\centering\n\\small\n"
        "\\caption{Baseline vs fusion et ablations (généré depuis "
        "\\texttt{reports/comparison\\_table.csv}).}\n\\label{tab:comparison}\n"
        "\\begin{tabular}{llrrrrrr}\n\\toprule\n"
        f"{header}\n\\midrule\n" + "\n".join(body_lines) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from reports.")
    parser.add_argument("--reports", default=str(PROJECT_ROOT / "reports"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "rapport" / "generated"))
    args = parser.parse_args(argv)

    reports = Path(args.reports)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    metrics_path = reports / "metrics.json"
    if metrics_path.exists():
        (output / "metrics_table.tex").write_text(
            metrics_table(json.loads(metrics_path.read_text("utf-8"))), "utf-8"
        )
    else:
        (output / "metrics_table.tex").write_text(
            "% metrics.json absent — lancer python -m src.models.evaluate.\n", "utf-8"
        )

    (output / "comparison_table.tex").write_text(comparison_table(reports / "comparison_table.csv"), "utf-8")
    print(f"Tables written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

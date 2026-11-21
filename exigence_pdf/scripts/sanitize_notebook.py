"""Strip outputs and secrets from a Jupyter notebook.

    python scripts/sanitize_notebook.py --input ../Deepfake_...ipynb --output ../notebooks/...
    python scripts/sanitize_notebook.py --input notebook.ipynb --in-place

Every match of ``KGAT_...`` is replaced by ``REDACTED_KAGGLE_TOKEN``. Code-cell
outputs and execution counts are cleared unless ``--keep-outputs`` is passed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TOKEN_PATTERN = re.compile(r"KGAT_[A-Za-z0-9]+")


def sanitize_source(source) -> tuple[object, int]:
    if isinstance(source, list):
        text = "".join(source)
        cleaned, count = TOKEN_PATTERN.subn("REDACTED_KAGGLE_TOKEN", text)
        return cleaned.splitlines(keepends=True), count
    text = str(source)
    cleaned, count = TOKEN_PATTERN.subn("REDACTED_KAGGLE_TOKEN", text)
    return cleaned, count


def sanitize_notebook(payload: dict, *, keep_outputs: bool) -> tuple[dict, int, int, int]:
    secrets = 0
    cleared = 0
    dropped = 0
    kept: list[dict] = []
    for cell in payload.get("cells", []):
        source = (
            "".join(cell.get("source", []))
            if isinstance(cell.get("source"), list)
            else str(cell.get("source", ""))
        )
        # Drop the dedicated credential cell and the useless API-listing cell.
        if TOKEN_PATTERN.search(source) or "kaggle competitions list" in source:
            dropped += 1
            continue
        if "source" in cell:
            cell["source"], found = sanitize_source(cell["source"])
            secrets += found
        if cell.get("cell_type") == "code" and not keep_outputs:
            if cell.get("outputs") or cell.get("execution_count") is not None:
                cleared += 1
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("widgets", None)
        kept.append(cell)
    payload["cells"] = kept
    payload.get("metadata", {}).pop("widgets", None)
    return payload, secrets, cleared, dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strip outputs and Kaggle tokens from a notebook.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--keep-outputs", action="store_true")
    args = parser.parse_args(argv)

    source = Path(args.input)
    if not source.exists():
        print(f"Notebook not found: {source}")
        return 1

    payload = json.loads(source.read_text("utf-8"))
    payload, secrets, cleared, dropped = sanitize_notebook(payload, keep_outputs=args.keep_outputs)

    if args.in_place:
        target = source
    elif args.output:
        target = Path(args.output)
    else:
        target = source.with_name(f"{source.stem}_sanitized{source.suffix}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print(
        f"Wrote {target} | secrets redacted: {secrets} | outputs cleared: {cleared} | cells dropped: {dropped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

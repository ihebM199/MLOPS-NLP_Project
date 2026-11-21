"""Export 8 demo images (4 real / 4 fake) from the official *test* split.

    python scripts/export_examples.py --dataset-root data/processed/real_vs_fake/real-vs-fake

The images are resized to 512 px and written to ``frontend/public/examples``.
They come from the test split, so they were never seen during training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src.data.dataset import LABEL2ID, build_dataframe, subset_per_class  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export demo images from the test split.")
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "data/processed/real_vs_fake/real-vs-fake"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "frontend/public/examples"))
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frame = build_dataframe(args.dataset_root, splits=("test",))
    if frame.empty:
        print(f"No test split found under {args.dataset_root}. Run scripts/download_data.py first.")
        return 1

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, label in LABEL2ID.items():
        part = subset_per_class(frame, "test", args.per_class, seed=args.seed)
        part = part[part["label"] == label]
        for index, row in enumerate(part.itertuples(index=False), start=1):
            with Image.open(row.path) as image:
                resized = image.convert("RGB")
                resized.thumbnail((args.size, args.size))
                target = output / f"{name}_{index}.jpg"
                resized.save(target, format="JPEG", quality=90)
            manifest.append({"file": target.name, "label": name, "source": row.path})

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    print(f"Exported {len(manifest)} examples to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit an image-classification dataset at any local filesystem path.

Example:
    python3 -m src.audit --data-dir "/Volumes/GoogleDrive/My Drive/thai_data"

The expected convention is that an image's label is its parent folder. For a
different layout, use --label-level to choose a higher ancestor folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.paths import get_data_dir, prepare_image_directory

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # Keep --help and import errors understandable before setup.
    Image = None  # type: ignore[assignment]
    UnidentifiedImageError = OSError


DEFAULT_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_extensions(value: str) -> set[str]:
    """Return normalized extensions from a comma-separated CLI value."""
    extensions = {item.strip().lower() for item in value.split(",") if item.strip()}
    return {item if item.startswith(".") else f".{item}" for item in extensions}


def image_paths(data_dir: Path, extensions: set[str]) -> Iterable[Path]:
    """Yield non-hidden image files recursively in a deterministic order."""
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if any(part.startswith(".") for part in path.relative_to(data_dir).parts):
            continue
        yield path


def label_for_path(path: Path, data_dir: Path, label_level: int) -> str:
    """Read label from the Nth parent directory above an image file.

    label_level=1 maps ``root/class/image.jpg`` to ``class``.
    label_level=2 maps ``root/class/variant/image.jpg`` to ``class``.
    """
    relative_parents = path.relative_to(data_dir).parts[:-1]
    if len(relative_parents) < label_level:
        raise ValueError(
            f"{path} has only {len(relative_parents)} directory level(s) below "
            f"{data_dir}; cannot use --label-level {label_level}."
        )
    return relative_parents[-label_level]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> tuple[int, int, str]:
    """Validate an image and return width, height, and color mode."""
    if Image is None:
        raise RuntimeError("Pillow is required. Install dependencies with: python3 -m pip install -r requirements.txt")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.width, image.height, image.mode
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(str(error)) from error


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_dataset(data_dir: Path, output_dir: Path, label_level: int, extensions: set[str]) -> dict[str, Any]:
    """Audit images at data_dir and write reproducible manifests and a JSON report."""
    data_dir = prepare_image_directory(data_dir, extensions)
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist or is not a folder: {data_dir}")
    if label_level < 1:
        raise ValueError("--label-level must be at least 1")
    if Image is None:
        raise RuntimeError("Pillow is required. Install dependencies with: python3 -m pip install -r requirements.txt")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    corrupt: list[dict[str, str]] = []
    label_errors: list[dict[str, str]] = []
    class_counts: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    hashes: dict[str, list[str]] = {}

    paths = list(image_paths(data_dir, extensions))
    for path in paths:
        try:
            label = label_for_path(path, data_dir, label_level)
        except ValueError as error:
            label_errors.append({"path": str(path), "error": str(error)})
            continue
        try:
            width, height, mode = inspect_image(path)
            content_hash = sha256_file(path)
        except (OSError, ValueError) as error:
            corrupt.append({"path": str(path), "error": str(error)})
            continue

        class_counts[label] += 1
        modes[mode] += 1
        resolutions[f"{width}x{height}"] += 1
        hashes.setdefault(content_hash, []).append(str(path))
        records.append(
            {
                "path": str(path.resolve()),
                "label": label,
                "width": width,
                "height": height,
                "mode": mode,
                "sha256": content_hash,
            }
        )

    exact_duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    ordered_counts = sorted(class_counts.items(), key=lambda item: (item[1], item[0]))
    minimum = min(class_counts.values()) if class_counts else None
    maximum = max(class_counts.values()) if class_counts else None
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir.resolve()),
        "label_level": label_level,
        "extensions": sorted(extensions),
        "discovered_supported_files": len(paths),
        "valid_images": len(records),
        "corrupt_images": len(corrupt),
        "label_errors": len(label_errors),
        "class_count": len(class_counts),
        "images_per_class": dict(sorted(class_counts.items())),
        "minimum_images_per_class": minimum,
        "maximum_images_per_class": maximum,
        "imbalance_ratio_max_over_min": (maximum / minimum) if minimum else None,
        "image_modes": dict(sorted(modes.items())),
        "top_resolutions": dict(resolutions.most_common(20)),
        "exact_duplicate_groups": len(exact_duplicate_groups),
        "images_in_exact_duplicate_groups": sum(len(group) for group in exact_duplicate_groups),
        "notes": [
            "Exact duplicates use SHA-256 hashes. Near-duplicate detection is not included in this first audit.",
            "Inspect image_manifest.csv and class_counts.csv before deciding transformations or class weights.",
        ],
    }

    write_csv(output_dir / "image_manifest.csv", ["path", "label", "width", "height", "mode", "sha256"], records)
    write_csv(
        output_dir / "class_counts.csv",
        ["label", "image_count"],
        [{"label": label, "image_count": count} for label, count in ordered_counts],
    )
    write_csv(output_dir / "corrupt_images.csv", ["path", "error"], corrupt)
    write_csv(output_dir / "label_errors.csv", ["path", "error"], label_errors)
    write_csv(
        output_dir / "exact_duplicate_groups.csv",
        ["sha256", "path"],
        [
            {"sha256": digest, "path": path}
            for digest, group in hashes.items()
            if len(group) > 1
            for path in group
        ],
    )
    (output_dir / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a local image-classification dataset at any path.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional dataset directory. If omitted, uses THAI_CHAR_DATA_DIR or repository data/raw.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/audit"), help="Directory for CSV/JSON artifacts.")
    parser.add_argument(
        "--label-level",
        type=int,
        default=1,
        help="Directory level above each image that represents its label (default: 1 = parent folder).",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(DEFAULT_EXTENSIONS)),
        help="Comma-separated image extensions to scan.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        data_dir = args.data_dir if args.data_dir else get_data_dir()
        report = audit_dataset(data_dir, args.output_dir, args.label_level, parse_extensions(args.extensions))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        return 2
    print(
        "Audit complete: "
        f"{report['valid_images']} valid images, {report['class_count']} classes, "
        f"{report['corrupt_images']} corrupt images. Results: {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

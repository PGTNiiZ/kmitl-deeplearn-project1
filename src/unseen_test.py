"""Create and crop a writer-independent 72-class test set.

The generated form is ordered by the numeric prefix in each label (0_ก, 1_ข,
...).  A photographed form is rectified from four manually selected grid
corners, then split into one isolated-character image per class.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


A4_SIZE = (2480, 3508)  # 300 DPI
GRID_BOX = (120, 400, 2360, 3280)
GRID_COLUMNS = 8
LABEL_BAND = 0.16
DEFAULT_LABELS = Path("data/splits/clean_32x32/label_to_index.json")
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def load_labels(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or len(mapping) != 72:
        raise ValueError(f"Expected a 72-class label mapping: {path}")
    if sorted(mapping.values()) != list(range(72)):
        raise ValueError("Label mapping indices must be exactly 0..71")
    try:
        return sorted(mapping, key=lambda label: int(label.split("_", 1)[0]))
    except (ValueError, IndexError) as exc:
        raise ValueError("Labels must begin with a numeric prefix such as '0_ก'") from exc


def _font(path: Path | None, size: int) -> ImageFont.FreeTypeFont:
    candidates = ([path] if path else []) + [
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    raise FileNotFoundError("No Thai-capable font found; pass --font /path/to/font.ttf")


def _display_label(label: str) -> str:
    number, character = label.split("_", 1)
    if character and unicodedata.category(character[0]).startswith("M"):
        character = "◌" + character
    return f"{number}  {character}"


def make_form(labels: list[str], output: Path, font_path: Path | None = None) -> None:
    page = Image.new("RGB", A4_SIZE, "white")
    draw = ImageDraw.Draw(page)
    title_font, label_font, note_font = _font(font_path, 54), _font(font_path, 31), _font(font_path, 28)
    draw.text((120, 70), "Thai Character Unseen-Writer Test Form — 72 classes", fill="black", font=title_font)
    draw.text((120, 160), "Writer ID: ____________________   Date: __________   Pen: __________", fill="black", font=note_font)
    draw.text((120, 215), "Write ONE character in each box. Do not copy the small printed prompt.", fill="black", font=note_font)
    draw.text((120, 260), "Photograph the whole grid; keep all four black corner squares visible.", fill="black", font=note_font)

    left, top, right, bottom = GRID_BOX
    rows = math.ceil(len(labels) / GRID_COLUMNS)
    cell_w, cell_h = (right - left) / GRID_COLUMNS, (bottom - top) / rows
    for column in range(GRID_COLUMNS + 1):
        x = round(left + column * cell_w)
        draw.line((x, top, x, bottom), fill="black", width=4)
    for row in range(rows + 1):
        y = round(top + row * cell_h)
        draw.line((left, y, right, y), fill="black", width=4)
    for index, label in enumerate(labels):
        row, column = divmod(index, GRID_COLUMNS)
        x, y = round(left + column * cell_w), round(top + row * cell_h)
        draw.text((x + 10, y + 5), _display_label(label), fill=(150, 150, 150), font=label_font)
        band_y = round(y + cell_h * LABEL_BAND)
        draw.line((x + 4, band_y, round(x + cell_w) - 4, band_y), fill=(220, 220, 220), width=2)

    marker = 32
    for x, y in ((left, top), (right, top), (right, bottom), (left, bottom)):
        draw.rectangle((x - marker, y - marker, x + marker, y + marker), fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    page.save(output, dpi=(300, 300))


def parse_corners(value: str) -> list[tuple[float, float]]:
    try:
        points = [tuple(map(float, item.split(","))) for item in value.split()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use 'x,y x,y x,y x,y'") from exc
    if len(points) != 4 or any(len(point) != 2 for point in points):
        raise argparse.ArgumentTypeError("Provide TL TR BR BL as 'x,y x,y x,y x,y'")
    return points


def _solve(matrix: list[list[float]], values: list[float]) -> list[float]:
    """Solve a small linear system with pivoted Gaussian elimination."""
    rows = [row[:] + [value] for row, value in zip(matrix, values)]
    for column in range(len(values)):
        pivot = max(range(column, len(rows)), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-10:
            raise ValueError("Corner points do not form a usable quadrilateral")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for row in range(len(rows)):
            if row == column:
                continue
            factor = rows[row][column]
            rows[row] = [a - factor * b for a, b in zip(rows[row], rows[column])]
    return [row[-1] for row in rows]


def perspective_coefficients(corners: list[tuple[float, float]], size: tuple[int, int]) -> list[float]:
    width, height = size
    destinations = [(0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))]
    matrix, values = [], []
    for (u, v), (x, y) in zip(destinations, corners):
        matrix.extend(([u, v, 1, 0, 0, 0, -x * u, -x * v],
                       [0, 0, 0, u, v, 1, -y * u, -y * v]))
        values.extend((x, y))
    return _solve(matrix, values)


def build_manifest(output_dir: Path) -> dict[str, int]:
    rows = []
    for metadata_path in sorted((output_dir / "pages").glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for item in metadata["crops"]:
            image_path = output_dir / item["path"]
            rows.append({**item, "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                         "writer_id": metadata["writer_id"], "capture_id": metadata["capture_id"],
                         "source_page": metadata["source_page"]})
    fields = ["path", "label", "writer_id", "capture_id", "source_page", "row", "column", "sha256"]
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["label"] for row in rows)
    report = {"images": len(rows), "writers": len({row["writer_id"] for row in rows}),
              "classes": len(counts), "minimum_per_class": min(counts.values(), default=0),
              "maximum_per_class": max(counts.values(), default=0)}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def score_predictions(dataset_dir: Path, predictions_path: Path, output_path: Path) -> dict:
    with (dataset_dir / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        truth = list(csv.DictReader(stream))
    with predictions_path.open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    if not truth:
        raise ValueError("Test manifest is empty")
    if len(predictions) != len(truth):
        raise ValueError(f"Prediction count {len(predictions)} does not match test images {len(truth)}")
    prediction_by_path = {Path(row["path"]).as_posix(): row for row in predictions}
    if len(prediction_by_path) != len(predictions):
        raise ValueError("Prediction CSV contains duplicate paths")
    scored = []
    for item in truth:
        relative = Path(item["path"]).as_posix()
        matches = [row for path, row in prediction_by_path.items()
                   if path == relative or path.endswith("/" + relative)]
        if len(matches) != 1:
            raise ValueError(f"Expected one prediction for {relative}, found {len(matches)}")
        predicted = matches[0].get("prediction", "")
        scored.append({**item, "prediction": predicted, "status": matches[0].get("status", ""),
                       "confidence": matches[0].get("confidence", ""),
                       "correct": predicted == item["label"]})
    by_class = {}
    for label in sorted({item["label"] for item in scored}, key=lambda value: int(value.split("_", 1)[0])):
        items = [item for item in scored if item["label"] == label]
        by_class[label] = sum(item["correct"] for item in items) / len(items)
    by_writer = {}
    for writer_id in sorted({item["writer_id"] for item in scored}):
        items = [item for item in scored if item["writer_id"] == writer_id]
        by_writer[writer_id] = sum(item["correct"] for item in items) / len(items)
    report = {"images": len(scored), "accuracy": sum(item["correct"] for item in scored) / len(scored),
              "macro_recall": sum(by_class.values()) / len(by_class),
              "per_class_recall": by_class, "per_writer_accuracy": by_writer}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["path", "label", "prediction", "correct", "confidence", "status", "writer_id", "capture_id"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored)
    return report


def crop_form(source: Path, corners: list[tuple[float, float]], labels: list[str], output_dir: Path,
              writer_id: str, capture_id: str, inner_margin: float = 0.07) -> dict[str, int]:
    if not SAFE_ID.fullmatch(writer_id) or not SAFE_ID.fullmatch(capture_id):
        raise ValueError("writer-id and capture-id may contain only letters, numbers, '_' and '-'")
    if not 0 <= inner_margin < 0.3:
        raise ValueError("inner-margin must be between 0 and 0.3")
    stem = f"{writer_id}__{capture_id}"
    pages_dir = output_dir / "pages"
    metadata_path = pages_dir / f"{stem}.json"
    if metadata_path.exists():
        raise FileExistsError(f"This writer/capture already exists: {metadata_path}")
    rows = math.ceil(len(labels) / GRID_COLUMNS)
    rectified_size = (GRID_COLUMNS * 300, rows * 300)
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        rectified = image.transform(rectified_size, Image.Transform.PERSPECTIVE,
                                    perspective_coefficients(corners, rectified_size),
                                    resample=Image.Resampling.BICUBIC)
    pages_dir.mkdir(parents=True, exist_ok=True)
    rectified_path = pages_dir / f"{stem}.jpg"
    rectified.save(rectified_path, quality=94)
    crops = []
    for index, label in enumerate(labels):
        row, column = divmod(index, GRID_COLUMNS)
        x0, x1 = column * 300, (column + 1) * 300
        y0, y1 = row * 300, (row + 1) * 300
        x_pad = round(300 * inner_margin)
        y_pad = round(300 * inner_margin)
        box = (x0 + x_pad, y0 + round(300 * LABEL_BAND) + y_pad, x1 - x_pad, y1 - y_pad)
        relative = Path("images") / label / f"{stem}.png"
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        rectified.crop(box).save(target)
        crops.append({"path": relative.as_posix(), "label": label, "row": row, "column": column})
    metadata = {"writer_id": writer_id, "capture_id": capture_id, "source_page": str(source.resolve()),
                "corners_tl_tr_br_bl": corners, "rectified_page": rectified_path.relative_to(output_dir).as_posix(),
                "crops": crops}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return build_manifest(output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="72-class label_to_index.json")
    commands = parser.add_subparsers(dest="command", required=True)
    form = commands.add_parser("make-form", help="Create a printable A4 PNG/PDF form")
    form.add_argument("--output", type=Path, default=Path("data/unseen_test/form_72.png"))
    form.add_argument("--font", type=Path)
    crop = commands.add_parser("crop", help="Rectify and split one photographed form")
    crop.add_argument("--input", type=Path, required=True)
    crop.add_argument("--corners", type=parse_corners, required=True, metavar='"TL_X,TL_Y TR_X,TR_Y BR_X,BR_Y BL_X,BL_Y"')
    crop.add_argument("--writer-id", required=True)
    crop.add_argument("--capture-id", default="natural")
    crop.add_argument("--output-dir", type=Path, default=Path("data/unseen_test/collected"))
    crop.add_argument("--inner-margin", type=float, default=0.07)
    score = commands.add_parser("score", help="Score an inference CSV against the frozen manifest")
    score.add_argument("--dataset-dir", type=Path, default=Path("data/unseen_test/collected"))
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, default=Path("results/unseen_test/score.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        labels = load_labels(args.labels)
        if args.command == "make-form":
            make_form(labels, args.output, args.font)
            print(f"Created printable 72-class form: {args.output}")
        elif args.command == "crop":
            report = crop_form(args.input, args.corners, labels, args.output_dir,
                               args.writer_id, args.capture_id, args.inner_margin)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            report = score_predictions(args.dataset_dir, args.predictions, args.output)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

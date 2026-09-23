"""Build a balanced 72-class public test set from isolated Thai handwriting."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import tarfile
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

SOURCE_DIR = Path("data/unseen_test/burapha_source")
OUTPUT_DIR = Path("data/unseen_test/burapha_72")
DEFAULT_MODEL_LABELS = Path("results/model_search/full/e90d3777438a/export/label_to_index.json")
DEFAULT_TRAIN_MANIFEST = Path("results/model_search/full/audit/image_manifest.csv")
BURAPHA_ARCHIVES = (SOURCE_DIR / "character-test.zip", SOURCE_DIR / "digit-test.zip")
ALICE_ARCHIVE = SOURCE_DIR / "alice-thi.tar.gz"
SOURCE_URLS = {
    BURAPHA_ARCHIVES[0]: "https://services.informatics.buu.ac.th/datasets/Burapha-TH/character/20210306-test.zip",
    BURAPHA_ARCHIVES[1]: "https://services.informatics.buu.ac.th/datasets/Burapha-TH/digit/20210307-test.zip",
    ALICE_ARCHIVE: "https://www.ai.rug.nl/~mrolarik/ALICE-THI/ALICE-THI-Dataset.tar.gz",
}
IAPP_ROWS = range(1450, 1550)
IAPP_EXCLUDED_ROWS = {
    1450, 1453, 1455, 1470, 1472, 1479, 1480, 1483, 1484, 1485, 1486,
    1487, 1489, 1490, 1492, 1493, 1494, 1496, 1499, 1501, 1503, 1505,
    1506, 1509, 1510, 1511, 1516, 1518, 1520, 1521, 1523, 1525, 1526,
    1527, 1528, 1531, 1536, 1540, 1543, 1548,
}
IAPP_TEXT = "จงฝ่าฟันพัฒนาวิชาการ อย่าล้างผลาญฤๅเข่นฆ่าบีฑาใคร"
CODE_PATTERN = re.compile(r"-(\d{3})-[0-9A-F]{2}-")


def _download_public_archives() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in SOURCE_URLS.items():
        if not path.is_file():
            print(f"Downloading {url} -> {path}")
            urllib.request.urlretrieve(url, path)


def tis_code(label: str) -> int:
    if label.isdigit():
        return int(label)
    character = label.split("_", 1)[1]
    encoded = character.encode("tis-620")
    if len(encoded) != 1:
        raise ValueError(f"Label is not one TIS-620 character: {label}")
    return encoded[0]


def load_model_labels(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or len(mapping) != 72 or sorted(mapping.values()) != list(range(72)):
        raise ValueError(f"Expected a 72-class model label mapping: {path}")
    return [label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]


def _burapha_members() -> tuple[dict[int, list[tuple[Path, str]]], dict[Path, zipfile.ZipFile]]:
    archives, members = {}, defaultdict(list)
    for path in BURAPHA_ARCHIVES:
        if not path.is_file():
            raise FileNotFoundError(f"Missing official test archive: {path}")
        archive = archives[path] = zipfile.ZipFile(path)
        for name in archive.namelist():
            match = CODE_PATTERN.search(name)
            if match and name.lower().endswith((".jpg", ".jpeg", ".png")):
                members[int(match.group(1))].append((path, name))
    return members, archives


def _alice_members() -> tuple[dict[int, list[tarfile.TarInfo]], tarfile.TarFile, tarfile.TarFile]:
    if not ALICE_ARCHIVE.is_file():
        raise FileNotFoundError(f"Missing ALICE-THI archive: {ALICE_ARCHIVE}")
    outer = tarfile.open(ALICE_ARCHIVE, "r:gz")
    nested = next((item for item in outer.getmembers() if item.name.endswith("ALICE-THI Dataset.tar.gz")), None)
    if nested is None:
        raise ValueError("ALICE-THI archive has no nested image archive")
    inner = tarfile.open(fileobj=io.BytesIO(outer.extractfile(nested).read()), mode="r:gz")  # type: ignore[union-attr]
    members = defaultdict(list)
    for item in inner.getmembers():
        match = CODE_PATTERN.search(item.name)
        if match and item.isfile() and item.name.lower().endswith((".jpg", ".jpeg", ".png")):
            members[int(match.group(1))].append(item)
    return members, outer, inner


def _ink_bounds(image: Image.Image) -> tuple[int, int, int, int]:
    gray = ImageOps.grayscale(image)
    pixels, width, height = gray.load(), *gray.size
    row_counts = [sum(pixels[x, y] < 190 for x in range(width)) for y in range(height)]
    ys = [y for y, count in enumerate(row_counts) if count >= max(2, width // 500)]
    if not ys:
        raise ValueError("No handwriting detected in iApp source image")
    top, bottom = min(ys), max(ys) + 1
    column_counts = [sum(pixels[x, y] < 190 for y in range(top, bottom)) for x in range(width)]
    xs = [x for x, count in enumerate(column_counts) if count >= 2]
    if not xs:
        raise ValueError("No handwriting columns detected in iApp source image")
    return min(xs), top, max(xs) + 1, bottom


def trim_to_ink(image: Image.Image, padding: float = 0.15) -> Image.Image:
    left, top, right, bottom = _ink_bounds(image)
    pad_x = max(2, round((right - left) * padding))
    pad_y = max(2, round((bottom - top) * padding))
    return image.crop((max(0, left - pad_x), max(0, top - pad_y),
                       min(image.width, right + pad_x), min(image.height, bottom + pad_y))).convert("RGB")


def crop_lakkhangyao(image: Image.Image) -> Image.Image:
    """Crop ๅ from the fixed BEST2019 sentence used by rows 1450..1549."""
    left, top, right, bottom = _ink_bounds(image)
    gray = ImageOps.grayscale(image)
    pixels = gray.load()
    profile = [sum(pixels[x, y] < 190 for y in range(top, bottom)) for x in range(left, right)]
    gaps, start = [], None
    for offset, count in enumerate(profile + [99]):
        if count <= 1 and start is None:
            start = offset
        elif count > 1 and start is not None:
            gaps.append((left + start, left + offset))
            start = None
    expected_space = left + (right - left) * 0.41
    phrase_gap = min((gap for gap in gaps if left + (right - left) * 0.25 < sum(gap) / 2 < left + (right - left) * 0.58),
                     key=lambda gap: abs(sum(gap) / 2 - expected_space) - (gap[1] - gap[0]) * 2,
                     default=(round(expected_space), round(expected_space)))
    phrase_left = phrase_gap[1]
    center = phrase_left + (right - phrase_left) * 0.50
    unit = (right - phrase_left) / 23
    left_gaps = [gap for gap in gaps if center - 1.4 * unit <= sum(gap) / 2 < center - 0.15 * unit]
    right_gaps = [gap for gap in gaps if center + 0.15 * unit < sum(gap) / 2 <= center + 1.4 * unit]
    crop_left = max((gap[1] for gap in left_gaps), default=round(center - unit * 0.75))
    crop_right = min((gap[0] for gap in right_gaps), default=round(center + unit * 0.75))
    pad_x, pad_y = max(2, round(unit * 0.2)), max(2, round((bottom - top) * 0.08))
    if crop_right <= crop_left:
        crop_left, crop_right = round(center - unit * 0.75), round(center + unit * 0.75)
    broad = image.crop((max(0, crop_left - pad_x), max(0, top - pad_y),
                        min(image.width, crop_right + pad_x), min(image.height, bottom + pad_y))).convert("RGB")
    return _central_component(broad)


def _central_component(image: Image.Image) -> Image.Image:
    """Remove neighbouring glyphs accidentally included by horizontal alignment."""
    gray = ImageOps.grayscale(image)
    mask = gray.point(lambda value: 255 if value < 200 else 0).filter(ImageFilter.MaxFilter(3))
    pixels, width, height = mask.load(), *mask.size
    seen, components = set(), []
    for y in range(height):
        for x in range(width):
            if not pixels[x, y] or (x, y) in seen:
                continue
            stack, points = [(x, y)], []
            seen.add((x, y))
            while stack:
                px, py = stack.pop()
                points.append((px, py))
                for nx, ny in ((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask.getpixel((nx, ny)) and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            if len(points) >= 6:
                components.append(points)
    if not components:
        return image
    largest = max(map(len, components))
    usable = [points for points in components if len(points) >= largest * 0.25]
    points = min(usable, key=lambda item: abs(sum(x for x, _ in item) / len(item) - width / 2))
    left, right = min(x for x, _ in points), max(x for x, _ in points) + 1
    top, bottom = min(y for _, y in points), max(y for _, y in points) + 1
    pad = max(2, round((bottom - top) * 0.12))
    return image.crop((max(0, left - pad), max(0, top - pad),
                       min(width, right + pad), min(height, bottom + pad)))


def _download_iapp_sources() -> dict[int, Path]:
    target = SOURCE_DIR / "iapp_lakkhangyao"
    target.mkdir(parents=True, exist_ok=True)
    cached = {row_id: target / f"row_{row_id}.jpg" for row_id in IAPP_ROWS}
    if all(path.is_file() for path in cached.values()):
        return cached
    query = urllib.parse.urlencode({"dataset": "iapp/thai_handwriting_dataset", "config": "default",
                                    "split": "train", "offset": min(IAPP_ROWS), "length": len(IAPP_ROWS)})
    with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{query}", timeout=60) as response:
        rows = json.load(response)["rows"]
    result = {}
    for item in rows:
        row_id, row = item["row_idx"], item["row"]
        if row_id not in IAPP_ROWS or row["text"] != IAPP_TEXT:
            raise ValueError(f"Unexpected iApp row {row_id}; source dataset may have changed")
        path = target / f"row_{row_id}.jpg"
        if not path.is_file():
            with urllib.request.urlopen(row["image"]["src"], timeout=60) as response:
                path.write_bytes(response.read())
        result[row_id] = path
    if set(result) != set(IAPP_ROWS):
        raise ValueError("Hugging Face did not return all requested ๅ source rows")
    return result


def _write_review_grid(paths: list[Path], output: Path) -> None:
    cell = 112
    canvas = Image.new("RGB", (cell * 10, cell * ((len(paths) + 9) // 10)), "white")
    draw = ImageDraw.Draw(canvas)
    multiple_classes = len({path.parent for path in paths}) > 1
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = ImageOps.contain(source.convert("RGB"), (96, 86))
        x, y = index % 10 * cell, index // 10 * cell
        canvas.paste(image, (x + (96 - image.width) // 2, y))
        draw.text((x + 2, y + 90), path.parent.name if multiple_classes else path.stem, fill="black")
    canvas.save(output)


def build_public_test(labels_path: Path = DEFAULT_MODEL_LABELS, output_dir: Path = OUTPUT_DIR,
                      per_class: int = 50, train_manifest: Path = DEFAULT_TRAIN_MANIFEST) -> dict:
    if per_class < 1:
        raise ValueError("per-class must be positive")
    labels = load_model_labels(labels_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _download_public_archives()
    burapha, zip_archives = _burapha_members()
    alice, alice_outer, alice_inner = _alice_members()
    iapp = _download_iapp_sources()
    train_hashes = ({row["sha256"] for row in csv.DictReader(train_manifest.open(encoding="utf-8"))}
                    if train_manifest.is_file() else set())
    rows, source_counts, review_paths, preview_paths, training_overlaps = [], Counter(), [], [], 0
    try:
        for label in labels:
            code = tis_code(label)
            selected: list[tuple[str, str, bytes | Image.Image]] = []
            if code in burapha:
                for archive_path, member in sorted(burapha[code], key=lambda item: item[1])[:per_class]:
                    writer_match = re.search(r"P-(\d+)", member)
                    writer_id = f"burapha_{writer_match.group(1) if writer_match else Path(member).stem}"
                    selected.append(("burapha", writer_id, zip_archives[archive_path].read(member)))
            elif code == 230 and code in alice:
                for member in sorted(alice[code], key=lambda item: item.name)[:per_class]:
                    sample = Path(member.name).stem.rsplit("-", 1)[-1]
                    selected.append(("alice", f"alice_{sample}", alice_inner.extractfile(member).read()))  # type: ignore[union-attr]
            elif code == 229:
                candidates = [(row_id, path) for row_id, path in sorted(iapp.items())
                              if row_id not in IAPP_EXCLUDED_ROWS]
                for row_id, path in candidates[:per_class]:
                    with Image.open(path) as source:
                        selected.append(("iapp", f"best2019_{row_id}", crop_lakkhangyao(source.convert("RGB"))))
            if len(selected) < per_class:
                raise ValueError(f"Only {len(selected)} public samples available for {label}")
            for index, (source_name, writer_id, payload) in enumerate(selected):
                relative = Path("images") / label / f"{source_name}_{index:03}.png"
                target = output_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(payload, Image.Image):
                    image = payload
                else:
                    with Image.open(io.BytesIO(payload)) as source:
                        image = trim_to_ink(source.convert("RGB"))
                image.save(target)
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                source_digest = (hashlib.sha256(payload).hexdigest()
                                 if isinstance(payload, bytes) else digest)
                training_overlaps += source_digest in train_hashes or digest in train_hashes
                rows.append({"path": relative.as_posix(), "label": label, "writer_id": writer_id,
                             "capture_id": source_name, "source_page": source_name,
                             "row": "", "column": "", "sha256": digest})
                source_counts[source_name] += 1
                if code == 229:
                    review_paths.append(target)
                if index == 0:
                    preview_paths.append(target)
    finally:
        for archive in zip_archives.values():
            archive.close()
        alice_inner.close()
        alice_outer.close()
    fields = ["path", "label", "writer_id", "capture_id", "source_page", "row", "column", "sha256"]
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["label"] for row in rows)
    duplicate_images = len(rows) - len({row["sha256"] for row in rows})
    report = {"images": len(rows), "classes": len(counts), "per_class": per_class,
              "minimum_per_class": min(counts.values()), "maximum_per_class": max(counts.values()),
              "sources": dict(source_counts), "exact_duplicate_images": duplicate_images,
              "exact_training_overlaps": training_overlaps,
              "notes": ["Burapha samples use the official test split.",
                        "Class 56_ๆ uses isolated ALICE-THI samples because Burapha omits it.",
                        "Class 55_ๅ uses manually screened crops from BEST2019 rows 1450..1549; inspect review_55_ๅ.png before scoring."]}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_review_grid(review_paths, output_dir / "review_55_ๅ.png")
    _write_review_grid(preview_paths, output_dir / "preview_all_classes.png")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_MODEL_LABELS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    arguments = parser.parse_args()
    print(json.dumps(build_public_test(arguments.labels, arguments.output_dir, arguments.per_class,
                                       arguments.train_manifest),
                     ensure_ascii=False, indent=2))

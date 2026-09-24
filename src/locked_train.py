"""Validate and run the frozen final-training protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
from typing import TYPE_CHECKING
from pathlib import Path

from src.split import fingerprint

if TYPE_CHECKING:
    from src.train import TrainConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPOSITORY_ROOT / "configs/experiments/densenet121_final_protocol.json"
TRAINING_CODE_FILES = ("train.py", "split.py")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest {path.name} missing columns: {sorted(missing)}")
        return list(reader)


def verify_dataset(
    data_dir: Path,
    split_dir: Path,
    *,
    expected_train: int,
    expected_val: int,
    expected_classes: int,
    verify_content_hashes: bool = False,
) -> dict:
    """Verify the manifest-defined dataset without decoding image pixels."""
    mapping_path = split_dir / "label_to_index.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if len(mapping) != expected_classes or sorted(mapping.values()) != list(range(expected_classes)):
        raise ValueError(f"Label mapping must contain exactly {expected_classes} contiguous indices")

    train_rows = _read_manifest(split_dir / "train.csv")
    val_rows = _read_manifest(split_dir / "val.csv")
    if len(train_rows) != expected_train or len(val_rows) != expected_val:
        raise ValueError(
            f"Manifest count mismatch: train {len(train_rows)} != {expected_train}, "
            f"validation {len(val_rows)} != {expected_val}"
        )

    seen_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    paths_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    hashes_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    source_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    class_counts: dict[str, dict[str, int]] = {}
    missing_files: list[str] = []
    bad_labels: list[str] = []
    duplicate_entries: list[str] = []
    hash_mismatches: list[str] = []

    for split_name, rows in (("train", train_rows), ("validation", val_rows)):
        for row in rows:
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Manifest path escapes dataset root: {row['path']}")
            key = relative.as_posix()
            if key in seen_by_split[split_name]:
                duplicate_entries.append(key)
            seen_by_split[split_name].add(key)
            paths_by_split[split_name].add(key)

            label = row["label"]
            if label not in mapping:
                bad_labels.append(f"{split_name}:{key}:{label}")
            counts = class_counts.setdefault(label, {"expected_train": 0, "actual_train": 0,
                                                       "expected_val": 0, "actual_val": 0})
            counts["expected_train" if split_name == "train" else "expected_val"] += 1

            target = data_dir / relative
            if not target.is_file() or not os.access(target, os.R_OK):
                missing_files.append(key)
                continue
            if row.get("sha256"):
                hashes_by_split[split_name].add(row["sha256"])
            if row.get("source_id"):
                source_by_split[split_name].add(row["source_id"])
            counts["actual_train" if split_name == "train" else "actual_val"] += 1
            if verify_content_hashes and row.get("sha256"):
                actual = _sha256(target)
                if actual != row["sha256"]:
                    hash_mismatches.append(f"{key}: {actual} != {row['sha256']}")

    if duplicate_entries:
        raise ValueError(f"Duplicate manifest entries: {duplicate_entries[:10]}")
    if bad_labels:
        raise ValueError(f"Manifest contains unknown labels: {bad_labels[:10]}")
    if missing_files:
        raise ValueError(
            f"Missing canonical files: expected {len(train_rows) + len(val_rows)}, "
            f"present {len(train_rows) + len(val_rows) - len(set(missing_files))}, "
            f"missing {len(set(missing_files))}; sample={missing_files[:10]}"
        )
    if hash_mismatches:
        raise ValueError(f"Manifest content hash mismatch: {hash_mismatches[:10]}")

    path_overlap = paths_by_split["train"] & paths_by_split["validation"]
    hash_overlap = hashes_by_split["train"] & hashes_by_split["validation"]
    source_available = bool(source_by_split["train"] or source_by_split["validation"])
    source_overlap = source_by_split["train"] & source_by_split["validation"]
    if path_overlap:
        raise ValueError(f"Train/validation path overlap: {sorted(path_overlap)[:10]}")
    if hash_overlap:
        raise ValueError(f"Train/validation hash overlap: {sorted(hash_overlap)[:10]}")
    if source_overlap:
        raise ValueError(f"Train/validation source ID overlap: {sorted(source_overlap)[:10]}")

    for label in mapping:
        class_counts.setdefault(label, {"expected_train": 0, "actual_train": 0,
                                         "expected_val": 0, "actual_val": 0})
    for counts in class_counts.values():
        if counts["expected_train"] != counts["actual_train"] or counts["expected_val"] != counts["actual_val"]:
            raise ValueError(f"Per-class file count mismatch: {counts}")

    canonical_paths = {data_dir / Path(row["path"]) for row in train_rows + val_rows}
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    extra_files = sum(1 for path in data_dir.rglob("*")
                      if path.is_file() and path.suffix.lower() in image_suffixes and path not in canonical_paths)
    return {
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "total_canonical_files": len(train_rows) + len(val_rows),
        "classes": len(mapping),
        "missing_files": missing_files,
        "duplicate_manifest_entries": len(duplicate_entries),
        "path_overlap": len(path_overlap),
        "hash_overlap": len(hash_overlap),
        "source_id_overlap": len(source_overlap) if source_available else None,
        "source_id_status": "PASS" if source_available else "NOT AVAILABLE",
        "content_hash_status": "PASS" if verify_content_hashes else "NOT RUN",
        "extra_image_files": extra_files,
        "class_counts": class_counts,
    }


def _print_preflight(report: dict, data_dir: Path, split_hash: str) -> None:
    source_status = report["source_id_status"]
    print("LOCKED DATASET PREFLIGHT")
    print(f"\nDataset root:\n{data_dir}")
    print(f"\nCanonical split:\n{split_hash}")
    print(f"\nClasses:\n{report['classes']} / {report['classes']} PASS")
    print(f"\nTrain manifest:\n{report['train_count']} / {report['train_count']} PASS")
    print(f"\nValidation manifest:\n{report['val_count']} / {report['val_count']} PASS")
    print(f"\nTotal canonical files:\n{report['total_canonical_files']} / {report['total_canonical_files']} PASS")
    print(f"\nPer-class counts:\n{report['classes']} / {report['classes']} PASS")
    print("\nMissing canonical files:\n0")
    print(f"\nDuplicate manifest entries:\n{report['duplicate_manifest_entries']}")
    print(f"\nTrain/validation path overlap:\n{report['path_overlap']}")
    print(f"\nTrain/validation hash overlap:\n{report['hash_overlap']}")
    print(f"\nTrain/validation source ID overlap:\n{report['source_id_overlap'] if source_status == 'PASS' else source_status}")
    print("\nLabel mapping:\nPASS")
    print(f"\nExisting manifest content hashes:\n{report['content_hash_status']}")
    print(f"\nExtra image files (not used):\n{report['extra_image_files']}")
    print("\nProtocol integrity:\nPASS")
    print("\nDataset readiness:\nPASS")


def validate_files(protocol_path: Path, *, verify_content_hashes: bool = False):
    protocol_path = protocol_path.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "locked":
        raise ValueError("Protocol status must be 'locked'")

    paths = protocol["paths"]
    data_dir = (REPOSITORY_ROOT / paths["data_dir"]).resolve()
    split_dir = (REPOSITORY_ROOT / paths["split_dir"]).resolve()
    output_dir = (REPOSITORY_ROOT / paths["output_dir"]).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Canonical dataset missing: {data_dir}")

    split = protocol["split"]
    for name, expected in split["files"].items():
        path = split_dir / name
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen split file changed: {name}: {actual} != {expected}")
    meta = json.loads((split_dir / "split_meta.json").read_text(encoding="utf-8"))
    if meta.get("split_hash") != split["split_hash"]:
        raise ValueError(f"Canonical split hash mismatch: {meta.get('split_hash')} != {split['split_hash']}")

    report = verify_dataset(
        data_dir,
        split_dir,
        expected_train=split["train_images"],
        expected_val=split["validation_images"],
        expected_classes=split["classes"],
        verify_content_hashes=verify_content_hashes,
    )

    code_hash = fingerprint({name: (REPOSITORY_ROOT / "src" / name).read_text()
                             for name in TRAINING_CODE_FILES})
    expected_code_hash = protocol["environment"]["training_code_hash"]
    if code_hash != expected_code_hash:
        raise ValueError(f"Training code changed: {code_hash} != {expected_code_hash}")
    _print_preflight(report, data_dir, split["split_hash"])
    return protocol, protocol["config"], data_dir, split_dir, output_dir


def validate_environment(protocol: dict, config: TrainConfig) -> None:
    from src.train import choose_device, device_name
    expected = protocol["environment"]
    actual_packages = {name: importlib.metadata.version(name) for name in expected["packages"]}
    if actual_packages != expected["packages"]:
        raise ValueError(f"Package environment mismatch: {actual_packages} != {expected['packages']}")
    device = choose_device(config.device)
    if device.type != expected["device_type"] or device_name(device) != expected["device_name"]:
        raise ValueError(
            f"Training device mismatch: {device.type}/{device_name(device)} != "
            f"{expected['device_type']}/{expected['device_name']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--check-files-only", action="store_true",
                        help="Verify immutable files/code without starting training or requiring the target GPU")
    parser.add_argument("--verify-content-hashes", action="store_true",
                        help="Also SHA256-hash every canonical file against its manifest entry")
    args = parser.parse_args()

    try:
        protocol, config_data, data_dir, split_dir, output_dir = validate_files(
            args.protocol, verify_content_hashes=args.verify_content_hashes
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        print("DATASET PREFLIGHT FAILED")
        print(str(error))
        return 1
    protocol_hash = _sha256(args.protocol.resolve())
    print(f"Protocol OK: {protocol['protocol_id']} ({protocol_hash})")
    if args.check_files_only:
        return 0

    from src.train import TrainConfig
    config = TrainConfig(**config_data)
    validate_environment(protocol, config)
    from src.train import fit
    receipt = fit(config, data_dir, split_dir, output_dir, phase=protocol["protocol_id"])
    run_dir = Path(receipt["checkpoint_path"]).parent
    shutil.copy2(args.protocol.resolve(), run_dir / "canonical_protocol.json")
    (run_dir / "canonical_protocol.sha256").write_text(protocol_hash + "\n", encoding="utf-8")
    checkpoint_hash = _sha256(Path(receipt["checkpoint_path"]))
    (run_dir / "checkpoint.sha256").write_text(checkpoint_hash + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(f"Inference: python -m src.inference --weights {receipt['checkpoint_path']} --input <image-or-folder>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

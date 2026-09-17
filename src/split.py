"""Frozen relative-path manifests, stratified by label and grouped by file hash."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from src.audit import write_csv


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def prepare_split(data_dir: Path, audit_dir: Path, split_dir: Path,
                  seed: int = 42, val_fraction: float = 0.2,
                  expected_classes: int | None = 72,
                  conflicting_label_policy: str = "exclude") -> dict:
    """Reuse only an identical split; never silently replace the team's split.

    The target is approximately 80/20 per class: indivisible duplicate groups
    can prevent an exact ratio. By default, every member of a hash group whose
    labels disagree is quarantined from both splits; the source images are not
    modified and the audit directory records the affected paths. Pass
    ``conflicting_label_policy='error'`` to require manual resolution instead.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    if conflicting_label_policy not in {"exclude", "error"}:
        raise ValueError("conflicting_label_policy must be 'exclude' or 'error'")
    data_dir = data_dir.resolve()
    rows = read_rows(audit_dir / "image_manifest.csv")
    if not rows:
        raise ValueError("No valid images. Check DATA_DIR, archives and LABEL_LEVEL.")
    for row in rows:
        row["path"] = Path(row["path"]).resolve().relative_to(data_dir).as_posix()
    rows = sorted(({k: row[k] for k in ("path", "label", "sha256")} for row in rows),
                  key=lambda row: row["path"])
    source_rows = rows
    labels = sorted({row["label"] for row in rows})
    if expected_classes is not None and len(labels) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, found {len(labels)}. Check label folders.")
    if len(labels) < 2:
        raise ValueError("Classification needs at least two classes")
    groups = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row)
    conflicting_groups = [group for group in groups.values() if len({row["label"] for row in group}) > 1]
    excluded_rows = sorted((row for group in conflicting_groups for row in group), key=lambda row: row["path"])
    write_csv(audit_dir / "conflicting_label_duplicates.csv", ["path", "label", "sha256"], excluded_rows)
    if conflicting_groups and conflicting_label_policy == "error":
        raise ValueError("Identical files have conflicting labels; see conflicting_label_duplicates.csv and resolve them before splitting.")
    if excluded_rows:
        excluded_hashes = {row["sha256"] for row in excluded_rows}
        rows = [row for row in rows if row["sha256"] not in excluded_hashes]
        groups = defaultdict(list)
        for row in rows:
            groups[row["sha256"]].append(row)
        labels = sorted({row["label"] for row in rows})
        if expected_classes is not None and len(labels) != expected_classes:
            raise ValueError("Excluding conflicting duplicates removed a class; resolve labels manually before splitting.")
        if len(labels) < 2:
            raise ValueError("Classification needs at least two classes after excluding conflicting duplicates")
    label_map = {label: i for i, label in enumerate(labels)}
    identity = {"dataset_hash": fingerprint(source_rows), "seed": seed,
                "val_fraction": val_fraction, "algorithm": "sha256-label-greedy-v2",
                "conflicting_label_policy": conflicting_label_policy,
                "excluded_conflicting_hashes": len(conflicting_groups),
                "excluded_conflicting_images": len(excluded_rows)}
    split_dir.mkdir(parents=True, exist_ok=True)
    meta_path = split_dir / "split_meta.json"
    if any(split_dir.iterdir()):
        required = [meta_path, split_dir / "train.csv", split_dir / "val.csv",
                    split_dir / "label_to_index.json"]
        if not all(path.is_file() for path in required):
            raise ValueError("Incomplete/external split directory. Use a new SPLIT_DIR or restore all shared files.")
        meta = json.loads(meta_path.read_text())
        if any(meta.get(key) != value for key, value in identity.items()):
            raise ValueError("Dataset or split settings changed. Restore team data/settings or use a new SPLIT_DIR.")
        if json.loads(required[-1].read_text()) != label_map:
            raise ValueError("Label mapping was modified")
        if fingerprint([read_rows(required[1]), read_rows(required[2]), label_map]) != meta["split_hash"]:
            raise ValueError("Frozen manifests were modified")
        return meta
    rng = random.Random(seed)
    train, val, train_only_classes = [], [], []
    for label in labels:
        buckets = [group for group in groups.values() if group[0]["label"] == label]
        if len(buckets) < 2:
            # A singleton class cannot appear in both partitions without data
            # leakage. Keep its real example in training and expose this fact
            # in metadata so validation metrics are interpreted correctly.
            train.extend(row for group in buckets for row in group)
            train_only_classes.append(label)
            continue
        rng.shuffle(buckets)
        target = sum(map(len, buckets)) * val_fraction
        # Start with the closest-sized group, not an arbitrary huge duplicate
        # group. Reserve at least one whole group for training.
        first = min(range(len(buckets)), key=lambda i: abs(len(buckets[i]) - target))
        chosen, remaining = list(buckets.pop(first)), []
        count = len(chosen)
        for index, group in enumerate(buckets):
            if index < len(buckets) - 1 and abs(count + len(group) - target) < abs(count - target):
                chosen.extend(group)
                count += len(group)
            else:
                remaining.extend(group)
        train.extend(remaining)
        val.extend(chosen)
    train.sort(key=lambda row: row["path"])
    val.sort(key=lambda row: row["path"])
    meta = {**identity, "split_hash": fingerprint([train, val, label_map]),
            "train_images": len(train), "val_images": len(val),
            "source_images": len(source_rows), "included_images": len(rows),
            "actual_val_fraction": len(val) / len(rows), "num_classes": len(labels),
            "train_only_classes": train_only_classes}
    write_csv(split_dir / "train.csv", ["path", "label", "sha256"], train)
    write_csv(split_dir / "val.csv", ["path", "label", "sha256"], val)
    (split_dir / "label_to_index.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2))
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta

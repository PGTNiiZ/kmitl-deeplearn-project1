"""Frozen manifests stratified by label and grouped by source family/hash."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from src.audit import write_csv


DERIVED_SUFFIX = re.compile(r"__(?:aug|erode|dilate)_[^/]*")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def source_id_for_path(path: str) -> str:
    """Return the original-image family shared by pre-generated variants."""
    relative = Path(path)
    stem = DERIVED_SUFFIX.split(relative.stem, maxsplit=1)[0]
    return (relative.parent / stem).as_posix()


def grouped_rows(rows: list[dict]) -> list[list[dict]]:
    """Group source families, also joining families with identical files."""
    by_source = defaultdict(list)
    sources_by_hash = defaultdict(set)
    for row in rows:
        by_source[row["source_id"]].append(row)
        sources_by_hash[row["sha256"]].add(row["source_id"])
    neighbours = defaultdict(set)
    for sources in sources_by_hash.values():
        first, *others = sources
        neighbours[first].update(others)
        for source in others:
            neighbours[source].add(first)
    groups, seen = [], set()
    for source in by_source:
        if source in seen:
            continue
        stack, component = [source], []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.extend(by_source[current])
            stack.extend(neighbours[current] - seen)
        groups.append(component)
    return groups


def prepare_split(data_dir: Path, audit_dir: Path, split_dir: Path,
                  seed: int = 42, val_fraction: float = 0.2,
                  expected_classes: int | None = 72,
                  conflicting_label_policy: str = "exclude") -> dict:
    """Reuse only an identical split; never silently replace the team's split.

    The target is approximately 80/20 original images per class. Pre-generated
    variants (``__aug_*``, ``__erode_*`` and ``__dilate_*``) follow their source
    family into training, while held-out families contribute only originals to
    validation. Indivisible source/duplicate groups can prevent an exact ratio.
    By default, every member of a hash group whose labels disagree is
    quarantined from both splits; the source images are not modified and the
    audit directory records the affected paths. Pass
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
        row["source_id"] = source_id_for_path(row["path"])
    rows = sorted(({k: row[k] for k in ("path", "label", "sha256", "source_id")} for row in rows),
                  key=lambda row: row["path"])
    source_rows = rows
    labels = sorted({row["label"] for row in rows})
    if expected_classes is not None and len(labels) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, found {len(labels)}. Check label folders.")
    if len(labels) < 2:
        raise ValueError("Classification needs at least two classes")
    hash_groups = defaultdict(list)
    for row in rows:
        hash_groups[row["sha256"]].append(row)
    conflicting_groups = [group for group in hash_groups.values() if len({row["label"] for row in group}) > 1]
    excluded_rows = sorted((row for group in conflicting_groups for row in group), key=lambda row: row["path"])
    write_csv(audit_dir / "conflicting_label_duplicates.csv", ["path", "label", "sha256"],
              [{key: row[key] for key in ("path", "label", "sha256")} for row in excluded_rows])
    if conflicting_groups and conflicting_label_policy == "error":
        raise ValueError("Identical files have conflicting labels; see conflicting_label_duplicates.csv and resolve them before splitting.")
    if excluded_rows:
        excluded_hashes = {row["sha256"] for row in excluded_rows}
        rows = [row for row in rows if row["sha256"] not in excluded_hashes]
        labels = sorted({row["label"] for row in rows})
        if expected_classes is not None and len(labels) != expected_classes:
            raise ValueError("Excluding conflicting duplicates removed a class; resolve labels manually before splitting.")
        if len(labels) < 2:
            raise ValueError("Classification needs at least two classes after excluding conflicting duplicates")
    label_map = {label: i for i, label in enumerate(labels)}
    identity = {"dataset_hash": fingerprint(source_rows), "seed": seed,
                "val_fraction": val_fraction, "algorithm": "source-family-sha256-greedy-v3",
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
    train, val, train_only_classes, held_out_derived = [], [], [], []
    groups = grouped_rows(rows)
    for label in labels:
        buckets = [group for group in groups if group[0]["label"] == label]
        eligible = [group for group in buckets if any(not DERIVED_SUFFIX.search(Path(row["path"]).stem)
                                                     for row in group)]
        if len(eligible) < 2:
            # A singleton class cannot appear in both partitions without data
            # leakage. Keep its real example in training and expose this fact
            # in metadata so validation metrics are interpreted correctly.
            train.extend(row for group in buckets for row in group)
            train_only_classes.append(label)
            continue
        rng.shuffle(eligible)
        original_count = lambda group: sum(not DERIVED_SUFFIX.search(Path(row["path"]).stem)
                                           for row in group)
        target = sum(map(original_count, eligible)) * val_fraction
        # Start with the closest-sized group, not an arbitrary huge duplicate
        # group. Reserve at least one whole group for training.
        first = min(range(len(eligible)), key=lambda i: abs(original_count(eligible[i]) - target))
        chosen = [eligible.pop(first)]
        count = original_count(chosen[0])
        for index, group in enumerate(eligible):
            size = original_count(group)
            if index < len(eligible) - 1 and abs(count + size - target) < abs(count - target):
                chosen.append(group)
                count += size
        chosen_ids = {id(group) for group in chosen}
        train.extend(row for group in buckets if id(group) not in chosen_ids for row in group)
        for group in chosen:
            val.extend(row for row in group if not DERIVED_SUFFIX.search(Path(row["path"]).stem))
            held_out_derived.extend(row for row in group if DERIVED_SUFFIX.search(Path(row["path"]).stem))
    train.sort(key=lambda row: row["path"])
    val.sort(key=lambda row: row["path"])
    meta = {**identity, "split_hash": fingerprint([train, val, label_map]),
            "train_images": len(train), "val_images": len(val),
            "source_images": len(source_rows), "included_images": len(train) + len(val),
            "original_images": sum(not DERIVED_SUFFIX.search(Path(row["path"]).stem) for row in rows),
            "derived_images": sum(bool(DERIVED_SUFFIX.search(Path(row["path"]).stem)) for row in rows),
            "held_out_derived_images": len(held_out_derived),
            "actual_val_fraction": len(val) / sum(not DERIVED_SUFFIX.search(Path(row["path"]).stem)
                                                   for row in rows),
            "num_classes": len(labels),
            "train_only_classes": train_only_classes}
    fields = ["path", "label", "sha256", "source_id"]
    write_csv(split_dir / "train.csv", fields, train)
    write_csv(split_dir / "val.csv", fields, val)
    (split_dir / "label_to_index.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2))
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta

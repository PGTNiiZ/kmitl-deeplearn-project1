"""Regression tests for locked canonical dataset preflight."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.locked_train import verify_dataset


class DatasetPreflightTest(unittest.TestCase):
    def make_fixture(self, *, duplicate_train=False, wrong_label=False,
                     missing=False, overlap=False, extra=False):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name) / "images"
        split = Path(temp.name) / "split"
        root.mkdir(parents=True)
        split.mkdir()
        mapping = {"a": 0, "b": 1}
        (split / "label_to_index.json").write_text(json.dumps(mapping), encoding="utf-8")
        train = [("train/a.png", "a", "source-a"), ("train/b.png", "b", "source-b")]
        val = [("val/a.png", "a", "source-c"), ("val/b.png", "b", "source-d")]
        if duplicate_train:
            train[1] = train[0]
        if wrong_label:
            train[0] = (train[0][0], "not-in-mapping", train[0][2])
        if overlap:
            val[0] = train[0]
        for rows in (train, val):
            for path, label, source_id in rows:
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                if not (missing and path == "val/b.png"):
                    target.write_bytes((label + source_id).encode())
        if extra:
            (root / "unrelated.png").write_bytes(b"extra")
        def write_manifest(name, rows):
            with (split / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["path", "label", "sha256", "source_id"])
                writer.writeheader()
                for path, label, source_id in rows:
                    file_path = root / path
                    digest = hashlib.sha256(file_path.read_bytes()).hexdigest() if file_path.exists() else "0" * 64
                    writer.writerow({"path": path, "label": label, "sha256": digest, "source_id": source_id})
        write_manifest("train.csv", train)
        write_manifest("val.csv", val)
        return temp, root, split

    def test_complete_manifest_passes(self):
        temp, root, split = self.make_fixture()
        self.addCleanup(temp.cleanup)
        report = verify_dataset(root, split, expected_train=2, expected_val=2, expected_classes=2)
        self.assertEqual(report["missing_files"], [])
        self.assertEqual(report["total_canonical_files"], 4)
        self.assertEqual(report["extra_image_files"], 0)
        self.assertEqual(report["content_hash_status"], "NOT RUN")

    def test_missing_canonical_file_fails(self):
        temp, root, split = self.make_fixture(missing=True)
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "Missing canonical files"):
            verify_dataset(root, split, expected_train=2, expected_val=2, expected_classes=2)

    def test_duplicate_manifest_entry_fails(self):
        temp, root, split = self.make_fixture(duplicate_train=True)
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "Duplicate manifest entries"):
            verify_dataset(root, split, expected_train=2, expected_val=2, expected_classes=2)

    def test_wrong_label_fails(self):
        temp, root, split = self.make_fixture(wrong_label=True)
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "unknown labels"):
            verify_dataset(root, split, expected_train=2, expected_val=2, expected_classes=2)

    def test_train_validation_overlap_fails(self):
        temp, root, split = self.make_fixture(overlap=True)
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "Train/validation path overlap"):
            verify_dataset(root, split, expected_train=2, expected_val=2, expected_classes=2)

    def test_extra_image_does_not_fail_canonical_check(self):
        temp, root, split = self.make_fixture(extra=True)
        self.addCleanup(temp.cleanup)
        report = verify_dataset(root, split, expected_train=2, expected_val=2, expected_classes=2)
        self.assertEqual(report["extra_image_files"], 1)

    def test_optional_content_hash_check_passes(self):
        temp, root, split = self.make_fixture()
        self.addCleanup(temp.cleanup)
        report = verify_dataset(root, split, expected_train=2, expected_val=2,
                               expected_classes=2, verify_content_hashes=True)
        self.assertEqual(report["content_hash_status"], "PASS")

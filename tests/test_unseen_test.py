import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.unseen_test import crop_form, load_labels, perspective_coefficients, score_predictions
from src.build_public_test import load_model_labels, tis_code


class UnseenTestSetTest(unittest.TestCase):
    def test_project_labels_map_to_expected_tis_codes(self) -> None:
        self.assertEqual(tis_code("0_ก"), 161)
        self.assertEqual(tis_code("55_ๅ"), 229)
        self.assertEqual(tis_code("71_๙"), 249)
        self.assertEqual(tis_code("229"), 229)

    def test_public_test_uses_checkpoint_label_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps({str(code): index for index, code in enumerate(range(161, 233))}))
            self.assertEqual(load_model_labels(path)[:2], ["161", "162"])

    def test_labels_use_semantic_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps({f"{i}_x": i for i in range(72)}))
            labels = load_labels(path)
            self.assertEqual(labels[:3], ["0_x", "1_x", "2_x"])
            self.assertEqual(labels[-1], "71_x")

    def test_identity_perspective_coefficients(self) -> None:
        coefficients = perspective_coefficients([(0, 0), (800, 0), (800, 900), (0, 900)], (800, 900))
        self.assertEqual([round(value, 8) for value in coefficients], [1, 0, 0, 0, 1, 0, 0, 0])

    def test_crop_writes_balanced_manifest(self) -> None:
        labels = [f"{i}_x" for i in range(72)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "page.png"
            Image.new("RGB", (2400, 2700), "white").save(source)
            report = crop_form(source, [(0, 0), (2400, 0), (2400, 2700), (0, 2700)],
                               labels, root / "out", "writer01", "daylight")
            with (root / "out" / "manifest.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(report, {"images": 72, "writers": 1, "classes": 72,
                                      "minimum_per_class": 1, "maximum_per_class": 1})
            self.assertEqual(len(rows), 72)
            self.assertTrue((root / "out" / rows[0]["path"]).is_file())

    def test_score_joins_paths_and_reports_macro_recall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            with (dataset / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["path", "label", "writer_id", "capture_id"])
                writer.writeheader()
                writer.writerows([
                    {"path": "images/0_ก/w1.png", "label": "0_ก", "writer_id": "w1", "capture_id": "day"},
                    {"path": "images/1_ข/w1.png", "label": "1_ข", "writer_id": "w1", "capture_id": "day"},
                ])
            predictions = root / "predictions.csv"
            with predictions.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["path", "prediction", "status", "confidence"])
                writer.writeheader()
                writer.writerows([
                    {"path": "/tmp/images/0_ก/w1.png", "prediction": "0_ก", "status": "OK", "confidence": ".9"},
                    {"path": "/tmp/images/1_ข/w1.png", "prediction": "0_ก", "status": "OK", "confidence": ".8"},
                ])
            report = score_predictions(dataset, predictions, root / "score.json")
            self.assertEqual(report["accuracy"], 0.5)
            self.assertEqual(report["macro_recall"], 0.5)
            self.assertEqual(report["per_writer_accuracy"], {"w1": 0.5})


if __name__ == "__main__":
    unittest.main()

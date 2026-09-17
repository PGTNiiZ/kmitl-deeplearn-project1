"""Check visual answers against paths, including missing truth and escaped labels."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from PIL import Image

from src.visualize import load_predictions, validation_gallery, live_prediction_gallery


class VisualReportTest(unittest.TestCase):
    def test_same_images_align_by_path_and_reject_wrong_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"path": "b.png", "label": "002"}, {"path": "a.png", "label": "001"}]
            for row in rows:
                Image.new("RGB", (24, 32), "white").save(root / row["path"])
            predictions = pd.DataFrame([
                {"path": "a.png", "true_label": "001", "predicted_label": "002", "confidence": .9},
                {"path": "b.png", "true_label": "002", "predicted_label": "002", "confidence": .6},
            ])
            predictions.to_csv(root / "predictions.csv", index=False)
            receipt = {"checkpoint_path": str(root / "best.pt"), "run_id": "model-a", "backbone": "A",
                       "seed": 42, "split_hash": "frozen", "phase": "screening", "val_macro_f1": .3,
                       "best_epoch": 2}
            second = {**receipt, "run_id": "model-b", "backbone": "B"}
            html = validation_gallery([receipt, second], root, rows, root / "reports", sample_count=2,
                                      labels={"001": "<script>ก</script>"}, context="smoke")
            self.assertIn("data:image/png;base64,", html)
            self.assertIn("&lt;script&gt;", html)
            self.assertNotIn("<script>", html)
            self.assertIn("ภาพจำลอง", html)
            comparison = pd.read_csv(root / "reports/same_images_comparison.csv", dtype={"true_label": str})
            self.assertEqual(comparison.groupby("path").size().tolist(), [2, 2])
            self.assertFalse(comparison[comparison.path == "a.png"].correct.any())
            self.assertTrue(comparison[comparison.path == "b.png"].correct.all())
            self.assertEqual(comparison.iloc[0].true_label in {"001", "002"}, True)
            self.assertTrue((root / "reports/model-a.html").is_file())
            predictions.loc[0, "true_label"] = "002"
            predictions.to_csv(root / "predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "ground truth"):
                load_predictions(receipt, rows)

    def test_live_unknown_truth_is_not_marked_correct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            Image.new("RGB", (24, 32), "white").save(path)
            predictor = SimpleNamespace(
                predict=lambda paths, top_k: [{"predicted_label": "001", "confidence": .8,
                    "top_k": [{"label": "001", "confidence": .8}, {"label": "002", "confidence": .2}]}],
                transform=lambda image: torch.ones(3, 32, 32),
                config=SimpleNamespace(mean=(0, 0, 0), std=(1, 1, 1)))
            html, table = live_prediction_gallery(predictor, [path])
            self.assertIn("ยังประเมินถูก/ผิดไม่ได้", html)
            self.assertTrue(pd.isna(table.iloc[0].correct))
            self.assertEqual(html.count('src="data:image'), 2)
            html, table = live_prediction_gallery(predictor, [path], truth={str(path.resolve()): "002"})
            self.assertFalse(table.iloc[0].correct)
            self.assertIn("Top-3", html)


if __name__ == "__main__":
    unittest.main()

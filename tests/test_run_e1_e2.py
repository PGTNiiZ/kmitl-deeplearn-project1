"""Exercise orchestration with fake training and real saved-image reports."""
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.run_e1_e2 import REPOSITORY_ROOT, run
from src.split import fingerprint
from src.train import TrainConfig


class ControlledPairTest(unittest.TestCase):
    def test_pair_reports_and_contract_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, split, output = root / "data", root / "split", root / "output"
            data.mkdir()
            split.mkdir()
            Image.new("RGB", (32, 32), "white").save(data / "a.png")
            (split / "val.csv").write_text("path,label\na.png,a\n", encoding="utf-8")
            config = asdict(TrainConfig(architecture="densenet121.ra_in1k"))
            protocol = {"environment": {"training_code_hash": fingerprint({
                name: (REPOSITORY_ROOT / "src" / name).read_text()
                for name in ("train.py", "split.py")})}}
            configs = []

            def fake_fit(config, data_dir, split_dir, output_dir, phase):
                configs.append(asdict(config))
                run_dir = output_dir / "test_run"
                run_dir.mkdir(parents=True)
                (run_dir / "best.pt").write_bytes(b"test checkpoint")
                (run_dir / "history.csv").write_text("epoch\n1\n")
                (run_dir / "confusion_matrix.csv").write_text(",a\na,1\n")
                (run_dir / "predictions.csv").write_text(
                    "path,true_label,predicted_label,confidence\na.png,a,a,0.9\n")
                return {"status": "complete", "run_id": output_dir.name, "phase": phase,
                        "split_hash": "test", "seed": config.seed, "backbone": config.architecture,
                        "checkpoint_path": str(run_dir / "best.pt"), "best_epoch": 1,
                        "val_macro_f1": 0.95 if config.augmentation else 0.9, "val_accuracy": 0.95}

            with patch("src.run_e1_e2.validate_files", return_value=(protocol, config, data, split, output)), \
                 patch("src.run_e1_e2.validate_environment"), patch("src.train.fit", side_effect=fake_fit):
                ranked = run(root / "protocol.json", output)
                self.assertEqual(ranked[0]["experiment"], "E2")
                self.assertEqual({key for key in configs[0] if configs[0][key] != configs[1][key]},
                                 {"augmentation"})
                self.assertFalse(configs[0]["augmentation"])
                self.assertTrue((output / "error_analysis/index.html").is_file())
                state = json.loads((output / "status.json").read_text())
                self.assertEqual(state["status"], "awaiting_e5_review")
                self.assertEqual(state["completed"], ["E1_NEW", "E2"])
                self.assertFalse((output / "RUNNING.lock").exists())
                self.assertFalse((output / "final").exists())
                protocol["changed"] = True
                with self.assertRaisesRegex(ValueError, "protocol/runner changed"):
                    run(root / "protocol.json", output)
                self.assertEqual(json.loads((output / "status.json").read_text()), state)

    def test_preflight_failure_never_trains(self):
        with patch("src.run_e1_e2.validate_files", side_effect=ValueError("missing data")), \
             patch("src.train.fit") as fit:
            with self.assertRaisesRegex(ValueError, "missing data"):
                run()
            fit.assert_not_called()

    def test_training_failure_stops_before_e2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = asdict(TrainConfig(architecture="densenet121.ra_in1k"))
            protocol = {"environment": {"training_code_hash": fingerprint({
                name: (REPOSITORY_ROOT / "src" / name).read_text()
                for name in ("train.py", "split.py")})}}
            with patch("src.run_e1_e2.validate_files", return_value=(protocol, config, root, root, root)), \
                 patch("src.run_e1_e2.validate_environment"), \
                 patch("src.train.fit", side_effect=RuntimeError("training failed")) as fit:
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    run(root / "protocol.json", root)
                self.assertEqual(fit.call_count, 1)
                self.assertEqual(json.loads((root / "status.json").read_text())["status"], "failed")
                self.assertFalse((root / "RUNNING.lock").exists())


if __name__ == "__main__":
    unittest.main()
